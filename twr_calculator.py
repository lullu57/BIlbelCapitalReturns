import pandas as pd
import numpy as np
from datetime import datetime
import argparse
from typing import Tuple, List, Dict, Optional
import os
import re
import IBKR
import Exante
import logging
import openpyxl
import utils
import flow_utils
from config_loader import load_config, Config, PeriodsConfig
from gips_fee_tracker import (
    GIPSFeeTracker,
    calculate_net_twr_from_monthly,
    calculate_period_net_return,
    annual_to_monthly_fee
)


def _normalize_exante_client_name(filename: str) -> str:
    base = os.path.splitext(os.path.basename(filename))[0]
    if base.lower().startswith('exante '):
        base = base[len('exante '):]
    if ' -- ' in base:
        base = base.split(' -- ', 1)[0]
    return base


_CURRENCY_SYMBOLS = {
    'EUR': '€',
    'USD': '$',
    'GBP': '£',
    'CHF': 'CHF ',
    'JPY': '¥',
}


def _normalize_currency_code(code: Optional[str]) -> str:
    if not code:
        return ''
    return str(code).strip().upper()


def _get_report_currency(config: Optional[Config]) -> str:
    if config is None:
        return 'EUR'
    base_currency = _normalize_currency_code(config.currency.base_currency) or 'EUR'
    report_currency = _normalize_currency_code(config.currency.report_currency)
    return report_currency or base_currency


def _get_currency_symbol(currency_code: str) -> str:
    code = _normalize_currency_code(currency_code)
    return _CURRENCY_SYMBOLS.get(code, f'{code} ')


def _find_column_case_insensitive(columns: List[str], target: str) -> Optional[str]:
    target_lower = target.lower()
    for col in columns:
        if str(col).lower() == target_lower:
            return col
    return None


def _select_fx_rate_column(
    df: pd.DataFrame,
    base_currency: str,
    report_currency: str,
    preferred: Optional[str] = None
) -> Tuple[str, bool]:
    columns = [str(col) for col in df.columns]
    base = _normalize_currency_code(base_currency)
    report = _normalize_currency_code(report_currency)
    direct = f'{base}/{report}'
    inverse = f'{report}/{base}'

    def normalize(col: str) -> str:
        return re.sub(r'[^A-Za-z/]', '', col).upper()

    if preferred:
        preferred_match = preferred if preferred in columns else _find_column_case_insensitive(columns, preferred)
        if preferred_match:
            norm = normalize(preferred_match)
            if inverse in norm:
                return preferred_match, True
            return preferred_match, False

    for col in columns:
        norm = normalize(col)
        if direct in norm:
            return col, False
        if inverse in norm:
            return col, True

    raise ValueError(
        f"Could not find FX rate column for {report}/{base}. "
        f"Available columns: {columns}"
    )


def _load_fx_rates(config: Optional[Config]) -> Optional[pd.DataFrame]:
    if config is None:
        return None

    base_currency = _normalize_currency_code(config.currency.base_currency) or 'EUR'
    report_currency = _normalize_currency_code(config.currency.report_currency) or base_currency
    if report_currency == base_currency:
        return None

    fx_file = config.currency.fx_rates_file
    if not fx_file:
        raise ValueError(
            f"FX rates file is required to report in {report_currency} when base is {base_currency}."
        )

    sheet = config.currency.fx_rates_sheet if config.currency.fx_rates_sheet else 0
    fx_df = pd.read_excel(fx_file, sheet_name=sheet)
    if fx_df.empty:
        raise ValueError(f"FX rates sheet is empty: {fx_file}")

    date_col = config.currency.fx_date_column or 'TIME_PERIOD'
    if date_col not in fx_df.columns:
        date_col = _find_column_case_insensitive(list(fx_df.columns), date_col) or date_col
    if date_col not in fx_df.columns:
        raise ValueError(f"FX date column '{date_col}' not found in {fx_file}")

    rate_col, invert = _select_fx_rate_column(
        fx_df,
        base_currency=base_currency,
        report_currency=report_currency,
        preferred=config.currency.fx_rate_column
    )

    fx_df = fx_df[[date_col, rate_col]].copy()
    fx_df = fx_df.rename(columns={date_col: 'Date', rate_col: 'Rate'})
    fx_df['Date'] = pd.to_datetime(fx_df['Date'], errors='coerce').dt.normalize()
    fx_df['Rate'] = pd.to_numeric(fx_df['Rate'], errors='coerce')
    fx_df = fx_df.dropna(subset=['Date', 'Rate']).sort_values('Date')
    fx_df = fx_df.drop_duplicates(subset=['Date'], keep='last')

    if invert:
        fx_df['Rate'] = fx_df['Rate'].rdiv(1.0)

    return fx_df


def _apply_fx_rates(
    nav_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    fx_rates: pd.DataFrame,
    flow_columns: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if fx_rates is None or fx_rates.empty:
        return nav_df, trades_df

    nav_df = nav_df.copy()
    nav_df['Date'] = pd.to_datetime(nav_df['Date'], errors='coerce').dt.normalize()
    nav_df = nav_df.sort_values('Date')
    nav_df = pd.merge_asof(nav_df, fx_rates, on='Date', direction='backward')
    missing_nav_rates = nav_df['Rate'].isna().sum()
    if missing_nav_rates:
        logging.warning(f"Missing FX rates for {missing_nav_rates} NAV rows.")
    nav_df['Net Asset Value'] = pd.to_numeric(nav_df['Net Asset Value'], errors='coerce') * nav_df['Rate']
    nav_df = nav_df.drop(columns=['Rate'])

    if trades_df is None or trades_df.empty:
        return nav_df, trades_df

    trades_df = trades_df.copy()
    trades_df['When'] = pd.to_datetime(trades_df['When'], errors='coerce').dt.normalize()
    trades_df = trades_df.sort_values('When')
    trades_df = pd.merge_asof(
        trades_df,
        fx_rates.rename(columns={'Date': 'When'}),
        on='When',
        direction='backward'
    )
    missing_trade_rates = trades_df['Rate'].isna().sum()
    if missing_trade_rates:
        logging.warning(f"Missing FX rates for {missing_trade_rates} flow rows.")

    for col in flow_columns:
        if col in trades_df.columns:
            trades_df[col] = pd.to_numeric(trades_df[col], errors='coerce') * trades_df['Rate']

    trades_df = trades_df.drop(columns=['Rate'], errors='ignore')
    return nav_df, trades_df


def _group_exante_files(input_path: str) -> Dict[str, Dict[str, List[str]]]:
    grouped: Dict[str, Dict[str, List[str]]] = {}
    def _exante_sort_key(filename: str) -> tuple:
        base = os.path.splitext(os.path.basename(filename))[0]
        range_part = ''
        if ' -- ' in base:
            range_part = base.split(' -- ', 1)[1]
        is_inception = range_part.lower().startswith('inception')
        return (0 if is_inception else 1, range_part.lower())

    for file in sorted(os.listdir(input_path), key=_exante_sort_key):
        if not file.endswith('.csv'):
            continue
        client = _normalize_exante_client_name(file)
        csv_path = os.path.join(input_path, file)
        output_dir = os.path.join(input_path, os.path.splitext(file)[0])
        entry = grouped.setdefault(client, {'source_csv': [], 'output_dir': []})
        entry['source_csv'].append(csv_path)
        entry['output_dir'].append(output_dir)
    return grouped


def get_period_windows(config: Optional[Config] = None) -> Dict[str, Tuple[str, str]]:
    """
    Get period windows for return calculations.
    
    Uses configuration if provided, otherwise uses defaults.
    
    Args:
        config: Optional configuration object
    
    Returns:
        Dictionary mapping period labels to (start_date, end_date) tuples
    """
    if config is not None:
        return config.periods.get_period_windows()
    
    # Default periods if no config provided
    return PeriodsConfig().get_period_windows()



"""
@TODO
Update comments
- Remove those that aren't meaningful
- Add more and explain better in the parts that do the mathematical calculations

Flow
- Several functions are meant to return boolean based on success or failure however this is not done and checked for consistently
- Refer to brokerages process() functions for example
"""


class ReturnsCalculator:
    """
    GIPS-compliant returns calculator for multiple brokerage accounts.
    
    Replaces global state with a class-based approach for better maintainability.
    Configuration is loaded from config.yaml.
    """
    
    def __init__(self, config_path: str = 'config.yaml', input_path: Optional[str] = None):
        """
        Initialize the returns calculator.
        
        Args:
            config_path: Path to configuration file
            input_path: Override input path from config (optional)
        """
        self.config = load_config(config_path)
        self.input_path = input_path or self.config.paths.input_dir
        self.output_path = self.config.paths.output_dir
        self.report_currency = _get_report_currency(self.config)
        self.currency_symbol = _get_currency_symbol(self.report_currency)
        self.fx_rates = _load_fx_rates(self.config)
        
        # State
        self.brokerages: Dict[str, Dict] = {}
        self.all_results: Dict = {}
        self.all_client_returns: List[Dict] = []
        
        # Load brokerage modules dynamically
        self.brokerage_modules: Dict[str, tuple] = {}
        self._load_brokerage_modules()
    
    def _load_brokerage_modules(self):
        """Load brokerage modules based on configuration."""
        import importlib
        
        for brokerage_config in self.config.get_enabled_brokerages():
            try:
                module = importlib.import_module(brokerage_config.module)
                self.brokerage_modules[brokerage_config.name] = (
                    getattr(module, 'process'),
                    getattr(module, 'read_data')
                )
                self.brokerages[brokerage_config.name] = {}
                logging.info(f"Loaded brokerage module: {brokerage_config.name}")
            except (ImportError, AttributeError) as e:
                logging.warning(f"Could not load brokerage module {brokerage_config.module}: {e}")
    
    def scan_for_accounts(self):
        """Scan input directory for account CSV files."""
        print('Scanning for accounts...')
        
        for brokerage_name in self.brokerages.keys():
            brokerage_input_path = os.path.join(self.input_path, brokerage_name)
            
            if os.path.exists(brokerage_input_path):
                if brokerage_name == 'Exante':
                    self.brokerages[brokerage_name].update(
                        _group_exante_files(brokerage_input_path)
                    )
                else:
                    for file in os.listdir(brokerage_input_path):
                        if file.endswith('.csv'):
                            client = os.path.splitext(file)[0]
                            csv_path = os.path.join(brokerage_input_path, file)
                            client_dir = os.path.join(brokerage_input_path, client)
                            self.brokerages[brokerage_name][client] = {
                                'source_csv': csv_path,
                                'output_dir': client_dir
                            }

            if len(self.brokerages[brokerage_name]):
                print(f"Found {len(self.brokerages[brokerage_name])} {brokerage_name} accounts")
    
    def process_brokerage(self, broker_name: str):
        """
        Process all clients for a specific brokerage.
        
        Args:
            broker_name: Name of the brokerage to process
        """
        if broker_name not in self.brokerage_modules:
            logging.error(f"No module loaded for brokerage: {broker_name}")
            return
        
        process_func, read_data_func = self.brokerage_modules[broker_name]
        large_flow_threshold = self.config.thresholds.large_cash_flow_pct
        
        for client, files in self.brokerages[broker_name].items():
            try:
                print(f"Processing {broker_name} data for {client}...")
                source_csv = files['source_csv']
                if isinstance(source_csv, (list, tuple)):
                    for csv_path in source_csv:
                        process_func(csv_path)
                else:
                    process_func(source_csv)

                # Processing logic
                output_dir = files['output_dir']
                if isinstance(output_dir, (list, tuple)):
                    # Exante can provide multiple output folders (one per CSV range).
                    # Merge them into a single NAV/flow set before return calculations.
                    nav_frames = []
                    trade_frames = []
                    for out_dir in output_dir:
                        nav_part, trade_part = read_data_func(out_dir)
                        if isinstance(nav_part, pd.DataFrame) and not nav_part.empty:
                            nav_frames.append(nav_part)
                        if isinstance(trade_part, pd.DataFrame) and not trade_part.empty:
                            trade_frames.append(trade_part)
                    if nav_frames:
                        nav_df = pd.concat(nav_frames, ignore_index=True)
                        # If overlapping dates exist, keep the last entry per date.
                        nav_df['Date'] = pd.to_datetime(nav_df['Date']).dt.normalize()
                        nav_df = nav_df.sort_values('Date').drop_duplicates(subset=['Date'], keep='last')
                    else:
                        nav_df = pd.DataFrame(columns=['Date', 'Net Asset Value'])
                    if trade_frames:
                        trades_df = pd.concat(trade_frames, ignore_index=True)
                        trades_df['When'] = pd.to_datetime(trades_df['When']).dt.normalize()
                        trades_df = trades_df.sort_values('When').drop_duplicates()
                    else:
                        trades_df = pd.DataFrame(columns=['When', 'Operation type', 'EUR equivalent'])
                else:
                    nav_df, trades_df = read_data_func(output_dir)
                if self.fx_rates is not None:
                    nav_df, trades_df = _apply_fx_rates(
                        nav_df,
                        trades_df,
                        self.fx_rates,
                        self.config.currency.flow_columns
                    )
                sub_period_returns = calculate_sub_period_returns(
                    nav_df, trades_df, 
                    large_flow_threshold=large_flow_threshold
                )
                monthly_returns = calculate_monthly_twr(sub_period_returns)
                # Build daily series directly from NAV and flows to mirror reference TWR
                daily_returns = build_daily_returns_direct(nav_df, trades_df)
                # Use daily-compounded TWR for totals to ensure no days are missed
                absolute_return, annualized_return = calculate_total_returns_from_daily(daily_returns)

                results = {
                    'monthly_returns': monthly_returns,
                    'daily_returns': daily_returns,
                    'absolute_return': absolute_return,
                    'annualized_return': annualized_return,
                    'sub_period_returns': sub_period_returns,
                    'account_name': f'{broker_name}_{client}'
                }

                self.all_results[f'{broker_name}_{client}'] = results
                self.all_client_returns.append(results)
                
            except Exception as e:
                logging.error(f"Error processing {broker_name} client {client}: {e}")
            continue

    def process_all_brokerages(self):
        """Process all enabled brokerages."""
        for brokerage_name in self.brokerages.keys():
            if self.brokerages[brokerage_name]:  # Only process if accounts found
                self.process_brokerage(brokerage_name)
    
    def aggregate_returns(self):
        """Aggregate client returns into composite."""
        aggregate_client_returns(self.all_client_returns, self.all_results, config=self.config)
    
    def save_results(self):
        """Save all results to output directory."""
        save_all_results(self.all_results, self.output_path, config=self.config)
    
    def run(self):
        """Run the full returns calculation pipeline."""
        self.scan_for_accounts()
        
        total_accounts = sum(len(clients) for clients in self.brokerages.values())
        if total_accounts == 0:
            print("No accounts found to process")
            return
        
        self.process_all_brokerages()
        self.aggregate_returns()
        self.save_results()
        
        print("Returns calculation process completed")


# Legacy functions for backward compatibility
def scan_for_accounts():
    """Legacy function - use ReturnsCalculator class instead."""
    global brokerages, args
    print('Scanning for accounts...')
    for brokerage in brokerages:
        input_path = os.path.join(args.input_path, brokerage)
        if os.path.exists(input_path):
            if brokerage == 'Exante':
                brokerages[brokerage].update(_group_exante_files(input_path))
            else:
                for file in os.listdir(input_path):
                    if file.endswith('.csv'):
                        client = os.path.splitext(file)[0]
                        csv_path = os.path.join(input_path, file)
                        client_dir = os.path.join(input_path, client)
                        brokerages[brokerage][client] = {
                            'source_csv': csv_path,
                            'output_dir': client_dir
                        }

        _add_manual_brokerage_accounts(
            brokerage,
            input_path,
            brokerages[brokerage]
        )
        if len(brokerages[brokerage]):
            print(f"Found {len(brokerages[brokerage])} {brokerage} accounts")


def process_brokerage(broker_name, process, read_data):
    """Legacy function - use ReturnsCalculator class instead."""
    global all_results, all_client_returns, brokerages

    for client, files in brokerages[broker_name].items():
        try:
            print(f"Processing {broker_name} data for {client}...")
            source_csv = files['source_csv']
            if isinstance(source_csv, (list, tuple)):
                for csv_path in source_csv:
                    process(csv_path)
            else:
                process(source_csv)

            # Processing logic
            output_dir = files['output_dir']
            if isinstance(output_dir, (list, tuple)):
                nav_frames = []
                trade_frames = []
                for out_dir in output_dir:
                    nav_part, trade_part = read_data(out_dir)
                    if isinstance(nav_part, pd.DataFrame) and not nav_part.empty:
                        nav_frames.append(nav_part)
                    if isinstance(trade_part, pd.DataFrame) and not trade_part.empty:
                        trade_frames.append(trade_part)
                if nav_frames:
                    nav_df = pd.concat(nav_frames, ignore_index=True)
                    nav_df['Date'] = pd.to_datetime(nav_df['Date']).dt.normalize()
                    nav_df = nav_df.sort_values('Date').drop_duplicates(subset=['Date'], keep='last')
                else:
                    nav_df = pd.DataFrame(columns=['Date', 'Net Asset Value'])
                if trade_frames:
                    trades_df = pd.concat(trade_frames, ignore_index=True)
                    trades_df['When'] = pd.to_datetime(trades_df['When']).dt.normalize()
                    trades_df = trades_df.sort_values('When').drop_duplicates()
                else:
                    trades_df = pd.DataFrame(columns=['When', 'Operation type', 'EUR equivalent'])
            else:
                nav_df, trades_df = read_data(output_dir)
            sub_period_returns = calculate_sub_period_returns(nav_df, trades_df)
            monthly_returns = calculate_monthly_twr(sub_period_returns)
            # Build daily series directly from NAV and flows to mirror reference TWR
            daily_returns = build_daily_returns_direct(nav_df, trades_df)
            # Use daily-compounded TWR for totals to ensure no days are missed
            absolute_return, annualized_return = calculate_total_returns_from_daily(daily_returns)

            results = {
                'monthly_returns': monthly_returns,
                'daily_returns': daily_returns,
                'absolute_return': absolute_return,
                'annualized_return': annualized_return,
                'sub_period_returns': sub_period_returns,
                'account_name': f'{broker_name}_{client}'
            }

            all_results[f'{broker_name}_{client}'] = results
            all_client_returns.append(results)
        except Exception as e:
            logging.error(f"Error processing {broker_name} client {client}: {e}")
            continue


def calculate_sub_period_returns(
    nav_df: pd.DataFrame, 
    trades_df: pd.DataFrame,
    large_flow_threshold: float = 0.10
) -> pd.DataFrame:
    """
    Calculate sub-period returns between each cash flow using TWR methodology.
    
    Also flags large cash flows (>threshold of NAV) for GIPS compliance.
    
    Args:
        nav_df: DataFrame with Date and Net Asset Value columns
        trades_df: DataFrame with When and flow amount columns
        large_flow_threshold: Threshold for flagging large flows (default 10%)
    
    Returns:
        DataFrame with sub-period returns and large flow flags
    """
    logging.info("Calculating sub-period returns")
    
    # Get unique dates
    all_dates = pd.concat([
        nav_df['Date']
    ]).sort_values().unique()
    
    logging.info(f"Found {len(all_dates)} unique dates for sub-period calculations")
    
    # Flag large cash flows using flow_utils
    if not trades_df.empty:
        trades_with_flags = flow_utils.flag_large_cash_flows(
            trades_df, nav_df, 
            threshold=large_flow_threshold,
            date_column='When',
            nav_date_column='Date',
            nav_value_column='Net Asset Value'
        )
    else:
        trades_with_flags = trades_df.copy()
        if 'large_flow_flag' not in trades_with_flags.columns:
            trades_with_flags['large_flow_flag'] = False
    
    returns = []
    for i in range(len(all_dates) - 1):
        start_date = all_dates[i]
        end_date = all_dates[i + 1]
        
        try:
            start_nav = nav_df[nav_df['Date'] == start_date]['Net Asset Value'].iloc[0]
            end_nav = nav_df[nav_df['Date'] == end_date]['Net Asset Value'].iloc[0]
        except IndexError:
            logging.warning(f"Missing NAV data for period {start_date} to {end_date}")
            continue
        
        # Get flows for period using flow_utils
        period_flows = flow_utils.get_flows_for_period(
            trades_with_flags, start_date, end_date, date_column='When'
        )
        
        # Use flow_utils to get the total flow
        total_flow = flow_utils.get_total_flow_for_period(
            trades_df, start_date, end_date, date_column='When'
        )
        
        # Check if period has large flows
        has_large_flow = False
        if not period_flows.empty and 'large_flow_flag' in period_flows.columns:
            has_large_flow = period_flows['large_flow_flag'].any()
        
        if start_nav <= 0:
            if start_nav == 0:
                if end_nav == 0 and total_flow == 0:
                    sub_period_return = 0.0
                elif total_flow != 0:
                    if total_flow > 0:
                        sub_period_return = (end_nav - total_flow) / total_flow
                        logging.info(
                            f"Computed return using flow as base for zero NAV period {start_date} to {end_date}"
                        )
                    else:
                        logging.warning(
                            f"Negative or zero flow with zero NAV for period {start_date} to {end_date}; skipping"
                        )
                        continue
                else:
                    logging.warning(f"Zero NAV with no flow for period {start_date} to {end_date}; skipping")
                    continue
            else:
                logging.warning(f"Invalid starting NAV ({start_nav}) for period {start_date}")
                continue
        else:
            sub_period_return = (end_nav - start_nav - total_flow) / start_nav

        returns.append({
            'start_date': start_date,
            'end_date': end_date,
            'return': sub_period_return,
            'start_nav': start_nav,
            'end_nav': end_nav,
            'total_flow': total_flow,
            'has_large_flow': has_large_flow
        })
    
    returns_df = pd.DataFrame(returns)
    
    # Drop rows where return, start_nav, end_nav are NaN and total_flow is 0.0
    initial_len = len(returns_df)
    returns_df = returns_df.dropna(
        subset=['return', 'start_nav', 'end_nav'],
        how='all'
    )
    # Replace the query with boolean indexing
    returns_df = returns_df[~((returns_df['total_flow'] == 0.0) & (returns_df['return'].isna()))]
    
    dropped_rows = initial_len - len(returns_df)
    if dropped_rows > 0:
        logging.info(f"Dropped {dropped_rows} invalid return records")
    
    # Log large flow periods
    if 'has_large_flow' in returns_df.columns:
        large_flow_periods = returns_df[returns_df['has_large_flow'] == True]
        if len(large_flow_periods) > 0:
            logging.warning(
                f"Found {len(large_flow_periods)} sub-periods with large cash flows "
                f"(>{large_flow_threshold*100:.0f}% of NAV)"
            )
    
    logging.info(f"Calculated {len(returns_df)} sub-period returns")
    return returns_df


def build_daily_returns_direct(nav_df: pd.DataFrame, trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build daily returns using the reference method:
      r_i = (end_nav - flow_on_end_date) / start_nav - 1
    where flows on the day are treated as occurring at the end of that day.
    """
    daily = build_account_daily_from_nav_flows(nav_df, trades_df)
    if daily.empty:
        return pd.DataFrame(columns=['date', 'return', 'start_nav', 'end_nav', 'flow'])

    daily = daily.sort_values('date')
    def compute_ret(row):
        start_v = float(row['start_nav']) if pd.notna(row['start_nav']) else 0.0
        end_v = float(row['end_nav']) if pd.notna(row['end_nav']) else start_v
        flow_v = float(row['flow']) if pd.notna(row['flow']) else 0.0
        
        if start_v > 0:
            # Normal case: calculate return based on starting NAV
            return (end_v - flow_v) / start_v - 1.0
        else:
            # No starting value and no flow - no return to calculate
            return 0.0

    daily['return'] = daily.apply(compute_ret, axis=1)
    return daily[['date', 'return', 'start_nav', 'end_nav', 'flow']]

def calculate_monthly_twr(sub_period_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate monthly TWR by geometrically linking daily-like sub-period returns.
    We attribute each sub-period's return to its end_date day, then compound all
    such daily values within the same calendar month. This prevents losing a
    cross-boundary sub-period that starts in one month and ends in the next.

    Adds start_of_month_nav, end_of_month_nav, and monthly_flow for auditability.
    """
    logging.info("Calculating monthly TWR from sub-period (daily) returns")

    if sub_period_returns.empty:
        logging.warning("No sub-period returns to calculate monthly TWR")
        return pd.DataFrame(
            columns=['month', 'return', 'start_of_month_nav', 'end_of_month_nav', 'monthly_flow', 'nav']
        )

    df = sub_period_returns.copy()
    df['end_date'] = pd.to_datetime(df['end_date']).dt.normalize()
    df = df.sort_values('end_date')
    # Month keyed by end_date so periods that end in a month are counted for that month
    df['month'] = df['end_date'].dt.to_period('M')

    monthly_rows: List[Dict] = []
    for month, in_month in df.groupby('month'):
        if in_month.empty:
            continue

        monthly_return = float(np.prod(1 + in_month['return']) - 1)
        if not np.isfinite(monthly_return):
            logging.warning(f"Invalid return calculation for month {month}")
            continue

        start_of_month_nav = float(in_month.iloc[0]['start_nav']) if 'start_nav' in in_month.columns else np.nan
        end_of_month_nav = float(in_month.iloc[-1]['end_nav']) if 'end_nav' in in_month.columns else np.nan
        monthly_flow = float(in_month['total_flow'].sum()) if 'total_flow' in in_month.columns else np.nan

        monthly_rows.append({
            'month': month,
            'return': monthly_return,
            'start_of_month_nav': start_of_month_nav,
            'end_of_month_nav': end_of_month_nav,
            'monthly_flow': monthly_flow,
            'nav': end_of_month_nav,
        })

    monthly_returns_df = pd.DataFrame(monthly_rows)
    logging.info(f"Calculated {len(monthly_returns_df)} monthly returns")
    return monthly_returns_df


def calculate_six_month_returns(monthly_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate rolling 6-month returns using geometric linking
    """
    if len(monthly_returns) < 6:
        logging.warning("Insufficient data for 6-month returns (need at least 6 months)")
        return pd.DataFrame(columns=['start_month', 'end_month', 'return'])

    print("Calculating 6-month rolling returns...")

    rolling_returns = []
    for i in range(len(monthly_returns) - 5):
        period = monthly_returns.iloc[i:i + 6]
        six_month_return = np.prod(1 + period['return']) - 1

        if not np.isfinite(six_month_return):
            logging.warning(f"Invalid 6-month return calculation starting {period.iloc[0]['month']}")
            continue

        rolling_returns.append({
            'start_month': period.iloc[0]['month'],
            'end_month': period.iloc[-1]['month'],
            'return': six_month_return
        })

    rolling_returns_df = pd.DataFrame(rolling_returns)
    logging.info(f"Calculated {len(rolling_returns_df)} 6-month rolling returns")
    return rolling_returns_df





def calculate_total_returns_from_subperiods(sub_period_returns: pd.DataFrame) -> Tuple[float, float]:
    """
    Calculate total absolute and annualized returns from sub-period returns.
    Annualization uses elapsed calendar time between first start_date and last end_date.
    """
    if sub_period_returns.empty:
        return 0.0, 0.0

    valid = sub_period_returns[np.isfinite(sub_period_returns['return'])]
    if valid.empty:
        return 0.0, 0.0

    absolute_return = float((1 + valid['return']).prod() - 1)

    start_dt = pd.to_datetime(sub_period_returns['start_date']).min()
    end_dt = pd.to_datetime(sub_period_returns['end_date']).max()
    elapsed_years = max((end_dt - start_dt).days, 0) / 365.25
    base = 1 + absolute_return
    if elapsed_years > 0 and base > 0:
        annualized_return = base ** (1 / elapsed_years) - 1
    else:
        if elapsed_years > 0 and base <= 0:
            logging.warning(
                "Annualized return undefined for total return <= -100%; "
                "using absolute return instead."
            )
        annualized_return = absolute_return

    return absolute_return, float(annualized_return)


def calculate_total_returns_from_daily(daily_returns: pd.DataFrame) -> Tuple[float, float]:
    """
    Calculate total absolute and annualized returns from a daily returns series.
    Annualization uses elapsed calendar time between first and last date.
    """
    if not isinstance(daily_returns, pd.DataFrame) or daily_returns.empty or 'return' not in daily_returns.columns:
        return 0.0, 0.0

    valid = daily_returns[np.isfinite(daily_returns['return'])]
    if valid.empty:
        return 0.0, 0.0

    absolute_return = float((1 + valid['return']).prod() - 1)
    start_dt = pd.to_datetime(valid['date']).min()
    end_dt = pd.to_datetime(valid['date']).max()
    elapsed_years = max((end_dt - start_dt).days, 0) / 365.25
    base = 1 + absolute_return
    if elapsed_years > 0 and base > 0:
        annualized_return = base ** (1 / elapsed_years) - 1
    else:
        if elapsed_years > 0 and base <= 0:
            logging.warning(
                "Annualized return undefined for total return <= -100%; "
                "using absolute return instead."
            )
        annualized_return = absolute_return

    return absolute_return, float(annualized_return)


def aggregate_client_returns(
    client_returns: List[Dict],
    results_dict: Optional[Dict] = None,
    config: Optional[Config] = None
):
    """
    Aggregate returns across multiple accounts using daily sub-period series.
    - Build a composite daily series by summing start/end NAV and flows, then
      applying the sub-period return formula at the composite level.
    - Derive composite monthly returns by geometrically linking daily returns.
    
    Args:
        client_returns: List of account result dictionaries
        results_dict: Optional dict to store results (uses global all_results if None)
    """
    global all_results
    
    # Use provided dict or fall back to global
    target_results = results_dict if results_dict is not None else all_results
    
    print("Aggregating client returns using daily composite method...")

    if not client_returns:
        logging.warning("No client returns to aggregate")
        return

    # Gather daily series per account
    daily_by_account: Dict[str, pd.DataFrame] = {}
    for account_data in client_returns:
        account_name = account_data.get('account_name', 'Unknown')
        daily_df = account_data.get('daily_returns')
        if isinstance(daily_df, pd.DataFrame) and not daily_df.empty:
            daily_by_account[account_name] = daily_df.copy()

    if not daily_by_account:
        logging.warning("No daily series available for composite aggregation")
        return

    # Composite daily via iterative summation
    composite_daily = build_daily_composite(daily_by_account)

    # Derive monthly from daily
    composite_daily['month'] = pd.to_datetime(composite_daily['date']).dt.to_period('M')
    monthly_rows: List[Dict] = []
    for month, grp in composite_daily.groupby('month'):
        monthly_return = (1 + grp['return']).prod() - 1
        start_of_month_nav = grp.iloc[0]['start_nav_total']
        end_of_month_nav = grp.iloc[-1]['end_nav_total']
        monthly_flow = grp['flow_total'].sum()
        monthly_rows.append({
            'month': month,
            'return': monthly_return,
            'start_of_month_nav': start_of_month_nav,
            'end_of_month_nav': end_of_month_nav,
            'monthly_flow': monthly_flow,
            'nav': end_of_month_nav,
        })
    composite_monthly = pd.DataFrame(monthly_rows)

    # Period returns and totals
    composite_df_for_periods = composite_monthly.rename(columns={'return': 'composite_return'})
    composite_df_for_periods['composite_growth'] = (1 + composite_df_for_periods['composite_return']).cumprod() - 1
    composite_df_for_periods['date'] = pd.to_datetime(composite_df_for_periods['month'].astype(str))

    # Use config-driven period windows
    period_windows = get_period_windows(config)
    period_returns: Dict = {}
    for label, (start_s, end_s) in period_windows.items():
        period_df = composite_df_for_periods[(composite_df_for_periods['date'] >= start_s) & (composite_df_for_periods['date'] <= end_s)]
        if not period_df.empty:
            ret = float((1 + period_df['composite_return']).prod() - 1)
            # Determine active accounts within the period window
            start_dt = pd.to_datetime(start_s)
            end_dt = pd.to_datetime(end_s)
            active_accounts: List[str] = []
            for acc_name, df in daily_by_account.items():
                if not isinstance(df, pd.DataFrame) or df.empty:
                    continue
                d = df.copy()
                d['date'] = pd.to_datetime(d['date']).dt.normalize()
                in_range = d[(d['date'] >= start_dt) & (d['date'] <= end_dt)]
                if in_range.empty:
                    continue
                # Consider account active if it had positive NAV in the window
                has_nav = False
                for col in ['start_nav', 'end_nav']:
                    if col in in_range.columns and np.isfinite(in_range[col]).any():
                        if (in_range[col] > 0).any():
                            has_nav = True
                            break
                if has_nav:
                    active_accounts.append(acc_name)
            period_returns[label] = {'return': ret, 'accounts': active_accounts}

    # Compute totals from daily to ensure exact TWR compounding
    abs_total, ann_total = calculate_total_returns_from_daily(composite_daily)

    target_results['Combined'] = {
        'monthly_returns': composite_monthly,
        'daily_returns': composite_daily[['date', 'return', 'start_nav_total', 'end_nav_total', 'flow_total']].rename(columns={
            'start_nav_total': 'start_nav',
            'end_nav_total': 'end_nav',
            'flow_total': 'flow',
        }),
        'absolute_return': abs_total,
        'annualized_return': float(ann_total),
        'period_returns': period_returns,
    }


def build_daily_composite(daily_by_account: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Given per-account daily series with columns date, return, start_nav, end_nav, flow,
    build a composite daily series using sub-period formula on totals.
    """
    # Normalize and collect date set
    date_set = set()
    normalized = {}
    for name, df in daily_by_account.items():
        d = df.copy()
        d['date'] = pd.to_datetime(d['date']).dt.normalize()
        d = d.sort_values('date')
        normalized[name] = d
        date_set.update(d['date'].tolist())

    if not date_set:
        return pd.DataFrame(columns=['date', 'return', 'start_nav_total', 'end_nav_total', 'flow_total'])

    all_dates = sorted(date_set)

    # Build maps for quick lookup and track last known end_nav per account
    per_account_by_date = {name: {row['date']: row for _, row in df.iterrows()} for name, df in normalized.items()}
    last_end_nav: Dict[str, float] = {}

    rows: List[Dict] = []
    for dt in all_dates:
        start_total = 0.0
        end_total = 0.0
        flow_total = 0.0

        for name, by_date in per_account_by_date.items():
            if dt in by_date:
                row = by_date[dt]
                start_nav = float(row['start_nav']) if pd.notna(row['start_nav']) else 0.0
                end_nav = float(row['end_nav']) if pd.notna(row['end_nav']) else start_nav
                flow = float(row.get('flow', 0.0))
                last_end_nav[name] = end_nav
            else:
                # If we have prior activity, carry forward NAV with 0 flow
                if name in last_end_nav:
                    start_nav = last_end_nav[name]
                    end_nav = start_nav
                    flow = 0.0
                else:
                    start_nav = 0.0
                    end_nav = 0.0
                    flow = 0.0

            start_total += start_nav
            end_total += end_nav
            flow_total += flow

        # Calculate composite return with end-of-day flows
        if start_total > 0:
            # Normal case: calculate return based on starting NAV
            daily_return = (end_total - start_total - flow_total) / start_total
        else:
            # No starting value and no flow - no return to calculate
            daily_return = 0.0
            
        rows.append({
            'date': dt,
            'return': daily_return,
            'start_nav_total': start_total,
            'end_nav_total': end_total,
            'flow_total': flow_total,
        })

    return pd.DataFrame(rows)


def build_gips_composite(monthly_returns_dfs: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, Dict]:
    """
    Build a GIPS-compliant composite from account-level monthly returns.
    Only includes accounts that were active during each period for period-specific returns.
    Uses NAV-weighted returns for each month and compounds them over time.
    """
    logging.info("Building GIPS composite from account returns")
    
    if not monthly_returns_dfs:
        logging.warning("No account returns provided for composite calculation")
        return pd.DataFrame(columns=['month', 'composite_return', 'composite_growth']), {}
    
    # Gather all unique months
    all_months = set()
    for df in monthly_returns_dfs.values():
        all_months.update(df['month'].unique())
    
    if not all_months:
        logging.warning("No months found in account returns")
        return pd.DataFrame(columns=['month', 'composite_return', 'composite_growth']), {}
    
    composite_results = []
    for month in sorted(all_months):
        active_accounts = []
        month_navs = {}  # Store end-of-month NAVs by account for this month
        month_returns = {}  # Store returns by account for this month
        start_month_navs = {}  # Store start-of-month NAVs by account if available
        month_flows = {}  # Store monthly flows by account if available
        
        # First gather all valid data for this month
        for account, df in monthly_returns_dfs.items():
            month_data = df[df['month'] == month]
            if not month_data.empty:
                monthly_return = month_data['return'].iloc[0]
                
                # Get start-of-month NAV for GIPS-compliant beginning-of-period weighting
                som_nav = None
                if 'start_of_month_nav' in month_data.columns:
                    som = month_data['start_of_month_nav'].iloc[0]
                    if np.isfinite(som) and som > 0:
                        som_nav = som
                
                # Get end-of-month NAV for reporting
                eom_nav = None
                if 'end_of_month_nav' in month_data.columns:
                    eom = month_data['end_of_month_nav'].iloc[0]
                    if np.isfinite(eom):
                        eom_nav = eom
                
                # Use start-of-month NAV for weighting (GIPS compliant)
                # Fall back to end-of-month only if start not available
                weighting_nav = som_nav if som_nav is not None else eom_nav
                
                if weighting_nav is not None and np.isfinite(monthly_return):
                    start_month_navs[account] = som_nav if som_nav is not None else 0.0
                    month_navs[account] = eom_nav if eom_nav is not None else 0.0
                    month_returns[account] = monthly_return
                    active_accounts.append(account)

                    if 'monthly_flow' in month_data.columns:
                        flow_val = month_data['monthly_flow'].iloc[0]
                        try:
                            flow_num = float(flow_val)
                        except Exception:
                            flow_num = np.nan
                        if np.isfinite(flow_num):
                            month_flows[account] = flow_num
        
        if start_month_navs and month_returns:  # Only proceed if we have valid data
            # Calculate total start-of-month NAV for GIPS-compliant weighting
            total_start_nav = sum(start_month_navs.values())
            total_end_nav = sum(month_navs.values()) if month_navs else np.nan
            monthly_flow_total = sum(month_flows.values()) if month_flows else np.nan
            
            # Calculate NAV-weighted composite return using BEGINNING-of-period weights
            # This is GIPS compliant - weights are based on start-of-month NAV
            if total_start_nav > 0:
                weighted_return = sum(
                    (start_month_navs[account] / total_start_nav) * month_returns[account]
                    for account in month_returns.keys()
                    if account in start_month_navs and start_month_navs[account] > 0
                )
            else:
                # Fallback to equal weighting if no start NAVs available
                weighted_return = np.mean(list(month_returns.values()))
            
            composite_results.append({
                'month': month,
                'composite_return': weighted_return,
                'total_nav': total_end_nav,  # End-of-month NAV for reporting
                'start_of_month_total_nav': total_start_nav,  # Used for weighting
                'monthly_flow_total': monthly_flow_total,
                'active_accounts': ','.join(active_accounts)
            })
    
    if not composite_results:
        logging.warning("No valid composite results calculated")
        return pd.DataFrame(columns=['month', 'composite_return', 'composite_growth']), {}
    
    composite_df = pd.DataFrame(composite_results)
    
    # Calculate cumulative growth through geometric linking
    composite_df['composite_growth'] = (1 + composite_df['composite_return']).cumprod() - 1
    
    # Add period returns calculations
    composite_df['date'] = pd.to_datetime(composite_df['month'].astype(str))
    
    # Calculate returns for specific periods using config-driven period windows
    period_returns = {}
    period_windows = get_period_windows()
    
    for period_label, (start_date, end_date) in period_windows.items():
        period_df = composite_df[
            (composite_df['date'] >= start_date) & 
            (composite_df['date'] <= end_date)
        ]
        if not period_df.empty:
        # Compound the monthly returns for the period
            period_return = np.prod(1 + period_df['composite_return']) - 1
            period_returns[period_label] = {
                'return': period_return,
                'accounts': list(set(','.join(period_df['active_accounts']).split(',')))
            }
            logging.info(f"{period_label} returns calculated using accounts: {period_returns[period_label]['accounts']}")
    logging.info(f"Calculated composite returns for {len(composite_df)} months")
    return composite_df, period_returns


def calculate_internal_dispersion(
    account_returns: Dict[str, float],
    account_navs: Optional[Dict[str, float]] = None,
    min_accounts: int = 6
) -> Optional[float]:
    """
    Calculate internal dispersion of account returns for GIPS compliance.
    
    Internal dispersion measures how individual account returns vary from the composite.
    GIPS requires this for composites with 6 or more accounts.
    
    This implementation uses equal-weighted standard deviation.
    For asset-weighted dispersion, provide account_navs.
    
    Args:
        account_returns: Dictionary mapping account names to their returns
        account_navs: Optional dictionary mapping account names to NAV for weighting
        min_accounts: Minimum number of accounts required (default 6 for GIPS)
    
    Returns:
        Standard deviation of returns, or None if fewer than min_accounts
    """
    if len(account_returns) < min_accounts:
        logging.info(
            f"Skipping dispersion calculation: {len(account_returns)} accounts "
            f"(minimum {min_accounts} required for GIPS)"
        )
        return None
    
    returns = np.array(list(account_returns.values()))
    
    if account_navs is not None and len(account_navs) == len(account_returns):
        # Asset-weighted standard deviation
        navs = np.array([account_navs.get(acc, 0) for acc in account_returns.keys()])
        total_nav = navs.sum()
        if total_nav > 0:
            weights = navs / total_nav
            weighted_mean = np.sum(weights * returns)
            variance = np.sum(weights * (returns - weighted_mean) ** 2)
            return float(np.sqrt(variance))
    
    # Equal-weighted standard deviation (sample std dev with ddof=1)
    return float(np.std(returns, ddof=1))


def calculate_composite_dispersion(
    monthly_returns_dfs: Dict[str, pd.DataFrame],
    period_start: str,
    period_end: str,
    min_accounts: int = 6
) -> Optional[float]:
    """
    Calculate internal dispersion for a specific period.
    
    Args:
        monthly_returns_dfs: Dictionary of account monthly returns DataFrames
        period_start: Start date of period (YYYY-MM-DD)
        period_end: End date of period (YYYY-MM-DD)
        min_accounts: Minimum accounts required
    
    Returns:
        Internal dispersion for the period, or None if not enough accounts
    """
    period_returns = {}
    period_navs = {}
    
    for account, df in monthly_returns_dfs.items():
        df = df.copy()
        df['date'] = pd.to_datetime(df['month'].astype(str))
        period_df = df[(df['date'] >= period_start) & (df['date'] <= period_end)]
        
        if not period_df.empty:
            # Calculate compound return for the period
            account_return = float((1 + period_df['return']).prod() - 1)
            period_returns[account] = account_return
            
            # Get average NAV for weighting
            if 'start_of_month_nav' in period_df.columns:
                avg_nav = period_df['start_of_month_nav'].mean()
                if np.isfinite(avg_nav):
                    period_navs[account] = avg_nav
    
    return calculate_internal_dispersion(period_returns, period_navs, min_accounts)


def build_synthetic_composite(
    nav_dfs: Dict[str, pd.DataFrame],
    flow_dfs: Dict[str, pd.DataFrame],
    flow_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Build a synthetic composite using daily values derived from NAV snapshots and flows.
    Steps:
      1) For each account, create a full daily series [date, start_nav, end_nav, flow]
         by carrying forward NAV on non-snapshot days and attributing flows on their dates.
      2) Build a composite daily series via totals and the sub-period formula.
      3) Aggregate composite daily to monthly using geometric linking.

    Returns DataFrame with columns: month (Period), return (float).
    """
    if not nav_dfs:
        return pd.DataFrame(columns=['month', 'return'])

    # Create per-account daily series from raw nav/flow data
    daily_by_account: Dict[str, pd.DataFrame] = {}
    for name, nav in nav_dfs.items():
        flows = flow_dfs.get(name, pd.DataFrame(columns=['When', 'EUR equivalent']))
        daily_by_account[name] = build_account_daily_from_nav_flows(nav, flows, flow_columns=flow_columns)

    # Build composite daily
    composite_daily = build_daily_composite(daily_by_account)
    if composite_daily.empty:
        return pd.DataFrame(columns=['month', 'return'])

    # Aggregate to monthly
    composite_daily['month'] = pd.to_datetime(composite_daily['date']).dt.to_period('M')
    monthly_rows: List[Dict] = []
    for month, grp in composite_daily.groupby('month'):
        monthly_return = float((1 + grp['return']).prod() - 1)
        monthly_rows.append({'month': month, 'return': monthly_return})
    return pd.DataFrame(monthly_rows)


def build_account_daily_from_nav_flows(
    nav_df: pd.DataFrame,
    flow_df: pd.DataFrame,
    flow_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Construct a daily series for a single account from NAV snapshots and flows.
    For each day in the continuous range between min(Date, When) and max(Date, When):
      - start_nav: prior day's end_nav (or 0 if none)
      - end_nav: NAV snapshot for the day if provided; else start_nav
      - flow: sum of flows occurring on that day
    """
    if not isinstance(nav_df, pd.DataFrame) or nav_df.empty:
        return pd.DataFrame(columns=['date', 'start_nav', 'end_nav', 'flow'])

    n = nav_df.copy()
    n['Date'] = pd.to_datetime(n['Date']).dt.normalize()
    n = n.sort_values('Date')
    nav_by_day = {d: float(v) for d, v in zip(n['Date'], n['Net Asset Value'])}

    if isinstance(flow_df, pd.DataFrame) and not flow_df.empty:
        f = flow_df.copy()
        f['When'] = pd.to_datetime(f['When']).dt.normalize()
        try:
            flow_col = flow_utils.get_flow_column(f, flow_columns)
            flows_by_day = f.groupby('When')[flow_col].sum().to_dict()
        except ValueError:
            flows_by_day = {}
        min_flow_date = f['When'].min()
        max_flow_date = f['When'].max()
    else:
        flows_by_day = {}
        min_flow_date = None
        max_flow_date = None

    min_date = n['Date'].min() if min_flow_date is None else min(n['Date'].min(), min_flow_date)
    max_date = n['Date'].max() if max_flow_date is None else max(n['Date'].max(), max_flow_date)

    rows: List[Dict] = []
    prev_end = 0.0
    for dt in pd.date_range(min_date, max_date, freq='D'):
        start_nav = prev_end
        end_nav = float(nav_by_day.get(dt, start_nav))
        flow = float(flows_by_day.get(dt, 0.0))
        rows.append({'date': dt.normalize(), 'start_nav': start_nav, 'end_nav': end_nav, 'flow': flow})
        prev_end = end_nav

    return pd.DataFrame(rows)





def save_all_results(results: Dict, output_dir: str = 'results', config: Optional[Config] = None) -> None:
    """
    Save results for all accounts and aggregations.
    
    Includes gross and net returns with fee breakdown.
    
    Args:
        results: Dictionary of results to save
        output_dir: Output directory path
        config: Optional configuration (uses defaults if not provided)
    """
    logging.info(f"Saving results to {output_dir}")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load fee configuration
    if config is None:
        try:
            config = load_config()
        except Exception:
            config = None

    report_currency = _get_report_currency(config)
    currency_symbol = _get_currency_symbol(report_currency)

    def fmt_currency(val: Optional[float]) -> str:
        if val is None or not np.isfinite(val):
            return ''
        return f'{currency_symbol}{val:,.2f}'
    
    # Get fee rates from config or use defaults
    if config is not None:
        annual_mgmt_fee = config.fees.management_fee_annual
        perf_fee_rate = config.fees.performance_fee_rate
        hurdle_rate = config.fees.hurdle_rate_annual
    else:
        annual_mgmt_fee = 0.01  # 1% default
        perf_fee_rate = 0.25   # 25% default
        hurdle_rate = 0.06     # 6% default
    
    monthly_mgmt_fee = annual_to_monthly_fee(annual_mgmt_fee)
    
    for category, data in results.items():
        category_dir = os.path.join(output_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        monthly_file = os.path.join(category_dir, f'monthly_returns_{timestamp}.xlsx')
        daily_file = os.path.join(category_dir, f'daily_returns_{timestamp}.xlsx')
        summary_file = os.path.join(category_dir, f'return_summary_{timestamp}.xlsx')
        
        # Ensure monthly returns exist; if missing or malformed, derive from daily
        monthly_returns_obj = data.get('monthly_returns')
        if isinstance(monthly_returns_obj, pd.DataFrame) and 'return' in monthly_returns_obj.columns:
            monthly_returns = monthly_returns_obj.copy()
        elif isinstance(data.get('daily_returns'), pd.DataFrame) and not data['daily_returns'].empty:
            dr = data['daily_returns'].copy()
            dr['date'] = pd.to_datetime(dr['date']).dt.normalize()
            dr['month'] = dr['date'].dt.to_period('M')
            monthly_rows: List[Dict] = []
            for month, grp in dr.groupby('month'):
                mret = (1 + grp['return']).prod() - 1
                som = grp.iloc[0].get('start_nav', np.nan)
                eom = grp.iloc[-1].get('end_nav', np.nan)
                mflow = grp.get('flow', pd.Series(dtype=float)).sum() if 'flow' in grp.columns else np.nan
                monthly_rows.append({'month': month, 'return': mret, 'start_of_month_nav': som, 'end_of_month_nav': eom, 'monthly_flow': mflow, 'nav': eom})
            monthly_returns = pd.DataFrame(monthly_rows)
        else:
            monthly_returns = pd.DataFrame(columns=['month', 'return', 'start_of_month_nav', 'end_of_month_nav', 'monthly_flow', 'nav'])

        # Prepare monthly data for fee tracking (GIPSFeeTracker)
        monthly_returns_for_fees = None
        monthly_net_returns = pd.DataFrame()
        nav_tracker = None
        fee_history = []
        quarterly_events = []
        daily_flows_for_fees = None
        initial_nav = 100.0
        initial_nav_display = None
        actual_final_gross_nav = 100.0
        total_flows = 0.0
        final_net_nav = initial_nav
        total_mgmt_fees = 0.0
        total_perf_fees = 0.0
        fee_drag = 0.0

        # Build monthly data with NAV and flows for fee tracking
        if isinstance(data.get('daily_returns'), pd.DataFrame) and not data['daily_returns'].empty:
            dr = data['daily_returns'].copy()
            dr['date'] = pd.to_datetime(dr['date']).dt.normalize()
            dr['month'] = dr['date'].dt.to_period('M')

            flow_col = 'flow'
            if flow_col not in dr.columns and 'flow_total' in dr.columns:
                flow_col = 'flow_total'
            if flow_col in dr.columns:
                daily_flows_for_fees = dr[['date', flow_col]].rename(columns={flow_col: 'flow'})

            monthly_rows: List[Dict] = []
            for month, grp in dr.groupby('month'):
                grp = grp.sort_values('date')
                mret = (1 + grp['return']).prod() - 1

                if 'start_nav_total' in grp.columns:
                    start_nav = grp.iloc[0]['start_nav_total']
                    end_nav = grp.iloc[-1]['end_nav_total'] if 'end_nav_total' in grp.columns else start_nav
                elif 'start_nav' in grp.columns:
                    start_nav = grp.iloc[0]['start_nav']
                    end_nav = grp.iloc[-1]['end_nav'] if 'end_nav' in grp.columns else start_nav
                else:
                    start_nav = np.nan
                    end_nav = np.nan

                if 'flow_total' in grp.columns:
                    month_flow = grp['flow_total'].sum()
                elif 'flow' in grp.columns:
                    month_flow = grp['flow'].sum()
                else:
                    month_flow = 0

                monthly_rows.append({
                    'month': month,
                    'return': mret,
                    'start_nav': start_nav if pd.notna(start_nav) else 0,
                    'end_nav': end_nav if pd.notna(end_nav) else 0,
                    'flow': month_flow if pd.notna(month_flow) else 0
                })

            monthly_returns_for_fees = pd.DataFrame(monthly_rows)

            if not monthly_returns_for_fees.empty:
                first_nav = monthly_returns_for_fees['start_nav'].iloc[0]
                if pd.notna(first_nav) and first_nav > 0:
                    initial_nav = first_nav
                last_nav = monthly_returns_for_fees['end_nav'].iloc[-1]
                if pd.notna(last_nav) and last_nav > 0:
                    actual_final_gross_nav = last_nav
                total_flows = monthly_returns_for_fees['flow'].sum()
        elif isinstance(monthly_returns, pd.DataFrame) and 'return' in monthly_returns.columns and not monthly_returns.empty:
            monthly_returns_for_fees = monthly_returns.rename(columns={
                'start_of_month_nav': 'start_nav',
                'end_of_month_nav': 'end_nav',
                'monthly_flow': 'flow'
            })[['month', 'return', 'start_nav', 'end_nav', 'flow']].copy()
            if 'start_nav' in monthly_returns_for_fees.columns:
                first_nav = monthly_returns_for_fees['start_nav'].iloc[0]
                if pd.notna(first_nav) and first_nav > 0:
                    initial_nav = first_nav
            if 'end_nav' in monthly_returns_for_fees.columns:
                last_nav = monthly_returns_for_fees['end_nav'].iloc[-1]
                if pd.notna(last_nav) and last_nav > 0:
                    actual_final_gross_nav = last_nav

        # Determine initial NAV for reporting (first positive actual NAV)
        if isinstance(data.get('daily_returns'), pd.DataFrame) and not data['daily_returns'].empty:
            dr_nav = data['daily_returns'].copy()
            dr_nav['date'] = pd.to_datetime(dr_nav['date']).dt.normalize()
            nav_col = 'end_nav'
            if nav_col not in dr_nav.columns and 'end_nav_total' in dr_nav.columns:
                nav_col = 'end_nav_total'
            if nav_col in dr_nav.columns:
                pos_navs = dr_nav[dr_nav[nav_col] > 0].sort_values('date')
                if not pos_navs.empty:
                    initial_nav_display = float(pos_navs.iloc[0][nav_col])

        if (initial_nav_display is None or not np.isfinite(initial_nav_display) or initial_nav_display <= 0):
            if monthly_returns_for_fees is not None and not monthly_returns_for_fees.empty:
                pos_end = monthly_returns_for_fees[monthly_returns_for_fees['end_nav'] > 0]
                if not pos_end.empty:
                    initial_nav_display = float(pos_end.iloc[0]['end_nav'])
                else:
                    pos_start = monthly_returns_for_fees[monthly_returns_for_fees['start_nav'] > 0]
                    if not pos_start.empty:
                        initial_nav_display = float(pos_start.iloc[0]['start_nav'])

        if initial_nav_display is None or not np.isfinite(initial_nav_display) or initial_nav_display <= 0:
            initial_nav_display = initial_nav

        # Run fee tracker if we have monthly data with NAV/flows
        # Default fiscal year starts in January (calendar year) unless config overrides.
        fiscal_year_start_month = 1
        quarterly_mgmt_fee = 0.0025  # 0.25% per quarter
        quarterly_hurdle = 0.015     # 1.5% per quarter
        if config is not None:
            fiscal_year_start_month = config.periods.fiscal_year_start_month
            quarterly_mgmt_fee = config.fees.management_fee_quarterly
            quarterly_hurdle = hurdle_rate / 4

        if monthly_returns_for_fees is not None and not monthly_returns_for_fees.empty:
            nav_tracker = GIPSFeeTracker(
                initial_nav=initial_nav,
                quarterly_mgmt_fee=quarterly_mgmt_fee,
                quarterly_hurdle_rate=quarterly_hurdle,
                perf_fee_rate=perf_fee_rate,
                fiscal_year_start_month=fiscal_year_start_month
            )

            fee_result = nav_tracker.process_monthly_data(
                monthly_returns_for_fees,
                month_column='month',
                gross_nav_column='end_nav',
                flow_column='flow',
                return_column='return',
                daily_flows=daily_flows_for_fees
            )

            initial_nav = fee_result['initial_nav']
            actual_final_gross_nav = fee_result['final_gross_nav']
            final_net_nav = fee_result['final_net_nav']
            total_mgmt_fees = fee_result['total_mgmt_fees']
            total_perf_fees = fee_result['total_perf_fees']
            total_flows = fee_result['total_flows']
            fee_drag = fee_result['total_fees']
            fee_history = fee_result.get('history', [])
            quarterly_events = fee_result.get('quarterly_events', [])
            monthly_net_returns = fee_result.get('monthly_net_returns', pd.DataFrame())
            flow_proration_warning = fee_result.get('flow_proration_warning')
            if not monthly_net_returns.empty:
                monthly_net_returns = monthly_net_returns.copy()
                monthly_net_returns['period'] = pd.to_datetime(
                    dict(year=monthly_net_returns['year'], month=monthly_net_returns['month'], day=1)
                ).dt.to_period('M')
        else:
            flow_proration_warning = None

        # Calculate gross and net returns with GIPS-compliant fee handling
        if 'return' in monthly_returns.columns and not monthly_returns.empty:
            raw_returns = monthly_returns['return'].copy()
            monthly_returns['gross_return'] = raw_returns
            monthly_returns['gross_return_raw'] = raw_returns

            net_return_map = {}
            if not monthly_net_returns.empty:
                net_return_map = dict(zip(monthly_net_returns['period'], monthly_net_returns['net_return']))

            def month_number(val):
                if hasattr(val, 'month'):
                    return val.month
                try:
                    return pd.Period(val).month
                except Exception:
                    return None

            monthly_returns['net_return'] = monthly_returns['month'].map(net_return_map)
            monthly_returns['net_return_raw'] = monthly_returns['net_return']
            if nav_tracker is not None:
                quarter_starts = set(nav_tracker.quarter_start_months)
                monthly_returns['management_fee'] = monthly_returns['month'].apply(
                    lambda p: quarterly_mgmt_fee if month_number(p) in quarter_starts else 0.0
                )
            else:
                monthly_returns['management_fee'] = np.nan
            monthly_returns['management_fee_raw'] = monthly_returns['management_fee']

            def fmt_pct(val):
                return '{:.4%}'.format(val) if pd.notna(val) else ''

            monthly_returns['gross_return'] = monthly_returns['gross_return'].map(fmt_pct)
            monthly_returns['net_return'] = monthly_returns['net_return'].map(fmt_pct)
            monthly_returns['management_fee'] = monthly_returns['management_fee'].map(fmt_pct)

            # Keep original return column for compatibility
            monthly_returns['return'] = monthly_returns['gross_return']
        
        # Format NAV/flow columns as currency if they exist
        if 'nav' in monthly_returns.columns:
            monthly_returns['nav'] = monthly_returns['nav'].map(fmt_currency)
        if 'end_of_month_nav' in monthly_returns.columns:
            monthly_returns['end_of_month_nav'] = monthly_returns['end_of_month_nav'].map(fmt_currency)
        if 'start_of_month_nav' in monthly_returns.columns:
            monthly_returns['start_of_month_nav'] = monthly_returns['start_of_month_nav'].map(fmt_currency)
        if 'monthly_flow' in monthly_returns.columns:
            monthly_returns['monthly_flow'] = monthly_returns['monthly_flow'].map(fmt_currency)
        
        # Save to Excel (Monthly)
        with pd.ExcelWriter(monthly_file, engine='openpyxl') as writer:
            monthly_returns.to_excel(writer, index=False)
            # Set column width for better readability
            worksheet = writer.sheets['Sheet1']
            worksheet.column_dimensions['B'].width = 15
            # Set wider columns for NAV/flow data if present
            if 'nav' in monthly_returns.columns:
                nav_col = chr(ord('A') + list(monthly_returns.columns).index('nav'))
                worksheet.column_dimensions[nav_col].width = 18
            if 'end_of_month_nav' in monthly_returns.columns:
                nav_col = chr(ord('A') + list(monthly_returns.columns).index('end_of_month_nav'))
                worksheet.column_dimensions[nav_col].width = 18
            if 'start_of_month_nav' in monthly_returns.columns:
                nav_col = chr(ord('A') + list(monthly_returns.columns).index('start_of_month_nav'))
                worksheet.column_dimensions[nav_col].width = 18
            if 'monthly_flow' in monthly_returns.columns:
                nav_col = chr(ord('A') + list(monthly_returns.columns).index('monthly_flow'))
                worksheet.column_dimensions[nav_col].width = 18
        
        # Save to Excel (Daily) if present
        if 'daily_returns' in data and isinstance(data['daily_returns'], pd.DataFrame):
            daily_returns = data['daily_returns'].copy()
            if not daily_returns.empty and 'return' in daily_returns.columns:
                daily_returns['return'] = daily_returns['return'].map('{:.4%}'.format)
            # Format NAV/flow
            for col in ['start_nav', 'end_nav', 'flow']:
                if col in daily_returns.columns:
                    daily_returns[col] = daily_returns[col].map(fmt_currency)
            with pd.ExcelWriter(daily_file, engine='openpyxl') as writer:
                daily_returns.to_excel(writer, index=False)
                worksheet = writer.sheets['Sheet1']
                # Set useful widths
                if 'date' in daily_returns.columns:
                    worksheet.column_dimensions['A'].width = 15
                # Widen numeric columns
                for idx, col in enumerate(daily_returns.columns, start=1):
                    if col in ['start_nav', 'end_nav', 'flow']:
                        worksheet.column_dimensions[chr(ord('A') + idx - 1)].width = 18
        
        # Create summary DataFrame with all return metrics based on daily compounding
        summary_data = []
        
        # Calculate gross and net total returns
        gross_absolute = data.get("absolute_return", 0.0)
        gross_annualized = data.get("annualized_return", 0.0)
        
        # Get quarterly management fee from config
        quarterly_mgmt_fee = 0.0025  # Default 0.25% per quarter
        if config is not None and hasattr(config.fees, 'management_fee_quarterly'):
            quarterly_mgmt_fee = config.fees.management_fee_quarterly
        
        # Get fiscal year start month
        # Default fiscal year starts in January (calendar year) unless config overrides.
        fiscal_year_start_month = 1
        if config is not None:
            fiscal_year_start_month = config.periods.fiscal_year_start_month
        
        # Calculate number of years for annualization
        if 'daily_returns' in data and isinstance(data['daily_returns'], pd.DataFrame):
            num_months = len(data['daily_returns']['date'].dt.to_period('M').unique()) if 'date' in data['daily_returns'].columns else 12
        else:
            num_months = 12  # Default assumption
        
        num_years = max(1, num_months / 12)
        
        net_absolute = gross_absolute
        net_annualized = gross_annualized
        fallback_used = False
        fallback_range = None
        if not monthly_net_returns.empty:
            net_absolute, net_annualized = calculate_net_twr_from_monthly(
                monthly_net_returns,
                return_column='net_return'
            )
        else:
            fallback_used = True
            if isinstance(data.get('daily_returns'), pd.DataFrame) and not data['daily_returns'].empty:
                dr_range = data['daily_returns'].copy()
                dr_range['date'] = pd.to_datetime(dr_range['date']).dt.normalize()
                start_dt = dr_range['date'].min()
                end_dt = dr_range['date'].max()
                fallback_range = (start_dt, end_dt)
            elif monthly_returns_for_fees is not None and not monthly_returns_for_fees.empty:
                month_vals = pd.to_datetime(monthly_returns_for_fees['month'].astype(str), errors='coerce').dropna()
                if not month_vals.empty:
                    start_dt = month_vals.min()
                    end_dt = (month_vals.max() + pd.offsets.MonthEnd(0)).normalize()
                    fallback_range = (start_dt, end_dt)

            total_fee_result = calculate_period_net_return(
                gross_return=gross_absolute,
                quarterly_mgmt_fee=quarterly_mgmt_fee,
                quarterly_hurdle=hurdle_rate / 4,
                perf_fee_rate=perf_fee_rate,
                num_quarters=max(1, int(num_months / 3))
            )
            net_absolute = total_fee_result['net_return']
            total_mgmt_fees = total_fee_result['mgmt_fee_impact'] * initial_nav
            total_perf_fees = total_fee_result['perf_fee_impact'] * initial_nav
            fee_drag = total_mgmt_fees + total_perf_fees
            final_net_nav = initial_nav * (1 + net_absolute)
            if net_absolute > -1:
                net_annualized = (1 + net_absolute) ** (1 / num_years) - 1
            else:
                net_annualized = -1.0

        def fmt_pct_value(val: Optional[float], default: str = 'N/A') -> str:
            if val is None:
                return default
            if not np.isfinite(val):
                return default
            return f'{val:.4%}'

        tracker_summary = nav_tracker.get_summary() if nav_tracker is not None else {}
        net_index_current = tracker_summary.get('net_index') if tracker_summary else None
        hwm_index_current = tracker_summary.get('high_water_mark') if tracker_summary else None
        cumulative_hurdle_pct_current = tracker_summary.get('cumulative_hurdle_pct') if tracker_summary else None
        hurdle_threshold_return = None
        hurdle_threshold_index = tracker_summary.get('hurdle_threshold') if tracker_summary else None
        if hurdle_threshold_index is not None:
            hurdle_threshold_return = hurdle_threshold_index - 1
        
        # Add section header
        summary_data.append({'Metric': '=== GROSS RETURNS (Before Fees) ===', 'Value': ''})
        
        summary_data.extend([
            {
                'Metric': 'Total Absolute Return (Gross TWR)',
                'Value': f'{gross_absolute:.4%}'
            },
            {
                'Metric': 'Annualized Return (Gross TWR)',
                'Value': f'{gross_annualized:.4%}'
            }
        ])
        
        # Add blank row for spacing
        summary_data.append({'Metric': '', 'Value': ''})
        summary_data.append({'Metric': '=== NET RETURNS (After Fees) ===', 'Value': ''})
        
        summary_data.extend([
            {
                'Metric': 'Total Absolute Return (Net TWR)',
                'Value': f'{net_absolute:.4%}'
            },
            {
                'Metric': 'Annualized Return (Net TWR)',
                'Value': f'{net_annualized:.4%}'
            },
        ])

        if fallback_used:
            if fallback_range is not None:
                start_str = fallback_range[0].strftime('%Y-%m-%d')
                end_str = fallback_range[1].strftime('%Y-%m-%d')
                warning_text = (
                    "WARNING: Net returns use fallback fee simulation "
                    f"(no flow proration) for {start_str} to {end_str}"
                )
            else:
                warning_text = (
                    "WARNING: Net returns use fallback fee simulation "
                    "(no flow proration); period unavailable"
                )
            summary_data.append({'Metric': warning_text, 'Value': ''})
            logging.warning(warning_text)

        if flow_proration_warning:
            summary_data.append({'Metric': flow_proration_warning, 'Value': ''})
            logging.error(flow_proration_warning)

        # Always include NAV/fee summary metrics (critical for interpretation),
        # regardless of warning state.
        summary_data.extend([
            {
                'Metric': 'Initial NAV (Actual)',
                'Value': fmt_currency(initial_nav_display)
            },
            {
                'Metric': 'Final NAV - Gross (Actual from Accounts)',
                'Value': fmt_currency(actual_final_gross_nav)
            },
            {
                'Metric': 'Final NAV - Net (After Fees)',
                'Value': fmt_currency(final_net_nav)
            },
            {
                'Metric': 'Total Flows (In/Out)',
                'Value': fmt_currency(total_flows)
            },
            {
                'Metric': 'Total Management Fees Paid',
                'Value': fmt_currency(total_mgmt_fees)
            },
            {
                'Metric': 'Total Performance Fees Paid',
                'Value': fmt_currency(total_perf_fees)
            },
            {
                'Metric': 'Fee Impact (Gross - Net)',
                'Value': fmt_currency(fee_drag)
            }
        ])
        
        # Add fee structure section
        summary_data.append({'Metric': '', 'Value': ''})
        summary_data.append({'Metric': '=== FEE STRUCTURE ===', 'Value': ''})
        summary_data.extend([
            {
                'Metric': 'Annual Management Fee',
                'Value': f'{annual_mgmt_fee:.2%}'
            },
            {
                'Metric': 'Monthly Management Fee (equivalent)',
                'Value': f'{monthly_mgmt_fee:.4%}'
            },
            {
                'Metric': 'Performance Fee Rate',
                'Value': f'{perf_fee_rate:.0%} of gains above hurdle'
            },
            {
                'Metric': 'Hurdle Rate (Annual)',
                'Value': f'{hurdle_rate:.2%}'
            },
            {
                'Metric': 'Hurdle Mechanism',
                'Value': 'Cumulative quarterly hurdle (no reset)'
            }
        ])
        
        # Add detailed quarterly NAV tracking section
        summary_data.append({'Metric': '', 'Value': ''})
        summary_data.append({'Metric': '=== QUARTERLY NAV TRACKING (Gross vs Net NAV with Flows) ===', 'Value': ''})
        
        # Use quarterly events from the GIPSFeeTracker for accurate tracking
        if quarterly_events:
            # Group events by quarter
            quarterly_actual = {}
            annual_summary = {}
            
            for event in quarterly_events:
                fy = event.get('fiscal_year', 'Unknown')
                q = event.get('quarter', 1)
                key = f"{fy}_Q{q}"
                
                if key not in quarterly_actual:
                    quarterly_actual[key] = {
                        'fiscal_year': fy,
                        'quarter': q,
                        'gross_nav': event.get('gross_nav', 0),
                        'mgmt_fee': 0,
                        'perf_fee': 0,
                        'net_nav_after_mgmt': 0,
                        'net_nav_after_perf': 0,
                        'high_water_mark': 0.0,
                        'cumulative_hurdle_pct': 0.0,
                        'net_index_after': 0.0,
                        'gross_return_factor': 1.0
                    }
                
                if event['event'] in ('management_fee', 'management_fee_adjustment'):
                    quarterly_actual[key]['gross_nav'] = event.get('gross_nav', 0)
                    quarterly_actual[key]['mgmt_fee'] += event.get('fee_amount', 0)
                    quarterly_actual[key]['net_nav_after_mgmt'] = event.get('net_nav_after', 0)
                    quarterly_actual[key]['high_water_mark'] = event.get('high_water_mark', 0.0)
                    quarterly_actual[key]['cumulative_hurdle_pct'] = event.get('cumulative_hurdle_pct', 0.0)
                    quarterly_actual[key]['net_index_after'] = event.get('net_index_after', 0.0)
                
                elif event['event'] == 'performance_fee':
                    quarterly_actual[key]['perf_fee'] = event.get('fee_amount', 0)
                    quarterly_actual[key]['net_nav_after_perf'] = event.get('net_nav_after', 0)
                    quarterly_actual[key]['excess_gain'] = event.get('excess_gain', 0)
                    quarterly_actual[key]['effective_hurdle'] = event.get('effective_hurdle', 0)
                    quarterly_actual[key]['hurdle_threshold'] = event.get('hurdle_threshold', 0.0)
                    quarterly_actual[key]['high_water_mark'] = event.get('high_water_mark_after', 0.0)
                    quarterly_actual[key]['cumulative_hurdle_pct'] = event.get('accumulated_hurdle_pct', 0.0)
                    quarterly_actual[key]['net_index_after'] = event.get('net_index_after', 0.0)
            
            # Get monthly NAV and flow data for each quarter
            if monthly_returns_for_fees is not None:
                for _, row in monthly_returns_for_fees.iterrows():
                    month_period = row['month']
                    if hasattr(month_period, 'month'):
                        month = month_period.month
                        year = month_period.year
                    else:
                        try:
                            period = pd.Period(month_period)
                            month = period.month
                            year = period.year
                        except:
                            continue
                    
                    if month >= fiscal_year_start_month:
                        fy = str(year)
                    else:
                        fy = str(year - 1)
                    
                    months_since_start = (month - fiscal_year_start_month) % 12
                    q = (months_since_start // 3) + 1
                    key = f"{fy}_Q{q}"
                    
                    if key in quarterly_actual:
                        end_nav_val = row.get('end_nav', 0)
                        flow_val = row.get('flow', 0)
                        monthly_return = row.get('return', 0)
                        if pd.notna(end_nav_val):
                            quarterly_actual[key]['end_nav_gross'] = end_nav_val
                        if pd.notna(flow_val):
                            quarterly_actual[key]['total_flows'] = quarterly_actual[key].get('total_flows', 0) + flow_val
                        if pd.notna(monthly_return):
                            try:
                                monthly_return = float(monthly_return)
                                quarterly_actual[key]['gross_return_factor'] *= (1 + monthly_return)
                            except Exception:
                                pass
            
            # Add initial NAV row
            summary_data.append({
                'Metric': 'Starting NAV (Actual Combined)',
                'Value': fmt_currency(initial_nav_display)
            })
            summary_data.append({'Metric': '', 'Value': ''})
            
            # Display quarterly breakdown with actual NAV values
            # Build ordered quarter list for sequential calculations
            def _quarter_sort_key(k: str) -> tuple:
                try:
                    fy, q_part = k.split('_Q')
                    return (int(fy), int(q_part))
                except Exception:
                    return (0, 0)

            ordered_quarters = sorted(quarterly_actual.keys(), key=_quarter_sort_key)
            computed_hurdles = {}
            prev_hurdle = None
            prev_net_index = None
            quarterly_hurdle_rate = hurdle_rate / 4 if hurdle_rate is not None else 0.015

            for key in ordered_quarters:
                q_data = quarterly_actual[key]
                existing = q_data.get('cumulative_hurdle_pct', 0.0) or 0.0
                if prev_hurdle is None:
                    current_hurdle = existing if existing > 0 else quarterly_hurdle_rate
                else:
                    current_hurdle = max(existing, prev_hurdle + quarterly_hurdle_rate)
                computed_hurdles[key] = current_hurdle
                prev_hurdle = current_hurdle

                net_index_after = q_data.get('net_index_after', None)
                if net_index_after is not None and prev_net_index is not None and prev_net_index > 0:
                    q_data['net_return_quarter'] = (net_index_after / prev_net_index) - 1
                else:
                    q_data['net_return_quarter'] = None
                if net_index_after is not None and net_index_after > 0:
                    prev_net_index = net_index_after

            for key in ordered_quarters:
                q_data = quarterly_actual[key]
                fy = q_data['fiscal_year']
                q = q_data['quarter']
                
                # Get NAV values
                gross_nav = q_data.get('gross_nav', 0)
                end_nav_gross = q_data.get('end_nav_gross', gross_nav)
                total_flows = q_data.get('total_flows', 0)
                net_nav_end = q_data.get('net_nav_after_perf', q_data.get('net_nav_after_mgmt', 0))
                
                # Initialize annual summary
                if fy not in annual_summary:
                    annual_summary[fy] = {
                        'mgmt_fee': 0, 
                        'perf_fee': 0, 
                        'start_nav': None, 
                        'end_nav_gross': 0,
                        'end_nav_net': 0,
                        'total_flows': 0
                    }
                
                if annual_summary[fy]['start_nav'] is None:
                    annual_summary[fy]['start_nav'] = gross_nav
                
                annual_summary[fy]['mgmt_fee'] += q_data['mgmt_fee']
                annual_summary[fy]['perf_fee'] += q_data['perf_fee']
                annual_summary[fy]['end_nav_gross'] = end_nav_gross
                annual_summary[fy]['end_nav_net'] = net_nav_end
                annual_summary[fy]['total_flows'] += total_flows
                
                summary_data.extend([
                    {
                        'Metric': f'{fy} Q{q} - Gross NAV (Actual)',
                        'Value': fmt_currency(end_nav_gross)
                    },
                    {
                        'Metric': f'{fy} Q{q} - Flows (In/Out)',
                        'Value': fmt_currency(total_flows)
                    },
                    {
                        'Metric': f'{fy} Q{q} - Management Fee (0.25%)',
                        'Value': fmt_currency(q_data["mgmt_fee"])
                    }
                ])
                
                if q_data['perf_fee'] > 0:
                    summary_data.append({
                        'Metric': f'{fy} Q{q} - Performance Fee (25% excess)',
                        'Value': fmt_currency(q_data["perf_fee"])
                    })
                else:
                    summary_data.append({
                        'Metric': f'{fy} Q{q} - Performance Fee',
                        'Value': 'None (below hurdle)'
                    })
                
                if net_nav_end > 0:
                    summary_data.append({
                        'Metric': f'{fy} Q{q} - Net NAV (After Fees)',
                        'Value': fmt_currency(net_nav_end)
                    })
                    # Show fee drag for the quarter
                    fee_drag_q = end_nav_gross - net_nav_end
                    if fee_drag_q > 0:
                        summary_data.append({
                            'Metric': f'{fy} Q{q} - Fee Impact (Cumulative)',
                            'Value': fmt_currency(fee_drag_q)
                        })

                net_index_after = q_data.get('net_index_after', None)
                hwm_index = q_data.get('high_water_mark', None)
                hurdle_threshold_index = q_data.get('hurdle_threshold', None)
                cumulative_hurdle_pct = computed_hurdles.get(key, q_data.get('cumulative_hurdle_pct', None))
                quarter_return = q_data.get('net_return_quarter', None)
                gross_quarter_return = None
                gross_factor = q_data.get('gross_return_factor', None)
                if gross_factor is not None:
                    try:
                        gross_quarter_return = float(gross_factor) - 1
                    except Exception:
                        gross_quarter_return = None

                if net_index_after is not None and net_index_after > 0:
                    summary_data.append({
                        'Metric': f'{fy} Q{q} - Net Total Return to Date',
                        'Value': fmt_pct_value(net_index_after - 1)
                    })
                if gross_quarter_return is not None:
                    summary_data.append({
                        'Metric': f'{fy} Q{q} - Gross Return (Quarter)',
                        'Value': fmt_pct_value(gross_quarter_return)
                    })
                if quarter_return is not None:
                    summary_data.append({
                        'Metric': f'{fy} Q{q} - Net Return (Quarter)',
                        'Value': fmt_pct_value(quarter_return)
                    })
                if hurdle_threshold_index is not None and hurdle_threshold_index > 0:
                    summary_data.append({
                        'Metric': f'{fy} Q{q} - High Water Mark (Hurdle-Adjusted, Net Return to Date)',
                        'Value': fmt_pct_value(hurdle_threshold_index - 1)
                    })
                if hwm_index is not None and hwm_index > 0:
                    summary_data.append({
                        'Metric': f'{fy} Q{q} - High Water Mark (Base, Net Return to Date)',
                        'Value': fmt_pct_value(hwm_index - 1)
                    })
                if cumulative_hurdle_pct is not None:
                    summary_data.append({
                        'Metric': f'{fy} Q{q} - Cumulative Hurdle (Since Inception)',
                        'Value': fmt_pct_value(cumulative_hurdle_pct)
                    })
                
                summary_data.append({'Metric': '', 'Value': ''})
            
            # Add annual fee totals
            summary_data.append({'Metric': '=== ANNUAL FEE TOTALS ===', 'Value': ''})
            
            cumulative_mgmt = 0
            cumulative_perf = 0
            cumulative_flows = 0
            
            for fy in sorted(annual_summary.keys()):
                a_data = annual_summary[fy]
                cumulative_mgmt += a_data['mgmt_fee']
                cumulative_perf += a_data['perf_fee']
                cumulative_flows += a_data['total_flows']
                total_annual_fees = a_data['mgmt_fee'] + a_data['perf_fee']
                
                summary_data.extend([
                    {
                        'Metric': f'{fy} - Start NAV (Actual)',
                        'Value': fmt_currency(a_data["start_nav"]) if a_data["start_nav"] else 'N/A'
                    },
                    {
                        'Metric': f'{fy} - End NAV Gross (Actual)',
                        'Value': fmt_currency(a_data["end_nav_gross"])
                    },
                    {
                        'Metric': f'{fy} - End NAV Net (After Fees)',
                        'Value': fmt_currency(a_data["end_nav_net"]) if a_data["end_nav_net"] > 0 else 'N/A'
                    },
                    {
                        'Metric': f'{fy} - Total Flows',
                        'Value': fmt_currency(a_data["total_flows"])
                    },
                    {
                        'Metric': f'{fy} - Management Fees',
                        'Value': fmt_currency(a_data["mgmt_fee"])
                    },
                    {
                        'Metric': f'{fy} - Performance Fees',
                        'Value': fmt_currency(a_data["perf_fee"])
                    },
                    {
                        'Metric': f'{fy} - Total Fees',
                        'Value': fmt_currency(total_annual_fees)
                    },
                    {'Metric': '', 'Value': ''}
                ])
            
            # Add cumulative totals
            summary_data.append({'Metric': '=== CUMULATIVE TOTALS ===', 'Value': ''})
            summary_data.extend([
                {
                    'Metric': 'Total Management Fees (All Years)',
                    'Value': fmt_currency(cumulative_mgmt)
                },
                {
                    'Metric': 'Total Performance Fees (All Years)',
                    'Value': fmt_currency(cumulative_perf)
                },
                {
                    'Metric': 'Total All Fees',
                    'Value': fmt_currency(cumulative_mgmt + cumulative_perf)
                },
                {
                    'Metric': 'Final NAV - Gross (Actual from Accounts)',
                    'Value': fmt_currency(actual_final_gross_nav)
                },
                {
                    'Metric': 'Final NAV - Net (After All Fees)',
                    'Value': fmt_currency(final_net_nav)
                },
                {
                    'Metric': 'Total Flows (All Years)',
                    'Value': fmt_currency(cumulative_flows)
                },
                {
                    'Metric': 'Fee Impact (Gross - Net)',
                    'Value': fmt_currency(actual_final_gross_nav - final_net_nav)
                }
            ])

            net_contributions_base = cumulative_flows
            if np.isfinite(net_contributions_base) and abs(net_contributions_base) > 0:
                summary_data.extend([
                    {
                        'Metric': 'Total Management Fees (% of Net Contributions)',
                        'Value': fmt_pct_value(cumulative_mgmt / net_contributions_base)
                    },
                    {
                        'Metric': 'Total Performance Fees (% of Net Contributions)',
                        'Value': fmt_pct_value(cumulative_perf / net_contributions_base)
                    },
                    {
                        'Metric': 'Total Fees (% of Net Contributions)',
                        'Value': fmt_pct_value((cumulative_mgmt + cumulative_perf) / net_contributions_base)
                    },
                    {
                        'Metric': 'Fee Impact (% of Net Contributions)',
                        'Value': fmt_pct_value((actual_final_gross_nav - final_net_nav) / net_contributions_base)
                    }
                ])

            summary_data.extend([
                {
                    'Metric': 'Gross Total Return (TWR)',
                    'Value': f'{gross_absolute:.4%}'
                },
                {
                    'Metric': 'Net Total Return (TWR)',
                    'Value': f'{net_absolute:.4%}'
                }
            ])

            if net_index_current is not None:
                summary_data.append({
                    'Metric': 'Net Total Return (Index-based, After Fees)',
                    'Value': fmt_pct_value(net_index_current - 1)
                })
            if hurdle_threshold_return is not None:
                summary_data.append({
                    'Metric': 'High Water Mark (Hurdle-Adjusted, Net Return to Date)',
                    'Value': fmt_pct_value(hurdle_threshold_return)
                })
            if hwm_index_current is not None:
                summary_data.append({
                    'Metric': 'High Water Mark (Base, Net Return to Date)',
                    'Value': fmt_pct_value(hwm_index_current - 1)
                })
            if cumulative_hurdle_pct_current is not None:
                summary_data.append({
                    'Metric': 'Cumulative Hurdle (Since Inception)',
                    'Value': fmt_pct_value(cumulative_hurdle_pct_current)
                })
        else:
            summary_data.append({
                'Metric': 'Note',
                'Value': 'No quarterly fee tracking data available'
            })
        
        # Add blank row for spacing
        summary_data.append({'Metric': '', 'Value': ''})
        summary_data.append({'Metric': '=== PERIOD RETURNS ===', 'Value': ''})
        
        # Add period returns if they exist
        if 'period_returns' in data:
            # Get quarterly management fee from config
            quarterly_mgmt_fee = 0.0025  # Default 0.25% per quarter
            quarterly_hurdle = 0.015     # Default 1.5% per quarter
            
            if config is not None and hasattr(config.fees, 'management_fee_quarterly'):
                quarterly_mgmt_fee = config.fees.management_fee_quarterly
            if config is not None:
                quarterly_hurdle = hurdle_rate / 4
            
            period_windows = config.periods.get_period_windows() if config is not None else {}
            for period, period_data in data['period_returns'].items():
                period_name = f'{period} Returns' if '_ytd' not in period else f'{period.replace("_ytd", "")} Returns (YTD)'
                gross_period_return = period_data["return"]
                final_net_return = None
                mgmt_fee = None
                perf_fee = None
                if not monthly_net_returns.empty and period in period_windows:
                    start_date, end_date = period_windows[period]
                    period_months = monthly_net_returns['period'].dt.to_timestamp('M')
                    mask = (period_months >= pd.to_datetime(start_date)) & (period_months <= pd.to_datetime(end_date))
                    period_net_returns = monthly_net_returns.loc[mask, 'net_return']
                    if not period_net_returns.empty:
                        final_net_return = float((1 + period_net_returns).prod() - 1)

                if final_net_return is None:
                    num_quarters = 4
                    if '_ytd' in period:
                        num_quarters = max(1, datetime.now().month // 3)

                    fee_result = calculate_period_net_return(
                        gross_return=gross_period_return,
                        quarterly_mgmt_fee=quarterly_mgmt_fee,
                        quarterly_hurdle=quarterly_hurdle,
                        perf_fee_rate=perf_fee_rate,
                        num_quarters=num_quarters,
                        high_water_mark_ratio=1.0
                    )

                    final_net_return = fee_result['net_return']
                    perf_fee = fee_result['perf_fee_impact']
                    mgmt_fee = fee_result['mgmt_fee_impact']

                mgmt_fee_display = 'See quarterly fee tracking'
                perf_fee_display = 'See quarterly fee tracking'
                if mgmt_fee is not None:
                    mgmt_fee_display = f'{mgmt_fee:.4%}'
                if perf_fee is not None:
                    perf_fee_display = f'{perf_fee:.4%}' if perf_fee > 0 else 'None (below hurdle/HWM)'
                
                summary_data.extend([
                    {
                        'Metric': f'{period_name} (Gross TWR)',
                        'Value': f'{gross_period_return:.4%}'
                    },
                    {
                        'Metric': f'{period_name} (Net TWR)',
                        'Value': f'{final_net_return:.4%}'
                    },
                    {
                        'Metric': f'{period_name} Management Fee Impact',
                        'Value': mgmt_fee_display
                    },
                    {
                        'Metric': f'{period_name} Performance Fee Impact',
                        'Value': perf_fee_display
                    },
                    {
                        'Metric': f'{period_name} Active Accounts',
                        'Value': ', '.join(period_data['accounts'])
                    }
                ])
                # Add blank row for spacing between periods
                summary_data.append({'Metric': '', 'Value': ''})
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Save with increased column width for better readability
        with pd.ExcelWriter(summary_file, engine='openpyxl') as writer:
            summary_df.to_excel(writer, index=False)
            worksheet = writer.sheets['Sheet1']
            worksheet.column_dimensions['A'].width = 40
            worksheet.column_dimensions['B'].width = 20
            
            # Add some basic formatting to make it more readable
            for row in worksheet.iter_rows():
                for cell in row:
                    cell.alignment = openpyxl.styles.Alignment(wrap_text=True)
        
        logging.info(f"Saved results for {category}")
        logging.info(f"- Monthly returns: {os.path.basename(monthly_file)}")
        if os.path.exists(daily_file):
            logging.info(f"- Daily returns: {os.path.basename(daily_file)}")
        logging.info(f"- Summary statistics: {os.path.basename(summary_file)}")


def main():
    """
    Main entry point using legacy global variables.
    For new code, use ReturnsCalculator class directly.
    """
    global brokerages, functions, all_results, all_client_returns, args
    
    utils.setup_logging('Returns')

    scan_for_accounts()

    total_accounts = sum(len(clients) for clients in brokerages.values())
    if total_accounts == 0:
        print("No accounts found to process")
        return

    for account in brokerages:
        if account in functions and brokerages[account]:
            process_brokerage(account, *functions[account])

    config = None
    if hasattr(args, 'config'):
        try:
            config = load_config(args.config)
        except Exception:
            config = None

    # Calculate aggregate results using asset-weighted composite returns
    aggregate_client_returns(all_client_returns, config=config)

    save_all_results(all_results, config=config)

    print("Returns calculation process completed")


def main_with_class(config_path: str = 'config.yaml', input_path: Optional[str] = None):
    """
    Main entry point using the ReturnsCalculator class.
    
    Args:
        config_path: Path to configuration file
        input_path: Override input path from config
    """
    utils.setup_logging('Returns')
    
    calculator = ReturnsCalculator(config_path=config_path, input_path=input_path)
    calculator.run()


if __name__ == "__main__":
    # Argument configuration
    parser = argparse.ArgumentParser(
        description='GIPS-compliant investment returns calculator for multiple brokerage accounts.'
    )
    parser.add_argument('-i', '--input-path', default=None, help='Input path (overrides config)')
    parser.add_argument('-c', '--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--legacy', action='store_true', help='Use legacy mode with global variables')
    args = parser.parse_args()

    if args.legacy:
        # Legacy mode for backward compatibility
        # Initialize global variables
        brokerages = {'Exante': {}, 'IBKR': {}}
        functions = {
            "IBKR": (IBKR.process, IBKR.read_data),
            "Exante": (Exante.process, Exante.read_data),
        }
        all_results = {}
        all_client_returns = []

        # Override input path if provided
        if args.input_path is None:
            args.input_path = 'input'

        main()
    else:
        # New class-based mode (default)
        main_with_class(config_path=args.config, input_path=args.input_path)
