import pandas as pd
import numpy as np
from datetime import datetime
import argparse
from typing import Tuple, List, Dict, Optional
import os
import IBKR
import Exante
import logging
import openpyxl
import utils
import flow_utils
from config_loader import load_config, Config, PeriodsConfig
from fee_calculator import (
    FeeCalculator, 
    PerformanceFeeCalculator,
    annual_to_monthly_fee,
    calculate_net_return
)


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
                process_func(files['source_csv'])

                # Processing logic
                nav_df, trades_df = read_data_func(files['output_dir'])
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
        aggregate_client_returns(self.all_client_returns, self.all_results)
    
    def save_results(self):
        """Save all results to output directory."""
        save_all_results(self.all_results, self.output_path)
    
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
            for file in os.listdir(input_path):
                if file.endswith('.csv'):
                    client = os.path.splitext(file)[0]
                    csv_path = os.path.join(input_path, file)
                    client_dir = os.path.join(input_path, client)
                    brokerages[brokerage][client] = {
                        'source_csv': csv_path,
                        'output_dir': client_dir
                    }
        if len(brokerages[brokerage]):
            print(f"Found {len(brokerages[brokerage])} {brokerage} accounts")


def process_brokerage(broker_name, process, read_data):
    """Legacy function - use ReturnsCalculator class instead."""
    global all_results, all_client_returns, brokerages

    for client, files in brokerages[broker_name].items():
        try:
            print(f"Processing {broker_name} data for {client}...")
            process(files['source_csv'])

            # Processing logic
            nav_df, trades_df = read_data(files['output_dir'])
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
        
        if start_nav <= 0:
            logging.warning(f"Invalid starting NAV ({start_nav}) for period {start_date}")
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
        
        sub_period_return = (end_nav - start_nav - total_flow) / start_nav if start_nav != 0 else 0

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
    where flows on the day are treated as occurring at the start of that day.
    """
    daily = build_account_daily_from_nav_flows(nav_df, trades_df)
    if daily.empty:
        return pd.DataFrame(columns=['date', 'return', 'start_nav', 'end_nav', 'flow'])

    daily = daily.sort_values('date')
    def compute_ret(row):
        start_v = float(row['start_nav']) if pd.notna(row['start_nav']) else 0.0
        if start_v <= 0:
            return 0.0
        end_v = float(row['end_nav']) if pd.notna(row['end_nav']) else start_v
        flow_v = float(row['flow']) if pd.notna(row['flow']) else 0.0
        return (end_v - flow_v) / start_v - 1.0

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
    if elapsed_years > 0:
        annualized_return = (1 + absolute_return) ** (1 / elapsed_years) - 1
    else:
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
    if elapsed_years > 0:
        annualized_return = (1 + absolute_return) ** (1 / elapsed_years) - 1
    else:
        annualized_return = absolute_return

    return absolute_return, float(annualized_return)


def aggregate_client_returns(client_returns: List[Dict], results_dict: Optional[Dict] = None):
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
    period_windows = get_period_windows()
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

        daily_return = (end_total - start_total - flow_total) / start_total if start_total > 0 else 0.0
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


def build_synthetic_composite(nav_dfs: Dict[str, pd.DataFrame], flow_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
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
        daily_by_account[name] = build_account_daily_from_nav_flows(nav, flows)

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


def build_account_daily_from_nav_flows(nav_df: pd.DataFrame, flow_df: pd.DataFrame) -> pd.DataFrame:
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
        flows_by_day = f.groupby('When')['EUR equivalent'].sum().to_dict()
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
        
        # Calculate gross and net returns with fee breakdown
        if 'return' in monthly_returns.columns and not monthly_returns.empty:
            # Store raw numeric returns for calculations
            raw_returns = monthly_returns['return'].copy()
            
            # Gross return is the original return
            monthly_returns['gross_return'] = raw_returns
            
            # Management fee (monthly)
            monthly_returns['management_fee'] = monthly_mgmt_fee
            
            # Net return = Gross - Management fee
            monthly_returns['net_return'] = raw_returns - monthly_mgmt_fee
            
            # Format for display
            monthly_returns['gross_return'] = monthly_returns['gross_return'].map('{:.4%}'.format)
            monthly_returns['management_fee'] = monthly_returns['management_fee'].map('{:.4%}'.format)
            monthly_returns['net_return'] = monthly_returns['net_return'].map('{:.4%}'.format)
            
            # Keep original return column for compatibility
            monthly_returns['return'] = monthly_returns['gross_return']
        
        # Format NAV/flow columns as currency if they exist
        if 'nav' in monthly_returns.columns:
            monthly_returns['nav'] = monthly_returns['nav'].map('€{:,.2f}'.format)
        if 'end_of_month_nav' in monthly_returns.columns:
            monthly_returns['end_of_month_nav'] = monthly_returns['end_of_month_nav'].map('€{:,.2f}'.format)
        if 'start_of_month_nav' in monthly_returns.columns:
            monthly_returns['start_of_month_nav'] = monthly_returns['start_of_month_nav'].map('€{:,.2f}'.format)
        if 'monthly_flow' in monthly_returns.columns:
            monthly_returns['monthly_flow'] = monthly_returns['monthly_flow'].map('€{:,.2f}'.format)
        
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
                    daily_returns[col] = daily_returns[col].map('€{:,.2f}'.format)
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
        
        # Calculate net returns (after management fees)
        # For multi-year periods, compound the monthly fee
        if 'daily_returns' in data and isinstance(data['daily_returns'], pd.DataFrame):
            num_months = len(data['daily_returns']['date'].dt.to_period('M').unique()) if 'date' in data['daily_returns'].columns else 12
        else:
            num_months = 12  # Default assumption
        
        total_mgmt_fee_impact = (1 + monthly_mgmt_fee) ** num_months - 1
        net_absolute = gross_absolute - total_mgmt_fee_impact
        
        # Calculate annualized net return
        if gross_annualized != 0:
            net_annualized = gross_annualized - annual_mgmt_fee
        else:
            net_annualized = 0.0
        
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
                'Value': 'Carry-forward shortfall (non-compounding)'
            }
        ])
        
        # Add blank row for spacing
        summary_data.append({'Metric': '', 'Value': ''})
        summary_data.append({'Metric': '=== PERIOD RETURNS ===', 'Value': ''})
        
        # Add period returns if they exist
        if 'period_returns' in data:
            # Create performance fee calculator for period returns
            perf_calc = PerformanceFeeCalculator(hurdle_rate=hurdle_rate, fee_rate=perf_fee_rate)
            
            for period, period_data in data['period_returns'].items():
                period_name = f'{period} Returns' if '_ytd' not in period else f'{period.replace("_ytd", "")} Returns (YTD)'
                gross_period_return = period_data["return"]
                
                # Net period return (after management fee, ~1% annual)
                net_period_return = gross_period_return - annual_mgmt_fee
                
                # Calculate performance fee for this period
                perf_fee, shortfall = perf_calc.calculate_annual_fee(net_period_return)
                final_net_return = net_period_return - perf_fee
                
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
                        'Metric': f'{period_name} Performance Fee',
                        'Value': f'{perf_fee:.4%}' if perf_fee > 0 else 'None (below hurdle)'
                    },
                    {
                        'Metric': f'{period_name} Carried Shortfall',
                        'Value': f'{shortfall:.4%}' if shortfall > 0 else 'None'
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

    # Calculate aggregate results using asset-weighted composite returns
    aggregate_client_returns(all_client_returns)

    save_all_results(all_results)

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
