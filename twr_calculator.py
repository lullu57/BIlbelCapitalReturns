import pandas as pd
import numpy as np
from datetime import datetime
import argparse
from typing import Tuple, List, Dict
import os
import IBKR
import Exante
import POEMS
import MoneyBase
import logging
import openpyxl
import utils



"""
@TODO
Update comments
- Remove those that aren't meaningful
- Add more and explain better in the parts that do the mathematical calculations

Flow
- Several functions are meant to return boolean based on success or failure however this is not done and checked for consistently
- Refer to brokerages process() functions for example
"""


def scan_for_accounts():
    global brokerages
    print('Scanning for accounts...')
    for brokerage in brokerages:
        # POEMS and MoneyBase are manual brokerages with hard-coded data, so we just register the two
        # known clients without looking for CSVs.
        if brokerage in ['POEMS', 'MoneyBase']:
            for client in ['Bernard', 'Roswitha']:
                brokerages[brokerage][client] = {
                    'source_csv': '',  # not used
                    'output_dir': client  # just the client name, POEMS.read_data ignores parent dirs
                }
            print(f"Registered {brokerage} accounts: {', '.join(brokerages[brokerage].keys())}")
            continue

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
    global all_results, all_client_returns

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


def calculate_sub_period_returns(nav_df: pd.DataFrame, trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate sub-period returns between each cash flow using TWR methodology
    """
    logging.info("Calculating sub-period returns")
    
    # Get unique dates
    all_dates = pd.concat([
        nav_df['Date']
    ]).sort_values().unique()
    
    logging.info(f"Found {len(all_dates)} unique dates for sub-period calculations")
    
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
            
        period_flows = trades_df[
            (trades_df['When'] > start_date) & 
            (trades_df['When'] <= end_date)
        ]
        
        flow_column = 'Adjusted EUR' if 'Adjusted EUR' in period_flows.columns else 'EUR equivalent'
        total_flow = period_flows[flow_column].sum()
        
        sub_period_return = (end_nav - start_nav - total_flow) / start_nav if start_nav != 0 else 0

        returns.append({
            'start_date': start_date,
            'end_date': end_date,
            'return': sub_period_return,
            'start_nav': start_nav,
            'end_nav': end_nav,
            'total_flow': total_flow
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
    
    logging.info(f"Calculated {len(returns_df)} sub-period returns")
    return returns_df


def build_daily_returns(nav_df: pd.DataFrame, trades_df: pd.DataFrame, sub_period_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-day series with TWR-consistent daily returns using sub-periods.
    - For periods with consecutive daily NAVs, this yields actual daily returns.
    - For sparse NAVs (e.g., monthly), returns are 0 on interim days and applied on the
      end date of the sub-period.
    The output has columns: date, return, start_nav, end_nav, flow
    """
    if sub_period_returns.empty:
        return pd.DataFrame(columns=['date', 'return', 'start_nav', 'end_nav', 'flow'])

    # Prepare flows per day
    flow_col = 'Adjusted EUR' if 'Adjusted EUR' in trades_df.columns else 'EUR equivalent'
    flows_by_day = (
        trades_df.groupby(pd.to_datetime(trades_df['When']).dt.normalize())[flow_col]
        .sum()
        .to_dict()
    )

    # Ensure NAV dates are normalized and sorted
    nav_df = nav_df.copy()
    nav_df['Date'] = pd.to_datetime(nav_df['Date']).dt.normalize()
    nav_df = nav_df.sort_values('Date')

    daily_rows: List[Dict] = []

    # Iterate each sub-period
    for _, sp in sub_period_returns.iterrows():
        period_start = pd.to_datetime(sp['start_date']).normalize()
        period_end = pd.to_datetime(sp['end_date']).normalize()
        r = float(sp['return']) if pd.notna(sp['return']) else 0.0

        # Track baseline NAV for period return
        period_start_nav = float(sp['start_nav']) if pd.notna(sp['start_nav']) else 0.0

        # current_nav represents the NAV carried day-to-day through the period
        current_nav = period_start_nav

        # Iterate calendar days strictly after start up to and including end
        for dt in pd.date_range(period_start + pd.Timedelta(days=1), period_end, freq='D'):
            start_nav_today = current_nav
            flow_today = float(flows_by_day.get(dt, 0.0))

            if dt == period_end:
                # Apply period return on the period baseline NAV, not on flows
                return_today = r
                return_impact = r * period_start_nav
            else:
                return_today = 0.0
                return_impact = 0.0

            end_nav_today = start_nav_today + flow_today + return_impact

            daily_rows.append({
                'date': dt,
                'return': return_today,
                'start_nav': start_nav_today,
                'end_nav': end_nav_today,
                'flow': flow_today,
            })

            current_nav = end_nav_today

    daily_df = pd.DataFrame(daily_rows)
    if not daily_df.empty:
        daily_df = daily_df.sort_values('date')
    return daily_df


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


def aggregate_client_returns(client_returns: List[Dict]):
    """
    Aggregate returns across multiple accounts using daily sub-period series.
    - Build a composite daily series by summing start/end NAV and flows, then
      applying the sub-period return formula at the composite level.
    - Derive composite monthly returns by geometrically linking daily returns.
    """
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

    # Reuse period window logic from build_gips_composite
    period_returns: Dict = {}
    for label, (start_s, end_s) in {
        '2022': ('2022-02-01', '2023-01-31'),
        '2023': ('2023-02-01', '2024-01-31'),
        '2024_ytd': ('2024-02-01', '2025-01-31'),
        '2025_ytd': ('2025-02-01', '2026-01-31'),
    }.items():
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

    all_results['Combined'] = {
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
                nav = month_data['end_of_month_nav'].iloc[0]
                monthly_return = month_data['return'].iloc[0]
                
                if np.isfinite(nav) and np.isfinite(monthly_return):
                    month_navs[account] = nav
                    month_returns[account] = monthly_return
                    active_accounts.append(account)

                    # Optional diagnostics: start-of-month NAV and monthly flow
                    if 'start_of_month_nav' in month_data.columns:
                        som = month_data['start_of_month_nav'].iloc[0]
                        if np.isfinite(som):
                            start_month_navs[account] = som
                    if 'monthly_flow' in month_data.columns:
                        flow_val = month_data['monthly_flow'].iloc[0]
                        try:
                            flow_num = float(flow_val)
                        except Exception:
                            flow_num = np.nan
                        if np.isfinite(flow_num):
                            month_flows[account] = flow_num
        
        if month_navs and month_returns:  # Only proceed if we have valid data
            # Calculate total NAV for this month
            total_nav = sum(month_navs.values())
            # Optional aggregates for auditability
            start_of_month_total_nav = sum(start_month_navs.values()) if start_month_navs else np.nan
            monthly_flow_total = sum(month_flows.values()) if month_flows else np.nan
            
            # Calculate NAV-weighted composite return for this month
            weighted_return = sum(
                (nav / total_nav) * month_returns[account]
                for account, nav in month_navs.items()
            )
            
            composite_results.append({
                'month': month,
                'composite_return': weighted_return,
                'total_nav': total_nav,
                'start_of_month_total_nav': start_of_month_total_nav,
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
    
    # Calculate returns for specific periods
    period_returns = {}
    
    # 2022 returns (2nd Feb 2022 - 31st Jan 2023)
    period_2022 = composite_df[
        (composite_df['date'] >= '2022-02-01') & 
        (composite_df['date'] <= '2023-01-31')
    ]
    if not period_2022.empty:
        # Compound the monthly returns for the period
        period_2022_return = np.prod(1 + period_2022['composite_return']) - 1
        period_returns['2022'] = {
            'return': period_2022_return,
            'accounts': list(set(','.join(period_2022['active_accounts']).split(',')))
        }
        logging.info(f"2022 returns calculated using accounts: {period_returns['2022']['accounts']}")
    
    # 2023 returns (2nd Feb 2023 - 31st Jan 2024)
    period_2023 = composite_df[
        (composite_df['date'] >= '2023-02-01') & 
        (composite_df['date'] <= '2024-01-31')
    ]
    if not period_2023.empty:
        # Compound the monthly returns for the period
        period_2023_return = np.prod(1 + period_2023['composite_return']) - 1
        period_returns['2023'] = {
            'return': period_2023_return,
            'accounts': list(set(','.join(period_2023['active_accounts']).split(',')))
        }
        logging.info(f"2023 returns calculated using accounts: {period_returns['2023']['accounts']}")
    
    # 2024 returns (2nd Feb 2024 - 31st Jan 2025)
    period_2024 = composite_df[
        (composite_df['date'] >= '2024-02-01') & 
        (composite_df['date'] <= '2025-01-31')
    ]
    if not period_2024.empty:
        # Compound the monthly returns for the period
        period_2024_return = np.prod(1 + period_2024['composite_return']) - 1
        period_returns['2024_ytd'] = {
            'return': period_2024_return,
            'accounts': list(set(','.join(period_2024['active_accounts']).split(',')))
        }
        logging.info(f"2024 YTD returns calculated using accounts: {period_returns['2024_ytd']['accounts']}")
    
    # 2025 returns (2nd Feb 2025 - 31st Jan 2026)
    period_2025 = composite_df[
        (composite_df['date'] >= '2025-02-01') & 
        (composite_df['date'] <= '2026-01-31')
    ]
    if not period_2025.empty:
        # Compound the monthly returns for the period
        period_2025_return = np.prod(1 + period_2025['composite_return']) - 1
        period_returns['2025_ytd'] = {
            'return': period_2025_return,
            'accounts': list(set(','.join(period_2025['active_accounts']).split(',')))
        }
        logging.info(f"2025 YTD returns calculated using accounts: {period_returns['2025_ytd']['accounts']}")
    logging.info(f"Calculated composite returns for {len(composite_df)} months")
    return composite_df, period_returns


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





def save_all_results(results: Dict, output_dir: str = 'results') -> None:
    """
    Save results for all accounts and aggregations
    """
    logging.info(f"Saving results to {output_dir}")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(output_dir, exist_ok=True)
    
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

        # Format monthly returns as percentages
        if 'return' in monthly_returns.columns:
            monthly_returns['return'] = monthly_returns['return'].map('{:.4%}'.format)
        
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
        
        # Add blank row for spacing
        summary_data.append({'Metric': '', 'Value': ''})
        
        # Add TWR returns (from daily compounding if available)
        summary_data.extend([
            {
                'Metric': 'Total Absolute Return (TWR)',
                'Value': f'{data.get("absolute_return", 0.0):.4%}'
            },
            {
                'Metric': 'Annualized Return (TWR)',
                'Value': f'{data.get("annualized_return", 0.0):.4%}'
            }
        ])
        
        # Add blank row for spacing
        summary_data.append({'Metric': '', 'Value': ''})
        

        
        # Add period returns if they exist
        if 'period_returns' in data:
            for period, period_data in data['period_returns'].items():
                period_name = '2024 Returns (YTD)' if period == '2024_ytd' else f'{period} Returns'
                summary_data.extend([
                    {
                        'Metric': f'{period_name} (TWR)',
                        'Value': f'{period_data["return"]:.4%}'
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
    utils.setup_logging('Returns')

    scan_for_accounts()

    if not brokerages:
        print("No accounts found to process")
        return

    for account in brokerages:
        process_brokerage(account, *functions[account])

    # Calculate aggregate results using asset-weighted composite returns
    aggregate_client_returns(all_client_returns)

    save_all_results(all_results)

    print("Returns calculation process completed")


if __name__ == "__main__":
    # Dictionary that holds the accounts for each brokerage
    brokerages = {'Exante': {}, 'IBKR': {}, 'POEMS': {}, 'MoneyBase': {}}

    # Functions used by process_brokerage() for each brokerage
    functions = {
        "IBKR": (IBKR.process, IBKR.read_data),
        "Exante": (Exante.process, Exante.read_data),
        "POEMS": (POEMS.process, POEMS.read_data),
        "MoneyBase": (MoneyBase.process, MoneyBase.read_data),
    }

    # Data structures storing results
    all_results = {}
    all_client_returns = []

    # Argument configuration
    parser = argparse.ArgumentParser(description='A Python-based tool for calculating investment returns across multiple brokerage accounts for Bilbel Capital.')
    parser.add_argument('-i', '--input-path', default='input', help='Input path')
    args = parser.parse_args()

    main()
