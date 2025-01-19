import pandas as pd
import numpy as np
from datetime import datetime
import argparse
from typing import Tuple, List, Dict
import os
import IBKR
import Exante
import logging
import openpyxl
import utils
import numpy_financial as npf


"""
Update comments
- Remove those that aren't meaningful
- Add more and explain better in the parts that do the mathematical calculations
- Ensure the formatting is consistent (I added some above methods) 

Flow
- Several functions are meant to return boolean based on success or failure however this is not done and checked for consistently
- Refer to brokerages process() functions for example
"""


def scan_for_accounts():
    global brokerages
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
    global all_results, all_client_returns

    for client, files in brokerages[broker_name].items():
        try:
            print(f"Processing {broker_name} data for {client}...")
            process(files['source_csv'])

            # Processing logic
            nav_df, trades_df = read_data(files['output_dir'])
            sub_period_returns = calculate_sub_period_returns(nav_df, trades_df)
            monthly_returns = calculate_monthly_twr(sub_period_returns)
            six_month_returns = calculate_six_month_returns(monthly_returns)
            absolute_return, annualized_return = calculate_total_returns(monthly_returns)

            # Calculate IRR
            irr = calculate_irr(nav_df, trades_df)

            results = {
                'monthly_returns': monthly_returns,
                'six_month_returns': six_month_returns,
                'absolute_return': absolute_return,
                'annualized_return': annualized_return,
                'irr': irr,
                'sub_period_returns': sub_period_returns,
                'account_name': f'{broker_name}_{client}'
            }

            all_results[f'{broker_name}_{client}'] = results
            all_client_returns.append(results)
        except Exception as e:
            logging.error(f"Error processing {broker_name} client {client}: {e}")
            continue


# Calculate sub-period returns between each cash flow using TWR methodology
def calculate_sub_period_returns(nav_df, trades_df):
    print("Calculating sub-period returns...")

    # Get unique dates
    all_dates = pd.concat([nav_df['Date']]).sort_values().unique()

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

        sub_period_return = (end_nav - start_nav - total_flow) / (start_nav + total_flow)

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


def calculate_monthly_twr(sub_period_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate monthly TWR by geometrically linking sub-period returns.
    Also includes start-of-month NAV for GIPS composite calculation.
    """
    print("Calculating monthly TWR...")

    if sub_period_returns.empty:
        logging.warning("No sub-period returns to calculate monthly TWR")
        return pd.DataFrame(columns=['month', 'return', 'start_of_month_nav'])

    sub_period_returns['month'] = pd.to_datetime(sub_period_returns['start_date']).dt.to_period('M')

    monthly_returns = []
    for month, group in sub_period_returns.groupby('month'):
        monthly_return = np.prod(1 + group['return']) - 1

        if not np.isfinite(monthly_return):
            logging.warning(f"Invalid return calculation for month {month}")
            continue

        # Get start-of-month NAV (first start_nav in the month)
        start_of_month_nav = group.iloc[0]['start_nav']

        monthly_returns.append({
            'month': month,
            'return': monthly_return,
            'start_of_month_nav': start_of_month_nav
        })

    monthly_returns_df = pd.DataFrame(monthly_returns)
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


def calculate_total_returns(monthly_returns: pd.DataFrame) -> Tuple[float, float]:
    """
    Calculate total absolute and annualized returns
    """
    print("Calculating total returns...")

    if monthly_returns.empty:
        logging.warning("No monthly returns to calculate total returns")
        return 0.0, 0.0

    valid_returns = monthly_returns[np.isfinite(monthly_returns['return'])]

    if valid_returns.empty:
        logging.warning("No valid returns to calculate total returns")
        return 0.0, 0.0

    absolute_return = np.prod(1 + valid_returns['return']) - 1
    num_years = len(valid_returns) / 12

    if num_years > 0:
        annualized_return = (1 + absolute_return) ** (1 / num_years) - 1
    else:
        annualized_return = absolute_return

    logging.info(f"Calculated total returns: Absolute={absolute_return:.2%}, Annualized={annualized_return:.2%}")
    return absolute_return, annualized_return


def aggregate_client_returns(client_returns: List[Dict]):
    """
    Aggregate returns across multiple accounts using GIPS-style asset-weighted composite returns.
    """
    print("Aggregating client returns using GIPS composite method...")

    if not client_returns:
        logging.warning("No client returns to aggregate")
        return

    # Collect monthly returns from all accounts
    monthly_returns_by_account = {}
    for account_data in client_returns:
        monthly_returns = account_data.get('monthly_returns')
        account_name = account_data.get('account_name', 'Unknown')
        if not monthly_returns.empty:
            monthly_returns_by_account[account_name] = monthly_returns

    # Calculate GIPS composite
    composite_df, period_returns = build_gips_composite(monthly_returns_by_account)

    if composite_df.empty:
        logging.warning("No composite returns calculated")
        return

    # Calculate six-month rolling returns from composite returns
    composite_monthly = composite_df[['month', 'composite_return']].rename(
        columns={'composite_return': 'return'}
    )
    composite_six_month = calculate_six_month_returns(composite_monthly)

    # Calculate total returns from composite growth
    final_growth = composite_df['composite_growth'].iloc[-1]
    num_years = len(composite_df) / 12
    annualized_return = (1 + final_growth) ** (1 / num_years) - 1 if num_years > 0 else final_growth

    # Calculate composite IRR
    composite_irr = calculate_composite_irr(client_returns)

    logging.info(f"Composite absolute return: {final_growth:.2%}")
    logging.info(f"Composite annualized return: {annualized_return:.2%}")
    if composite_irr is not None:
        logging.info(f"Composite IRR: {composite_irr:.2%}")

    all_results['Combined'] = {
        'monthly_returns': composite_monthly,
        'six_month_returns': composite_six_month,
        'absolute_return': final_growth,
        'annualized_return': annualized_return,
        'irr': composite_irr,  # Add composite IRR to results
        'period_returns': period_returns
    }


def build_gips_composite(monthly_returns_dfs: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, Dict]:
    """
    Build a GIPS-compliant composite from account-level monthly returns.
    Only includes accounts that were active during each period for period-specific returns.
    Uses NAV-weighted returns for each month and compounds them over time.
    """
    print("Building GIPS composite from account returns...")

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
        month_navs = {}  # Store NAVs by account for this month
        month_returns = {}  # Store returns by account for this month

        # First gather all valid data for this month
        for account, df in monthly_returns_dfs.items():
            month_data = df[df['month'] == month]
            if not month_data.empty:
                nav = month_data['start_of_month_nav'].iloc[0]
                monthly_return = month_data['return'].iloc[0]

                if np.isfinite(nav) and np.isfinite(monthly_return):
                    month_navs[account] = nav
                    month_returns[account] = monthly_return
                    active_accounts.append(account)

        if month_navs and month_returns:  # Only proceed if we have valid data
            # Calculate total NAV for this month
            total_nav = sum(month_navs.values())

            # Calculate NAV-weighted composite return for this month
            weighted_return = sum(
                (nav / total_nav) * month_returns[account]
                for account, nav in month_navs.items()
            )

            composite_results.append({
                'month': month,
                'composite_return': weighted_return,
                'total_nav': total_nav,
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

    # 2022 returns (2nd Feb 2022 - 2nd Feb 2023)
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

    # 2023 returns (2nd Feb 2023 - 2nd Feb 2024)
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

    # 2024 returns (2nd Feb 2024 - latest)
    period_2024 = composite_df[composite_df['date'] >= '2024-02-01']
    if not period_2024.empty:
        # Compound the monthly returns for the period
        period_2024_return = np.prod(1 + period_2024['composite_return']) - 1
        period_returns['2024_ytd'] = {
            'return': period_2024_return,
            'accounts': list(set(','.join(period_2024['active_accounts']).split(',')))
        }
        logging.info(f"2024 YTD returns calculated using accounts: {period_returns['2024_ytd']['accounts']}")

    logging.info(f"Calculated composite returns for {len(composite_df)} months")
    return composite_df, period_returns


def calculate_irr(nav_df: pd.DataFrame, trades_df: pd.DataFrame):
    """
    Calculate the Internal Rate of Return (IRR) using cash flows and final NAV.

    The IRR calculation:
    - Uses all cash flows (deposits/withdrawals)
    - Includes the initial NAV as a negative flow (investment)
    - Includes the final NAV as a positive flow (current value)
    - Returns the annualized IRR
    """
    print("Calculating IRR...")

    try:
        # Get initial and final NAV
        initial_nav = nav_df.iloc[0]['Net Asset Value']
        final_nav = nav_df.iloc[-1]['Net Asset Value']
        start_date = nav_df.iloc[0]['Date']
        end_date = nav_df.iloc[-1]['Date']

        # Prepare cash flows
        flows = []
        dates = []

        # Add initial NAV as negative flow (investment)
        flows.append(-initial_nav)
        dates.append(start_date)

        # Add all intermediate cash flows
        flow_column = 'Adjusted EUR' if 'Adjusted EUR' in trades_df.columns else 'EUR equivalent'
        for _, row in trades_df.iterrows():
            flows.append(-row[flow_column])  # Negative because deposits are outflows from investor perspective
            dates.append(row['When'])

        # Add final NAV as positive flow (current value)
        flows.append(final_nav)
        dates.append(end_date)

        # Calculate IRR
        irr = npf.irr(flows)

        # Convert to annual rate
        annual_irr = (1 + irr) ** 365 - 1

        logging.info(f"Calculated IRR: {annual_irr:.2%}")
        return annual_irr

    except Exception as e:
        logging.error(f"Error calculating IRR: {e}")
        return None


def calculate_composite_irr(client_returns: List[Dict]) -> float:
    """
    Calculate IRR for combined accounts by aggregating all cash flows and NAVs.
    """
    print("Calculating composite IRR for all accounts...")

    try:
        # Collect all NAVs and trades from all accounts
        all_navs = []
        all_trades = []

        for account_data in client_returns:
            if 'sub_period_returns' in account_data:
                sub_period_returns = account_data['sub_period_returns']
                if not sub_period_returns.empty:
                    # Get NAV data points
                    nav_data = pd.DataFrame({
                        'Date': sub_period_returns['start_date'],
                        'Net Asset Value': sub_period_returns['start_nav']
                    })
                    # Add final NAV point
                    nav_data = nav_data.append({
                        'Date': sub_period_returns['end_date'].iloc[-1],
                        'Net Asset Value': sub_period_returns['end_nav'].iloc[-1]
                    }, ignore_index=True)
                    all_navs.append(nav_data)

                # Get trade data
                if 'trades' in account_data:
                    all_trades.append(account_data['trades'])

        if not all_navs:
            logging.warning("No NAV data available for composite IRR calculation")
            return None

        # Combine all NAV data
        combined_nav = pd.concat(all_navs)
        # Group by date and sum NAVs
        combined_nav = combined_nav.groupby('Date')['Net Asset Value'].sum().reset_index()
        combined_nav = combined_nav.sort_values('Date')

        # Combine all trades if available
        if all_trades:
            combined_trades = pd.concat(all_trades)
            combined_trades = combined_trades.sort_values('When')
        else:
            combined_trades = pd.DataFrame(columns=['When', 'Operation type', 'EUR equivalent'])

        # Calculate IRR using combined data
        return calculate_irr(combined_nav, combined_trades)

    except Exception as e:
        logging.error(f"Error calculating composite IRR: {e}")
        return


def save_all_results(results: Dict, output_dir='Results'):
    print(f"Saving results to {output_dir}...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(output_dir, exist_ok=True)

    for category, data in results.items():
        category_dir = os.path.join(output_dir, category)
        os.makedirs(category_dir, exist_ok=True)

        monthly_file = os.path.join(category_dir, f'monthly_returns_{timestamp}.xlsx')
        six_month_file = os.path.join(category_dir, f'six_month_returns_{timestamp}.xlsx')
        summary_file = os.path.join(category_dir, f'return_summary_{timestamp}.xlsx')

        data['monthly_returns'].to_excel(monthly_file, index=False)
        data['six_month_returns'].to_excel(six_month_file, index=False)

        # Create summary DataFrame with all return metrics
        summary_data = [{'Metric': '', 'Value': ''}]

        # Add TWR returns
        summary_data.extend([
            {
                'Metric': 'Total Absolute Return (TWR)',
                'Value': f'{data["absolute_return"]:.2%}'
            },
            {
                'Metric': 'Annualized Return (TWR)',
                'Value': f'{data["annualized_return"]:.2%}'
            }
        ])

        # Add blank row for spacing
        summary_data.append({'Metric': '', 'Value': ''})

        # Add IRR if available
        if 'irr' in data and data['irr'] is not None:
            summary_data.append({
                'Metric': 'Internal Rate of Return (IRR)',
                'Value': f'{data["irr"]:.2%}'
            })
            # Add blank row for spacing
            summary_data.append({'Metric': '', 'Value': ''})

        # Add period returns if they exist
        if 'period_returns' in data:
            for period, period_data in data['period_returns'].items():
                period_name = '2024 Returns (YTD)' if period == '2024_ytd' else f'{period} Returns'
                summary_data.extend([
                    {
                        'Metric': f'{period_name} (TWR)',
                        'Value': f'{period_data["return"]:.2%}'
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
        logging.info(f"- Six-month returns: {os.path.basename(six_month_file)}")
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
    brokerages = {'Exante': {}, 'IBKR': {}}

    # Functions used by process_brokerage() for each brokerage
    functions = {
        "IBKR": (IBKR.process, IBKR.read_data),
        "Exante": (Exante.process, Exante.read_data),
    }

    # Data structures storing results
    all_results = {}
    all_client_returns = []

    # Argument configuration
    parser = argparse.ArgumentParser(description='Description')
    parser.add_argument('-i', '--input-path', default='Input', help='Input path')
    args = parser.parse_args()

    main()
