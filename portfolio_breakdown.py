import argparse
import logging
import os
from datetime import datetime
from typing import Dict, Tuple, Optional, List

import pandas as pd

import utils
from twr_calculator import ReturnsCalculator


def _ensure_processed(process_func, source_csv) -> None:
    if isinstance(source_csv, (list, tuple)):
        for csv_path in source_csv:
            if csv_path:
                process_func(csv_path)
    elif source_csv:
        process_func(source_csv)


def _get_latest_nav_date(nav_df: pd.DataFrame) -> Optional[pd.Timestamp]:
    if nav_df.empty or 'Date' not in nav_df.columns:
        return None
    nav_df = nav_df.copy()
    nav_df['Date'] = pd.to_datetime(nav_df['Date'], errors='coerce')
    nav_df = nav_df.dropna(subset=['Date'])
    if nav_df.empty:
        return None
    return nav_df['Date'].max()


def _get_nav_as_of(nav_df: pd.DataFrame, as_of: pd.Timestamp) -> Tuple[Optional[float], Optional[pd.Timestamp]]:
    if nav_df.empty or 'Date' not in nav_df.columns or 'Net Asset Value' not in nav_df.columns:
        return None, None
    df = nav_df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date')
    df = df[df['Date'] <= as_of]
    if df.empty:
        return None, None
    value = pd.to_numeric(df['Net Asset Value'], errors='coerce').dropna()
    if value.empty:
        return None, None
    return float(value.iloc[-1]), pd.to_datetime(df['Date'].iloc[-1])


def _build_ibkr_positions(client_dir: str, client: str) -> pd.DataFrame:
    # IBKR trade summary provides quantities and proceeds in base currency.
    trade_path = os.path.join(client_dir, 'trade_summary.csv')
    if not os.path.exists(trade_path):
        return pd.DataFrame()

    df = pd.read_csv(trade_path)
    if 'Symbol' not in df.columns:
        return pd.DataFrame()

    qty_bought = pd.to_numeric(df.get('Quantity Bought'), errors='coerce').fillna(0.0)
    qty_sold = pd.to_numeric(df.get('Quantity Sold'), errors='coerce').fillna(0.0)
    proceeds_bought_base = pd.to_numeric(df.get('Proceeds Bought in Base'), errors='coerce').fillna(0.0)
    proceeds_sold_base = pd.to_numeric(df.get('Proceeds Sold in Base'), errors='coerce').fillna(0.0)

    df = df.assign(
        net_quantity=qty_bought - qty_sold,
        net_trade_cash_base=proceeds_bought_base + proceeds_sold_base
    )

    group_cols = ['Symbol']
    for col in ['Description', 'Currency', 'Financial Instrument']:
        if col in df.columns:
            group_cols.append(col)

    # Aggregate by symbol (and available descriptors) to show net quantity
    # and net trade cash since the report start.
    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(
            net_quantity=('net_quantity', 'sum'),
            net_trade_cash_base=('net_trade_cash_base', 'sum')
        )
        .reset_index()
    )

    grouped = grouped[(grouped['net_quantity'] != 0) | (grouped['net_trade_cash_base'] != 0)]
    if grouped.empty:
        return grouped

    grouped = grouped.rename(columns={
        'net_quantity': 'Net Quantity',
        'net_trade_cash_base': 'Net Trade Cash (EUR)'
    })
    grouped.insert(0, 'Brokerage', 'IBKR')
    grouped.insert(1, 'Client', client)
    grouped['ISIN'] = None
    grouped['Net Trade Cash (Asset)'] = None
    grouped['Last Trade Date'] = None
    grouped['Data Note'] = 'Net quantity and cash from Trade Summary (since report start)'
    return grouped


def _build_exante_trade_amounts(output_dirs: List[str], client: str) -> pd.DataFrame:
    # Exante exports provide trade cash amounts but not position quantities.
    frames = []
    for out_dir in output_dirs:
        trade_path = os.path.join(out_dir, 'Trades.xlsx')
        if not os.path.exists(trade_path):
            continue
        df = pd.read_excel(trade_path)
        if df.empty:
            continue
        df.columns = [str(c).strip() for c in df.columns]
        if 'Operation Type' in df.columns and 'Operation type' not in df.columns:
            df = df.rename(columns={'Operation Type': 'Operation type'})
        if 'Operation type' not in df.columns:
            continue
        df['Operation type'] = df['Operation type'].astype(str).str.upper().str.strip()
        df = df[df['Operation type'] == 'TRADE']
        if df.empty:
            continue
        if 'When' in df.columns:
            df['When'] = pd.to_datetime(df['When'], errors='coerce')
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    trades = pd.concat(frames, ignore_index=True)
    for col in ['Sum', 'EUR equivalent']:
        if col in trades.columns:
            trades[col] = pd.to_numeric(trades[col], errors='coerce').fillna(0.0)

    group_cols = []
    for col in ['Symbol ID', 'ISIN', 'Asset']:
        if col in trades.columns:
            group_cols.append(col)

    if not group_cols:
        return pd.DataFrame()

    agg_map = {
        'Sum': 'sum',
        'EUR equivalent': 'sum',
        'When': 'max'
    }
    # Aggregate cash amounts by symbol/ISIN/asset for a consolidated view.
    grouped = trades.groupby(group_cols, dropna=False).agg(agg_map).reset_index()

    grouped = grouped.rename(columns={
        'Sum': 'Net Trade Cash (Asset)',
        'EUR equivalent': 'Net Trade Cash (EUR)',
        'When': 'Last Trade Date'
    })
    grouped.insert(0, 'Brokerage', 'Exante')
    grouped.insert(1, 'Client', client)
    grouped['Net Quantity'] = None
    grouped['Description'] = None
    grouped['Currency'] = grouped.get('Asset')
    grouped['Financial Instrument'] = None
    grouped['Data Note'] = 'Net trade cash amounts (Exante does not expose quantities)'
    grouped = grouped.rename(columns={'Symbol ID': 'Symbol'})
    return grouped


def build_portfolio_breakdown(config_path: str, input_path: Optional[str], output_path: str, as_of: Optional[str]):
    utils.setup_logging('PortfolioBreakdown')
    logging.info("Building portfolio breakdown report")

    calc = ReturnsCalculator(config_path=config_path, input_path=input_path)
    calc.scan_for_accounts()

    # Ensure processed files exist
    for brokerage_name, clients in calc.brokerages.items():
        if brokerage_name not in calc.brokerage_modules:
            continue
        process_func, _ = calc.brokerage_modules[brokerage_name]
        for _, files in clients.items():
            _ensure_processed(process_func, files.get('source_csv'))

    # Determine as-of date:
    # - explicit date if provided
    # - otherwise latest NAV date across all accounts
    if as_of:
        as_of_dt = pd.to_datetime(as_of)
    else:
        latest_dates = []
        for brokerage_name, clients in calc.brokerages.items():
            if brokerage_name not in calc.brokerage_modules:
                continue
            _, read_data_func = calc.brokerage_modules[brokerage_name]
            for _, files in clients.items():
                nav_df, _ = read_data_func(files.get('output_dir'))
                last_date = _get_latest_nav_date(nav_df)
                if last_date is not None:
                    latest_dates.append(last_date)
        as_of_dt = max(latest_dates) if latest_dates else pd.Timestamp.now().normalize()

    rows = []
    position_frames: List[pd.DataFrame] = []
    for brokerage_name, clients in calc.brokerages.items():
        if brokerage_name not in calc.brokerage_modules:
            continue
        _, read_data_func = calc.brokerage_modules[brokerage_name]
        for client, files in clients.items():
            nav_df, _ = read_data_func(files.get('output_dir'))
            # Use last known NAV on or before the as-of date.
            nav_value, nav_date = _get_nav_as_of(nav_df, as_of_dt)
            rows.append({
                'Brokerage': brokerage_name,
                'Client': client,
                'As Of': as_of_dt.date().isoformat(),
                'Data Date': nav_date.date().isoformat() if nav_date is not None else None,
                'Final NAV': nav_value,
            })

            if brokerage_name == 'IBKR':
                ibkr_positions = _build_ibkr_positions(files.get('output_dir', ''), client)
                if not ibkr_positions.empty:
                    position_frames.append(ibkr_positions)
            elif brokerage_name == 'Exante':
                output_dirs = files.get('output_dir') or []
                if not isinstance(output_dirs, list):
                    output_dirs = [output_dirs]
                exante_positions = _build_exante_trade_amounts(output_dirs, client)
                if not exante_positions.empty:
                    position_frames.append(exante_positions)

    accounts_df = pd.DataFrame(rows)
    total_nav = accounts_df['Final NAV'].sum(skipna=True)
    accounts_df['Weight of Total'] = accounts_df['Final NAV'] / total_nav if total_nav else None

    brokerage_totals = (
        accounts_df.groupby('Brokerage', dropna=False)['Final NAV']
        .sum()
        .reset_index()
    )
    brokerage_totals['Weight of Total'] = brokerage_totals['Final NAV'] / total_nav if total_nav else None

    client_totals = (
        accounts_df.groupby('Client', dropna=False)['Final NAV']
        .sum()
        .reset_index()
    )
    client_totals['Weight of Total'] = client_totals['Final NAV'] / total_nav if total_nav else None

    os.makedirs(output_path, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = os.path.join(output_path, f'portfolio_breakdown_{timestamp}.xlsx')

    # Write all summary tabs in a single workbook for easy review.
    with pd.ExcelWriter(out_file) as writer:
        accounts_df.to_excel(writer, sheet_name='accounts', index=False)
        brokerage_totals.to_excel(writer, sheet_name='brokerage_totals', index=False)
        client_totals.to_excel(writer, sheet_name='client_totals', index=False)
        if position_frames:
            position_frames = [frame for frame in position_frames if not frame.empty]
            if position_frames:
                positions_df = pd.concat(position_frames, ignore_index=True)
                positions_df.to_excel(writer, sheet_name='positions', index=False)

                pos_work = positions_df.copy()
                if 'ISIN' not in pos_work.columns:
                    pos_work['ISIN'] = None
                if 'Symbol' not in pos_work.columns:
                    pos_work['Symbol'] = None
                if 'Net Trade Cash (EUR)' not in pos_work.columns:
                    pos_work['Net Trade Cash (EUR)'] = None
                if 'Net Quantity' not in pos_work.columns:
                    pos_work['Net Quantity'] = None

                # Prefer ISIN as the consolidated key; fallback to Symbol.
                pos_work['Position Key'] = pos_work['ISIN'].fillna(pos_work['Symbol'])
                pos_work['Net Trade Cash (EUR)'] = pd.to_numeric(pos_work['Net Trade Cash (EUR)'], errors='coerce').fillna(0.0)
                pos_work['Net Quantity'] = pd.to_numeric(pos_work['Net Quantity'], errors='coerce').fillna(0.0)

                consolidated = (
                    pos_work.groupby('Position Key', dropna=False)
                    .agg(
                        Symbol=('Symbol', 'first'),
                        ISIN=('ISIN', 'first'),
                        Net_Trade_Cash_EUR=('Net Trade Cash (EUR)', 'sum'),
                        Net_Quantity=('Net Quantity', 'sum'),
                        Brokerages=('Brokerage', lambda x: ', '.join(sorted(set(x))))
                    )
                    .reset_index()
                )
                consolidated.to_excel(writer, sheet_name='positions_consolidated', index=False)

    logging.info(f"Wrote portfolio breakdown report to {out_file}")
    return out_file, accounts_df, brokerage_totals, client_totals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate portfolio breakdown report.")
    parser.add_argument('-c', '--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('-i', '--input-path', default=None, help='Input path (overrides config)')
    parser.add_argument('-o', '--output-path', default='results', help='Output directory for report')
    parser.add_argument('--as-of', default=None, help='As-of date (YYYY-MM-DD). Defaults to latest date in data.')
    args = parser.parse_args()

    build_portfolio_breakdown(args.config, args.input_path, args.output_path, args.as_of)
