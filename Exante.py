from typing import Tuple, List, Union

import pandas as pd
import os
import logging
import utils
import argparse


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    rename_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower == 'operation type':
            rename_map[col] = 'Operation type'
        elif col_lower == 'eur equivalent':
            rename_map[col] = 'EUR equivalent'
        elif col_lower == 'net asset value':
            rename_map[col] = 'Net Asset Value'
        elif col_lower == 'date':
            rename_map[col] = 'Date'
        elif col_lower == 'when':
            rename_map[col] = 'When'

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def _parse_dates(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, errors='coerce', format='mixed', dayfirst=True)
    except TypeError:
        return pd.to_datetime(series, errors='coerce', dayfirst=True)


def _read_exante_output(output_dir: str, source_order: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    nav_file = os.path.join(output_dir, 'NAV.xlsx')
    trades_file = os.path.join(output_dir, 'Trades.xlsx')

    if not os.path.exists(nav_file):
        logging.error(f"NAV file not found: {nav_file}")
        return pd.DataFrame(), pd.DataFrame()

    nav_df = pd.read_excel(nav_file)
    nav_df = _normalize_columns(nav_df)
    nav_df['_source_order'] = source_order

    if not os.path.exists(trades_file):
        logging.warning(f"Trades file not found: {trades_file}. Using empty trades.")
        trades_df = pd.DataFrame(columns=['When', 'Operation type', 'EUR equivalent'])
    else:
        trades_df = pd.read_excel(trades_file)
        trades_df = _normalize_columns(trades_df)

    trades_df['_source_order'] = source_order
    return nav_df, trades_df


def read_data(output_dir: Union[str, List[str]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read and process NAV and Trades data from Excel files for Exante format
    """
    output_dirs = output_dir if isinstance(output_dir, (list, tuple)) else [output_dir]
    logging.info(f"Reading Exante data from {len(output_dirs)} output directory(ies)")

    nav_frames = []
    trade_frames = []
    for idx, output_path in enumerate(output_dirs):
        logging.info(f"Reading Exante data from {output_path}")
        nav_df, trades_df = _read_exante_output(output_path, idx)
        if not nav_df.empty:
            nav_frames.append(nav_df)
        if not trades_df.empty:
            trade_frames.append(trades_df)

    empty_nav = pd.DataFrame(columns=['Date', 'Net Asset Value'])
    empty_trades = pd.DataFrame(columns=['When', 'Operation type', 'EUR equivalent'])

    if not nav_frames:
        logging.error("No Exante NAV data found across output directories")
        return empty_nav, empty_trades

    nav_df = pd.concat(nav_frames, ignore_index=True)
    trades_df = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()

    logging.info(f"Read {len(nav_df)} NAV records and {len(trades_df)} trade records")

    # Drop rows with NaN values in Net Asset Value
    if 'Net Asset Value' not in nav_df.columns:
        logging.error("Missing 'Net Asset Value' column in NAV data")
        return empty_nav, empty_trades

    initial_len = len(nav_df)
    nav_df = nav_df.dropna(subset=['Net Asset Value'])
    dropped_rows = initial_len - len(nav_df)
    if dropped_rows > 0:
        logging.info(f"Dropped {dropped_rows} rows with NaN Net Asset Value")

    # Process NAV data
    if 'Date' not in nav_df.columns:
        logging.error("Missing 'Date' column in NAV data")
        return empty_nav, empty_trades

    nav_df['Date'] = _parse_dates(nav_df['Date']).dt.normalize()
    invalid_dates = nav_df['Date'].isna().sum()
    if invalid_dates > 0:
        logging.warning(f"Dropping {invalid_dates} NAV rows with invalid dates")
        nav_df = nav_df.dropna(subset=['Date'])
    nav_df['Net Asset Value'] = pd.to_numeric(nav_df['Net Asset Value'], errors='coerce')
    nav_df = nav_df.sort_values(['Date', '_source_order'])

    nav_dupes = nav_df.duplicated(subset=['Date']).sum()
    if nav_dupes > 0:
        logging.warning(f"Found {nav_dupes} duplicate NAV dates across files; keeping the latest")
        nav_df = nav_df.drop_duplicates(subset=['Date'], keep='last')

    # Process Trades data
    if trades_df.empty:
        trades_df = empty_trades.copy()
    else:
        if 'Operation type' not in trades_df.columns:
            logging.error("Missing 'Operation type' column in trades data")
            return nav_df, empty_trades
        if 'When' not in trades_df.columns:
            logging.error("Missing 'When' column in trades data")
            return nav_df, empty_trades

        trades_df['Operation type'] = trades_df['Operation type'].astype(str).str.strip().str.upper()
        trades_df['When'] = _parse_dates(trades_df['When']).dt.normalize()
        invalid_trade_dates = trades_df['When'].isna().sum()
        if invalid_trade_dates > 0:
            logging.warning(f"Dropping {invalid_trade_dates} trade rows with invalid dates")
            trades_df = trades_df.dropna(subset=['When'])
        if 'EUR equivalent' in trades_df.columns:
            trades_df['EUR equivalent'] = pd.to_numeric(trades_df['EUR equivalent'], errors='coerce')

        # Filter for only FUNDING/WITHDRAWAL operations
        trades_df = trades_df[trades_df['Operation type'] == 'FUNDING/WITHDRAWAL']

        if 'Transaction ID' in trades_df.columns:
            trade_dupes = trades_df.duplicated(subset=['Transaction ID']).sum()
            if trade_dupes > 0:
                logging.warning(f"Found {trade_dupes} duplicate Transaction IDs across files; keeping the latest")
            trades_df = trades_df.drop_duplicates(subset=['Transaction ID'], keep='last')
        else:
            trades_df = trades_df.drop_duplicates()

        trades_df = trades_df.sort_values(['When', '_source_order'])

    if '_source_order' in nav_df.columns:
        nav_df = nav_df.drop(columns=['_source_order'])
    if '_source_order' in trades_df.columns:
        trades_df = trades_df.drop(columns=['_source_order'])

    logging.info(f"Filtered to {len(trades_df)} funding/withdrawal records")
    logging.info(f"Processed data from {nav_df['Date'].min()} to {nav_df['Date'].max()}")
    return nav_df, trades_df


def find_section_start(
    data: pd.DataFrame,
    header_pattern: str,
    match: str = 'first'
) -> tuple[int, list]:
    """
    Find the start of a section and its headers.
    
    Args:
        data (pd.DataFrame): The raw DataFrame
        header_pattern (str): Pattern to match for section start
        match (str): 'first' or 'last' match for the header pattern
        
    Returns:
        tuple[int, list]: Index where section starts and list of headers
    """
    match_indices = []
    for idx in range(len(data)):
        row = data.iloc[idx]
        row_str = '\t'.join([str(x) for x in row if pd.notna(x)])
        if header_pattern in row_str:
            match_indices.append(idx)
            if match == 'first':
                break

    if not match_indices:
        return -1, []

    idx = match_indices[-1] if match == 'last' else match_indices[0]
    row = data.iloc[idx]
    row_values = [str(x).strip() for x in row if pd.notna(x)]

    # If the matched row is a section title, use the next row as headers.
    if len(row_values) <= 1 and idx + 1 < len(data):
        headers = [str(x).strip() for x in data.iloc[idx + 1] if pd.notna(x)]
        return idx + 1, headers

    return idx, row_values


def extract_section(
    data: pd.DataFrame,
    header_pattern: str,
    section_name: str,
    match: str = 'first'
) -> pd.DataFrame:
    """
    Extract a specific section from the CSV data based on its header pattern.
    
    Args:
        data (pd.DataFrame): The raw DataFrame containing all data
        header_pattern (str): The pattern that identifies the section header
        section_name (str): Name of the section being extracted
        
    Returns:
        pd.DataFrame: Processed section data
    """
    try:
        # Find the section start and headers
        start_idx, headers = find_section_start(data, header_pattern, match=match)
        if start_idx == -1:
            logging.warning(f"Header pattern '{header_pattern}' not found for {section_name} section")
            return pd.DataFrame()

        # Start collecting data from the row after headers
        section_data = []
        current_idx = start_idx + 1

        # Collect data rows until we hit an empty row or end of file
        while current_idx < len(data):
            row = data.iloc[current_idx]
            if pd.isna(row[0]) or all(pd.isna(x) for x in row):
                break
            section_data.append([str(x) if pd.notna(x) else '' for x in row[:len(headers)]])
            current_idx += 1

        if not section_data:
            logging.warning(f"No data found for {section_name} section")
            return pd.DataFrame()

        # Create DataFrame with proper headers
        processed_df = pd.DataFrame(section_data, columns=headers)
        processed_df['Section'] = section_name

        return processed_df

    except Exception as e:
        logging.error(f"Error processing {section_name} section: {e}")
        return pd.DataFrame()


def read_file_with_encoding(file_path: str) -> tuple[str, str, list]:
    """
    Read the file and determine its encoding and delimiter.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        tuple[str, str, list]: Encoding, delimiter, and list of lines
    """
    encodings = ['utf-16', 'utf-8-sig', 'utf-8', 'latin1']

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                lines = f.readlines()
                if len(lines) > 1:
                    # Look at the first few non-empty lines to determine delimiter
                    for line in lines[1:5]:  # Check first 5 lines
                        line = line.strip()
                        if not line:
                            continue
                        # Count occurrences of potential delimiters
                        tab_count = line.count('\t')
                        comma_count = line.count(',')
                        if tab_count > comma_count:
                            return encoding, '\t', lines
                        elif comma_count > 0:
                            return encoding, ',', lines

        except UnicodeDecodeError:
            continue
        except Exception as e:
            logging.error(f"Error reading file with {encoding} encoding: {e}")
            continue

    return None, None, []


def process(file_path: str):
    """
    Process an Exante CSV file and extract NAV and Activity sections.
    
    Args:
        file_path (str): Path to the Exante CSV file
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        # Set up logging
        utils.setup_logging('Exante')
        logging.info(f"Processing Exante CSV file: {file_path}")

        # Validate file exists
        if not os.path.exists(file_path):
            logging.error(f"File {file_path} does not exist")
            return

        # First read the file to determine encoding
        encoding = None
        for enc in ['utf-16', 'utf-8-sig', 'utf-8', 'latin1']:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    f.read()
                    encoding = enc
                    break
            except UnicodeDecodeError:
                continue

        if not encoding:
            logging.error("Could not determine file encoding")
            return

        logging.info(f"Detected encoding: {encoding}")

        # Read the file line by line and process manually
        rows = []
        with open(file_path, 'r', encoding=encoding) as f:
            for line in f:
                # Split on tab while preserving quoted strings
                row = []
                current_field = ''
                in_quotes = False

                for char in line.strip():
                    if char == '"':
                        in_quotes = not in_quotes
                    elif char == '\t' and not in_quotes:
                        row.append(current_field.strip('"'))
                        current_field = ''
                    else:
                        current_field += char

                if current_field:
                    row.append(current_field.strip('"'))

                rows.append(row)

        # Convert to DataFrame
        raw_data = pd.DataFrame(rows)

        logging.info(f"Successfully read {len(raw_data)} rows from the file")

        # Create output directory based on the CSV filename and location
        input_dir = os.path.dirname(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = os.path.join(input_dir, base_name)
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Created output directory: {output_dir}")

        # Find the transaction header row for activity data
        transaction_header = "Transaction ID"
        activity_header_row = None
        for idx in range(len(raw_data)):
            row = raw_data.iloc[idx]
            row_str = '\t'.join([str(x) for x in row if pd.notna(x)])
            if transaction_header in row_str:
                activity_header_row = idx
                break

        if activity_header_row is not None:
            # Extract activity data starting from the header row
            headers = [str(x).strip() for x in raw_data.iloc[activity_header_row] if pd.notna(x)]
            activity_data = []
            for idx in range(activity_header_row + 1, len(raw_data)):
                row = raw_data.iloc[idx]
                if pd.isna(row[0]) or all(pd.isna(x) for x in row):
                    break
                activity_data.append([str(x) if pd.notna(x) else '' for x in row[:len(headers)]])

            if activity_data:
                activity_df = pd.DataFrame(activity_data, columns=headers)
                trades_file = os.path.join(output_dir, "Trades.xlsx")
                activity_df.to_excel(trades_file, index=False)
                logging.info(f"Saved trades data to {trades_file}")
            else:
                logging.warning("No trades data found")

        # Extract and save NAV section
        nav_header = "Net Asset Value"
        nav_df = extract_section(raw_data, nav_header, "NAV", match='last')
        if not nav_df.empty:
            nav_file = os.path.join(output_dir, "NAV.xlsx")
            nav_df.to_excel(nav_file, index=False)
            logging.info(f"Saved NAV data to {nav_file}")
        else:
            logging.warning("No NAV data found")

    except Exception as e:
        logging.error(f"Error processing file: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    # Argument configuration
    parser = argparse.ArgumentParser(description="Process an Exante CSV file.")
    parser.add_argument("file_path", help="Path to the CSV file")
    args = parser.parse_args()

    process(args.file_path)
