from typing import Tuple

import pandas as pd
import os
import logging
import utils
import argparse


def read_data(output_dir) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read and process NAV and Trades data from Excel files for Exante format
    """

    nav_file = os.path.join(output_dir, 'NAV.xlsx')
    trades_file = os.path.join(output_dir, 'Trades.xlsx')

    logging.info(f"Reading Exante data from {nav_file} and {trades_file}")

    # Read files
    nav_df = pd.read_excel(nav_file)
    trades_df = pd.read_excel(trades_file)

    logging.info(f"Read {len(nav_df)} NAV records and {len(trades_df)} trade records")

    # Drop rows with NaN values in Net Asset Value
    initial_len = len(nav_df)
    nav_df = nav_df.dropna(subset=['Net Asset Value'])
    dropped_rows = initial_len - len(nav_df)
    if dropped_rows > 0:
        logging.info(f"Dropped {dropped_rows} rows with NaN Net Asset Value")

    # Process NAV data
    nav_df['Date'] = pd.to_datetime(nav_df['Date']).dt.normalize()
    nav_df = nav_df.sort_values('Date')

    # Process Trades data
    trades_df['When'] = pd.to_datetime(trades_df['When']).dt.normalize()
    # Filter for only FUNDING/WITHDRAWAL operations
    trades_df = trades_df[trades_df['Operation type'] == 'FUNDING/WITHDRAWAL']
    trades_df = trades_df.sort_values('When')

    logging.info(f"Filtered to {len(trades_df)} funding/withdrawal records")
    logging.info(f"Processed data from {nav_df['Date'].min()} to {nav_df['Date'].max()}")
    return nav_df, trades_df


def find_section_start(data: pd.DataFrame, header_pattern: str) -> tuple[int, list]:
    """
    Find the start of a section and its headers.
    
    Args:
        data (pd.DataFrame): The raw DataFrame
        header_pattern (str): Pattern to match for section start
        
    Returns:
        tuple[int, list]: Index where section starts and list of headers
    """
    for idx in range(len(data)):
        row = data.iloc[idx]
        row_str = '\t'.join([str(x) for x in row if pd.notna(x)])
        if header_pattern in row_str:
            # Get the next row as headers if it exists
            if idx + 1 < len(data):
                headers = [str(x).strip() for x in data.iloc[idx + 1] if pd.notna(x)]
                return idx + 1, headers
            return idx, [str(x).strip() for x in row if pd.notna(x)]
    return -1, []


def extract_section(data: pd.DataFrame, header_pattern: str, section_name: str) -> pd.DataFrame:
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
        start_idx, headers = find_section_start(data, header_pattern)
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
        nav_header = "NAV"
        nav_df = extract_section(raw_data, nav_header, "NAV")
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
