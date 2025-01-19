import argparse
from typing import Tuple

import pandas as pd
import os
import logging
import utils


def read_data(client_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read and process IBKR data from the processed CSV files.
    Uses allocation_by_asset_class.csv for NAV and deposits_and_withdrawals.csv for flows.
    Only considers 'Cash Receipts / Electronic Fund Transfers' as valid flows.
    """
    logging.info(f"Reading IBKR data from {client_dir}")

    # Read allocation data for NAV
    nav_file = os.path.join(client_dir, 'allocation_by_asset_class.csv')
    logging.info(f"Reading NAV data from {nav_file}")
    nav_df = pd.read_csv(nav_file)

    # Process NAV data - now using proper column names
    try:
        nav_df['Date'] = pd.to_datetime(nav_df['Date'], format='%Y%m%d')
    except ValueError:
        logging.info("Trying alternative date format for NAV data")
        nav_df['Date'] = pd.to_datetime(nav_df['Date'], format='%m/%d/%y')

    nav_df = nav_df.rename(columns={'NAV': 'Net Asset Value'})
    nav_df = nav_df[['Date', 'Net Asset Value']].copy()
    nav_df['Net Asset Value'] = pd.to_numeric(nav_df['Net Asset Value'], errors='coerce')
    nav_df = nav_df.sort_values('Date')

    logging.info(f"Processed {len(nav_df)} NAV records from {nav_df['Date'].min()} to {nav_df['Date'].max()}")

    # Read deposits and withdrawals data
    flows_file = os.path.join(client_dir, 'deposits_and_withdrawals.csv')
    logging.info(f"Reading flows data from {flows_file}")
    flows_df = pd.read_csv(flows_file)

    # Filter for only valid flow descriptions
    valid_descriptions = [
        'Cash Receipts / Electronic Fund Transfers ',
        'Cash Receipts / Electronic Fund Transfers (n/a) '
    ]
    flows_df = flows_df[flows_df['Description'].isin(valid_descriptions)]
    logging.info(f"Filtered to {len(flows_df)} valid flow records")

    # Process flows - using proper column names and handling MM/DD/YY format
    try:
        flows_df['When'] = pd.to_datetime(flows_df['Date'], format='%m/%d/%y')
        logging.info("Successfully parsed dates in MM/DD/YY format")
    except ValueError as e:
        logging.warning(f"Failed to parse dates in MM/DD/YY format: {e}")
        try:
            flows_df['When'] = pd.to_datetime(flows_df['Date'], format='%Y%m%d')
            logging.info("Successfully parsed dates in YYYYMMDD format")
        except ValueError as e:
            logging.error(f"Failed to parse dates in both formats: {e}")
            raise

    flows_df['EUR equivalent'] = pd.to_numeric(flows_df['Amount'], errors='coerce')
    flows_df['Operation type'] = 'FUNDING/WITHDRAWAL'
    flows_df['Adjusted EUR'] = flows_df['EUR equivalent']  # No adjustment needed for deposits/withdrawals

    # Sort and finalize
    nav_df = nav_df.sort_values('Date')
    flows_df = flows_df[['When', 'Operation type', 'EUR equivalent', 'Adjusted EUR']].sort_values('When')

    logging.info(f"Final processed data: {len(nav_df)} NAV records, {len(flows_df)} flow records")
    return nav_df, flows_df


# Process section data to ensure consistent column names and format
def process_section_data(section_df: pd.DataFrame, section_name: str) -> pd.DataFrame:
    # Get the header row (first row after section name)
    header_row = section_df.iloc[1]
    data_rows = section_df.iloc[2:]

    # Create new dataframe with proper headers
    processed_df = pd.DataFrame(data_rows.values, columns=header_row)

    # Add standard columns
    processed_df['Section'] = section_name
    processed_df['Type'] = 'Data'

    return processed_df


def process(file_path: str):
    """
    Process an IBKR CSV file and split it into separate section files.

    Args:
        file_path (str): Path to the IBKR CSV file

    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        # Set up logging
        utils.setup_logging('IBKR')
        logging.info(f"Processing IBKR CSV file: {file_path}")

        # Validate file exists
        if not os.path.exists(file_path):
            logging.error(f"File {file_path} does not exist")
            return

        # First read to get max columns
        with open(file_path, 'r') as f:
            max_cols = max(len(line.split(',')) for line in f)
        logging.info(f"Detected {max_cols} columns in the file")

        # Now read with the correct number of columns
        raw_data = pd.read_csv(file_path, header=None, on_bad_lines='warn',
                               dtype=str, engine='python',
                               names=range(max_cols))
        logging.info(f"Successfully read {len(raw_data)} rows from the file")

        # Create output directory based on the CSV filename and location
        input_dir = os.path.dirname(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = os.path.join(input_dir, base_name)
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Created output directory: {output_dir}")

        # Step 1: Identify unique sections and exclude specific sections
        excluded_sections = ['Introduction', 'Disclosure']
        unique_sections = [section for section in raw_data[0].dropna().unique()
                           if section not in excluded_sections]
        logging.info(f"Found {len(unique_sections)} sections to process")

        # Step 2: Process each section
        for section in unique_sections:
            logging.info(f"Processing section: {section}")

            # Filter rows belonging to the current section
            section_data = raw_data[raw_data[0] == section]
            start_idx = section_data.index[0]
            end_idx = section_data.index[-1]

            # Find the end of the section
            while end_idx < len(raw_data) - 1 and pd.isna(raw_data.loc[end_idx + 1, 0]):
                end_idx += 1

            # Extract and process the section
            section_df = raw_data.loc[start_idx:end_idx].reset_index(drop=True)
            processed_df = process_section_data(section_df, section)

            # Save to CSV
            file_name = os.path.join(output_dir, f"{section.replace(' ', '_').lower()}.csv")
            processed_df.to_csv(file_name, index=False)
            logging.info(f"Saved processed data to {file_name}")

        logging.info(f"Successfully processed all sections and saved to {output_dir}")
        return True

    except Exception as e:
        logging.error(f"Error processing file: {e}", exc_info=True)


if __name__ == "__main__":
    # Argument configuration
    parser = argparse.ArgumentParser(description="Process an IBKR CSV file.")
    parser.add_argument("file_path", help="Path to the CSV file")
    args = parser.parse_args()

    process(args.file_path)
