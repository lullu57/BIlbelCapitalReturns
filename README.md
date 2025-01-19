# Returns Calculator

A Python-based tool for calculating investment returns across multiple brokerage accounts (IBKR, Exante, and Poems). The system processes broker-specific reports and generates standardized returns metrics including Time-Weighted Returns (TWR) and Internal Rate of Return (IRR).

## Setup

### Requirements

```
pandas>=1.5.0
numpy>=1.21.0
openpyxl>=3.0.0
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### Directory Structure

Create the following directory structure:
```
ReturnsCalculator/
├── IBKR/                  # IBKR reports and processed data
├── Exante/               # Exante reports and processed data
├── docs/                 # Documentation
├── logs/                 # Processing logs
└── results/              # Generated results
```

## Usage

1. Export broker reports:
   - **IBKR**: Export performance report as CSV from Client Portal/TWS
   - **Exante**: Export account statement as CSV (tab-delimited)

2. Place the exported files in their respective directories:
   ```
   IBKR/{client_name}.csv
   Exante/{client_name}.csv
   ```

3. Calculate returns:
   ```bash
   python calculation.py
   ```

   This will:
   - Automatically process all IBKR and Exante files in their respective directories
   - Generate processed data files in client-specific subdirectories
   - Calculate returns and create result files
   - No need to run `ibkr.py` or `exante.py` separately

   The results will be available in the `results/` directory.

> Note: While you can run `ibkr.py` and `exante.py` separately for debugging purposes, it's not necessary for normal operation as `calculation.py` handles the entire workflow.

## How It Works

### 1. Data Processing

#### IBKR Processing (`ibkr.py`)
- Reads the performance report CSV
- Splits the file into sections:
  - Allocation by Asset Class (daily portfolio values)
  - Deposits and Withdrawals
  - Dividends
  - Fees
  - Interest
  - Trade Summary
- Each section is saved as a separate CSV file in `IBKR/{client_name}/`

#### Exante Processing (`exante.py`)
- Reads the tab-delimited CSV report
- Extracts and processes:
  - NAV data (daily portfolio values)
  - Transaction history
- Generates two Excel files in `Exante/{client_name}/`:
  - NAV.xlsx: Daily portfolio values
  - Trades.xlsx: Transaction history

### 2. Returns Calculation (`calculation.py`)

The main calculation process follows these steps:

1. **Data Loading**
   - Reads processed files from both IBKR and Exante directories
   - Validates data consistency and formats

2. **Sub-period Returns**
   - Calculates returns between each cash flow
   - Uses Time-Weighted Return (TWR) methodology
   - Accounts for deposits, withdrawals, and other cash flows

3. **Monthly Returns**
   - Aggregates sub-period returns into monthly returns
   - Geometrically links returns within each month
   - Captures start-of-month NAV for composite calculations

4. **Rolling Returns**
   - Calculates rolling 6-month returns
   - Provides medium-term performance perspective

5. **Composite Returns**
   - Creates asset-weighted composite for all accounts
   - Follows GIPS-like methodology
   - Includes only active accounts in each period

6. **Performance Metrics**
   - Calculates absolute and annualized returns
   - Computes Internal Rate of Return (IRR)
   - Generates period-specific returns (2022, 2023, 2024 YTD)

### 3. Output Generation

Results are saved in the `results/` directory with timestamps:
- Individual account results
- Combined portfolio results
- Monthly and rolling returns
- Performance summaries

## File Formats

For detailed information about file formats and structures, see [docs/files.md](docs/files.md).

## Notes

- All monetary values are converted to EUR
- Returns are calculated net of fees
- IBKR reports use varying date formats (MM/DD/YY or YYYYMMDD)
- Exante dates are standardized to YYYY-MM-DD
- Logs are generated in the `logs/` directory for troubleshooting

## Error Handling

The system includes comprehensive error handling:
- Validates input file formats
- Checks for missing or corrupted data
- Logs processing errors and warnings
- Gracefully handles missing sections in IBKR reports

## Limitations

- IBKR reports must be in the standard performance report format
- Exante reports must be tab-delimited CSV files
- All accounts must use EUR as the base currency
- Historical data limited to available broker reports 