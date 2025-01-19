# Returns Calculator

A Python-based tool for calculating investment returns across multiple brokerage accounts (IBKR, Exante, and Poems). The system processes broker-specific reports and generates standardized returns metrics including Time-Weighted Returns (TWR) and Internal Rate of Return (IRR).

## Setup

### Requirements

```
pandas>=2.2.3
numpy>=2.0.2
numpy_financial>=1.0.0
openpyxl>=3.1.5
python-dateutil>=2.9.0
pytz>=2024.2
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### Directory Structure

Create the following directory structure:
```
ReturnsCalculator/
├── Input/                 # Input directory for raw broker files
│   ├── IBKR/             # IBKR input files
│   └── Exante/          # Exante input files
├── logs/                 # Processing logs
└── results/             # Generated results
    ├── Combined/        # Asset-weighted composite results
    ├── IBKR_{client_name}/
    └── Exante_{client_name}/
```

## Usage

1. Export broker reports:
   - **IBKR**: Export performance report as CSV from Client Portal/TWS
   - **Exante**: Export account statement as CSV (tab-delimited)

2. Place the exported files in their respective directories:
   ```
   Input/IBKR/{client_name}.csv
   Input/Exante/{client_name}.csv
   ```

3. Calculate returns:
   ```bash
   python TWR_Calculator.py -i Input
   ```

   This will:
   - Automatically scan for and process all broker files
   - Generate processed data files in client-specific subdirectories
   - Calculate individual and composite returns
   - Create timestamped result files in the results directory

## How It Works

### 1. Data Processing

The system processes broker files in three main steps:

1. **Account Discovery**
   - Scans Input directory for CSV files
   - Identifies client names from filenames
   - Creates processing directories

2. **Broker-specific Processing**
   - IBKR reports are split into component files:
     - Allocation by Asset Class (NAV data)
     - Deposits and Withdrawals
     - Dividends, Fees, Interest details
   - Exante reports are converted to:
     - NAV.xlsx (daily portfolio values)
     - Trades.xlsx (transactions and flows)

3. **Returns Calculation**
   - Calculates sub-period returns between cash flows
   - Computes monthly TWR through geometric linking
   - Generates rolling 6-month returns
   - Calculates IRR using cash flows and NAVs
   - Creates asset-weighted composite returns

### 2. Return Methodologies

#### Time-Weighted Return (TWR)
- Eliminates impact of cash flow timing
- Calculated between each cash flow
- Geometrically linked for longer periods
- Used for performance comparison

#### Internal Rate of Return (IRR)
- Considers timing and size of cash flows
- Uses actual day count for annualization
- Represents investor's actual experience
- Calculated using numpy_financial.irr

#### GIPS-style Composite
- Asset-weighted using start-of-month NAVs
- Only includes active accounts
- Provides period-specific account lists
- Composite IRR from aggregated flows

### 3. Output Files

Results are saved with timestamps (YYYYMMDD_HHMMSS):

1. **Monthly Returns**
   - Monthly TWR values
   - Start-of-month NAVs
   - Used for composite calculations

2. **Six-Month Returns**
   - Rolling 6-month periods
   - Geometrically linked returns
   - Performance trending analysis

3. **Return Summary**
   - Total and annualized TWR
   - Internal Rate of Return
   - Period returns (2022, 2023, 2024 YTD)
   - Active account listings

For detailed file formats and structures, see [docs/files.md](docs/files.md).

## Notes

- All monetary values are converted to EUR
- Returns are calculated net of fees
- Reporting periods align with Feb-Jan cycle
- Logs are generated in the logs directory
- Results use YYYYMMDD_HHMMSS timestamps

## Error Handling

- Validates input file formats
- Handles missing or corrupted data
- Logs processing errors and warnings
- Skips invalid return calculations
- Provides detailed error messages

## Dependencies

The system requires Python 3.8+ and the following key packages:
- pandas: Data processing and analysis
- numpy: Numerical computations
- numpy_financial: IRR calculations
- openpyxl: Excel file handling

See requirements.txt for complete dependency list. 