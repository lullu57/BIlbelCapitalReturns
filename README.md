# GIPS-Compliant Returns Calculator

A Python-based tool for calculating GIPS-compliant investment returns across multiple brokerage accounts (IBKR and Exante). The system processes broker-specific reports and generates standardized returns metrics including Time-Weighted Returns (TWR), fee-adjusted net returns, and comprehensive NAV tracking.

## Features

- **GIPS-Compliant Calculations**: Beginning-of-period weighting, internal dispersion, large cash flow detection
- **Multi-Brokerage Support**: IBKR and Exante with extensible architecture
- **Comprehensive Fee Tracking**: Management fees (quarterly) and performance fees (with hurdle rate)
- **NAV Tracking**: Gross NAV (actual) and Net NAV (after fees) with flow handling
- **Configurable**: YAML-based configuration for clients, fees, and reporting periods
- **Composite Returns**: Asset-weighted composite across all accounts

## Setup

### Requirements

```
pandas==2.2.3
numpy==2.0.2
openpyxl==3.1.5
PyYAML==6.0.1
numpy_financial==1.0.0
pytest==8.2.1
pytest-cov==5.0.0
hypothesis==6.98.8
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### Configuration

Copy the example configuration and customize:
```bash
cp config.example.yaml config.yaml
```

Edit `config.yaml` to configure:
- **Brokerages**: Enable/disable IBKR, Exante
- **Clients**: List of clients to process
- **Fees**: Management fee (quarterly), performance fee rate, hurdle rate
- **Periods**: Fiscal year start month, reporting years

Example configuration:
```yaml
# Fee structure
fees:
  management_fee_quarterly: 0.0025  # 0.25% per quarter = 1% annual
  performance_fee_rate: 0.25        # 25% of gains above hurdle
  hurdle_rate_annual: 0.06          # 6% annual hurdle

# Fiscal year configuration
periods:
  fiscal_year_start_month: 2        # February
  reporting_years: [2022, 2023, 2024, 2025]
```

### Directory Structure

```
ReturnsCalculator/
├── input/                    # Input directory for raw broker files
│   ├── IBKR/                # IBKR input files (*.csv)
│   └── Exante/              # Exante input files (*.csv)
├── results/                  # Generated results
│   ├── Combined/            # Asset-weighted composite results
│   ├── IBKR_{client}/       # Per-client IBKR results
│   └── Exante_{client}/     # Per-client Exante results
├── logs/                     # Processing logs
├── tests/                    # Test suite
├── config.yaml              # Your configuration (gitignored)
├── config.example.yaml      # Example configuration
├── twr_calculator.py        # Main calculator engine
├── fee_calculator.py        # Fee calculation logic
├── flow_utils.py            # Cash flow utilities
├── config_loader.py         # Configuration loading
├── IBKR.py                  # IBKR data processing
├── Exante.py                # Exante data processing
└── utils.py                 # Utility functions
```

## Usage

### Quick Start

1. Export broker reports:
   - **IBKR**: Export performance report as CSV from Client Portal/TWS
   - **Exante**: Export account statement as CSV

2. Place files in input directories:
   ```
   input/IBKR/{client_name}.csv
   input/Exante/{client_name}.csv
   ```

3. Run the calculator:
   ```bash
   python twr_calculator.py
   ```

### Command Line Options

```bash
# Use default config.yaml
python twr_calculator.py

# Specify custom config
python twr_calculator.py --config my_config.yaml

# Override input path
python twr_calculator.py --input-path /path/to/input

# Legacy mode (for backward compatibility)
python twr_calculator.py --legacy -i input
```

## How It Works

### 1. Data Processing

The system processes broker files in three main steps:

1. **Account Discovery**
   - Scans input directory for CSV files
   - Identifies clients from filenames
   - Creates processing directories

2. **Broker-specific Processing**
   - **IBKR**: Extracts NAV from allocation reports, flows from deposits/withdrawals
   - **Exante**: Converts to NAV.xlsx and Trades.xlsx

3. **Returns Calculation**
   - Calculates sub-period returns between cash flows
   - Computes monthly TWR through geometric linking
   - Applies fees and tracks Net NAV

### 2. Return Methodologies

#### Time-Weighted Return (TWR)
- Eliminates impact of cash flow timing
- Calculated between each cash flow (sub-period returns)
- Geometrically linked for monthly/annual periods
- GIPS-compliant for performance comparison

#### Gross vs Net Returns
- **Gross TWR**: Pure investment performance before fees
- **Net TWR**: Returns after management and performance fees
- Net TWR = Compounded product of annual net returns

### 3. Fee Structure

#### Management Fee
- **Rate**: 0.25% per quarter (1% annual)
- **Timing**: Deducted at start of each fiscal quarter
- **Base**: Current Net NAV

#### Performance Fee
- **Rate**: 25% of gains above hurdle
- **Hurdle**: 6% annual, accumulates quarterly (Q1: 1.5%, Q2: 3%, Q3: 4.5%, Q4: 6%)
- **Timing**: Deducted at end of each fiscal quarter
- **Base**: Investment gain since year start (excluding flows)
- **Carry-forward**: Shortfall carries to next year (non-compounding)

### 4. NAV Tracking

The system maintains two parallel NAV series:

| NAV Type | Description |
|----------|-------------|
| **Gross NAV** | Actual NAV from accounts (includes flows) |
| **Net NAV** | Gross NAV minus accumulated fees |

```
Net NAV = Gross NAV - Accumulated Fees
Fee Drag = Gross NAV - Net NAV
```

### 5. Output Files

Results are saved with timestamps (YYYYMMDD_HHMMSS):

#### Monthly Returns (`monthly_returns_*.xlsx`)
- Monthly TWR values
- Start/end-of-month NAVs
- Monthly flows

#### Daily Returns (`daily_returns_*.xlsx`)
- Sub-period returns
- Daily NAV snapshots
- Flow attribution

#### Return Summary (`return_summary_*.xlsx`)
Contains comprehensive metrics:

**Performance Metrics:**
- Total Absolute Return (Gross/Net TWR)
- Annualized Return (Gross/Net)
- Initial and Final NAV
- Total Flows
- Fee Impact

**Quarterly NAV Tracking:**
- Gross NAV (Actual from accounts)
- Net NAV (After fees)
- Management fee per quarter
- Performance fee per quarter
- Cumulative fee impact

**Annual Summaries:**
- Year-over-year NAV progression
- Annual fee totals
- Active accounts per period

## Example Output

```
Total Absolute Return (Gross TWR): 2037.52%
Total Absolute Return (Net TWR): 1130.88%
Annualized Return (Gross TWR): 140.26%
Annualized Return (Net TWR): 104.88%

Initial NAV (Actual): €7,809.27
Final NAV - Gross: €1,496,663.75
Final NAV - Net: €1,036,211.72
Total Flows: €404,361.50
Total Fees: €460,452.04 (Mgmt: €8,239 + Perf: €452,213)
Fee Impact: €460,452.04

2022 Returns: Gross 188.24% → Net 141.68%
2023 Returns: Gross 89.17% → Net 67.38%
2024 Returns: Gross 142.13% → Net 107.09%
2025 Returns: Gross 61.91% → Net 46.93%
```

## GIPS Compliance Features

- **Beginning-of-Period Weighting**: Composite weights use start-of-month NAV
- **Internal Dispersion**: Calculated for composites with 6+ accounts
- **Large Cash Flow Detection**: Flags flows >10% of NAV
- **Active Account Tracking**: Only includes accounts with data for each period
- **Geometric Linking**: Sub-period returns properly compounded

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_fee_calculator.py

# Run in parallel
pytest -n auto
```

## Notes

- All monetary values are converted to EUR
- Fiscal year defaults to February start (Feb 1 - Jan 31)
- Logs are generated in the logs directory
- Configuration file (`config.yaml`) is gitignored for security

## Error Handling

- Validates input file formats
- Handles missing or corrupted data
- Logs processing errors and warnings
- Skips invalid return calculations
- Provides detailed error messages

## Dependencies

Requires Python 3.8+ with:
- **pandas**: Data processing and analysis
- **numpy**: Numerical computations
- **openpyxl**: Excel file handling
- **PyYAML**: Configuration loading

See `requirements.txt` for complete list.
