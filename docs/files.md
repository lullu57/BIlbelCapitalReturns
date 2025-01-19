# File Formats Explanation

## Root Directory Structure

The system expects the following structure in the root directory:

```
ReturnsCalculator/
├── Input/                 # Input directory for raw broker files
│   ├── IBKR/             # IBKR input files
│   │   ├── {client_name}.csv           # Raw IBKR performance report CSV
│   │   └── {client_name}/              # Generated after processing
│   │       ├── allocation_by_asset_class.csv
│   │       ├── deposits_and_withdrawals.csv
│   │       ├── dividends.csv
│   │       ├── fee_summary.csv
│   │       ├── interest_details.csv
│   │       └── trade_summary.csv
│   └── Exante/          # Exante input files
│       ├── {client_name}.csv           # Raw Exante report CSV (tab-delimited)
│       └── {client_name}/              # Generated after processing
│           ├── NAV.xlsx                # Daily portfolio values
│           └── Trades.xlsx             # All transactions and cash flows
├── logs/                 # Processing logs directory
└── results/             # Generated results
    ├── Combined/        # Asset-weighted composite results
    │   ├── monthly_returns_{timestamp}.xlsx
    │   ├── six_month_returns_{timestamp}.xlsx
    │   └── return_summary_{timestamp}.xlsx
    ├── IBKR_{client_name}/        # Individual IBKR account results
    │   ├── monthly_returns_{timestamp}.xlsx
    │   ├── six_month_returns_{timestamp}.xlsx
    │   └── return_summary_{timestamp}.xlsx
    ├── Exante_{client_name}/      # Individual Exante account results
    │   ├── monthly_returns_{timestamp}.xlsx
    │   ├── six_month_returns_{timestamp}.xlsx
    │   └── return_summary_{timestamp}.xlsx
    └── Poems_{client_name}/       # Individual Poems account results
```

## Broker Export Formats

### 1. Interactive Brokers (IBKR)
IBKR reports are exported as CSV files from the IBKR Client Portal or TWS platform. The report contains multiple sections separated by headers, which may include:
- Allocation by Asset Class (daily portfolio allocation across asset types)
- Deposits and Withdrawals (cash movements in/out of the account)
- Trade Summary (details of all trading activity)
- Dividends (dividend payments received)
- Fee Summary (breakdown of fees charged)
- Interest Details (interest earned or paid)
- Other performance metrics

When processed by `ibkr.py`, each section is split into separate CSV files in the client's subdirectory. The exact sections present may vary depending on the account activity during the reporting period.

### 2. Exante
Exante reports are exported as tab-delimited CSV files containing multiple sections:
- Costs and Charges Report (fees and charges breakdown)
- Margin Structure (daily margin requirements)
- Transaction History
- NAV History

When processed by `exante.py`, the data is split into two Excel files:

#### 2.1 NAV.xlsx
Contains daily portfolio values and performance metrics:
- Date (YYYY-MM-DD format)
- Net Asset Value (total portfolio value)
- Daily P&L (profit/loss for the day)
- Cumulative Change (running total of P&L)

#### 2.2 Trades.xlsx
Contains all transactions affecting the portfolio:
- Transaction ID (unique identifier)
- Operation Type (e.g., FUNDING/WITHDRAWAL, TRADE)
- When (timestamp in YYYY-MM-DD HH:MM:SS format)
- Symbol ID (for trades)
- ISIN (international security identifier)
- Amount (transaction amount)
- Currency (transaction currency)
- EUR Equivalent (amount in EUR)
- Comment (transaction details)

The raw Exante CSV includes additional sections like:
- Investment Services costs
- Investment Instruments costs
- Returns before and after costs
- Margin requirements per asset

## Calculation Process and Output Files

### 1. Sub-period Returns
Intermediate calculations that form the basis for all other return metrics:

| Column Name | Description |
|------------|-------------|
| **start_date** | Beginning of the sub-period |
| **end_date** | End of the sub-period |
| **return** | Sub-period return calculated using TWR methodology |
| **start_nav** | Portfolio value at start of period |
| **end_nav** | Portfolio value at end of period |
| **total_flow** | Sum of cash flows during the period |

### 2. Monthly Returns (`monthly_returns_{timestamp}.xlsx`)
Monthly time-weighted returns calculated by geometrically linking sub-period returns:

| Column Name | Description |
|------------|-------------|
| **month** | Month in YYYY-MM format |
| **return** | Monthly return (geometrically linked) |
| **start_of_month_nav** | Portfolio value at start of month (for GIPS composite) |

### 3. Six-Month Returns (`six_month_returns_{timestamp}.xlsx`)
Rolling 6-month returns calculated from monthly returns:

| Column Name | Description |
|------------|-------------|
| **start_month** | Start month of the 6-month period |
| **end_month** | End month of the 6-month period |
| **return** | 6-month rolling return |

### 4. Return Summary (`return_summary_{timestamp}.xlsx`)
Comprehensive performance metrics for each account/composite:

| Metric | Description |
|--------|-------------|
| **Total Absolute Return (TWR)** | Total return over entire period |
| **Annualized Return (TWR)** | Annualized time-weighted return |
| **Internal Rate of Return (IRR)** | Money-weighted return considering cash flows |
| **2022 Returns** | Returns from Feb 2022 to Jan 2023 |
| **2023 Returns** | Returns from Feb 2023 to Jan 2024 |
| **2024 Returns (YTD)** | Returns from Feb 2024 onwards |
| **Active Accounts** | List of accounts included in each period |

### 5. Combined Results
The `Combined` directory contains asset-weighted composite returns:
- Uses GIPS-style methodology
- Weights each account by its start-of-month NAV
- Only includes accounts active in each period
- Provides composite IRR using aggregated cash flows
- Shows which accounts contributed to each period's returns

### Notes:
- All returns are calculated net of fees
- Timestamps in filenames use format: YYYYMMDD_HHMMSS
- IRR calculations use actual day counts for annualization
- The Combined results represent the asset-weighted performance across all accounts
- Period returns (2022, 2023, 2024) align with reporting year (Feb-Jan)

## 1. NAV.xlsx

This file contains data on the Net Asset Value (NAV) of the investment fund over time, as well as daily profit and loss. The structure is as follows:

| Column Name           | Description                                                    |
| --------------------- | -------------------------------------------------------------- |
| **Date**              | The date of the NAV entry.                                     |
| **Net Asset Value**   | The total value of the investment portfolio on the given date. |
| **Daily P&L**         | The profit or loss for that particular day.                    |
| **Cumulative Change** | The cumulative profit or loss from the start of the dataset.   |

### Notes:

- Dates are formatted as `YYYY-MM-DD`.
- Data is sorted by date in ascending order.

## 2. Trades.xlsx

This file contains information about trades and cash flow operations related to the investment fund. The structure is as follows:

| Column Name        | Description                                                                  |
| ------------------ | ---------------------------------------------------------------------------- |
| **Transaction ID** | Unique identifier for the transaction.                                       |
| **Account ID**     | Identifier for the account where the transaction occurred.                   |
| **Symbol ID**      | Identifier for the security involved in the trade (if applicable).           |
| **ISIN**           | ISIN code of the security (if applicable).                                   |
| **Operation Type** | Type of operation (e.g., `FUNDING/WITHDRAWAL`, `TRADE`, etc.).               |
| **When**           | Timestamp of the transaction in `YYYY-MM-DD HH:MM:SS` format.               |
| **Sum Asset**      | The total amount of the asset involved in the transaction.                   |
| **EUR Equivalent** | Value of the transaction in EUR.                                             |
| **Comment**        | Additional details about the transaction.                                    |
| **UUID**           | Unique identifier for the transaction (globally unique).                     |
| **Parent UUID**    | Identifier linking this transaction to a parent transaction (if applicable). |
| **Amount**         | Specific amount related to the transaction (may be blank for some rows).     |
| **Currency**       | Currency of the transaction amount (e.g., EUR, USD).                         |
| **Merchant Name**  | Name of the merchant involved in the transaction (if applicable).            |

### Operation Types:
- **FUNDING/WITHDRAWAL**
- **AUTOCONVERSION**
- **COMMISSION**
- **TRADE**
- **EXCESS MARGIN FEE**
- **INTEREST**
- **CORPORATE ACTION**
- **FEE**
- **DIVIDEND**

### Notes:

- Trades and funding/withdrawals are distinguished by the `Operation Type` column.
- The `EUR Equivalent` column is adjusted by a fee in calculations (e.g., a 0.5% fee for trades).
- Timestamps allow precise alignment with NAV data for time-weighted return calculations.

### Usage:

- This file is essential for identifying cash flows that affect the NAV and calculating returns.
- The `Adjusted EUR` column is derived during processing to reflect the application of fees on trades.

## 3. IBKR Report Files

These files are generated from processing the IBKR performance report CSV. Each file represents a different section of the report:

### 3.1 allocation_by_asset_class.csv

Contains daily portfolio allocation across different asset classes. Structure:

| Column Name | Description |
|------------|-------------|
| **Section** | Always "Allocation by Asset Class" |
| **Type** | Row type: Header or Data |
| **Date** | Date of the allocation entry (YYYYMMDD format) |
| **Equities** | Value held in equities |
| **Cash** | Value held in cash |
| **NAV** | Total Net Asset Value (sum of all allocations) |

### 3.2 deposits_and_withdrawals.csv

Records all cash movements in and out of the account. Structure:

| Column Name | Description |
|------------|-------------|
| **Section** | Always "Deposits And Withdrawals" |
| **Type** | Row type: Header or Data |
| **Date** | Date of the transaction (MM/DD/YY format) |
| **Type** | Always "Data" |
| **Description** | Description of the transaction (e.g., "Cash Receipts / Electronic Fund Transfers") |
| **Amount** | Transaction amount in base currency |

### 3.3 dividends.csv

Records all dividend payments received. Structure:

| Column Name | Description |
|------------|-------------|
| **Section** | Always "Dividends" |
| **Type** | Row type: Header or Data |
| **PayDate** | Date the dividend was paid (YYYYMMDD format) |
| **Ex-Date** | Ex-dividend date (YYYYMMDD format) |
| **Symbol** | Stock symbol |
| **Note** | Type of dividend payment |
| **Quantity** | Number of shares |
| **DividendPerShare** | Dividend amount per share |
| **Amount** | Total dividend amount |

### 3.4 fee_summary.csv

Details of fees charged to the account. Structure:

| Column Name | Description |
|------------|-------------|
| **Section** | Always "Fee Summary" |
| **Type** | Row type: Header or Data |
| **Date** | Date of the fee (YYYYMMDD format) |
| **Description** | Description of the fee |
| **Amount** | Fee amount (negative for charges, positive for refunds) |

### 3.5 interest_details.csv

Records interest earned and paid. Structure:

| Column Name | Description |
|------------|-------------|
| **Section** | Always "Interest Details" |
| **Type** | Row type: Header or Data |
| **Date** | Date of the interest entry (YYYYMMDD format) |
| **Description** | Type of interest (e.g., "Credit Interest", "Debit Interest") with currency |
| **Amount** | Interest amount (negative for paid, positive for received) |

### 3.6 trade_summary.csv

Comprehensive summary of all trading activity. Structure:

| Column Name | Description |
|------------|-------------|
| **Section** | Always "Trade Summary" |
| **Type** | Row type: Header or Data |
| **Financial Instrument** | Type of instrument (e.g., Stocks, Forex) |
| **Currency** | Trading currency |
| **Symbol** | Trading symbol/ticker |
| **Description** | Instrument description |
| **Sector** | Market sector (if applicable) |
| **Quantity Bought** | Number of units purchased |
| **Average Price Bought** | Average purchase price |
| **Proceeds Bought** | Total cost in original currency |
| **Proceeds Bought in Base** | Total cost in base currency (EUR) |
| **Quantity Sold** | Number of units sold |
| **Average Price Sold** | Average selling price |
| **Proceeds Sold** | Total proceeds in original currency |
| **Proceeds Sold in Base** | Total proceeds in base currency (EUR) |

### Notes for IBKR Files:

- All files follow a consistent format with Section and Type columns
- Dates may be in different formats (MM/DD/YY or YYYYMMDD) depending on the section
- All monetary values are in the account's base currency (EUR) unless specified otherwise
- Files are generated only if there is relevant activity in that category
- The trade summary includes both Forex and stock transactions with their respective sectors

