"""
Flow utilities for the GIPS-Compliant Returns Calculator.

Provides unified handling of cash flow columns across different brokerages.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Tuple
from datetime import datetime


# Default flow column names in order of preference
DEFAULT_FLOW_COLUMNS = ['Adjusted EUR', 'EUR equivalent']


def get_flow_column(df: pd.DataFrame, flow_columns: Optional[List[str]] = None) -> str:
    """
    Detect and return the appropriate flow column name from a DataFrame.
    
    Args:
        df: DataFrame to check for flow columns
        flow_columns: List of column names to check, in order of preference.
                     Defaults to ['Adjusted EUR', 'EUR equivalent']
    
    Returns:
        Name of the first matching flow column found
        
    Raises:
        ValueError: If no recognized flow column is found
    """
    if flow_columns is None:
        flow_columns = DEFAULT_FLOW_COLUMNS
    
    for col in flow_columns:
        if col in df.columns:
            return col
    
    # Try case-insensitive match as fallback
    df_cols_lower = {c.lower(): c for c in df.columns}
    for col in flow_columns:
        if col.lower() in df_cols_lower:
            return df_cols_lower[col.lower()]
    
    raise ValueError(
        f"No recognized flow column found. "
        f"Expected one of {flow_columns}, got {list(df.columns)}"
    )


def get_flows_for_period(
    trades_df: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    flow_columns: Optional[List[str]] = None,
    date_column: str = 'When',
    inclusive: str = 'right'
) -> pd.DataFrame:
    """
    Extract flows for a specific period with consistent logic.
    
    Args:
        trades_df: DataFrame containing trade/flow data
        start_date: Start of the period (exclusive by default)
        end_date: End of the period (inclusive by default)
        flow_columns: List of column names to check for flow values
        date_column: Name of the date column in trades_df
        inclusive: Which bounds to include - 'right' (default), 'left', 'both', 'neither'
    
    Returns:
        DataFrame of flows within the specified period
    """
    if trades_df.empty:
        return trades_df
    
    # Ensure dates are normalized
    df = trades_df.copy()
    df[date_column] = pd.to_datetime(df[date_column]).dt.normalize()
    start_date = pd.to_datetime(start_date).normalize()
    end_date = pd.to_datetime(end_date).normalize()
    
    # Filter based on inclusive parameter
    if inclusive == 'right':
        mask = (df[date_column] > start_date) & (df[date_column] <= end_date)
    elif inclusive == 'left':
        mask = (df[date_column] >= start_date) & (df[date_column] < end_date)
    elif inclusive == 'both':
        mask = (df[date_column] >= start_date) & (df[date_column] <= end_date)
    else:  # neither
        mask = (df[date_column] > start_date) & (df[date_column] < end_date)
    
    return df[mask]


def get_total_flow_for_period(
    trades_df: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    flow_columns: Optional[List[str]] = None,
    date_column: str = 'When'
) -> float:
    """
    Calculate total flow amount for a specific period.
    
    Args:
        trades_df: DataFrame containing trade/flow data
        start_date: Start of the period
        end_date: End of the period
        flow_columns: List of column names to check for flow values
        date_column: Name of the date column
    
    Returns:
        Total flow amount for the period
    """
    period_flows = get_flows_for_period(
        trades_df, start_date, end_date, flow_columns, date_column
    )
    
    if period_flows.empty:
        return 0.0
    
    try:
        flow_col = get_flow_column(period_flows, flow_columns)
        return float(period_flows[flow_col].sum())
    except ValueError:
        return 0.0


def get_flows_by_day(
    trades_df: pd.DataFrame,
    flow_columns: Optional[List[str]] = None,
    date_column: str = 'When'
) -> dict:
    """
    Aggregate flows by day.
    
    Args:
        trades_df: DataFrame containing trade/flow data
        flow_columns: List of column names to check for flow values
        date_column: Name of the date column
    
    Returns:
        Dictionary mapping dates to total flow amounts
    """
    if trades_df.empty:
        return {}
    
    try:
        flow_col = get_flow_column(trades_df, flow_columns)
    except ValueError:
        return {}
    
    df = trades_df.copy()
    df[date_column] = pd.to_datetime(df[date_column]).dt.normalize()
    
    return df.groupby(date_column)[flow_col].sum().to_dict()


def is_large_cash_flow(
    flow_amount: float,
    portfolio_nav: float,
    threshold: float = 0.10
) -> bool:
    """
    Check if a cash flow exceeds the threshold as percentage of NAV.
    
    According to GIPS, large external cash flows (typically >10% of portfolio)
    may require special handling or temporary removal from composite.
    
    Args:
        flow_amount: The cash flow amount (positive or negative)
        portfolio_nav: The portfolio NAV at the time of flow
        threshold: Threshold as decimal (default 0.10 = 10%)
    
    Returns:
        True if the flow exceeds the threshold
    """
    if portfolio_nav <= 0:
        return False
    return abs(flow_amount) / portfolio_nav > threshold


def flag_large_cash_flows(
    trades_df: pd.DataFrame,
    nav_df: pd.DataFrame,
    threshold: float = 0.10,
    flow_columns: Optional[List[str]] = None,
    date_column: str = 'When',
    nav_date_column: str = 'Date',
    nav_value_column: str = 'Net Asset Value'
) -> pd.DataFrame:
    """
    Flag large cash flows in a trades DataFrame.
    
    Adds a 'large_flow_flag' column indicating if each flow exceeds
    the threshold as a percentage of the portfolio NAV.
    
    Args:
        trades_df: DataFrame containing trade/flow data
        nav_df: DataFrame containing NAV data
        threshold: Threshold as decimal (default 0.10 = 10%)
        flow_columns: List of column names to check for flow values
        date_column: Name of the date column in trades_df
        nav_date_column: Name of the date column in nav_df
        nav_value_column: Name of the NAV column in nav_df
    
    Returns:
        trades_df with 'large_flow_flag' and 'flow_pct_of_nav' columns added
    """
    if trades_df.empty:
        result = trades_df.copy()
        result['large_flow_flag'] = pd.Series(dtype=bool)
        result['flow_pct_of_nav'] = pd.Series(dtype=float)
        return result
    
    try:
        flow_col = get_flow_column(trades_df, flow_columns)
    except ValueError:
        result = trades_df.copy()
        result['large_flow_flag'] = False
        result['flow_pct_of_nav'] = 0.0
        return result
    
    result = trades_df.copy()
    result[date_column] = pd.to_datetime(result[date_column]).dt.normalize()
    
    # Prepare NAV lookup
    nav = nav_df.copy()
    nav[nav_date_column] = pd.to_datetime(nav[nav_date_column]).dt.normalize()
    nav = nav.sort_values(nav_date_column)
    nav_lookup = nav.set_index(nav_date_column)[nav_value_column].to_dict()
    
    def get_nav_for_date(flow_date):
        """Get NAV for a date, using most recent prior NAV if exact match not found."""
        if flow_date in nav_lookup:
            return nav_lookup[flow_date]
        # Find most recent prior NAV
        prior_dates = [d for d in nav_lookup.keys() if d <= flow_date]
        if prior_dates:
            return nav_lookup[max(prior_dates)]
        return None
    
    # Calculate flow percentage and flag
    def calc_flow_pct(row):
        nav_value = get_nav_for_date(row[date_column])
        if nav_value and nav_value > 0:
            return abs(row[flow_col]) / nav_value
        return 0.0
    
    result['flow_pct_of_nav'] = result.apply(calc_flow_pct, axis=1)
    result['large_flow_flag'] = result['flow_pct_of_nav'] > threshold
    
    # Log any large flows found
    large_flows = result[result['large_flow_flag']]
    if len(large_flows) > 0:
        logging.warning(
            f"Found {len(large_flows)} large cash flows (>{threshold*100:.0f}% of NAV)"
        )
        for _, row in large_flows.iterrows():
            logging.warning(
                f"  {row[date_column].strftime('%Y-%m-%d')}: "
                f"{row[flow_col]:,.2f} ({row['flow_pct_of_nav']*100:.1f}% of NAV)"
            )
    
    return result


def normalize_flow_dataframe(
    trades_df: pd.DataFrame,
    date_column: str = 'When',
    flow_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Normalize a flow DataFrame to have consistent column names and types.
    
    Args:
        trades_df: DataFrame containing trade/flow data
        date_column: Name of the date column
        flow_columns: List of column names to check for flow values
    
    Returns:
        DataFrame with normalized 'date' and 'flow' columns
    """
    if trades_df.empty:
        return pd.DataFrame(columns=['date', 'flow'])
    
    result = trades_df.copy()
    
    # Normalize date column
    result['date'] = pd.to_datetime(result[date_column]).dt.normalize()
    
    # Normalize flow column
    try:
        flow_col = get_flow_column(result, flow_columns)
        result['flow'] = pd.to_numeric(result[flow_col], errors='coerce').fillna(0.0)
    except ValueError:
        result['flow'] = 0.0
    
    return result[['date', 'flow']]

