"""
Fee calculator for the GIPS-Compliant Returns Calculator.

Handles gross-to-net return calculations including:
- Management fees (0.25% of start NAV per quarter = 1% annual)
- Performance fees with carry-forward shortfall hurdle mechanism

Fee Calculation Methodology:
- Management fee: 0.25% of start-of-quarter NAV, deducted quarterly
- Performance fee: 25% of gains above 6% hurdle, calculated annually
- All fees are deducted from NAV, then net return is calculated
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass, field


def annual_to_monthly_fee(annual_rate: float) -> float:
    """
    Convert annual fee rate to monthly equivalent using geometric conversion.
    
    Formula: monthly_rate = (1 + annual_rate)^(1/12) - 1
    
    This ensures that compounding monthly gives the same result as annual.
    Example: 1% annual = 0.0829% monthly (not 0.0833%)
    
    Args:
        annual_rate: Annual fee rate as decimal (e.g., 0.01 for 1%)
    
    Returns:
        Monthly fee rate as decimal
    """
    return (1 + annual_rate) ** (1/12) - 1


def quarterly_fee_to_annual_impact(quarterly_rate: float, num_quarters: int = 4) -> float:
    """
    Calculate the total annual impact of quarterly fees on NAV.
    
    When fees are deducted quarterly from NAV, the impact compounds.
    For a 0.25% quarterly fee applied 4 times, the total is slightly
    less than 1% due to the reduced base after each deduction.
    
    Args:
        quarterly_rate: Fee rate per quarter as decimal (e.g., 0.0025 for 0.25%)
        num_quarters: Number of quarters to calculate for
    
    Returns:
        Total fee impact as decimal (e.g., 0.00997 for ~1%)
    """
    # Each quarter, NAV is reduced by the fee
    # NAV_after = NAV_before * (1 - quarterly_rate)
    # After n quarters: NAV_final = NAV_start * (1 - quarterly_rate)^n
    # Total fee impact = 1 - (1 - quarterly_rate)^n
    return 1 - (1 - quarterly_rate) ** num_quarters


def calculate_net_return_from_nav(
    gross_return: float,
    quarterly_mgmt_fee: float = 0.0025,
    performance_fee_rate: float = 0.25,
    hurdle_rate: float = 0.06,
    num_quarters: int = 4
) -> Dict[str, float]:
    """
    Calculate net return by applying fees to NAV.
    
    This is the correct method for fee calculation:
    1. Start with NAV = 100
    2. Apply gross return: NAV = 100 * (1 + gross_return)
    3. Deduct management fee (quarterly, applied to each quarter's start NAV)
    4. Deduct performance fee (on gains above hurdle)
    5. Calculate net return from final NAV
    
    Args:
        gross_return: Gross return as decimal (e.g., 1.88 for 188%)
        quarterly_mgmt_fee: Quarterly management fee rate (default 0.25%)
        performance_fee_rate: Performance fee rate on excess gains (default 25%)
        hurdle_rate: Annual hurdle rate for performance fee (default 6%)
        num_quarters: Number of quarters in the period
    
    Returns:
        Dictionary with fee breakdown:
        - net_return: Final net return
        - mgmt_fee_impact: Management fee as % of starting NAV
        - perf_fee_amount: Performance fee as % of starting NAV
        - gross_return: Original gross return
    """
    start_nav = 100.0  # Use 100 as base for percentage calculations
    
    # End NAV before any fees
    end_nav_gross = start_nav * (1 + gross_return)
    gain = end_nav_gross - start_nav
    
    # Management fee: 0.25% of start NAV per quarter
    # For simplicity, we apply it as a simple annual amount (not compounded quarterly)
    # Total annual impact = quarterly_fee * 4 = 1%
    mgmt_fee_amount = start_nav * quarterly_mgmt_fee * num_quarters
    
    # Performance fee: 25% of gains above hurdle
    gain_above_hurdle = max(0, gain - (start_nav * hurdle_rate))
    perf_fee_amount = gain_above_hurdle * performance_fee_rate
    
    # Final NAV after all fees
    end_nav_net = end_nav_gross - mgmt_fee_amount - perf_fee_amount
    
    # Net return
    net_return = (end_nav_net - start_nav) / start_nav
    
    return {
        'net_return': net_return,
        'mgmt_fee_impact': mgmt_fee_amount / start_nav,
        'perf_fee_amount': perf_fee_amount / start_nav,
        'gross_return': gross_return
    }


def calculate_net_return(gross_return: float, monthly_mgmt_fee: float) -> float:
    """
    Calculate net return after deducting management fee.
    
    DEPRECATED: Use calculate_net_return_from_nav for accurate fee calculation.
    
    Net return = Gross return - Management fee
    
    Note: This is a simple subtraction. For more precise accounting,
    the fee could be deducted from NAV before calculating returns.
    
    Args:
        gross_return: Gross return as decimal (e.g., 0.02 for 2%)
        monthly_mgmt_fee: Monthly management fee as decimal
    
    Returns:
        Net return as decimal
    """
    return gross_return - monthly_mgmt_fee


def calculate_monthly_net_returns(
    monthly_gross_returns: pd.DataFrame,
    annual_mgmt_fee: float = 0.01,
    return_column: str = 'return'
) -> pd.DataFrame:
    """
    Calculate monthly net returns from gross returns.
    
    Args:
        monthly_gross_returns: DataFrame with monthly returns
        annual_mgmt_fee: Annual management fee rate (default 1%)
        return_column: Name of the return column
    
    Returns:
        DataFrame with both gross and net returns, plus fee breakdown
    """
    result = monthly_gross_returns.copy()
    
    monthly_fee = annual_to_monthly_fee(annual_mgmt_fee)
    
    result['gross_return'] = result[return_column]
    result['management_fee'] = monthly_fee
    result['net_return'] = result['gross_return'] - monthly_fee
    
    return result


@dataclass
class PerformanceFeeCalculator:
    """
    Performance fee calculator with carry-forward shortfall hurdle mechanism.
    
    Rules:
    - Base hurdle rate: 6% annual (configurable)
    - Performance fee: 25% of gains above hurdle (configurable)
    - If year's return < hurdle, the shortfall carries forward to next year
    - Shortfall accumulates but does NOT compound
    - Once hurdle is exceeded, shortfall resets to zero
    
    Example scenario:
    - Year 1: Return = 4%, Hurdle = 6% → Shortfall = 2%, No fee
    - Year 2: Return = 8%, Carried Hurdle = 6% + 2% = 8% → No fee (exactly met)
    - Year 3: Return = 10%, Hurdle = 6% → Excess = 4%, Fee = 25% * 4% = 1%
    
    Attributes:
        hurdle_rate: Annual hurdle rate (default 6%)
        fee_rate: Fee rate on gains above hurdle (default 25%)
        carried_shortfall: Accumulated shortfall from prior years
        history: List of annual fee calculations for audit trail
    """
    hurdle_rate: float = 0.06
    fee_rate: float = 0.25
    carried_shortfall: float = 0.0
    history: List[Dict] = field(default_factory=list)
    
    def calculate_annual_fee(self, annual_return: float, year: Optional[int] = None) -> Tuple[float, float]:
        """
        Calculate performance fee for a year.
        
        Args:
            annual_return: The year's total return as decimal (e.g., 0.08 for 8%)
            year: Optional year label for audit trail
        
        Returns:
            Tuple of (fee_amount, new_carried_shortfall)
            - fee_amount: Performance fee to charge (as decimal)
            - new_carried_shortfall: Updated shortfall to carry forward
        """
        # Calculate effective hurdle (base + any carried shortfall)
        effective_hurdle = self.hurdle_rate + self.carried_shortfall
        
        # Calculate excess over effective hurdle
        excess = annual_return - effective_hurdle
        
        # Record for audit trail
        record = {
            'year': year,
            'annual_return': annual_return,
            'base_hurdle': self.hurdle_rate,
            'carried_shortfall_in': self.carried_shortfall,
            'effective_hurdle': effective_hurdle,
            'excess': excess,
        }
        
        if excess > 0:
            # Exceeded hurdle - charge fee and reset shortfall
            fee = excess * self.fee_rate
            self.carried_shortfall = 0.0
            record['fee'] = fee
            record['carried_shortfall_out'] = 0.0
        else:
            # Below hurdle - no fee, calculate shortfall
            fee = 0.0
            # Shortfall is how much we missed the BASE hurdle by (not effective)
            # Only accumulate if we're below the base hurdle
            if annual_return < self.hurdle_rate:
                shortfall_this_year = self.hurdle_rate - annual_return
                self.carried_shortfall += shortfall_this_year
            # If we're between effective and base hurdle, we just didn't clear
            # the carried shortfall but don't add more
            record['fee'] = 0.0
            record['carried_shortfall_out'] = self.carried_shortfall
        
        self.history.append(record)
        
        return fee, self.carried_shortfall
    
    def reset(self):
        """Reset the calculator state."""
        self.carried_shortfall = 0.0
        self.history = []
    
    def get_history_df(self) -> pd.DataFrame:
        """Get calculation history as DataFrame for audit purposes."""
        if not self.history:
            return pd.DataFrame(columns=[
                'year', 'annual_return', 'base_hurdle', 'carried_shortfall_in',
                'effective_hurdle', 'excess', 'fee', 'carried_shortfall_out'
            ])
        return pd.DataFrame(self.history)


@dataclass
class QuarterlyFeeTracker:
    """
    Tracks NAV with proper quarterly fee deductions.
    
    Fee Timing Rules:
    1. Management fee: Deducted on 1st day of each fiscal quarter (0.25% of NAV)
    2. Performance fee: Deducted on last day of each fiscal quarter
       - Hurdle: 6% annual = 1.5% per quarter (accumulated)
       - Fee: 25% of gains above accumulated hurdle
    
    For February fiscal year (Feb 1 - Jan 31):
    - Q1: Feb 1 (mgmt fee) → Apr 30 (perf fee)
    - Q2: May 1 (mgmt fee) → Jul 31 (perf fee)
    - Q3: Aug 1 (mgmt fee) → Oct 31 (perf fee)
    - Q4: Nov 1 (mgmt fee) → Jan 31 (perf fee)
    
    Attributes:
        initial_nav: Starting NAV (use actual consolidated NAV)
        quarterly_mgmt_fee: Management fee rate per quarter (default 0.25%)
        annual_hurdle_rate: Annual hurdle rate (default 6%)
        quarterly_hurdle: Hurdle per quarter (1.5%)
        perf_fee_rate: Performance fee rate (default 25%)
        fiscal_year_start_month: Month when fiscal year starts (default 2 = February)
    """
    initial_nav: float
    quarterly_mgmt_fee: float = 0.0025  # 0.25% per quarter
    annual_hurdle_rate: float = 0.06
    perf_fee_rate: float = 0.25
    fiscal_year_start_month: int = 2  # February
    
    def __post_init__(self):
        self.nav = self.initial_nav
        self.quarterly_hurdle = self.annual_hurdle_rate / 4  # 1.5% per quarter
        self.accumulated_hurdle = 0.0  # Accumulates over quarters within a year
        self.carried_shortfall = 0.0  # Carries between years
        self.quarter_start_nav = self.initial_nav  # NAV at start of current quarter
        self.year_start_nav = self.initial_nav  # NAV at start of fiscal year
        self.history: List[Dict] = []
        self.current_quarter = 1
        self.current_fiscal_year = None
        
        # Calculate quarter end months based on fiscal year start
        self._calculate_quarter_boundaries()
    
    def _calculate_quarter_boundaries(self):
        """
        Calculate which months are quarter starts and ends.
        
        Quarter boundaries are calculated relative to fiscal year start:
        - Q1: fiscal_year_start_month to fiscal_year_start_month + 2
        - Q2: fiscal_year_start_month + 3 to fiscal_year_start_month + 5
        - Q3: fiscal_year_start_month + 6 to fiscal_year_start_month + 8
        - Q4: fiscal_year_start_month + 9 to fiscal_year_start_month + 11
        
        All months are normalized to 1-12 range using modular arithmetic.
        
        Examples:
        - January fiscal year (start=1): Q1=Jan-Mar, Q2=Apr-Jun, Q3=Jul-Sep, Q4=Oct-Dec
        - February fiscal year (start=2): Q1=Feb-Apr, Q2=May-Jul, Q3=Aug-Oct, Q4=Nov-Jan
        - April fiscal year (start=4): Q1=Apr-Jun, Q2=Jul-Sep, Q3=Oct-Dec, Q4=Jan-Mar
        """
        start = self.fiscal_year_start_month
        
        # Helper to normalize month to 1-12 range
        def normalize_month(m: int) -> int:
            return ((m - 1) % 12) + 1
        
        # Quarter start months (1st of these months = mgmt fee day)
        # Q1 starts at fiscal_year_start_month, Q2 at +3, Q3 at +6, Q4 at +9
        self.quarter_start_months = [
            normalize_month(start),
            normalize_month(start + 3),
            normalize_month(start + 6),
            normalize_month(start + 9)
        ]
        
        # Quarter end months (last day = perf fee day)
        # Q1 ends at start+2, Q2 at start+5, Q3 at start+8, Q4 at start+11
        self.quarter_end_months = [
            normalize_month(start + 2),
            normalize_month(start + 5),
            normalize_month(start + 8),
            normalize_month(start + 11)
        ]
    
    def get_quarter_for_month(self, month: int) -> int:
        """
        Determine which fiscal quarter a calendar month belongs to.
        
        Args:
            month: Calendar month (1-12)
        
        Returns:
            Fiscal quarter (1-4)
        """
        # Calculate months since fiscal year start
        months_since_start = (month - self.fiscal_year_start_month) % 12
        
        # Each quarter is 3 months
        return (months_since_start // 3) + 1
    
    def is_quarter_start(self, month: int) -> bool:
        """Check if month is the start of a fiscal quarter."""
        return month in self.quarter_start_months
    
    def is_quarter_end(self, month: int) -> bool:
        """Check if month is the end of a fiscal quarter."""
        return month in self.quarter_end_months
    
    def is_fiscal_year_start(self, month: int) -> bool:
        """Check if month is the start of fiscal year."""
        return month == self.fiscal_year_start_month
    
    def apply_management_fee(self, quarter: int, fiscal_year: str) -> Dict:
        """
        Apply management fee at start of quarter (0.25% of current NAV).
        
        Returns:
            Dictionary with fee details
        """
        nav_before = self.nav
        fee_amount = self.nav * self.quarterly_mgmt_fee
        self.nav = self.nav - fee_amount
        self.quarter_start_nav = self.nav  # This is NAV for calculating quarter return
        
        record = {
            'event': 'management_fee',
            'quarter': quarter,
            'fiscal_year': fiscal_year,
            'nav_before': nav_before,
            'fee_rate': self.quarterly_mgmt_fee,
            'fee_amount': fee_amount,
            'nav_after': self.nav
        }
        self.history.append(record)
        return record
    
    def apply_month_return(self, month_return: float, month: int, year: int) -> Dict:
        """
        Apply a month's gross return to NAV.
        
        Args:
            month_return: Gross return for the month as decimal
            month: Calendar month (1-12)
            year: Calendar year
        
        Returns:
            Dictionary with return details
        """
        nav_before = self.nav
        self.nav = self.nav * (1 + month_return)
        
        record = {
            'event': 'monthly_return',
            'month': month,
            'calendar_year': year,
            'nav_before': nav_before,
            'return': month_return,
            'nav_after': self.nav
        }
        self.history.append(record)
        return record
    
    def apply_performance_fee(self, quarter: int, fiscal_year: str) -> Dict:
        """
        Apply performance fee at end of quarter.
        
        Performance fee is 25% of gains above the accumulated quarterly hurdle.
        Hurdle = 1.5% per quarter, accumulated (so Q2 hurdle = 3%, Q3 = 4.5%, Q4 = 6%)
        
        Args:
            quarter: Fiscal quarter (1-4)
            fiscal_year: Fiscal year label
        
        Returns:
            Dictionary with performance fee details
        """
        nav_before = self.nav
        
        # Accumulate hurdle for this quarter
        self.accumulated_hurdle += self.quarterly_hurdle
        
        # Add any carried shortfall from prior years (only applies at Q1)
        effective_hurdle = self.accumulated_hurdle + (self.carried_shortfall if quarter == 1 else 0)
        
        # Calculate gain since year start
        gain_since_year_start = (self.nav - self.year_start_nav) / self.year_start_nav if self.year_start_nav > 0 else 0
        
        # Calculate hurdle amount in NAV terms
        hurdle_amount = self.year_start_nav * effective_hurdle
        actual_gain = self.nav - self.year_start_nav
        excess_gain = max(0, actual_gain - hurdle_amount)
        
        # Performance fee on excess
        perf_fee = excess_gain * self.perf_fee_rate
        self.nav = self.nav - perf_fee
        
        # Track for shortfall (only at year end, Q4)
        if quarter == 4:
            annual_return = (self.nav - self.year_start_nav) / self.year_start_nav if self.year_start_nav > 0 else 0
            if annual_return < self.annual_hurdle_rate:
                shortfall_this_year = self.annual_hurdle_rate - annual_return
                self.carried_shortfall += shortfall_this_year
            else:
                self.carried_shortfall = 0.0  # Reset if hurdle exceeded
            # Reset accumulated hurdle for new year
            self.accumulated_hurdle = 0.0
        
        record = {
            'event': 'performance_fee',
            'quarter': quarter,
            'fiscal_year': fiscal_year,
            'year_start_nav': self.year_start_nav,
            'nav_before': nav_before,
            'gain_since_year_start': gain_since_year_start,
            'accumulated_hurdle': effective_hurdle,
            'hurdle_amount': hurdle_amount,
            'actual_gain': actual_gain,
            'excess_gain': excess_gain,
            'fee_rate': self.perf_fee_rate,
            'fee_amount': perf_fee,
            'nav_after': self.nav,
            'carried_shortfall': self.carried_shortfall
        }
        self.history.append(record)
        return record
    
    def start_new_fiscal_year(self, fiscal_year: str):
        """Mark the start of a new fiscal year."""
        self.year_start_nav = self.nav
        self.current_fiscal_year = fiscal_year
        self.current_quarter = 1
        self.accumulated_hurdle = 0.0
    
    def process_monthly_returns(
        self,
        monthly_returns: pd.DataFrame,
        month_column: str = 'month',
        return_column: str = 'return'
    ) -> Dict:
        """
        Process a DataFrame of monthly returns with proper fee timing.
        
        Args:
            monthly_returns: DataFrame with columns for month and return
            month_column: Name of the month column (should be datetime or Period)
            return_column: Name of the return column
        
        Returns:
            Dictionary with:
            - final_nav: NAV after all fees
            - total_gross_return: Gross return ignoring fees
            - total_net_return: Net return after all fees
            - total_mgmt_fees: Total management fees paid
            - total_perf_fees: Total performance fees paid
            - history: List of all events
        """
        if monthly_returns.empty:
            return {
                'final_nav': self.initial_nav,
                'total_gross_return': 0.0,
                'total_net_return': 0.0,
                'total_mgmt_fees': 0.0,
                'total_perf_fees': 0.0,
                'history': []
            }
        
        # Sort by month
        df = monthly_returns.copy()
        df = df.sort_values(month_column)
        
        # Track gross NAV (without fees) for comparison
        gross_nav = self.initial_nav
        
        for _, row in df.iterrows():
            month_period = row[month_column]
            month_return = row[return_column]
            
            # Extract month and year
            if hasattr(month_period, 'month'):
                month = month_period.month
                year = month_period.year if hasattr(month_period, 'year') else 2024
            else:
                # Try to parse as period
                try:
                    period = pd.Period(month_period)
                    month = period.month
                    year = period.year
                except:
                    continue
            
            # Determine fiscal year
            if month >= self.fiscal_year_start_month:
                fiscal_year = str(year)
            else:
                fiscal_year = str(year - 1)
            
            # Check if this is start of fiscal year
            if month == self.fiscal_year_start_month:
                self.start_new_fiscal_year(fiscal_year)
            
            # At quarter start: apply management fee
            if self.is_quarter_start(month):
                quarter = self.get_quarter_for_month(month)
                self.apply_management_fee(quarter, fiscal_year)
            
            # Apply the month's return
            self.apply_month_return(month_return, month, year)
            gross_nav = gross_nav * (1 + month_return)
            
            # At quarter end: apply performance fee
            if self.is_quarter_end(month):
                quarter = self.get_quarter_for_month(month)
                self.apply_performance_fee(quarter, fiscal_year)
        
        # Calculate totals
        total_mgmt_fees = sum(h.get('fee_amount', 0) for h in self.history if h['event'] == 'management_fee')
        total_perf_fees = sum(h.get('fee_amount', 0) for h in self.history if h['event'] == 'performance_fee')
        
        total_gross_return = (gross_nav - self.initial_nav) / self.initial_nav
        total_net_return = (self.nav - self.initial_nav) / self.initial_nav
        
        return {
            'final_nav': self.nav,
            'final_gross_nav': gross_nav,
            'total_gross_return': total_gross_return,
            'total_net_return': total_net_return,
            'total_mgmt_fees': total_mgmt_fees,
            'total_perf_fees': total_perf_fees,
            'total_fees': total_mgmt_fees + total_perf_fees,
            'history': self.history
        }
    
    def process_monthly_with_actual_nav(
        self,
        monthly_data: pd.DataFrame,
        month_column: str = 'month',
        return_column: str = 'return',
        nav_column: str = 'end_nav',
        start_nav_column: str = 'start_nav',
        flow_column: str = 'flow'
    ) -> Dict:
        """
        Process monthly data using ACTUAL NAV values (not computed from returns).
        
        This method tracks a theoretical "Net NAV" that starts at the actual NAV
        but has fees deducted. The Gross NAV is the actual NAV from the data.
        
        Args:
            monthly_data: DataFrame with NAV and flow data
            month_column: Name of the month column
            return_column: Name of the return column
            nav_column: Name of the end-of-month NAV column
            start_nav_column: Name of the start-of-month NAV column
            flow_column: Name of the flow column
        
        Returns:
            Dictionary with actual NAV tracking and fee deductions
        """
        if monthly_data.empty:
            return {
                'final_gross_nav': self.initial_nav,
                'final_net_nav': self.initial_nav,
                'total_mgmt_fees': 0.0,
                'total_perf_fees': 0.0,
                'total_flows': 0.0,
                'history': []
            }
        
        df = monthly_data.copy()
        df = df.sort_values(month_column)
        
        # Initialize tracking
        # net_nav starts equal to gross_nav but will diverge as fees are deducted
        first_row = df.iloc[0]
        initial_gross_nav = first_row.get(start_nav_column, first_row.get(nav_column, self.initial_nav))
        if pd.isna(initial_gross_nav) or initial_gross_nav <= 0:
            initial_gross_nav = self.initial_nav
        
        # Track the ratio of net to gross NAV (starts at 1.0, decreases with fees)
        net_to_gross_ratio = 1.0
        total_flows = 0.0
        
        for _, row in df.iterrows():
            month_period = row[month_column]
            end_nav = row.get(nav_column, 0)
            start_nav = row.get(start_nav_column, end_nav)
            flow = row.get(flow_column, 0) if flow_column in row else 0
            month_return = row.get(return_column, 0)
            
            if pd.isna(end_nav):
                end_nav = 0
            if pd.isna(start_nav):
                start_nav = end_nav
            if pd.isna(flow):
                flow = 0
            
            total_flows += flow
            
            # Extract month and year
            if hasattr(month_period, 'month'):
                month = month_period.month
                year = month_period.year if hasattr(month_period, 'year') else 2024
            else:
                try:
                    period = pd.Period(month_period)
                    month = period.month
                    year = period.year
                except:
                    continue
            
            # Determine fiscal year
            if month >= self.fiscal_year_start_month:
                fiscal_year = str(year)
            else:
                fiscal_year = str(year - 1)
            
            # Check if this is start of fiscal year
            if month == self.fiscal_year_start_month:
                self.start_new_fiscal_year(fiscal_year)
            
            # Current gross NAV for this month
            gross_nav_start = start_nav if start_nav > 0 else self.nav
            gross_nav_end = end_nav if end_nav > 0 else gross_nav_start
            
            # Calculate net NAV (gross NAV * net_to_gross_ratio)
            net_nav_start = gross_nav_start * net_to_gross_ratio
            self.nav = net_nav_start  # Update tracker's nav
            
            # At quarter start: apply management fee
            if self.is_quarter_start(month):
                quarter = self.get_quarter_for_month(month)
                # Management fee reduces net NAV
                fee_result = self.apply_management_fee(quarter, fiscal_year)
                # Update ratio based on fee
                if net_nav_start > 0:
                    net_to_gross_ratio *= (1 - self.quarterly_mgmt_fee)
            
            # Apply the month's return (net NAV grows by same % as gross)
            self.nav = self.nav * (1 + month_return) if month_return else self.nav
            
            # Record the actual NAV values
            self.history.append({
                'event': 'monthly_nav',
                'month': month,
                'calendar_year': year,
                'fiscal_year': fiscal_year,
                'gross_nav_start': gross_nav_start,
                'gross_nav_end': gross_nav_end,
                'net_nav': self.nav,
                'flow': flow,
                'return': month_return,
                'net_to_gross_ratio': net_to_gross_ratio
            })
            
            # At quarter end: apply performance fee
            if self.is_quarter_end(month):
                quarter = self.get_quarter_for_month(month)
                perf_result = self.apply_performance_fee(quarter, fiscal_year)
                # Update ratio based on performance fee
                if gross_nav_end > 0 and perf_result.get('fee_amount', 0) > 0:
                    # Recalculate ratio
                    net_to_gross_ratio = self.nav / gross_nav_end if gross_nav_end > 0 else net_to_gross_ratio
        
        # Get final values
        final_gross_nav = df.iloc[-1].get(nav_column, 0) if not df.empty else 0
        if pd.isna(final_gross_nav):
            final_gross_nav = 0
        
        total_mgmt_fees = sum(h.get('fee_amount', 0) for h in self.history if h.get('event') == 'management_fee')
        total_perf_fees = sum(h.get('fee_amount', 0) for h in self.history if h.get('event') == 'performance_fee')
        
        return {
            'initial_gross_nav': initial_gross_nav,
            'final_gross_nav': final_gross_nav,
            'final_net_nav': self.nav,
            'total_mgmt_fees': total_mgmt_fees,
            'total_perf_fees': total_perf_fees,
            'total_fees': total_mgmt_fees + total_perf_fees,
            'total_flows': total_flows,
            'net_to_gross_ratio': net_to_gross_ratio,
            'history': self.history
        }
    
    def get_history_df(self) -> pd.DataFrame:
        """Get event history as DataFrame."""
        if not self.history:
            return pd.DataFrame()
        return pd.DataFrame(self.history)
    
    def reset(self):
        """Reset tracker to initial state."""
        self.nav = self.initial_nav
        self.accumulated_hurdle = 0.0
        self.carried_shortfall = 0.0
        self.quarter_start_nav = self.initial_nav
        self.year_start_nav = self.initial_nav
        self.history = []
        self.current_quarter = 1
        self.current_fiscal_year = None


@dataclass
class NetNAVTracker:
    """
    Tracks Net NAV (after fees) alongside actual Gross NAV, properly handling flows.
    
    This tracker maintains two parallel NAV series:
    1. Gross NAV: The actual NAV from accounts (with flows)
    2. Net NAV: Gross NAV minus accumulated fee deductions
    
    Flows (deposits/withdrawals) affect both equally.
    Fees are deducted only from Net NAV at quarter boundaries.
    
    Fee Timing:
    - Management fee: 0.25% of Net NAV at START of each quarter
    - Performance fee: 25% of gains above hurdle at END of each quarter
    
    Attributes:
        initial_gross_nav: Starting combined NAV from all accounts
        quarterly_mgmt_fee: Management fee rate per quarter (default 0.25%)
        annual_hurdle_rate: Annual hurdle rate (default 6%)
        perf_fee_rate: Performance fee rate (default 25%)
        fiscal_year_start_month: Month when fiscal year starts (default 2 = February)
    """
    initial_gross_nav: float
    quarterly_mgmt_fee: float = 0.0025
    annual_hurdle_rate: float = 0.06
    perf_fee_rate: float = 0.25
    fiscal_year_start_month: int = 2
    
    def __post_init__(self):
        self.gross_nav = self.initial_gross_nav
        self.net_nav = self.initial_gross_nav  # Starts equal to gross
        self.accumulated_fees = 0.0  # Total fees deducted
        self.quarterly_hurdle = self.annual_hurdle_rate / 4
        
        # Track flows since year start for accurate performance fee calculation
        self.flows_since_year_start = 0.0
        self.current_quarter = 0
        self.year_start_gross_nav = self.initial_gross_nav
        self.accumulated_hurdle = 0.0
        self.carried_shortfall = 0.0
        self.year_start_net_nav = self.initial_gross_nav
        self.total_flows = 0.0
        self.history: List[Dict] = []
        self.quarterly_events: List[Dict] = []
        
        # Calculate quarter boundaries
        self._calculate_quarter_boundaries()
    
    def _calculate_quarter_boundaries(self):
        """Calculate which months are quarter starts and ends."""
        start = self.fiscal_year_start_month
        
        def normalize_month(m: int) -> int:
            return ((m - 1) % 12) + 1
        
        self.quarter_start_months = [
            normalize_month(start),
            normalize_month(start + 3),
            normalize_month(start + 6),
            normalize_month(start + 9)
        ]
        
        self.quarter_end_months = [
            normalize_month(start + 2),
            normalize_month(start + 5),
            normalize_month(start + 8),
            normalize_month(start + 11)
        ]
    
    def get_quarter_for_month(self, month: int) -> int:
        """Determine which fiscal quarter a calendar month belongs to."""
        months_since_start = (month - self.fiscal_year_start_month) % 12
        return (months_since_start // 3) + 1
    
    def is_quarter_start(self, month: int) -> bool:
        return month in self.quarter_start_months
    
    def is_quarter_end(self, month: int) -> bool:
        return month in self.quarter_end_months
    
    def apply_flow(self, flow_amount: float, date_info: str = ''):
        """
        Apply a cash flow (deposit or withdrawal).
        Flows affect both gross and net NAV equally.
        """
        self.gross_nav += flow_amount
        self.net_nav += flow_amount
        self.total_flows += flow_amount
        
        self.history.append({
            'event': 'flow',
            'date': date_info,
            'flow_amount': flow_amount,
            'gross_nav_after': self.gross_nav,
            'net_nav_after': self.net_nav
        })
    
    def apply_return(self, return_pct: float, period_info: str = ''):
        """
        Apply investment return for a period.
        Returns are applied to both gross and net NAV.
        """
        gross_before = self.gross_nav
        net_before = self.net_nav
        
        self.gross_nav *= (1 + return_pct)
        self.net_nav *= (1 + return_pct)
        
        self.history.append({
            'event': 'return',
            'period': period_info,
            'return_pct': return_pct,
            'gross_nav_before': gross_before,
            'gross_nav_after': self.gross_nav,
            'net_nav_before': net_before,
            'net_nav_after': self.net_nav
        })
    
    def apply_management_fee(self, quarter: int, fiscal_year: str):
        """
        Apply management fee at start of quarter.
        Fee is 0.25% of current Net NAV, added to accumulated_fees.
        """
        # Calculate fee on current net NAV (gross - accumulated fees so far)
        current_net_nav = self.gross_nav - self.accumulated_fees
        fee_amount = current_net_nav * self.quarterly_mgmt_fee
        self.accumulated_fees += fee_amount
        
        # Update net NAV
        self.net_nav = self.gross_nav - self.accumulated_fees
        
        event = {
            'event': 'management_fee',
            'quarter': quarter,
            'fiscal_year': fiscal_year,
            'gross_nav': self.gross_nav,
            'net_nav_before': current_net_nav,
            'fee_rate': self.quarterly_mgmt_fee,
            'fee_amount': fee_amount,
            'net_nav_after': self.net_nav,
            'cumulative_fees': self.accumulated_fees
        }
        self.history.append(event)
        self.quarterly_events.append(event)
        
        return fee_amount
    
    def apply_performance_fee(self, quarter: int, fiscal_year: str):
        """
        Apply performance fee at end of quarter.
        
        The hurdle accumulates quarterly:
        - Q1: 1.5% (6% / 4)
        - Q2: 3.0% (cumulative)
        - Q3: 4.5% (cumulative)
        - Q4: 6.0% (cumulative)
        
        Performance fee is 25% of (investment gain - hurdle amount).
        Investment gain = Gross NAV change - flows (pure investment performance).
        Fee is deducted from accumulated_fees which reduces net_nav.
        """
        self.current_quarter = quarter
        
        # Accumulate hurdle for the quarter (1.5% per quarter)
        self.accumulated_hurdle = quarter * self.quarterly_hurdle
        
        # Add carried shortfall from previous years
        effective_hurdle = self.accumulated_hurdle + self.carried_shortfall
        
        # Calculate INVESTMENT gain since year start using GROSS NAV (excluding flows)
        # Investment gain = Current Gross NAV - Year Start Gross NAV - Flows
        # This represents pure investment performance before fees
        year_start = getattr(self, 'year_start_gross_nav', self.initial_gross_nav)
        investment_gain = self.gross_nav - year_start - self.flows_since_year_start
        
        # Hurdle amount is based on the year start NAV (the capital base at risk)
        hurdle_amount = year_start * effective_hurdle
        
        # Excess gain is investment gain above the hurdle
        excess_gain = max(0, investment_gain - hurdle_amount)
        
        # Performance fee is 25% of excess
        fee_amount = excess_gain * self.perf_fee_rate
        self.accumulated_fees += fee_amount
        
        # Update net NAV: Gross NAV - Accumulated Fees
        self.net_nav = self.gross_nav - self.accumulated_fees
        
        # Handle year-end shortfall tracking
        if quarter == 4:
            # Calculate investment return (excluding flows)
            investment_return = investment_gain / year_start if year_start > 0 else 0
            if investment_return < self.annual_hurdle_rate:
                self.carried_shortfall += (self.annual_hurdle_rate - investment_return)
            else:
                self.carried_shortfall = 0.0
            self.accumulated_hurdle = 0.0  # Reset for new year
        
        event = {
            'event': 'performance_fee',
            'quarter': quarter,
            'fiscal_year': fiscal_year,
            'gross_nav': self.gross_nav,
            'net_nav_before': self.gross_nav - self.accumulated_fees + fee_amount,
            'year_start_gross_nav': year_start,
            'flows_since_year_start': self.flows_since_year_start,
            'investment_gain': investment_gain,
            'effective_hurdle_pct': effective_hurdle,
            'hurdle_amount': hurdle_amount,
            'excess_gain': excess_gain,
            'fee_rate': self.perf_fee_rate,
            'fee_amount': fee_amount,
            'net_nav_after': self.net_nav,
            'cumulative_fees': self.accumulated_fees,
            'carried_shortfall': self.carried_shortfall
        }
        self.history.append(event)
        self.quarterly_events.append(event)
        
        return fee_amount
    
    def start_new_fiscal_year(self, fiscal_year: str):
        """Mark the start of a new fiscal year."""
        self.year_start_gross_nav = self.gross_nav  # Track gross NAV for performance calc
        self.year_start_net_nav = self.net_nav
        self.accumulated_hurdle = 0.0
        self.flows_since_year_start = 0.0  # Reset flow tracking
        self.current_quarter = 0
        
        self.history.append({
            'event': 'fiscal_year_start',
            'fiscal_year': fiscal_year,
            'gross_nav': self.gross_nav,
            'net_nav': self.net_nav
        })
    
    def process_monthly_data(
        self,
        monthly_df: pd.DataFrame,
        month_column: str = 'month',
        return_column: str = 'return',
        start_nav_column: str = 'start_nav',
        gross_nav_column: str = 'end_nav',
        flow_column: str = 'flow'
    ) -> Dict:
        """
        Process monthly data, tracking both gross and net NAV.
        
        Net NAV starts equal to Gross NAV and diverges as fees are deducted.
        Both NAVs include flows - they affect gross and net equally.
        
        Args:
            monthly_df: DataFrame with monthly data
            month_column: Column with month (Period or datetime)
            return_column: Column with monthly return
            start_nav_column: Column with start-of-month NAV
            gross_nav_column: Column with end-of-month NAV (actual)
            flow_column: Column with monthly flows
        
        Returns:
            Dictionary with tracking results
        """
        if monthly_df.empty:
            return {
                'initial_gross_nav': self.initial_gross_nav,
                'final_gross_nav': self.gross_nav,
                'final_net_nav': self.net_nav,
                'total_fees': self.accumulated_fees,
                'total_flows': self.total_flows,
                'quarterly_events': [],
                'history': []
            }
        
        df = monthly_df.copy().sort_values(month_column)
        
        # Initialize from the first actual NAV value (skip zero/empty values)
        # Use the first END_NAV that is > 0 (after initial deposit)
        first_valid_nav = None
        first_flow = 0
        for idx, row in df.iterrows():
            start_val = row.get(start_nav_column, 0)
            end_val = row.get(gross_nav_column, 0)
            flow_val = row.get(flow_column, 0)
            
            # Handle string values (e.g., "€1,234.56")
            def parse_nav(val):
                if pd.isna(val):
                    return 0
                if isinstance(val, str):
                    try:
                        return float(val.replace('€', '').replace(',', '').strip())
                    except:
                        return 0
                return float(val)
            
            start_val = parse_nav(start_val)
            end_val = parse_nav(end_val)
            flow_val = parse_nav(flow_val)
            
            # Look for first non-zero end NAV (this is after the first deposit)
            if end_val > 0:
                first_valid_nav = end_val
                first_flow = flow_val
                break
        
        if first_valid_nav is not None and first_valid_nav > 0:
            self.initial_gross_nav = first_valid_nav
            self.gross_nav = first_valid_nav
            self.net_nav = first_valid_nav  # Net starts equal to gross
            self.year_start_net_nav = first_valid_nav
            self.year_start_gross_nav = first_valid_nav
            # The first flow contributed to the initial NAV
            self.flows_since_year_start = first_flow
        
        for idx, row in df.iterrows():
            month_period = row[month_column]
            
            # Extract month and year
            if hasattr(month_period, 'month'):
                month = month_period.month
                year = month_period.year
            else:
                try:
                    period = pd.Period(month_period)
                    month = period.month
                    year = period.year
                except:
                    continue
            
            # Determine fiscal year
            if month >= self.fiscal_year_start_month:
                fiscal_year = str(year)
            else:
                fiscal_year = str(year - 1)
            
            # Get actual values from data
            start_nav = row.get(start_nav_column, 0)
            end_nav = row.get(gross_nav_column, 0)
            flow = row.get(flow_column, 0)
            monthly_return = row.get(return_column, 0)
            
            # Handle string values (e.g., "€1,234.56" or "10.5%")
            def parse_value(val, default=0):
                if pd.isna(val):
                    return default
                if isinstance(val, str):
                    try:
                        cleaned = val.replace('€', '').replace(',', '').replace('%', '').strip()
                        return float(cleaned) / 100 if '%' in val else float(cleaned)
                    except:
                        return default
                return float(val)
            
            start_nav = parse_value(start_nav, self.gross_nav)
            end_nav = parse_value(end_nav, start_nav)
            flow = parse_value(flow, 0)
            monthly_return = parse_value(monthly_return, 0)
            
            self.total_flows += flow
            
            # Check if fiscal year start
            if month == self.fiscal_year_start_month:
                self.start_new_fiscal_year(fiscal_year)
            
            # Track flows for performance fee calculation
            self.flows_since_year_start += flow
            
            # At quarter start: apply management fee
            # Fee is calculated on current net_nav and deducted
            if self.is_quarter_start(month):
                quarter = self.get_quarter_for_month(month)
                self.apply_management_fee(quarter, fiscal_year)
            
            # Update gross NAV to actual end value from data
            self.gross_nav = end_nav
            
            # Net NAV = Gross NAV - Accumulated Fees
            # This ensures net_nav always tracks gross_nav minus fees
            self.net_nav = self.gross_nav - self.accumulated_fees
            
            # Record monthly state
            self.history.append({
                'event': 'monthly_update',
                'month': month,
                'year': year,
                'fiscal_year': fiscal_year,
                'return': monthly_return,
                'start_nav_gross': start_nav,
                'end_nav_gross': end_nav,
                'flow': flow,
                'gross_nav': self.gross_nav,
                'net_nav': self.net_nav,
                'fee_drag': self.accumulated_fees,
                'cumulative_fees': self.accumulated_fees
            })
            
            # At quarter end: apply performance fee
            if self.is_quarter_end(month):
                quarter = self.get_quarter_for_month(month)
                self.apply_performance_fee(quarter, fiscal_year)
        
        # Calculate net return properly
        # Net return should account for flows being "invested" at different points
        # For simplicity, we use: (final_net - initial - flows_cumulative) / initial
        # But more accurately, we should use the TWR calculation
        
        return {
            'initial_gross_nav': self.initial_gross_nav,
            'final_gross_nav': self.gross_nav,
            'final_net_nav': self.net_nav,
            'total_mgmt_fees': sum(e.get('fee_amount', 0) for e in self.quarterly_events if e['event'] == 'management_fee'),
            'total_perf_fees': sum(e.get('fee_amount', 0) for e in self.quarterly_events if e['event'] == 'performance_fee'),
            'total_fees': self.accumulated_fees,
            'total_flows': self.total_flows,
            'fee_drag': self.gross_nav - self.net_nav,
            'quarterly_events': self.quarterly_events,
            'history': self.history
        }
    
    def get_quarterly_summary(self) -> pd.DataFrame:
        """Get quarterly summary as DataFrame."""
        if not self.quarterly_events:
            return pd.DataFrame()
        return pd.DataFrame(self.quarterly_events)
    
    def get_history_df(self) -> pd.DataFrame:
        """Get full history as DataFrame."""
        if not self.history:
            return pd.DataFrame()
        return pd.DataFrame(self.history)


@dataclass
class FeeCalculator:
    """
    Combined fee calculator handling both management and performance fees.
    
    Attributes:
        annual_mgmt_fee: Annual management fee rate (default 1%)
        hurdle_rate: Annual hurdle rate for performance fee (default 6%)
        performance_fee_rate: Performance fee rate on excess gains (default 25%)
    """
    annual_mgmt_fee: float = 0.01
    hurdle_rate: float = 0.06
    performance_fee_rate: float = 0.25
    
    def __post_init__(self):
        self._perf_calc = PerformanceFeeCalculator(
            hurdle_rate=self.hurdle_rate,
            fee_rate=self.performance_fee_rate
        )
    
    @property
    def monthly_mgmt_fee(self) -> float:
        """Monthly management fee rate."""
        return annual_to_monthly_fee(self.annual_mgmt_fee)
    
    def calculate_monthly_fees(
        self,
        monthly_returns: pd.DataFrame,
        return_column: str = 'return',
        month_column: str = 'month'
    ) -> pd.DataFrame:
        """
        Calculate monthly gross and net returns with fee breakdown.
        
        Applies management fee monthly, and performance fee annually.
        
        Args:
            monthly_returns: DataFrame with monthly returns
            return_column: Name of the return column
            month_column: Name of the month column (used to group by year)
        
        Returns:
            DataFrame with gross_return, net_return, management_fee, performance_fee
        """
        result = monthly_returns.copy()
        
        # Add management fee (applied monthly)
        result['gross_return'] = result[return_column]
        result['management_fee'] = self.monthly_mgmt_fee
        
        # Calculate net return before performance fee
        result['net_before_perf'] = result['gross_return'] - result['management_fee']
        
        # Performance fee is calculated annually - we need to track by fiscal year
        # For now, we'll add a placeholder and calculate at year-end
        result['performance_fee'] = 0.0
        
        # Final net return
        result['net_return'] = result['net_before_perf'] - result['performance_fee']
        
        return result
    
    def calculate_annual_performance_fee(
        self,
        annual_return: float,
        year: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Calculate performance fee for a year.
        
        Args:
            annual_return: The year's total return (after management fees)
            year: Optional year label
        
        Returns:
            Tuple of (fee_amount, carried_shortfall)
        """
        return self._perf_calc.calculate_annual_fee(annual_return, year)
    
    def get_performance_fee_history(self) -> pd.DataFrame:
        """Get performance fee calculation history."""
        return self._perf_calc.get_history_df()
    
    def reset_performance_fee(self):
        """Reset performance fee calculator state."""
        self._perf_calc.reset()


def calculate_fees_for_period(
    period_returns: pd.DataFrame,
    annual_mgmt_fee: float = 0.01,
    hurdle_rate: float = 0.06,
    performance_fee_rate: float = 0.25,
    return_column: str = 'return'
) -> Dict:
    """
    Calculate all fees for a reporting period.
    
    Args:
        period_returns: DataFrame with periodic (monthly) returns
        annual_mgmt_fee: Annual management fee rate
        hurdle_rate: Annual hurdle rate for performance fee
        performance_fee_rate: Performance fee rate on excess gains
        return_column: Name of the return column
    
    Returns:
        Dictionary with fee summary:
        - total_gross_return: Compound gross return for period
        - total_net_return: Compound net return for period
        - total_management_fee: Total management fee impact
        - total_performance_fee: Total performance fee impact
        - monthly_breakdown: DataFrame with monthly details
    """
    if period_returns.empty:
        return {
            'total_gross_return': 0.0,
            'total_net_return': 0.0,
            'total_management_fee': 0.0,
            'total_performance_fee': 0.0,
            'monthly_breakdown': pd.DataFrame()
        }
    
    calc = FeeCalculator(
        annual_mgmt_fee=annual_mgmt_fee,
        hurdle_rate=hurdle_rate,
        performance_fee_rate=performance_fee_rate
    )
    
    # Calculate monthly fees
    monthly = calc.calculate_monthly_fees(period_returns, return_column)
    
    # Calculate compound returns
    total_gross = float((1 + monthly['gross_return']).prod() - 1)
    total_net_before_perf = float((1 + monthly['net_before_perf']).prod() - 1)
    
    # Calculate performance fee on annual return
    perf_fee, _ = calc.calculate_annual_performance_fee(total_net_before_perf)
    
    # Final net return
    total_net = total_net_before_perf - perf_fee
    
    # Total management fee impact (approximate)
    monthly_fee = annual_to_monthly_fee(annual_mgmt_fee)
    total_mgmt_fee = len(monthly) * monthly_fee  # Simplified
    
    return {
        'total_gross_return': total_gross,
        'total_net_return': total_net,
        'total_management_fee': total_mgmt_fee,
        'total_performance_fee': perf_fee,
        'monthly_breakdown': monthly
    }

