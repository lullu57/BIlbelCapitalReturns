"""
Fee calculator for the GIPS-Compliant Returns Calculator.

Handles gross-to-net return calculations including:
- Management fees (annual rate converted to monthly)
- Performance fees with carry-forward shortfall hurdle mechanism
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


def calculate_net_return(gross_return: float, monthly_mgmt_fee: float) -> float:
    """
    Calculate net return after deducting management fee.
    
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

