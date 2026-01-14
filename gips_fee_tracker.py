"""
GIPS-Compliant Fee Tracker

This module provides a clean, consolidated implementation of fee tracking
for GIPS-compliant fund returns.

Fee Structure:
1. Management Fee: 0.25% of NAV at beginning of each quarter
   - Prorated for flows based on days in quarter
   - Inflows on fee day excluded; outflows on fee day included

2. Performance Fee: 25% of profits above cumulative hurdle
   - Hurdle: 1.5% per quarter, cumulative non-compounding
   - Q1 = 1.5%, Q2 = 3.0%, Q3 = 4.5%, etc.
   - Only charged when return exceeds HWM + hurdle_since_HWM
   - HWM = highest post-fee net return achieved
   - Prorated for flows based on days in quarter

3. High Water Mark Logic:
   - At inception: HWM = 1.0 (0% return), threshold = 1.5% for Q1
   - Threshold = HWM + (1.5% × quarters_since_HWM_was_set)
   - When return exceeds threshold: charge fee, update HWM, reset counter
   - When return doesn't exceed: HWM stays, counter increases

4. Flow Timing:
   - All flows treated as end-of-day
   - Inflows on fee day: NOT included in fee calculation
   - Outflows on fee day: INCLUDED in fee calculation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date
from calendar import monthrange
import pandas as pd
import numpy as np
import logging


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


def get_quarter_dates(year: int, quarter: int, fiscal_year_start_month: int = 2) -> Tuple[date, date, int]:
    """
    Get the start date, end date, and number of days for a fiscal quarter.
    
    Args:
        year: Calendar year
        quarter: Fiscal quarter (1-4)
        fiscal_year_start_month: Month when fiscal year starts (default 2 = February)
    
    Returns:
        Tuple of (start_date, end_date, days_in_quarter)
    """
    # Calculate start month for this quarter
    start_month_offset = (quarter - 1) * 3
    start_month = ((fiscal_year_start_month - 1 + start_month_offset) % 12) + 1
    
    # Handle year rollover
    start_year = year if start_month >= fiscal_year_start_month else year + 1
    if fiscal_year_start_month > 1 and start_month < fiscal_year_start_month:
        start_year = year + 1
    
    # Calculate end month (2 months after start)
    end_month = ((start_month - 1 + 2) % 12) + 1
    end_year = start_year if end_month >= start_month else start_year + 1
    
    start_date = date(start_year, start_month, 1)
    _, last_day = monthrange(end_year, end_month)
    end_date = date(end_year, end_month, last_day)
    
    days_in_quarter = (end_date - start_date).days + 1
    
    return start_date, end_date, days_in_quarter


@dataclass
class FlowRecord:
    """Record of a cash flow for proration calculations."""
    date: date
    amount: float
    is_inflow: bool
    
    @property
    def is_outflow(self) -> bool:
        return not self.is_inflow


@dataclass
class GIPSFeeTracker:
    """
    GIPS-compliant fee tracker with high-water-mark and quarterly crystallization.
    
    Fee Structure:
    - Management Fee: 0.25% of NAV, deducted on first day of each quarter
    - Performance Fee: 25% of gains above (HWM + cumulative hurdle since HWM)
      - Quarterly hurdle: 1.5% per quarter, added cumulatively
      - Threshold = HWM + (1.5% × quarters since HWM was set)
      - Deducted on last day of each quarter
    
    Timing:
    - Quarter End (e.g., Apr 30): Calculate and deduct performance fee
    - Next Quarter Start (e.g., May 1): Calculate and deduct management fee
    
    Attributes:
        initial_nav: Starting combined NAV from all accounts
        quarterly_mgmt_fee: Management fee rate per quarter (default 0.25%)
        quarterly_hurdle_rate: Quarterly hurdle rate (default 1.5%)
        perf_fee_rate: Performance fee rate (default 25%)
        fiscal_year_start_month: Month when fiscal year starts (default 2 = February)
    """
    initial_nav: float
    quarterly_mgmt_fee: float = 0.0025  # 0.25% per quarter
    quarterly_hurdle_rate: float = 0.015  # 1.5% per quarter
    perf_fee_rate: float = 0.25  # 25% of excess
    fiscal_year_start_month: int = 2  # February
    
    def __post_init__(self):
        # NAV tracking
        self.gross_nav = self.initial_nav  # Actual NAV from accounts
        self.net_nav = self.initial_nav    # NAV after fees
        
        # Return index tracking (net-of-fees, flow-neutral)
        self.net_index = 1.0
        
        # High Water Mark tracking
        # HWM is the highest net return index achieved at any fee crystallization
        self.high_water_mark = 1.0  # Starting at 1.0 (0% return)
        self.quarters_since_hwm = 0  # Quarters elapsed since HWM was set
        
        # Legacy field for backward compatibility
        self.cumulative_hurdle_pct = 0.0
        
        # Fee accumulators
        self.accumulated_mgmt_fees = 0.0
        self.accumulated_perf_fees = 0.0
        
        # Period tracking
        self.current_quarter = 0
        self.total_quarters_elapsed = 0  # Total quarters since inception
        self.current_fiscal_year: Optional[str] = None
        
        # Flow tracking for proration
        self.total_flows = 0.0
        self.quarter_flows: List[FlowRecord] = []  # Flows within current quarter
        self.current_quarter_start: Optional[date] = None
        self.current_quarter_end: Optional[date] = None
        self.days_in_current_quarter: int = 90  # Default, will be updated
        
        # History for audit trail
        self.history: List[Dict] = []
        self.quarterly_events: List[Dict] = []
        
        # Calculate quarter boundaries
        self._calculate_quarter_boundaries()
    
    def _calculate_quarter_boundaries(self):
        """Calculate which months are quarter starts and ends."""
        start = self.fiscal_year_start_month
        
        def normalize_month(m: int) -> int:
            return ((m - 1) % 12) + 1
        
        # Quarter start months (first day = mgmt fee day)
        self.quarter_start_months = [
            normalize_month(start),
            normalize_month(start + 3),
            normalize_month(start + 6),
            normalize_month(start + 9)
        ]
        
        # Quarter end months (last day = perf fee day)
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
    
    def is_fiscal_year_start(self, month: int) -> bool:
        return month == self.fiscal_year_start_month
    
    def _get_current_hurdle_threshold(self) -> float:
        """
        Calculate the current hurdle threshold.
        
        Threshold = HWM + (quarterly_hurdle × quarters_since_HWM)
        
        Example:
        - HWM = 1.05 (5% return), quarters_since_hwm = 2
        - Threshold = 1.05 + (0.015 × 2) = 1.05 + 0.03 = 1.08 (8% return required)
        """
        hurdle_since_hwm = self.quarterly_hurdle_rate * self.quarters_since_hwm
        return self.high_water_mark + hurdle_since_hwm
    
    def _get_performance_fee_proration_factor(
        self,
        flow_date: date,
        fee_date: date,
        is_inflow: bool
    ) -> float:
        """
        Calculate the proration factor for performance fees at quarter end.

        Inflows: based on days present after the flow (end-of-day).
        Outflows: based on days present before the flow (end-of-day).
        
        Args:
            flow_date: Date of the flow
            fee_date: Date of fee calculation
            is_inflow: Whether this is an inflow (True) or outflow (False)
        
        Returns:
            Proration factor between 0 and 1
        """
        if self.current_quarter_start is None:
            return 0.0
        
        days_in_quarter = self.days_in_current_quarter if self.days_in_current_quarter > 0 else 1
        
        if is_inflow:
            if flow_date >= fee_date:
                return 0.0  # Inflows on fee day are excluded
            days_present = (fee_date - flow_date).days
        else:
            # Outflows are present from quarter start through flow date (end-of-day)
            days_present = (flow_date - self.current_quarter_start).days + 1
        
        return min(1.0, max(0.0, days_present / days_in_quarter))

    def _get_mgmt_fee_adjustment_factor(self, flow_date: date) -> float:
        """
        Calculate the proration factor for management fee adjustments.

        Uses days remaining after the flow date (end-of-day), which corresponds
        to the portion of the quarter NOT covered by the fee charged at quarter start.
        """
        if self.current_quarter_end is None:
            return 0.0
        
        days_in_quarter = self.days_in_current_quarter if self.days_in_current_quarter > 0 else 1
        days_remaining = (self.current_quarter_end - flow_date).days
        
        return min(1.0, max(0.0, days_remaining / days_in_quarter))

    def add_flow_for_proration(self, flow_amount: float, flow_date: date):
        """
        Record a flow for proration tracking without affecting NAV.
        """
        is_inflow = flow_amount > 0
        self.quarter_flows.append(FlowRecord(
            date=flow_date,
            amount=flow_amount,
            is_inflow=is_inflow
        ))
    
    def start_new_fiscal_year(self, fiscal_year: str):
        """Mark the start of a new fiscal year for reporting labels."""
        self.current_fiscal_year = fiscal_year
        self.current_quarter = 0
        
        self.history.append({
            'event': 'fiscal_year_start',
            'fiscal_year': fiscal_year,
            'gross_nav': self.gross_nav,
            'net_nav': self.net_nav,
            'high_water_mark': self.high_water_mark,
            'net_index': self.net_index,
            'quarters_since_hwm': self.quarters_since_hwm,
            'cumulative_hurdle_pct': self.cumulative_hurdle_pct
        })
    
    def start_new_quarter(self, quarter: int, fiscal_year: str, year: int, month: int):
        """Initialize tracking for a new quarter."""
        self.current_quarter = quarter
        self.quarter_flows = []  # Reset flow tracking for new quarter
        
        # Calculate quarter dates
        try:
            _, last_day = monthrange(year, month)
            self.current_quarter_start = date(year, month, 1)
            
            # Calculate end of quarter (2 months later)
            end_month = ((month - 1 + 2) % 12) + 1
            end_year = year if end_month >= month else year + 1
            _, end_last_day = monthrange(end_year, end_month)
            self.current_quarter_end = date(end_year, end_month, end_last_day)
            
            self.days_in_current_quarter = (self.current_quarter_end - self.current_quarter_start).days + 1
        except Exception:
            self.days_in_current_quarter = 90  # Default fallback
    
    def apply_management_fee(
        self,
        quarter: int,
        fiscal_year: str,
        fee_date: Optional[date] = None
    ) -> float:
        """
        Apply management fee at start of quarter.
        
        Fee is 0.25% of current NET NAV, prorated for any flows.
        
        Proration:
        - Original capital: full fee (0.25%)
        - New inflows: prorated based on days remaining in quarter
        - Outflows: reduce NAV base for fee calculation
        
        Same-day handling (flows are end-of-day):
        - Inflows on fee day: NOT included in fee base
        - Outflows on fee day: INCLUDED in fee base (not yet withdrawn)
        """
        net_index_before = self.net_index
        
        # Calculate fee base considering flow proration from PREVIOUS quarter
        # For management fee at quarter START, we use NAV as-is
        # (proration applies to flows within the quarter for THAT quarter's fee)
        fee_base = self.net_nav
        
        # Apply standard fee (proration will be handled in detailed flow tracking)
        fee_amount = fee_base * self.quarterly_mgmt_fee
        
        self.accumulated_mgmt_fees += fee_amount
        self.net_nav -= fee_amount
        self.net_index *= (1 - self.quarterly_mgmt_fee)
        
        event = {
            'event': 'management_fee',
            'quarter': quarter,
            'fiscal_year': fiscal_year,
            'gross_nav': self.gross_nav,
            'net_nav_before': self.net_nav + fee_amount,
            'fee_rate': self.quarterly_mgmt_fee,
            'fee_amount': fee_amount,
            'net_nav_after': self.net_nav,
            'high_water_mark': self.high_water_mark,
            'quarters_since_hwm': self.quarters_since_hwm,
            'net_index_before': net_index_before,
            'net_index_after': self.net_index,
            'cumulative_hurdle_pct': self.cumulative_hurdle_pct
        }
        self.history.append(event)
        self.quarterly_events.append(event)
        
        return fee_amount

    def apply_management_fee_adjustment(
        self,
        quarter: int,
        fiscal_year: str,
        fee_date: Optional[date] = None
    ) -> float:
        """
        Apply management fee adjustment for intra-quarter flows.

        Base fee is charged at quarter start. This adjustment prorates flows
        based on days remaining in the quarter and is applied at quarter end
        (before performance fee).
        """
        if not self.quarter_flows:
            return 0.0

        fee_date = fee_date or self.current_quarter_end
        if fee_date is None:
            return 0.0

        adjustment_base = 0.0
        for flow in self.quarter_flows:
            factor = self._get_mgmt_fee_adjustment_factor(flow.date)
            adjustment_base += flow.amount * factor

        if adjustment_base == 0.0:
            return 0.0

        fee_amount = adjustment_base * self.quarterly_mgmt_fee
        net_nav_before = self.net_nav

        self.accumulated_mgmt_fees += fee_amount
        self.net_nav -= fee_amount
        if net_nav_before > 0:
            self.net_index *= (1 - (fee_amount / net_nav_before))

        event = {
            'event': 'management_fee_adjustment',
            'quarter': quarter,
            'fiscal_year': fiscal_year,
            'gross_nav': self.gross_nav,
            'net_nav_before': net_nav_before,
            'fee_rate': self.quarterly_mgmt_fee,
            'proration_base': adjustment_base,
            'fee_amount': fee_amount,
            'net_nav_after': self.net_nav,
            'high_water_mark': self.high_water_mark,
            'quarters_since_hwm': self.quarters_since_hwm,
            'net_index_after': self.net_index,
            'cumulative_hurdle_pct': self.cumulative_hurdle_pct
        }
        self.history.append(event)
        self.quarterly_events.append(event)

        return fee_amount
    
    def apply_performance_fee(
        self,
        quarter: int,
        fiscal_year: str,
        fee_date: Optional[date] = None
    ) -> float:
        """
        Apply performance fee at end of quarter.
        
        Fee is 25% of net return above threshold.
        Threshold = HWM + (1.5% × quarters_since_HWM)
        
        HIGH-WATER-MARK LOGIC:
        - Each quarter adds to quarters_since_hwm counter
        - Threshold = HWM + (hurdle_rate × quarters_since_hwm)
        - If net_index > threshold: charge fee, update HWM, reset counter
        - If net_index <= threshold: no fee, counter stays for next quarter
        
        At inception (Q1): HWM=1.0, quarters_since_hwm becomes 1, threshold=1.015
        """
        self.current_quarter = quarter
        
        # Increment quarters since HWM (this quarter adds to the hurdle)
        self.quarters_since_hwm += 1
        self.total_quarters_elapsed += 1
        
        # Update cumulative hurdle for backward compatibility
        self.cumulative_hurdle_pct = self.quarterly_hurdle_rate * self.total_quarters_elapsed
        
        # Calculate threshold using ADDITIVE method
        # Threshold = HWM + (hurdle_rate × quarters_since_hwm)
        hwm_before = self.high_water_mark
        hurdle_since_hwm = self.quarterly_hurdle_rate * self.quarters_since_hwm
        hurdle_threshold = hwm_before + hurdle_since_hwm  # ADDITIVE, not multiplicative
        
        pre_fee_index = self.net_index
        fee_amount = 0.0
        fee_percent = 0.0

        fee_date = fee_date or self.current_quarter_end
        fee_base = self.net_nav

        # Adjust fee base for flow proration
        if fee_date is not None and self.quarter_flows:
            for flow in self.quarter_flows:
                # Only adjust for flows that have already impacted NAV
                if flow.date >= fee_date:
                    continue
                if flow.is_inflow:
                    factor = self._get_performance_fee_proration_factor(flow.date, fee_date, True)
                    fee_base -= flow.amount * (1 - factor)
                else:
                    factor = self._get_performance_fee_proration_factor(flow.date, fee_date, False)
                    fee_base += (-flow.amount) * factor

        fee_base = max(0.0, fee_base)
        
        # Performance fee only if net index exceeds the hurdle threshold
        if pre_fee_index > hurdle_threshold:
            # Excess is the difference between current index and threshold
            excess_return = pre_fee_index - hurdle_threshold
            
            # Fee is 25% of the excess return, applied to NAV
            # fee_percent = (excess_return / pre_fee_index) × perf_fee_rate
            fee_percent = self.perf_fee_rate * (excess_return / pre_fee_index)
            fee_amount = fee_base * fee_percent
            
            self.accumulated_perf_fees += fee_amount
            net_nav_before = self.net_nav
            self.net_nav -= fee_amount
            if net_nav_before > 0:
                self.net_index *= (1 - (fee_amount / net_nav_before))
            
            # Update high-water-mark to post-fee net index
            self.high_water_mark = self.net_index
            
            # Reset quarters counter since we just set a new HWM
            self.quarters_since_hwm = 0
        
        # Note: If no fee charged, HWM stays the same, quarters_since_hwm already incremented
        
        event = {
            'event': 'performance_fee',
            'quarter': quarter,
            'fiscal_year': fiscal_year,
            'gross_nav': self.gross_nav,
            'net_nav_before': self.net_nav + fee_amount,
            'high_water_mark_before': hwm_before,
            'quarters_since_hwm_before': self.quarters_since_hwm + 1 if fee_amount > 0 else self.quarters_since_hwm,
            'hurdle_since_hwm': hurdle_since_hwm,
            'hurdle_threshold': hurdle_threshold,
            'pre_fee_index': pre_fee_index,
            'fee_base': fee_base,
            'fee_percent': fee_percent,
            'excess_gain': max(0.0, pre_fee_index - hurdle_threshold),
            'effective_hurdle': hurdle_since_hwm,
            'accumulated_hurdle_pct': self.cumulative_hurdle_pct,
            'fee_rate': self.perf_fee_rate,
            'fee_amount': fee_amount,
            'net_nav_after': self.net_nav,
            'high_water_mark_after': self.high_water_mark,
            'quarters_since_hwm_after': self.quarters_since_hwm,
            'net_index_after': self.net_index,
            'cumulative_hurdle_pct': self.cumulative_hurdle_pct
        }
        self.history.append(event)
        self.quarterly_events.append(event)
        
        return fee_amount
    
    def record_flow(
        self,
        flow_amount: float,
        flow_date: date,
        date_info: str = ''
    ):
        """
        Record a cash flow for proration tracking.
        
        Flows are treated as end-of-day.
        
        Args:
            flow_amount: Amount of flow (positive for inflow, negative for outflow)
            flow_date: Date of the flow
            date_info: Optional string description
        """
        is_inflow = flow_amount > 0
        
        self.quarter_flows.append(FlowRecord(
            date=flow_date,
            amount=flow_amount,
            is_inflow=is_inflow
        ))
        
        self.gross_nav += flow_amount
        self.net_nav += flow_amount
        self.total_flows += flow_amount
        
        self.history.append({
            'event': 'flow',
            'date': date_info or str(flow_date),
            'flow_amount': flow_amount,
            'is_inflow': is_inflow,
            'gross_nav_after': self.gross_nav,
            'net_nav_after': self.net_nav,
            'high_water_mark': self.high_water_mark,
            'net_index': self.net_index
        })
    
    def apply_flow(self, flow_amount: float, date_info: str = ''):
        """
        Apply a cash flow (deposit or withdrawal).
        
        Flows affect both gross NAV and net NAV equally.
        This is the legacy method - use record_flow for proration tracking.
        """
        self.gross_nav += flow_amount
        self.net_nav += flow_amount
        self.total_flows += flow_amount
        
        self.history.append({
            'event': 'flow',
            'date': date_info,
            'flow_amount': flow_amount,
            'gross_nav_after': self.gross_nav,
            'net_nav_after': self.net_nav,
            'high_water_mark_after': self.high_water_mark,
            'net_index': self.net_index
        })
    
    def apply_return(self, return_pct: float, period_info: str = ''):
        """Apply investment return for a period."""
        gross_before = self.gross_nav
        
        self.gross_nav *= (1 + return_pct)
        gain = self.gross_nav - gross_before
        self.net_nav += gain
        self.net_index *= (1 + return_pct)
        
        self.history.append({
            'event': 'return',
            'period': period_info,
            'return_pct': return_pct,
            'gross_nav_before': gross_before,
            'gross_nav_after': self.gross_nav,
            'net_nav_after': self.net_nav,
            'net_index_after': self.net_index
        })
    
    def process_month(
        self,
        month: int,
        year: int,
        gross_nav_end: float,
        flow_amount: float = 0.0,
        monthly_return: Optional[float] = None,
        daily_flows: Optional[List[Tuple[date, float]]] = None
    ) -> Dict:
        """
        Process a single month's data.
        
        Args:
            month: Calendar month (1-12)
            year: Calendar year
            gross_nav_end: Actual gross NAV at end of month
            flow_amount: Total flows during the month
            monthly_return: Optional pre-calculated monthly return
        
        Returns:
            Dict with monthly results including net return
        """
        # Determine fiscal year
        if month >= self.fiscal_year_start_month:
            fiscal_year = str(year)
        else:
            fiscal_year = str(year - 1)
        
        quarter = self.get_quarter_for_month(month)
        
        # Check for fiscal year start
        if self.is_fiscal_year_start(month):
            self.start_new_fiscal_year(fiscal_year)

        net_index_start_month = self.net_index
        mgmt_fee = 0.0

        # At quarter start: initialize quarter and apply management fee
        if self.is_quarter_start(month):
            self.start_new_quarter(quarter, fiscal_year, year, month)
            mgmt_fee = self.apply_management_fee(quarter, fiscal_year)

        # Record daily flows for proration (flows are end-of-day)
        if daily_flows:
            for flow_date, flow_value in daily_flows:
                self.add_flow_for_proration(flow_value, flow_date)
        
        # Save NAV before flow for investment gain calculation
        gross_nav_before_flow = self.gross_nav
        net_nav_start_after_mgmt = self.net_nav
        
        # Calculate investment gain for this month
        investment_gain = gross_nav_end - gross_nav_before_flow - flow_amount
        
        # Update gross NAV to pre-flow value; flow is end-of-day
        self.gross_nav = gross_nav_end - flow_amount
        
        # Apply the investment gain to net NAV
        self.net_nav += investment_gain
        
        # Apply investment return to net index
        net_return_pre_fee = None
        if monthly_return is not None and np.isfinite(monthly_return):
            net_return_pre_fee = monthly_return
        elif net_nav_start_after_mgmt > 0:
            net_return_pre_fee = investment_gain / net_nav_start_after_mgmt
        else:
            net_return_pre_fee = 0.0

        self.net_index *= (1 + net_return_pre_fee)
        
        # At quarter end: apply management fee adjustment then performance fee (before flow)
        perf_fee = 0.0
        if self.is_quarter_end(month):
            self.apply_management_fee_adjustment(quarter, fiscal_year, fee_date=self.current_quarter_end)
            perf_fee = self.apply_performance_fee(quarter, fiscal_year)
        
        # End-of-day flow (does not affect net index)
        if flow_amount != 0:
            self.apply_flow(flow_amount, f"{year}-{month:02d}")
        
        # Calculate net return for this month
        if net_index_start_month > 0:
            net_return = (self.net_index / net_index_start_month) - 1
        else:
            net_return = 0.0
        
        # Calculate current hurdle threshold for reporting
        current_threshold = self._get_current_hurdle_threshold()
        
        result = {
            'month': month,
            'year': year,
            'fiscal_year': fiscal_year,
            'quarter': quarter,
            'gross_nav_end': gross_nav_end,
            'net_nav_end': self.net_nav,
            'flow': flow_amount,
            'investment_gain': investment_gain,
            'net_return': net_return,
            'management_fee': mgmt_fee,
            'perf_fee': perf_fee,
            'high_water_mark': self.high_water_mark,
            'quarters_since_hwm': self.quarters_since_hwm,
            'hurdle_threshold': current_threshold,
            'net_index': self.net_index,
            'cumulative_hurdle_pct': self.cumulative_hurdle_pct,
            'accumulated_mgmt_fees': self.accumulated_mgmt_fees,
            'accumulated_perf_fees': self.accumulated_perf_fees
        }
        
        self.history.append({
            'event': 'monthly_update',
            **result
        })
        
        return result
    
    def process_monthly_data(
        self,
        monthly_df: pd.DataFrame,
        month_column: str = 'month',
        gross_nav_column: str = 'end_nav',
        flow_column: str = 'flow',
        return_column: str = 'return',
        daily_flows: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Process a DataFrame of monthly data.
        
        Returns comprehensive results including:
        - Final gross and net NAV
        - Total fees paid
        - Monthly net returns for proper TWR calculation
        """
        df = monthly_df.copy()
        
        # Parse month column
        if month_column in df.columns:
            df['_month'] = pd.to_datetime(df[month_column].astype(str)).dt.month
            df['_year'] = pd.to_datetime(df[month_column].astype(str)).dt.year
        else:
            raise ValueError(f"Month column '{month_column}' not found")
        
        # Helper to parse values
        def parse_value(val, default=0.0):
            if pd.isna(val):
                return default
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):
                try:
                    cleaned = val.replace('€', '').replace(',', '').replace('%', '').strip()
                    return float(cleaned) / 100 if '%' in val else float(cleaned)
                except:
                    return default
            return default
        
        monthly_results = []

        daily_flows_by_month: Dict[Tuple[int, int], List[Tuple[date, float]]] = {}
        if isinstance(daily_flows, pd.DataFrame) and not daily_flows.empty:
            flow_df = daily_flows.copy()
            date_col = 'date' if 'date' in flow_df.columns else 'Date'
            if date_col not in flow_df.columns:
                date_col = flow_df.columns[0]
            flow_col = 'flow' if 'flow' in flow_df.columns else 'flow_total' if 'flow_total' in flow_df.columns else None
            if flow_col is None:
                flow_col = flow_df.columns[1] if len(flow_df.columns) > 1 else flow_df.columns[0]

            flow_df[date_col] = pd.to_datetime(flow_df[date_col]).dt.date
            flow_df[flow_col] = pd.to_numeric(flow_df[flow_col], errors='coerce').fillna(0.0)

            for _, row in flow_df.iterrows():
                flow_date = row[date_col]
                flow_value = float(row[flow_col])
                if not np.isfinite(flow_value) or flow_value == 0.0:
                    continue
                key = (flow_date.year, flow_date.month)
                daily_flows_by_month.setdefault(key, []).append((flow_date, flow_value))
        
        for idx, row in df.iterrows():
            month = int(row['_month'])
            year = int(row['_year'])
            
            gross_nav_end = parse_value(row.get(gross_nav_column, 0))
            flow = parse_value(row.get(flow_column, 0))
            monthly_return = parse_value(row.get(return_column, 0))
            
            # Skip rows with no NAV data
            if gross_nav_end <= 0 and self.gross_nav <= 0:
                continue
            
            # Initialize if this is the first valid data
            if self.initial_nav == 0 and gross_nav_end > 0:
                self.initial_nav = gross_nav_end - flow if flow > 0 else gross_nav_end
                self.gross_nav = self.initial_nav
                self.net_nav = self.initial_nav
                self.net_index = 1.0
                self.high_water_mark = 1.0
                self.quarters_since_hwm = 0
                self.cumulative_hurdle_pct = 0.0
            
            result = self.process_month(
                month=month,
                year=year,
                gross_nav_end=gross_nav_end,
                flow_amount=flow,
                monthly_return=monthly_return,
                daily_flows=daily_flows_by_month.get((year, month), [])
            )
            monthly_results.append(result)
        
        monthly_net_returns = pd.DataFrame(monthly_results)
        
        return {
            'initial_nav': self.initial_nav,
            'final_gross_nav': self.gross_nav,
            'final_net_nav': self.net_nav,
            'high_water_mark': self.high_water_mark,
            'quarters_since_hwm': self.quarters_since_hwm,
            'total_mgmt_fees': self.accumulated_mgmt_fees,
            'total_perf_fees': self.accumulated_perf_fees,
            'total_fees': self.accumulated_mgmt_fees + self.accumulated_perf_fees,
            'total_flows': self.total_flows,
            'monthly_net_returns': monthly_net_returns,
            'history': self.history,
            'quarterly_events': self.quarterly_events
        }
    
    def get_summary(self) -> Dict:
        """Get a summary of the fee tracker state."""
        current_threshold = self._get_current_hurdle_threshold()
        
        return {
            'gross_nav': self.gross_nav,
            'net_nav': self.net_nav,
            'high_water_mark': self.high_water_mark,
            'quarters_since_hwm': self.quarters_since_hwm,
            'hurdle_threshold': current_threshold,
            'net_index': self.net_index,
            'cumulative_hurdle_pct': self.cumulative_hurdle_pct,
            'total_mgmt_fees': self.accumulated_mgmt_fees,
            'total_perf_fees': self.accumulated_perf_fees,
            'total_fees': self.accumulated_mgmt_fees + self.accumulated_perf_fees,
            'fee_drag': self.gross_nav - self.net_nav,
            'total_flows': self.total_flows
        }


def calculate_net_twr_from_monthly(
    monthly_net_returns: pd.DataFrame,
    return_column: str = 'net_return'
) -> Tuple[float, float]:
    """
    Calculate Net TWR by geometrically linking monthly net returns.
    
    This is the GIPS-compliant method for calculating net-of-fee returns.
    
    Args:
        monthly_net_returns: DataFrame with monthly net returns
        return_column: Name of the column containing net returns
    
    Returns:
        Tuple of (absolute_net_twr, annualized_net_twr)
    """
    if monthly_net_returns.empty or return_column not in monthly_net_returns.columns:
        return 0.0, 0.0
    
    valid_returns = monthly_net_returns[return_column].dropna()
    if valid_returns.empty:
        return 0.0, 0.0
    
    net_multiplier = (1 + valid_returns).prod()
    absolute_net_twr = net_multiplier - 1
    
    num_months = len(valid_returns)
    num_years = num_months / 12
    
    if num_years > 0 and net_multiplier > 0:
        annualized_net_twr = net_multiplier ** (1 / num_years) - 1
    else:
        annualized_net_twr = absolute_net_twr
    
    return absolute_net_twr, annualized_net_twr


def calculate_period_net_return(
    gross_return: float,
    quarterly_mgmt_fee: float = 0.0025,
    quarterly_hurdle: float = 0.015,
    perf_fee_rate: float = 0.25,
    num_quarters: int = 4,
    high_water_mark_ratio: float = 1.0
) -> Dict:
    """
    Calculate net return for a period with proper quarterly fee crystallization.
    
    Uses a simulation approach with the correct ADDITIVE hurdle calculation:
    Threshold = HWM + (hurdle_rate × quarters_since_HWM)
    
    Args:
        gross_return: Gross return for the period (e.g., 0.88 for 88%)
        quarterly_mgmt_fee: Management fee per quarter
        quarterly_hurdle: Hurdle rate per quarter
        perf_fee_rate: Performance fee rate
        num_quarters: Number of quarters in the period
        high_water_mark_ratio: HWM as ratio of starting NAV (1.0 = at HWM)
    
    Returns:
        Dict with net_return, mgmt_fees, perf_fees
    """
    # Start with index = 1.0 (tracking returns, not NAV)
    nav = 100.0  # For fee calculations
    index = 1.0  # For return tracking
    hwm = 1.0 * high_water_mark_ratio
    quarters_since_hwm = 0
    
    # Distribute gross return evenly across quarters
    quarterly_gross_return = (1 + gross_return) ** (1 / num_quarters) - 1
    
    total_mgmt_fees = 0.0
    total_perf_fees = 0.0
    
    for q in range(1, num_quarters + 1):
        # Start of quarter: apply management fee
        mgmt_fee = nav * quarterly_mgmt_fee
        nav -= mgmt_fee
        index *= (1 - quarterly_mgmt_fee)
        total_mgmt_fees += mgmt_fee
        
        # During quarter: apply return
        nav *= (1 + quarterly_gross_return)
        index *= (1 + quarterly_gross_return)
        
        # End of quarter: apply performance fee with ADDITIVE hurdle
        quarters_since_hwm += 1
        hurdle_since_hwm = quarterly_hurdle * quarters_since_hwm
        hurdle_threshold = hwm + hurdle_since_hwm  # ADDITIVE
        
        if index > hurdle_threshold:
            excess = index - hurdle_threshold
            fee_pct = perf_fee_rate * (excess / index)
            perf_fee = nav * fee_pct
            nav -= perf_fee
            index *= (1 - fee_pct)
            total_perf_fees += perf_fee
            hwm = index
            quarters_since_hwm = 0
    
    net_return = (nav - 100.0) / 100.0
    
    return {
        'net_return': net_return,
        'gross_return': gross_return,
        'mgmt_fee_impact': total_mgmt_fees / 100.0,
        'perf_fee_impact': total_perf_fees / 100.0,
        'final_nav': nav
    }
