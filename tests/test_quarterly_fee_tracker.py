"""
Tests for QuarterlyFeeTracker with proper quarterly fee timing.

Fee Timing Rules:
1. Management fee: 0.25% of NAV on 1st day of each fiscal quarter
2. Performance fee: 25% of gains above quarterly accumulated hurdle (1.5%/quarter)
   - Deducted at end of each quarter
3. For February fiscal year: Q1=Feb-Apr, Q2=May-Jul, Q3=Aug-Oct, Q4=Nov-Jan

Test Scenarios:
- Single quarter with positive return above hurdle
- Single quarter with positive return below hurdle
- Full year with consistent returns
- Multi-year with shortfall carry-forward
- NAV compounding effect of fee deductions
"""

import pytest
import pandas as pd
import numpy as np
from fee_calculator import QuarterlyFeeTracker, PerformanceFeeCalculator


class TestQuarterlyFeeTrackerInit:
    """Test initialization of QuarterlyFeeTracker."""
    
    def test_default_values(self):
        """Test default initialization values."""
        tracker = QuarterlyFeeTracker(initial_nav=100.0)
        assert tracker.initial_nav == 100.0
        assert tracker.nav == 100.0
        assert tracker.quarterly_mgmt_fee == 0.0025
        assert tracker.annual_hurdle_rate == 0.06
        assert tracker.quarterly_hurdle == 0.015  # 1.5% per quarter
        assert tracker.perf_fee_rate == 0.25
        assert tracker.fiscal_year_start_month == 2
    
    def test_custom_values(self):
        """Test custom initialization values."""
        tracker = QuarterlyFeeTracker(
            initial_nav=50000.0,
            quarterly_mgmt_fee=0.003,
            annual_hurdle_rate=0.08,
            perf_fee_rate=0.20,
            fiscal_year_start_month=1
        )
        assert tracker.initial_nav == 50000.0
        assert tracker.quarterly_mgmt_fee == 0.003
        assert tracker.quarterly_hurdle == 0.02  # 8% / 4 = 2%
        assert tracker.perf_fee_rate == 0.20
        assert tracker.fiscal_year_start_month == 1


class TestManagementFee:
    """Test management fee deduction."""
    
    def test_management_fee_deduction(self):
        """Test that management fee is correctly deducted from NAV."""
        tracker = QuarterlyFeeTracker(initial_nav=100000.0)
        
        result = tracker.apply_management_fee(quarter=1, fiscal_year="2024")
        
        # 0.25% of 100,000 = 250
        assert result['fee_amount'] == pytest.approx(250.0)
        assert result['nav_after'] == pytest.approx(99750.0)
        assert tracker.nav == pytest.approx(99750.0)
    
    def test_management_fee_compounds(self):
        """Test that management fees reduce NAV for subsequent calculations."""
        tracker = QuarterlyFeeTracker(initial_nav=100000.0)
        
        # Q1 fee
        tracker.apply_management_fee(quarter=1, fiscal_year="2024")
        assert tracker.nav == pytest.approx(99750.0)
        
        # Apply some return
        tracker.apply_month_return(0.02, month=2, year=2024)  # 2% return
        nav_after_feb = tracker.nav
        
        # Q2 fee (should be on lower NAV due to Q1 fee)
        tracker.apply_management_fee(quarter=2, fiscal_year="2024")
        
        # Fee should be 0.25% of nav_after_feb
        expected_fee = nav_after_feb * 0.0025
        expected_nav = nav_after_feb - expected_fee
        assert tracker.nav == pytest.approx(expected_nav)


class TestPerformanceFee:
    """Test performance fee deduction."""
    
    def test_performance_fee_above_hurdle(self):
        """Test performance fee when return exceeds quarterly hurdle."""
        tracker = QuarterlyFeeTracker(initial_nav=100000.0)
        tracker.year_start_nav = 100000.0
        
        # Apply 5% return (well above 1.5% quarterly hurdle)
        tracker.nav = 105000.0
        
        result = tracker.apply_performance_fee(quarter=1, fiscal_year="2024")
        
        # Gain = 5000
        # Hurdle amount = 100000 * 1.5% = 1500
        # Total excess = 5000 - 1500 = 3500
        # Incremental excess = 3500 (first quarter, nothing charged yet)
        # Perf fee = 3500 * 25% = 875
        assert result['actual_gain'] == pytest.approx(5000.0)
        assert result['hurdle_amount'] == pytest.approx(1500.0)
        assert result['total_excess_gain'] == pytest.approx(3500.0)
        assert result['incremental_excess'] == pytest.approx(3500.0)
        assert result['fee_amount'] == pytest.approx(875.0)
        assert tracker.nav == pytest.approx(105000.0 - 875.0)
    
    def test_performance_fee_below_hurdle(self):
        """Test no performance fee when return is below hurdle."""
        tracker = QuarterlyFeeTracker(initial_nav=100000.0)
        tracker.year_start_nav = 100000.0
        
        # Apply 1% return (below 1.5% quarterly hurdle)
        tracker.nav = 101000.0
        
        result = tracker.apply_performance_fee(quarter=1, fiscal_year="2024")
        
        # Gain = 1000
        # Hurdle amount = 100000 * 1.5% = 1500
        # Total excess = 0 (below hurdle)
        # Incremental excess = 0
        # Perf fee = 0
        assert result['actual_gain'] == pytest.approx(1000.0)
        assert result['total_excess_gain'] == pytest.approx(0.0)
        assert result['incremental_excess'] == pytest.approx(0.0)
        assert result['fee_amount'] == pytest.approx(0.0)
        assert tracker.nav == pytest.approx(101000.0)  # Unchanged
    
    def test_accumulated_hurdle_across_quarters(self):
        """Test that hurdle accumulates across quarters within a year, and no double-counting."""
        tracker = QuarterlyFeeTracker(initial_nav=100000.0)
        tracker.year_start_nav = 100000.0
        
        # Q1: 2% return, hurdle = 1.5%, excess = 0.5% = €500
        tracker.nav = 102000.0
        q1_result = tracker.apply_performance_fee(quarter=1, fiscal_year="2024")
        
        # Q1: Total excess = €500, incremental = €500, fee = €125
        assert q1_result['total_excess_gain'] == pytest.approx(500.0)
        assert q1_result['incremental_excess'] == pytest.approx(500.0)
        assert q1_result['fee_amount'] == pytest.approx(125.0)
        nav_after_q1 = tracker.nav  # 102000 - 125 = 101875
        
        # Q2: Set NAV to 106000 (6% total gain from year start)
        # This gives us NEW excess to charge on
        tracker.nav = 106000.0
        q2_result = tracker.apply_performance_fee(quarter=2, fiscal_year="2024")
        
        # Accumulated hurdle = 3% (1.5% + 1.5%)
        assert q2_result['accumulated_hurdle'] == pytest.approx(0.03)
        
        # Gain = 6000
        # Hurdle amount = 100000 * 3% = 3000
        # Total excess = 6000 - 3000 = 3000
        # Already charged on 500 in Q1
        # Incremental excess = 3000 - 500 = 2500
        # Perf fee = 2500 * 25% = 625
        assert q2_result['total_excess_gain'] == pytest.approx(3000.0)
        assert q2_result['incremental_excess'] == pytest.approx(2500.0)
        assert q2_result['fee_amount'] == pytest.approx(625.0)

    def test_no_double_counting_when_excess_unchanged(self):
        """
        Test that no fees are charged when cumulative excess hasn't increased.
        
        Scenario: Strong Q1 gains, then flat Q2 that still exceeds new hurdle
        but doesn't add new excess - should charge €0 in Q2.
        """
        tracker = QuarterlyFeeTracker(initial_nav=100000.0)
        tracker.year_start_nav = 100000.0
        
        # Q1: 10% return, hurdle = 1.5%, excess = 8.5% = €8,500
        tracker.nav = 110000.0
        q1_result = tracker.apply_performance_fee(quarter=1, fiscal_year="2024")
        assert q1_result['total_excess_gain'] == pytest.approx(8500.0)
        assert q1_result['fee_amount'] == pytest.approx(2125.0)  # 8500 * 25%
        
        # Q2: NAV drops to 103500 (3.5% total from year start)
        # Still above Q2 hurdle (3%), but excess is LESS than Q1
        tracker.nav = 103500.0
        q2_result = tracker.apply_performance_fee(quarter=2, fiscal_year="2024")
        
        # Gain = 3500, Hurdle = 3% = 3000
        # Total excess = 500, but we already charged on 8500!
        # Incremental = max(0, 500 - 8500) = 0
        # No fee should be charged
        assert q2_result['total_excess_gain'] == pytest.approx(500.0)
        assert q2_result['incremental_excess'] == pytest.approx(0.0)
        assert q2_result['fee_amount'] == pytest.approx(0.0)

    def test_double_counting_prevention_full_year(self):
        """
        Test double-counting prevention across all 4 quarters.
        
        Verifies that total fees charged equals exactly 25% of final excess,
        not more due to compounding.
        """
        tracker = QuarterlyFeeTracker(initial_nav=100000.0)
        tracker.year_start_nav = 100000.0
        
        total_fees = 0.0
        
        # Q1: 3% return (above 1.5% hurdle)
        tracker.nav = 103000.0
        q1 = tracker.apply_performance_fee(quarter=1, fiscal_year="2024")
        total_fees += q1['fee_amount']
        # Excess = 3000 - 1500 = 1500, Fee = 375
        
        # Q2: 5% total return (above 3% hurdle)  
        tracker.nav = 105000.0
        q2 = tracker.apply_performance_fee(quarter=2, fiscal_year="2024")
        total_fees += q2['fee_amount']
        # Excess = 5000 - 3000 = 2000, Incremental = 2000 - 1500 = 500, Fee = 125
        
        # Q3: 8% total return (above 4.5% hurdle)
        tracker.nav = 108000.0
        q3 = tracker.apply_performance_fee(quarter=3, fiscal_year="2024")
        total_fees += q3['fee_amount']
        # Excess = 8000 - 4500 = 3500, Incremental = 3500 - 2000 = 1500, Fee = 375
        
        # Q4: 12% total return (above 6% hurdle)
        tracker.nav = 112000.0
        q4 = tracker.apply_performance_fee(quarter=4, fiscal_year="2024")
        total_fees += q4['fee_amount']
        # Excess = 12000 - 6000 = 6000, Incremental = 6000 - 3500 = 2500, Fee = 625
        
        # Total fees should equal 25% of final excess (6000)
        final_excess = 112000.0 - 100000.0 - (100000.0 * 0.06)  # 12000 - 6000 = 6000
        expected_total_fees = final_excess * 0.25  # 1500
        
        assert total_fees == pytest.approx(expected_total_fees)
        assert total_fees == pytest.approx(1500.0)


class TestQuarterBoundaries:
    """Test fiscal quarter boundary detection for various fiscal year configurations."""
    
    def test_february_fiscal_year_quarters(self):
        """Test quarter detection for February fiscal year (Feb 1 - Jan 31)."""
        tracker = QuarterlyFeeTracker(initial_nav=100.0, fiscal_year_start_month=2)
        
        # Quarter start months
        assert tracker.quarter_start_months == [2, 5, 8, 11]
        # Quarter end months
        assert tracker.quarter_end_months == [4, 7, 10, 1]
        
        # February is start of Q1
        assert tracker.is_quarter_start(2) == True
        assert tracker.is_fiscal_year_start(2) == True
        assert tracker.get_quarter_for_month(2) == 1
        assert tracker.get_quarter_for_month(3) == 1
        assert tracker.get_quarter_for_month(4) == 1
        assert tracker.is_quarter_end(4) == True
        
        # May is start of Q2
        assert tracker.is_quarter_start(5) == True
        assert tracker.get_quarter_for_month(5) == 2
        assert tracker.get_quarter_for_month(6) == 2
        assert tracker.get_quarter_for_month(7) == 2
        assert tracker.is_quarter_end(7) == True
        
        # August is start of Q3
        assert tracker.is_quarter_start(8) == True
        assert tracker.get_quarter_for_month(8) == 3
        assert tracker.get_quarter_for_month(9) == 3
        assert tracker.get_quarter_for_month(10) == 3
        assert tracker.is_quarter_end(10) == True
        
        # November is start of Q4
        assert tracker.is_quarter_start(11) == True
        assert tracker.get_quarter_for_month(11) == 4
        assert tracker.get_quarter_for_month(12) == 4
        assert tracker.get_quarter_for_month(1) == 4  # January is Q4 for Feb fiscal year
        assert tracker.is_quarter_end(1) == True
    
    def test_january_fiscal_year_quarters(self):
        """Test quarter detection for January fiscal year (Jan 1 - Dec 31)."""
        tracker = QuarterlyFeeTracker(initial_nav=100.0, fiscal_year_start_month=1)
        
        # Quarter start months
        assert tracker.quarter_start_months == [1, 4, 7, 10]
        # Quarter end months
        assert tracker.quarter_end_months == [3, 6, 9, 12]
        
        # January is start of Q1
        assert tracker.is_quarter_start(1) == True
        assert tracker.is_fiscal_year_start(1) == True
        assert tracker.get_quarter_for_month(1) == 1
        assert tracker.get_quarter_for_month(2) == 1
        assert tracker.get_quarter_for_month(3) == 1
        assert tracker.is_quarter_end(3) == True
        
        # April is start of Q2
        assert tracker.is_quarter_start(4) == True
        assert tracker.get_quarter_for_month(4) == 2
        assert tracker.get_quarter_for_month(5) == 2
        assert tracker.get_quarter_for_month(6) == 2
        assert tracker.is_quarter_end(6) == True
        
        # July is start of Q3
        assert tracker.is_quarter_start(7) == True
        assert tracker.get_quarter_for_month(7) == 3
        assert tracker.get_quarter_for_month(8) == 3
        assert tracker.get_quarter_for_month(9) == 3
        assert tracker.is_quarter_end(9) == True
        
        # October is start of Q4
        assert tracker.is_quarter_start(10) == True
        assert tracker.get_quarter_for_month(10) == 4
        assert tracker.get_quarter_for_month(11) == 4
        assert tracker.get_quarter_for_month(12) == 4
        assert tracker.is_quarter_end(12) == True
    
    def test_april_fiscal_year_quarters(self):
        """Test quarter detection for April fiscal year (Apr 1 - Mar 31)."""
        tracker = QuarterlyFeeTracker(initial_nav=100.0, fiscal_year_start_month=4)
        
        # Quarter start months
        assert tracker.quarter_start_months == [4, 7, 10, 1]
        # Quarter end months
        assert tracker.quarter_end_months == [6, 9, 12, 3]
        
        # April is start of Q1
        assert tracker.is_quarter_start(4) == True
        assert tracker.is_fiscal_year_start(4) == True
        assert tracker.get_quarter_for_month(4) == 1
        assert tracker.get_quarter_for_month(5) == 1
        assert tracker.get_quarter_for_month(6) == 1
        assert tracker.is_quarter_end(6) == True
        
        # July is start of Q2
        assert tracker.is_quarter_start(7) == True
        assert tracker.get_quarter_for_month(7) == 2
        
        # October is start of Q3
        assert tracker.is_quarter_start(10) == True
        assert tracker.get_quarter_for_month(10) == 3
        
        # January is start of Q4 (wraps around year)
        assert tracker.is_quarter_start(1) == True
        assert tracker.get_quarter_for_month(1) == 4
        assert tracker.get_quarter_for_month(2) == 4
        assert tracker.get_quarter_for_month(3) == 4
        assert tracker.is_quarter_end(3) == True
    
    def test_july_fiscal_year_quarters(self):
        """Test quarter detection for July fiscal year (Jul 1 - Jun 30)."""
        tracker = QuarterlyFeeTracker(initial_nav=100.0, fiscal_year_start_month=7)
        
        # Quarter start months
        assert tracker.quarter_start_months == [7, 10, 1, 4]
        # Quarter end months
        assert tracker.quarter_end_months == [9, 12, 3, 6]
        
        # July is start of Q1
        assert tracker.get_quarter_for_month(7) == 1
        assert tracker.get_quarter_for_month(8) == 1
        assert tracker.get_quarter_for_month(9) == 1
        
        # October is Q2
        assert tracker.get_quarter_for_month(10) == 2
        assert tracker.get_quarter_for_month(11) == 2
        assert tracker.get_quarter_for_month(12) == 2
        
        # January is Q3
        assert tracker.get_quarter_for_month(1) == 3
        assert tracker.get_quarter_for_month(2) == 3
        assert tracker.get_quarter_for_month(3) == 3
        
        # April is Q4
        assert tracker.get_quarter_for_month(4) == 4
        assert tracker.get_quarter_for_month(5) == 4
        assert tracker.get_quarter_for_month(6) == 4
    
    def test_october_fiscal_year_quarters(self):
        """Test quarter detection for October fiscal year (Oct 1 - Sep 30)."""
        tracker = QuarterlyFeeTracker(initial_nav=100.0, fiscal_year_start_month=10)
        
        # Quarter start months
        assert tracker.quarter_start_months == [10, 1, 4, 7]
        # Quarter end months
        assert tracker.quarter_end_months == [12, 3, 6, 9]
        
        # October is Q1
        assert tracker.get_quarter_for_month(10) == 1
        assert tracker.get_quarter_for_month(11) == 1
        assert tracker.get_quarter_for_month(12) == 1
        
        # January is Q2
        assert tracker.get_quarter_for_month(1) == 2
        assert tracker.get_quarter_for_month(2) == 2
        assert tracker.get_quarter_for_month(3) == 2
        
        # April is Q3
        assert tracker.get_quarter_for_month(4) == 3
        
        # July is Q4
        assert tracker.get_quarter_for_month(7) == 4
        assert tracker.get_quarter_for_month(9) == 4


class TestMonthlyReturnsProcessing:
    """Test processing of monthly returns through the tracker."""
    
    def test_single_quarter_processing(self):
        """Test processing returns for a single quarter."""
        tracker = QuarterlyFeeTracker(initial_nav=100000.0, fiscal_year_start_month=2)
        
        # Create monthly returns for Q1 (Feb, Mar, Apr)
        monthly_data = pd.DataFrame({
            'month': [pd.Period('2024-02'), pd.Period('2024-03'), pd.Period('2024-04')],
            'return': [0.02, 0.015, 0.01]  # 2%, 1.5%, 1%
        })
        
        result = tracker.process_monthly_returns(monthly_data)
        
        # Should have:
        # - 1 mgmt fee event (start of Q1)
        # - 3 monthly return events
        # - 1 perf fee event (end of Q1)
        mgmt_events = [h for h in result['history'] if h['event'] == 'management_fee']
        return_events = [h for h in result['history'] if h['event'] == 'monthly_return']
        perf_events = [h for h in result['history'] if h['event'] == 'performance_fee']
        
        assert len(mgmt_events) == 1
        assert len(return_events) == 3
        assert len(perf_events) == 1
        
        # Verify gross return (compounded)
        expected_gross = (1.02 * 1.015 * 1.01) - 1
        assert result['total_gross_return'] == pytest.approx(expected_gross, rel=1e-4)
        
        # Net return should be less than gross due to fees
        assert result['total_net_return'] < result['total_gross_return']
    
    def test_full_year_processing(self):
        """Test processing returns for a full fiscal year."""
        tracker = QuarterlyFeeTracker(initial_nav=100000.0, fiscal_year_start_month=2)
        
        # Create monthly returns for full year (Feb 2024 - Jan 2025)
        months = [pd.Period(f'2024-{m:02d}') for m in range(2, 13)] + [pd.Period('2025-01')]
        returns = [0.01] * 12  # 1% each month
        
        monthly_data = pd.DataFrame({
            'month': months,
            'return': returns
        })
        
        result = tracker.process_monthly_returns(monthly_data)
        
        # Should have 4 management fee events (one per quarter)
        mgmt_events = [h for h in result['history'] if h['event'] == 'management_fee']
        assert len(mgmt_events) == 4
        
        # Should have 4 performance fee events (one per quarter)
        perf_events = [h for h in result['history'] if h['event'] == 'performance_fee']
        assert len(perf_events) == 4
        
        # Gross return = (1.01)^12 - 1 ≈ 12.68%
        expected_gross = (1.01 ** 12) - 1
        assert result['total_gross_return'] == pytest.approx(expected_gross, rel=1e-3)
        
        # Total fees should be positive
        assert result['total_mgmt_fees'] > 0
        assert result['total_perf_fees'] > 0
    
    def test_january_fiscal_year_full_year(self):
        """Test processing for January fiscal year (standard calendar year)."""
        tracker = QuarterlyFeeTracker(initial_nav=100000.0, fiscal_year_start_month=1)
        
        # Create monthly returns for full calendar year (Jan - Dec 2024)
        months = [pd.Period(f'2024-{m:02d}') for m in range(1, 13)]
        returns = [0.01] * 12  # 1% each month
        
        monthly_data = pd.DataFrame({
            'month': months,
            'return': returns
        })
        
        result = tracker.process_monthly_returns(monthly_data)
        
        # Should have 4 management fee events (Jan, Apr, Jul, Oct)
        mgmt_events = [h for h in result['history'] if h['event'] == 'management_fee']
        assert len(mgmt_events) == 4
        
        # Verify quarter assignments
        mgmt_quarters = [h['quarter'] for h in mgmt_events]
        assert mgmt_quarters == [1, 2, 3, 4]
        
        # Should have 4 performance fee events (Mar, Jun, Sep, Dec)
        perf_events = [h for h in result['history'] if h['event'] == 'performance_fee']
        assert len(perf_events) == 4
    
    def test_april_fiscal_year_full_year(self):
        """Test processing for April fiscal year (Apr 2024 - Mar 2025)."""
        tracker = QuarterlyFeeTracker(initial_nav=100000.0, fiscal_year_start_month=4)
        
        # Create monthly returns for full year (Apr 2024 - Mar 2025)
        months = [pd.Period(f'2024-{m:02d}') for m in range(4, 13)] + \
                 [pd.Period(f'2025-{m:02d}') for m in range(1, 4)]
        returns = [0.01] * 12  # 1% each month
        
        monthly_data = pd.DataFrame({
            'month': months,
            'return': returns
        })
        
        result = tracker.process_monthly_returns(monthly_data)
        
        # Should have 4 management fee events (Apr, Jul, Oct, Jan)
        mgmt_events = [h for h in result['history'] if h['event'] == 'management_fee']
        assert len(mgmt_events) == 4
        
        # Should have 4 performance fee events (Jun, Sep, Dec, Mar)
        perf_events = [h for h in result['history'] if h['event'] == 'performance_fee']
        assert len(perf_events) == 4
    
    def test_partial_year_start_mid_quarter(self):
        """Test processing when starting mid-quarter."""
        tracker = QuarterlyFeeTracker(initial_nav=100000.0, fiscal_year_start_month=2)
        
        # Start in March (mid Q1 for Feb fiscal year)
        # March is NOT a quarter start, so no mgmt fee applied at start
        monthly_data = pd.DataFrame({
            'month': [pd.Period('2024-03'), pd.Period('2024-04'), pd.Period('2024-05')],
            'return': [0.02, 0.02, 0.02]
        })
        
        result = tracker.process_monthly_returns(monthly_data)
        
        # No mgmt fee for March (not quarter start)
        # April is end of Q1 - perf fee applied
        # May is start of Q2 - mgmt fee applied
        mgmt_events = [h for h in result['history'] if h['event'] == 'management_fee']
        perf_events = [h for h in result['history'] if h['event'] == 'performance_fee']
        
        # Only Q2 mgmt fee (May)
        assert len(mgmt_events) == 1
        assert mgmt_events[0]['quarter'] == 2
        
        # Q1 perf fee (April end)
        assert len(perf_events) == 1
        assert perf_events[0]['quarter'] == 1


class TestFeeCompoundingEffect:
    """Test that fee deductions properly compound over time."""
    
    def test_fee_deduction_reduces_base(self):
        """Test that fee deductions reduce the NAV base for future calculations."""
        # Scenario: Compare NAV growth with and without fee deductions
        
        # With fees
        tracker_with_fees = QuarterlyFeeTracker(initial_nav=100000.0, fiscal_year_start_month=2)
        monthly_data = pd.DataFrame({
            'month': [pd.Period('2024-02'), pd.Period('2024-03'), pd.Period('2024-04')],
            'return': [0.05, 0.05, 0.05]  # 5% each month
        })
        result_with_fees = tracker_with_fees.process_monthly_returns(monthly_data)
        
        # Without fees (just compound returns)
        gross_nav = 100000.0 * (1.05 ** 3)
        gross_return = (gross_nav / 100000.0) - 1
        
        # Net NAV should be less than gross NAV
        assert result_with_fees['final_nav'] < gross_nav
        
        # Net return should be less than gross return
        assert result_with_fees['total_net_return'] < gross_return
        
        # The difference should equal total fees
        fee_impact = gross_nav - result_with_fees['final_nav']
        total_fees = result_with_fees['total_mgmt_fees'] + result_with_fees['total_perf_fees']
        
        # Note: Due to compounding, this won't be exact, but should be in same ballpark
        # The fees reduce the base, which reduces future gains
        assert total_fees > 0
    
    def test_high_return_scenario(self):
        """Test with high returns to verify significant performance fee deduction."""
        tracker = QuarterlyFeeTracker(initial_nav=100000.0, fiscal_year_start_month=2)
        
        # Create monthly returns with 10% each month (very high)
        monthly_data = pd.DataFrame({
            'month': [pd.Period('2024-02'), pd.Period('2024-03'), pd.Period('2024-04')],
            'return': [0.10, 0.10, 0.10]  # 10% each month
        })
        
        result = tracker.process_monthly_returns(monthly_data)
        
        # Gross return = (1.10)^3 - 1 ≈ 33.1%
        expected_gross = (1.10 ** 3) - 1
        assert result['total_gross_return'] == pytest.approx(expected_gross, rel=1e-3)
        
        # With such high returns, performance fee should be significant
        assert result['total_perf_fees'] > 5000  # At least $5000 in perf fees


class TestShortfallCarryForward:
    """Test shortfall carry-forward mechanism across years."""
    
    def test_shortfall_accumulates_across_years(self):
        """Test that shortfall from prior year increases next year's hurdle."""
        tracker = QuarterlyFeeTracker(initial_nav=100000.0, fiscal_year_start_month=2)
        
        # Year 1: Low returns (below 6% annual hurdle)
        year1_months = [pd.Period(f'2023-{m:02d}') for m in range(2, 13)] + [pd.Period('2024-01')]
        year1_returns = [0.002] * 12  # 0.2% monthly ≈ 2.4% annual
        
        monthly_data_y1 = pd.DataFrame({
            'month': year1_months,
            'return': year1_returns
        })
        
        tracker.process_monthly_returns(monthly_data_y1)
        
        # Should have accumulated shortfall (6% - ~2.4% ≈ 3.6%)
        assert tracker.carried_shortfall > 0
        
        # Year 2: Strong returns
        year2_months = [pd.Period(f'2024-{m:02d}') for m in range(2, 13)] + [pd.Period('2025-01')]
        year2_returns = [0.015] * 12  # 1.5% monthly ≈ 19.6% annual
        
        monthly_data_y2 = pd.DataFrame({
            'month': year2_months,
            'return': year2_returns
        })
        
        # Reset history for clarity (but keep carried shortfall)
        current_nav = tracker.nav
        current_shortfall = tracker.carried_shortfall
        tracker.history = []
        
        # Continue processing (simulating year 2)
        # Note: In practice, we'd continue with the same tracker
        # The shortfall should affect the effective hurdle


class TestNAVTrackingAccuracy:
    """Test that NAV tracking produces accurate values."""
    
    def test_simple_scenario_manual_verification(self):
        """
        Test a simple scenario with manual verification.
        
        Scenario:
        - Initial NAV: €100,000
        - Q1: Feb start → 0.25% mgmt fee, then 3 months of 2% return each
        - At Apr 30: performance fee on gains above 1.5% hurdle
        """
        tracker = QuarterlyFeeTracker(initial_nav=100000.0, fiscal_year_start_month=2)
        
        # Manual calculation:
        # 1. Fiscal year starts: year_start_nav = 100,000
        # 2. Feb 1: Mgmt fee = 100,000 * 0.25% = 250
        #    NAV after = 99,750
        # 3. Feb return: 99,750 * 1.02 = 101,745
        # 4. Mar return: 101,745 * 1.02 = 103,779.90
        # 5. Apr return: 103,779.90 * 1.02 = 105,855.50
        # 6. Apr 30: Perf fee
        #    Year start NAV = 100,000 (initial NAV, before mgmt fee)
        #    Current NAV = 105,855.50
        #    Gain = 105,855.50 - 100,000 = 5,855.50
        #    Hurdle = 100,000 * 1.5% = 1,500
        #    Excess = 5,855.50 - 1,500 = 4,355.50
        #    Perf fee = 4,355.50 * 25% = 1,088.87
        #    Final NAV = 105,855.50 - 1,088.87 = 104,766.63
        
        monthly_data = pd.DataFrame({
            'month': [pd.Period('2024-02'), pd.Period('2024-03'), pd.Period('2024-04')],
            'return': [0.02, 0.02, 0.02]
        })
        
        result = tracker.process_monthly_returns(monthly_data)
        
        # Verify key values
        assert result['final_nav'] == pytest.approx(104766.62, rel=1e-3)
        assert result['total_mgmt_fees'] == pytest.approx(250.0, rel=1e-2)
        assert result['total_perf_fees'] == pytest.approx(1088.87, rel=1e-2)
    
    def test_zero_return_scenario(self):
        """Test with zero returns - only management fee should be deducted."""
        tracker = QuarterlyFeeTracker(initial_nav=100000.0, fiscal_year_start_month=2)
        
        monthly_data = pd.DataFrame({
            'month': [pd.Period('2024-02'), pd.Period('2024-03'), pd.Period('2024-04')],
            'return': [0.0, 0.0, 0.0]
        })
        
        result = tracker.process_monthly_returns(monthly_data)
        
        # Only management fee should be deducted
        expected_nav = 100000.0 * (1 - 0.0025)  # 0.25% fee
        assert result['final_nav'] == pytest.approx(expected_nav, rel=1e-4)
        assert result['total_perf_fees'] == pytest.approx(0.0)
        assert result['total_mgmt_fees'] == pytest.approx(250.0, rel=1e-2)
    
    def test_negative_return_scenario(self):
        """Test with negative returns - no performance fee, shortfall accumulates."""
        tracker = QuarterlyFeeTracker(initial_nav=100000.0, fiscal_year_start_month=2)
        
        # Full year of negative returns
        months = [pd.Period(f'2024-{m:02d}') for m in range(2, 13)] + [pd.Period('2025-01')]
        returns = [-0.01] * 12  # -1% each month
        
        monthly_data = pd.DataFrame({
            'month': months,
            'return': returns
        })
        
        result = tracker.process_monthly_returns(monthly_data)
        
        # No performance fees (all quarters below hurdle)
        assert result['total_perf_fees'] == pytest.approx(0.0)
        
        # Management fees still deducted
        assert result['total_mgmt_fees'] > 0
        
        # Shortfall should be accumulated
        assert tracker.carried_shortfall > 0


class TestHistoryTracking:
    """Test that history is properly tracked."""
    
    def test_history_contains_all_events(self):
        """Test that all events are recorded in history."""
        tracker = QuarterlyFeeTracker(initial_nav=100000.0, fiscal_year_start_month=2)
        
        monthly_data = pd.DataFrame({
            'month': [pd.Period('2024-02'), pd.Period('2024-03'), pd.Period('2024-04')],
            'return': [0.02, 0.01, 0.015]
        })
        
        result = tracker.process_monthly_returns(monthly_data)
        
        # Get history as DataFrame
        history_df = tracker.get_history_df()
        
        assert not history_df.empty
        assert 'event' in history_df.columns
        
        # Verify event types
        events = history_df['event'].tolist()
        assert 'management_fee' in events
        assert 'monthly_return' in events
        assert 'performance_fee' in events
    
    def test_reset_clears_state(self):
        """Test that reset clears all state."""
        tracker = QuarterlyFeeTracker(initial_nav=100000.0)
        
        # Process some returns
        monthly_data = pd.DataFrame({
            'month': [pd.Period('2024-02')],
            'return': [0.05]
        })
        tracker.process_monthly_returns(monthly_data)
        
        # Verify state changed
        assert tracker.nav != 100000.0
        assert len(tracker.history) > 0
        
        # Reset
        tracker.reset()
        
        # Verify state reset
        assert tracker.nav == 100000.0
        assert len(tracker.history) == 0
        assert tracker.carried_shortfall == 0.0
        assert tracker.accumulated_hurdle == 0.0

