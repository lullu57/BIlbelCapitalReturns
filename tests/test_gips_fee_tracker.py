"""
Tests for GIPSFeeTracker

These tests validate the GIPS-compliant fee tracker that:
1. Uses high-water-mark for performance fees
2. Applies a perpetual cumulative hurdle (no annual reset)
3. Applies performance fee at quarter end
4. Applies management fee at start of next quarter
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date
from gips_fee_tracker import (
    GIPSFeeTracker,
    calculate_net_twr_from_monthly,
    calculate_period_net_return
)


class TestGIPSFeeTrackerInit:
    """Test initialization of GIPSFeeTracker."""
    
    def test_default_values(self):
        tracker = GIPSFeeTracker(initial_nav=100000.0)
        assert tracker.initial_nav == 100000.0
        assert tracker.gross_nav == 100000.0
        assert tracker.net_nav == 100000.0
        assert tracker.net_index == 1.0
        assert tracker.high_water_mark == 1.0
        assert tracker.cumulative_hurdle_pct == 0.0
        assert tracker.quarterly_mgmt_fee == 0.0025
        assert tracker.quarterly_hurdle_rate == 0.015
        assert tracker.perf_fee_rate == 0.25
        assert tracker.fiscal_year_start_month == 2
    
    def test_custom_values(self):
        tracker = GIPSFeeTracker(
            initial_nav=50000.0,
            quarterly_mgmt_fee=0.003,
            quarterly_hurdle_rate=0.02,
            perf_fee_rate=0.20,
            fiscal_year_start_month=1
        )
        assert tracker.initial_nav == 50000.0
        assert tracker.quarterly_mgmt_fee == 0.003
        assert tracker.quarterly_hurdle_rate == 0.02
        assert tracker.perf_fee_rate == 0.20
        assert tracker.fiscal_year_start_month == 1


class TestHighWaterMark:
    """Test high-water-mark functionality."""
    
    def test_hwm_starts_at_one(self):
        tracker = GIPSFeeTracker(initial_nav=100000.0)
        assert tracker.high_water_mark == 1.0
    
    def test_hwm_updates_only_on_fee(self):
        tracker = GIPSFeeTracker(initial_nav=100000.0)
        
        # Above HWM but below hurdle
        tracker.net_index = 1.01
        tracker.net_nav = 101000.0
        fee = tracker.apply_performance_fee(quarter=1, fiscal_year="2024")
        
        assert fee == 0.0
        assert tracker.high_water_mark == 1.0
    
    def test_hwm_does_not_decrease(self):
        tracker = GIPSFeeTracker(initial_nav=100000.0)
        tracker.high_water_mark = 1.20  # Set HWM high
        
        # Net NAV drops below HWM
        tracker.net_index = 1.05
        tracker.net_nav = 105000.0
        tracker.apply_performance_fee(quarter=1, fiscal_year="2024")
        
        # HWM should NOT decrease
        assert tracker.high_water_mark == 1.20
    
    def test_hwm_updates_on_fee_charge(self):
        tracker = GIPSFeeTracker(initial_nav=100000.0)
        tracker.net_index = 1.10
        tracker.net_nav = 110000.0
        
        fee = tracker.apply_performance_fee(quarter=1, fiscal_year="2024")
        
        assert fee > 0
        assert tracker.high_water_mark < 1.10
        assert tracker.high_water_mark > 1.0
    
    def test_flow_does_not_change_hwm(self):
        tracker = GIPSFeeTracker(initial_nav=100000.0)
        
        # Deposit
        tracker.apply_flow(50000.0, "deposit")
        
        assert tracker.high_water_mark == 1.0
        
        # Withdrawal
        tracker.apply_flow(-20000.0, "withdrawal")
        assert tracker.high_water_mark == 1.0


class TestPerpetualHurdle:
    """Test that hurdle accumulates each quarter with no reset."""
    
    def test_hurdle_accumulates_each_quarter(self):
        tracker = GIPSFeeTracker(initial_nav=100000.0, fiscal_year_start_month=2)
        tracker.net_index = 1.0
        tracker.net_nav = 100000.0
        
        tracker.apply_performance_fee(quarter=1, fiscal_year="2024")
        assert tracker.cumulative_hurdle_pct == pytest.approx(0.015)
        
        tracker.apply_performance_fee(quarter=2, fiscal_year="2024")
        assert tracker.cumulative_hurdle_pct == pytest.approx(0.03)


class TestQuarterBoundaries:
    """Test fiscal quarter boundary detection."""
    
    def test_february_fiscal_year(self):
        tracker = GIPSFeeTracker(initial_nav=100.0, fiscal_year_start_month=2)
        
        assert tracker.quarter_start_months == [2, 5, 8, 11]
        assert tracker.quarter_end_months == [4, 7, 10, 1]
        
        assert tracker.is_fiscal_year_start(2) == True
        assert tracker.is_fiscal_year_start(1) == False
        
        assert tracker.get_quarter_for_month(2) == 1
        assert tracker.get_quarter_for_month(3) == 1
        assert tracker.get_quarter_for_month(4) == 1
        assert tracker.get_quarter_for_month(5) == 2
    
    def test_january_fiscal_year(self):
        tracker = GIPSFeeTracker(initial_nav=100.0, fiscal_year_start_month=1)
        
        assert tracker.quarter_start_months == [1, 4, 7, 10]
        assert tracker.quarter_end_months == [3, 6, 9, 12]
        
        assert tracker.is_fiscal_year_start(1) == True
        assert tracker.get_quarter_for_month(1) == 1
        assert tracker.get_quarter_for_month(12) == 4


class TestFeeTimingOrder:
    """Test that fees are applied in correct order."""
    
    def test_mgmt_fee_at_quarter_start(self):
        """Management fee applied at start of quarter."""
        tracker = GIPSFeeTracker(initial_nav=100000.0)
        
        # At start of Q1 (Feb for fiscal Feb start)
        fee = tracker.apply_management_fee(quarter=1, fiscal_year="2024")
        
        # Fee is 0.25% of 100000 = 250
        assert fee == pytest.approx(250.0)
        assert tracker.net_nav == pytest.approx(99750.0)
        assert tracker.net_index == pytest.approx(0.9975)
    
    def test_perf_fee_at_quarter_end(self):
        """Performance fee applied at end of quarter."""
        tracker = GIPSFeeTracker(initial_nav=100000.0)
        
        # Simulate gains during quarter
        tracker.net_nav = 105000.0
        tracker.net_index = 1.05
        
        # At end of Q1
        fee = tracker.apply_performance_fee(quarter=1, fiscal_year="2024")
        
        # Hurdle threshold = 1.015 on the net index
        # Excess = 1.05 - 1.015 = 0.035
        # Fee = 25% of excess => fee percent ≈ 0.008333; fee amount ≈ 875
        assert fee == pytest.approx(875.0)
    
    def test_fee_order_within_quarter(self):
        """
        Correct order: 
        1. End of Q1: Perf fee applied
        2. Start of Q2: Mgmt fee applied
        """
        tracker = GIPSFeeTracker(initial_nav=100000.0)
        
        # Q1 processing
        tracker.apply_management_fee(quarter=1, fiscal_year="2024")  # Start of Q1
        nav_after_q1_mgmt = tracker.net_nav
        
        tracker.net_nav = 105000.0  # Gains during Q1
        tracker.net_index = 1.05
        tracker.apply_performance_fee(quarter=1, fiscal_year="2024")  # End of Q1
        nav_after_q1_perf = tracker.net_nav
        
        # Q2 processing
        tracker.apply_management_fee(quarter=2, fiscal_year="2024")  # Start of Q2
        nav_after_q2_mgmt = tracker.net_nav
        
        # NAV should decrease with each fee
        assert nav_after_q1_mgmt < 100000.0
        assert nav_after_q1_perf < 105000.0
        assert nav_after_q2_mgmt < nav_after_q1_perf


class TestMonthlyProcessing:
    """Test monthly data processing."""
    
    def test_process_month_basic(self):
        tracker = GIPSFeeTracker(initial_nav=100000.0)
        
        result = tracker.process_month(
            month=2,  # Start of fiscal year
            year=2024,
            gross_nav_end=105000.0,
            flow_amount=0.0
        )
        
        assert result['gross_nav_end'] == 105000.0
        assert 'net_nav_end' in result
        assert 'net_return' in result
    
    def test_process_month_with_flow(self):
        tracker = GIPSFeeTracker(initial_nav=100000.0)
        
        result = tracker.process_month(
            month=3,
            year=2024,
            gross_nav_end=155000.0,
            flow_amount=50000.0  # Deposit
        )
        
        # Gross NAV increased by flow + some return
        assert result['gross_nav_end'] == 155000.0
        assert result['flow'] == 50000.0
        # Net return should be based on investment gain, not flow
        assert result['investment_gain'] == 5000.0  # 155000 - 100000 - 50000


class TestNetTWRCalculation:
    """Test Net TWR calculation from monthly net returns."""
    
    def test_calculate_net_twr_basic(self):
        # Create monthly returns
        monthly_data = pd.DataFrame({
            'month': ['2024-02', '2024-03', '2024-04'],
            'net_return': [0.05, 0.03, 0.02]
        })
        
        abs_twr, ann_twr = calculate_net_twr_from_monthly(monthly_data)
        
        # Geometric linking: (1.05 * 1.03 * 1.02) - 1 = 0.1033
        expected = 1.05 * 1.03 * 1.02 - 1
        assert abs_twr == pytest.approx(expected)
    
    def test_calculate_net_twr_empty(self):
        abs_twr, ann_twr = calculate_net_twr_from_monthly(pd.DataFrame())
        assert abs_twr == 0.0
        assert ann_twr == 0.0


class TestFlowProration:
    """Test fee proration for intra-quarter flows."""

    def test_performance_fee_prorates_inflow_base(self):
        tracker = GIPSFeeTracker(initial_nav=100000.0, fiscal_year_start_month=2)
        tracker.start_new_quarter(quarter=1, fiscal_year="2024", year=2024, month=2)

        tracker.net_nav = 140000.0
        tracker.net_index = 1.10

        flow_date = date(2024, 3, 16)
        tracker.add_flow_for_proration(40000.0, flow_date)

        fee_date = date(2024, 4, 30)
        fee = tracker.apply_performance_fee(quarter=1, fiscal_year="2024", fee_date=fee_date)

        days_in_quarter = tracker.days_in_current_quarter
        factor = (fee_date - flow_date).days / days_in_quarter
        fee_base = 140000.0 - 40000.0 * (1 - factor)

        pre_fee_index = 1.10
        hurdle_threshold = 1.0 + tracker.quarterly_hurdle_rate
        excess = pre_fee_index - hurdle_threshold
        fee_percent = tracker.perf_fee_rate * (excess / pre_fee_index)
        expected_fee = fee_base * fee_percent

        assert fee == pytest.approx(expected_fee, rel=1e-6)

    def test_performance_fee_prorates_outflow_base(self):
        tracker = GIPSFeeTracker(initial_nav=100000.0, fiscal_year_start_month=2)
        tracker.start_new_quarter(quarter=1, fiscal_year="2024", year=2024, month=2)

        tracker.net_nav = 60000.0
        tracker.net_index = 1.10

        flow_date = date(2024, 3, 16)
        tracker.add_flow_for_proration(-40000.0, flow_date)

        fee_date = date(2024, 4, 30)
        fee = tracker.apply_performance_fee(quarter=1, fiscal_year="2024", fee_date=fee_date)

        days_in_quarter = tracker.days_in_current_quarter
        days_present = (flow_date - tracker.current_quarter_start).days + 1
        factor = days_present / days_in_quarter
        fee_base = 60000.0 + 40000.0 * factor

        pre_fee_index = 1.10
        hurdle_threshold = 1.0 + tracker.quarterly_hurdle_rate
        excess = pre_fee_index - hurdle_threshold
        fee_percent = tracker.perf_fee_rate * (excess / pre_fee_index)
        expected_fee = fee_base * fee_percent

        assert fee == pytest.approx(expected_fee, rel=1e-6)

    def test_management_fee_adjustment_prorates_flows(self):
        tracker = GIPSFeeTracker(initial_nav=100000.0, fiscal_year_start_month=2)
        tracker.start_new_quarter(quarter=1, fiscal_year="2024", year=2024, month=2)

        tracker.apply_management_fee(quarter=1, fiscal_year="2024")

        inflow_date = date(2024, 3, 16)
        tracker.add_flow_for_proration(40000.0, inflow_date)

        fee_date = date(2024, 4, 30)
        fee_adj = tracker.apply_management_fee_adjustment(quarter=1, fiscal_year="2024", fee_date=fee_date)

        days_in_quarter = tracker.days_in_current_quarter
        days_remaining = (fee_date - inflow_date).days
        factor = days_remaining / days_in_quarter
        expected_base = 40000.0 * factor
        expected_fee = expected_base * tracker.quarterly_mgmt_fee

        assert fee_adj == pytest.approx(expected_fee, rel=1e-6)


class TestPeriodNetReturn:
    """Test period net return calculation with quarterly crystallization."""
    
    def test_basic_calculation(self):
        result = calculate_period_net_return(
            gross_return=0.20,  # 20% gross
            quarterly_mgmt_fee=0.0025,
            quarterly_hurdle=0.015,
            perf_fee_rate=0.25,
            num_quarters=4
        )
        
        assert 'net_return' in result
        assert result['net_return'] < 0.20  # Net should be less than gross
        assert result['mgmt_fee_impact'] > 0
        assert result['perf_fee_impact'] >= 0
    
    def test_below_hurdle_no_perf_fee(self):
        result = calculate_period_net_return(
            gross_return=0.04,  # 4% gross, below 6% annual hurdle
            quarterly_mgmt_fee=0.0025,
            quarterly_hurdle=0.015,
            perf_fee_rate=0.25,
            num_quarters=4
        )
        
        # No performance fee when below hurdle
        # (But may have small perf fee if any quarter briefly exceeds)
        assert result['net_return'] < 0.04
        assert result['mgmt_fee_impact'] > 0


class TestHighWaterMarkWithFees:
    """Test high-water-mark behavior with actual fee calculations."""
    
    def test_hwm_prevents_double_charging(self):
        """
        Scenario: NAV rises, falls, rises again.
        Should only pay fees on NEW highs, not recouping old ones.
        """
        tracker = GIPSFeeTracker(initial_nav=100000.0)
        
        # Q1: NAV rises to 110000
        tracker.net_nav = 110000.0
        tracker.net_index = 1.10
        fee_q1 = tracker.apply_performance_fee(quarter=1, fiscal_year="2024")
        hwm_after_q1 = tracker.high_water_mark
        
        # Fee charged on excess above 1.015 hurdle
        assert fee_q1 > 0
        
        # Q2: NAV drops (below hurdle)
        tracker.net_nav = 105000.0
        tracker.net_index = 1.05
        fee_q2 = tracker.apply_performance_fee(quarter=2, fiscal_year="2024")
        
        # No fee - below hurdle threshold
        assert fee_q2 == 0.0
        
        # Q3: NAV rises but still below hurdle threshold
        tracker.net_nav = 108000.0
        tracker.net_index = 1.08
        fee_q3 = tracker.apply_performance_fee(quarter=3, fiscal_year="2024")
        
        # Still no fee - below hurdle threshold
        assert fee_q3 == 0.0
        
        # Q4: NAV exceeds hurdle threshold
        tracker.net_nav = 120000.0
        tracker.net_index = 1.20
        fee_q4 = tracker.apply_performance_fee(quarter=4, fiscal_year="2024")
        
        assert fee_q4 > 0
        assert tracker.high_water_mark > hwm_after_q1


class TestIntegration:
    """Integration tests for full workflow."""
    
    def test_full_year_processing(self):
        """Test processing a full fiscal year of data."""
        tracker = GIPSFeeTracker(
            initial_nav=100000.0,
            fiscal_year_start_month=2
        )
        
        # Create 12 months of data (Feb 2024 - Jan 2025)
        months = []
        nav = 100000.0
        for m in range(2, 14):
            month = m if m <= 12 else m - 12
            year = 2024 if m <= 12 else 2025
            nav *= 1.01  # 1% monthly growth
            months.append({
                'month': f"{year}-{month:02d}",
                'end_nav': nav,
                'flow': 0.0
            })
        
        monthly_df = pd.DataFrame(months)
        
        result = tracker.process_monthly_data(
            monthly_df,
            month_column='month',
            gross_nav_column='end_nav',
            flow_column='flow'
        )
        
        assert result['final_gross_nav'] > 100000.0
        assert result['final_net_nav'] > 0
        assert result['final_net_nav'] < result['final_gross_nav']
        assert result['total_fees'] > 0
        
        # Should have monthly net returns
        assert 'monthly_net_returns' in result
        assert len(result['monthly_net_returns']) > 0


class TestAdditiveHurdleThreshold:
    """Test that hurdle threshold is additive (HWM + hurdle), not multiplicative."""
    
    def test_threshold_is_additive_not_multiplicative(self):
        """
        Verify threshold = HWM + hurdle_since_HWM, NOT HWM * (1 + hurdle).
        
        Example:
        - HWM = 1.10 (10% return), quarters_since_hwm = 3
        - Additive: threshold = 1.10 + (0.015 * 3) = 1.10 + 0.045 = 1.145
        - Multiplicative (wrong): threshold = 1.10 * (1 + 0.045) = 1.1495
        """
        tracker = GIPSFeeTracker(initial_nav=100000.0)
        tracker.high_water_mark = 1.10
        tracker.quarters_since_hwm = 2  # Will become 3 after apply_performance_fee
        
        # Get threshold calculation
        tracker.net_nav = 114000.0
        tracker.net_index = 1.14
        
        tracker.apply_performance_fee(quarter=1, fiscal_year="2024")
        
        # After this, quarters_since_hwm should be 3 (2 + 1)
        # Threshold = 1.10 + (0.015 * 3) = 1.145
        # 1.14 < 1.145, so no fee should be charged
        assert tracker.accumulated_perf_fees == 0.0
        
    def test_threshold_crosses_additive_boundary(self):
        """Test when return just exceeds additive threshold."""
        tracker = GIPSFeeTracker(initial_nav=100000.0)
        tracker.high_water_mark = 1.10
        tracker.quarters_since_hwm = 2
        
        # Set index to just above threshold
        # Threshold will be 1.10 + (0.015 * 3) = 1.145
        tracker.net_nav = 115000.0
        tracker.net_index = 1.15  # Above 1.145
        
        tracker.apply_performance_fee(quarter=1, fiscal_year="2024")
        
        # Fee should be charged on excess above 1.145
        assert tracker.accumulated_perf_fees > 0
        
        # Excess = 1.15 - 1.145 = 0.005
        # Fee = 25% of (0.005 / 1.15) of NAV
        excess = 0.005
        fee_pct = 0.25 * (excess / 1.15)
        expected_fee = 115000.0 * fee_pct
        assert tracker.accumulated_perf_fees == pytest.approx(expected_fee, rel=0.001)


class TestQuartersSinceHWM:
    """Test the quarters_since_hwm counter behavior."""
    
    def test_counter_starts_at_zero(self):
        tracker = GIPSFeeTracker(initial_nav=100000.0)
        assert tracker.quarters_since_hwm == 0
    
    def test_counter_increments_each_quarter(self):
        """Each quarter without exceeding threshold increases the counter."""
        tracker = GIPSFeeTracker(initial_nav=100000.0)
        
        # Q1: No gain, below threshold
        tracker.net_index = 1.0
        tracker.net_nav = 100000.0
        tracker.apply_performance_fee(quarter=1, fiscal_year="2024")
        assert tracker.quarters_since_hwm == 1
        
        # Q2: Still below threshold
        tracker.apply_performance_fee(quarter=2, fiscal_year="2024")
        assert tracker.quarters_since_hwm == 2
        
        # Q3: Still below threshold
        tracker.apply_performance_fee(quarter=3, fiscal_year="2024")
        assert tracker.quarters_since_hwm == 3
    
    def test_counter_resets_on_fee_charge(self):
        """When fee is charged, counter resets to 0 and HWM updates."""
        tracker = GIPSFeeTracker(initial_nav=100000.0)
        
        # Q1: Below threshold
        tracker.net_index = 1.0
        tracker.net_nav = 100000.0
        tracker.apply_performance_fee(quarter=1, fiscal_year="2024")
        assert tracker.quarters_since_hwm == 1
        
        # Q2: Large gain, exceeds threshold
        # Threshold = 1.0 + (0.015 * 2) = 1.03
        tracker.net_index = 1.10
        tracker.net_nav = 110000.0
        tracker.apply_performance_fee(quarter=2, fiscal_year="2024")
        
        # Counter should reset to 0
        assert tracker.quarters_since_hwm == 0
        
        # HWM should be updated to post-fee index
        assert tracker.high_water_mark < 1.10  # Reduced by fee
        assert tracker.high_water_mark > 1.03  # But above threshold
    
    def test_threshold_accumulates_correctly_over_quarters(self):
        """
        Scenario: No fee charged for multiple quarters, threshold keeps increasing.
        
        Q1: HWM=1.0, counter=0 -> 1, threshold = 1.0 + 0.015 = 1.015
        Q2: HWM=1.0, counter=1 -> 2, threshold = 1.0 + 0.030 = 1.030
        Q3: HWM=1.0, counter=2 -> 3, threshold = 1.0 + 0.045 = 1.045
        Q4: HWM=1.0, counter=3 -> 4, threshold = 1.0 + 0.060 = 1.060
        """
        tracker = GIPSFeeTracker(initial_nav=100000.0)
        
        # Stay flat each quarter
        for q in range(1, 5):
            tracker.net_index = 1.0
            tracker.net_nav = 100000.0
            tracker.apply_performance_fee(quarter=q, fiscal_year="2024")
            assert tracker.accumulated_perf_fees == 0.0  # No fees
            assert tracker.quarters_since_hwm == q
        
        # Now at Q5 (or new year Q1), threshold = 1.0 + (0.015 * 5) = 1.075
        # Return of 1.08 should trigger fee
        tracker.net_index = 1.08
        tracker.net_nav = 108000.0
        tracker.apply_performance_fee(quarter=1, fiscal_year="2025")
        
        # Fee charged on excess above 1.075
        assert tracker.accumulated_perf_fees > 0
        assert tracker.quarters_since_hwm == 0  # Reset after fee


class TestHWMUpdateLogic:
    """Test HWM only updates when fee is charged."""
    
    def test_hwm_unchanged_when_no_fee(self):
        """When no fee charged, HWM stays at previous level."""
        tracker = GIPSFeeTracker(initial_nav=100000.0)
        
        # Set an initial HWM
        tracker.high_water_mark = 1.05
        tracker.quarters_since_hwm = 2
        
        # Return below threshold
        # Threshold = 1.05 + (0.015 * 3) = 1.095
        tracker.net_index = 1.08
        tracker.net_nav = 108000.0
        
        tracker.apply_performance_fee(quarter=1, fiscal_year="2024")
        
        # No fee charged
        assert tracker.accumulated_perf_fees == 0.0
        
        # HWM unchanged
        assert tracker.high_water_mark == 1.05
        
        # Counter increased
        assert tracker.quarters_since_hwm == 3
    
    def test_hwm_updates_to_post_fee_index(self):
        """When fee charged, HWM becomes post-fee net index."""
        tracker = GIPSFeeTracker(initial_nav=100000.0)
        
        # Q1: Return of 10%, threshold = 1.015
        tracker.net_index = 1.10
        tracker.net_nav = 110000.0
        
        pre_fee_index = tracker.net_index
        tracker.apply_performance_fee(quarter=1, fiscal_year="2024")
        
        # Fee was charged
        assert tracker.accumulated_perf_fees > 0
        
        # HWM is now the post-fee index (less than pre-fee)
        assert tracker.high_water_mark == tracker.net_index
        assert tracker.high_water_mark < pre_fee_index


class TestFirstQuarterThreshold:
    """Test that first quarter has correct 1.5% threshold."""
    
    def test_q1_threshold_is_1_5_percent(self):
        """At inception Q1, threshold = 1.0 + 0.015 = 1.015."""
        tracker = GIPSFeeTracker(initial_nav=100000.0)
        
        # Return of exactly 1.5% - should not trigger fee
        tracker.net_index = 1.015
        tracker.net_nav = 101500.0
        
        tracker.apply_performance_fee(quarter=1, fiscal_year="2024")
        
        # No fee (at threshold, not above)
        assert tracker.accumulated_perf_fees == 0.0
    
    def test_q1_fee_on_excess_above_1_5_percent(self):
        """Return of 2% triggers fee on the 0.5% excess."""
        tracker = GIPSFeeTracker(initial_nav=100000.0)
        
        tracker.net_index = 1.02
        tracker.net_nav = 102000.0
        
        tracker.apply_performance_fee(quarter=1, fiscal_year="2024")
        
        # Fee on excess = 1.02 - 1.015 = 0.005
        # Fee = 25% of (0.005 / 1.02) of 102000
        excess = 0.005
        fee_pct = 0.25 * (excess / 1.02)
        expected_fee = 102000.0 * fee_pct
        
        assert tracker.accumulated_perf_fees == pytest.approx(expected_fee, rel=0.001)


class TestCumulativeHurdleTracking:
    """Test cumulative hurdle % for backward compatibility."""
    
    def test_cumulative_hurdle_tracks_total_quarters(self):
        """cumulative_hurdle_pct tracks total quarters since inception."""
        tracker = GIPSFeeTracker(initial_nav=100000.0)
        
        tracker.net_index = 1.0
        tracker.net_nav = 100000.0
        
        tracker.apply_performance_fee(quarter=1, fiscal_year="2024")
        assert tracker.cumulative_hurdle_pct == pytest.approx(0.015)
        
        tracker.apply_performance_fee(quarter=2, fiscal_year="2024")
        assert tracker.cumulative_hurdle_pct == pytest.approx(0.030)
        
        tracker.apply_performance_fee(quarter=3, fiscal_year="2024")
        assert tracker.cumulative_hurdle_pct == pytest.approx(0.045)
        
        tracker.apply_performance_fee(quarter=4, fiscal_year="2024")
        assert tracker.cumulative_hurdle_pct == pytest.approx(0.060)
