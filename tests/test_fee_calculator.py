"""Tests for the fee calculator module."""

import os
import sys
import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, PROJECT_ROOT)

from fee_calculator import (
    annual_to_monthly_fee,
    calculate_net_return,
    calculate_monthly_net_returns,
    PerformanceFeeCalculator,
    FeeCalculator,
    calculate_fees_for_period,
)


class TestAnnualToMonthlyFee:
    """Tests for annual to monthly fee conversion."""
    
    def test_one_percent_annual(self):
        """Test 1% annual fee conversion."""
        monthly = annual_to_monthly_fee(0.01)
        # Compound monthly back to annual
        annual_check = (1 + monthly) ** 12 - 1
        assert abs(annual_check - 0.01) < 0.0001
    
    def test_twelve_percent_annual(self):
        """Test 12% annual fee conversion."""
        monthly = annual_to_monthly_fee(0.12)
        annual_check = (1 + monthly) ** 12 - 1
        assert abs(annual_check - 0.12) < 0.0001
    
    def test_zero_fee(self):
        """Test zero fee conversion."""
        monthly = annual_to_monthly_fee(0.0)
        assert monthly == 0.0
    
    def test_monthly_less_than_simple_division(self):
        """Geometric monthly should be less than simple division."""
        annual = 0.12
        monthly = annual_to_monthly_fee(annual)
        simple_monthly = annual / 12
        assert monthly < simple_monthly


class TestCalculateNetReturn:
    """Tests for net return calculation."""
    
    def test_positive_gross_return(self):
        """Test net return with positive gross return."""
        gross = 0.05  # 5% gross
        fee = 0.01    # 1% fee
        net = calculate_net_return(gross, fee)
        assert net == 0.04
    
    def test_gross_equals_fee(self):
        """Test net return when gross equals fee."""
        gross = 0.01
        fee = 0.01
        net = calculate_net_return(gross, fee)
        assert net == 0.0
    
    def test_negative_gross_return(self):
        """Test net return with negative gross return."""
        gross = -0.02  # -2% gross
        fee = 0.01     # 1% fee
        net = calculate_net_return(gross, fee)
        assert net == -0.03


class TestPerformanceFeeCalculator:
    """Tests for the performance fee calculator with carry-forward shortfall."""
    
    def test_above_hurdle_no_shortfall(self):
        """Test fee charged when return exceeds hurdle with no prior shortfall."""
        calc = PerformanceFeeCalculator(hurdle_rate=0.06, fee_rate=0.25)
        
        # Return of 10%, hurdle 6% -> excess 4% -> fee 1%
        fee, shortfall = calc.calculate_annual_fee(0.10)
        
        assert abs(fee - 0.01) < 0.0001  # 25% of 4% = 1%
        assert shortfall == 0.0
    
    def test_below_hurdle_creates_shortfall(self):
        """Test that return below hurdle creates shortfall, no fee."""
        calc = PerformanceFeeCalculator(hurdle_rate=0.06, fee_rate=0.25)
        
        # Return of 4%, hurdle 6% -> shortfall 2%, no fee
        fee, shortfall = calc.calculate_annual_fee(0.04)
        
        assert fee == 0.0
        assert abs(shortfall - 0.02) < 0.0001
    
    def test_exactly_at_hurdle(self):
        """Test no fee and no shortfall when return equals hurdle."""
        calc = PerformanceFeeCalculator(hurdle_rate=0.06, fee_rate=0.25)
        
        fee, shortfall = calc.calculate_annual_fee(0.06)
        
        assert fee == 0.0
        assert shortfall == 0.0
    
    def test_shortfall_carries_forward(self):
        """Test that shortfall carries forward to next year."""
        calc = PerformanceFeeCalculator(hurdle_rate=0.06, fee_rate=0.25)
        
        # Year 1: 4% return, 6% hurdle -> 2% shortfall
        fee1, shortfall1 = calc.calculate_annual_fee(0.04, year=2022)
        assert fee1 == pytest.approx(0.0, abs=1e-10)
        assert shortfall1 == pytest.approx(0.02, abs=0.0001)
        
        # Year 2: 7% return, effective hurdle = 6% + 2% = 8%
        # Since 7% < 8% effective hurdle, no fee
        # Since 7% >= 6% base hurdle, no additional shortfall
        fee2, shortfall2 = calc.calculate_annual_fee(0.07, year=2023)
        assert fee2 == pytest.approx(0.0, abs=1e-10)
        # Shortfall should remain at 0.02 (not increased, not cleared)
        assert shortfall2 == pytest.approx(0.02, abs=0.0001)
    
    def test_excess_over_effective_hurdle(self):
        """Test fee on excess over effective hurdle (base + shortfall)."""
        calc = PerformanceFeeCalculator(hurdle_rate=0.06, fee_rate=0.25)
        
        # Year 1: 4% return -> 2% shortfall
        calc.calculate_annual_fee(0.04, year=2022)
        
        # Year 2: 12% return, effective hurdle = 8% -> excess 4% -> fee 1%
        fee, shortfall = calc.calculate_annual_fee(0.12, year=2023)
        
        assert abs(fee - 0.01) < 0.0001  # 25% of 4%
        assert shortfall == 0.0  # Shortfall cleared
    
    def test_shortfall_accumulates(self):
        """Test that shortfall accumulates over multiple years."""
        calc = PerformanceFeeCalculator(hurdle_rate=0.06, fee_rate=0.25)
        
        # Year 1: 4% return -> 2% shortfall
        calc.calculate_annual_fee(0.04, year=2022)
        
        # Year 2: 3% return -> additional 3% shortfall
        fee2, shortfall2 = calc.calculate_annual_fee(0.03, year=2023)
        
        assert fee2 == 0.0
        assert abs(shortfall2 - 0.05) < 0.0001  # 2% + 3% = 5%
    
    def test_negative_return(self):
        """Test negative return creates maximum shortfall."""
        calc = PerformanceFeeCalculator(hurdle_rate=0.06, fee_rate=0.25)
        
        fee, shortfall = calc.calculate_annual_fee(-0.10)  # -10% return
        
        assert fee == 0.0
        # Shortfall = 6% - (-10%) = 16%? No, shortfall is just 6% (hurdle - return capped at hurdle)
        # Actually based on logic: shortfall_this_year = max(0, hurdle - return) = max(0, 0.06 - (-0.10)) = 0.16
        assert abs(shortfall - 0.16) < 0.0001
    
    def test_history_tracking(self):
        """Test that history is tracked correctly."""
        calc = PerformanceFeeCalculator(hurdle_rate=0.06, fee_rate=0.25)
        
        calc.calculate_annual_fee(0.04, year=2022)
        calc.calculate_annual_fee(0.10, year=2023)
        
        history = calc.get_history_df()
        
        assert len(history) == 2
        assert list(history['year']) == [2022, 2023]
    
    def test_reset(self):
        """Test that reset clears state."""
        calc = PerformanceFeeCalculator(hurdle_rate=0.06, fee_rate=0.25)
        
        calc.calculate_annual_fee(0.04)
        calc.reset()
        
        assert calc.carried_shortfall == 0.0
        assert len(calc.history) == 0


class TestFeeCalculator:
    """Tests for the combined fee calculator."""
    
    def test_monthly_mgmt_fee_property(self):
        """Test monthly management fee calculation."""
        calc = FeeCalculator(annual_mgmt_fee=0.01)
        monthly = calc.monthly_mgmt_fee
        assert abs((1 + monthly) ** 12 - 1.01) < 0.0001
    
    def test_calculate_monthly_fees(self):
        """Test monthly fee calculation with DataFrame."""
        calc = FeeCalculator(annual_mgmt_fee=0.01)
        
        months = pd.period_range('2024-01', periods=3, freq='M')
        monthly_returns = pd.DataFrame({
            'month': months,
            'return': [0.02, 0.01, 0.03]
        })
        
        result = calc.calculate_monthly_fees(monthly_returns)
        
        assert 'gross_return' in result.columns
        assert 'net_before_perf' in result.columns
        assert 'management_fee' in result.columns
        assert all(result['net_before_perf'] < result['gross_return'])


class TestCalculateFeesForPeriod:
    """Tests for period fee calculation."""
    
    def test_empty_returns(self):
        """Test with empty returns DataFrame."""
        empty_df = pd.DataFrame(columns=['month', 'return'])
        result = calculate_fees_for_period(empty_df)
        
        assert result['total_gross_return'] == 0.0
        assert result['total_net_return'] == 0.0
    
    def test_with_returns(self):
        """Test with actual returns."""
        months = pd.period_range('2024-01', periods=6, freq='M')
        returns = pd.DataFrame({
            'month': months,
            'return': [0.01, 0.02, 0.01, 0.015, 0.02, 0.01]
        })
        
        result = calculate_fees_for_period(returns)
        
        assert result['total_gross_return'] > 0
        assert result['total_net_return'] < result['total_gross_return']
        assert 'monthly_breakdown' in result

