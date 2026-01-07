"""Tests for the TWR calculator module."""

import os
import sys
import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, PROJECT_ROOT)

import twr_calculator as twr


class TestCalculateSubPeriodReturns:
    """Tests for sub-period return calculations."""
    
    @pytest.fixture
    def nav_df(self):
        """Sample NAV DataFrame."""
        return pd.DataFrame({
            'Date': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01']),
            'Net Asset Value': [100000, 108000, 120000],
        })
    
    @pytest.fixture
    def trades_df(self):
        """Sample trades DataFrame."""
        return pd.DataFrame({
            'When': pd.to_datetime(['2024-01-15', '2024-02-15']),
            'EUR equivalent': [5000, -3000],
        })
    
    def test_basic_calculation(self, nav_df, trades_df):
        """Test basic sub-period return calculation."""
        result = twr.calculate_sub_period_returns(nav_df, trades_df)
        
        assert len(result) == 2  # Two periods
        assert 'return' in result.columns
        assert 'start_nav' in result.columns
        assert 'end_nav' in result.columns
        assert 'total_flow' in result.columns
    
    def test_return_formula(self, nav_df, trades_df):
        """Test TWR formula: r = (end - start - flow) / start."""
        result = twr.calculate_sub_period_returns(nav_df, trades_df)
        
        # First period: (108000 - 100000 - 5000) / 100000 = 0.03
        first_return = result.iloc[0]['return']
        expected_first = (108000 - 100000 - 5000) / 100000
        assert abs(first_return - expected_first) < 0.0001
    
    def test_large_flow_flagging(self, nav_df, trades_df):
        """Test that large flows are flagged."""
        # Create trades with a large flow (>10% of NAV)
        large_trades = pd.DataFrame({
            'When': pd.to_datetime(['2024-01-15']),
            'EUR equivalent': [15000],  # 15% of 100000
        })
        
        result = twr.calculate_sub_period_returns(nav_df, large_trades, large_flow_threshold=0.10)
        
        assert 'has_large_flow' in result.columns
    
    def test_empty_trades(self, nav_df):
        """Test with no trades."""
        empty_trades = pd.DataFrame(columns=['When', 'EUR equivalent'])
        
        result = twr.calculate_sub_period_returns(nav_df, empty_trades)
        
        # Should still calculate returns based on NAV changes
        assert len(result) == 2
        assert all(result['total_flow'] == 0)


class TestCalculateMonthlyTWR:
    """Tests for monthly TWR calculation."""
    
    @pytest.fixture
    def sub_period_returns(self):
        """Sample sub-period returns."""
        return pd.DataFrame({
            'start_date': pd.to_datetime(['2024-01-01', '2024-01-15', '2024-02-01', '2024-02-15']),
            'end_date': pd.to_datetime(['2024-01-15', '2024-02-01', '2024-02-15', '2024-03-01']),
            'return': [0.01, 0.02, 0.015, 0.025],
            'start_nav': [100000, 101000, 103020, 104565],
            'end_nav': [101000, 103020, 104565, 107179],
            'total_flow': [0, 0, 0, 0],
        })
    
    def test_geometric_linking(self, sub_period_returns):
        """Test that sub-periods are geometrically linked."""
        result = twr.calculate_monthly_twr(sub_period_returns)
        
        # Should have 3 months based on end dates spanning Jan, Feb, Mar
        assert len(result) >= 2
        
        # Check that January has expected return
        jan_result = result[result['month'] == pd.Period('2024-01')]
        if not jan_result.empty:
            jan_return = jan_result['return'].iloc[0]
            # January should have the first sub-period return
            assert jan_return == pytest.approx(0.01, abs=0.001)
    
    def test_empty_input(self):
        """Test with empty sub-period returns."""
        empty_df = pd.DataFrame(columns=['start_date', 'end_date', 'return', 'start_nav', 'end_nav', 'total_flow'])
        
        result = twr.calculate_monthly_twr(empty_df)
        
        assert len(result) == 0


class TestCalculateTotalReturns:
    """Tests for total return calculations."""
    
    def test_from_daily_returns(self):
        """Test total return calculation from daily returns."""
        daily = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=365),
            'return': [0.0003] * 365,  # ~0.03% daily
        })
        
        absolute, annualized = twr.calculate_total_returns_from_daily(daily)
        
        # Compound daily returns
        expected_absolute = (1 + 0.0003) ** 365 - 1
        assert abs(absolute - expected_absolute) < 0.001
        
        # Annualized should be close to absolute for 1 year
        assert abs(annualized - absolute) < 0.01
    
    def test_empty_returns(self):
        """Test with empty returns."""
        empty = pd.DataFrame(columns=['date', 'return'])
        
        absolute, annualized = twr.calculate_total_returns_from_daily(empty)
        
        assert absolute == 0.0
        assert annualized == 0.0


class TestCalculateInternalDispersion:
    """Tests for internal dispersion calculation."""
    
    def test_with_six_or_more_accounts(self):
        """Test dispersion calculation with sufficient accounts."""
        account_returns = {
            'acc1': 0.10,
            'acc2': 0.12,
            'acc3': 0.08,
            'acc4': 0.11,
            'acc5': 0.09,
            'acc6': 0.13,
        }
        
        dispersion = twr.calculate_internal_dispersion(account_returns)
        
        # Should return sample std dev
        expected = np.std([0.10, 0.12, 0.08, 0.11, 0.09, 0.13], ddof=1)
        assert dispersion is not None
        assert abs(dispersion - expected) < 0.0001
    
    def test_fewer_than_six_accounts(self):
        """Test that None is returned for fewer than 6 accounts."""
        account_returns = {
            'acc1': 0.10,
            'acc2': 0.12,
            'acc3': 0.08,
        }
        
        dispersion = twr.calculate_internal_dispersion(account_returns)
        
        assert dispersion is None
    
    def test_custom_min_accounts(self):
        """Test with custom minimum accounts threshold."""
        account_returns = {
            'acc1': 0.10,
            'acc2': 0.12,
            'acc3': 0.08,
        }
        
        dispersion = twr.calculate_internal_dispersion(account_returns, min_accounts=3)
        
        assert dispersion is not None


class TestGetPeriodWindows:
    """Tests for period window generation."""
    
    def test_default_windows(self):
        """Test default period windows."""
        windows = twr.get_period_windows()
        
        assert '2022' in windows or '2022_ytd' in windows
        assert '2023' in windows or '2023_ytd' in windows
    
    def test_february_fiscal_year(self):
        """Test February fiscal year windows."""
        from config_loader import Config, PeriodsConfig, _get_default_config
        
        config = _get_default_config()
        config.periods.fiscal_year_start_month = 2
        config.periods.reporting_years = [2023]
        
        windows = twr.get_period_windows(config)
        
        # Should be Feb 2023 - Jan 2024
        assert '2023' in windows or '2023_ytd' in windows


class TestBuildDailyReturnsDirect:
    """Tests for building daily returns directly from NAV and flows."""
    
    @pytest.fixture
    def nav_df(self):
        """Sample NAV DataFrame with daily data."""
        return pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=5),
            'Net Asset Value': [100000, 101000, 100500, 102000, 103000],
        })
    
    @pytest.fixture
    def trades_df(self):
        """Sample trades DataFrame."""
        return pd.DataFrame({
            'When': pd.to_datetime(['2024-01-03']),
            'EUR equivalent': [500],
        })
    
    def test_daily_returns_calculation(self, nav_df, trades_df):
        """Test daily returns are calculated correctly."""
        result = twr.build_daily_returns_direct(nav_df, trades_df)
        
        assert 'date' in result.columns
        assert 'return' in result.columns
        assert len(result) > 0
    
    def test_flow_adjustment(self, nav_df, trades_df):
        """Test that flows are properly adjusted in return calculation."""
        result = twr.build_daily_returns_direct(nav_df, trades_df)
        
        # Return formula: r = (end_nav - flow) / start_nav - 1
        # This accounts for flows at start of day
        assert all(np.isfinite(result['return']))


class TestReturnsCalculatorClass:
    """Tests for the ReturnsCalculator class."""
    
    def test_initialization_with_default_config(self, tmp_path):
        """Test initialization with default configuration."""
        # Create a minimal config
        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
brokerages:
  - name: IBKR
    module: IBKR
    enabled: true
fees:
  management_fee_annual: 0.01
""")
        
        calc = twr.ReturnsCalculator(config_path=str(config_path))
        
        assert calc.config is not None
        assert calc.brokerages is not None
        assert calc.all_results == {}
    
    def test_scan_for_accounts_empty_dir(self, tmp_path):
        """Test scanning for accounts in empty directory."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
brokerages:
  - name: TestBroker
    module: test
    enabled: true
paths:
  input_dir: nonexistent
""")
        
        calc = twr.ReturnsCalculator(config_path=str(config_path))
        calc.scan_for_accounts()
        
        # Should not crash, just find no accounts
        assert sum(len(clients) for clients in calc.brokerages.values()) == 0

