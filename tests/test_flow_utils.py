"""Tests for the flow utilities module."""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, PROJECT_ROOT)

from flow_utils import (
    get_flow_column,
    get_flows_for_period,
    get_total_flow_for_period,
    get_flows_by_day,
    is_large_cash_flow,
    flag_large_cash_flows,
    normalize_flow_dataframe,
    DEFAULT_FLOW_COLUMNS,
)


class TestGetFlowColumn:
    """Tests for flow column detection."""
    
    def test_adjusted_eur_preferred(self):
        """Test that 'Adjusted EUR' is preferred over 'EUR equivalent'."""
        df = pd.DataFrame({
            'Adjusted EUR': [100, 200],
            'EUR equivalent': [100, 200]
        })
        assert get_flow_column(df) == 'Adjusted EUR'
    
    def test_eur_equivalent_fallback(self):
        """Test fallback to 'EUR equivalent' when 'Adjusted EUR' not present."""
        df = pd.DataFrame({
            'EUR equivalent': [100, 200],
            'Other': [1, 2]
        })
        assert get_flow_column(df) == 'EUR equivalent'
    
    def test_custom_columns(self):
        """Test with custom column list."""
        df = pd.DataFrame({
            'Amount': [100, 200],
            'Other': [1, 2]
        })
        assert get_flow_column(df, ['Amount', 'Value']) == 'Amount'
    
    def test_case_insensitive_fallback(self):
        """Test case-insensitive matching as fallback."""
        df = pd.DataFrame({
            'adjusted eur': [100, 200],  # lowercase
        })
        assert get_flow_column(df) == 'adjusted eur'
    
    def test_no_matching_column_raises(self):
        """Test that ValueError is raised when no matching column found."""
        df = pd.DataFrame({
            'Other': [100, 200]
        })
        with pytest.raises(ValueError, match="No recognized flow column"):
            get_flow_column(df)


class TestGetFlowsForPeriod:
    """Tests for period flow extraction."""
    
    @pytest.fixture
    def sample_trades(self):
        """Sample trades DataFrame."""
        return pd.DataFrame({
            'When': pd.to_datetime([
                '2024-01-15',
                '2024-02-01',
                '2024-02-15',
                '2024-03-01',
            ]),
            'EUR equivalent': [1000, 2000, -500, 1500]
        })
    
    def test_default_inclusive_right(self, sample_trades):
        """Test default inclusive='right' behavior."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 2, 1)
        
        result = get_flows_for_period(sample_trades, start, end)
        
        # Should include Jan 15 and Feb 1, exclude Mar dates
        assert len(result) == 2
    
    def test_inclusive_both(self, sample_trades):
        """Test inclusive='both' behavior."""
        start = datetime(2024, 1, 15)
        end = datetime(2024, 2, 1)
        
        result = get_flows_for_period(
            sample_trades, start, end, inclusive='both'
        )
        
        assert len(result) == 2
    
    def test_empty_result(self, sample_trades):
        """Test when no flows in period."""
        start = datetime(2023, 1, 1)
        end = datetime(2023, 12, 31)
        
        result = get_flows_for_period(sample_trades, start, end)
        
        assert len(result) == 0


class TestGetTotalFlowForPeriod:
    """Tests for total flow calculation."""
    
    @pytest.fixture
    def sample_trades(self):
        """Sample trades DataFrame."""
        return pd.DataFrame({
            'When': pd.to_datetime([
                '2024-01-15',
                '2024-01-20',
                '2024-02-15',
            ]),
            'EUR equivalent': [1000, 500, 2000]
        })
    
    def test_sum_flows_in_period(self, sample_trades):
        """Test summing flows in a period."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        
        total = get_total_flow_for_period(sample_trades, start, end)
        
        assert total == 1500  # 1000 + 500
    
    def test_empty_period_returns_zero(self, sample_trades):
        """Test that empty period returns 0."""
        start = datetime(2023, 1, 1)
        end = datetime(2023, 12, 31)
        
        total = get_total_flow_for_period(sample_trades, start, end)
        
        assert total == 0.0


class TestGetFlowsByDay:
    """Tests for daily flow aggregation."""
    
    def test_aggregate_multiple_flows_same_day(self):
        """Test aggregating multiple flows on the same day."""
        trades = pd.DataFrame({
            'When': pd.to_datetime([
                '2024-01-15',
                '2024-01-15',
                '2024-01-16',
            ]),
            'EUR equivalent': [1000, 500, 2000]
        })
        
        result = get_flows_by_day(trades)
        
        jan_15 = pd.Timestamp('2024-01-15')
        jan_16 = pd.Timestamp('2024-01-16')
        
        assert result[jan_15] == 1500
        assert result[jan_16] == 2000
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        trades = pd.DataFrame(columns=['When', 'EUR equivalent'])
        
        result = get_flows_by_day(trades)
        
        assert result == {}


class TestIsLargeCashFlow:
    """Tests for large cash flow detection."""
    
    def test_flow_exceeds_threshold(self):
        """Test flow exceeding threshold is flagged."""
        assert is_large_cash_flow(15000, 100000, 0.10) is True  # 15% > 10%
    
    def test_flow_below_threshold(self):
        """Test flow below threshold is not flagged."""
        assert is_large_cash_flow(5000, 100000, 0.10) is False  # 5% < 10%
    
    def test_exactly_at_threshold(self):
        """Test flow exactly at threshold is not flagged."""
        assert is_large_cash_flow(10000, 100000, 0.10) is False  # 10% = 10%
    
    def test_negative_flow(self):
        """Test withdrawal (negative flow) is handled correctly."""
        assert is_large_cash_flow(-15000, 100000, 0.10) is True  # Uses abs()
    
    def test_zero_nav(self):
        """Test zero NAV returns False (avoid division by zero)."""
        assert is_large_cash_flow(1000, 0, 0.10) is False
    
    def test_negative_nav(self):
        """Test negative NAV returns False."""
        assert is_large_cash_flow(1000, -10000, 0.10) is False


class TestFlagLargeCashFlows:
    """Tests for flagging large cash flows in DataFrame."""
    
    @pytest.fixture
    def sample_data(self):
        """Sample NAV and trades data."""
        nav_df = pd.DataFrame({
            'Date': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01']),
            'Net Asset Value': [100000, 110000, 120000]
        })
        
        trades_df = pd.DataFrame({
            'When': pd.to_datetime(['2024-01-15', '2024-02-15']),
            'EUR equivalent': [5000, 15000]  # 5% and ~13.6% of NAV
        })
        
        return nav_df, trades_df
    
    def test_flags_large_flows(self, sample_data):
        """Test that large flows are flagged."""
        nav_df, trades_df = sample_data
        
        result = flag_large_cash_flows(trades_df, nav_df, threshold=0.10)
        
        assert 'large_flow_flag' in result.columns
        assert 'flow_pct_of_nav' in result.columns
        
        # First flow (5000) should not be flagged
        # Second flow (15000) should be flagged
        assert result.iloc[0]['large_flow_flag'] == False
        assert result.iloc[1]['large_flow_flag'] == True
    
    def test_empty_trades(self, sample_data):
        """Test with empty trades DataFrame."""
        nav_df, _ = sample_data
        empty_trades = pd.DataFrame(columns=['When', 'EUR equivalent'])
        
        result = flag_large_cash_flows(empty_trades, nav_df)
        
        assert 'large_flow_flag' in result.columns
        assert len(result) == 0


class TestNormalizeFlowDataFrame:
    """Tests for flow DataFrame normalization."""
    
    def test_normalizes_columns(self):
        """Test that columns are normalized to 'date' and 'flow'."""
        trades = pd.DataFrame({
            'When': pd.to_datetime(['2024-01-15', '2024-01-20']),
            'EUR equivalent': [1000, 500]
        })
        
        result = normalize_flow_dataframe(trades)
        
        assert 'date' in result.columns
        assert 'flow' in result.columns
        assert len(result) == 2
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        empty = pd.DataFrame(columns=['When', 'EUR equivalent'])
        
        result = normalize_flow_dataframe(empty)
        
        assert 'date' in result.columns
        assert 'flow' in result.columns
        assert len(result) == 0

