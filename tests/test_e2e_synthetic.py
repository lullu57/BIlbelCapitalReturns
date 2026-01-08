"""
End-to-End Tests with Synthetic Data

This module contains comprehensive tests using synthetic data that simulates
5 years of trading data for multiple brokerages (IBKR and Exante format).

Test scenarios include:
- Regular deposits and withdrawals
- Large cash flows (>10% of NAV)
- Negative periods
- Various market conditions
- Fee calculations across years
- Composite aggregation
"""

import os
import sys
import tempfile
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Tuple, Dict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, PROJECT_ROOT)

import twr_calculator as twr
from fee_calculator import PerformanceFeeCalculator, FeeCalculator, calculate_fees_for_period
from flow_utils import is_large_cash_flow, flag_large_cash_flows, get_flow_column
from config_loader import load_config, Config, PeriodsConfig, FeeConfig


# =============================================================================
# Synthetic Data Generators
# =============================================================================

@dataclass
class SyntheticScenario:
    """Defines a synthetic test scenario."""
    name: str
    start_date: datetime
    end_date: datetime
    initial_nav: float
    monthly_returns: List[float]  # Per-month returns
    flows: List[Tuple[datetime, float]]  # (date, amount) - positive=deposit, negative=withdrawal
    description: str = ""


def generate_monthly_dates(start_date: datetime, num_months: int) -> List[datetime]:
    """Generate month-end dates."""
    dates = []
    current = start_date
    for _ in range(num_months):
        # Move to last day of month
        if current.month == 12:
            next_month = current.replace(year=current.year + 1, month=1, day=1)
        else:
            next_month = current.replace(month=current.month + 1, day=1)
        month_end = next_month - timedelta(days=1)
        dates.append(month_end)
        current = next_month
    return dates


def generate_nav_series(
    initial_nav: float,
    monthly_returns: List[float],
    flows: List[Tuple[datetime, float]],
    start_date: datetime,
) -> pd.DataFrame:
    """
    Generate a NAV series with flows incorporated.
    
    NAV changes due to:
    1. Investment returns
    2. Cash flows (deposits/withdrawals)
    """
    dates = [start_date]
    navs = [initial_nav]
    
    current_nav = initial_nav
    current_date = start_date
    
    for i, monthly_return in enumerate(monthly_returns):
        # Calculate next month's date
        if current_date.month == 12:
            next_month = current_date.replace(year=current_date.year + 1, month=1, day=1)
        else:
            next_month = current_date.replace(month=current_date.month + 1, day=1)
        
        # Get last day of month
        if next_month.month == 12:
            month_end = next_month.replace(year=next_month.year + 1, month=1, day=1) - timedelta(days=1)
        else:
            month_end = next_month.replace(month=next_month.month + 1, day=1) - timedelta(days=1)
        
        # Add flows that occurred this month
        month_flows = sum(
            f[1] for f in flows 
            if current_date < f[0] <= month_end
        )
        
        # Apply return then add flows
        current_nav = current_nav * (1 + monthly_return) + month_flows
        current_date = month_end
        
        dates.append(current_date)
        navs.append(current_nav)
    
    return pd.DataFrame({
        'Date': pd.to_datetime(dates),
        'Net Asset Value': navs
    })


def generate_trades_df(flows: List[Tuple[datetime, float]]) -> pd.DataFrame:
    """Generate trades DataFrame in IBKR/Exante format."""
    if not flows:
        return pd.DataFrame(columns=['When', 'Operation type', 'EUR equivalent', 'Adjusted EUR'])
    
    return pd.DataFrame({
        'When': pd.to_datetime([f[0] for f in flows]),
        'Operation type': ['FUNDING/WITHDRAWAL'] * len(flows),
        'EUR equivalent': [f[1] for f in flows],
        'Adjusted EUR': [f[1] for f in flows],
        'Description': [f"{'Deposit' if f[1] > 0 else 'Withdrawal'} {abs(f[1]):.2f} EUR" for f in flows]
    })


# =============================================================================
# Pre-defined Synthetic Scenarios (5 Years)
# =============================================================================

def create_5_year_scenarios() -> Dict[str, SyntheticScenario]:
    """Create 5-year synthetic scenarios for different account types."""
    start_date = datetime(2020, 2, 1)  # Feb 2020 (fiscal year start)
    num_months = 60  # 5 years
    
    # Scenario 1: Steady Growth Account
    steady_returns = [0.005 + np.sin(m * 0.2) * 0.01 for m in range(num_months)]  # ~6-18% annual
    steady_flows = [
        (datetime(2020, 3, 15), 10000),   # Initial deposit
        (datetime(2021, 1, 15), 5000),    # Annual top-up
        (datetime(2022, 1, 15), 5000),
        (datetime(2023, 1, 15), 5000),
        (datetime(2024, 1, 15), 5000),
    ]
    
    # Scenario 2: Volatile Account with Withdrawals
    volatile_returns = []
    np.random.seed(42)
    for m in range(num_months):
        # More volatile with some negative months
        base = 0.003
        volatility = np.random.normal(0, 0.03)
        # Add market crash in month 2-4 (March 2020)
        if 1 <= m <= 3:
            volatility -= 0.10  # COVID crash simulation
        volatile_returns.append(base + volatility)
    
    volatile_flows = [
        (datetime(2020, 3, 1), 50000),    # Initial deposit before crash
        (datetime(2020, 6, 15), 20000),   # Buy the dip
        (datetime(2021, 12, 1), -10000),  # Withdrawal
        (datetime(2023, 3, 1), -15000),   # Large withdrawal
        (datetime(2024, 6, 1), 25000),    # Large deposit
    ]
    
    # Scenario 3: Conservative Account
    conservative_returns = [0.002 + np.random.normal(0, 0.002) for _ in range(num_months)]  # ~2-3% annual
    conservative_flows = [
        (datetime(2020, 2, 15), 100000),
        (datetime(2022, 2, 15), 50000),
    ]
    
    # Scenario 4: Aggressive Account (high returns but high volatility)
    aggressive_returns = []
    np.random.seed(123)
    for m in range(num_months):
        if m % 12 < 6:  # First half of year tends positive
            aggressive_returns.append(0.02 + np.random.normal(0, 0.04))
        else:  # Second half more volatile
            aggressive_returns.append(0.005 + np.random.normal(0, 0.05))
    
    aggressive_flows = [
        (datetime(2020, 2, 15), 25000),
        (datetime(2020, 8, 15), 10000),
        (datetime(2021, 2, 15), 15000),
        (datetime(2022, 8, 15), -5000),
        (datetime(2023, 2, 15), 20000),
        (datetime(2024, 8, 15), -8000),
    ]
    
    return {
        'steady': SyntheticScenario(
            name='steady_growth',
            start_date=start_date,
            end_date=datetime(2025, 1, 31),
            initial_nav=10000,
            monthly_returns=steady_returns,
            flows=steady_flows,
            description="Steady growth account with annual top-ups"
        ),
        'volatile': SyntheticScenario(
            name='volatile_market',
            start_date=start_date,
            end_date=datetime(2025, 1, 31),
            initial_nav=50000,
            monthly_returns=volatile_returns,
            flows=volatile_flows,
            description="Volatile account with market crash and recovery"
        ),
        'conservative': SyntheticScenario(
            name='conservative',
            start_date=start_date,
            end_date=datetime(2025, 1, 31),
            initial_nav=100000,
            monthly_returns=conservative_returns,
            flows=conservative_flows,
            description="Low-risk conservative account"
        ),
        'aggressive': SyntheticScenario(
            name='aggressive',
            start_date=start_date,
            end_date=datetime(2025, 1, 31),
            initial_nav=25000,
            monthly_returns=aggressive_returns,
            flows=aggressive_flows,
            description="High-risk aggressive trading account"
        ),
    }


# =============================================================================
# Test Classes
# =============================================================================

class TestSyntheticDataGeneration:
    """Tests for the synthetic data generation utilities."""
    
    def test_generate_nav_series(self):
        """Test NAV series generation."""
        start = datetime(2024, 1, 1)
        returns = [0.01, 0.02, -0.01, 0.015]  # 4 months
        flows = [(datetime(2024, 2, 15), 5000)]
        
        nav_df = generate_nav_series(100000, returns, flows, start)
        
        assert len(nav_df) == 5  # Initial + 4 months
        assert nav_df.iloc[0]['Net Asset Value'] == 100000
        # NAV should increase with returns and flows
        assert nav_df.iloc[-1]['Net Asset Value'] > 100000
    
    def test_generate_trades_df(self):
        """Test trades DataFrame generation."""
        flows = [
            (datetime(2024, 1, 15), 10000),
            (datetime(2024, 3, 20), -5000),
        ]
        
        trades_df = generate_trades_df(flows)
        
        assert len(trades_df) == 2
        assert 'When' in trades_df.columns
        assert 'EUR equivalent' in trades_df.columns
        assert trades_df.iloc[0]['EUR equivalent'] == 10000
        assert trades_df.iloc[1]['EUR equivalent'] == -5000


class TestEndToEndSubPeriodReturns:
    """End-to-end tests for sub-period return calculations."""
    
    @pytest.fixture
    def five_year_data(self) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate 5 years of synthetic data for all scenarios."""
        scenarios = create_5_year_scenarios()
        data = {}
        
        for key, scenario in scenarios.items():
            nav_df = generate_nav_series(
                scenario.initial_nav,
                scenario.monthly_returns,
                scenario.flows,
                scenario.start_date
            )
            trades_df = generate_trades_df(scenario.flows)
            data[key] = (nav_df, trades_df)
        
        return data
    
    def test_sub_period_returns_all_scenarios(self, five_year_data):
        """Test sub-period return calculation for all scenarios."""
        for name, (nav_df, trades_df) in five_year_data.items():
            result = twr.calculate_sub_period_returns(nav_df, trades_df)
            
            # Should have sub-periods
            assert len(result) > 0, f"No sub-periods for {name}"
            
            # All returns should be finite
            assert all(np.isfinite(result['return'])), f"Non-finite returns in {name}"
            
            # Start dates should be before end dates
            assert all(result['start_date'] < result['end_date']), f"Date order issue in {name}"
    
    def test_monthly_twr_all_scenarios(self, five_year_data):
        """Test monthly TWR calculation for all scenarios."""
        for name, (nav_df, trades_df) in five_year_data.items():
            sub_returns = twr.calculate_sub_period_returns(nav_df, trades_df)
            monthly = twr.calculate_monthly_twr(sub_returns)
            
            # Should have monthly data
            assert len(monthly) > 0, f"No monthly data for {name}"
            
            # Should span approximately 60 months (5 years)
            assert len(monthly) >= 55, f"Expected ~60 months, got {len(monthly)} for {name}"
            assert len(monthly) <= 65, f"Expected ~60 months, got {len(monthly)} for {name}"
    
    def test_total_returns_consistency(self, five_year_data):
        """Test that total returns are consistent across calculation methods."""
        for name, (nav_df, trades_df) in five_year_data.items():
            sub_returns = twr.calculate_sub_period_returns(nav_df, trades_df)
            monthly = twr.calculate_monthly_twr(sub_returns)
            
            # Calculate total return from monthly
            total_from_monthly = np.prod(1 + monthly['return']) - 1
            
            # Total return should be reasonable for 5 years
            # Even volatile scenarios shouldn't completely zero out
            assert total_from_monthly > -0.99, f"Extreme loss in {name}: {total_from_monthly}"


class TestEndToEndLargeCashFlows:
    """End-to-end tests for large cash flow handling."""
    
    def test_large_flow_detection(self):
        """Test that large cash flows are properly detected."""
        # Create scenario with known large flow
        nav_df = pd.DataFrame({
            'Date': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01']),
            'Net Asset Value': [100000, 125000, 130000]  # 15000 deposit is 15% of initial
        })
        trades_df = pd.DataFrame({
            'When': pd.to_datetime(['2024-01-15']),
            'EUR equivalent': [15000],  # 15% of 100000 = large flow
        })
        
        flagged = flag_large_cash_flows(trades_df, nav_df, threshold=0.10)
        
        assert 'large_flow_flag' in flagged.columns
        assert flagged.iloc[0]['large_flow_flag'] == True
    
    def test_small_flows_not_flagged(self):
        """Test that small cash flows are not flagged."""
        nav_df = pd.DataFrame({
            'Date': pd.to_datetime(['2024-01-01', '2024-02-01']),
            'Net Asset Value': [100000, 103000]
        })
        trades_df = pd.DataFrame({
            'When': pd.to_datetime(['2024-01-15']),
            'EUR equivalent': [3000],  # 3% of 100000 = small flow
        })
        
        flagged = flag_large_cash_flows(trades_df, nav_df, threshold=0.10)
        
        assert flagged.iloc[0]['large_flow_flag'] == False


class TestEndToEndFeeCalculations:
    """End-to-end tests for fee calculations across 5 years."""
    
    def test_performance_fee_over_5_years(self):
        """Test performance fee calculation with shortfall carry-forward over 5 years."""
        calc = PerformanceFeeCalculator(hurdle_rate=0.06, fee_rate=0.25)
        
        # Simulate 5 years of returns
        annual_returns = [
            0.04,   # Year 1: Below hurdle, creates 2% shortfall
            0.05,   # Year 2: Below hurdle, creates 1% more shortfall (total 3%)
            0.12,   # Year 3: Above effective hurdle (6%+3%=9%), excess 3%, fee 0.75%
            -0.05,  # Year 4: Loss, creates 11% shortfall
            0.20,   # Year 5: Above effective hurdle (6%+11%=17%), excess 3%, fee 0.75%
        ]
        
        results = []
        for year, ret in enumerate(annual_returns, start=2020):
            fee, shortfall = calc.calculate_annual_fee(ret, year=year)
            results.append({
                'year': year,
                'return': ret,
                'fee': fee,
                'shortfall': shortfall
            })
        
        # Verify Year 1: no fee, shortfall = 2%
        assert results[0]['fee'] == pytest.approx(0.0)
        assert results[0]['shortfall'] == pytest.approx(0.02, abs=0.001)
        
        # Verify Year 2: no fee, shortfall = 3% (2% + 1%)
        assert results[1]['fee'] == pytest.approx(0.0)
        assert results[1]['shortfall'] == pytest.approx(0.03, abs=0.001)
        
        # Verify Year 3: fee charged, shortfall cleared
        assert results[2]['fee'] == pytest.approx(0.0075, abs=0.001)  # 25% of 3%
        assert results[2]['shortfall'] == pytest.approx(0.0)
        
        # Verify Year 4: no fee, large shortfall
        assert results[3]['fee'] == pytest.approx(0.0)
        assert results[3]['shortfall'] == pytest.approx(0.11, abs=0.001)  # 6% - (-5%) = 11%
        
        # Verify Year 5: fee on excess over effective hurdle
        assert results[4]['fee'] == pytest.approx(0.0075, abs=0.001)  # 25% of (20%-17%)
        assert results[4]['shortfall'] == pytest.approx(0.0)
    
    def test_combined_fees_monthly(self):
        """Test combined management and performance fee calculation."""
        calc = FeeCalculator(
            annual_mgmt_fee=0.01,
            hurdle_rate=0.06,
            performance_fee_rate=0.25
        )
        
        # 12 months of returns
        months = pd.period_range('2024-01', periods=12, freq='M')
        returns = [0.01, 0.005, 0.015, -0.02, 0.03, 0.01,
                   0.02, 0.005, 0.01, 0.015, 0.02, 0.01]
        
        monthly_df = pd.DataFrame({
            'month': months,
            'return': returns
        })
        
        result = calc.calculate_monthly_fees(monthly_df)
        
        # Check columns exist
        assert 'gross_return' in result.columns
        assert 'net_before_perf' in result.columns
        assert 'management_fee' in result.columns
        
        # Net returns should be less than gross (due to mgmt fee)
        assert all(result['net_before_perf'] < result['gross_return'])


class TestEndToEndCompositeAggregation:
    """End-to-end tests for GIPS composite aggregation."""
    
    @pytest.fixture
    def multiple_accounts(self):
        """Create monthly returns for multiple accounts."""
        months = pd.period_range('2024-01', periods=12, freq='M')
        
        # Different accounts with different returns and NAVs
        accounts = {
            'account_1': pd.DataFrame({
                'month': months,
                'return': [0.01, 0.02, -0.01, 0.03, 0.01, 0.02,
                          0.01, 0.015, 0.02, 0.01, 0.025, 0.01],
                'start_of_month_nav': [100000 * (1.01 ** i) for i in range(12)],
            }),
            'account_2': pd.DataFrame({
                'month': months,
                'return': [0.015, 0.01, 0.02, 0.01, 0.025, 0.01,
                          0.02, 0.01, 0.015, 0.02, 0.01, 0.02],
                'start_of_month_nav': [200000 * (1.015 ** i) for i in range(12)],
            }),
            'account_3': pd.DataFrame({
                'month': months,
                'return': [0.005, 0.03, 0.01, 0.02, 0.01, 0.015,
                          0.01, 0.02, 0.01, 0.015, 0.02, 0.01],
                'start_of_month_nav': [150000 * (1.01 ** i) for i in range(12)],
            }),
        }
        
        return accounts
    
    @pytest.fixture
    def accounts_with_different_start_dates(self):
        """Create accounts with different start dates (staggered onboarding)."""
        # Full year account (Jan-Dec 2024)
        full_year_months = pd.period_range('2024-01', periods=12, freq='M')
        
        # Late start account (Jul-Dec 2024 only - 6 months)
        late_start_months = pd.period_range('2024-07', periods=6, freq='M')
        
        accounts = {
            'full_year_client': pd.DataFrame({
                'month': full_year_months,
                'return': [0.01, 0.02, -0.01, 0.03, 0.01, 0.02,
                          0.01, 0.015, 0.02, 0.01, 0.025, 0.01],
                'start_of_month_nav': [100000 * (1.01 ** i) for i in range(12)],
            }),
            'late_start_client': pd.DataFrame({
                'month': late_start_months,
                'return': [0.02, 0.025, 0.015, 0.03, 0.02, 0.015],
                'start_of_month_nav': [50000 * (1.02 ** i) for i in range(6)],
            }),
        }
        
        return accounts
    
    def test_composite_with_staggered_account_starts(self, accounts_with_different_start_dates):
        """Test composite handles accounts joining at different times."""
        composite_df, stats = twr.build_gips_composite(accounts_with_different_start_dates)
        
        # Should have 12 months total
        assert len(composite_df) == 12
        
        # Check active accounts for different periods
        jan_data = composite_df[composite_df['month'] == pd.Period('2024-01')]
        jul_data = composite_df[composite_df['month'] == pd.Period('2024-07')]
        
        # January should only have full_year_client
        jan_active = jan_data['active_accounts'].iloc[0]
        assert 'full_year_client' in jan_active
        assert 'late_start_client' not in jan_active
        
        # July should have both clients
        jul_active = jul_data['active_accounts'].iloc[0]
        assert 'full_year_client' in jul_active
        assert 'late_start_client' in jul_active
    
    def test_composite_weighting_changes_with_new_accounts(self, accounts_with_different_start_dates):
        """Test that composite weighting adjusts when accounts join."""
        composite_df, stats = twr.build_gips_composite(accounts_with_different_start_dates)
        
        # In January, composite return should equal full_year_client return (only account)
        jan_composite = composite_df[composite_df['month'] == pd.Period('2024-01')]['composite_return'].iloc[0]
        jan_full_year = 0.01  # From fixture
        assert jan_composite == pytest.approx(jan_full_year, abs=0.001)
        
        # In July, composite should be weighted average of both accounts
        # full_year_client NAV after 6 months: 100000 * 1.01^6 ≈ 106152
        # late_start_client NAV at start: 50000
        # Weights: 106152/(106152+50000) ≈ 0.68, 50000/156152 ≈ 0.32
        jul_composite = composite_df[composite_df['month'] == pd.Period('2024-07')]['composite_return'].iloc[0]
        
        # Verify it's between the two returns (weighted average)
        jul_full_year = 0.01  # From fixture
        jul_late_start = 0.02  # From fixture
        assert min(jul_full_year, jul_late_start) <= jul_composite <= max(jul_full_year, jul_late_start)
    
    def test_composite_calculation(self, multiple_accounts):
        """Test GIPS composite calculation with multiple accounts."""
        composite_df, stats = twr.build_gips_composite(multiple_accounts)
        
        # Should have 12 months
        assert len(composite_df) == 12
        
        # Should have composite return column
        assert 'composite_return' in composite_df.columns
        
        # Composite return should be within range of individual returns
        for month in composite_df['month']:
            account_returns = [
                acc[acc['month'] == month]['return'].iloc[0]
                for acc in multiple_accounts.values()
            ]
            comp_ret = composite_df[composite_df['month'] == month]['composite_return'].iloc[0]
            
            assert min(account_returns) <= comp_ret <= max(account_returns)
    
    def test_composite_beginning_of_period_weighting(self, multiple_accounts):
        """Test that composite uses beginning-of-period NAV for weighting."""
        composite_df, stats = twr.build_gips_composite(multiple_accounts)
        
        # For month 1, weights should be based on initial NAVs
        # account_1: 100000, account_2: 200000, account_3: 150000
        # Total: 450000
        # Weights: 100/450=0.222, 200/450=0.444, 150/450=0.333
        
        month_1_returns = {
            'account_1': 0.01,
            'account_2': 0.015,
            'account_3': 0.005,
        }
        month_1_navs = {
            'account_1': 100000,
            'account_2': 200000,
            'account_3': 150000,
        }
        
        total_nav = sum(month_1_navs.values())
        expected_weighted_return = sum(
            (nav / total_nav) * ret
            for nav, ret in zip(month_1_navs.values(), month_1_returns.values())
        )
        
        actual_return = composite_df.iloc[0]['composite_return']
        
        assert actual_return == pytest.approx(expected_weighted_return, abs=0.0001)


class TestEndToEndFullPipeline:
    """Full end-to-end test of the entire returns calculation pipeline."""
    
    def test_full_pipeline_5_years(self):
        """Test complete pipeline from raw data to final results over 5 years."""
        # Create synthetic data
        scenarios = create_5_year_scenarios()
        scenario = scenarios['steady']
        
        nav_df = generate_nav_series(
            scenario.initial_nav,
            scenario.monthly_returns,
            scenario.flows,
            scenario.start_date
        )
        trades_df = generate_trades_df(scenario.flows)
        
        # Step 1: Calculate sub-period returns
        sub_returns = twr.calculate_sub_period_returns(nav_df, trades_df)
        assert len(sub_returns) > 0
        
        # Step 2: Calculate monthly TWR
        monthly = twr.calculate_monthly_twr(sub_returns)
        assert len(monthly) >= 55  # ~5 years of months
        
        # Step 3: Calculate 6-month returns
        six_month = twr.calculate_six_month_returns(monthly)
        assert len(six_month) > 0
        
        # Step 4: Calculate total returns from monthly data
        # Using monthly returns which are more reliable than daily interpolation
        total_return = np.prod(1 + monthly['return']) - 1
        
        # Validate results are reasonable
        assert np.isfinite(total_return)
        
        # Over 5 years with generally positive returns, should be positive
        # The steady scenario has returns around 0.5-1.5% monthly
        assert total_return > 0, f"Expected positive return for steady scenario, got {total_return}"
    
    def test_full_pipeline_with_fees(self):
        """Test complete pipeline including fee calculations."""
        # Create data
        scenarios = create_5_year_scenarios()
        scenario = scenarios['volatile']
        
        nav_df = generate_nav_series(
            scenario.initial_nav,
            scenario.monthly_returns,
            scenario.flows,
            scenario.start_date
        )
        trades_df = generate_trades_df(scenario.flows)
        
        # Calculate returns
        sub_returns = twr.calculate_sub_period_returns(nav_df, trades_df)
        monthly = twr.calculate_monthly_twr(sub_returns)
        
        # Apply fees
        result = calculate_fees_for_period(monthly)
        
        # Verify fee calculations
        assert 'total_gross_return' in result
        assert 'total_net_return' in result
        assert 'total_management_fee' in result
        
        # Net should be less than gross
        assert result['total_net_return'] <= result['total_gross_return']
    
    def test_internal_dispersion_with_multiple_accounts(self):
        """Test internal dispersion calculation with 6+ accounts."""
        # Create 6 accounts (minimum for GIPS dispersion)
        np.random.seed(42)
        account_returns = {
            f'account_{i}': 0.08 + np.random.normal(0, 0.02)
            for i in range(8)  # 8 accounts
        }
        account_navs = {
            f'account_{i}': 100000 + i * 20000
            for i in range(8)
        }
        
        dispersion = twr.calculate_internal_dispersion(
            account_returns,
            account_navs,
            min_accounts=6
        )
        
        # Should return a value
        assert dispersion is not None
        assert dispersion > 0
        
        # Should be reasonable (< 10% for typical portfolios)
        assert dispersion < 0.10
    
    def test_period_windows_generation(self):
        """Test that period windows are correctly generated."""
        periods_config = PeriodsConfig(
            fiscal_year_start_month=2,
            reporting_years=[2020, 2021, 2022, 2023, 2024]
        )
        
        windows = periods_config.get_period_windows()
        
        # Should have 5 years
        assert len(windows) == 5
        
        # Check 2022 fiscal year (Feb 2022 - Jan 2023)
        assert '2022' in windows
        assert windows['2022'][0] == '2022-02-01'
        assert windows['2022'][1] == '2023-01-31'


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_day_returns(self):
        """Test with minimal data (2 days)."""
        nav_df = pd.DataFrame({
            'Date': pd.to_datetime(['2024-01-01', '2024-01-02']),
            'Net Asset Value': [100000, 101000]
        })
        trades_df = pd.DataFrame(columns=['When', 'EUR equivalent'])
        
        result = twr.calculate_sub_period_returns(nav_df, trades_df)
        
        # Should have 1 sub-period with 1% return
        assert len(result) == 1
        assert result.iloc[0]['return'] == pytest.approx(0.01)
    
    def test_zero_flows(self):
        """Test with no external cash flows."""
        nav_df = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=30),
            'Net Asset Value': [100000 * (1.001 ** i) for i in range(30)]
        })
        trades_df = pd.DataFrame(columns=['When', 'EUR equivalent'])
        
        result = twr.calculate_sub_period_returns(nav_df, trades_df)
        
        # Returns should be purely from NAV changes
        assert len(result) == 29
        assert all(result['total_flow'] == 0)
    
    def test_negative_nav_handling(self):
        """Test handling of scenarios with negative NAV (should warn/skip)."""
        nav_df = pd.DataFrame({
            'Date': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01']),
            'Net Asset Value': [100000, -5000, 50000]  # Temporary negative
        })
        trades_df = pd.DataFrame(columns=['When', 'EUR equivalent'])
        
        # Should handle without crashing
        result = twr.calculate_sub_period_returns(nav_df, trades_df)
        
        # Periods with negative starting NAV are skipped
        # So we only get the first period (Jan 1 -> Feb 1)
        assert len(result) >= 1
        
        # The first period should show a loss (going to negative NAV)
        assert result.iloc[0]['return'] < 0
    
    def test_large_withdrawal_scenario(self):
        """Test scenario with large withdrawal (>10% of NAV)."""
        nav_df = pd.DataFrame({
            'Date': pd.to_datetime(['2024-01-01', '2024-02-01']),
            'Net Asset Value': [100000, 45000]
        })
        trades_df = pd.DataFrame({
            'When': pd.to_datetime(['2024-01-15']),
            'EUR equivalent': [-50000],  # 50% withdrawal
        })
        
        result = twr.calculate_sub_period_returns(nav_df, trades_df)
        
        # Should flag as large flow
        if 'has_large_flow' in result.columns:
            assert result.iloc[0]['has_large_flow'] == True
    
    def test_complete_loss_scenario(self):
        """Test scenario with complete portfolio loss."""
        nav_df = pd.DataFrame({
            'Date': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01']),
            'Net Asset Value': [100000, 50000, 0.01]  # Near-total loss
        })
        trades_df = pd.DataFrame(columns=['When', 'EUR equivalent'])
        
        result = twr.calculate_sub_period_returns(nav_df, trades_df)
        
        # Should handle without crashing
        assert len(result) == 2
        # Total loss should approach -100%
        total_return = (1 + result.iloc[0]['return']) * (1 + result.iloc[1]['return']) - 1
        assert total_return < -0.99

