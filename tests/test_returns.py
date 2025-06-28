import importlib.util
import os
import sys
import pandas as pd
import numpy as np
import numpy_financial as npf
import pytest

# Ensure project root is on sys.path for module imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, PROJECT_ROOT)

# Load the TWR Calculator module which has a space in its filename
MODULE_PATH = os.path.join(PROJECT_ROOT, 'TWR Calculator.py')
spec = importlib.util.spec_from_file_location('twr', MODULE_PATH)
twr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(twr)

def test_calculate_sub_period_returns_basic():
    nav_df = pd.DataFrame({
        'Date': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01']),
        'Net Asset Value': [100.0, 110.0, 150.0],
    })
    trades_df = pd.DataFrame({
        'When': pd.to_datetime(['2024-01-15', '2024-02-15']),
        'EUR equivalent': [20.0, -10.0],
    })

    result = twr.calculate_sub_period_returns(nav_df, trades_df)
    expected_returns = [-0.0833333333, 0.5]
    assert result['return'].tolist() == pytest.approx(expected_returns)
    assert list(result['start_date']) == list(pd.to_datetime(['2024-01-01', '2024-02-01']))
    assert list(result['end_date']) == list(pd.to_datetime(['2024-02-01', '2024-03-01']))

def test_calculate_monthly_twr_basic():
    sub_period_returns = pd.DataFrame({
        'start_date': pd.to_datetime(['2024-01-01', '2024-02-01']),
        'end_date': pd.to_datetime(['2024-02-01', '2024-03-01']),
        'return': [-0.0833333333, 0.5],
        'start_nav': [100.0, 110.0],
    })

    monthly = twr.calculate_monthly_twr(sub_period_returns)
    expected_months = [pd.Period('2024-01'), pd.Period('2024-02')]
    expected_returns = [-0.0833333333, 0.5]
    assert monthly['month'].tolist() == expected_months
    assert monthly['return'].tolist() == pytest.approx(expected_returns)
    assert monthly['start_of_month_nav'].tolist() == [100.0, 110.0]

def test_calculate_six_month_returns():
    months = pd.period_range('2024-01', periods=6, freq='M')
    returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    monthly_df = pd.DataFrame({'month': months, 'return': returns})

    result = twr.calculate_six_month_returns(monthly_df)
    expected = np.prod(1 + returns) - 1
    assert len(result) == 1
    assert result.iloc[0]['return'] == pytest.approx(expected)
    assert result.iloc[0]['start_month'] == months[0]
    assert result.iloc[0]['end_month'] == months[-1]

def test_calculate_total_returns():
    months = pd.period_range('2024-01', periods=6, freq='M')
    returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    monthly_df = pd.DataFrame({'month': months, 'return': returns})

    abs_ret, ann_ret = twr.calculate_total_returns(monthly_df)
    expected_abs = np.prod(1 + returns) - 1
    num_years = len(returns) / 12
    expected_ann = (1 + expected_abs) ** (1 / num_years) - 1
    assert abs_ret == pytest.approx(expected_abs)
    assert ann_ret == pytest.approx(expected_ann)

def test_calculate_irr():
    nav_df = pd.DataFrame({
        'Date': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01']),
        'Net Asset Value': [100.0, 110.0, 150.0],
    })
    trades_df = pd.DataFrame({
        'When': pd.to_datetime(['2024-01-15', '2024-02-15']),
        'EUR equivalent': [20.0, -10.0],
    })
    flows = [-100.0, -20.0, 10.0, 150.0]
    irr = npf.irr(flows)
    expected = (1 + irr) ** 365 - 1
    result = twr.calculate_irr(nav_df, trades_df)
    assert result == pytest.approx(expected)

def test_build_gips_composite():
    acc1 = pd.DataFrame({
        'month': pd.period_range('2024-01', periods=2, freq='M'),
        'return': [0.1, 0.2],
        'start_of_month_nav': [100.0, 120.0],
    })
    acc2 = pd.DataFrame({
        'month': pd.period_range('2024-01', periods=2, freq='M'),
        'return': [0.05, 0.0],
        'start_of_month_nav': [200.0, 200.0],
    })

    composite, _ = twr.build_gips_composite({'acc1': acc1, 'acc2': acc2})
    expected_returns = [
        (100/300)*0.1 + (200/300)*0.05,
        (120/320)*0.2 + (200/320)*0.0,
    ]
    expected_growth = [
        (1 + expected_returns[0]) - 1,
        (1 + expected_returns[0]) * (1 + expected_returns[1]) - 1,
    ]

    assert composite['composite_return'].tolist() == pytest.approx(expected_returns)
    assert composite['composite_growth'].tolist() == pytest.approx(expected_growth)

def test_calculate_composite_irr():
    nav_df = pd.DataFrame({
        'Date': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01']),
        'Net Asset Value': [100.0, 110.0, 150.0],
    })
    trades_df = pd.DataFrame({
        'When': pd.to_datetime(['2024-01-15', '2024-02-15']),
        'EUR equivalent': [20.0, -10.0],
    })
    sub_returns = twr.calculate_sub_period_returns(nav_df, trades_df)
    client_data = [{'sub_period_returns': sub_returns, 'trades': trades_df}]
    flows = [-100.0, -20.0, 10.0, 150.0]
    irr = npf.irr(flows)
    expected = (1 + irr) ** 365 - 1
    result = twr.calculate_composite_irr(client_data)
    assert result is not None
    assert result == pytest.approx(expected)
