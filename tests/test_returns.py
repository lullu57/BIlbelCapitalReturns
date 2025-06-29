import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, PROJECT_ROOT)
from pathlib import Path
import utils
import Exante
import IBKR
import importlib
import pandas as pd
import numpy as np
import numpy_financial as npf
import pytest

# Ensure the production module is properly named
try:
    import twr_calculator as twr
except ImportError as exc:
    raise ImportError("Rename 'TWR Calculator.py' to 'twr_calculator.py'") from exc


def test_old_name_import_fails():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module('TWR Calculator')


def test_calculate_sub_period_returns_basic(nav_df, trades_df):
    result = twr.calculate_sub_period_returns(nav_df, trades_df)
    expected_returns = [-0.0833333333, 0.5]
    assert result['return'].tolist() == pytest.approx(expected_returns)
    assert list(result['start_date']) == list(pd.to_datetime(['2024-01-01', '2024-02-01']))
    assert list(result['end_date']) == list(pd.to_datetime(['2024-02-01', '2024-03-01']))


def test_calculate_monthly_twr_basic(sub_period_returns):
    monthly = twr.calculate_monthly_twr(sub_period_returns)
    expected_months = [pd.Period('2024-01'), pd.Period('2024-02')]
    expected_returns = [-0.0833333333, 0.5]
    assert monthly['month'].tolist() == expected_months
    assert monthly['return'].tolist() == pytest.approx(expected_returns)
    assert monthly['start_of_month_nav'].tolist() == [100.0, 110.0]


def test_calculate_six_month_returns(monthly_df):
    result = twr.calculate_six_month_returns(monthly_df)
    expected = np.prod(1 + monthly_df['return']) - 1
    assert len(result) == 1
    assert result.iloc[0]['return'] == pytest.approx(expected)
    assert result.iloc[0]['start_month'] == monthly_df['month'].iloc[0]
    assert result.iloc[0]['end_month'] == monthly_df['month'].iloc[-1]


def test_calculate_total_returns(monthly_df):
    abs_ret, ann_ret = twr.calculate_total_returns(monthly_df)
    expected_abs = np.prod(1 + monthly_df['return']) - 1
    num_years = len(monthly_df) / 12
    expected_ann = (1 + expected_abs) ** (1 / num_years) - 1
    assert abs_ret == pytest.approx(expected_abs)
    assert ann_ret == pytest.approx(expected_ann)


def test_calculate_irr(nav_df, trades_df):
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


def test_calculate_composite_irr(nav_df, trades_df):
    sub_returns = twr.calculate_sub_period_returns(nav_df, trades_df)
    client_data = [{'sub_period_returns': sub_returns, 'trades': trades_df}]
    flows = [-100.0, -20.0, 10.0, 150.0]
    irr = npf.irr(flows)
    expected = (1 + irr) ** 365 - 1
    result = twr.calculate_composite_irr(client_data)
    assert result is not None
    assert result == pytest.approx(expected)


# Edge case tests

def test_empty_inputs():
    empty_nav = pd.DataFrame({'Date': pd.to_datetime([]), 'Net Asset Value': []})
    empty_trades = pd.DataFrame({'When': pd.to_datetime([]), 'EUR equivalent': []})
    with pytest.raises(KeyError):
        twr.calculate_sub_period_returns(empty_nav, empty_trades)


def test_single_nav_row(trades_df):
    nav = pd.DataFrame({'Date': pd.to_datetime(['2024-01-01']), 'Net Asset Value': [100.0]})
    with pytest.raises(KeyError):
        twr.calculate_sub_period_returns(nav, trades_df)


def test_negative_nav_warns(nav_df, trades_df, caplog):
    nav_df.loc[1, 'Net Asset Value'] = -10.0
    with caplog.at_level('WARNING'):
        twr.calculate_sub_period_returns(nav_df, trades_df)
        assert any('Invalid starting NAV' in m for m in caplog.text.splitlines())

from hypothesis import given, strategies as st

@given(st.lists(st.floats(min_value=-0.9, max_value=2, allow_nan=False), min_size=1, max_size=12))
def test_total_returns_identity(returns):
    months = pd.period_range('2024-01', periods=len(returns), freq='M')
    df = pd.DataFrame({'month': months, 'return': returns})
    abs_ret, ann_ret = twr.calculate_total_returns(df)
    expected_abs = np.prod(1 + df['return']) - 1
    num_years = len(returns) / 12
    expected_ann = (1 + expected_abs) ** (1 / num_years) - 1
    assert abs_ret == pytest.approx(expected_abs)
    assert ann_ret == pytest.approx(expected_ann)

def test_exante_process(tmp_path, monkeypatch):
    src = tmp_path / "sample.csv"
    dst_dir = tmp_path / "sample"
    data = Path(os.path.join(os.path.dirname(__file__), "fixtures", "exante_sample.csv")).read_text()
    src.write_text(data)
    monkeypatch.setattr(utils, "setup_logging", lambda *_: None)
    Exante.process(str(src))
    assert (dst_dir / "NAV.xlsx").exists()
    assert (dst_dir / "Trades.xlsx").exists()
    nav = pd.read_excel(dst_dir / "NAV.xlsx")
    trades = pd.read_excel(dst_dir / "Trades.xlsx")
    assert len(nav) > 0
    assert len(trades) > 0


def test_ibkr_process(tmp_path, monkeypatch):
    src = tmp_path / "ib.csv"
    dst_dir = tmp_path / "ib"
    data = Path(os.path.join(os.path.dirname(__file__), "fixtures", "ibkr_sample.csv")).read_text()
    src.write_text(data)
    monkeypatch.setattr(utils, "setup_logging", lambda *_: None)
    IBKR.process(str(src))
    assert (dst_dir / "allocation_by_asset_class.csv").exists()
    assert (dst_dir / "deposits_and_withdrawals.csv").exists()
    nav = pd.read_csv(dst_dir / "allocation_by_asset_class.csv")
    flows = pd.read_csv(dst_dir / "deposits_and_withdrawals.csv")
    assert len(nav) > 0
    assert len(flows) > 0
