import pandas as pd
import pytest

@pytest.fixture
def nav_df():
    return pd.DataFrame({
        'Date': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01']),
        'Net Asset Value': [100.0, 110.0, 150.0],
    })

@pytest.fixture
def trades_df():
    return pd.DataFrame({
        'When': pd.to_datetime(['2024-01-15', '2024-02-15']),
        'EUR equivalent': [20.0, -10.0],
    })

@pytest.fixture
def sub_period_returns(nav_df, trades_df):
    from twr_calculator import calculate_sub_period_returns
    return calculate_sub_period_returns(nav_df, trades_df)

@pytest.fixture
def monthly_df():
    months = pd.period_range('2024-01', periods=6, freq='M')
    returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    return pd.DataFrame({'month': months, 'return': returns})
