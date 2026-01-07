"""Tests for the configuration loader module."""

import os
import sys
import tempfile
import pytest
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, PROJECT_ROOT)

from config_loader import (
    load_config,
    Config,
    BrokerageConfig,
    FeeConfig,
    ThresholdsConfig,
    PeriodsConfig,
    PathsConfig,
    CurrencyConfig,
    validate_config,
    _get_default_config,
    _parse_config,
)


class TestFeeConfig:
    """Tests for FeeConfig class."""
    
    def test_default_values(self):
        """Test default fee configuration values."""
        fee_config = FeeConfig()
        assert fee_config.management_fee_annual == 0.01
        assert fee_config.performance_fee_rate == 0.25
        assert fee_config.hurdle_rate_annual == 0.06
    
    def test_monthly_fee_conversion(self):
        """Test annual to monthly fee conversion."""
        fee_config = FeeConfig(management_fee_annual=0.12)  # 12% annual
        monthly = fee_config.management_fee_monthly
        # Monthly should be approximately 0.12/12 but geometric
        assert monthly > 0
        assert monthly < 0.12 / 12 * 1.1  # Should be close to simple division
        # Verify compounding: (1 + monthly)^12 ≈ 1.12
        assert abs((1 + monthly) ** 12 - 1.12) < 0.001
    
    def test_one_percent_monthly(self):
        """Test 1% annual fee converts correctly."""
        fee_config = FeeConfig(management_fee_annual=0.01)
        monthly = fee_config.management_fee_monthly
        # (1 + monthly)^12 should equal 1.01
        assert abs((1 + monthly) ** 12 - 1.01) < 0.0001


class TestPeriodsConfig:
    """Tests for PeriodsConfig class."""
    
    def test_default_values(self):
        """Test default period configuration."""
        config = PeriodsConfig()
        assert config.fiscal_year_start_month == 2  # February
        assert 2022 in config.reporting_years
        assert 2023 in config.reporting_years
    
    def test_get_period_windows_february_start(self):
        """Test period windows with February fiscal year start."""
        config = PeriodsConfig(
            fiscal_year_start_month=2,
            reporting_years=[2022, 2023]
        )
        windows = config.get_period_windows()
        
        # 2022 fiscal year: Feb 1, 2022 - Jan 31, 2023
        assert '2022' in windows
        assert windows['2022'][0] == '2022-02-01'
        assert windows['2022'][1] == '2023-01-31'
        
        # 2023 fiscal year: Feb 1, 2023 - Jan 31, 2024
        assert '2023' in windows
        assert windows['2023'][0] == '2023-02-01'
        assert windows['2023'][1] == '2024-01-31'
    
    def test_get_period_windows_january_start(self):
        """Test period windows with January fiscal year start (calendar year)."""
        config = PeriodsConfig(
            fiscal_year_start_month=1,
            reporting_years=[2023]
        )
        windows = config.get_period_windows()
        
        # 2023 calendar year: Jan 1, 2023 - Dec 31, 2023
        assert '2023' in windows
        assert windows['2023'][0] == '2023-01-01'
        assert windows['2023'][1] == '2023-12-31'


class TestBrokerageConfig:
    """Tests for BrokerageConfig class."""
    
    def test_basic_config(self):
        """Test basic brokerage configuration."""
        config = BrokerageConfig(name='IBKR', module='IBKR', enabled=True)
        assert config.name == 'IBKR'
        assert config.module == 'IBKR'
        assert config.enabled is True
    
    def test_disabled_brokerage(self):
        """Test disabled brokerage configuration."""
        config = BrokerageConfig(name='Test', module='test', enabled=False)
        assert config.enabled is False


class TestLoadConfig:
    """Tests for load_config function."""
    
    def test_load_default_when_file_missing(self):
        """Test loading default config when file doesn't exist."""
        config = load_config('/nonexistent/path/config.yaml')
        assert isinstance(config, Config)
        assert len(config.brokerages) > 0
    
    def test_load_from_yaml(self):
        """Test loading config from a YAML file."""
        yaml_content = {
            'brokerages': [
                {'name': 'TestBroker', 'module': 'test_broker', 'enabled': True}
            ],
            'fees': {
                'management_fee_annual': 0.02,
                'performance_fee_rate': 0.20,
                'hurdle_rate_annual': 0.05
            },
            'thresholds': {
                'large_cash_flow_pct': 0.15
            },
            'periods': {
                'fiscal_year_start_month': 1,
                'reporting_years': [2024]
            },
            'paths': {
                'input_dir': 'test_input',
                'output_dir': 'test_output',
                'log_dir': 'test_logs'
            },
            'currency': {
                'base_currency': 'USD',
                'flow_columns': ['Amount', 'USD Amount']
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_content, f)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            
            assert len(config.brokerages) == 1
            assert config.brokerages[0].name == 'TestBroker'
            assert config.fees.management_fee_annual == 0.02
            assert config.fees.performance_fee_rate == 0.20
            assert config.thresholds.large_cash_flow_pct == 0.15
            assert config.periods.fiscal_year_start_month == 1
            assert config.paths.input_dir == 'test_input'
            assert config.currency.base_currency == 'USD'
        finally:
            os.unlink(temp_path)
    
    def test_get_enabled_brokerages(self):
        """Test filtering for enabled brokerages."""
        config = _get_default_config()
        enabled = config.get_enabled_brokerages()
        assert all(b.enabled for b in enabled)


class TestValidateConfig:
    """Tests for config validation."""
    
    def test_valid_config(self):
        """Test that valid config produces no warnings."""
        config = _get_default_config()
        issues = validate_config(config)
        assert len(issues) == 0
    
    def test_unusual_management_fee(self):
        """Test warning for unusual management fee."""
        config = _get_default_config()
        config.fees.management_fee_annual = 0.50  # 50% - too high
        issues = validate_config(config)
        assert any('Management fee' in issue for issue in issues)
    
    def test_invalid_fiscal_month(self):
        """Test warning for invalid fiscal year start month."""
        config = _get_default_config()
        config.periods.fiscal_year_start_month = 13  # Invalid
        issues = validate_config(config)
        assert any('fiscal year start month' in issue for issue in issues)

