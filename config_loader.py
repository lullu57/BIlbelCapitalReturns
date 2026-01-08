"""
Configuration loader for the GIPS-Compliant Returns Calculator.

Loads and validates configuration from config.yaml file.
"""

import os
import yaml
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime


@dataclass
class BrokerageConfig:
    """Configuration for a single brokerage."""
    name: str
    module: str
    enabled: bool = True


@dataclass
class FeeConfig:
    """Fee structure configuration."""
    management_fee_quarterly: float = 0.0025  # 0.25% per quarter (1% annual)
    performance_fee_rate: float = 0.25   # 25% of gains above hurdle
    hurdle_rate_annual: float = 0.06     # 6% annual hurdle
    # Keep annual for backward compatibility
    management_fee_annual: float = 0.01  # 1% annual (derived from quarterly)
    
    def __post_init__(self):
        # Derive annual from quarterly if not explicitly set
        if self.management_fee_quarterly > 0:
            self.management_fee_annual = self.management_fee_quarterly * 4
    
    @property
    def management_fee_monthly(self) -> float:
        """Convert annual fee to monthly equivalent: (1 + r)^(1/12) - 1"""
        return (1 + self.management_fee_annual) ** (1/12) - 1


@dataclass
class ThresholdsConfig:
    """Threshold configuration."""
    large_cash_flow_pct: float = 0.10  # 10% of portfolio


@dataclass
class PeriodsConfig:
    """Reporting periods configuration."""
    fiscal_year_start_month: int = 2  # February
    reporting_years: List[int] = field(default_factory=lambda: [2022, 2023, 2024, 2025])
    
    def get_period_windows(self) -> Dict[str, Tuple[str, str]]:
        """
        Generate period windows based on fiscal year config.
        
        Returns dict like:
        {
            '2022': ('2022-02-01', '2023-01-31'),
            '2023': ('2023-02-01', '2024-01-31'),
            ...
        }
        """
        windows = {}
        start_month = self.fiscal_year_start_month
        
        for year in self.reporting_years:
            # Fiscal year starts in start_month of year and ends in (start_month - 1) of year + 1
            start_date = f"{year}-{start_month:02d}-01"
            
            # End month is start_month - 1, or 12 if start_month is 1
            end_month = start_month - 1 if start_month > 1 else 12
            end_year = year + 1 if start_month > 1 else year
            
            # Get last day of end month
            if end_month in [1, 3, 5, 7, 8, 10, 12]:
                end_day = 31
            elif end_month in [4, 6, 9, 11]:
                end_day = 30
            else:  # February
                # Check for leap year
                end_day = 29 if (end_year % 4 == 0 and (end_year % 100 != 0 or end_year % 400 == 0)) else 28
            
            end_date = f"{end_year}-{end_month:02d}-{end_day:02d}"
            
            # Use year as key, or year_ytd if it's the current/future period
            current_year = datetime.now().year
            if year >= current_year:
                windows[f'{year}_ytd'] = (start_date, end_date)
            else:
                windows[str(year)] = (start_date, end_date)
        
        return windows


@dataclass
class PathsConfig:
    """Path configuration."""
    input_dir: str = 'input'
    output_dir: str = 'results'
    log_dir: str = 'logs'


@dataclass
class CurrencyConfig:
    """Currency configuration."""
    base_currency: str = 'EUR'
    flow_columns: List[str] = field(default_factory=lambda: ['Adjusted EUR', 'EUR equivalent'])


@dataclass
class Config:
    """Main configuration class."""
    brokerages: List[BrokerageConfig]
    fees: FeeConfig
    thresholds: ThresholdsConfig
    periods: PeriodsConfig
    paths: PathsConfig
    currency: CurrencyConfig
    
    def get_enabled_brokerages(self) -> List[BrokerageConfig]:
        """Return list of enabled brokerages."""
        return [b for b in self.brokerages if b.enabled]
    
    def get_brokerage_names(self) -> List[str]:
        """Return list of enabled brokerage names."""
        return [b.name for b in self.get_enabled_brokerages()]


def load_config(config_path: str = 'config.yaml') -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Config object with validated configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    if not os.path.exists(config_path):
        logging.warning(f"Config file {config_path} not found, using defaults")
        return _get_default_config()
    
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    
    return _parse_config(raw_config)


def _get_default_config() -> Config:
    """Return default configuration."""
    return Config(
        brokerages=[
            BrokerageConfig(name='IBKR', module='IBKR', enabled=True),
            BrokerageConfig(name='Exante', module='Exante', enabled=True),
        ],
        fees=FeeConfig(),
        thresholds=ThresholdsConfig(),
        periods=PeriodsConfig(),
        paths=PathsConfig(),
        currency=CurrencyConfig(),
    )


def _parse_config(raw: dict) -> Config:
    """Parse raw YAML dict into Config object."""
    # Parse brokerages
    brokerages = []
    for b in raw.get('brokerages', []):
        brokerages.append(BrokerageConfig(
            name=b.get('name', ''),
            module=b.get('module', ''),
            enabled=b.get('enabled', True),
        ))
    
    # Parse fees
    fees_raw = raw.get('fees', {})
    fees = FeeConfig(
        management_fee_annual=fees_raw.get('management_fee_annual', 0.01),
        performance_fee_rate=fees_raw.get('performance_fee_rate', 0.25),
        hurdle_rate_annual=fees_raw.get('hurdle_rate_annual', 0.06),
    )
    
    # Parse thresholds
    thresholds_raw = raw.get('thresholds', {})
    thresholds = ThresholdsConfig(
        large_cash_flow_pct=thresholds_raw.get('large_cash_flow_pct', 0.10),
    )
    
    # Parse periods
    periods_raw = raw.get('periods', {})
    periods = PeriodsConfig(
        fiscal_year_start_month=periods_raw.get('fiscal_year_start_month', 2),
        reporting_years=periods_raw.get('reporting_years', [2022, 2023, 2024, 2025]),
    )
    
    # Parse paths
    paths_raw = raw.get('paths', {})
    paths = PathsConfig(
        input_dir=paths_raw.get('input_dir', 'input'),
        output_dir=paths_raw.get('output_dir', 'results'),
        log_dir=paths_raw.get('log_dir', 'logs'),
    )
    
    # Parse currency
    currency_raw = raw.get('currency', {})
    currency = CurrencyConfig(
        base_currency=currency_raw.get('base_currency', 'EUR'),
        flow_columns=currency_raw.get('flow_columns', ['Adjusted EUR', 'EUR equivalent']),
    )
    
    return Config(
        brokerages=brokerages,
        fees=fees,
        thresholds=thresholds,
        periods=periods,
        paths=paths,
        currency=currency,
    )


def validate_config(config: Config) -> List[str]:
    """
    Validate configuration and return list of warnings/errors.
    
    Args:
        config: Configuration to validate
        
    Returns:
        List of warning/error messages (empty if valid)
    """
    issues = []
    
    # Validate fees
    if config.fees.management_fee_annual < 0 or config.fees.management_fee_annual > 0.10:
        issues.append(f"Management fee {config.fees.management_fee_annual} seems unusual (expected 0-10%)")
    
    if config.fees.performance_fee_rate < 0 or config.fees.performance_fee_rate > 0.50:
        issues.append(f"Performance fee rate {config.fees.performance_fee_rate} seems unusual (expected 0-50%)")
    
    if config.fees.hurdle_rate_annual < 0 or config.fees.hurdle_rate_annual > 0.20:
        issues.append(f"Hurdle rate {config.fees.hurdle_rate_annual} seems unusual (expected 0-20%)")
    
    # Validate thresholds
    if config.thresholds.large_cash_flow_pct < 0.01 or config.thresholds.large_cash_flow_pct > 0.50:
        issues.append(f"Large cash flow threshold {config.thresholds.large_cash_flow_pct} seems unusual (expected 1-50%)")
    
    # Validate periods
    if config.periods.fiscal_year_start_month < 1 or config.periods.fiscal_year_start_month > 12:
        issues.append(f"Invalid fiscal year start month: {config.periods.fiscal_year_start_month}")
    
    # Validate brokerages have required fields
    for b in config.brokerages:
        if not b.name:
            issues.append("Brokerage missing name")
        if not b.module:
            issues.append(f"Brokerage {b.name} missing module")
    
    return issues

