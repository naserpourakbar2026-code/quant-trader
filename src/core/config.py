"""Configuration loading and validation.

Loads config/settings.yaml, config/strategies.yaml and config/brokers.yaml
into validated pydantic models. No trading-relevant value should be
hard-coded elsewhere in the codebase — add it to a YAML file and a field
here instead.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


class CooldownConfig(BaseModel):
    losses: int = Field(gt=0)
    cooldown_bars: int = Field(gt=0)


class RiskConfig(BaseModel):
    risk_per_trade: float = Field(gt=0, le=0.05)
    allowed_risk_levels: list[float]
    max_daily_loss: float = Field(gt=0, le=1)
    max_weekly_loss: float = Field(gt=0, le=1)
    max_portfolio_drawdown: float = Field(gt=0, le=1)
    max_open_positions: int = Field(gt=0)
    max_correlated_exposure: float = Field(gt=0, le=1)
    max_leverage: float = Field(gt=0)
    max_margin_usage: float = Field(gt=0, le=1)
    daily_trade_limit: int = Field(gt=0)
    cooldown_after_consecutive_losses: CooldownConfig

    @field_validator("risk_per_trade")
    @classmethod
    def risk_per_trade_must_be_allowed(cls, v: float, info: Any) -> float:
        # allowed_risk_levels may not be populated yet during this validator
        # (pydantic v2 validates fields in declaration order); re-checked
        # again below in AppConfig.
        return v


class AccountConfig(BaseModel):
    initial_capital: float = Field(gt=0)
    currency: str


class DataConfig(BaseModel):
    symbols: list[str]
    optional_symbols: list[str] = Field(default_factory=list)
    timeframes: list[str]
    source: str

    @field_validator("source")
    @classmethod
    def source_must_be_supported(cls, v: str) -> str:
        allowed = {"csv", "mt5"}
        if v not in allowed:
            raise ValueError(f"data.source must be one of {allowed}, got {v!r}")
        return v


class ExecutionConfig(BaseModel):
    scenarios: list[str]
    default_scenario: str

    @field_validator("default_scenario")
    @classmethod
    def default_scenario_in_scenarios(cls, v: str, info: Any) -> str:
        scenarios = info.data.get("scenarios", [])
        if scenarios and v not in scenarios:
            raise ValueError(f"default_scenario {v!r} not in scenarios {scenarios}")
        return v


class PathsConfig(BaseModel):
    data_raw: str
    data_processed: str
    data_cache: str
    reports: str
    logs: str


class LoggingConfig(BaseModel):
    level: str = "INFO"
    trade_log_file: str
    system_log_file: str


class AppConfig(BaseModel):
    account: AccountConfig
    risk: RiskConfig
    data: DataConfig
    execution: ExecutionConfig
    live_trading: bool
    paths: PathsConfig
    logging: LoggingConfig

    @field_validator("risk")
    @classmethod
    def risk_per_trade_allowed(cls, v: RiskConfig) -> RiskConfig:
        if v.allowed_risk_levels and v.risk_per_trade not in v.allowed_risk_levels:
            raise ValueError(
                f"risk_per_trade {v.risk_per_trade} not in allowed_risk_levels "
                f"{v.allowed_risk_levels}"
            )
        return v


class StrategiesConfig(BaseModel):
    stop_loss_types: list[str]
    take_profit_ratios: list[float]
    strategies: dict[str, Any] = Field(default_factory=dict)


class BrokerEntryConfig(BaseModel):
    enabled: bool = False
    name: str | None = None
    environment: str = "demo"

    model_config = {"extra": "allow"}


class BrokersConfig(BaseModel):
    brokers: dict[str, BrokerEntryConfig]
    active_broker: str | None = None

    @field_validator("active_broker")
    @classmethod
    def active_broker_must_exist(cls, v: str | None, info: Any) -> str | None:
        if v is None:
            return v
        brokers = info.data.get("brokers", {})
        if v not in brokers:
            raise ValueError(f"active_broker {v!r} not defined in brokers.yaml")
        return v


def is_live_trading_enabled() -> bool:
    """Live trading requires BOTH env flags to be true. Never on by default."""
    live = os.getenv("LIVE_TRADING", "false").strip().lower() == "true"
    confirmed = os.getenv("LIVE_CONFIRMATION", "false").strip().lower() == "true"
    return live and confirmed


@lru_cache(maxsize=1)
def load_settings(config_dir: Path | None = None) -> AppConfig:
    directory = config_dir or CONFIG_DIR
    raw = _load_yaml(directory / "settings.yaml")
    return AppConfig(**raw)


@lru_cache(maxsize=1)
def load_strategies(config_dir: Path | None = None) -> StrategiesConfig:
    directory = config_dir or CONFIG_DIR
    raw = _load_yaml(directory / "strategies.yaml")
    return StrategiesConfig(**raw)


@lru_cache(maxsize=1)
def load_brokers(config_dir: Path | None = None) -> BrokersConfig:
    directory = config_dir or CONFIG_DIR
    raw = _load_yaml(directory / "brokers.yaml")
    return BrokersConfig(**raw)
