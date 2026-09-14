import pytest
from pydantic import ValidationError

from src.core.config import (
    AppConfig,
    BrokersConfig,
    StrategiesConfig,
    is_live_trading_enabled,
    load_brokers,
    load_settings,
    load_strategies,
)


def test_load_settings_returns_valid_app_config():
    settings = load_settings()
    assert isinstance(settings, AppConfig)
    assert settings.account.initial_capital == 2000
    assert settings.risk.risk_per_trade == 0.005
    assert settings.live_trading is False


def test_risk_per_trade_must_be_in_allowed_levels():
    with pytest.raises(ValidationError):
        AppConfig(
            account={"initial_capital": 2000, "currency": "EUR"},
            risk={
                "risk_per_trade": 0.02,  # not in allowed_risk_levels
                "allowed_risk_levels": [0.0025, 0.005, 0.0075, 0.01],
                "max_daily_loss": 0.03,
                "max_weekly_loss": 0.08,
                "max_portfolio_drawdown": 0.15,
                "max_open_positions": 5,
                "max_correlated_exposure": 0.02,
                "max_leverage": 30,
                "max_margin_usage": 0.5,
                "daily_trade_limit": 10,
                "cooldown_after_consecutive_losses": {"losses": 3, "cooldown_bars": 12},
            },
            data={
                "symbols": ["EURUSD"],
                "timeframes": ["H1"],
                "source": "csv",
            },
            execution={"scenarios": ["realistic"], "default_scenario": "realistic"},
            live_trading=False,
            paths={
                "data_raw": "data/raw",
                "data_processed": "data/processed",
                "data_cache": "data/cache",
                "reports": "reports",
                "logs": "logs",
            },
            logging={"trade_log_file": "logs/trades.log", "system_log_file": "logs/system.log"},
        )


def test_data_source_rejects_unsupported_value():
    settings = load_settings()
    with pytest.raises(ValidationError):
        type(settings.data)(**{**settings.data.model_dump(), "source": "unsupported"})


def test_load_strategies_returns_empty_registry_before_phase5():
    strategies = load_strategies()
    assert isinstance(strategies, StrategiesConfig)
    assert strategies.strategies == {}
    assert "atr" in strategies.stop_loss_types


def test_load_brokers_disabled_by_default():
    brokers = load_brokers()
    assert isinstance(brokers, BrokersConfig)
    assert brokers.active_broker is None
    for entry in brokers.brokers.values():
        assert entry.enabled is False


def test_live_trading_disabled_by_default(monkeypatch):
    monkeypatch.delenv("LIVE_TRADING", raising=False)
    monkeypatch.delenv("LIVE_CONFIRMATION", raising=False)
    assert is_live_trading_enabled() is False


def test_live_trading_requires_both_flags(monkeypatch):
    monkeypatch.setenv("LIVE_TRADING", "true")
    monkeypatch.delenv("LIVE_CONFIRMATION", raising=False)
    assert is_live_trading_enabled() is False

    monkeypatch.setenv("LIVE_CONFIRMATION", "true")
    assert is_live_trading_enabled() is True
