import pytest
from pydantic import ValidationError

from src.core.config import (
    AppConfig,
    BrokersConfig,
    FeaturesConfig,
    OptimizationConfig,
    ParamSpec,
    StrategiesConfig,
    is_live_trading_enabled,
    load_brokers,
    load_settings,
    load_strategies,
)

VALID_FEATURES_KWARGS = {
    "ema_fast_period": 20,
    "ema_slow_period": 50,
    "atr_period": 14,
    "donchian_period": 20,
    "bollinger_period": 20,
    "bollinger_std": 2.0,
    "rsi_period": 14,
    "adx_period": 14,
    "momentum_period": 10,
    "trend_slope_lookback": 10,
    "trend_strong_threshold": 1.0,
    "trend_weak_threshold": 0.3,
    "volatility_lookback": 100,
    "volatility_low_percentile": 25.0,
    "volatility_high_percentile": 75.0,
    "range_squeeze_lookback": 20,
}


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
            execution={
                "scenarios": ["realistic"],
                "default_scenario": "realistic",
                "costs": {"realistic": {"commission_pct": 0.00007, "slippage_pct": 0.0001}},
            },
            validation={
                "weekend_close_day": 4,
                "weekend_close_hour": 21,
                "weekend_open_day": 6,
                "weekend_open_hour": 21,
                "spread_outlier_zscore": 3.0,
            },
            features=VALID_FEATURES_KWARGS,
            optimization={
                "n_trials": 50,
                "min_trades": 10,
                "random_seed": 42,
                "weights": {"profit_factor_cap": 5.0, "drawdown_penalty": 10.0, "trade_count_penalty": 0.5},
                "stability": {"neighbor_step_pct": 0.1, "neighbor_step_int": 1},
            },
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


def test_features_config_accepts_valid_defaults():
    cfg = FeaturesConfig(**VALID_FEATURES_KWARGS)
    assert cfg.ema_slow_period > cfg.ema_fast_period


def test_features_config_rejects_slow_ema_not_greater_than_fast():
    with pytest.raises(ValidationError):
        FeaturesConfig(**{**VALID_FEATURES_KWARGS, "ema_slow_period": 10, "ema_fast_period": 20})


def test_features_config_rejects_strong_threshold_not_greater_than_weak():
    with pytest.raises(ValidationError):
        FeaturesConfig(**{**VALID_FEATURES_KWARGS, "trend_strong_threshold": 0.2, "trend_weak_threshold": 0.3})


def test_features_config_rejects_high_percentile_not_greater_than_low():
    with pytest.raises(ValidationError):
        FeaturesConfig(
            **{**VALID_FEATURES_KWARGS, "volatility_high_percentile": 20.0, "volatility_low_percentile": 25.0}
        )


def test_load_settings_loads_features_config():
    settings = load_settings()
    assert isinstance(settings.features, FeaturesConfig)
    assert settings.features.ema_slow_period > settings.features.ema_fast_period


def test_param_spec_rejects_low_not_less_than_high():
    with pytest.raises(ValidationError):
        ParamSpec(type="float", low=2.0, high=1.0)


def test_param_spec_rejects_unsupported_type():
    with pytest.raises(ValidationError):
        ParamSpec(type="str", low=0.0, high=1.0)


def test_load_settings_loads_optimization_config():
    settings = load_settings()
    assert isinstance(settings.optimization, OptimizationConfig)
    assert settings.optimization.n_trials > 0


def test_load_strategies_loads_optimization_space():
    strategies = load_strategies()
    trend = strategies.strategies["trend_following"]
    assert "atr_stop_multiplier" in trend.optimization_space
    assert trend.optimization_space["atr_stop_multiplier"].type == "float"


def test_load_strategies_returns_three_strategy_families():
    strategies = load_strategies()
    assert isinstance(strategies, StrategiesConfig)
    assert set(strategies.strategies) == {"trend_following", "mean_reversion", "momentum_multi_factor"}
    for name, definition in strategies.strategies.items():
        assert definition.family == name
        assert definition.enabled is True
        assert definition.symbols
        assert definition.timeframes
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
