import pandas as pd
import pytest

from src.strategies.base import SignalDirection
from src.strategies.trend_following import TrendFollowingStrategy


def _df(trend_regime, ema_fast, ema_slow, breakout_strength, volatility_regime="normal_vol", atr=1.0, close=100.0):
    return pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-02T00:00:00Z")],
            "symbol": ["EURUSD"],
            "timeframe": ["H1"],
            "close": [close],
            "atr": [atr],
            "ema_fast": [ema_fast],
            "ema_slow": [ema_slow],
            "trend_regime": [trend_regime],
            "volatility_regime": [volatility_regime],
            "breakout_strength": [breakout_strength],
        }
    )


def test_long_on_strong_uptrend_with_upside_breakout():
    strat = TrendFollowingStrategy()
    df = _df(trend_regime="strong_trend", ema_fast=101.0, ema_slow=100.0, breakout_strength=1.5)
    signal = strat.generate_signal(df)
    assert signal.direction == SignalDirection.LONG
    assert signal.confidence == pytest.approx(0.8)
    assert signal.stop_loss < signal.entry_price < signal.take_profit
    assert strat.validate_signal(signal, df) is True


def test_short_on_weak_downtrend_with_downside_breakout():
    strat = TrendFollowingStrategy()
    df = _df(trend_regime="weak_trend", ema_fast=99.0, ema_slow=100.0, breakout_strength=-1.2)
    signal = strat.generate_signal(df)
    assert signal.direction == SignalDirection.SHORT
    assert signal.confidence == pytest.approx(0.5)
    assert signal.take_profit < signal.entry_price < signal.stop_loss
    assert strat.validate_signal(signal, df) is True


def test_flat_when_no_trend():
    strat = TrendFollowingStrategy()
    df = _df(trend_regime="no_trend", ema_fast=101.0, ema_slow=100.0, breakout_strength=1.5)
    signal = strat.generate_signal(df)
    assert signal.direction == SignalDirection.FLAT
    assert signal.entry_price is None


def test_flat_when_trend_undefined_during_warmup():
    strat = TrendFollowingStrategy()
    df = _df(trend_regime=float("nan"), ema_fast=101.0, ema_slow=100.0, breakout_strength=1.5)
    signal = strat.generate_signal(df)
    assert signal.direction == SignalDirection.FLAT


def test_flat_when_uptrend_but_no_breakout_yet():
    strat = TrendFollowingStrategy()
    df = _df(trend_regime="strong_trend", ema_fast=101.0, ema_slow=100.0, breakout_strength=0.0)
    signal = strat.generate_signal(df)
    assert signal.direction == SignalDirection.FLAT


def test_flat_when_low_volatility_blocks_breakout_by_default():
    strat = TrendFollowingStrategy()
    df = _df(
        trend_regime="strong_trend",
        ema_fast=101.0,
        ema_slow=100.0,
        breakout_strength=1.5,
        volatility_regime="low_vol",
    )
    signal = strat.generate_signal(df)
    assert signal.direction == SignalDirection.FLAT


def test_low_volatility_filter_can_be_disabled_via_params():
    strat = TrendFollowingStrategy(params={"require_non_low_vol": False})
    df = _df(
        trend_regime="strong_trend",
        ema_fast=101.0,
        ema_slow=100.0,
        breakout_strength=1.5,
        volatility_regime="low_vol",
    )
    signal = strat.generate_signal(df)
    assert signal.direction == SignalDirection.LONG


def test_stop_loss_and_take_profit_use_configured_multipliers():
    strat = TrendFollowingStrategy(params={"atr_stop_multiplier": 3.0, "take_profit_r_multiple": 1.5})
    df = _df(trend_regime="strong_trend", ema_fast=101.0, ema_slow=100.0, breakout_strength=1.5, atr=2.0, close=100.0)
    signal = strat.generate_signal(df)
    assert signal.stop_loss == pytest.approx(100.0 - 3.0 * 2.0)
    risk = 100.0 - signal.stop_loss
    assert signal.take_profit == pytest.approx(100.0 + 1.5 * risk)
