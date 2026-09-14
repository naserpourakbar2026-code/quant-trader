import pandas as pd
import pytest

from src.strategies.base import SignalDirection
from src.strategies.momentum import MomentumMultiFactorStrategy


def _df(
    n=30,
    momentum_pct=0.0,
    breakout_strength=0.0,
    ema_fast=100.0,
    ema_slow=100.0,
    trend_regime="no_trend",
    last_tick_volume=100.0,
    baseline_tick_volume=100.0,
    extra_low=None,
    extra_high=None,
    close=100.0,
):
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-02", periods=n, freq="1h", tz="UTC"),
            "symbol": ["EURUSD"] * n,
            "timeframe": ["H1"] * n,
            "close": [close] * n,
            "high": [close + 2.0] * n,
            "low": [close - 2.0] * n,
            "tick_volume": [baseline_tick_volume] * n,
        }
    )
    if extra_low is not None:
        df.loc[df.index[-5], "low"] = extra_low
    if extra_high is not None:
        df.loc[df.index[-5], "high"] = extra_high
    df["momentum_pct"] = momentum_pct
    df["breakout_strength"] = breakout_strength
    df["ema_fast"] = ema_fast
    df["ema_slow"] = ema_slow
    df["trend_regime"] = trend_regime
    df.loc[df.index[-1], "tick_volume"] = last_tick_volume
    return df


def test_long_on_strong_aligned_bullish_factors():
    strat = MomentumMultiFactorStrategy()
    df = _df(
        momentum_pct=2.0,
        breakout_strength=2.0,
        ema_fast=101.0,
        ema_slow=100.0,
        trend_regime="strong_trend",
        extra_low=90.0,
    )
    signal = strat.generate_signal(df)
    assert signal.direction == SignalDirection.LONG
    assert signal.confidence == pytest.approx(1.0)
    assert signal.stop_loss == pytest.approx(90.0)  # swing low within lookback
    assert signal.take_profit == pytest.approx(100.0 + 2.0 * (100.0 - 90.0))
    assert strat.validate_signal(signal, df) is True


def test_short_on_strong_aligned_bearish_factors():
    strat = MomentumMultiFactorStrategy()
    df = _df(
        momentum_pct=-2.0,
        breakout_strength=-2.0,
        ema_fast=99.0,
        ema_slow=100.0,
        trend_regime="strong_trend",
        extra_high=110.0,
    )
    signal = strat.generate_signal(df)
    assert signal.direction == SignalDirection.SHORT
    assert signal.confidence == pytest.approx(1.0)
    assert signal.stop_loss == pytest.approx(110.0)
    assert signal.take_profit == pytest.approx(100.0 - 2.0 * (110.0 - 100.0))
    assert strat.validate_signal(signal, df) is True


def test_flat_when_factors_are_weak_and_mixed():
    strat = MomentumMultiFactorStrategy()
    df = _df(momentum_pct=0.3, breakout_strength=0.2, ema_fast=100.1, ema_slow=100.0, trend_regime="weak_trend")
    signal = strat.generate_signal(df)
    assert signal.direction == SignalDirection.FLAT


def test_volume_spike_alone_never_creates_a_signal():
    """Section 5C: tick volume may only confirm, never drive, direction —
    a huge volume spike with zero directional factors must stay FLAT."""
    strat = MomentumMultiFactorStrategy()
    df = _df(
        momentum_pct=0.0,
        breakout_strength=0.0,
        ema_fast=100.0,
        ema_slow=100.0,
        trend_regime="no_trend",
        last_tick_volume=10_000.0,
        baseline_tick_volume=100.0,
    )
    signal = strat.generate_signal(df)
    assert signal.direction == SignalDirection.FLAT


def test_volume_confirmation_can_tip_a_borderline_signal():
    """A composite score just below threshold on price factors alone can
    be tipped over by strong volume confirmation (capped at 1.2x) — but
    only because the price factors were already meaningfully positive."""
    strat = MomentumMultiFactorStrategy()
    borderline = _df(
        momentum_pct=0.675,
        breakout_strength=0.675,
        ema_fast=100.0,
        ema_slow=100.0,
        trend_regime="no_trend",  # trend_factor = 0 -> base_score = 0.45
        last_tick_volume=100.0,
        baseline_tick_volume=100.0,
    )
    assert strat.generate_signal(borderline).direction == SignalDirection.FLAT

    with_volume = _df(
        momentum_pct=0.675,
        breakout_strength=0.675,
        ema_fast=100.0,
        ema_slow=100.0,
        trend_regime="no_trend",
        last_tick_volume=1000.0,  # far above the ~100 baseline -> capped 1.2x multiplier
        baseline_tick_volume=100.0,
        extra_low=90.0,
    )
    signal = strat.generate_signal(with_volume)
    assert signal.direction == SignalDirection.LONG
    assert signal.confidence == pytest.approx(0.45 * 1.2, rel=1e-3)


def test_trend_regime_undefined_during_warmup_is_treated_as_zero_weight():
    strat = MomentumMultiFactorStrategy()
    df = _df(momentum_pct=0.6, breakout_strength=0.6, ema_fast=101.0, ema_slow=100.0, trend_regime=float("nan"))
    score = strat.composite_score_series(df).iloc[-1]
    # trend_factor contributes 0 (unknown regime -> zero weight), so score is the mean of the other two
    assert score == pytest.approx((0.6 + 0.6) / 3)
