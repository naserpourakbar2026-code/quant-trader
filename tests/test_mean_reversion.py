import pandas as pd
import pytest

from src.strategies.base import SignalDirection
from src.strategies.mean_reversion import MeanReversionStrategy


def _df(
    close,
    rsi,
    bb_lower,
    bb_upper,
    bb_middle,
    trend_regime="no_trend",
    ema_fast=100.0,
    ema_slow=100.0,
    volatility_regime="normal_vol",
    atr=1.0,
):
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
            "rsi": [rsi],
            "bb_lower": [bb_lower],
            "bb_upper": [bb_upper],
            "bb_middle": [bb_middle],
        }
    )


def test_long_on_oversold_lower_band_touch():
    strat = MeanReversionStrategy()
    df = _df(close=95.0, rsi=25.0, bb_lower=96.0, bb_upper=104.0, bb_middle=100.0)
    signal = strat.generate_signal(df)
    assert signal.direction == SignalDirection.LONG
    assert signal.take_profit == pytest.approx(100.0)  # reverts to the mean
    assert signal.stop_loss < signal.entry_price < signal.take_profit
    assert strat.validate_signal(signal, df) is True


def test_short_on_overbought_upper_band_touch():
    strat = MeanReversionStrategy()
    df = _df(close=105.0, rsi=75.0, bb_lower=96.0, bb_upper=104.0, bb_middle=100.0)
    signal = strat.generate_signal(df)
    assert signal.direction == SignalDirection.SHORT
    assert signal.take_profit == pytest.approx(100.0)
    assert signal.take_profit < signal.entry_price < signal.stop_loss
    assert strat.validate_signal(signal, df) is True


def test_blocked_when_fading_a_strong_downtrend():
    """Price at the lower band during a *strong* downtrend is exactly the
    "aggressively trading against a strong trend" case Section 5B forbids."""
    strat = MeanReversionStrategy()
    df = _df(
        close=95.0,
        rsi=25.0,
        bb_lower=96.0,
        bb_upper=104.0,
        bb_middle=100.0,
        trend_regime="strong_trend",
        ema_fast=90.0,
        ema_slow=100.0,  # fast << slow: strong downtrend
    )
    signal = strat.generate_signal(df)
    assert signal.direction == SignalDirection.FLAT


def test_blocked_when_fading_a_strong_uptrend():
    strat = MeanReversionStrategy()
    df = _df(
        close=105.0,
        rsi=75.0,
        bb_lower=96.0,
        bb_upper=104.0,
        bb_middle=100.0,
        trend_regime="strong_trend",
        ema_fast=110.0,
        ema_slow=100.0,  # fast >> slow: strong uptrend
    )
    signal = strat.generate_signal(df)
    assert signal.direction == SignalDirection.FLAT


def test_allowed_against_only_a_weak_trend():
    """Section 5B only requires avoiding *strong* trends; a weak one
    shouldn't block a mean-reversion entry."""
    strat = MeanReversionStrategy()
    df = _df(
        close=95.0,
        rsi=25.0,
        bb_lower=96.0,
        bb_upper=104.0,
        bb_middle=100.0,
        trend_regime="weak_trend",
        ema_fast=98.0,
        ema_slow=100.0,
    )
    signal = strat.generate_signal(df)
    assert signal.direction == SignalDirection.LONG


def test_flat_when_price_inside_bands():
    strat = MeanReversionStrategy()
    df = _df(close=100.0, rsi=50.0, bb_lower=96.0, bb_upper=104.0, bb_middle=100.0)
    signal = strat.generate_signal(df)
    assert signal.direction == SignalDirection.FLAT


def test_flat_when_band_touched_but_rsi_not_confirming():
    strat = MeanReversionStrategy()
    df = _df(close=95.0, rsi=45.0, bb_lower=96.0, bb_upper=104.0, bb_middle=100.0)
    signal = strat.generate_signal(df)
    assert signal.direction == SignalDirection.FLAT


def test_confidence_increases_with_rsi_extremity():
    strat = MeanReversionStrategy()
    mild = strat.generate_signal(_df(close=95.0, rsi=29.0, bb_lower=96.0, bb_upper=104.0, bb_middle=100.0))
    extreme = strat.generate_signal(_df(close=95.0, rsi=5.0, bb_lower=96.0, bb_upper=104.0, bb_middle=100.0))
    assert extreme.confidence > mild.confidence
