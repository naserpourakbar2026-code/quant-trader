import math

import pandas as pd
import pytest

from src.features import regime


def _ramp_ema(n: int, slope_per_bar: float) -> pd.Series:
    return pd.Series(range(n)).astype(float) * slope_per_bar


def test_classify_trend_strong_when_aligned_and_slope_well_above_threshold():
    n = 15
    ema_slow = _ramp_ema(n, slope_per_bar=2.0)  # diff(5) = 10 per window
    ema_fast = ema_slow + 1.0  # fast above slow: aligned with the positive slope
    atr = pd.Series([1.0] * n)  # normalized_slope = 10 / (1*5) = 2.0 >= strong_threshold
    label = regime.classify_trend(
        ema_fast, ema_slow, atr, slope_lookback=5, strong_threshold=1.0, weak_threshold=0.3
    )
    assert label.iloc[:5].isna().all()
    assert (label.iloc[5:] == regime.TREND_STRONG).all()


def test_classify_trend_weak_when_slope_between_thresholds():
    n = 15
    ema_slow = _ramp_ema(n, slope_per_bar=0.5)  # diff(5) = 2.5 per window
    ema_fast = ema_slow + 1.0
    atr = pd.Series([1.0] * n)  # normalized_slope = 2.5 / (1*5) = 0.5, in [0.3, 1.0)
    label = regime.classify_trend(
        ema_fast, ema_slow, atr, slope_lookback=5, strong_threshold=1.0, weak_threshold=0.3
    )
    assert (label.iloc[5:] == regime.TREND_WEAK).all()


def test_classify_trend_none_when_flat():
    n = 15
    ema_slow = pd.Series([100.0] * n)
    ema_fast = pd.Series([100.0] * n)
    atr = pd.Series([1.0] * n)
    label = regime.classify_trend(
        ema_fast, ema_slow, atr, slope_lookback=5, strong_threshold=1.0, weak_threshold=0.3
    )
    assert (label.iloc[5:] == regime.TREND_NONE).all()


def test_classify_trend_none_when_fast_disagrees_with_slope_direction():
    """A rising slow EMA with price currently below it (fast < slow) is not
    a confirmed trend structure — even though the slope magnitude alone
    would qualify as strong."""
    n = 15
    ema_slow = _ramp_ema(n, slope_per_bar=2.0)
    ema_fast = ema_slow - 1.0  # disagrees with the positive slope direction
    atr = pd.Series([1.0] * n)
    label = regime.classify_trend(
        ema_fast, ema_slow, atr, slope_lookback=5, strong_threshold=1.0, weak_threshold=0.3
    )
    assert (label.iloc[5:] == regime.TREND_NONE).all()


def test_classify_volatility_flags_spike_as_high_and_baseline_as_normal():
    atr = pd.Series([1.0] * 60 + [10.0] * 5)
    label = regime.classify_volatility(atr, lookback=50, low_percentile=25, high_percentile=75)
    assert label.iloc[:49].isna().all()
    assert label.iloc[60:65].tolist() == [regime.VOL_HIGH] * 5


def test_classify_volatility_flags_low_percentile():
    atr = pd.Series([5.0] * 60 + [0.1] * 5)
    label = regime.classify_volatility(atr, lookback=50, low_percentile=25, high_percentile=75)
    assert label.iloc[60:65].tolist() == [regime.VOL_LOW] * 5


def test_classify_range_true_on_squeeze_false_when_trending_nan_during_warmup():
    trend_label = pd.Series(["no_trend"] * 6 + ["weak_trend"] * 4)
    width = pd.Series([5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    result = regime.classify_range(trend_label, width, squeeze_lookback=5)

    assert result.iloc[:4].isna().all()
    assert result.iloc[5] is True or result.iloc[5] == True  # noqa: E712 -- squeeze while no_trend
    assert result.iloc[6] is False or result.iloc[6] == False  # noqa: E712 -- trend present now


def test_classify_range_nan_when_trend_undefined():
    trend_label = pd.Series([float("nan")] * 10)
    width = pd.Series([1.0] * 10)
    result = regime.classify_range(trend_label, width, squeeze_lookback=5)
    assert all(v is None or (isinstance(v, float) and math.isnan(v)) for v in result)
