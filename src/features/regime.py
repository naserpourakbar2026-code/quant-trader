"""Objective, rule-based regime classification (CLAUDE.md Section 5).

Every classification here is derived purely from indicators in
src/features/indicators.py using fixed, configurable thresholds — never a
subjective/manual judgment call, per the brief's requirement that trend
classification be objective.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TREND_STRONG = "strong_trend"
TREND_WEAK = "weak_trend"
TREND_NONE = "no_trend"

VOL_LOW = "low_vol"
VOL_NORMAL = "normal_vol"
VOL_HIGH = "high_vol"


def classify_trend(
    ema_fast: pd.Series,
    ema_slow: pd.Series,
    atr_series: pd.Series,
    *,
    slope_lookback: int,
    strong_threshold: float,
    weak_threshold: float,
) -> pd.Series:
    """Trend strength from the ATR-normalized slope of the slow EMA,
    counted only when the fast EMA sits on the same side as that slope's
    direction (i.e. trend structure is actually aligned, not just the slow
    average drifting on its own)."""
    slope = ema_slow.diff(slope_lookback)
    normalized_slope = slope / (atr_series * slope_lookback).replace(0, np.nan)
    aligned = np.sign(ema_fast - ema_slow) == np.sign(slope)
    magnitude = normalized_slope.abs()

    label = pd.Series(TREND_NONE, index=ema_slow.index, dtype=object)
    label = label.mask(aligned & (magnitude >= weak_threshold), TREND_WEAK)
    label = label.mask(aligned & (magnitude >= strong_threshold), TREND_STRONG)
    return label.mask(normalized_slope.isna(), np.nan)


def classify_volatility(
    atr_series: pd.Series,
    *,
    lookback: int,
    low_percentile: float,
    high_percentile: float,
) -> pd.Series:
    """Volatility regime from the ATR's own trailing rolling percentile —
    relative to its recent history, not an absolute pip threshold (which
    would not generalize across symbols)."""
    low_q = atr_series.rolling(lookback, min_periods=lookback).quantile(low_percentile / 100)
    high_q = atr_series.rolling(lookback, min_periods=lookback).quantile(high_percentile / 100)

    label = pd.Series(VOL_NORMAL, index=atr_series.index, dtype=object)
    label = label.mask(atr_series <= low_q, VOL_LOW)
    label = label.mask(atr_series >= high_q, VOL_HIGH)
    return label.mask(low_q.isna() | high_q.isna(), np.nan)


def classify_range(trend_label: pd.Series, bb_width_pct: pd.Series, *, squeeze_lookback: int) -> pd.Series:
    """A market is "ranging" when there's no trend AND the Bollinger Band
    width has contracted below its own trailing median (a squeeze) —
    distinguishing genuine consolidation from directionless-but-still-
    volatile chop."""
    median_width = bb_width_pct.rolling(squeeze_lookback, min_periods=squeeze_lookback).median()
    is_range = (trend_label == TREND_NONE) & (bb_width_pct <= median_width)
    undefined = trend_label.isna() | median_width.isna()
    return is_range.astype(object).mask(undefined, np.nan)
