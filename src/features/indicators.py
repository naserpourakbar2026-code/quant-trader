"""Shared technical indicators (Phase 4, CLAUDE.md Section 5).

Pure functions on standardized candle data (src/data/schema.py columns).
Each function assumes it is given one contiguous, time-sorted series for a
single symbol/timeframe — src/features/engine.py enforces that before
calling in. Warm-up periods are left as NaN rather than backfilled or
guessed, consistent with never fabricating data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Wilder's ATR: EMA of true range with alpha = 1/period."""
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    tr = true_range(df)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def donchian_channel(df: pd.DataFrame, period: int) -> pd.DataFrame:
    upper = df["high"].rolling(period, min_periods=period).max()
    lower = df["low"].rolling(period, min_periods=period).min()
    middle = (upper + lower) / 2
    return pd.DataFrame({"donchian_upper": upper, "donchian_lower": lower, "donchian_middle": middle})


def bollinger_bands(series: pd.Series, period: int, num_std: float) -> pd.DataFrame:
    middle = series.rolling(period, min_periods=period).mean()
    std = series.rolling(period, min_periods=period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    width_pct = (upper - lower) / middle.replace(0, np.nan)
    return pd.DataFrame({"bb_upper": upper, "bb_middle": middle, "bb_lower": lower, "bb_width_pct": width_pct})


def rsi(series: pd.Series, period: int) -> pd.Series:
    """Wilder's RSI. Undefined (NaN) when there has been no price movement
    at all in the lookback window, rather than defaulting to an arbitrary
    neutral value."""
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def rate_of_change(series: pd.Series, period: int) -> pd.Series:
    """Momentum as a percentage change over `period` bars."""
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    return series.pct_change(periods=period) * 100


def distance_from_ma_pct(series: pd.Series, ma: pd.Series) -> pd.Series:
    return (series - ma) / ma.replace(0, np.nan) * 100


def distance_from_ma_atr(series: pd.Series, ma: pd.Series, atr_series: pd.Series) -> pd.Series:
    """Distance from a moving average expressed in ATR units — comparable
    across symbols and volatility regimes, unlike a raw price distance."""
    return (series - ma) / atr_series.replace(0, np.nan)
