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


def adx(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """Wilder's ADX (+DI/-DI/ADX): an objective measure of trend strength,
    independent of direction — derived purely from directional price
    movement (+DM/-DM), smoothed the same way as ATR, rather than from
    EMA slope (how src/features/regime.classify_trend measures strength)."""
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")

    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    smoothed_tr = true_range(df).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    smoothed_plus_dm = plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    smoothed_minus_dm = minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    safe_tr = smoothed_tr.replace(0, np.nan)
    plus_di = 100 * smoothed_plus_dm / safe_tr
    minus_di = 100 * smoothed_minus_dm / safe_tr

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    adx_series = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": adx_series})
