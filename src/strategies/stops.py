"""Stop-loss calculators (CLAUDE.md Section 9).

Which type performs best is an empirical backtesting question for later
phases (7+) — this module only implements the mechanics for each type so
strategies can select one via config, and later phases can compare them.

Most functions work in two modes with the *same* formula, so a strategy's
per-bar (`calculate_stop_loss`) and whole-series (`generate_signals_vectorized`,
used by the Phase 6 vectorbt screening engine) code paths can never drift
apart into two different implementations of "the same" rule:
  - scalar mode: `entry_price`/`atr_value` are floats, `direction` is the
    string "LONG"/"SHORT" -> returns a float.
  - vectorized mode: `entry_price`/`atr_value` are aligned pd.Series,
    `direction` is a same-length Series (of "LONG"/"SHORT"/"FLAT", or
    SignalDirection values, which compare equal to those strings) ->
    returns a pd.Series.
`swing_stop` needs a rolling window instead of a plain formula, so it gets
a separate vectorized sibling (`swing_stop_series`) rather than trying to
force one signature over both shapes.
"""
from __future__ import annotations

import pandas as pd

_VOLATILITY_MULTIPLIER_SCALE = {"low_vol": 0.75, "normal_vol": 1.0, "high_vol": 1.5}


def _select(is_long, long_value, short_value):
    if isinstance(is_long, pd.Series):
        return long_value.where(is_long, short_value)
    return long_value if is_long else short_value


def atr_stop(entry_price, atr_value, direction, multiplier: float = 2.0):
    is_long = direction == "LONG"
    return _select(is_long, entry_price - multiplier * atr_value, entry_price + multiplier * atr_value)


def fixed_pct_stop(entry_price, direction, pct: float = 0.01):
    is_long = direction == "LONG"
    return _select(is_long, entry_price * (1 - pct), entry_price * (1 + pct))


def swing_stop(df: pd.DataFrame, direction: str, lookback: int = 10) -> float:
    """Stop beyond the most recent swing low/high over `lookback` bars
    (including the current, last bar of `df`)."""
    window = df.iloc[-lookback:]
    if direction == "LONG":
        return float(window["low"].min())
    return float(window["high"].max())


def swing_stop_series(df: pd.DataFrame, direction: pd.Series, lookback: int = 10) -> pd.Series:
    """Vectorized `swing_stop`: the same rolling swing low/high, computed
    at every bar instead of just the last one."""
    rolling_low = df["low"].rolling(lookback, min_periods=lookback).min()
    rolling_high = df["high"].rolling(lookback, min_periods=lookback).max()
    is_long = direction == "LONG"
    return rolling_low.where(is_long, rolling_high)


def volatility_stop(entry_price, atr_value, direction, atr_multiplier: float, volatility_regime):
    """Like `atr_stop`, but the multiplier itself widens in a high-volatility
    regime and tightens in a low-volatility one, rather than trusting one
    fixed multiplier to fit every regime. `volatility_regime` may be a
    scalar string or a pd.Series (vectorized mode)."""
    if isinstance(volatility_regime, pd.Series):
        scale = volatility_regime.map(_VOLATILITY_MULTIPLIER_SCALE).fillna(1.0)
    else:
        scale = _VOLATILITY_MULTIPLIER_SCALE.get(volatility_regime, 1.0)
    return atr_stop(entry_price, atr_value, direction, multiplier=atr_multiplier * scale)
