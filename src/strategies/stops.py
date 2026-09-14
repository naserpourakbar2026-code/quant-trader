"""Stop-loss calculators (CLAUDE.md Section 9).

Which type performs best is an empirical backtesting question for later
phases (7+) — this module only implements the mechanics for each type so
strategies can select one via config, and later phases can compare them.
"""
from __future__ import annotations

import pandas as pd

_VOLATILITY_MULTIPLIER_SCALE = {"low_vol": 0.75, "normal_vol": 1.0, "high_vol": 1.5}


def atr_stop(entry_price: float, atr_value: float, direction: str, multiplier: float = 2.0) -> float:
    if direction == "LONG":
        return entry_price - multiplier * atr_value
    return entry_price + multiplier * atr_value


def fixed_pct_stop(entry_price: float, direction: str, pct: float = 0.01) -> float:
    if direction == "LONG":
        return entry_price * (1 - pct)
    return entry_price * (1 + pct)


def swing_stop(df: pd.DataFrame, direction: str, lookback: int = 10) -> float:
    """Stop beyond the most recent swing low/high over `lookback` bars
    (including the current bar)."""
    window = df.iloc[-lookback:]
    if direction == "LONG":
        return float(window["low"].min())
    return float(window["high"].max())


def volatility_stop(
    entry_price: float,
    atr_value: float,
    direction: str,
    atr_multiplier: float,
    volatility_regime: str,
) -> float:
    """Like `atr_stop`, but the multiplier itself widens in a high-volatility
    regime and tightens in a low-volatility one, rather than trusting one
    fixed multiplier to fit every regime."""
    scale = _VOLATILITY_MULTIPLIER_SCALE.get(volatility_regime, 1.0)
    return atr_stop(entry_price, atr_value, direction, multiplier=atr_multiplier * scale)
