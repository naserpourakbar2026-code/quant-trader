"""Take-profit calculators (CLAUDE.md Section 10).

Whether a higher R:R is actually better is an empirical backtesting
question for later phases — this module only implements the mechanics.

`r_multiple_target` and `atr_target` work in scalar or vectorized mode,
same convention as src/strategies/stops.py, so a strategy's per-bar and
whole-series code paths share one formula (see that module's docstring).
"""
from __future__ import annotations

import pandas as pd


def _select(is_long, long_value, short_value):
    if isinstance(is_long, pd.Series):
        return long_value.where(is_long, short_value)
    return long_value if is_long else short_value


def r_multiple_target(entry_price, stop_loss, direction, r_multiple: float):
    is_long = direction == "LONG"
    risk = (entry_price - stop_loss).abs() if isinstance(entry_price, pd.Series) else abs(entry_price - stop_loss)
    return _select(is_long, entry_price + r_multiple * risk, entry_price - r_multiple * risk)


def atr_target(entry_price, atr_value, direction, multiplier: float):
    is_long = direction == "LONG"
    return _select(is_long, entry_price + multiplier * atr_value, entry_price - multiplier * atr_value)


def mean_reversion_target(mean_price):
    """Target = reversion to the mean (e.g. the Bollinger middle band) —
    dynamic, not a fixed R-multiple, matching mean reversion's own logic.
    Works unchanged in scalar or vectorized mode: it's a passthrough."""
    return mean_price
