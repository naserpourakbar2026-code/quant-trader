"""Take-profit calculators (CLAUDE.md Section 10).

Whether a higher R:R is actually better is an empirical backtesting
question for later phases — this module only implements the mechanics.
"""
from __future__ import annotations


def r_multiple_target(entry_price: float, stop_loss: float, direction: str, r_multiple: float) -> float:
    risk = abs(entry_price - stop_loss)
    if direction == "LONG":
        return entry_price + r_multiple * risk
    return entry_price - r_multiple * risk


def atr_target(entry_price: float, atr_value: float, direction: str, multiplier: float) -> float:
    if direction == "LONG":
        return entry_price + multiplier * atr_value
    return entry_price - multiplier * atr_value


def mean_reversion_target(mean_price: float) -> float:
    """Target = reversion to the mean (e.g. the Bollinger middle band) —
    dynamic, not a fixed R-multiple, matching mean reversion's own logic."""
    return mean_price
