"""Trend Following / Breakout strategy (CLAUDE.md Section 5A).

Trades Donchian-style breakouts, but only in the direction of an already
objectively-classified trend (src.features.regime.classify_trend) — never
a breakout in isolation, and never against the prevailing EMA structure.
"""
from __future__ import annotations

import pandas as pd

from src.strategies import stops, targets
from src.strategies.base import BaseStrategy, Signal, SignalDirection

DEFAULT_PARAMS = {
    "atr_stop_multiplier": 2.0,
    "take_profit_r_multiple": 2.0,
    "require_non_low_vol": True,
}


class TrendFollowingStrategy(BaseStrategy):
    strategy_name = "trend_following"

    def __init__(self, params: dict | None = None):
        super().__init__({**DEFAULT_PARAMS, **(params or {})})

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        last = df.iloc[-1]

        direction = SignalDirection.FLAT
        confidence = 0.0

        trend = last["trend_regime"]
        is_uptrend = last["ema_fast"] > last["ema_slow"]
        is_downtrend = last["ema_fast"] < last["ema_slow"]
        vol_ok = (not self.params["require_non_low_vol"]) or last["volatility_regime"] != "low_vol"
        breakout = last["breakout_strength"]

        if trend in ("weak_trend", "strong_trend") and vol_ok:
            if is_uptrend and breakout > 0:
                direction = SignalDirection.LONG
            elif is_downtrend and breakout < 0:
                direction = SignalDirection.SHORT

        if direction != SignalDirection.FLAT:
            confidence = 0.8 if trend == "strong_trend" else 0.5

        return self._build_signal(df, direction, confidence)

    def calculate_stop_loss(self, df: pd.DataFrame, direction: SignalDirection) -> float:
        last = df.iloc[-1]
        return stops.atr_stop(
            entry_price=float(last["close"]),
            atr_value=float(last["atr"]),
            direction=direction.value,
            multiplier=self.params["atr_stop_multiplier"],
        )

    def calculate_take_profit(self, df: pd.DataFrame, direction: SignalDirection, stop_loss: float) -> float:
        last = df.iloc[-1]
        return targets.r_multiple_target(
            entry_price=float(last["close"]),
            stop_loss=stop_loss,
            direction=direction.value,
            r_multiple=self.params["take_profit_r_multiple"],
        )
