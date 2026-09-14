"""Momentum / Multi-Factor strategy (CLAUDE.md Section 5C).

Combines only factors with a clear statistical/economic justification into
one composite score:
  - momentum_pct       — price persistence over the recent past
  - breakout_strength  — structural confirmation (already broke the recent range)
  - trend alignment    — broader-context filter, weighted by the objectively
                          classified trend strength

Tick volume is used only as a *minor* confirmation multiplier (capped to
+/-20%), never a primary or standalone signal: broker-reported tick volume
for retail forex is a count of price changes, not real traded volume, so
it is not statistically reliable enough to drive direction on its own.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies import stops, targets
from src.strategies.base import BaseStrategy, Signal, SignalDirection

DEFAULT_PARAMS = {
    "momentum_normalizer_pct": 1.0,  # |momentum_pct| at/beyond this counts as "full strength"
    "breakout_normalizer_atr": 1.0,  # |breakout_strength| at/beyond this counts as "full strength"
    "entry_threshold": 0.5,
    "volume_lookback": 20,
    "swing_lookback": 10,
    "take_profit_r_multiple": 2.0,
}

_TREND_STRENGTH_WEIGHT = {"no_trend": 0.0, "weak_trend": 0.5, "strong_trend": 1.0}


class MomentumMultiFactorStrategy(BaseStrategy):
    strategy_name = "momentum_multi_factor"

    def __init__(self, params: dict | None = None):
        super().__init__({**DEFAULT_PARAMS, **(params or {})})

    def _composite_score(self, df: pd.DataFrame) -> float:
        last = df.iloc[-1]

        momentum_factor = np.clip(last["momentum_pct"] / self.params["momentum_normalizer_pct"], -1.0, 1.0)
        breakout_factor = np.clip(last["breakout_strength"] / self.params["breakout_normalizer_atr"], -1.0, 1.0)

        trend_direction = np.sign(last["ema_fast"] - last["ema_slow"])
        trend_weight = _TREND_STRENGTH_WEIGHT.get(last["trend_regime"], 0.0)
        trend_factor = trend_direction * trend_weight

        factors = [f for f in (momentum_factor, breakout_factor, trend_factor) if pd.notna(f)]
        if not factors:
            return 0.0
        base_score = sum(factors) / len(factors)

        volume_window = df["tick_volume"].iloc[-self.params["volume_lookback"] :]
        avg_volume = volume_window.mean()
        volume_ratio = last["tick_volume"] / avg_volume if avg_volume and avg_volume > 0 else 1.0
        volume_multiplier = float(np.clip(volume_ratio, 0.8, 1.2))

        return float(np.clip(base_score * volume_multiplier, -1.0, 1.0))

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        score = self._composite_score(df)

        direction = SignalDirection.FLAT
        if score >= self.params["entry_threshold"]:
            direction = SignalDirection.LONG
        elif score <= -self.params["entry_threshold"]:
            direction = SignalDirection.SHORT

        confidence = min(abs(score), 1.0) if direction != SignalDirection.FLAT else 0.0
        return self._build_signal(df, direction, confidence)

    def calculate_stop_loss(self, df: pd.DataFrame, direction: SignalDirection) -> float:
        return stops.swing_stop(df, direction=direction.value, lookback=self.params["swing_lookback"])

    def calculate_take_profit(self, df: pd.DataFrame, direction: SignalDirection, stop_loss: float) -> float:
        last = df.iloc[-1]
        return targets.r_multiple_target(
            entry_price=float(last["close"]),
            stop_loss=stop_loss,
            direction=direction.value,
            r_multiple=self.params["take_profit_r_multiple"],
        )
