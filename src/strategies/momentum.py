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
from src.strategies.base import BaseStrategy, SignalDirection

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

    def composite_score_series(self, df: pd.DataFrame) -> pd.Series:
        """The multi-factor score, computed for every bar at once. Used by
        generate_signals_vectorized() below and, unchanged, by the Phase 6
        vectorbt research engine for whole-series screening."""
        momentum_factor = (df["momentum_pct"] / self.params["momentum_normalizer_pct"]).clip(-1.0, 1.0)
        breakout_factor = (df["breakout_strength"] / self.params["breakout_normalizer_atr"]).clip(-1.0, 1.0)

        trend_direction = np.sign(df["ema_fast"] - df["ema_slow"])
        trend_weight = df["trend_regime"].map(_TREND_STRENGTH_WEIGHT).fillna(0.0)
        trend_factor = trend_direction * trend_weight

        factors = pd.concat([momentum_factor, breakout_factor, trend_factor], axis=1)
        base_score = factors.mean(axis=1, skipna=True).fillna(0.0)

        lookback = self.params["volume_lookback"]
        avg_volume = df["tick_volume"].rolling(lookback, min_periods=lookback).mean()
        volume_ratio = (df["tick_volume"] / avg_volume).where(avg_volume > 0, 1.0)
        volume_multiplier = volume_ratio.clip(0.8, 1.2).fillna(1.0)

        return (base_score * volume_multiplier).clip(-1.0, 1.0)

    def generate_signals_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        score = self.composite_score_series(df)

        # Plain strings, not SignalDirection instances -- see the note in
        # src/strategies/trend_following.py's generate_signals_vectorized.
        direction = pd.Series("FLAT", index=df.index, dtype=object)
        long_cond = score >= self.params["entry_threshold"]
        short_cond = score <= -self.params["entry_threshold"]
        direction = direction.mask(long_cond, "LONG")
        direction = direction.mask(short_cond, "SHORT")
        actionable = long_cond | short_cond

        confidence = score.abs().clip(upper=1.0).where(actionable, 0.0)

        stop_loss = stops.swing_stop_series(df, direction, lookback=self.params["swing_lookback"]).where(actionable)
        take_profit = targets.r_multiple_target(
            df["close"], stop_loss, direction, r_multiple=self.params["take_profit_r_multiple"]
        ).where(actionable)

        return pd.DataFrame(
            {"direction": direction, "confidence": confidence, "stop_loss": stop_loss, "take_profit": take_profit}
        )

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
