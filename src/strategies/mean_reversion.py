"""Mean Reversion strategy (CLAUDE.md Section 5B).

Fades a Bollinger Band extreme confirmed by RSI, but explicitly refuses to
fade the direction of an already-classified strong trend — Section 5B's
requirement to "avoid aggressively trading against strong trends".
"""
from __future__ import annotations

import pandas as pd

import numpy as np

from src.strategies import stops, targets
from src.strategies.base import BaseStrategy, SignalDirection

DEFAULT_PARAMS = {
    "rsi_oversold": 30.0,
    "rsi_overbought": 70.0,
    "atr_stop_multiplier": 1.5,
}


class MeanReversionStrategy(BaseStrategy):
    strategy_name = "mean_reversion"

    def __init__(self, params: dict | None = None):
        super().__init__({**DEFAULT_PARAMS, **(params or {})})

    def generate_signals_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        trend = df["trend_regime"]
        is_uptrend = df["ema_fast"] > df["ema_slow"]
        is_downtrend = df["ema_fast"] < df["ema_slow"]
        strong_downtrend = (trend == "strong_trend") & is_downtrend
        strong_uptrend = (trend == "strong_trend") & is_uptrend

        close, rsi = df["close"], df["rsi"]
        bb_lower, bb_upper = df["bb_lower"], df["bb_upper"]

        touched_lower = bb_lower.notna() & (close <= bb_lower)
        touched_upper = bb_upper.notna() & (close >= bb_upper)
        oversold = rsi.notna() & (rsi <= self.params["rsi_oversold"])
        overbought = rsi.notna() & (rsi >= self.params["rsi_overbought"])

        long_cond = touched_lower & oversold & ~strong_downtrend
        short_cond = touched_upper & overbought & ~strong_uptrend

        # Plain strings, not SignalDirection instances -- see the note in
        # src/strategies/trend_following.py's generate_signals_vectorized.
        direction = pd.Series("FLAT", index=df.index, dtype=object)
        direction = direction.mask(long_cond, "LONG")
        direction = direction.mask(short_cond, "SHORT")
        actionable = long_cond | short_cond

        long_extremity = (self.params["rsi_oversold"] - rsi).clip(lower=0.0)
        short_extremity = (rsi - self.params["rsi_overbought"]).clip(lower=0.0)
        confidence = pd.Series(0.0, index=df.index)
        confidence = confidence.mask(long_cond, np.minimum(0.5 + long_extremity / 100.0, 0.9))
        confidence = confidence.mask(short_cond, np.minimum(0.5 + short_extremity / 100.0, 0.9))

        stop_loss = stops.volatility_stop(
            df["close"],
            df["atr"],
            direction,
            atr_multiplier=self.params["atr_stop_multiplier"],
            volatility_regime=df["volatility_regime"],
        ).where(actionable)
        take_profit = targets.mean_reversion_target(df["bb_middle"]).where(actionable)

        return pd.DataFrame(
            {"direction": direction, "confidence": confidence, "stop_loss": stop_loss, "take_profit": take_profit}
        )

    def calculate_stop_loss(self, df: pd.DataFrame, direction: SignalDirection) -> float:
        last = df.iloc[-1]
        return stops.volatility_stop(
            entry_price=float(last["close"]),
            atr_value=float(last["atr"]),
            direction=direction.value,
            atr_multiplier=self.params["atr_stop_multiplier"],
            volatility_regime=last["volatility_regime"],
        )

    def calculate_take_profit(self, df: pd.DataFrame, direction: SignalDirection, stop_loss: float) -> float:
        last = df.iloc[-1]
        return targets.mean_reversion_target(float(last["bb_middle"]))
