"""Trend Following / Breakout strategy (CLAUDE.md Section 5A).

Trades Donchian-style breakouts, but only in the direction of an already
objectively-classified trend (src.features.regime.classify_trend) — never
a breakout in isolation, and never against the prevailing EMA structure.
"""
from __future__ import annotations

import pandas as pd

from src.strategies import stops, targets
from src.strategies.base import BaseStrategy, SignalDirection

DEFAULT_PARAMS = {
    "atr_stop_multiplier": 2.0,
    "take_profit_r_multiple": 2.0,
    "require_non_low_vol": True,
}


class TrendFollowingStrategy(BaseStrategy):
    strategy_name = "trend_following"

    def __init__(self, params: dict | None = None):
        super().__init__({**DEFAULT_PARAMS, **(params or {})})

    def generate_signals_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        trend = df["trend_regime"]
        is_uptrend = df["ema_fast"] > df["ema_slow"]
        is_downtrend = df["ema_fast"] < df["ema_slow"]
        breakout = df["breakout_strength"]

        if self.params["require_non_low_vol"]:
            vol_ok = df["volatility_regime"] != "low_vol"
        else:
            vol_ok = pd.Series(True, index=df.index)

        trending = trend.isin(["weak_trend", "strong_trend"]) & vol_ok
        long_cond = trending & is_uptrend & (breakout > 0)
        short_cond = trending & is_downtrend & (breakout < 0)

        # Plain strings, not SignalDirection instances: pandas' .mask()
        # silently corrupts a str-subclass Enum used as a scalar
        # replacement value (treats it as an array-like of characters
        # internally), even though comparing *against* one works fine.
        # generate_signal() (base.py) converts the last row back to a
        # real SignalDirection before building a Signal.
        direction = pd.Series("FLAT", index=df.index, dtype=object)
        direction = direction.mask(long_cond, "LONG")
        direction = direction.mask(short_cond, "SHORT")
        actionable = long_cond | short_cond

        confidence = pd.Series(0.0, index=df.index)
        confidence = confidence.mask(actionable, (trend == "strong_trend").map({True: 0.8, False: 0.5}))

        stop_loss = stops.atr_stop(
            df["close"], df["atr"], direction, multiplier=self.params["atr_stop_multiplier"]
        ).where(actionable)
        take_profit = targets.r_multiple_target(
            df["close"], stop_loss, direction, r_multiple=self.params["take_profit_r_multiple"]
        ).where(actionable)

        return pd.DataFrame(
            {"direction": direction, "confidence": confidence, "stop_loss": stop_loss, "take_profit": take_profit}
        )

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
