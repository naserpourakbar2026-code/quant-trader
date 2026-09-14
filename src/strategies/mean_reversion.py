"""Mean Reversion strategy (CLAUDE.md Section 5B).

Fades a Bollinger Band extreme confirmed by RSI, but explicitly refuses to
fade the direction of an already-classified strong trend — Section 5B's
requirement to "avoid aggressively trading against strong trends".
"""
from __future__ import annotations

import pandas as pd

from src.strategies import stops, targets
from src.strategies.base import BaseStrategy, Signal, SignalDirection

DEFAULT_PARAMS = {
    "rsi_oversold": 30.0,
    "rsi_overbought": 70.0,
    "atr_stop_multiplier": 1.5,
}


class MeanReversionStrategy(BaseStrategy):
    strategy_name = "mean_reversion"

    def __init__(self, params: dict | None = None):
        super().__init__({**DEFAULT_PARAMS, **(params or {})})

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        last = df.iloc[-1]

        direction = SignalDirection.FLAT
        confidence = 0.0

        trend = last["trend_regime"]
        is_uptrend = last["ema_fast"] > last["ema_slow"]
        is_downtrend = last["ema_fast"] < last["ema_slow"]
        strong_downtrend = trend == "strong_trend" and is_downtrend
        strong_uptrend = trend == "strong_trend" and is_uptrend

        close, rsi = last["close"], last["rsi"]
        bb_lower, bb_upper = last["bb_lower"], last["bb_upper"]

        touched_lower = pd.notna(bb_lower) and close <= bb_lower
        touched_upper = pd.notna(bb_upper) and close >= bb_upper
        oversold = pd.notna(rsi) and rsi <= self.params["rsi_oversold"]
        overbought = pd.notna(rsi) and rsi >= self.params["rsi_overbought"]

        if touched_lower and oversold and not strong_downtrend:
            direction = SignalDirection.LONG
        elif touched_upper and overbought and not strong_uptrend:
            direction = SignalDirection.SHORT

        if direction == SignalDirection.LONG:
            extremity = max(self.params["rsi_oversold"] - rsi, 0.0)
            confidence = min(0.5 + extremity / 100.0, 0.9)
        elif direction == SignalDirection.SHORT:
            extremity = max(rsi - self.params["rsi_overbought"], 0.0)
            confidence = min(0.5 + extremity / 100.0, 0.9)

        return self._build_signal(df, direction, confidence)

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
