"""Standard strategy interface every strategy family implements
(CLAUDE.md Section 6).

generate_signal(), calculate_stop_loss() and calculate_take_profit() are
strategy-specific (each family's own economic logic); calculate_position_size()
and validate_signal() are genuinely common risk/sanity logic and have a
shared implementation here that subclasses can still override.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import pandas as pd


class SignalDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass
class Signal:
    direction: SignalDirection
    entry_price: float | None
    stop_loss: float | None
    take_profit: float | None
    confidence: float
    strategy_name: str
    timestamp: pd.Timestamp
    symbol: str
    timeframe: str

    def is_actionable(self) -> bool:
        return self.direction != SignalDirection.FLAT


@dataclass
class PositionSizeResult:
    units: float
    risk_amount: float
    stop_distance: float


class BaseStrategy(ABC):
    strategy_name: str = "base"

    def __init__(self, params: dict | None = None):
        self.params: dict = params or {}

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """`df` is one symbol/timeframe's feature-engineered candle history
        (src.features.engine.compute_features output); the signal is
        generated from its last row."""

    @abstractmethod
    def calculate_stop_loss(self, df: pd.DataFrame, direction: SignalDirection) -> float:
        ...

    @abstractmethod
    def calculate_take_profit(self, df: pd.DataFrame, direction: SignalDirection, stop_loss: float) -> float:
        ...

    def calculate_position_size(
        self,
        equity: float,
        risk_pct: float,
        entry_price: float,
        stop_loss: float,
        pip_value_per_unit: float = 1.0,
    ) -> PositionSizeResult:
        """Percentage-risk position sizing (CLAUDE.md Sections 7-8):
        units = (equity * risk_pct) / stop_distance / pip_value_per_unit.

        `pip_value_per_unit` defaults to 1.0, i.e. this assumes the
        instrument's quote currency equals the account currency (so one
        price unit of movement equals one unit of account currency per
        traded unit). That is a real simplification: true pip value /
        contract size / cross-currency conversion is broker- and
        symbol-specific (Section 8) and isn't available until a broker
        adapter is connected (Phase 12/13) — pass a real
        `pip_value_per_unit` once that data exists instead of trusting
        this default. Margin/leverage verification against broker specs
        also belongs there, not here.
        """
        if equity <= 0:
            raise ValueError(f"equity must be > 0, got {equity}")
        if risk_pct <= 0:
            raise ValueError(f"risk_pct must be > 0, got {risk_pct}")
        if pip_value_per_unit <= 0:
            raise ValueError(f"pip_value_per_unit must be > 0, got {pip_value_per_unit}")
        stop_distance = abs(entry_price - stop_loss)
        if stop_distance <= 0:
            raise ValueError("stop_loss must differ from entry_price")

        risk_amount = equity * risk_pct
        units = risk_amount / stop_distance / pip_value_per_unit
        return PositionSizeResult(units=units, risk_amount=risk_amount, stop_distance=stop_distance)

    def _build_signal(self, df: pd.DataFrame, direction: SignalDirection, confidence: float) -> Signal:
        """Shared plumbing: an actionable direction gets its stop/take-profit
        computed via the subclass's own calculate_stop_loss/calculate_take_profit;
        FLAT gets None for all trade-specific fields."""
        last = df.iloc[-1]
        actionable = direction != SignalDirection.FLAT
        stop_loss = self.calculate_stop_loss(df, direction) if actionable else None
        take_profit = self.calculate_take_profit(df, direction, stop_loss) if actionable else None
        return Signal(
            direction=direction,
            entry_price=float(last["close"]) if actionable else None,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            strategy_name=self.strategy_name,
            timestamp=last["timestamp"],
            symbol=last["symbol"],
            timeframe=last["timeframe"],
        )

    def validate_signal(self, signal: Signal, df: pd.DataFrame) -> bool:
        """Sanity checks any signal must pass regardless of family-specific
        logic. Returns False rather than raising — callers decide whether
        to log/drop a signal that fails."""
        if signal.direction == SignalDirection.FLAT:
            return True
        if signal.entry_price is None or signal.entry_price <= 0:
            return False
        if signal.stop_loss is None or signal.take_profit is None:
            return False
        if not (0.0 <= signal.confidence <= 1.0):
            return False
        if signal.direction == SignalDirection.LONG:
            return signal.stop_loss < signal.entry_price < signal.take_profit
        return signal.take_profit < signal.entry_price < signal.stop_loss
