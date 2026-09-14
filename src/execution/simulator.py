"""Execution Simulator (CLAUDE.md Sections 11, 26).

Converts a virtual order's intended price into a simulated fill: applies
slippage (price gets worse for the trader, direction-aware) and a
commission cost that folds in the bar's own real `spread` value the same
way `src.backtest.backtrader_engine` does — spread expressed as a
percentage of price and added to the configured `commission_pct`, rather
than treated as a separate, disconnected cost. Uses the same
`execution.costs[scenario]` (CostScenario) as every other engine, so a
strategy's paper-trading costs are directly comparable to its
screening/validation costs under the same scenario name.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from src.core.config import CostScenario
from src.strategies.base import SignalDirection


@dataclass
class Fill:
    price: float
    commission: float
    slippage_pct: float
    spread: float


class ExecutionSimulator:
    def __init__(self, cost: CostScenario) -> None:
        self.cost = cost

    @staticmethod
    def _clean_spread(bar_spread: float) -> float:
        """A CSV source that never had a `spread` column leaves it NaN
        (src.data.schema: "not fabricated"), which is correct upstream
        but can't be the logged/persisted value here — treat it as 0.0
        (no spread data -> no spread cost applied), the same
        "unavailable means zero, not fabricated-nonzero" stance already
        applied to commission below."""
        return bar_spread if math.isfinite(bar_spread) else 0.0

    def _effective_commission_pct(self, bar_spread: float, price: float) -> float:
        spread_pct = bar_spread / price if price else 0.0
        return self.cost.commission_pct + spread_pct

    def simulate_entry(self, *, direction: SignalDirection, bar_open: float, bar_spread: float, size: float) -> Fill:
        """A BUY (LONG entry) fills slightly above the bar's open; a SELL
        (SHORT entry) fills slightly below — slippage always works
        against the trader, never in their favor."""
        bar_spread = self._clean_spread(bar_spread)
        slip = self.cost.slippage_pct
        price = bar_open * (1 + slip) if direction == SignalDirection.LONG else bar_open * (1 - slip)
        commission = price * size * self._effective_commission_pct(bar_spread, bar_open)
        return Fill(price=price, commission=commission, slippage_pct=slip, spread=bar_spread)

    def simulate_exit(self, *, direction: SignalDirection, exit_price: float, bar_spread: float, size: float) -> Fill:
        """Exiting a LONG is a sell (fills slightly below the requested
        exit price); exiting a SHORT is a buy (fills slightly above) --
        the opposite bias from entry, same principle: always against the
        trader."""
        bar_spread = self._clean_spread(bar_spread)
        slip = self.cost.slippage_pct
        price = exit_price * (1 - slip) if direction == SignalDirection.LONG else exit_price * (1 + slip)
        commission = price * size * self._effective_commission_pct(bar_spread, exit_price)
        return Fill(price=price, commission=commission, slippage_pct=slip, spread=bar_spread)
