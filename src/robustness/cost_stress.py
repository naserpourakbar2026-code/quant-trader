"""Transaction Cost Stress Test (CLAUDE.md Section 34): re-run with
normal / 1.5x / 2x / 3x spread (and, via the three existing named cost
scenarios, varying commission/slippage) to see whether profitability
survives a small cost increase. If it disappears, the strategy is
COST FRAGILE — a real risk this framework never papers over.

Uses the Backtrader engine (Phase 7), not vectorbt, because it's the
one that actually folds the data's own `spread` column into cost the
same way a real broker's spread would bite (src.backtest.backtrader_engine);
vectorbt's screening path does not consume the spread column at all.
Spread is stressed by scaling the raw data's own `spread` column
directly — the multiplier is a property of the market, not of any one
cost scenario — while cycling through the three configured scenarios
covers "varying slippage" (optimistic/realistic/stress already differ
specifically in slippage_pct/commission_pct by construction).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import pandas as pd

from src.backtest.backtrader_engine import run_validation
from src.core.config import RobustnessConfig, load_settings


@dataclass
class CostStressPoint:
    spread_multiplier: float
    scenario: str
    metrics: dict


@dataclass
class CostStressResult:
    run_id: str
    strategy: str
    symbol: str
    timeframe: str
    baseline_scenario: str
    points: list[CostStressPoint] = field(default_factory=list)

    def point(self, spread_multiplier: float, scenario: str) -> CostStressPoint | None:
        return next(
            (p for p in self.points if p.spread_multiplier == spread_multiplier and p.scenario == scenario), None
        )

    @property
    def baseline(self) -> CostStressPoint | None:
        return self.point(min(p.spread_multiplier for p in self.points), self.baseline_scenario) if self.points else None

    @property
    def is_cost_fragile(self) -> bool:
        """Section 34's own wording: "if profitability disappears with a
        SMALL cost increase." The smallest configured multiplier above
        1.0x (not the most extreme one) is what "small" means here —
        collapsing only at 3x is a different, milder finding than
        collapsing at 1.5x, and this flag is reserved for the latter."""
        base = self.baseline
        base_pf = base.metrics.get("profit_factor") if base else None
        # None means zero losing trades (vectorbt's own convention for
        # an undefined ratio) -- that is "excellent", not "unprofitable";
        # only a real, finite value <= 1 means never profitable to begin with.
        if base is None or (base_pf is not None and base_pf <= 1):
            return False  # never profitable to begin with -- not "fragile", just not profitable
        multipliers_above_baseline = sorted({p.spread_multiplier for p in self.points if p.spread_multiplier > 1.0})
        if not multipliers_above_baseline:
            return False
        smallest_increase = self.point(multipliers_above_baseline[0], self.baseline_scenario)
        if smallest_increase is None:
            return False
        stressed_pf = smallest_increase.metrics.get("profit_factor")
        return stressed_pf is not None and stressed_pf <= 1

    def to_text(self) -> str:
        lines = [
            "TRANSACTION COST STRESS TEST",
            f"Strategy: {self.strategy} | Symbol: {self.symbol} | Timeframe: {self.timeframe}",
            "",
        ]
        for p in sorted(self.points, key=lambda p: (p.scenario, p.spread_multiplier)):
            lines.append(
                f"  scenario={p.scenario:10s} spread x{p.spread_multiplier:<4} "
                f"return={p.metrics.get('total_return')!s:>10} profit_factor={p.metrics.get('profit_factor')!s:>8} "
                f"trades={p.metrics.get('trade_count')}"
            )
        lines.append("")
        lines.append(f"Verdict: {'COST FRAGILE' if self.is_cost_fragile else 'not cost fragile'} (CLAUDE.md Section 34)")
        return "\n".join(lines)


def run_cost_stress_test(
    strategy_family: str,
    symbol: str,
    timeframe: str,
    raw_df: pd.DataFrame,
    *,
    strategy_params: dict | None = None,
    baseline_scenario: str | None = None,
    scenarios: list[str] | None = None,
    spread_multipliers: list[float] | None = None,
) -> CostStressResult:
    settings = load_settings()
    cfg: RobustnessConfig = settings.robustness
    baseline_scenario = baseline_scenario or settings.execution.default_scenario
    scenarios = scenarios or list(settings.execution.costs)
    spread_multipliers = spread_multipliers or cfg.cost_stress_spread_multipliers

    points = []
    for multiplier in spread_multipliers:
        stressed_df = raw_df.copy()
        stressed_df["spread"] = raw_df["spread"] * multiplier
        for scenario in scenarios:
            validation = run_validation(
                strategy_family, symbol, timeframe, stressed_df,
                strategy_params=strategy_params, scenario=scenario, persist=False,
            )
            points.append(CostStressPoint(spread_multiplier=multiplier, scenario=scenario, metrics=validation.metrics))

    return CostStressResult(
        run_id=str(uuid.uuid4()), strategy=strategy_family, symbol=symbol, timeframe=timeframe,
        baseline_scenario=baseline_scenario, points=points,
    )
