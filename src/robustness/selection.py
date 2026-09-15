"""Strategy Selection (CLAUDE.md Section 32): pick the best, second, and
third robust strategy — never just the highest backtest profit, and
never one that failed the minimum robustness bar. If nothing passes,
"NO ROBUST STRATEGY FOUND" is an acceptable and expected outcome
(Section 32's own words) — never manufactured into a false PASS.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.robustness.score import RobustnessScore, RobustnessStatus

_MEDALS = ["🥇", "🥈", "🥉"]


@dataclass
class StrategySelection:
    medalists: list[RobustnessScore]  # 0-3 entries, best first

    @property
    def found_any(self) -> bool:
        return len(self.medalists) > 0

    def to_text(self) -> str:
        if not self.medalists:
            return "NO ROBUST STRATEGY FOUND (CLAUDE.md Section 32 — an acceptable, expected outcome, not a failure to report)."
        lines = ["STRATEGY SELECTION"]
        for medal, evaluation in zip(_MEDALS, self.medalists):
            lines.append(
                f"{medal} {evaluation.strategy}/{evaluation.symbol}/{evaluation.timeframe} "
                f"— score {evaluation.score:.1f}/100"
            )
        return "\n".join(lines)


def select_top_strategies(evaluations: list[RobustnessScore], *, top_n: int = 3) -> StrategySelection:
    passing = [e for e in evaluations if e.status == RobustnessStatus.PASS and e.score is not None]
    ranked = sorted(passing, key=lambda e: e.score, reverse=True)
    return StrategySelection(medalists=ranked[:top_n])
