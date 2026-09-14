"""Final Robustness Score and Status (CLAUDE.md Sections 18, 31, 43).

`compute_robustness_score()` combines every prior phase's own already-
validated output — walk-forward OOS results (Phase 9), Monte Carlo
(Phase 10), parameter stability (Phase 8), and the cost stress test
(this phase, Section 34) — into one 0-100 score, weighted per Section
31's table so no single metric dominates. It never invents a number
when the inputs it needs aren't there: missing walk-forward or Monte
Carlo results mean `INSUFFICIENT_DATA`, not a partial, falsely-confident
score (Section 44: never fabricate a result the evidence doesn't
support).

Every normalization reuses a config value that already exists
elsewhere for the same purpose — the profit-factor cap from Optuna's
own composite objective (Section 14), the drawdown ceiling from the
risk engine's own kill-switch threshold (Section 7), Monte Carlo's own
fragility cutoff (Section 17) — rather than inventing a second set of
thresholds that could quietly drift from the first.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from src.core.config import AppConfig
from src.montecarlo.mc_engine import MonteCarloResult
from src.robustness.cost_stress import CostStressResult
from src.walkforward.wfa_engine import WalkForwardReport


class RobustnessStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    COST_FRAGILE = "COST FRAGILE"
    OVERFIT = "OVERFIT"
    INSUFFICIENT_DATA = "INSUFFICIENT DATA"
    NOT_ROBUST = "NOT ROBUST"


@dataclass
class RobustnessScore:
    strategy: str
    symbol: str
    timeframe: str
    status: RobustnessStatus
    score: float | None  # 0-100, or None when status is INSUFFICIENT_DATA
    sub_scores: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        lines = [
            "FINAL ROBUSTNESS EVALUATION",
            f"Strategy: {self.strategy} | Symbol: {self.symbol} | Timeframe: {self.timeframe}",
            f"Status: {self.status.value}",
            f"Score: {self.score:.1f}/100" if self.score is not None else "Score: n/a",
        ]
        if self.sub_scores:
            lines.append("")
            lines.append("Sub-scores:")
            for name, value in self.sub_scores.items():
                lines.append(f"  {name:26s}: {value:.1f}")
        if self.notes:
            lines.append("")
            lines.append("Notes:")
            for note in self.notes:
                lines.append(f"  - {note}")
        return "\n".join(lines)


def _clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _profit_factor_or_cap(profit_factor: float | None, cap: float) -> float:
    """vectorbt reports profit_factor as None for a run with zero losing
    trades (division by zero, gross_loss=0) — the same convention
    src.optimization.optuna_engine.composite_objective already treats
    as "excellent" (capped, not zero): a strategy that never lost is not
    scored as if it never won."""
    return cap if profit_factor is None else min(profit_factor, cap)


def compute_robustness_score(
    *,
    strategy: str,
    symbol: str,
    timeframe: str,
    settings: AppConfig,
    walkforward_report: WalkForwardReport | None,
    montecarlo_result: MonteCarloResult | None,
    cost_stress_result: CostStressResult | None,
    stability_score: float | None,
    n_trades: int,
) -> RobustnessScore:
    notes: list[str] = []
    cfg = settings.robustness

    if walkforward_report is None or not walkforward_report.windows:
        return RobustnessScore(
            strategy=strategy, symbol=symbol, timeframe=timeframe, status=RobustnessStatus.INSUFFICIENT_DATA,
            score=None, notes=["No walk-forward result — out-of-sample performance is unknown (CLAUDE.md Section 18)."],
        )
    if montecarlo_result is None:
        return RobustnessScore(
            strategy=strategy, symbol=symbol, timeframe=timeframe, status=RobustnessStatus.INSUFFICIENT_DATA,
            score=None, notes=["No Monte Carlo result — robustness to trade-sequence variation is unknown (Section 17)."],
        )
    if n_trades < settings.optimization.min_trades:
        return RobustnessScore(
            strategy=strategy, symbol=symbol, timeframe=timeframe, status=RobustnessStatus.INSUFFICIENT_DATA,
            score=None,
            notes=[f"Only {n_trades} trades observed (< optimization.min_trades={settings.optimization.min_trades}) "
                   "— too few for a statistically reliable evaluation (Section 18/31)."],
        )

    last_window = walkforward_report.windows[-1]
    oos_metrics = last_window.oos_metrics
    oos_pass_rate = walkforward_report.oos_pass_rate

    # -- OOS performance (Section 18's central requirement) --------------------
    oos_score = _clip(oos_pass_rate * 100)

    # -- Drawdown, against the kill-switch threshold every session already lives by --
    max_dd = abs(oos_metrics.get("max_drawdown") or 0.0)
    drawdown_score = _clip(100 * (1 - max_dd / settings.risk.max_portfolio_drawdown))

    # -- Profit factor, capped the same way Optuna's own objective caps it --
    profit_factor = oos_metrics.get("profit_factor")
    pf_cap = settings.optimization.weights.profit_factor_cap
    profit_factor_score = _clip(100 * _profit_factor_or_cap(profit_factor, pf_cap) / pf_cap)

    # -- Sharpe/Sortino, against a stated reference (not a claim that's optimal) --
    sharpe = oos_metrics.get("sharpe_ratio") or 0.0
    sortino = oos_metrics.get("sortino_ratio") or 0.0
    sharpe_sortino_score = _clip(100 * ((sharpe + sortino) / 2) / cfg.sharpe_reference)

    # -- Parameter stability (Phase 8, Section 16) ------------------------------
    stability_pct = _clip((stability_score or 0.0) * 100)
    if stability_score is None:
        notes.append("No parameter-stability result available — scored as 0 (Section 16).")

    # -- Monte Carlo robustness, against its own fragility cutoff (Section 17) --
    mc_score = _clip(100 * (1 - montecarlo_result.probability_of_ruin / settings.montecarlo.fragility.max_probability_of_ruin))
    if montecarlo_result.is_fragile:
        notes.append("Monte Carlo flagged this strategy as fragile (Section 17).")

    # -- Cost sensitivity (this phase, Section 34) ------------------------------
    if cost_stress_result is not None and cost_stress_result.points:
        baseline = cost_stress_result.baseline
        stressed = cost_stress_result.point(
            max(p.spread_multiplier for p in cost_stress_result.points), cost_stress_result.baseline_scenario
        )
        base_pf = _profit_factor_or_cap(baseline.metrics.get("profit_factor"), pf_cap) if baseline else 0.0
        stressed_pf = _profit_factor_or_cap(stressed.metrics.get("profit_factor"), pf_cap) if stressed else 0.0
        cost_sensitivity_score = _clip(100 * (stressed_pf / base_pf)) if base_pf > 0 else 0.0
        if cost_stress_result.is_cost_fragile:
            notes.append("Profitability disappeared under a small transaction-cost increase (Section 34).")
    else:
        cost_sensitivity_score = 0.0
        notes.append("No cost-stress result available — scored as 0 (Section 34).")

    # -- Trade count / statistical reliability, against Optuna's own min_trades --
    trade_count_score = _clip(100 * n_trades / (settings.optimization.min_trades * 3))

    weights = cfg.weights
    sub_scores = {
        "oos_performance": oos_score, "drawdown": drawdown_score, "profit_factor": profit_factor_score,
        "sharpe_sortino": sharpe_sortino_score, "parameter_stability": stability_pct,
        "monte_carlo_robustness": mc_score, "cost_sensitivity": cost_sensitivity_score,
        "trade_count_reliability": trade_count_score,
    }
    weight_map = {
        "oos_performance": weights.oos_performance, "drawdown": weights.drawdown,
        "profit_factor": weights.profit_factor, "sharpe_sortino": weights.sharpe_sortino,
        "parameter_stability": weights.parameter_stability, "monte_carlo_robustness": weights.monte_carlo_robustness,
        "cost_sensitivity": weights.cost_sensitivity, "trade_count_reliability": weights.trade_count_reliability,
    }
    score = sum(sub_scores[name] * weight_map[name] for name in sub_scores)

    # -- Status: first matching rule wins, most severe first --------------------
    train_objective = last_window.train_objective
    is_overfit = (
        train_objective > 0 and oos_pass_rate < 0.5 and (stability_score or 0.0) < 0.5
    )

    oos_profit_factor = oos_metrics.get("profit_factor")
    oos_unprofitable = oos_profit_factor is not None and oos_profit_factor <= 1.0
    if oos_unprofitable or oos_pass_rate == 0.0:
        status = RobustnessStatus.FAIL
        notes.append("Out-of-sample performance failed to clear breakeven (Section 18: reject a strategy that fails OOS).")
    elif cost_stress_result is not None and cost_stress_result.is_cost_fragile:
        status = RobustnessStatus.COST_FRAGILE
    elif is_overfit:
        status = RobustnessStatus.OVERFIT
        notes.append("Strong training performance did not survive out-of-sample with stable parameters — a classic overfit signature (Section 16/18).")
    elif montecarlo_result.is_fragile or score < cfg.min_pass_score:
        status = RobustnessStatus.NOT_ROBUST
    else:
        status = RobustnessStatus.PASS

    return RobustnessScore(
        strategy=strategy, symbol=symbol, timeframe=timeframe, status=status, score=score,
        sub_scores=sub_scores, notes=notes,
    )
