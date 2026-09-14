"""Phase 18: Final robustness evaluation (CLAUDE.md Sections 18, 31-36,
43-44) — the last phase, and deliberately a thin orchestrator: every
heavy-lifting step below is a previous phase's own already-tested
engine, run once more and combined, not reimplemented. This is the
concrete answer to Section 44's "not the strategy with the highest
backtest profit — the strategy with the strongest combination of
profitability, risk-adjusted return, statistical reliability,
out-of-sample performance, parameter stability, execution realism, and
robustness."
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

import pandas as pd

from src.core.config import ParamSpec, load_settings
from src.core.db import Database
from src.core.reproducibility import git_commit_hash, library_versions, python_version
from src.execution.paper_trading import run_paper_trading_session
from src.montecarlo.mc_engine import run_monte_carlo
from src.robustness.capital_simulation import CapitalSimulationResult, run_capital_simulation
from src.robustness.cost_stress import CostStressResult, run_cost_stress_test
from src.robustness.run_store import RobustnessEvaluationRecord, save_evaluation
from src.robustness.score import RobustnessScore, compute_robustness_score
from src.walkforward.wfa_engine import WalkForwardReport, run_walk_forward


@dataclass
class RobustnessEvaluation:
    run_id: str
    score: RobustnessScore
    walkforward_report: WalkForwardReport
    cost_stress_result: CostStressResult
    capital_simulation: CapitalSimulationResult
    best_params: dict

    def to_text(self) -> str:
        return "\n\n".join([
            self.score.to_text(),
            self.walkforward_report.to_text(),
            self.cost_stress_result.to_text(),
            self.capital_simulation.to_text(),
        ])


def run_robustness_evaluation(
    strategy_family: str,
    symbol: str,
    timeframe: str,
    raw_df: pd.DataFrame,
    space: dict[str, ParamSpec],
    *,
    scenario: str | None = None,
    n_trials: int | None = None,
    mc_simulations: int | None = None,
    window_bars: int | None = None,
    step_bars: int | None = None,
    seed: int | None = None,
    persist: bool = True,
    db: Database | None = None,
) -> RobustnessEvaluation:
    settings = load_settings()

    # 1. Walk-forward (Phase 9): the OOS discipline and parameter-
    #    stability score everything else below is evaluated in light of.
    wf_report = run_walk_forward(
        strategy_family, symbol, timeframe, raw_df, space, window_bars=window_bars, step_bars=step_bars,
        n_trials=n_trials, scenario=scenario, seed=seed, persist=persist, db=db,
    )
    best_params = wf_report.windows[-1].best_params if wf_report.windows else {}
    stability_score = wf_report.windows[-1].stability_score if wf_report.windows else None

    # 2. Monte Carlo (Phase 10) on the frozen parameters from the most
    #    recent window.
    mc_result = run_monte_carlo(
        strategy_family, symbol, timeframe, raw_df, strategy_params=best_params, scenario=scenario,
        n_simulations=mc_simulations, seed=seed, persist=persist, db=db,
    )

    # 3. Transaction cost stress test (Section 34), same frozen parameters.
    cost_stress = run_cost_stress_test(strategy_family, symbol, timeframe, raw_df, strategy_params=best_params)

    # 4. A paper-trading session (Phase 14) supplies real per-trade R-
    #    multiples for the capital simulation (Section 33) below --
    #    reusing the exact same position-sizing/execution-cost pipeline
    #    a live/paper account would actually see, not a synthetic proxy.
    paper_result = run_paper_trading_session(
        strategy_family, symbol, timeframe, raw_df, strategy_params=best_params, scenario=scenario, persist=False, db=db,
    )
    capital_sim = run_capital_simulation(
        paper_result.closed_trades, strategy=strategy_family, symbol=symbol, timeframe=timeframe,
        initial_capital=settings.account.initial_capital, risk_levels=settings.risk.allowed_risk_levels,
        max_leverage=settings.risk.max_leverage, n_simulations=mc_simulations or settings.montecarlo.n_simulations,
        ruin_threshold=settings.montecarlo.ruin_threshold, seed=seed,
    )

    score = compute_robustness_score(
        strategy=strategy_family, symbol=symbol, timeframe=timeframe, settings=settings,
        walkforward_report=wf_report, montecarlo_result=mc_result, cost_stress_result=cost_stress,
        stability_score=stability_score, n_trades=len(paper_result.closed_trades),
    )

    run_id = str(uuid.uuid4())
    if persist:
        save_evaluation(
            RobustnessEvaluationRecord(
                run_id=run_id, strategy=strategy_family, symbol=symbol, timeframe=timeframe,
                status=score.status.value, score=score.score, sub_scores=score.sub_scores, notes=score.notes,
                oos_pass_rate=wf_report.oos_pass_rate, probability_of_ruin=mc_result.probability_of_ruin,
                is_cost_fragile=cost_stress.is_cost_fragile, n_trades=len(paper_result.closed_trades),
                best_params=best_params, walkforward_run_id=wf_report.run_id, montecarlo_run_id=mc_result.run_id,
                code_version=git_commit_hash(), python_version=python_version(),
                library_versions=library_versions(["pandas", "numpy", "vectorbt", "backtrader", "optuna"]),
            ),
            db=db,
        )

    return RobustnessEvaluation(
        run_id=run_id, score=score, walkforward_report=wf_report, cost_stress_result=cost_stress,
        capital_simulation=capital_sim, best_params=best_params,
    )
