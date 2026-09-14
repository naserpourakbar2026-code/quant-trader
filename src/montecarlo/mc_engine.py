"""Phase 10: Monte Carlo analysis (CLAUDE.md Section 17).

For a strategy/symbol/timeframe combination, bootstrap-resample the
*observed* per-trade returns from the vectorbt screening engine (Phase 6,
src.backtest.vectorbt_engine.get_trade_returns) thousands of times, each
time reshuffling trade order and adding extra execution-variation noise,
then build a simulated equity curve per draw. This estimates a
*distribution* of plausible outcomes from one historical trade sample —
never a claim that the future will repeat that sample, and never a
substitute for out-of-sample testing (Phase 9). A strategy whose Monte
Carlo distribution is fragile (high probability of ruin, high probability
of a negative return) is not "robust" even if its single historical
backtest looked good (Section 17's own words) — see `is_fragile` below.

Deliberately NOT built on top of run_optimization()/run_walk_forward(): a
Monte Carlo run needs only trade returns (not a fresh parameter search),
so it stays cheap and can be run again and again for the same fixed,
already-chosen parameters.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.backtest.vectorbt_engine import get_trade_returns
from src.core.config import MonteCarloConfig, load_settings
from src.core.db import Database
from src.core.reproducibility import git_commit_hash, library_versions, python_version
from src.features.engine import FeatureParams
from src.montecarlo.run_store import MonteCarloRecord, save_run


@dataclass
class MonteCarloResult:
    run_id: str
    strategy: str
    symbol: str
    timeframe: str
    scenario: str
    n_simulations: int
    n_trades_observed: int
    initial_capital: float
    ruin_threshold: float
    median_return: float
    p5_return: float
    p95_return: float
    worst_drawdown: float
    p95_drawdown: float
    median_losing_streak: float
    p95_losing_streak: float
    worst_losing_streak: int
    probability_of_ruin: float
    probability_of_negative_return: float
    is_fragile: bool
    insufficient_data: bool = False
    parameters: dict = field(default_factory=dict)

    def to_text(self) -> str:
        lines = [
            "MONTE CARLO ANALYSIS REPORT",
            f"Strategy: {self.strategy} | Symbol: {self.symbol} | Timeframe: {self.timeframe} | "
            f"Scenario: {self.scenario}",
            f"Simulations: {self.n_simulations} | Trades observed (resampled from): {self.n_trades_observed} | "
            f"Initial capital: {self.initial_capital}",
        ]
        if self.insufficient_data:
            lines.append(
                f"WARNING: only {self.n_trades_observed} observed trades — below the configured min_trades "
                "threshold. Results below are computed but statistically unreliable (CLAUDE.md Section 18/31)."
            )
        lines += [
            "",
            f"Median return        : {self.median_return:+.2%}",
            f"5th pct return       : {self.p5_return:+.2%}",
            f"95th pct return      : {self.p95_return:+.2%}",
            f"Worst drawdown       : {self.worst_drawdown:.2%}",
            f"95th pct drawdown    : {self.p95_drawdown:.2%}",
            f"Median losing streak : {self.median_losing_streak:.1f}",
            f"95th pct los. streak : {self.p95_losing_streak:.1f}",
            f"Worst losing streak  : {self.worst_losing_streak}",
            f"P(ruin, equity<={self.ruin_threshold:.0%} of capital): {self.probability_of_ruin:.2%}",
            f"P(negative return)   : {self.probability_of_negative_return:.2%}",
            "",
            f"Verdict: {'FRAGILE UNDER MONTE CARLO' if self.is_fragile else 'not flagged as fragile'} "
            "(CLAUDE.md Section 17 — not a guarantee of future profitability either way).",
        ]
        return "\n".join(lines)


def _max_consecutive_true(mask: np.ndarray) -> np.ndarray:
    """Per-row (simulation) longest run of consecutive True values in a
    2D boolean array — used for each simulation's max losing streak."""
    n_rows, n_cols = mask.shape
    streak = np.zeros(n_rows, dtype=int)
    best = np.zeros(n_rows, dtype=int)
    for col in range(n_cols):
        streak = np.where(mask[:, col], streak + 1, 0)
        best = np.maximum(best, streak)
    return best


def simulate(
    trade_returns: np.ndarray,
    *,
    n_simulations: int,
    initial_capital: float,
    ruin_threshold: float,
    execution_noise_std: float,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Core bootstrap: for each of `n_simulations` draws, resample
    len(trade_returns) trades with replacement (randomizes trade
    sequence and, via replacement, trade-return composition — Section
    17), add independent Gaussian noise per resampled trade
    (execution-variation/slippage uncertainty beyond the fixed cost
    scenario baked into `trade_returns` already), then compound them
    into an equity curve. Returns the raw per-simulation arrays so
    run_monte_carlo() can summarize them.
    """
    n_trades = len(trade_returns)
    rng = np.random.default_rng(seed)

    sampled = rng.choice(trade_returns, size=(n_simulations, n_trades), replace=True)
    if execution_noise_std > 0:
        sampled = sampled + rng.normal(0.0, execution_noise_std, size=sampled.shape)

    equity = initial_capital * np.cumprod(1.0 + sampled, axis=1)
    running_max = np.maximum.accumulate(equity, axis=1)
    drawdown = (running_max - equity) / running_max
    max_drawdown = drawdown.max(axis=1)

    total_return = equity[:, -1] / initial_capital - 1.0
    ruin = (equity <= ruin_threshold * initial_capital).any(axis=1)
    losing_streak = _max_consecutive_true(sampled < 0.0)

    return {
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "ruin": ruin,
        "losing_streak": losing_streak,
    }


def run_monte_carlo(
    strategy_family: str,
    symbol: str,
    timeframe: str,
    raw_df: pd.DataFrame,
    *,
    strategy_params: dict | None = None,
    feature_params: FeatureParams | None = None,
    scenario: str | None = None,
    initial_capital: float | None = None,
    n_simulations: int | None = None,
    seed: int | None = None,
    persist: bool = True,
    db: Database | None = None,
) -> MonteCarloResult:
    """Run a Monte Carlo analysis for one strategy/symbol/timeframe/
    parameter combination (CLAUDE.md Section 17).

    The observed trade-return sample comes from a fresh, unpersisted
    vectorbt screening run (src.backtest.vectorbt_engine.get_trade_returns)
    over `raw_df` under `scenario`'s cost assumptions — the same
    generate_signals_vectorized() signal table every other engine reads,
    so this can never silently diverge from what the strategy actually
    decided historically. Only the Monte Carlo summary itself is
    persisted (as a MonteCarloRun row), not the underlying screening.
    """
    settings = load_settings()
    cfg: MonteCarloConfig = settings.montecarlo
    scenario = scenario or settings.execution.default_scenario
    capital = initial_capital if initial_capital is not None else settings.account.initial_capital
    n_simulations = n_simulations if n_simulations is not None else cfg.n_simulations
    seed = seed if seed is not None else cfg.random_seed

    trade_returns = get_trade_returns(
        strategy_family,
        raw_df,
        strategy_params=strategy_params,
        feature_params=feature_params,
        scenario=scenario,
        initial_capital=capital,
    )
    n_trades_observed = len(trade_returns)
    if n_trades_observed == 0:
        raise ValueError(
            f"{strategy_family}/{symbol}/{timeframe} produced zero trades under scenario={scenario!r} — "
            "nothing to resample. Monte Carlo needs at least one observed trade."
        )

    sim = simulate(
        trade_returns,
        n_simulations=n_simulations,
        initial_capital=capital,
        ruin_threshold=cfg.ruin_threshold,
        execution_noise_std=cfg.execution_noise_std,
        seed=seed,
    )

    probability_of_ruin = float(sim["ruin"].mean())
    probability_of_negative_return = float((sim["total_return"] < 0.0).mean())
    is_fragile = (
        probability_of_ruin > cfg.fragility.max_probability_of_ruin
        or probability_of_negative_return > cfg.fragility.max_probability_of_negative_return
    )

    result = MonteCarloResult(
        run_id=str(uuid.uuid4()),
        strategy=strategy_family,
        symbol=symbol,
        timeframe=timeframe,
        scenario=scenario,
        n_simulations=n_simulations,
        n_trades_observed=n_trades_observed,
        initial_capital=capital,
        ruin_threshold=cfg.ruin_threshold,
        median_return=float(np.median(sim["total_return"])),
        p5_return=float(np.percentile(sim["total_return"], 5)),
        p95_return=float(np.percentile(sim["total_return"], 95)),
        worst_drawdown=float(sim["max_drawdown"].max()),
        p95_drawdown=float(np.percentile(sim["max_drawdown"], 95)),
        median_losing_streak=float(np.median(sim["losing_streak"])),
        p95_losing_streak=float(np.percentile(sim["losing_streak"], 95)),
        worst_losing_streak=int(sim["losing_streak"].max()),
        probability_of_ruin=probability_of_ruin,
        probability_of_negative_return=probability_of_negative_return,
        is_fragile=is_fragile,
        insufficient_data=n_trades_observed < cfg.min_trades,
        parameters=dict(strategy_params or {}),
    )

    if persist:
        save_run(
            MonteCarloRecord(
                run_id=result.run_id,
                strategy=result.strategy,
                symbol=result.symbol,
                timeframe=result.timeframe,
                scenario=result.scenario,
                n_simulations=result.n_simulations,
                n_trades_observed=result.n_trades_observed,
                initial_capital=result.initial_capital,
                ruin_threshold=result.ruin_threshold,
                median_return=result.median_return,
                p5_return=result.p5_return,
                p95_return=result.p95_return,
                worst_drawdown=result.worst_drawdown,
                p95_drawdown=result.p95_drawdown,
                median_losing_streak=result.median_losing_streak,
                p95_losing_streak=result.p95_losing_streak,
                worst_losing_streak=result.worst_losing_streak,
                probability_of_ruin=result.probability_of_ruin,
                probability_of_negative_return=result.probability_of_negative_return,
                is_fragile=result.is_fragile,
                parameters=result.parameters,
                code_version=git_commit_hash(),
                python_version=python_version(),
                library_versions=library_versions(["pandas", "numpy", "vectorbt"]),
                random_seed=seed,
            ),
            db=db,
        )

    return result
