"""Phase 8: Optuna parameter optimization (CLAUDE.md Section 14).

Optimizes only the parameters listed in each strategy's
`optimization_space` (config/strategies.yaml) — a small, economically
justified set (breakout period, stop/TP multipliers, Bollinger period/
deviation, RSI thresholds, momentum period; never "dozens of parameters
at once"). Each trial is evaluated via the vectorbt screening engine
(Phase 6) — the same generate_signals_vectorized() table Phase 5/6/7 all
read, so optimization can never silently diverge from what the strategy
actually decides. The composite objective is Section 14's own formula:
Profit Factor + Sharpe + Sortino + Expectancy, penalized by drawdown and
low trade count. Parameter *sensitivity* (Section 16's "never select an
isolated spike") is assessed once, after the search, via
`assess_parameter_stability()` — doing that inside every trial would
multiply the trial cost by the neighborhood size for no benefit.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace

import optuna
import pandas as pd

from src.backtest.experiment_store import ExperimentRecord, save_experiment
from src.backtest.vectorbt_engine import run_screening
from src.core.config import OptimizationConfig, ParamSpec, load_settings
from src.core.db import Database
from src.core.reproducibility import git_commit_hash, library_versions, python_version
from src.data.ingestion import data_version_for
from src.features.engine import FeatureParams

_FEATURE_FIELDS = set(FeatureParams.__dataclass_fields__.keys())


def split_params(params: dict) -> tuple[dict, dict]:
    """Route each suggested parameter to feature-level (recomputing
    indicators) or strategy-level (just the strategy's own .params),
    based on which dataclass actually defines that field name."""
    feature_kwargs = {k: v for k, v in params.items() if k in _FEATURE_FIELDS}
    strategy_kwargs = {k: v for k, v in params.items() if k not in _FEATURE_FIELDS}
    return feature_kwargs, strategy_kwargs


def composite_objective(metrics: dict, cfg: OptimizationConfig) -> float:
    """Section 14's composite objective: Profit Factor + Sharpe + Sortino
    + Expectancy, penalized by max drawdown and low trade count. Returns
    a large negative value for a run with zero trades — there is nothing
    to evaluate, not a score of exactly 0."""
    trade_count = metrics.get("trade_count") or 0
    if trade_count == 0:
        return -100.0

    profit_factor = metrics.get("profit_factor")
    profit_factor_capped = min(profit_factor, cfg.weights.profit_factor_cap) if profit_factor is not None else (
        cfg.weights.profit_factor_cap
    )
    sharpe = metrics.get("sharpe_ratio") or 0.0
    sortino = metrics.get("sortino_ratio") or 0.0
    expectancy = metrics.get("expectancy") or 0.0
    max_drawdown = abs(metrics.get("max_drawdown") or 0.0)

    raw = profit_factor_capped + sharpe + sortino + expectancy
    drawdown_penalty = max_drawdown * cfg.weights.drawdown_penalty
    trade_count_penalty = max(0.0, cfg.min_trades - trade_count) * cfg.weights.trade_count_penalty

    return raw - drawdown_penalty - trade_count_penalty


def _suggest(trial: "optuna.Trial", name: str, spec: ParamSpec):
    if spec.type == "int":
        return trial.suggest_int(name, int(spec.low), int(spec.high), step=int(spec.step or 1))
    return trial.suggest_float(name, spec.low, spec.high, step=spec.step)


@dataclass
class OptimizationResult:
    experiment_id: str
    strategy: str
    symbol: str
    timeframe: str
    scenario: str
    best_params: dict
    best_metrics: dict
    best_objective: float
    n_trials: int


def run_optimization(
    strategy_family: str,
    symbol: str,
    timeframe: str,
    raw_df: pd.DataFrame,
    space: dict[str, ParamSpec],
    *,
    n_trials: int | None = None,
    scenario: str | None = None,
    seed: int | None = None,
    persist: bool = True,
    db: Database | None = None,
) -> OptimizationResult:
    """Search `space` for the strategy_family/symbol/timeframe combination
    that maximizes composite_objective(), evaluating each trial via
    vectorbt screening (Phase 6). Only the final best result is persisted
    as an Experiment — individual trials are not (they'd flood the table
    for no benefit; the study itself isn't reproducible-record material
    the way one strategy/parameter/data combination's outcome is).
    """
    if not space:
        raise ValueError("space must not be empty — nothing to optimize")

    settings = load_settings()
    cfg = settings.optimization
    scenario = scenario or settings.execution.default_scenario
    n_trials = n_trials if n_trials is not None else cfg.n_trials
    seed = seed if seed is not None else cfg.random_seed

    def objective(trial: "optuna.Trial") -> float:
        params = {name: _suggest(trial, name, spec) for name, spec in space.items()}
        feature_kwargs, strategy_kwargs = split_params(params)
        feature_params = replace(FeatureParams.from_settings(), **feature_kwargs)
        result = run_screening(
            strategy_family,
            symbol,
            timeframe,
            raw_df,
            strategy_params=strategy_kwargs,
            feature_params=feature_params,
            scenario=scenario,
            persist=False,
        )
        return composite_objective(result.metrics, cfg)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)

    best_feature_kwargs, best_strategy_kwargs = split_params(study.best_params)
    best_feature_params = replace(FeatureParams.from_settings(), **best_feature_kwargs)
    best_result = run_screening(
        strategy_family,
        symbol,
        timeframe,
        raw_df,
        strategy_params=best_strategy_kwargs,
        feature_params=best_feature_params,
        scenario=scenario,
        persist=False,
    )

    experiment_id = str(uuid.uuid4())
    if persist:
        indexed_ts = pd.DatetimeIndex(raw_df["timestamp"])
        save_experiment(
            ExperimentRecord(
                experiment_id=experiment_id,
                engine="optuna",
                strategy=strategy_family,
                symbol=symbol,
                timeframe=timeframe,
                scenario=scenario,
                parameters=study.best_params,
                date_range_start=indexed_ts[0].to_pydatetime(),
                date_range_end=indexed_ts[-1].to_pydatetime(),
                metrics={**best_result.metrics, "objective": study.best_value},
                data_version=data_version_for(symbol, timeframe),
                code_version=git_commit_hash(),
                python_version=python_version(),
                library_versions=library_versions(["pandas", "numpy", "optuna", "vectorbt"]),
                random_seed=seed,
            ),
            db=db,
        )

    return OptimizationResult(
        experiment_id=experiment_id,
        strategy=strategy_family,
        symbol=symbol,
        timeframe=timeframe,
        scenario=scenario,
        best_params=study.best_params,
        best_metrics=best_result.metrics,
        best_objective=study.best_value,
        n_trials=n_trials,
    )


@dataclass
class StabilityResult:
    score: float  # 0 (isolated spike) .. 1 (stable region)
    neighbor_objectives: dict[str, list[float]] = field(default_factory=dict)


def assess_parameter_stability(
    strategy_family: str,
    symbol: str,
    timeframe: str,
    raw_df: pd.DataFrame,
    best_params: dict,
    space: dict[str, ParamSpec],
    *,
    scenario: str | None = None,
) -> StabilityResult:
    """CLAUDE.md Section 16: never trust an isolated spike. Perturbs each
    tunable parameter by one step (holding the others at `best_params`)
    and re-scores; a stable region has neighbors scoring nearly as well
    as the best point, an isolated spike collapses on its neighbors.
    Runs len(space)*2 additional vectorbt screens — cheap enough to do
    once after the search, prohibitively expensive to do inside every
    trial (which is why this is a separate, post-hoc step).
    """
    settings = load_settings()
    cfg = settings.optimization
    scenario = scenario or settings.execution.default_scenario

    def score(params: dict) -> float:
        feature_kwargs, strategy_kwargs = split_params(params)
        feature_params = replace(FeatureParams.from_settings(), **feature_kwargs)
        result = run_screening(
            strategy_family,
            symbol,
            timeframe,
            raw_df,
            strategy_params=strategy_kwargs,
            feature_params=feature_params,
            scenario=scenario,
            persist=False,
        )
        return composite_objective(result.metrics, cfg)

    best_score = score(best_params)
    neighbor_objectives: dict[str, list[float]] = {}

    for name, spec in space.items():
        if spec.type == "int":
            step = int(spec.step or cfg.stability.neighbor_step_int)
        else:
            step = spec.step or abs(best_params[name] * cfg.stability.neighbor_step_pct) or cfg.stability.neighbor_step_pct

        neighbor_scores = []
        for direction in (-1, 1):
            neighbor_value = best_params[name] + direction * step
            neighbor_value = max(spec.low, min(spec.high, neighbor_value))
            if neighbor_value == best_params[name]:
                continue
            neighbor_params = {**best_params, name: neighbor_value}
            neighbor_scores.append(score(neighbor_params))
        neighbor_objectives[name] = neighbor_scores

    all_neighbor_scores = [s for scores in neighbor_objectives.values() for s in scores]
    if not all_neighbor_scores or best_score <= 0:
        stability_score = 0.0
    else:
        ratios = [max(0.0, min(1.0, s / best_score)) for s in all_neighbor_scores]
        stability_score = sum(ratios) / len(ratios)

    return StabilityResult(score=stability_score, neighbor_objectives=neighbor_objectives)
