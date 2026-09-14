"""Read/write access to the `montecarlo_runs` table (CLAUDE.md Sections
17, 29). Same shape/conventions as src.backtest.experiment_store and
src.walkforward.window_store.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime

from src.core.db import Database, get_default_database
from src.montecarlo.models import MonteCarloRun


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


@dataclass
class MonteCarloRecord:
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
    parameters: dict
    code_version: str = "unknown"
    python_version: str = "unknown"
    library_versions: dict | None = None
    random_seed: int | None = None
    created_at: datetime | None = None


def save_run(record: MonteCarloRecord, db: Database | None = None) -> None:
    database = db or get_default_database()
    with database.session() as session:
        session.add(
            MonteCarloRun(
                run_id=record.run_id,
                strategy=record.strategy,
                symbol=record.symbol,
                timeframe=record.timeframe,
                scenario=record.scenario,
                n_simulations=record.n_simulations,
                n_trades_observed=record.n_trades_observed,
                initial_capital=record.initial_capital,
                ruin_threshold=record.ruin_threshold,
                median_return=record.median_return,
                p5_return=record.p5_return,
                p95_return=record.p95_return,
                worst_drawdown=record.worst_drawdown,
                p95_drawdown=record.p95_drawdown,
                median_losing_streak=record.median_losing_streak,
                p95_losing_streak=record.p95_losing_streak,
                worst_losing_streak=record.worst_losing_streak,
                probability_of_ruin=record.probability_of_ruin,
                probability_of_negative_return=record.probability_of_negative_return,
                is_fragile=record.is_fragile,
                parameters_json=json.dumps(_json_safe(record.parameters)),
                code_version=record.code_version,
                python_version=record.python_version,
                library_versions_json=json.dumps(record.library_versions or {}),
                random_seed=record.random_seed,
            )
        )
        session.commit()


def _to_record(row: MonteCarloRun) -> MonteCarloRecord:
    return MonteCarloRecord(
        run_id=row.run_id,
        strategy=row.strategy,
        symbol=row.symbol,
        timeframe=row.timeframe,
        scenario=row.scenario,
        n_simulations=row.n_simulations,
        n_trades_observed=row.n_trades_observed,
        initial_capital=row.initial_capital,
        ruin_threshold=row.ruin_threshold,
        median_return=row.median_return,
        p5_return=row.p5_return,
        p95_return=row.p95_return,
        worst_drawdown=row.worst_drawdown,
        p95_drawdown=row.p95_drawdown,
        median_losing_streak=row.median_losing_streak,
        p95_losing_streak=row.p95_losing_streak,
        worst_losing_streak=row.worst_losing_streak,
        probability_of_ruin=row.probability_of_ruin,
        probability_of_negative_return=row.probability_of_negative_return,
        is_fragile=row.is_fragile,
        parameters=json.loads(row.parameters_json),
        code_version=row.code_version,
        python_version=row.python_version,
        library_versions=json.loads(row.library_versions_json),
        random_seed=row.random_seed,
        created_at=row.created_at,
    )


def get_run(run_id: str, db: Database | None = None) -> MonteCarloRecord | None:
    database = db or get_default_database()
    with database.session() as session:
        row = session.query(MonteCarloRun).filter_by(run_id=run_id).one_or_none()
        return _to_record(row) if row else None


def list_runs(
    *,
    strategy: str | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
    db: Database | None = None,
) -> list[MonteCarloRecord]:
    database = db or get_default_database()
    with database.session() as session:
        query = session.query(MonteCarloRun)
        if strategy is not None:
            query = query.filter_by(strategy=strategy)
        if symbol is not None:
            query = query.filter_by(symbol=symbol)
        if timeframe is not None:
            query = query.filter_by(timeframe=timeframe)
        return [_to_record(row) for row in query.order_by(MonteCarloRun.created_at.desc()).all()]
