"""Read/write access to the `experiments` table (CLAUDE.md Sections 12, 29).

Kept separate from src.backtest.vectorbt_engine so the persistence layer
can be reused unchanged by Backtrader validation (Phase 7), Optuna (Phase
8) and walk-forward/Monte Carlo (Phases 9-10) — they all produce the same
shape of "experiment" record, just from a different engine.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime

from src.core.db import Database, get_default_database
from src.backtest.models import Experiment


def _json_safe(value):
    """Replace non-finite floats (inf/-inf/nan) with None so a metrics
    dict round-trips through JSON cleanly — vectorbt legitimately produces
    these (e.g. profit factor with zero losing trades)."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


@dataclass
class ExperimentRecord:
    experiment_id: str
    strategy: str
    symbol: str
    timeframe: str
    scenario: str
    parameters: dict
    date_range_start: datetime
    date_range_end: datetime
    metrics: dict
    data_version: str
    code_version: str
    python_version: str
    library_versions: dict
    random_seed: int | None = None


def save_experiment(record: ExperimentRecord, db: Database | None = None) -> None:
    database = db or get_default_database()
    with database.session() as session:
        session.add(
            Experiment(
                experiment_id=record.experiment_id,
                strategy=record.strategy,
                symbol=record.symbol,
                timeframe=record.timeframe,
                scenario=record.scenario,
                parameters_json=json.dumps(_json_safe(record.parameters)),
                date_range_start=record.date_range_start,
                date_range_end=record.date_range_end,
                metrics_json=json.dumps(_json_safe(record.metrics)),
                data_version=record.data_version,
                code_version=record.code_version,
                python_version=record.python_version,
                library_versions_json=json.dumps(record.library_versions),
                random_seed=record.random_seed,
            )
        )
        session.commit()


def _to_record(row: Experiment) -> ExperimentRecord:
    return ExperimentRecord(
        experiment_id=row.experiment_id,
        strategy=row.strategy,
        symbol=row.symbol,
        timeframe=row.timeframe,
        scenario=row.scenario,
        parameters=json.loads(row.parameters_json),
        date_range_start=row.date_range_start,
        date_range_end=row.date_range_end,
        metrics=json.loads(row.metrics_json),
        data_version=row.data_version,
        code_version=row.code_version,
        python_version=row.python_version,
        library_versions=json.loads(row.library_versions_json),
        random_seed=row.random_seed,
    )


def get_experiment(experiment_id: str, db: Database | None = None) -> ExperimentRecord | None:
    database = db or get_default_database()
    with database.session() as session:
        row = session.query(Experiment).filter_by(experiment_id=experiment_id).one_or_none()
        return _to_record(row) if row else None


def list_experiments(
    *,
    strategy: str | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
    db: Database | None = None,
) -> list[ExperimentRecord]:
    database = db or get_default_database()
    with database.session() as session:
        query = session.query(Experiment)
        if strategy is not None:
            query = query.filter_by(strategy=strategy)
        if symbol is not None:
            query = query.filter_by(symbol=symbol)
        if timeframe is not None:
            query = query.filter_by(timeframe=timeframe)
        return [_to_record(row) for row in query.order_by(Experiment.created_at.desc()).all()]
