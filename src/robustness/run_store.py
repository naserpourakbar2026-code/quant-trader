"""Read/write access to the `robustness_evaluations` table (CLAUDE.md
Sections 18, 31, 32, 36). Same shape/conventions as every other phase's
run store.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime

from src.core.db import Database, get_default_database
from src.robustness.models import RobustnessEvaluationRow


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


@dataclass
class RobustnessEvaluationRecord:
    run_id: str
    strategy: str
    symbol: str
    timeframe: str
    status: str
    score: float | None
    sub_scores: dict
    notes: list
    oos_pass_rate: float
    probability_of_ruin: float
    is_cost_fragile: bool
    n_trades: int
    best_params: dict
    walkforward_run_id: str
    montecarlo_run_id: str
    code_version: str = "unknown"
    python_version: str = "unknown"
    library_versions: dict = field(default_factory=dict)
    created_at: datetime | None = None


def save_evaluation(record: RobustnessEvaluationRecord, db: Database | None = None) -> None:
    database = db or get_default_database()
    with database.session() as session:
        session.add(
            RobustnessEvaluationRow(
                run_id=record.run_id, strategy=record.strategy, symbol=record.symbol, timeframe=record.timeframe,
                status=record.status, score=record.score, sub_scores_json=json.dumps(_json_safe(record.sub_scores)),
                notes_json=json.dumps(_json_safe(record.notes)), oos_pass_rate=record.oos_pass_rate,
                probability_of_ruin=record.probability_of_ruin, is_cost_fragile=record.is_cost_fragile,
                n_trades=record.n_trades, best_params_json=json.dumps(_json_safe(record.best_params)),
                walkforward_run_id=record.walkforward_run_id, montecarlo_run_id=record.montecarlo_run_id,
                code_version=record.code_version, python_version=record.python_version,
                library_versions_json=json.dumps(record.library_versions),
            )
        )
        session.commit()


def _to_record(row: RobustnessEvaluationRow) -> RobustnessEvaluationRecord:
    return RobustnessEvaluationRecord(
        run_id=row.run_id, strategy=row.strategy, symbol=row.symbol, timeframe=row.timeframe, status=row.status,
        score=row.score, sub_scores=json.loads(row.sub_scores_json), notes=json.loads(row.notes_json),
        oos_pass_rate=row.oos_pass_rate, probability_of_ruin=row.probability_of_ruin,
        is_cost_fragile=row.is_cost_fragile, n_trades=row.n_trades, best_params=json.loads(row.best_params_json),
        walkforward_run_id=row.walkforward_run_id, montecarlo_run_id=row.montecarlo_run_id,
        code_version=row.code_version, python_version=row.python_version,
        library_versions=json.loads(row.library_versions_json), created_at=row.created_at,
    )


def get_evaluation(run_id: str, db: Database | None = None) -> RobustnessEvaluationRecord | None:
    database = db or get_default_database()
    with database.session() as session:
        row = session.query(RobustnessEvaluationRow).filter_by(run_id=run_id).one_or_none()
        return _to_record(row) if row else None


def list_evaluations(
    *, strategy: str | None = None, symbol: str | None = None, timeframe: str | None = None,
    db: Database | None = None,
) -> list[RobustnessEvaluationRecord]:
    database = db or get_default_database()
    with database.session() as session:
        query = session.query(RobustnessEvaluationRow)
        if strategy is not None:
            query = query.filter_by(strategy=strategy)
        if symbol is not None:
            query = query.filter_by(symbol=symbol)
        if timeframe is not None:
            query = query.filter_by(timeframe=timeframe)
        return [_to_record(row) for row in query.order_by(RobustnessEvaluationRow.created_at.desc()).all()]
