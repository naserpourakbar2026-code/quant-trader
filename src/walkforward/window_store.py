"""Read/write access to the `walkforward_windows` table (CLAUDE.md
Sections 15, 29). Same shape/conventions as src.backtest.experiment_store.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime

from src.core.db import Database, get_default_database
from src.walkforward.models import WalkForwardWindow


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


@dataclass
class WindowRecord:
    run_id: str
    window_index: int
    strategy: str
    symbol: str
    timeframe: str
    scenario: str
    train_start: datetime
    train_end: datetime
    validation_start: datetime
    validation_end: datetime
    oos_start: datetime
    oos_end: datetime
    best_params: dict
    train_objective: float
    stability_score: float
    validation_metrics: dict
    oos_metrics: dict
    oos_objective: float
    oos_passed: bool
    code_version: str = "unknown"
    random_seed: int | None = None


def save_window(record: WindowRecord, db: Database | None = None) -> None:
    database = db or get_default_database()
    with database.session() as session:
        session.add(
            WalkForwardWindow(
                run_id=record.run_id,
                window_index=record.window_index,
                strategy=record.strategy,
                symbol=record.symbol,
                timeframe=record.timeframe,
                scenario=record.scenario,
                train_start=record.train_start,
                train_end=record.train_end,
                validation_start=record.validation_start,
                validation_end=record.validation_end,
                oos_start=record.oos_start,
                oos_end=record.oos_end,
                best_params_json=json.dumps(_json_safe(record.best_params)),
                train_objective=record.train_objective,
                stability_score=record.stability_score,
                validation_metrics_json=json.dumps(_json_safe(record.validation_metrics)),
                oos_metrics_json=json.dumps(_json_safe(record.oos_metrics)),
                oos_objective=record.oos_objective,
                oos_passed=record.oos_passed,
                code_version=record.code_version,
                random_seed=record.random_seed,
            )
        )
        session.commit()


def _to_record(row: WalkForwardWindow) -> WindowRecord:
    return WindowRecord(
        run_id=row.run_id,
        window_index=row.window_index,
        strategy=row.strategy,
        symbol=row.symbol,
        timeframe=row.timeframe,
        scenario=row.scenario,
        train_start=row.train_start,
        train_end=row.train_end,
        validation_start=row.validation_start,
        validation_end=row.validation_end,
        oos_start=row.oos_start,
        oos_end=row.oos_end,
        best_params=json.loads(row.best_params_json),
        train_objective=row.train_objective,
        stability_score=row.stability_score,
        validation_metrics=json.loads(row.validation_metrics_json),
        oos_metrics=json.loads(row.oos_metrics_json),
        oos_objective=row.oos_objective,
        oos_passed=row.oos_passed,
        code_version=row.code_version,
        random_seed=row.random_seed,
    )


def list_windows(
    *,
    run_id: str | None = None,
    strategy: str | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
    db: Database | None = None,
) -> list[WindowRecord]:
    database = db or get_default_database()
    with database.session() as session:
        query = session.query(WalkForwardWindow)
        if run_id is not None:
            query = query.filter_by(run_id=run_id)
        if strategy is not None:
            query = query.filter_by(strategy=strategy)
        if symbol is not None:
            query = query.filter_by(symbol=symbol)
        if timeframe is not None:
            query = query.filter_by(timeframe=timeframe)
        return [_to_record(row) for row in query.order_by(WalkForwardWindow.window_index.asc()).all()]
