"""Read/write access to the `portfolio_runs` table (CLAUDE.md Sections
20, 29). Same shape/conventions as src.backtest.experiment_store,
src.walkforward.window_store and src.montecarlo.run_store.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime

from src.core.db import Database, get_default_database
from src.portfolio.models import PortfolioRun


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


@dataclass
class PortfolioRecord:
    run_id: str
    scenario: str
    n_components: int
    components: list
    correlation: dict
    allocations: dict
    best_component_label: str
    code_version: str = "unknown"
    python_version: str = "unknown"
    library_versions: dict = field(default_factory=dict)
    created_at: datetime | None = None


def save_run(record: PortfolioRecord, db: Database | None = None) -> None:
    database = db or get_default_database()
    with database.session() as session:
        session.add(
            PortfolioRun(
                run_id=record.run_id,
                scenario=record.scenario,
                n_components=record.n_components,
                components_json=json.dumps(_json_safe(record.components)),
                correlation_json=json.dumps(_json_safe(record.correlation)),
                allocations_json=json.dumps(_json_safe(record.allocations)),
                best_component_label=record.best_component_label,
                code_version=record.code_version,
                python_version=record.python_version,
                library_versions_json=json.dumps(record.library_versions),
            )
        )
        session.commit()


def _to_record(row: PortfolioRun) -> PortfolioRecord:
    return PortfolioRecord(
        run_id=row.run_id,
        scenario=row.scenario,
        n_components=row.n_components,
        components=json.loads(row.components_json),
        correlation=json.loads(row.correlation_json),
        allocations=json.loads(row.allocations_json),
        best_component_label=row.best_component_label,
        code_version=row.code_version,
        python_version=row.python_version,
        library_versions=json.loads(row.library_versions_json),
        created_at=row.created_at,
    )


def get_run(run_id: str, db: Database | None = None) -> PortfolioRecord | None:
    database = db or get_default_database()
    with database.session() as session:
        row = session.query(PortfolioRun).filter_by(run_id=run_id).one_or_none()
        return _to_record(row) if row else None


def list_runs(db: Database | None = None) -> list[PortfolioRecord]:
    database = db or get_default_database()
    with database.session() as session:
        query = session.query(PortfolioRun)
        return [_to_record(row) for row in query.order_by(PortfolioRun.created_at.desc()).all()]
