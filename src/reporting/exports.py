"""CSV/JSON exports (CLAUDE.md Section 30: "Outputs: HTML reports, CSV
exports, JSON results."). One CSV per persisted table, always written
with a header even when there are zero rows yet — an empty export is a
legitimate "nothing here yet," not something to skip and leave the user
wondering whether the export ran at all.
"""
from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path

from src.backtest.experiment_store import ExperimentRecord
from src.execution.trade_store import PaperSessionRecord, PaperTradeRecord
from src.montecarlo.run_store import MonteCarloRecord
from src.portfolio.run_store import PortfolioRecord
from src.reporting.data import ReportData
from src.risk.kill_switch import KillSwitchEventRecord
from src.walkforward.window_store import WindowRecord


def _asdict_json_safe(record) -> dict:
    """dataclasses.asdict(), then JSON-encode any nested dict/list field
    so it survives a flat CSV cell; datetimes become ISO strings."""

    def _default(value):
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return str(value)

    flat = {}
    for key, value in dataclasses.asdict(record).items():
        if isinstance(value, (dict, list)):
            flat[key] = json.dumps(value, default=_default)
        elif hasattr(value, "isoformat"):
            flat[key] = value.isoformat()
        else:
            flat[key] = value
    return flat


def _write_csv(path: Path, records: list, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(_asdict_json_safe(record))


# field name on ReportData -> (output filename, the dataclass its list holds).
# Fieldnames are derived from the dataclass itself (dataclasses.fields()),
# not hand-typed, so this can never drift out of sync with a model change
# the way a manually-maintained column list just did during development.
_EXPORT_TABLES: dict[str, tuple[str, type]] = {
    "experiments": ("experiments.csv", ExperimentRecord),
    "walkforward_windows": ("walkforward_windows.csv", WindowRecord),
    "montecarlo_runs": ("montecarlo_runs.csv", MonteCarloRecord),
    "portfolio_runs": ("portfolio_runs.csv", PortfolioRecord),
    "paper_sessions": ("paper_sessions.csv", PaperSessionRecord),
    "kill_switch_events": ("kill_switch_events.csv", KillSwitchEventRecord),
}


def write_csv_exports(data: ReportData, output_dir: Path) -> list[Path]:
    written = []
    for field_name, (filename, record_type) in _EXPORT_TABLES.items():
        records = getattr(data, field_name)
        columns = [f.name for f in dataclasses.fields(record_type)]
        path = output_dir / filename
        _write_csv(path, records, columns)
        written.append(path)

    # paper_trades is keyed by run_id -- flatten into one CSV across every
    # charted session's trades, same convention as the rest.
    all_trades = [trade for trades in data.paper_trades_by_run.values() for trade in trades]
    trades_path = output_dir / "paper_trades.csv"
    _write_csv(trades_path, all_trades, [f.name for f in dataclasses.fields(PaperTradeRecord)])
    written.append(trades_path)
    return written


def write_json_export(data: ReportData, output_path: Path) -> Path:
    def _default(value):
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return str(value)

    payload = {
        "experiments": [dataclasses.asdict(r) for r in data.experiments],
        "walkforward_windows": [dataclasses.asdict(r) for r in data.walkforward_windows],
        "montecarlo_runs": [dataclasses.asdict(r) for r in data.montecarlo_runs],
        "portfolio_runs": [dataclasses.asdict(r) for r in data.portfolio_runs],
        "paper_sessions": [dataclasses.asdict(r) for r in data.paper_sessions],
        "paper_trades": {
            run_id: [dataclasses.asdict(t) for t in trades] for run_id, trades in data.paper_trades_by_run.items()
        },
        "kill_switch_events": [dataclasses.asdict(r) for r in data.kill_switch_events],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=_default), encoding="utf-8")
    return output_path
