"""Phase 16: Reporting (CLAUDE.md Section 30).

`run_report()` gathers every persisted engine's results
(src.reporting.data), and writes:
  - `reports/report.html` — the primary deliverable: a self-contained,
    pre-rendered HTML file. Opening it in a browser is enough; no server,
    no notebook, no command needed to *view* it (Section 30's explicit
    user preference) — generating it is the only command required.
  - `reports/exports/*.csv` — one CSV per persisted table.
  - `reports/exports/report.json` — the same data as one JSON document.

Every output reflects only what has actually been persisted by running
the other phases' commands against real (or test) data — nothing here
fabricates a result to make the report look more complete.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.core.config import PROJECT_ROOT
from src.core.db import Database
from src.reporting.data import ReportData, gather_report_data
from src.reporting.exports import write_csv_exports, write_json_export
from src.reporting.html_report import render_html_report


@dataclass
class ReportResult:
    data: ReportData
    html_path: Path
    csv_paths: list[Path]
    json_path: Path


def run_report(
    *, db: Database | None = None, output_dir: Path | None = None, max_paper_sessions: int = 5,
) -> ReportResult:
    output_dir = output_dir or (PROJECT_ROOT / "reports")
    exports_dir = output_dir / "exports"

    data = gather_report_data(db=db, max_paper_sessions=max_paper_sessions)

    html_path = output_dir / "report.html"
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path.write_text(render_html_report(data), encoding="utf-8")

    csv_paths = write_csv_exports(data, exports_dir)
    json_path = write_json_export(data, exports_dir / "report.json")

    return ReportResult(data=data, html_path=html_path, csv_paths=csv_paths, json_path=json_path)
