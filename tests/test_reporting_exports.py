import csv
import json

import pandas as pd
import pytest

from src.montecarlo.run_store import MonteCarloRecord
from src.portfolio.run_store import PortfolioRecord
from src.reporting.data import ReportData
from src.reporting.exports import write_csv_exports, write_json_export


def _empty_data() -> ReportData:
    return ReportData()


def test_write_csv_exports_creates_every_file_with_a_header_even_when_empty(tmp_path):
    paths = write_csv_exports(_empty_data(), tmp_path)
    names = {p.name for p in paths}
    assert names == {
        "experiments.csv", "walkforward_windows.csv", "montecarlo_runs.csv", "portfolio_runs.csv",
        "paper_sessions.csv", "kill_switch_events.csv", "paper_trades.csv", "robustness_evaluations.csv",
    }
    for path in paths:
        with path.open() as f:
            reader = csv.reader(f)
            header = next(reader)
            assert len(header) > 0
            assert list(reader) == []  # no data rows, but the file exists and has a header


def test_write_csv_exports_json_encodes_nested_fields(tmp_path):
    portfolio_run = PortfolioRecord(
        run_id="p1", scenario="realistic", n_components=2, components=[{"strategy": "a"}],
        correlation={"a": {"a": 1.0, "b": 0.2}}, allocations={"equal_weight": {"weights": {"a": 0.5}}},
        best_component_label="a",
    )
    data = ReportData(portfolio_runs=[portfolio_run])

    paths = write_csv_exports(data, tmp_path)
    portfolio_csv = next(p for p in paths if p.name == "portfolio_runs.csv")
    with portfolio_csv.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    correlation_cell = json.loads(rows[0]["correlation"])
    assert correlation_cell == {"a": {"a": 1.0, "b": 0.2}}


def test_write_json_export_is_valid_and_reflects_counts(tmp_path):
    mc_run = MonteCarloRecord(
        run_id="mc1", strategy="trend_following", symbol="EURUSD", timeframe="H1", scenario="realistic",
        n_simulations=100, n_trades_observed=50, initial_capital=2000.0, ruin_threshold=0.5, median_return=0.1,
        p5_return=-0.05, p95_return=0.3, worst_drawdown=-0.2, p95_drawdown=-0.15, median_losing_streak=2.0,
        p95_losing_streak=4.0, worst_losing_streak=6, probability_of_ruin=0.01, probability_of_negative_return=0.1,
        is_fragile=False, parameters={}, created_at=pd.Timestamp("2024-01-01", tz="UTC"),
    )
    data = ReportData(montecarlo_runs=[mc_run])
    path = write_json_export(data, tmp_path / "report.json")

    payload = json.loads(path.read_text())
    assert len(payload["montecarlo_runs"]) == 1
    assert payload["montecarlo_runs"][0]["run_id"] == "mc1"
    assert payload["experiments"] == []
    assert isinstance(payload["montecarlo_runs"][0]["created_at"], str)  # datetime serialized, didn't crash json.dumps


def test_write_json_export_creates_parent_directories(tmp_path):
    path = write_json_export(_empty_data(), tmp_path / "nested" / "dir" / "report.json")
    assert path.exists()
