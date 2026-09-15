import numpy as np
import pandas as pd
import pytest

from src.core.db import Database
from src.data.schema import STANDARD_COLUMNS
from src.execution.paper_trading import run_paper_trading_session
from src.montecarlo.mc_engine import run_monte_carlo
from src.reporting.report_engine import run_report


@pytest.fixture()
def db():
    database = Database(url="sqlite:///:memory:")
    database.init_db()
    return database


def _synthetic_raw_candles(n=1500, seed=1, symbol="EURUSD", timeframe="H1"):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2022-01-01T00:00:00Z", periods=n, freq="1h")
    third = n // 3
    drift = np.concatenate([np.full(third, 0.0008), np.full(third, 0.0), np.full(n - 2 * third, -0.0008)])
    noise_std = np.concatenate([np.full(third, 0.0003), np.full(third, 0.0006), np.full(n - 2 * third, 0.0003)])
    close = 1.10 + np.cumsum(drift + rng.normal(0, 1, size=n) * noise_std)
    high = close + rng.uniform(0.0001, 0.0004, size=n)
    low = close - rng.uniform(0.0001, 0.0004, size=n)
    open_ = close + rng.normal(0, 0.0002, size=n)
    return pd.DataFrame(
        {
            "timestamp": ts, "symbol": symbol, "timeframe": timeframe, "open": open_, "high": high, "low": low,
            "close": close, "tick_volume": 100.0, "spread": 0.00015, "real_volume": float("nan"),
        }
    )[STANDARD_COLUMNS]


def test_run_report_on_an_empty_database_writes_valid_placeholder_files(db, tmp_path):
    result = run_report(db=db, output_dir=tmp_path)
    assert result.html_path.exists()
    assert result.html_path.read_text().startswith("<!doctype html>")
    assert "no server needed" in result.html_path.read_text()
    for path in result.csv_paths:
        assert path.exists()
    assert result.json_path.exists()


def test_run_report_html_mentions_no_data_yet_when_nothing_persisted(db, tmp_path):
    result = run_report(db=db, output_dir=tmp_path)
    html = result.html_path.read_text()
    assert "No experiments yet" in html
    assert "No Monte Carlo runs yet" in html
    assert "No paper-trading sessions yet" in html
    assert "No robustness evaluations yet" in html
    assert "NO ROBUST STRATEGY FOUND" in html


def test_run_report_shows_strategy_selection_for_a_passing_robustness_evaluation(db, tmp_path):
    from src.core.config import load_strategies
    from src.robustness.evaluation import run_robustness_evaluation

    raw = _synthetic_raw_candles()
    space = load_strategies().strategies["trend_following"].optimization_space
    run_robustness_evaluation("trend_following", "EURUSD", "H1", raw, space, n_trials=3, mc_simulations=150, db=db)

    result = run_report(db=db, output_dir=tmp_path)
    html = result.html_path.read_text()
    assert len(result.data.robustness_evaluations) == 1
    if result.data.robustness_evaluations[0].status == "PASS":
        assert "🥇" in html
        assert "NO ROBUST STRATEGY FOUND" not in html
    else:
        assert "NO ROBUST STRATEGY FOUND" in html


def test_run_report_reflects_real_persisted_results(db, tmp_path):
    raw = _synthetic_raw_candles()
    run_monte_carlo("trend_following", "EURUSD", "H1", raw, n_simulations=150, db=db)
    run_paper_trading_session("trend_following", "EURUSD", "H1", raw, db=db)

    result = run_report(db=db, output_dir=tmp_path)
    html = result.html_path.read_text()
    assert "trend_following" in html
    assert "EURUSD" in html
    assert "No Monte Carlo runs yet" not in html
    assert "No paper-trading sessions yet" not in html
    assert len(result.data.montecarlo_runs) == 1
    assert len(result.data.paper_sessions) == 1


def test_run_report_writes_under_the_given_output_dir(db, tmp_path):
    result = run_report(db=db, output_dir=tmp_path)
    assert result.html_path.parent == tmp_path
    assert result.json_path.parent == tmp_path / "exports"


def test_run_report_respects_max_paper_sessions(db, tmp_path):
    raw = _synthetic_raw_candles()
    for _ in range(3):
        run_paper_trading_session("trend_following", "EURUSD", "H1", raw, db=db)

    result = run_report(db=db, output_dir=tmp_path, max_paper_sessions=1)
    assert len(result.data.paper_sessions) == 3
    assert len(result.data.paper_trades_by_run) == 1
