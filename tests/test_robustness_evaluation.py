import numpy as np
import pandas as pd
import pytest

from src.core.config import load_strategies
from src.core.db import Database
from src.data.schema import STANDARD_COLUMNS
from src.robustness.evaluation import run_robustness_evaluation
from src.robustness.run_store import list_evaluations


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


def test_run_robustness_evaluation_end_to_end_persists(db):
    raw = _synthetic_raw_candles()
    space = load_strategies().strategies["trend_following"].optimization_space

    result = run_robustness_evaluation(
        "trend_following", "EURUSD", "H1", raw, space, n_trials=3, mc_simulations=150, db=db,
    )

    assert result.score.status is not None
    assert result.walkforward_report.windows
    assert result.cost_stress_result.points
    assert result.capital_simulation.risk_levels

    records = list_evaluations(db=db)
    assert len(records) == 1
    assert records[0].run_id == result.run_id
    assert records[0].status == result.score.status.value
    assert records[0].best_params == result.best_params


def test_run_robustness_evaluation_persist_false_does_not_write_to_db(db):
    raw = _synthetic_raw_candles()
    space = load_strategies().strategies["trend_following"].optimization_space
    run_robustness_evaluation(
        "trend_following", "EURUSD", "H1", raw, space, n_trials=3, mc_simulations=150, db=db, persist=False,
    )
    assert list_evaluations(db=db) == []


def test_to_text_includes_every_sub_report(db):
    raw = _synthetic_raw_candles()
    space = load_strategies().strategies["trend_following"].optimization_space
    result = run_robustness_evaluation(
        "trend_following", "EURUSD", "H1", raw, space, n_trials=3, mc_simulations=150, db=db,
    )
    text = result.to_text()
    assert "FINAL ROBUSTNESS EVALUATION" in text
    assert "WALK-FORWARD ANALYSIS REPORT" in text
    assert "TRANSACTION COST STRESS TEST" in text
    assert "CAPITAL SIMULATION" in text
