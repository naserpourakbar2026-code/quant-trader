import numpy as np
import pandas as pd
import pytest

from src.core.db import Database
from src.data.schema import STANDARD_COLUMNS
from src.execution.paper_trading import run_paper_trading_session
from src.montecarlo.mc_engine import run_monte_carlo
from src.reporting.data import gather_report_data


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


def test_gather_report_data_is_all_empty_on_a_fresh_database(db):
    data = gather_report_data(db=db)
    assert data.experiments == []
    assert data.walkforward_windows == []
    assert data.montecarlo_runs == []
    assert data.portfolio_runs == []
    assert data.paper_sessions == []
    assert data.paper_trades_by_run == {}
    assert data.kill_switch_events == []


def test_gather_report_data_includes_montecarlo_and_paper_trading_results(db):
    raw = _synthetic_raw_candles()
    run_monte_carlo("trend_following", "EURUSD", "H1", raw, n_simulations=100, db=db)
    session = run_paper_trading_session("trend_following", "EURUSD", "H1", raw, db=db)

    data = gather_report_data(db=db)
    assert len(data.montecarlo_runs) == 1
    assert len(data.paper_sessions) == 1
    assert session.run_id in data.paper_trades_by_run
    assert len(data.paper_trades_by_run[session.run_id]) == len(session.closed_trades)


def test_gather_report_data_only_charts_the_most_recent_sessions(db):
    raw = _synthetic_raw_candles()
    sessions = [run_paper_trading_session("trend_following", "EURUSD", "H1", raw, db=db) for _ in range(3)]

    data = gather_report_data(db=db, max_paper_sessions=2)
    assert len(data.paper_sessions) == 3  # every session still listed
    assert len(data.paper_trades_by_run) == 2  # but only the 2 most recent are charted
    most_recent_two = {s.run_id for s in data.paper_sessions[:2]}
    assert set(data.paper_trades_by_run) == most_recent_two
