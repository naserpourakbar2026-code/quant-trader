import numpy as np
import pandas as pd
import pytest

from src.backtest.backtrader_engine import compare_with_screening, run_validation
from src.backtest.experiment_store import list_experiments
from src.backtest.vectorbt_engine import run_screening
from src.core.db import Database
from src.data.schema import STANDARD_COLUMNS


@pytest.fixture()
def db():
    database = Database(url="sqlite:///:memory:")
    database.init_db()
    return database


def _synthetic_raw_candles(n=400, seed=1, symbol="EURUSD", timeframe="H1"):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-02T00:00:00Z", periods=n, freq="1h")
    third = n // 3
    drift = np.concatenate([np.full(third, 0.0015), np.full(third, 0.0), np.full(n - 2 * third, -0.0015)])
    noise_std = np.concatenate([np.full(third, 0.0003), np.full(third, 0.0007), np.full(n - 2 * third, 0.0003)])
    close = 1.10 + np.cumsum(drift + rng.normal(0, 1, size=n) * noise_std)
    high = close + rng.uniform(0.0001, 0.0004, size=n)
    low = close - rng.uniform(0.0001, 0.0004, size=n)
    open_ = close + rng.normal(0, 0.0002, size=n)
    tick_volume = rng.uniform(50, 200, size=n)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": symbol,
            "timeframe": timeframe,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "tick_volume": tick_volume,
            "spread": 0.00015,  # ~1.5 pips for EURUSD, same price units as OHLC
            "real_volume": float("nan"),
        }
    )[STANDARD_COLUMNS]


def test_run_validation_returns_sane_metrics_and_persists(db):
    raw = _synthetic_raw_candles()
    result = run_validation("trend_following", "EURUSD", "H1", raw, db=db)

    assert result.strategy == "trend_following"
    assert result.symbol == "EURUSD"
    # Real activity, not a silently-broken engine that never fills anything:
    # this pins the fix for a real bug found while building this test (see
    # backtrader_engine.py's SignalDrivenStrategy docstring/comments) where
    # every single order was margin-rejected.
    assert result.metrics["trade_count"] > 0
    assert result.metrics["rejected_orders"] == 0
    assert result.metrics["margin_orders"] == 0
    assert result.metrics["win_rate"] is not None

    records = list_experiments(db=db, engine="backtrader")
    assert len(records) == 1
    assert records[0].engine == "backtrader"
    assert records[0].experiment_id == result.experiment_id


def test_max_drawdown_sign_matches_vectorbt_convention(db):
    """Backtrader's own DrawDown analyzer reports a positive percentage;
    vectorbt's max_drawdown() is negative. Metrics from both engines get
    compared (compare_with_screening), so a mismatched sign would silently
    produce a meaningless delta -- this pins the conversion."""
    raw = _synthetic_raw_candles()
    result = run_validation("trend_following", "EURUSD", "H1", raw, db=db)
    assert result.metrics["max_drawdown"] <= 0


def test_run_validation_never_exceeds_available_margin(db):
    """A strategy/cost combination that can't afford its own sizing should
    show up as rejected/margin orders, never as a silent zero-trade run or
    a crash — regression guard for the margin-rejection bug."""
    raw = _synthetic_raw_candles()
    result = run_validation("trend_following", "EURUSD", "H1", raw, db=db)
    assert result.metrics["rejected_orders"] == 0
    assert result.metrics["margin_orders"] == 0


def test_all_three_strategy_families_produce_real_trades(db):
    raw = _synthetic_raw_candles()
    for family in ("trend_following", "mean_reversion", "momentum_multi_factor"):
        result = run_validation(family, "EURUSD", "H1", raw, db=db)
        assert result.metrics["trade_count"] > 0, f"{family} produced no trades at all"
        assert result.metrics["margin_orders"] == 0, f"{family} hit margin rejections"


def test_run_validation_persist_false_does_not_write_to_db(db):
    raw = _synthetic_raw_candles()
    run_validation("mean_reversion", "EURUSD", "H1", raw, db=db, persist=False)
    assert list_experiments(db=db) == []


def test_run_validation_rejects_unknown_scenario(db):
    raw = _synthetic_raw_candles()
    with pytest.raises(ValueError, match="Unknown execution scenario"):
        run_validation("trend_following", "EURUSD", "H1", raw, scenario="not_a_scenario", db=db)


def test_vectorbt_and_backtrader_can_be_compared_for_the_same_run(db):
    raw = _synthetic_raw_candles()
    screening = run_screening("trend_following", "EURUSD", "H1", raw, db=db)
    validation = run_validation("trend_following", "EURUSD", "H1", raw, db=db)

    comparison = compare_with_screening(screening.metrics, validation.metrics)
    assert "total_return" in comparison
    assert "sharpe_ratio" in comparison
    for entry in comparison.values():
        assert "screening" in entry and "validation" in entry and "delta" in entry

    # Both engines should be querying the same underlying data/strategy,
    # so both runs must exist side by side, distinguishable by engine.
    all_records = list_experiments(db=db)
    assert {r.engine for r in all_records} == {"vectorbt", "backtrader"}


def test_higher_costs_never_improve_backtrader_return(db):
    raw = _synthetic_raw_candles()
    optimistic = run_validation("trend_following", "EURUSD", "H1", raw, scenario="optimistic", db=db)
    stress = run_validation("trend_following", "EURUSD", "H1", raw, scenario="stress", db=db)
    assert optimistic.metrics["total_return"] >= stress.metrics["total_return"] - 1e-9
