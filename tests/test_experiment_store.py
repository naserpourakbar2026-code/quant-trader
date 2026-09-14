from datetime import datetime, timezone

import pytest

from src.core.db import Database
from src.backtest.experiment_store import (
    ExperimentRecord,
    get_experiment,
    list_experiments,
    save_experiment,
)


@pytest.fixture()
def db():
    database = Database(url="sqlite:///:memory:")
    database.init_db()
    return database


def _make_record(experiment_id="exp-1", strategy="trend_following", symbol="EURUSD", timeframe="H1", **overrides):
    defaults = dict(
        experiment_id=experiment_id,
        strategy=strategy,
        symbol=symbol,
        timeframe=timeframe,
        scenario="realistic",
        parameters={"atr_stop_multiplier": 2.0},
        date_range_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        date_range_end=datetime(2024, 6, 1, tzinfo=timezone.utc),
        metrics={"sharpe_ratio": 1.2, "profit_factor": float("inf"), "max_drawdown": float("nan")},
        data_version="abc123",
        code_version="deadbeef",
        python_version="3.11.15",
        library_versions={"pandas": "2.3.3"},
        random_seed=42,
    )
    defaults.update(overrides)
    return ExperimentRecord(**defaults)


def test_save_and_get_experiment_round_trips(db):
    record = _make_record()
    save_experiment(record, db=db)

    fetched = get_experiment("exp-1", db=db)
    assert fetched is not None
    assert fetched.strategy == "trend_following"
    assert fetched.parameters == {"atr_stop_multiplier": 2.0}
    assert fetched.random_seed == 42


def test_non_finite_metrics_are_sanitized_to_none(db):
    save_experiment(_make_record(), db=db)
    fetched = get_experiment("exp-1", db=db)
    assert fetched.metrics["profit_factor"] is None
    assert fetched.metrics["max_drawdown"] is None
    assert fetched.metrics["sharpe_ratio"] == pytest.approx(1.2)


def test_get_experiment_returns_none_when_absent(db):
    assert get_experiment("does-not-exist", db=db) is None


def test_list_experiments_filters_by_strategy_symbol_timeframe(db):
    save_experiment(_make_record(experiment_id="a", strategy="trend_following", symbol="EURUSD"), db=db)
    save_experiment(_make_record(experiment_id="b", strategy="mean_reversion", symbol="EURUSD"), db=db)
    save_experiment(_make_record(experiment_id="c", strategy="trend_following", symbol="GBPUSD"), db=db)

    all_records = list_experiments(db=db)
    assert len(all_records) == 3

    trend_only = list_experiments(strategy="trend_following", db=db)
    assert {r.experiment_id for r in trend_only} == {"a", "c"}

    eurusd_trend = list_experiments(strategy="trend_following", symbol="EURUSD", db=db)
    assert {r.experiment_id for r in eurusd_trend} == {"a"}


def test_experiment_id_must_be_unique(db):
    save_experiment(_make_record(experiment_id="dup"), db=db)
    with pytest.raises(Exception):
        save_experiment(_make_record(experiment_id="dup"), db=db)
