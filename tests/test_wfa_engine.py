import numpy as np
import pandas as pd
import pytest

from src.core.config import ParamSpec, load_strategies
from src.core.db import Database
from src.data.schema import STANDARD_COLUMNS
from src.walkforward.wfa_engine import (
    WalkForwardReport,
    WalkForwardWindowResult,
    generate_windows,
    run_walk_forward,
)
from src.walkforward.window_store import list_windows


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
            "spread": 0.00015,
            "real_volume": float("nan"),
        }
    )[STANDARD_COLUMNS]


def _trend_following_space():
    return load_strategies().strategies["trend_following"].optimization_space


# --- generate_windows -----------------------------------------------------


def test_generate_windows_single_window_spans_whole_series():
    windows = generate_windows(1000, train_pct=0.6, validation_pct=0.2, oos_pct=0.2)
    assert len(windows) == 1
    w = windows[0]
    assert w.train == (0, 600)
    assert w.validation == (600, 800)
    assert w.oos == (800, 1000)


def test_generate_windows_rejects_percentages_not_summing_to_one():
    with pytest.raises(ValueError, match="must sum to 1.0"):
        generate_windows(1000, train_pct=0.5, validation_pct=0.2, oos_pct=0.2)


def test_generate_windows_rejects_window_bars_larger_than_data():
    with pytest.raises(ValueError, match="exceeds available rows"):
        generate_windows(100, window_bars=200)


def test_generate_windows_rejects_too_small_a_window():
    with pytest.raises(ValueError, match="too small"):
        generate_windows(1000, window_bars=3)


def test_generate_windows_rolling_windows_advance_by_step():
    windows = generate_windows(1500, window_bars=600, step_bars=300)
    assert len(windows) == 4
    assert windows[0].train == (0, 360)
    assert windows[1].train == (300, 660)
    # each window's segments are strictly sequential and non-overlapping
    for w in windows:
        assert w.train[1] == w.validation[0]
        assert w.validation[1] == w.oos[0]
        assert w.train[1] - w.train[0] + w.validation[1] - w.validation[0] + w.oos[1] - w.oos[0] == 600


def test_generate_windows_default_step_is_oos_length():
    default_step = generate_windows(1500, window_bars=600)
    explicit_step = generate_windows(1500, window_bars=600, step_bars=120)  # 600*0.2 = 120
    assert [w.train for w in default_step] == [w.train for w in explicit_step]


# --- run_walk_forward: single window (60/20/20 over the whole series) ----


def test_run_walk_forward_single_window_end_to_end(db):
    raw = _synthetic_raw_candles(n=1500)
    space = _trend_following_space()

    report = run_walk_forward("trend_following", "EURUSD", "H1", raw, space, n_trials=5, db=db)

    assert len(report.windows) == 1
    window = report.windows[0]
    assert set(window.best_params) == set(space)
    assert isinstance(window.oos_passed, bool)
    # sequential, non-overlapping periods -- optimization never saw validation/OOS data
    assert window.train_range[1] <= window.validation_range[0]
    assert window.validation_range[1] <= window.oos_range[0]
    assert 0.0 <= window.stability_score <= 1.0

    records = list_windows(db=db, run_id=report.run_id)
    assert len(records) == 1
    assert records[0].oos_passed == window.oos_passed


def test_run_walk_forward_persist_false_does_not_write_to_db(db):
    raw = _synthetic_raw_candles(n=1500)
    space = _trend_following_space()
    run_walk_forward("trend_following", "EURUSD", "H1", raw, space, n_trials=5, db=db, persist=False)
    assert list_windows(db=db) == []


def test_run_walk_forward_rolling_windows(db):
    raw = _synthetic_raw_candles(n=1500)
    space = _trend_following_space()

    report = run_walk_forward(
        "trend_following", "EURUSD", "H1", raw, space, n_trials=5, window_bars=600, step_bars=300, db=db
    )

    assert len(report.windows) == 4
    assert [w.index for w in report.windows] == [0, 1, 2, 3]
    assert 0.0 <= report.oos_pass_rate <= 1.0
    assert len(list_windows(db=db, run_id=report.run_id)) == 4


def test_run_walk_forward_isolates_runs_by_run_id(db):
    raw = _synthetic_raw_candles(n=1500)
    space = _trend_following_space()

    first = run_walk_forward("trend_following", "EURUSD", "H1", raw, space, n_trials=5, db=db)
    second = run_walk_forward("trend_following", "EURUSD", "H1", raw, space, n_trials=5, db=db)

    assert first.run_id != second.run_id
    assert len(list_windows(db=db)) == 2
    assert len(list_windows(db=db, run_id=first.run_id)) == 1
    assert len(list_windows(db=db, run_id=second.run_id)) == 1


# --- WalkForwardReport helpers (constructed directly -- no expensive run) -


def _make_window(index, params, oos_passed, oos_metrics=None):
    ts = pd.Timestamp("2024-01-01", tz="UTC")
    return WalkForwardWindowResult(
        index=index,
        train_range=(ts, ts),
        validation_range=(ts, ts),
        oos_range=(ts, ts),
        best_params=params,
        train_objective=1.0,
        stability_score=0.9,
        validation_metrics={},
        oos_metrics=oos_metrics or {"trade_count": 5, "total_return": 0.01},
        oos_objective=1.0 if oos_passed else -1.0,
        oos_passed=oos_passed,
    )


def test_oos_pass_rate_computes_fraction_correctly():
    report = WalkForwardReport(
        run_id="r",
        strategy="s",
        symbol="EURUSD",
        timeframe="H1",
        windows=[
            _make_window(0, {"a": 1}, True),
            _make_window(1, {"a": 1}, True),
            _make_window(2, {"a": 1}, False),
            _make_window(3, {"a": 1}, False),
        ],
    )
    assert report.oos_pass_rate == pytest.approx(0.5)


def test_oos_pass_rate_zero_windows_is_zero_not_a_crash():
    report = WalkForwardReport(run_id="r", strategy="s", symbol="EURUSD", timeframe="H1", windows=[])
    assert report.oos_pass_rate == 0.0


def test_parameter_consistency_empty_for_single_window():
    report = WalkForwardReport(
        run_id="r", strategy="s", symbol="EURUSD", timeframe="H1", windows=[_make_window(0, {"a": 1.0}, True)]
    )
    assert report.parameter_consistency() == {}


def test_parameter_consistency_low_cv_for_identical_params():
    report = WalkForwardReport(
        run_id="r",
        strategy="s",
        symbol="EURUSD",
        timeframe="H1",
        windows=[_make_window(i, {"a": 2.0}, True) for i in range(3)],
    )
    assert report.parameter_consistency()["a"] == pytest.approx(0.0)


def test_parameter_consistency_high_cv_for_scattered_params():
    report = WalkForwardReport(
        run_id="r",
        strategy="s",
        symbol="EURUSD",
        timeframe="H1",
        windows=[_make_window(0, {"a": 1.0}, True), _make_window(1, {"a": 100.0}, True)],
    )
    assert report.parameter_consistency()["a"] > 0.5


def test_to_text_includes_header_and_window_lines():
    report = WalkForwardReport(
        run_id="r",
        strategy="trend_following",
        symbol="EURUSD",
        timeframe="H1",
        windows=[_make_window(0, {"atr_stop_multiplier": 2.0}, True)],
    )
    text = report.to_text()
    assert text.startswith("WALK-FORWARD ANALYSIS REPORT")
    assert "trend_following" in text
    assert "PASS" in text
