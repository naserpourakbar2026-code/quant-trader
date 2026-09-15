import numpy as np
import pandas as pd
import pytest

from src.backtest.vectorbt_engine import get_trade_returns, run_screening
from src.core.db import Database
from src.data.schema import STANDARD_COLUMNS
from src.montecarlo.mc_engine import MonteCarloResult, _max_consecutive_true, run_monte_carlo, simulate
from src.montecarlo.run_store import list_runs


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


# --- get_trade_returns (vectorbt_engine) -----------------------------------


def test_get_trade_returns_length_matches_run_screening_trade_count():
    raw = _synthetic_raw_candles()
    screening = run_screening("trend_following", "EURUSD", "H1", raw, persist=False)
    returns = get_trade_returns("trend_following", raw)
    assert len(returns) == screening.metrics["trade_count"]


# --- simulate() (pure numeric core, no I/O) ---------------------------------


def test_simulate_is_reproducible_with_same_seed():
    trade_returns = np.array([0.01, -0.02, 0.015, -0.01, 0.03, -0.005])
    a = simulate(trade_returns, n_simulations=200, initial_capital=2000, ruin_threshold=0.5,
                 execution_noise_std=0.0002, seed=7)
    b = simulate(trade_returns, n_simulations=200, initial_capital=2000, ruin_threshold=0.5,
                 execution_noise_std=0.0002, seed=7)
    np.testing.assert_array_equal(a["total_return"], b["total_return"])
    np.testing.assert_array_equal(a["max_drawdown"], b["max_drawdown"])


def test_simulate_differs_with_different_seed():
    trade_returns = np.array([0.01, -0.02, 0.015, -0.01, 0.03, -0.005])
    a = simulate(trade_returns, n_simulations=200, initial_capital=2000, ruin_threshold=0.5,
                 execution_noise_std=0.0002, seed=1)
    b = simulate(trade_returns, n_simulations=200, initial_capital=2000, ruin_threshold=0.5,
                 execution_noise_std=0.0002, seed=2)
    assert not np.array_equal(a["total_return"], b["total_return"])


def test_simulate_flags_ruin_when_a_single_trade_wipes_out_half_of_equity():
    """Every resampled trade is a -60% loss -> the very first trade in
    every simulated sequence already breaches a 50% ruin threshold."""
    trade_returns = np.array([-0.6, -0.6, -0.6, -0.6])
    result = simulate(trade_returns, n_simulations=100, initial_capital=2000, ruin_threshold=0.5,
                       execution_noise_std=0.0, seed=1)
    assert result["ruin"].all()
    assert (result["total_return"] < 0).all()


def test_simulate_no_ruin_when_all_trades_are_small_wins():
    trade_returns = np.array([0.01, 0.02, 0.015, 0.01])
    result = simulate(trade_returns, n_simulations=100, initial_capital=2000, ruin_threshold=0.5,
                       execution_noise_std=0.0, seed=1)
    assert not result["ruin"].any()
    assert (result["total_return"] > 0).all()


def test_simulate_losing_streak_matches_all_losing_trades():
    trade_returns = np.array([-0.01, -0.01, -0.01])
    result = simulate(trade_returns, n_simulations=50, initial_capital=2000, ruin_threshold=0.1,
                       execution_noise_std=0.0, seed=1)
    assert (result["losing_streak"] == 3).all()


# --- _max_consecutive_true ---------------------------------------------------


def test_max_consecutive_true_handles_mixed_rows():
    mask = np.array(
        [
            [True, True, False, True, True, True],
            [False, False, False, False, False, False],
            [True, True, True, True, True, True],
        ]
    )
    result = _max_consecutive_true(mask)
    np.testing.assert_array_equal(result, [3, 0, 6])


def test_max_consecutive_true_handles_zero_columns():
    mask = np.zeros((3, 0), dtype=bool)
    result = _max_consecutive_true(mask)
    np.testing.assert_array_equal(result, [0, 0, 0])


# --- run_monte_carlo end-to-end ----------------------------------------------


def test_run_monte_carlo_end_to_end(db):
    raw = _synthetic_raw_candles()
    result = run_monte_carlo("trend_following", "EURUSD", "H1", raw, n_simulations=300, db=db)

    assert result.n_trades_observed > 0
    assert result.n_simulations == 300
    assert 0.0 <= result.probability_of_ruin <= 1.0
    assert 0.0 <= result.probability_of_negative_return <= 1.0
    assert result.p5_return <= result.median_return <= result.p95_return
    assert result.worst_drawdown >= result.p95_drawdown >= 0.0
    assert isinstance(result.is_fragile, bool)

    records = list_runs(db=db)
    assert len(records) == 1
    assert records[0].run_id == result.run_id
    assert records[0].n_trades_observed == result.n_trades_observed

    assert result.return_histogram["counts"]
    assert sum(result.return_histogram["counts"]) == result.n_simulations
    assert len(result.return_histogram["bin_edges"]) == len(result.return_histogram["counts"]) + 1
    assert records[0].return_histogram == result.return_histogram


def test_run_monte_carlo_persist_false_does_not_write_to_db(db):
    raw = _synthetic_raw_candles()
    run_monte_carlo("trend_following", "EURUSD", "H1", raw, n_simulations=100, db=db, persist=False)
    assert list_runs(db=db) == []


def test_run_monte_carlo_is_reproducible_with_explicit_seed(db):
    raw = _synthetic_raw_candles()
    first = run_monte_carlo("trend_following", "EURUSD", "H1", raw, n_simulations=200, seed=99, db=db, persist=False)
    second = run_monte_carlo("trend_following", "EURUSD", "H1", raw, n_simulations=200, seed=99, db=db, persist=False)
    assert first.median_return == second.median_return
    assert first.probability_of_ruin == second.probability_of_ruin


def test_run_monte_carlo_raises_value_error_on_zero_trades(db):
    """A too-short series never produces a real trade -- Monte Carlo has
    nothing to resample and must say so rather than fabricate a result."""
    raw = _synthetic_raw_candles(n=5)
    with pytest.raises(ValueError, match="zero trades"):
        run_monte_carlo("trend_following", "EURUSD", "H1", raw, db=db)


def test_run_monte_carlo_flags_insufficient_data_below_min_trades(db):
    """A short-but-not-empty series can produce a handful of trades --
    fewer than settings.montecarlo.min_trades. The run still completes
    (never fabricated), just flagged as statistically unreliable."""
    raw = _synthetic_raw_candles(n=120)
    result = run_monte_carlo("trend_following", "EURUSD", "H1", raw, n_simulations=50, db=db, persist=False)
    from src.core.config import load_settings

    if result.n_trades_observed < load_settings().montecarlo.min_trades:
        assert result.insufficient_data
        assert "WARNING" in result.to_text()
    else:
        assert not result.insufficient_data


def test_to_text_includes_header_and_verdict():
    result = MonteCarloResult(
        run_id="r",
        strategy="trend_following",
        symbol="EURUSD",
        timeframe="H1",
        scenario="realistic",
        n_simulations=1000,
        n_trades_observed=50,
        initial_capital=2000.0,
        ruin_threshold=0.5,
        median_return=0.1,
        p5_return=-0.05,
        p95_return=0.3,
        worst_drawdown=0.2,
        p95_drawdown=0.15,
        median_losing_streak=2.0,
        p95_losing_streak=4.0,
        worst_losing_streak=6,
        probability_of_ruin=0.01,
        probability_of_negative_return=0.1,
        is_fragile=False,
    )
    text = result.to_text()
    assert text.startswith("MONTE CARLO ANALYSIS REPORT")
    assert "not flagged as fragile" in text


def test_to_text_flags_fragile_and_insufficient_data():
    result = MonteCarloResult(
        run_id="r",
        strategy="trend_following",
        symbol="EURUSD",
        timeframe="H1",
        scenario="realistic",
        n_simulations=1000,
        n_trades_observed=3,
        initial_capital=2000.0,
        ruin_threshold=0.5,
        median_return=-0.4,
        p5_return=-0.9,
        p95_return=0.1,
        worst_drawdown=0.9,
        p95_drawdown=0.8,
        median_losing_streak=3.0,
        p95_losing_streak=5.0,
        worst_losing_streak=6,
        probability_of_ruin=0.4,
        probability_of_negative_return=0.7,
        is_fragile=True,
        insufficient_data=True,
    )
    text = result.to_text()
    assert "FRAGILE UNDER MONTE CARLO" in text
    assert "WARNING" in text
