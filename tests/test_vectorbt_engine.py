import numpy as np
import pandas as pd
import pytest

from src.backtest.experiment_store import list_experiments
from src.backtest.vectorbt_engine import filter_top_candidates, run_screening, screen_parameter_grid
from src.core.db import Database
from src.data.schema import STANDARD_COLUMNS


@pytest.fixture()
def db():
    database = Database(url="sqlite:///:memory:")
    database.init_db()
    return database


def _synthetic_raw_candles(n=400, seed=1, symbol="EURUSD", timeframe="H1"):
    """Three phases (up/range/down) so trend/breakout/range conditions all
    plausibly occur, matching the pattern already validated for Phase 5's
    integration tests."""
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


def test_warmup_df_gives_indicators_proper_history_without_scoring_it(db):
    """A 60-bar scored window alone is far too short for volatility_lookback
    (100) to ever produce a non-NaN volatility_regime — every bar would be
    stuck in "warm-up", making the window artificially untradeable. With
    400 bars of real preceding history as warmup_df, the same window's
    indicators have proper context, while only the 60 scored bars remain
    the actual date range persisted/evaluated (warmup contributes no
    trade and no metric)."""
    from src.features.engine import compute_features

    full = _synthetic_raw_candles(n=460)
    warmup = full.iloc[:400].reset_index(drop=True)
    scored = full.iloc[400:].reset_index(drop=True)

    # Premise: scored alone can never leave warm-up (proves the test is
    # meaningful, not just "it ran").
    features_alone = compute_features(scored)
    assert features_alone["volatility_regime"].isna().all()

    result = run_screening("trend_following", "EURUSD", "H1", scored, warmup_df=warmup, db=db)

    records = list_experiments(db=db)
    assert len(records) == 1
    # SQLite's DateTime column round-trips as naive, so compare naive-to-naive.
    assert records[0].date_range_start == scored["timestamp"].iloc[0].tz_localize(None).to_pydatetime()
    assert records[0].date_range_end == scored["timestamp"].iloc[-1].tz_localize(None).to_pydatetime()
    # never the warmup range
    assert records[0].date_range_start != warmup["timestamp"].iloc[0].tz_localize(None).to_pydatetime()


def test_run_screening_returns_sane_metrics_and_persists(db):
    raw = _synthetic_raw_candles()
    result = run_screening("trend_following", "EURUSD", "H1", raw, db=db)

    assert result.strategy == "trend_following"
    assert result.symbol == "EURUSD"
    assert result.scenario == "realistic"
    assert "sharpe_ratio" in result.metrics
    assert "trade_count" in result.metrics
    assert result.metrics["trade_count"] >= 0

    records = list_experiments(db=db)
    assert len(records) == 1
    assert records[0].experiment_id == result.experiment_id
    assert records[0].strategy == "trend_following"
    assert records[0].data_version  # non-empty, even if "unknown" (no processed file for this symbol)


def test_run_screening_persist_false_does_not_write_to_db(db):
    raw = _synthetic_raw_candles()
    run_screening("mean_reversion", "EURUSD", "H1", raw, db=db, persist=False)
    assert list_experiments(db=db) == []


def test_run_screening_rejects_unknown_scenario(db):
    raw = _synthetic_raw_candles()
    with pytest.raises(ValueError, match="Unknown execution scenario"):
        run_screening("trend_following", "EURUSD", "H1", raw, scenario="not_a_scenario", db=db)


def test_optimistic_scenario_has_zero_costs_realistic_has_nonzero(db):
    raw = _synthetic_raw_candles()
    optimistic = run_screening("trend_following", "EURUSD", "H1", raw, scenario="optimistic", db=db)
    realistic = run_screening("trend_following", "EURUSD", "H1", raw, scenario="realistic", db=db)
    assert optimistic.scenario == "optimistic"
    assert realistic.scenario == "realistic"
    # Same signals, different costs -> optimistic should never do worse
    assert optimistic.metrics["total_return"] >= realistic.metrics["total_return"] - 1e-9


def test_screen_parameter_grid_runs_every_combination(db):
    raw = _synthetic_raw_candles()
    grid = {"atr_stop_multiplier": [1.5, 2.5], "take_profit_r_multiple": [1.5, 2.0]}
    results = screen_parameter_grid("trend_following", "EURUSD", "H1", raw, grid, db=db)

    assert len(results) == 4
    seen_params = {(r.parameters["atr_stop_multiplier"], r.parameters["take_profit_r_multiple"]) for r in results}
    assert seen_params == {(1.5, 1.5), (1.5, 2.0), (2.5, 1.5), (2.5, 2.0)}
    assert len(list_experiments(db=db)) == 4


def test_screen_parameter_grid_with_no_grid_runs_default_params_once(db):
    raw = _synthetic_raw_candles()
    results = screen_parameter_grid("mean_reversion", "EURUSD", "H1", raw, {}, db=db)
    assert len(results) == 1


def test_filter_top_candidates_excludes_low_trade_count_and_ranks_by_sharpe():
    from src.backtest.vectorbt_engine import ScreeningResult

    results = [
        ScreeningResult("a", "s", "EURUSD", "H1", "realistic", {}, {"sharpe_ratio": 0.5, "trade_count": 20}),
        ScreeningResult("b", "s", "EURUSD", "H1", "realistic", {}, {"sharpe_ratio": 2.0, "trade_count": 3}),
        ScreeningResult("c", "s", "EURUSD", "H1", "realistic", {}, {"sharpe_ratio": 1.5, "trade_count": 15}),
    ]
    top = filter_top_candidates(results, top_n=5, min_trades=10)
    assert [r.experiment_id for r in top] == ["c", "a"]  # "b" excluded (too few trades)


def test_filter_top_candidates_respects_top_n():
    from src.backtest.vectorbt_engine import ScreeningResult

    results = [
        ScreeningResult(str(i), "s", "EURUSD", "H1", "realistic", {}, {"sharpe_ratio": float(i), "trade_count": 20})
        for i in range(10)
    ]
    top = filter_top_candidates(results, top_n=3, min_trades=1)
    assert len(top) == 3
    assert [r.experiment_id for r in top] == ["9", "8", "7"]
