import numpy as np
import pandas as pd
import pytest

from src.backtest.vectorbt_engine import get_bar_returns, run_screening
from src.core.db import Database
from src.data.schema import STANDARD_COLUMNS
from src.portfolio.portfolio_engine import (
    PortfolioComponent,
    align_returns,
    allocate_equal_weight,
    allocate_inverse_volatility,
    combine,
    compare_to_best_component,
    correlation_matrix,
    evaluate_returns,
    run_portfolio_analysis,
)
from src.portfolio.run_store import list_runs


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


def _component(label_parts, returns_values, index=None, metrics=None):
    strategy, symbol, timeframe = label_parts
    if index is None:
        index = pd.date_range("2024-01-01", periods=len(returns_values), freq="1h", tz="UTC")
    return PortfolioComponent(
        strategy=strategy,
        symbol=symbol,
        timeframe=timeframe,
        scenario="realistic",
        metrics=metrics or {},
        returns=pd.Series(returns_values, index=index),
    )


# --- get_bar_returns (vectorbt_engine) --------------------------------------


def test_get_bar_returns_is_a_series_indexed_by_timestamp_matching_screening():
    raw = _synthetic_raw_candles()
    screening = run_screening("trend_following", "EURUSD", "H1", raw, persist=False)
    returns = get_bar_returns("trend_following", raw)
    assert isinstance(returns, pd.Series)
    assert isinstance(returns.index, pd.DatetimeIndex)
    # same scored range as the screening run (both built off the same raw_df, no warmup)
    assert len(returns) == len(raw)
    assert returns.index[0] == pd.Timestamp(raw["timestamp"].iloc[0])


# --- evaluate_returns (pure numeric core) -----------------------------------


def test_evaluate_returns_on_all_positive_returns():
    idx = pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC")
    returns = pd.Series(0.001, index=idx)
    metrics = evaluate_returns(returns)
    assert metrics["total_return"] == pytest.approx((1.001) ** 100 - 1)
    assert metrics["sharpe_ratio"] > 0
    assert metrics["sortino_ratio"] is None  # no downside returns at all -> undefined
    assert metrics["max_drawdown"] == pytest.approx(0.0)  # monotonically rising equity, no drawdown
    assert metrics["positive_month_ratio"] == pytest.approx(1.0)


def test_evaluate_returns_max_drawdown_is_negative_matching_vectorbt_convention():
    idx = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    returns = pd.Series([0.10, -0.20, 0.0, 0.0, 0.0], index=idx)  # up 10%, down 20% from the peak
    metrics = evaluate_returns(returns)
    assert metrics["max_drawdown"] < 0
    # peak = 1.10, trough = 1.10*0.8 = 0.88 -> drawdown = (1.10-0.88)/1.10
    assert metrics["max_drawdown"] == pytest.approx(-((1.10 - 0.88) / 1.10))


def test_evaluate_returns_empty_series_returns_all_none():
    metrics = evaluate_returns(pd.Series([], dtype=float))
    assert all(v is None for v in metrics.values())


def test_evaluate_returns_constant_zero_returns_has_no_sharpe():
    idx = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")
    metrics = evaluate_returns(pd.Series(0.0, index=idx))
    assert metrics["sharpe_ratio"] is None  # zero std -> undefined, not inf
    assert metrics["sortino_ratio"] is None
    assert metrics["total_return"] == pytest.approx(0.0)


# --- align_returns / correlation_matrix -------------------------------------


def test_align_returns_outer_joins_and_fills_missing_with_zero():
    idx_a = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
    idx_b = pd.date_range("2024-01-01 01:00", periods=3, freq="1h", tz="UTC")
    a = _component(("s1", "EURUSD", "H1"), [0.01, 0.02, 0.03], index=idx_a)
    b = _component(("s2", "GBPUSD", "H1"), [0.05, 0.06, 0.07], index=idx_b)

    frame = align_returns([a, b])
    assert set(frame.columns) == {a.label, b.label}
    assert len(frame) == 4  # union of 3+3 with 2 overlapping timestamps
    assert frame.loc[idx_a[0], b.label] == 0.0  # b has no data at a's first timestamp


def test_correlation_matrix_of_identical_series_is_one():
    idx = pd.date_range("2024-01-01", periods=20, freq="1h", tz="UTC")
    values = np.linspace(-0.01, 0.01, 20)
    a = _component(("s1", "EURUSD", "H1"), values, index=idx)
    b = _component(("s2", "EURUSD", "H1"), values, index=idx)
    corr = correlation_matrix(align_returns([a, b]))
    assert corr.loc[a.label, b.label] == pytest.approx(1.0)


# --- allocation schemes -------------------------------------------------------


def test_allocate_equal_weight_sums_to_one_and_is_uniform():
    idx = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")
    components = [_component((f"s{i}", "EURUSD", "H1"), np.random.default_rng(i).normal(0, 0.01, 10), index=idx) for i in range(4)]
    weights = allocate_equal_weight(components)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert len(set(weights.values())) == 1


def test_allocate_inverse_volatility_favors_lower_volatility_component():
    idx = pd.date_range("2024-01-01", periods=200, freq="1h", tz="UTC")
    low_vol = _component(("s_low", "EURUSD", "H1"), np.random.default_rng(1).normal(0, 0.001, 200), index=idx)
    high_vol = _component(("s_high", "GBPUSD", "H1"), np.random.default_rng(1).normal(0, 0.05, 200), index=idx)
    weights = allocate_inverse_volatility([low_vol, high_vol])
    assert weights[low_vol.label] > weights[high_vol.label]
    assert sum(weights.values()) == pytest.approx(1.0)


def test_allocate_inverse_volatility_falls_back_to_equal_weight_when_a_component_has_zero_volatility():
    idx = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")
    flat = _component(("s_flat", "EURUSD", "H1"), np.zeros(10), index=idx)
    normal = _component(("s_normal", "GBPUSD", "H1"), np.random.default_rng(1).normal(0, 0.01, 10), index=idx)
    weights = allocate_inverse_volatility([flat, normal])
    assert weights[flat.label] == pytest.approx(0.5)
    assert weights[normal.label] == pytest.approx(0.5)


def test_combine_matches_manual_weighted_sum():
    idx = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
    a = _component(("s1", "EURUSD", "H1"), [0.01, 0.02, 0.03], index=idx)
    b = _component(("s2", "GBPUSD", "H1"), [0.10, 0.20, 0.30], index=idx)
    frame = align_returns([a, b])
    combined = combine(frame, {a.label: 0.25, b.label: 0.75})
    expected = pd.Series([0.01 * 0.25 + 0.10 * 0.75, 0.02 * 0.25 + 0.20 * 0.75, 0.03 * 0.25 + 0.30 * 0.75], index=idx)
    pd.testing.assert_series_equal(combined, expected, check_names=False)


# --- compare_to_best_component -----------------------------------------------


def test_compare_to_best_component_drawdown_uses_vectorbt_sign_convention():
    """max_drawdown is <= 0 in both inputs (vectorbt's own convention) --
    a portfolio drawdown of -0.01 (smaller magnitude) must count as
    IMPROVED over a best-component drawdown of -0.05 (larger magnitude)."""
    portfolio_metrics = {"sharpe_ratio": 1.0, "sortino_ratio": 1.0, "max_drawdown": -0.01, "positive_month_ratio": 0.6}
    best_metrics = {"sharpe_ratio": 2.0, "sortino_ratio": 2.0, "max_drawdown": -0.05, "positive_month_ratio": 0.6}
    verdict = compare_to_best_component(portfolio_metrics, best_metrics)
    assert verdict["drawdown_improved"] is True
    assert verdict["sharpe_improved"] is False
    assert verdict["consistency_improved"] is False


def test_compare_to_best_component_handles_none_gracefully():
    verdict = compare_to_best_component({"sharpe_ratio": None}, {"sharpe_ratio": 1.0})
    assert verdict["sharpe_improved"] is None


# --- run_portfolio_analysis end-to-end ---------------------------------------


def test_run_portfolio_analysis_end_to_end(db):
    components_data = [
        ("trend_following", "EURUSD", "H1", _synthetic_raw_candles(seed=1, symbol="EURUSD")),
        ("mean_reversion", "EURUSD", "H1", _synthetic_raw_candles(seed=1, symbol="EURUSD")),
        ("trend_following", "GBPUSD", "H1", _synthetic_raw_candles(seed=2, symbol="GBPUSD")),
    ]
    result = run_portfolio_analysis(components_data, db=db, top_n=10, min_trades=5)

    assert len(result.components) >= 1
    assert result.best_component_label
    if len(result.components) > 1:
        assert not result.correlation.empty
        assert set(result.allocations) == {"equal_weight", "inverse_volatility"}
        for outcome in result.allocations.values():
            assert sum(outcome["weights"].values()) == pytest.approx(1.0)

    records = list_runs(db=db)
    assert len(records) == 1
    assert records[0].run_id == result.run_id
    assert records[0].n_components == len(result.components)


def test_run_portfolio_analysis_persist_false_does_not_write_to_db(db):
    components_data = [("trend_following", "EURUSD", "H1", _synthetic_raw_candles())]
    run_portfolio_analysis(components_data, db=db, min_trades=5, persist=False)
    assert list_runs(db=db) == []


def test_run_portfolio_analysis_single_component_has_no_allocations(db):
    components_data = [("trend_following", "EURUSD", "H1", _synthetic_raw_candles())]
    result = run_portfolio_analysis(components_data, db=db, min_trades=5)
    assert len(result.components) == 1
    assert result.allocations == {}


def test_run_portfolio_analysis_raises_when_nothing_meets_min_trades(db):
    components_data = [("trend_following", "EURUSD", "H1", _synthetic_raw_candles())]
    with pytest.raises(ValueError, match="nothing statistically usable"):
        run_portfolio_analysis(components_data, db=db, min_trades=10_000)


def test_run_portfolio_analysis_requires_at_least_one_combination():
    with pytest.raises(ValueError, match="must not be empty"):
        run_portfolio_analysis([])


def test_run_portfolio_analysis_only_combines_within_the_largest_timeframe_group(db):
    """Components on different timeframes are never combined (a bar of
    H1 and a bar of H4 aren't the same unit of time) -- only the largest
    same-timeframe group enters the allocation step."""
    h1_a = _synthetic_raw_candles(seed=1, symbol="EURUSD")
    h1_b = _synthetic_raw_candles(seed=2, symbol="GBPUSD")
    h4_only = _synthetic_raw_candles(seed=3, symbol="USDJPY", timeframe="H4")
    components_data = [
        ("trend_following", "EURUSD", "H1", h1_a),
        ("trend_following", "GBPUSD", "H1", h1_b),
        ("trend_following", "USDJPY", "H4", h4_only),
    ]
    result = run_portfolio_analysis(components_data, db=db, min_trades=5)
    combined_labels = {label for outcome in result.allocations.values() for label in outcome["weights"]}
    assert all(label.endswith("|H1") for label in combined_labels)


def test_to_text_includes_header_and_correlation(db):
    components_data = [
        ("trend_following", "EURUSD", "H1", _synthetic_raw_candles(seed=1, symbol="EURUSD")),
        ("trend_following", "GBPUSD", "H1", _synthetic_raw_candles(seed=2, symbol="GBPUSD")),
    ]
    result = run_portfolio_analysis(components_data, db=db, min_trades=5)
    text = result.to_text()
    assert text.startswith("PORTFOLIO ANALYSIS REPORT")
    assert "Best single component" in text
