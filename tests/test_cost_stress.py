import numpy as np
import pandas as pd
import pytest

from src.data.schema import STANDARD_COLUMNS
from src.robustness.cost_stress import CostStressPoint, CostStressResult, run_cost_stress_test


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


def test_run_cost_stress_test_produces_one_point_per_multiplier_and_scenario():
    raw = _synthetic_raw_candles()
    result = run_cost_stress_test(
        "trend_following", "EURUSD", "H1", raw, scenarios=["optimistic", "realistic"], spread_multipliers=[1.0, 2.0],
    )
    assert len(result.points) == 4
    keys = {(p.spread_multiplier, p.scenario) for p in result.points}
    assert keys == {(1.0, "optimistic"), (1.0, "realistic"), (2.0, "optimistic"), (2.0, "realistic")}


def test_higher_spread_multiplier_never_improves_return_when_a_real_spread_column_exists():
    """With a real (non-NaN) spread column, widening it can only ever
    add cost, never subtract it -- return at 3x must be <= return at 1x
    for the same scenario."""
    raw = _synthetic_raw_candles()
    result = run_cost_stress_test("trend_following", "EURUSD", "H1", raw, scenarios=["realistic"])
    by_multiplier = {p.spread_multiplier: p.metrics["total_return"] for p in result.points}
    assert by_multiplier[3.0] <= by_multiplier[1.0]


def test_missing_spread_column_makes_the_multiplier_a_no_op_not_fabricated():
    """A CSV with no real spread column leaves it NaN throughout
    (src.data.schema) -- NaN * any multiplier is still NaN, so every
    multiplier must produce an identical result. This is the honest
    behavior: the test can't stress a cost that was never measured, and
    must not invent one."""
    raw = _synthetic_raw_candles()
    raw["spread"] = float("nan")
    result = run_cost_stress_test("trend_following", "EURUSD", "H1", raw, scenarios=["realistic"])
    returns = {p.spread_multiplier: p.metrics["total_return"] for p in result.points}
    assert len(set(returns.values())) == 1  # every multiplier gave the identical result


def _point(multiplier, scenario, profit_factor, total_return=0.1):
    return CostStressPoint(
        spread_multiplier=multiplier, scenario=scenario,
        metrics={"profit_factor": profit_factor, "total_return": total_return, "trade_count": 20},
    )


def test_is_cost_fragile_true_when_a_small_increase_kills_profitability():
    result = CostStressResult(
        run_id="r", strategy="s", symbol="EURUSD", timeframe="H1", baseline_scenario="realistic",
        points=[_point(1.0, "realistic", 2.0), _point(1.5, "realistic", 0.9), _point(2.0, "realistic", 0.5)],
    )
    assert result.is_cost_fragile is True


def test_is_cost_fragile_false_when_profitability_survives_a_small_increase():
    result = CostStressResult(
        run_id="r", strategy="s", symbol="EURUSD", timeframe="H1", baseline_scenario="realistic",
        points=[_point(1.0, "realistic", 3.0), _point(1.5, "realistic", 2.5), _point(2.0, "realistic", 0.8)],
    )
    assert result.is_cost_fragile is False  # only collapses at 2x, not the smallest (1.5x) increase


def test_is_cost_fragile_false_when_never_profitable_to_begin_with():
    result = CostStressResult(
        run_id="r", strategy="s", symbol="EURUSD", timeframe="H1", baseline_scenario="realistic",
        points=[_point(1.0, "realistic", 0.8), _point(1.5, "realistic", 0.5)],
    )
    assert result.is_cost_fragile is False


def test_is_cost_fragile_treats_none_profit_factor_as_excellent_not_zero():
    """vectorbt reports profit_factor=None for a run with zero losing
    trades -- that must never be read as "unprofitable"."""
    result = CostStressResult(
        run_id="r", strategy="s", symbol="EURUSD", timeframe="H1", baseline_scenario="realistic",
        points=[_point(1.0, "realistic", None), _point(1.5, "realistic", None)],
    )
    assert result.is_cost_fragile is False


def test_to_text_includes_header_and_verdict():
    result = CostStressResult(
        run_id="r", strategy="trend_following", symbol="EURUSD", timeframe="H1", baseline_scenario="realistic",
        points=[_point(1.0, "realistic", 2.0)],
    )
    text = result.to_text()
    assert text.startswith("TRANSACTION COST STRESS TEST")
    assert "not cost fragile" in text
