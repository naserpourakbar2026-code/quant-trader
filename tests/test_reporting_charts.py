from datetime import datetime, timezone

import pandas as pd
import pytest

from src.backtest.experiment_store import ExperimentRecord
from src.execution.trade_store import PaperTradeRecord
from src.montecarlo.run_store import MonteCarloRecord
from src.portfolio.run_store import PortfolioRecord
from src.reporting import charts
from src.walkforward.window_store import WindowRecord


def _trade(exit_time, pnl, r_multiple=None):
    return PaperTradeRecord(
        trade_id="t", run_id="r", strategy="trend_following", symbol="EURUSD", timeframe="H1",
        direction="LONG", entry_time=exit_time, entry_price=1.10, stop_loss=1.09, take_profit=1.12,
        size=1000.0, risk_amount=10.0, spread=0.0001, slippage_pct=0.0001,
        exit_time=exit_time, exit_price=1.11, pnl=pnl, r_multiple=r_multiple if r_multiple is not None else pnl / 10.0,
        reason="take_profit",
    )


def _trades_across_months(n=40, start="2024-01-01"):
    base = pd.Timestamp(start, tz="UTC")
    return [_trade(base + pd.Timedelta(days=3 * i), pnl=(5.0 if i % 3 else -3.0)) for i in range(n)]


# --- equity/drawdown/monthly/rolling-sharpe/distribution ---------------------


def test_build_equity_series_with_no_trades_is_a_single_point():
    series = charts.build_equity_series([], initial_capital=2000.0)
    assert len(series) == 1
    assert series.iloc[0] == 2000.0


def test_build_equity_series_orders_by_exit_time_and_compounds_pnl():
    t1 = _trade(pd.Timestamp("2024-01-02", tz="UTC"), pnl=10.0)
    t2 = _trade(pd.Timestamp("2024-01-01", tz="UTC"), pnl=5.0)  # out of order on purpose
    series = charts.build_equity_series([t1, t2], initial_capital=1000.0)
    assert list(series.values) == [1005.0, 1015.0]  # t2 (earlier) applied first


def test_equity_curve_chart_none_with_fewer_than_two_points():
    series = charts.build_equity_series([], initial_capital=2000.0)
    assert charts.equity_curve_chart(series) is None


def test_equity_curve_chart_returns_a_data_uri_with_enough_trades():
    trades = _trades_across_months(10)
    series = charts.build_equity_series(trades, 2000.0)
    result = charts.equity_curve_chart(series)
    assert result is not None
    assert result.startswith("data:image/png;base64,")


def test_drawdown_chart_present_with_enough_data():
    trades = _trades_across_months(10)
    series = charts.build_equity_series(trades, 2000.0)
    assert charts.drawdown_chart(series) is not None


def test_monthly_returns_chart_none_when_all_trades_in_one_month():
    trades = [_trade(pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=i), pnl=5.0) for i in range(5)]
    series = charts.build_equity_series(trades, 2000.0)
    assert charts.monthly_returns_chart(series) is None


def test_monthly_returns_chart_present_across_multiple_months():
    trades = _trades_across_months(40)  # ~3.6 months of 3-day spacing
    series = charts.build_equity_series(trades, 2000.0)
    assert charts.monthly_returns_chart(series) is not None


def test_rolling_sharpe_chart_none_below_window():
    trades = _trades_across_months(10)
    assert charts.rolling_sharpe_chart(trades, window=20) is None


def test_rolling_sharpe_chart_present_at_or_above_window():
    trades = _trades_across_months(25)
    assert charts.rolling_sharpe_chart(trades, window=20) is not None


def test_trade_distribution_chart_none_with_fewer_than_two_r_multiples():
    assert charts.trade_distribution_chart([_trade(pd.Timestamp("2024-01-01", tz="UTC"), pnl=5.0)]) is None


def test_trade_distribution_chart_present_with_enough_trades():
    trades = _trades_across_months(10)
    assert charts.trade_distribution_chart(trades) is not None


# --- parameter heatmap --------------------------------------------------------


def _experiment(strategy, symbol, timeframe, params, sharpe, engine="vectorbt"):
    return ExperimentRecord(
        experiment_id="e", engine=engine, strategy=strategy, symbol=symbol, timeframe=timeframe, scenario="realistic",
        parameters=params, date_range_start=datetime(2024, 1, 1), date_range_end=datetime(2024, 2, 1),
        metrics={"sharpe_ratio": sharpe, "total_return": 0.1, "trade_count": 20}, data_version="v1",
        code_version="abc", python_version="3.11", library_versions={},
    )


def test_parameter_heatmap_chart_none_with_too_few_runs():
    experiments = [_experiment("trend_following", "EURUSD", "H1", {"a": 1, "b": 1}, 1.0)]
    assert charts.parameter_heatmap_chart(experiments) is None


def test_parameter_heatmap_chart_none_when_only_one_parameter_varies():
    experiments = [
        _experiment("trend_following", "EURUSD", "H1", {"a": a, "b": 1}, float(a)) for a in range(5)
    ]
    assert charts.parameter_heatmap_chart(experiments) is None


def test_parameter_heatmap_chart_present_with_a_two_parameter_grid():
    experiments = [
        _experiment("trend_following", "EURUSD", "H1", {"a": a, "b": b}, float(a + b))
        for a in range(3) for b in range(3)
    ]
    result = charts.parameter_heatmap_chart(experiments)
    assert result is not None
    data_uri, description = result
    assert data_uri.startswith("data:image/png;base64,")
    assert "a x b" in description or "a" in description


def test_parameter_heatmap_chart_ignores_non_vectorbt_engines():
    experiments = [
        _experiment("trend_following", "EURUSD", "H1", {"a": a, "b": b}, float(a + b), engine="optuna")
        for a in range(3) for b in range(3)
    ]
    assert charts.parameter_heatmap_chart(experiments) is None


def test_parameter_heatmap_chart_picks_the_largest_group():
    small_group = [_experiment("mean_reversion", "EURUSD", "H1", {"a": a, "b": 1}, float(a)) for a in range(4)]
    large_group = [
        _experiment("trend_following", "EURUSD", "H1", {"a": a, "b": b}, float(a + b))
        for a in range(3) for b in range(3)
    ]
    result = charts.parameter_heatmap_chart(small_group + large_group)
    assert result is not None
    _, description = result
    assert "9 runs" in description


# --- walk-forward --------------------------------------------------------------


def _window(run_id, index, oos_passed, oos_objective, strategy="trend_following"):
    ts = pd.Timestamp("2024-01-01", tz="UTC")
    return WindowRecord(
        run_id=run_id, window_index=index, strategy=strategy, symbol="EURUSD", timeframe="H1", scenario="realistic",
        train_start=ts, train_end=ts, validation_start=ts, validation_end=ts, oos_start=ts, oos_end=ts,
        best_params={}, train_objective=1.0, stability_score=0.5, validation_metrics={},
        oos_metrics={}, oos_objective=oos_objective, oos_passed=oos_passed,
    )


def test_walk_forward_chart_none_when_no_windows():
    assert charts.walk_forward_chart([]) is None


def test_walk_forward_chart_picks_the_run_with_the_most_windows():
    small_run = [_window("run-a", i, True, 1.0) for i in range(2)]
    large_run = [_window("run-b", i, i % 2 == 0, 0.5) for i in range(5)]
    result = charts.walk_forward_chart(small_run + large_run)
    assert result is not None
    _, description = result
    assert "5 windows" in description


# --- monte carlo distribution --------------------------------------------------


def _mc_run(histogram):
    return MonteCarloRecord(
        run_id="mc1", strategy="trend_following", symbol="EURUSD", timeframe="H1", scenario="realistic",
        n_simulations=100, n_trades_observed=50, initial_capital=2000.0, ruin_threshold=0.5,
        median_return=0.1, p5_return=-0.05, p95_return=0.3, worst_drawdown=-0.2, p95_drawdown=-0.15,
        median_losing_streak=2.0, p95_losing_streak=4.0, worst_losing_streak=6, probability_of_ruin=0.01,
        probability_of_negative_return=0.1, is_fragile=False, parameters={}, return_histogram=histogram,
    )


def test_monte_carlo_distribution_chart_none_without_histogram():
    assert charts.monte_carlo_distribution_chart(_mc_run({})) is None


def test_monte_carlo_distribution_chart_present_with_histogram():
    histogram = {"bin_edges": [0.0, 0.1, 0.2, 0.3], "counts": [10, 60, 30]}
    assert charts.monte_carlo_distribution_chart(_mc_run(histogram)) is not None


# --- correlation heatmap -------------------------------------------------------


def _portfolio_run(correlation):
    return PortfolioRecord(
        run_id="p1", scenario="realistic", n_components=len(correlation), components=[], correlation=correlation,
        allocations={}, best_component_label="a",
    )


def test_correlation_heatmap_chart_none_with_a_single_component():
    assert charts.correlation_heatmap_chart(_portfolio_run({"a": {"a": 1.0}})) is None


def test_correlation_heatmap_chart_present_with_multiple_components():
    correlation = {"a": {"a": 1.0, "b": 0.3}, "b": {"a": 0.3, "b": 1.0}}
    assert charts.correlation_heatmap_chart(_portfolio_run(correlation)) is not None
