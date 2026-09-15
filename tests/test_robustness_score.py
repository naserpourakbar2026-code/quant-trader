import pandas as pd
import pytest

from src.core.config import load_settings
from src.montecarlo.mc_engine import MonteCarloResult
from src.robustness.cost_stress import CostStressPoint, CostStressResult
from src.robustness.score import RobustnessStatus, compute_robustness_score
from src.walkforward.wfa_engine import WalkForwardReport, WalkForwardWindowResult


@pytest.fixture()
def settings():
    return load_settings()


def _window(oos_metrics, *, oos_passed=True, train_objective=1.0, stability_score=0.9, best_params=None):
    ts = pd.Timestamp("2024-01-01", tz="UTC")
    return WalkForwardWindowResult(
        index=0, train_range=(ts, ts), validation_range=(ts, ts), oos_range=(ts, ts),
        best_params=best_params or {"a": 1}, train_objective=train_objective, stability_score=stability_score,
        validation_metrics={}, oos_metrics=oos_metrics, oos_objective=1.0 if oos_passed else -1.0,
        oos_passed=oos_passed,
    )


def _wf_report(windows):
    return WalkForwardReport(run_id="wf1", strategy="trend_following", symbol="EURUSD", timeframe="H1", windows=windows)


def _mc_result(*, probability_of_ruin=0.0, is_fragile=False):
    return MonteCarloResult(
        run_id="mc1", strategy="trend_following", symbol="EURUSD", timeframe="H1", scenario="realistic",
        n_simulations=100, n_trades_observed=50, initial_capital=2000.0, ruin_threshold=0.5, median_return=0.1,
        p5_return=-0.05, p95_return=0.3, worst_drawdown=-0.2, p95_drawdown=-0.15, median_losing_streak=2.0,
        p95_losing_streak=4.0, worst_losing_streak=6, probability_of_ruin=probability_of_ruin,
        probability_of_negative_return=0.1, is_fragile=is_fragile,
    )


def _cost_stress(*, is_fragile_points=False):
    if is_fragile_points:
        points = [
            CostStressPoint(1.0, "realistic", {"profit_factor": 2.0}),
            CostStressPoint(1.5, "realistic", {"profit_factor": 0.8}),
        ]
    else:
        points = [
            CostStressPoint(1.0, "realistic", {"profit_factor": 3.0}),
            CostStressPoint(1.5, "realistic", {"profit_factor": 2.8}),
            CostStressPoint(3.0, "realistic", {"profit_factor": 2.5}),
        ]
    return CostStressResult(run_id="cs1", strategy="trend_following", symbol="EURUSD", timeframe="H1",
                             baseline_scenario="realistic", points=points)


_GOOD_OOS_METRICS = {
    "total_return": 0.1, "sharpe_ratio": 1.5, "sortino_ratio": 1.8, "max_drawdown": -0.05,
    "trade_count": 40, "win_rate": 0.6, "profit_factor": 2.0, "expectancy": 5.0,
}


def test_insufficient_data_when_no_walkforward_report(settings):
    score = compute_robustness_score(
        strategy="s", symbol="EURUSD", timeframe="H1", settings=settings, walkforward_report=None,
        montecarlo_result=_mc_result(), cost_stress_result=_cost_stress(), stability_score=0.9, n_trades=50,
    )
    assert score.status == RobustnessStatus.INSUFFICIENT_DATA
    assert score.score is None


def test_insufficient_data_when_walkforward_has_no_windows(settings):
    score = compute_robustness_score(
        strategy="s", symbol="EURUSD", timeframe="H1", settings=settings, walkforward_report=_wf_report([]),
        montecarlo_result=_mc_result(), cost_stress_result=_cost_stress(), stability_score=0.9, n_trades=50,
    )
    assert score.status == RobustnessStatus.INSUFFICIENT_DATA


def test_insufficient_data_when_no_montecarlo_result(settings):
    score = compute_robustness_score(
        strategy="s", symbol="EURUSD", timeframe="H1", settings=settings,
        walkforward_report=_wf_report([_window(_GOOD_OOS_METRICS)]), montecarlo_result=None,
        cost_stress_result=_cost_stress(), stability_score=0.9, n_trades=50,
    )
    assert score.status == RobustnessStatus.INSUFFICIENT_DATA


def test_insufficient_data_when_too_few_trades(settings):
    score = compute_robustness_score(
        strategy="s", symbol="EURUSD", timeframe="H1", settings=settings,
        walkforward_report=_wf_report([_window(_GOOD_OOS_METRICS)]), montecarlo_result=_mc_result(),
        cost_stress_result=_cost_stress(), stability_score=0.9, n_trades=1,
    )
    assert score.status == RobustnessStatus.INSUFFICIENT_DATA


def test_pass_for_a_solid_all_around_result(settings):
    score = compute_robustness_score(
        strategy="s", symbol="EURUSD", timeframe="H1", settings=settings,
        walkforward_report=_wf_report([_window(_GOOD_OOS_METRICS, oos_passed=True, stability_score=0.9)]),
        montecarlo_result=_mc_result(probability_of_ruin=0.0), cost_stress_result=_cost_stress(),
        stability_score=0.9, n_trades=50,
    )
    assert score.status == RobustnessStatus.PASS
    assert score.score is not None and score.score >= settings.robustness.min_pass_score


def test_fail_when_oos_profit_factor_is_at_or_below_one(settings):
    metrics = {**_GOOD_OOS_METRICS, "profit_factor": 0.8}
    score = compute_robustness_score(
        strategy="s", symbol="EURUSD", timeframe="H1", settings=settings,
        walkforward_report=_wf_report([_window(metrics, oos_passed=False)]), montecarlo_result=_mc_result(),
        cost_stress_result=_cost_stress(), stability_score=0.9, n_trades=50,
    )
    assert score.status == RobustnessStatus.FAIL


def test_none_profit_factor_is_not_treated_as_a_failure():
    """vectorbt reports profit_factor=None for zero losing trades --
    that must never read as "failed OOS"."""
    settings = load_settings()
    metrics = {**_GOOD_OOS_METRICS, "profit_factor": None, "win_rate": 1.0}
    score = compute_robustness_score(
        strategy="s", symbol="EURUSD", timeframe="H1", settings=settings,
        walkforward_report=_wf_report([_window(metrics, oos_passed=True, stability_score=0.9)]),
        montecarlo_result=_mc_result(), cost_stress_result=_cost_stress(), stability_score=0.9, n_trades=50,
    )
    assert score.status != RobustnessStatus.FAIL
    assert score.sub_scores["profit_factor"] == pytest.approx(100.0)


def test_fail_when_oos_pass_rate_is_zero(settings):
    score = compute_robustness_score(
        strategy="s", symbol="EURUSD", timeframe="H1", settings=settings,
        walkforward_report=_wf_report([_window(_GOOD_OOS_METRICS, oos_passed=False)]),
        montecarlo_result=_mc_result(), cost_stress_result=_cost_stress(), stability_score=0.9, n_trades=50,
    )
    assert score.status == RobustnessStatus.FAIL


def test_cost_fragile_status(settings):
    score = compute_robustness_score(
        strategy="s", symbol="EURUSD", timeframe="H1", settings=settings,
        walkforward_report=_wf_report([_window(_GOOD_OOS_METRICS, oos_passed=True, stability_score=0.9)]),
        montecarlo_result=_mc_result(), cost_stress_result=_cost_stress(is_fragile_points=True),
        stability_score=0.9, n_trades=50,
    )
    assert score.status == RobustnessStatus.COST_FRAGILE


def test_overfit_status_when_training_strong_but_oos_and_stability_weak(settings):
    # OVERFIT needs 0 < oos_pass_rate < 0.5 (a nonzero pass rate --
    # otherwise FAIL takes precedence) with weak parameter stability:
    # 1 passing window out of 3 gives a pass rate of 1/3.
    windows = [
        _window(_GOOD_OOS_METRICS, oos_passed=True, train_objective=5.0, stability_score=0.2),
        _window(_GOOD_OOS_METRICS, oos_passed=False, train_objective=5.0, stability_score=0.2),
        _window(_GOOD_OOS_METRICS, oos_passed=False, train_objective=5.0, stability_score=0.2),
    ]
    report = WalkForwardReport(run_id="wf1", strategy="s", symbol="EURUSD", timeframe="H1", windows=windows)
    score = compute_robustness_score(
        strategy="s", symbol="EURUSD", timeframe="H1", settings=settings, walkforward_report=report,
        montecarlo_result=_mc_result(), cost_stress_result=_cost_stress(), stability_score=0.2, n_trades=50,
    )
    assert score.status == RobustnessStatus.OVERFIT


def test_not_robust_when_montecarlo_flags_fragile(settings):
    score = compute_robustness_score(
        strategy="s", symbol="EURUSD", timeframe="H1", settings=settings,
        walkforward_report=_wf_report([_window(_GOOD_OOS_METRICS, oos_passed=True, stability_score=0.9)]),
        montecarlo_result=_mc_result(is_fragile=True), cost_stress_result=_cost_stress(), stability_score=0.9,
        n_trades=50,
    )
    assert score.status == RobustnessStatus.NOT_ROBUST


def test_weighted_sub_scores_sum_matches_reported_score(settings):
    score = compute_robustness_score(
        strategy="s", symbol="EURUSD", timeframe="H1", settings=settings,
        walkforward_report=_wf_report([_window(_GOOD_OOS_METRICS, oos_passed=True, stability_score=0.9)]),
        montecarlo_result=_mc_result(), cost_stress_result=_cost_stress(), stability_score=0.9, n_trades=50,
    )
    weights = settings.robustness.weights
    expected = (
        score.sub_scores["oos_performance"] * weights.oos_performance
        + score.sub_scores["drawdown"] * weights.drawdown
        + score.sub_scores["profit_factor"] * weights.profit_factor
        + score.sub_scores["sharpe_sortino"] * weights.sharpe_sortino
        + score.sub_scores["parameter_stability"] * weights.parameter_stability
        + score.sub_scores["monte_carlo_robustness"] * weights.monte_carlo_robustness
        + score.sub_scores["cost_sensitivity"] * weights.cost_sensitivity
        + score.sub_scores["trade_count_reliability"] * weights.trade_count_reliability
    )
    assert score.score == pytest.approx(expected)


def test_to_text_includes_header_and_status():
    settings = load_settings()
    score = compute_robustness_score(
        strategy="trend_following", symbol="EURUSD", timeframe="H1", settings=settings,
        walkforward_report=_wf_report([_window(_GOOD_OOS_METRICS, oos_passed=True, stability_score=0.9)]),
        montecarlo_result=_mc_result(), cost_stress_result=_cost_stress(), stability_score=0.9, n_trades=50,
    )
    text = score.to_text()
    assert text.startswith("FINAL ROBUSTNESS EVALUATION")
    assert score.status.value in text
