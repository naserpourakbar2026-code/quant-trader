import pytest

from src.core.config import CostScenario
from src.execution.simulator import ExecutionSimulator
from src.strategies.base import SignalDirection


def test_simulate_entry_long_fills_above_open():
    sim = ExecutionSimulator(CostScenario(commission_pct=0.0, slippage_pct=0.001))
    fill = sim.simulate_entry(direction=SignalDirection.LONG, bar_open=1.1000, bar_spread=0.0, size=100.0)
    assert fill.price == pytest.approx(1.1000 * 1.001)


def test_simulate_entry_short_fills_below_open():
    sim = ExecutionSimulator(CostScenario(commission_pct=0.0, slippage_pct=0.001))
    fill = sim.simulate_entry(direction=SignalDirection.SHORT, bar_open=1.1000, bar_spread=0.0, size=100.0)
    assert fill.price == pytest.approx(1.1000 * 0.999)


def test_simulate_exit_long_fills_below_price():
    sim = ExecutionSimulator(CostScenario(commission_pct=0.0, slippage_pct=0.001))
    fill = sim.simulate_exit(direction=SignalDirection.LONG, exit_price=1.2000, bar_spread=0.0, size=100.0)
    assert fill.price == pytest.approx(1.2000 * 0.999)


def test_simulate_exit_short_fills_above_price():
    sim = ExecutionSimulator(CostScenario(commission_pct=0.0, slippage_pct=0.001))
    fill = sim.simulate_exit(direction=SignalDirection.SHORT, exit_price=1.2000, bar_spread=0.0, size=100.0)
    assert fill.price == pytest.approx(1.2000 * 1.001)


def test_commission_folds_in_bar_spread_as_pct_of_price():
    sim = ExecutionSimulator(CostScenario(commission_pct=0.0001, slippage_pct=0.0))
    fill = sim.simulate_entry(direction=SignalDirection.LONG, bar_open=1.0000, bar_spread=0.0001, size=1000.0)
    # spread_pct = 0.0001/1.0 = 0.0001; effective_commission_pct = 0.0001 + 0.0001 = 0.0002
    expected_commission = 1.0000 * 1000.0 * 0.0002
    assert fill.commission == pytest.approx(expected_commission)


def test_zero_cost_scenario_and_zero_spread_gives_zero_commission_and_no_slippage():
    sim = ExecutionSimulator(CostScenario(commission_pct=0.0, slippage_pct=0.0))
    fill = sim.simulate_entry(direction=SignalDirection.LONG, bar_open=1.1000, bar_spread=0.0, size=500.0)
    assert fill.price == pytest.approx(1.1000)
    assert fill.commission == pytest.approx(0.0)


def test_fill_records_slippage_pct_and_spread_for_the_trade_log():
    sim = ExecutionSimulator(CostScenario(commission_pct=0.0, slippage_pct=0.0005))
    fill = sim.simulate_entry(direction=SignalDirection.LONG, bar_open=1.1000, bar_spread=0.00015, size=100.0)
    assert fill.slippage_pct == 0.0005
    assert fill.spread == 0.00015


def test_nan_spread_is_treated_as_zero_not_propagated():
    """A CSV source with no spread column leaves the bar's spread NaN
    (src.data.schema: "not fabricated") -- correct upstream, but it must
    never reach a persisted Fill/trade record as NaN (sqlite's NOT NULL
    rejects it outright for a REAL column bound to NaN)."""
    import math

    sim = ExecutionSimulator(CostScenario(commission_pct=0.0001, slippage_pct=0.0))
    fill = sim.simulate_entry(direction=SignalDirection.LONG, bar_open=1.1000, bar_spread=float("nan"), size=100.0)
    assert math.isfinite(fill.commission)
    assert fill.commission == pytest.approx(1.1000 * 100.0 * 0.0001)
    assert fill.spread == 0.0
