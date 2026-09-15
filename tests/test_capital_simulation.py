import pandas as pd
import pytest

from src.execution.trade_store import PaperTradeRecord
from src.robustness.capital_simulation import run_capital_simulation


def _trade(exit_time, pnl, risk_amount=10.0, entry_price=1.10, stop_loss=1.09):
    return PaperTradeRecord(
        trade_id="t", run_id="r", strategy="trend_following", symbol="EURUSD", timeframe="H1", direction="LONG",
        entry_time=exit_time, entry_price=entry_price, stop_loss=stop_loss, take_profit=1.12, size=1000.0,
        risk_amount=risk_amount, spread=0.0001, slippage_pct=0.0001, exit_time=exit_time, exit_price=1.11,
        pnl=pnl, r_multiple=(pnl / risk_amount) if risk_amount else None, reason="take_profit",
    )


def _trades(n=20, start="2024-01-01"):
    base = pd.Timestamp(start, tz="UTC")
    # alternating win/win/loss pattern, R-multiples of +1.0 twice then -0.5
    return [
        _trade(base + pd.Timedelta(days=5 * i), pnl=(10.0 if i % 3 != 2 else -5.0))
        for i in range(n)
    ]


def test_no_trades_returns_flat_result_at_initial_capital():
    result = run_capital_simulation(
        [], strategy="s", symbol="EURUSD", timeframe="H1", initial_capital=2000.0,
        risk_levels=[0.005], max_leverage=30, n_simulations=50,
    )
    assert result.n_trades == 0
    level = result.risk_levels[0]
    assert level.final_equity == 2000.0
    assert level.max_drawdown_eur == 0.0
    assert level.probability_of_ruin == 0.0


def test_higher_risk_level_compounds_to_higher_final_equity_for_a_net_winning_series():
    trades = _trades(20)
    result = run_capital_simulation(
        trades, strategy="s", symbol="EURUSD", timeframe="H1", initial_capital=2000.0,
        risk_levels=[0.0025, 0.005, 0.01], max_leverage=30, n_simulations=200, seed=1,
    )
    final_equities = [r.final_equity for r in result.risk_levels]
    assert final_equities == sorted(final_equities)  # monotonically increasing with risk_pct
    assert all(fe > 2000.0 for fe in final_equities)  # net-winning series ends up ahead at every risk level


def test_margin_utilization_scales_linearly_with_risk_pct():
    trades = [_trade(pd.Timestamp("2024-01-01", tz="UTC"), pnl=10.0, entry_price=1.10, stop_loss=1.089)]  # 1% stop distance
    result = run_capital_simulation(
        trades, strategy="s", symbol="EURUSD", timeframe="H1", initial_capital=2000.0,
        risk_levels=[0.005, 0.01], max_leverage=10, n_simulations=10,
    )
    low, high = result.risk_levels
    # margin_utilization = risk_pct / (stop_distance_pct * max_leverage) -- doubling risk_pct doubles it
    assert high.avg_margin_utilization == pytest.approx(low.avg_margin_utilization * 2, rel=1e-6)


def test_max_losing_streak_counts_consecutive_negative_r_multiples():
    base = pd.Timestamp("2024-01-01", tz="UTC")
    trades = [
        _trade(base + pd.Timedelta(days=i), pnl=pnl)
        for i, pnl in enumerate([10.0, -5.0, -5.0, -5.0, 10.0, -5.0])
    ]
    result = run_capital_simulation(
        trades, strategy="s", symbol="EURUSD", timeframe="H1", initial_capital=2000.0,
        risk_levels=[0.005], max_leverage=30, n_simulations=10,
    )
    assert result.risk_levels[0].max_losing_streak == 3


def test_probability_of_ruin_is_high_when_every_trade_is_a_large_loss():
    base = pd.Timestamp("2024-01-01", tz="UTC")
    trades = [_trade(base + pd.Timedelta(days=i), pnl=-90.0, risk_amount=10.0) for i in range(5)]  # R = -9 each
    result = run_capital_simulation(
        trades, strategy="s", symbol="EURUSD", timeframe="H1", initial_capital=2000.0,
        risk_levels=[0.5], max_leverage=30, n_simulations=100, ruin_threshold=0.5, seed=1,
    )
    # risk_pct=0.5 * R=-9 => -450% on the very first trade -> instant ruin in every simulation
    assert result.risk_levels[0].probability_of_ruin == pytest.approx(1.0)


def test_trades_with_zero_risk_amount_are_excluded():
    trades = _trades(5) + [_trade(pd.Timestamp("2024-02-01", tz="UTC"), pnl=0.0, risk_amount=0.0)]
    result = run_capital_simulation(
        trades, strategy="s", symbol="EURUSD", timeframe="H1", initial_capital=2000.0,
        risk_levels=[0.005], max_leverage=30, n_simulations=10,
    )
    assert result.n_trades == 5


def test_to_text_includes_header_and_every_risk_level():
    trades = _trades(5)
    result = run_capital_simulation(
        trades, strategy="trend_following", symbol="EURUSD", timeframe="H1", initial_capital=2000.0,
        risk_levels=[0.0025, 0.005], max_leverage=30, n_simulations=10,
    )
    text = result.to_text()
    assert text.startswith("CAPITAL SIMULATION")
    assert "risk=0.25%" in text
    assert "risk=0.50%" in text
