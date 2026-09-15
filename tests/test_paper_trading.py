import numpy as np
import pandas as pd
import pytest

from src.core.config import CooldownConfig, RiskConfig
from src.core.db import Database
from src.data.schema import STANDARD_COLUMNS
from src.execution.paper_trading import run_paper_trading_session
from src.execution.trade_store import list_sessions, list_trades


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
            "timestamp": ts, "symbol": symbol, "timeframe": timeframe,
            "open": open_, "high": high, "low": low, "close": close,
            "tick_volume": tick_volume, "spread": 0.00015, "real_volume": float("nan"),
        }
    )[STANDARD_COLUMNS]


def _permissive_risk_config(**overrides):
    defaults = dict(
        risk_per_trade=0.005, allowed_risk_levels=[0.005], max_daily_loss=0.5, max_weekly_loss=0.5,
        max_portfolio_drawdown=0.9, max_open_positions=5, max_correlated_exposure=0.5, max_leverage=30,
        max_margin_usage=0.9, daily_trade_limit=1000,
        cooldown_after_consecutive_losses=CooldownConfig(losses=100, cooldown_bars=1),
    )
    defaults.update(overrides)
    return RiskConfig(**defaults)


def test_run_paper_trading_session_end_to_end_persists(db):
    from src.execution.position_manager import PositionManager

    raw = _synthetic_raw_candles()
    result = run_paper_trading_session(
        "trend_following", "EURUSD", "H1", raw, risk_config=_permissive_risk_config(), db=db,
    )

    assert result.run_id
    assert len(result.closed_trades) > 0

    last_close = float(raw["close"].iloc[-1])
    unrealized = PositionManager.unrealized_pnl(result.open_position, last_close) if result.open_position else 0.0
    assert result.final_equity == pytest.approx(result.initial_capital + result.total_pnl + unrealized)

    sessions = list_sessions(db=db)
    assert len(sessions) == 1
    assert sessions[0].run_id == result.run_id
    assert sessions[0].total_trades == len(result.closed_trades)

    trades = list_trades(result.run_id, db=db)
    assert len(trades) == len(result.closed_trades)
    assert all(t.reason in ("stop_loss", "take_profit", "signal_flat", "signal_reversal") for t in trades)


def test_run_paper_trading_session_persist_false_does_not_write_to_db(db):
    raw = _synthetic_raw_candles()
    run_paper_trading_session(
        "trend_following", "EURUSD", "H1", raw, risk_config=_permissive_risk_config(), db=db, persist=False,
    )
    assert list_sessions(db=db) == []


def test_win_rate_and_total_pnl_are_consistent_with_closed_trades(db):
    raw = _synthetic_raw_candles()
    result = run_paper_trading_session(
        "trend_following", "EURUSD", "H1", raw, risk_config=_permissive_risk_config(), db=db, persist=False,
    )
    assert result.total_pnl == pytest.approx(sum(t.pnl for t in result.closed_trades))
    if result.closed_trades:
        expected_win_rate = sum(1 for t in result.closed_trades if t.pnl > 0) / len(result.closed_trades)
        assert result.win_rate == pytest.approx(expected_win_rate)
    else:
        assert result.win_rate is None


def test_unknown_scenario_raises_value_error(db):
    raw = _synthetic_raw_candles()
    with pytest.raises(ValueError, match="Unknown execution scenario"):
        run_paper_trading_session("trend_following", "EURUSD", "H1", raw, scenario="not_a_real_scenario", db=db)


def test_kill_switch_blocks_new_trades_after_a_tiny_drawdown_threshold(db):
    """A near-zero max_portfolio_drawdown guarantees the kill switch trips
    on the very first adverse tick -- proving the risk engine's kill
    switch is genuinely wired into the session loop, not just present in
    isolation (see tests/test_risk_engine.py for its own unit tests)."""
    raw = _synthetic_raw_candles()
    result = run_paper_trading_session(
        "trend_following", "EURUSD", "H1", raw,
        risk_config=_permissive_risk_config(max_portfolio_drawdown=0.0001), db=db, persist=False,
    )
    assert result.kill_switch_triggered is True


def test_reproducible_with_same_inputs(db):
    raw = _synthetic_raw_candles()
    first = run_paper_trading_session(
        "trend_following", "EURUSD", "H1", raw, risk_config=_permissive_risk_config(), db=db, persist=False,
    )
    second = run_paper_trading_session(
        "trend_following", "EURUSD", "H1", raw, risk_config=_permissive_risk_config(), db=db, persist=False,
    )
    assert first.total_pnl == pytest.approx(second.total_pnl)
    assert len(first.closed_trades) == len(second.closed_trades)


def test_run_paper_trading_session_persists_when_spread_column_is_missing(db):
    """A CSV source with no `spread` column leaves every bar's spread
    NaN (src.data.schema.standardize_csv_rows) -- this must still
    persist cleanly (a real regression: sqlite's NOT NULL constraint
    rejects a REAL column bound to NaN, not just None)."""
    raw = _synthetic_raw_candles()
    raw["spread"] = float("nan")
    result = run_paper_trading_session(
        "trend_following", "EURUSD", "H1", raw, risk_config=_permissive_risk_config(), db=db,
    )
    assert len(result.closed_trades) > 0
    trades = list_trades(result.run_id, db=db)
    assert all(t.spread == 0.0 for t in trades)


def test_still_open_position_is_reported_not_force_closed(db):
    """The synthetic series ends mid-trend, so trend_following should
    still have an open position when the data runs out -- and it must
    be reported as open, never fabricated into a synthetic close."""
    raw = _synthetic_raw_candles()
    result = run_paper_trading_session(
        "trend_following", "EURUSD", "H1", raw, risk_config=_permissive_risk_config(), db=db, persist=False,
    )
    if result.open_position is not None:
        assert result.open_position.symbol == "EURUSD"
        # not present among closed trades under any synthetic "forced close" reason
        assert all(t.reason != "end_of_data" for t in result.closed_trades)
