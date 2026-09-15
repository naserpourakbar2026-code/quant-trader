import pandas as pd
import pytest

from src.core.config import CooldownConfig, RiskConfig
from src.core.db import Database
from src.risk.engine import RiskEngine
from src.strategies.base import PositionSizeResult


@pytest.fixture()
def db():
    database = Database(url="sqlite:///:memory:")
    database.init_db()
    return database


def _risk_config(**overrides):
    defaults = dict(
        risk_per_trade=0.005, allowed_risk_levels=[0.0025, 0.005, 0.0075, 0.01],
        max_daily_loss=0.03, max_weekly_loss=0.08, max_portfolio_drawdown=0.15,
        max_open_positions=5, max_correlated_exposure=0.02, max_leverage=30,
        max_margin_usage=0.5, daily_trade_limit=10,
        cooldown_after_consecutive_losses=CooldownConfig(losses=3, cooldown_bars=12),
    )
    defaults.update(overrides)
    return RiskConfig(**defaults)


def _ts(day="2024-01-01", hour=0):
    return pd.Timestamp(f"{day}T{hour:02d}:00:00Z")


def _size(units=1000.0, risk_amount=10.0, stop_distance=0.01):
    return PositionSizeResult(units=units, risk_amount=risk_amount, stop_distance=stop_distance)


def test_evaluate_approves_a_normal_order(db):
    engine = RiskEngine(_risk_config(), 2000.0, db=db)
    engine.update_equity(2000.0, _ts())
    decision = engine.evaluate(equity=2000.0, position_size=_size(units=100.0), entry_price=1.10, open_positions_count=0)
    assert decision.approved is True


def test_kill_switch_triggers_on_drawdown_and_stays_triggered(db):
    engine = RiskEngine(_risk_config(max_portfolio_drawdown=0.10), 2000.0, db=db)
    engine.update_equity(2000.0, _ts())
    engine.update_equity(1800.0, _ts(hour=1))  # exactly -10% from peak
    assert engine.kill_switch_triggered is True
    decision = engine.evaluate(equity=1800.0, position_size=_size(), entry_price=1.10, open_positions_count=0)
    assert decision.approved is False
    assert "kill switch" in decision.reason

    # Recovering equity does NOT un-trip the switch -- it's a hard switch, not a threshold check.
    engine.update_equity(2000.0, _ts(hour=2))
    decision = engine.evaluate(equity=2000.0, position_size=_size(), entry_price=1.10, open_positions_count=0)
    assert decision.approved is False


def test_kill_switch_trip_is_persisted_and_inherited_by_a_new_engine(db):
    """The whole point of a "hard" kill switch: once one RiskEngine
    instance trips it, a brand-new RiskEngine sharing the same database
    starts already triggered -- one session's breach stops every other
    session too, not just itself."""
    first = RiskEngine(_risk_config(max_portfolio_drawdown=0.10), 2000.0, db=db)
    first.update_equity(2000.0, _ts())
    first.update_equity(1800.0, _ts(hour=1))
    assert first.kill_switch_triggered is True

    second = RiskEngine(_risk_config(max_portfolio_drawdown=0.10), 2000.0, db=db)
    assert second.kill_switch_triggered is True
    decision = second.evaluate(equity=2000.0, position_size=_size(), entry_price=1.10, open_positions_count=0)
    assert decision.approved is False


def test_kill_switch_trip_notifies_registered_brokers(db):
    from unittest.mock import Mock

    broker = Mock()
    engine = RiskEngine(_risk_config(max_portfolio_drawdown=0.10), 2000.0, db=db, brokers=[broker])
    engine.update_equity(2000.0, _ts())
    engine.update_equity(1800.0, _ts(hour=1))
    broker.emergency_stop.assert_called_once()


def test_no_kill_switch_below_drawdown_threshold(db):
    engine = RiskEngine(_risk_config(max_portfolio_drawdown=0.10), 2000.0, db=db)
    engine.update_equity(2000.0, _ts())
    engine.update_equity(1850.0, _ts(hour=1))  # -7.5%, under the 10% threshold
    assert engine.kill_switch_triggered is False


def test_cooldown_activates_after_consecutive_losses_and_expires(db):
    engine = RiskEngine(_risk_config(cooldown_after_consecutive_losses=CooldownConfig(losses=2, cooldown_bars=3)), 2000.0, db=db)
    engine.update_equity(2000.0, _ts())
    engine.record_trade_close(-10.0)
    engine.record_trade_close(-10.0)  # 2nd consecutive loss -> cooldown starts
    assert engine.cooldown_remaining_bars == 3

    decision = engine.evaluate(equity=1980.0, position_size=_size(), entry_price=1.10, open_positions_count=0)
    assert decision.approved is False
    assert "cooldown" in decision.reason

    for hour in range(1, 4):
        engine.update_equity(1980.0, _ts(hour=hour))  # ticks cooldown down each bar
    assert engine.cooldown_remaining_bars == 0
    decision = engine.evaluate(equity=1980.0, position_size=_size(), entry_price=1.10, open_positions_count=0)
    assert decision.approved is True


def test_a_winning_trade_resets_consecutive_loss_count(db):
    engine = RiskEngine(_risk_config(cooldown_after_consecutive_losses=CooldownConfig(losses=2, cooldown_bars=3)), 2000.0, db=db)
    engine.record_trade_close(-10.0)
    engine.record_trade_close(15.0)  # win resets the streak
    assert engine.consecutive_losses == 0
    engine.record_trade_close(-10.0)
    assert engine.cooldown_remaining_bars == 0  # only 1 consecutive loss so far, not 2


def test_daily_trade_limit_blocks_further_orders_same_day(db):
    engine = RiskEngine(_risk_config(daily_trade_limit=2), 2000.0, db=db)
    engine.update_equity(2000.0, _ts())
    engine.notify_order_opened()
    engine.notify_order_opened()
    decision = engine.evaluate(equity=2000.0, position_size=_size(), entry_price=1.10, open_positions_count=0)
    assert decision.approved is False
    assert "daily trade limit" in decision.reason


def test_daily_trade_count_resets_on_a_new_calendar_day(db):
    engine = RiskEngine(_risk_config(daily_trade_limit=1), 2000.0, db=db)
    engine.update_equity(2000.0, _ts(day="2024-01-01"))
    engine.notify_order_opened()
    decision = engine.evaluate(equity=2000.0, position_size=_size(), entry_price=1.10, open_positions_count=0)
    assert decision.approved is False

    engine.update_equity(2000.0, _ts(day="2024-01-02"))  # a new day
    decision = engine.evaluate(equity=2000.0, position_size=_size(), entry_price=1.10, open_positions_count=0)
    assert decision.approved is True


def test_max_daily_loss_blocks_new_orders(db):
    engine = RiskEngine(_risk_config(max_daily_loss=0.02), 2000.0, db=db)
    engine.update_equity(2000.0, _ts())  # sets day_start_equity = 2000
    engine.update_equity(1950.0, _ts(hour=1))  # -2.5%, breaches 2% daily loss limit
    decision = engine.evaluate(equity=1950.0, position_size=_size(), entry_price=1.10, open_positions_count=0)
    assert decision.approved is False
    assert "daily loss" in decision.reason


def test_max_weekly_loss_blocks_new_orders(db):
    engine = RiskEngine(_risk_config(max_weekly_loss=0.05), 2000.0, db=db)
    engine.update_equity(2000.0, _ts(day="2024-01-01"))  # a Monday -- sets week_start_equity
    engine.update_equity(1880.0, _ts(day="2024-01-03"))  # -6%, breaches 5% weekly loss limit, same week
    decision = engine.evaluate(equity=1880.0, position_size=_size(), entry_price=1.10, open_positions_count=0)
    assert decision.approved is False
    assert "weekly loss" in decision.reason


def test_weekly_tracker_resets_on_a_new_iso_week(db):
    engine = RiskEngine(_risk_config(max_weekly_loss=0.05), 2000.0, db=db)
    engine.update_equity(2000.0, _ts(day="2024-01-01"))  # Monday, week 1
    engine.update_equity(1880.0, _ts(day="2024-01-08"))  # Monday, week 2 -- new week, tracker resets to 1880
    decision = engine.evaluate(equity=1880.0, position_size=_size(), entry_price=1.10, open_positions_count=0)
    assert decision.approved is True


def test_max_open_positions_blocks_further_orders(db):
    engine = RiskEngine(_risk_config(max_open_positions=1), 2000.0, db=db)
    engine.update_equity(2000.0, _ts())
    decision = engine.evaluate(equity=2000.0, position_size=_size(), entry_price=1.10, open_positions_count=1)
    assert decision.approved is False
    assert "max open positions" in decision.reason


def test_insufficient_margin_blocks_an_oversized_order(db):
    engine = RiskEngine(_risk_config(max_leverage=10, max_margin_usage=0.5), 2000.0, db=db)
    engine.update_equity(2000.0, _ts())
    # notional = 100000 units * 1.10 = 110,000; required margin = 11,000; available = 2000*0.5=1000
    decision = engine.evaluate(equity=2000.0, position_size=_size(units=100_000.0), entry_price=1.10, open_positions_count=0)
    assert decision.approved is False
    assert "margin" in decision.reason


def test_sufficient_margin_allows_a_reasonably_sized_order(db):
    engine = RiskEngine(_risk_config(max_leverage=30, max_margin_usage=0.5), 2000.0, db=db)
    engine.update_equity(2000.0, _ts())
    # notional = 1000 * 1.10 = 1100; required margin = 36.67; available = 1000
    decision = engine.evaluate(equity=2000.0, position_size=_size(units=1000.0), entry_price=1.10, open_positions_count=0)
    assert decision.approved is True
