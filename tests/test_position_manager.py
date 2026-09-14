import pandas as pd
import pytest

from src.execution.position_manager import PositionManager
from src.execution.simulator import Fill
from src.strategies.base import SignalDirection


def _fill(price=1.1000, commission=0.0, slippage_pct=0.0, spread=0.0001):
    return Fill(price=price, commission=commission, slippage_pct=slippage_pct, spread=spread)


def _open_long(entry_price=1.1000, stop_loss=1.0950, take_profit=1.1100, size=1000.0, risk_amount=50.0, commission=0.0):
    return PositionManager.open(
        strategy="trend_following", symbol="EURUSD", timeframe="H1", direction=SignalDirection.LONG,
        entry_time=pd.Timestamp("2024-01-01T00:00:00Z"), stop_loss=stop_loss, take_profit=take_profit,
        size=size, risk_amount=risk_amount, fill=_fill(price=entry_price, commission=commission),
    )


def _open_short(entry_price=1.1000, stop_loss=1.1050, take_profit=1.0900, size=1000.0, risk_amount=50.0):
    return PositionManager.open(
        strategy="trend_following", symbol="EURUSD", timeframe="H1", direction=SignalDirection.SHORT,
        entry_time=pd.Timestamp("2024-01-01T00:00:00Z"), stop_loss=stop_loss, take_profit=take_profit,
        size=size, risk_amount=risk_amount, fill=_fill(price=entry_price),
    )


def test_open_sets_entry_price_from_the_fill_not_the_raw_signal_price():
    position = _open_long(entry_price=1.1005)  # fill price differs from a hypothetical signal price due to slippage
    assert position.entry_price == 1.1005
    assert position.direction == SignalDirection.LONG


def test_unrealized_pnl_long_is_positive_when_price_rises():
    position = _open_long(entry_price=1.1000, size=1000.0, commission=1.0)
    pnl = PositionManager.unrealized_pnl(position, 1.1050)
    assert pnl == pytest.approx((1.1050 - 1.1000) * 1000.0 - 1.0)


def test_unrealized_pnl_short_is_positive_when_price_falls():
    position = _open_short(entry_price=1.1000, size=1000.0)
    pnl = PositionManager.unrealized_pnl(position, 1.0950)
    assert pnl == pytest.approx((1.1000 - 1.0950) * 1000.0)


def _bar(low, high):
    return pd.Series({"low": low, "high": high})


def test_check_stop_touch_long_hits_stop_loss():
    position = _open_long(stop_loss=1.0950, take_profit=1.1100)
    result = PositionManager.check_stop_touch(position, _bar(low=1.0940, high=1.0980))
    assert result == (1.0950, "stop_loss")


def test_check_stop_touch_long_hits_take_profit():
    position = _open_long(stop_loss=1.0950, take_profit=1.1100)
    result = PositionManager.check_stop_touch(position, _bar(low=1.1050, high=1.1120))
    assert result == (1.1100, "take_profit")


def test_check_stop_touch_long_returns_none_when_neither_touched():
    position = _open_long(stop_loss=1.0950, take_profit=1.1100)
    result = PositionManager.check_stop_touch(position, _bar(low=1.0990, high=1.1010))
    assert result is None


def test_check_stop_touch_prioritizes_stop_loss_when_both_touched_same_bar():
    position = _open_long(entry_price=1.1000, stop_loss=1.0950, take_profit=1.1100)
    result = PositionManager.check_stop_touch(position, _bar(low=1.0900, high=1.1200))
    assert result == (1.0950, "stop_loss")


def test_check_stop_touch_short_hits_stop_loss():
    position = _open_short(stop_loss=1.1050, take_profit=1.0900)
    result = PositionManager.check_stop_touch(position, _bar(low=1.0980, high=1.1060))
    assert result == (1.1050, "stop_loss")


def test_check_stop_touch_short_hits_take_profit():
    position = _open_short(stop_loss=1.1050, take_profit=1.0900)
    result = PositionManager.check_stop_touch(position, _bar(low=1.0890, high=1.0990))
    assert result == (1.0900, "take_profit")


def test_close_computes_pnl_and_positive_r_multiple_for_a_long_winner():
    position = _open_long(entry_price=1.1000, stop_loss=1.0950, take_profit=1.1100, size=1000.0, risk_amount=50.0)
    trade = PositionManager.close(
        position, exit_time=pd.Timestamp("2024-01-01T05:00:00Z"),
        fill=_fill(price=1.1100, commission=2.0), reason="take_profit",
    )
    expected_pnl = (1.1100 - 1.1000) * 1000.0 - 0.0 - 2.0  # entry commission was 0.0 by default
    assert trade.pnl == pytest.approx(expected_pnl)
    assert trade.r_multiple == pytest.approx(expected_pnl / 50.0)
    assert trade.reason == "take_profit"
    assert trade.status == "closed"


def test_close_computes_negative_pnl_for_a_long_loser():
    position = _open_long(entry_price=1.1000, stop_loss=1.0950, take_profit=1.1100, size=1000.0, risk_amount=50.0)
    trade = PositionManager.close(
        position, exit_time=pd.Timestamp("2024-01-01T05:00:00Z"), fill=_fill(price=1.0950), reason="stop_loss",
    )
    assert trade.pnl == pytest.approx((1.0950 - 1.1000) * 1000.0)
    assert trade.r_multiple < 0


def test_close_computes_pnl_for_a_short_winner():
    position = _open_short(entry_price=1.1000, stop_loss=1.1050, take_profit=1.0900, size=1000.0, risk_amount=50.0)
    trade = PositionManager.close(
        position, exit_time=pd.Timestamp("2024-01-01T05:00:00Z"), fill=_fill(price=1.0900), reason="take_profit",
    )
    assert trade.pnl == pytest.approx((1.1000 - 1.0900) * 1000.0)
    assert trade.r_multiple > 0


def test_close_r_multiple_is_none_when_risk_amount_is_zero():
    position = _open_long(risk_amount=0.0)
    trade = PositionManager.close(
        position, exit_time=pd.Timestamp("2024-01-01T05:00:00Z"), fill=_fill(price=1.1100), reason="take_profit",
    )
    assert trade.r_multiple is None


def test_close_carries_entry_spread_and_slippage_into_the_trade_log():
    position = PositionManager.open(
        strategy="trend_following", symbol="EURUSD", timeframe="H1", direction=SignalDirection.LONG,
        entry_time=pd.Timestamp("2024-01-01T00:00:00Z"), stop_loss=1.0950, take_profit=1.1100,
        size=1000.0, risk_amount=50.0, fill=_fill(price=1.1000, slippage_pct=0.0007, spread=0.00012),
    )
    trade = PositionManager.close(
        position, exit_time=pd.Timestamp("2024-01-01T05:00:00Z"), fill=_fill(price=1.1100), reason="take_profit",
    )
    assert trade.slippage_pct == 0.0007
    assert trade.spread == 0.00012
