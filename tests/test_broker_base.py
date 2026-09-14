from datetime import datetime, timezone

from src.brokers.base import OrderSide, Position, reconcile_positions


def _position(position_id, symbol="EURUSD", side=OrderSide.BUY, volume=0.1):
    return Position(
        position_id=position_id, symbol=symbol, side=side, volume=volume,
        entry_price=1.10, current_price=1.11, stop_loss=1.09, take_profit=1.15, profit=1.0,
        open_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


def test_reconcile_positions_clean_when_identical():
    p = _position("1")
    report = reconcile_positions([p], [p])
    assert report.is_clean
    assert report.missing_locally == []
    assert report.missing_at_broker == []
    assert report.mismatched == []


def test_reconcile_positions_detects_missing_locally():
    broker_only = _position("1")
    report = reconcile_positions([], [broker_only])
    assert not report.is_clean
    assert report.missing_locally == [broker_only]
    assert report.missing_at_broker == []


def test_reconcile_positions_detects_missing_at_broker():
    local_only = _position("1")
    report = reconcile_positions([local_only], [])
    assert not report.is_clean
    assert report.missing_at_broker == [local_only]
    assert report.missing_locally == []


def test_reconcile_positions_detects_volume_mismatch():
    local = _position("1", volume=0.1)
    broker = _position("1", volume=0.2)
    report = reconcile_positions([local], [broker])
    assert not report.is_clean
    assert report.mismatched == [(local, broker)]


def test_reconcile_positions_detects_side_mismatch():
    local = _position("1", side=OrderSide.BUY)
    broker = _position("1", side=OrderSide.SELL)
    report = reconcile_positions([local], [broker])
    assert report.mismatched == [(local, broker)]


def test_reconcile_positions_ignores_unrelated_field_differences():
    """Only volume/side matter for the mismatch check -- current_price
    and profit naturally differ moment to moment and shouldn't flag a
    reconciliation problem on their own."""
    local = _position("1")
    broker = _position("1")
    broker.current_price = 999.0
    broker.profit = -50.0
    report = reconcile_positions([local], [broker])
    assert report.is_clean
