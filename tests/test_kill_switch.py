from unittest.mock import Mock

import pytest

from src.core.db import Database
from src.risk.kill_switch import KillSwitch


@pytest.fixture()
def db():
    database = Database(url="sqlite:///:memory:")
    database.init_db()
    return database


def test_is_triggered_false_when_no_events_exist(db):
    switch = KillSwitch(db=db)
    assert switch.is_triggered() is False


def test_trip_sets_triggered_true(db):
    switch = KillSwitch(db=db)
    switch.trip("test breach", equity=1800.0, drawdown=0.10)
    assert switch.is_triggered() is True


def test_reset_clears_triggered_state(db):
    switch = KillSwitch(db=db)
    switch.trip("test breach")
    switch.reset("reviewed, resuming")
    assert switch.is_triggered() is False


def test_a_second_trip_after_reset_is_triggered_again(db):
    switch = KillSwitch(db=db)
    switch.trip("first breach")
    switch.reset("resumed")
    assert switch.is_triggered() is False
    switch.trip("second breach")
    assert switch.is_triggered() is True


def test_trip_calls_emergency_stop_on_every_broker(db):
    broker_a, broker_b = Mock(), Mock()
    switch = KillSwitch(db=db)
    switch.trip("test breach", brokers=[broker_a, broker_b])
    broker_a.emergency_stop.assert_called_once()
    broker_b.emergency_stop.assert_called_once()


def test_trip_with_no_brokers_does_not_raise(db):
    switch = KillSwitch(db=db)
    switch.trip("test breach")  # brokers=None -- must not blow up


def test_reset_calls_reset_emergency_stop_on_brokers_by_default(db):
    broker = Mock()
    switch = KillSwitch(db=db)
    switch.trip("test breach")
    switch.reset("resumed", brokers=[broker])
    broker.reset_emergency_stop.assert_called_once()


def test_reset_skips_brokers_when_reset_brokers_is_false(db):
    broker = Mock()
    switch = KillSwitch(db=db)
    switch.trip("test breach")
    switch.reset("resumed", brokers=[broker], reset_brokers=False)
    broker.reset_emergency_stop.assert_not_called()


def test_history_returns_events_newest_first(db):
    switch = KillSwitch(db=db)
    switch.trip("first breach", equity=1800.0, drawdown=0.10)
    switch.reset("resumed")
    switch.trip("second breach", equity=1700.0, drawdown=0.15)

    events = switch.history()
    assert [e.event_type for e in events] == ["trip", "reset", "trip"]
    assert events[0].reason == "second breach"
    assert events[0].drawdown == pytest.approx(0.15)


def test_history_respects_limit(db):
    switch = KillSwitch(db=db)
    for i in range(5):
        switch.trip(f"breach {i}")
        switch.reset(f"resumed {i}")
    events = switch.history(limit=3)
    assert len(events) == 3


def test_two_kill_switch_instances_sharing_a_db_see_the_same_state(db):
    """The durability point: independently constructed KillSwitch
    objects (e.g. from two different RiskEngine instances/sessions)
    against the same database agree on the current state."""
    first = KillSwitch(db=db)
    second = KillSwitch(db=db)
    assert second.is_triggered() is False
    first.trip("breach from session A")
    assert second.is_triggered() is True
