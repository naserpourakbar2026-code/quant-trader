import sys
import types
from datetime import datetime

import numpy as np
import pytest

from src.data.mt5_loader import MT5UnavailableError, connect, disconnect, fetch_rates


def _fake_mt5_module(*, initialize_returns=True, rates=None):
    mod = types.ModuleType("MetaTrader5")
    mod.TIMEFRAME_M5 = 5
    mod.TIMEFRAME_M15 = 15
    mod.TIMEFRAME_M30 = 30
    mod.TIMEFRAME_H1 = 16385
    mod.TIMEFRAME_H4 = 16388

    mod.initialize = lambda **kwargs: initialize_returns
    mod.shutdown = lambda: None
    mod.last_error = lambda: (1, "mock error")
    mod.copy_rates_range = lambda symbol, timeframe, start, end: rates
    return mod


def test_connect_raises_when_metatrader5_not_importable(monkeypatch):
    monkeypatch.setitem(sys.modules, "MetaTrader5", None)
    with pytest.raises(MT5UnavailableError, match="not installed/importable"):
        connect()


def test_connect_raises_when_initialize_fails(monkeypatch):
    monkeypatch.setitem(sys.modules, "MetaTrader5", _fake_mt5_module(initialize_returns=False))
    with pytest.raises(MT5UnavailableError, match="initialize\\(\\) failed"):
        connect()


def test_connect_succeeds_with_mocked_terminal(monkeypatch):
    monkeypatch.setitem(sys.modules, "MetaTrader5", _fake_mt5_module(initialize_returns=True))
    mt5 = connect(login=123, password="pw", server="Demo-Server")
    assert mt5 is not None
    disconnect()  # should not raise


def test_fetch_rates_standardizes_mocked_response(monkeypatch):
    dtype = [
        ("time", "i8"),
        ("open", "f8"),
        ("high", "f8"),
        ("low", "f8"),
        ("close", "f8"),
        ("tick_volume", "i8"),
        ("spread", "i8"),
        ("real_volume", "i8"),
    ]
    rates = np.array([(1704067200, 1.1, 1.15, 1.05, 1.12, 100, 2, 500)], dtype=dtype)
    monkeypatch.setitem(sys.modules, "MetaTrader5", _fake_mt5_module(rates=rates))

    df = fetch_rates("EURUSD", "H1", datetime(2024, 1, 1), datetime(2024, 1, 2))

    assert len(df) == 1
    assert df.loc[0, "symbol"] == "EURUSD"
    assert df.loc[0, "close"] == 1.12


def test_fetch_rates_rejects_unsupported_timeframe(monkeypatch):
    monkeypatch.setitem(sys.modules, "MetaTrader5", _fake_mt5_module())
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        fetch_rates("EURUSD", "W1", datetime(2024, 1, 1), datetime(2024, 1, 2))


def test_fetch_rates_raises_when_copy_rates_range_returns_none(monkeypatch):
    monkeypatch.setitem(sys.modules, "MetaTrader5", _fake_mt5_module(rates=None))
    with pytest.raises(MT5UnavailableError, match="copy_rates_range returned None"):
        fetch_rates("EURUSD", "H1", datetime(2024, 1, 1), datetime(2024, 1, 2))
