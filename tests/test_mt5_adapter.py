"""Mocked unit tests for the MT5 broker adapter (CLAUDE.md Section 22):
the `MetaTrader5` package cannot be installed/imported here, so every
test stubs `sys.modules["MetaTrader5"]` with a fake module exposing just
enough of the real API surface (namedtuples/namespaces + constants) to
exercise the adapter's own logic. None of this proves the real MT5
package behaves identically — only a real terminal on Windows can (see
CLAUDE.md Section 22) — these tests only pin down MT5Adapter's behavior
against whatever MetaTrader5 hands it back.
"""
from __future__ import annotations

import sys
import types
from collections import namedtuple
from datetime import datetime, timezone

import pytest

from src.brokers.base import (
    BrokerConnectionError,
    EmergencyStopActive,
    LiveTradingBlockedError,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
)
from src.brokers.mt5_adapter import MT5Adapter, _client_order_id_to_magic, build_from_config
from src.core.config import BrokerEntryConfig, BrokersConfig

OrderSendResult = namedtuple("OrderSendResult", ["retcode", "order", "comment", "deal", "volume", "price"])


def _account_info(**overrides):
    defaults = dict(
        login=12345, name="Test Account", server="Demo-Server", currency="EUR",
        leverage=30, balance=2000.0, equity=2010.0, margin=50.0, margin_free=1960.0,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def _symbol_info(**overrides):
    defaults = dict(
        digits=5, point=0.00001, trade_contract_size=100000.0, volume_min=0.01,
        volume_max=100.0, volume_step=0.01, trade_tick_value=1.0, trade_tick_size=0.00001, spread=15,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def _tick(**overrides):
    defaults = dict(bid=1.1000, ask=1.1002, time=1704067200)
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def _position(**overrides):
    defaults = dict(
        ticket=555, symbol="EURUSD", type=0, volume=0.1, price_open=1.1000, price_current=1.1050,
        sl=1.0950, tp=1.1100, profit=5.0, time=1704067200,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def _fake_mt5_module(*, initialize_returns=True, order_send_retcode=10009, positions=None, orders=None, history=None):
    mod = types.ModuleType("MetaTrader5")

    mod.ORDER_TYPE_BUY = 0
    mod.ORDER_TYPE_SELL = 1
    mod.ORDER_TYPE_BUY_LIMIT = 2
    mod.ORDER_TYPE_SELL_LIMIT = 3
    mod.ORDER_TYPE_BUY_STOP = 4
    mod.ORDER_TYPE_SELL_STOP = 5
    mod.TRADE_ACTION_DEAL = 1
    mod.TRADE_ACTION_PENDING = 5
    mod.TRADE_ACTION_SLTP = 6
    mod.TRADE_ACTION_MODIFY = 7
    mod.TRADE_ACTION_REMOVE = 8
    mod.ORDER_TIME_GTC = 0
    mod.ORDER_FILLING_IOC = 1
    mod.TRADE_RETCODE_DONE = 10009

    mod.initialize = lambda **kwargs: initialize_returns
    mod.shutdown = lambda: None
    mod.last_error = lambda: (1, "mock error")

    mod.account_info = lambda: _account_info()
    mod.symbol_info = lambda symbol: _symbol_info()
    mod.symbol_info_tick = lambda symbol: _tick()
    mod.positions_get = lambda **kwargs: positions if positions is not None else ()
    mod.orders_get = lambda **kwargs: orders if orders is not None else ()
    mod.history_orders_get = lambda **kwargs: history if history is not None else ()

    captured_requests: list[dict] = []

    def _order_send(request):
        captured_requests.append(request)
        return OrderSendResult(retcode=order_send_retcode, order=999, comment="ok", deal=1, volume=request.get("volume", 0.0), price=request.get("price", 0.0))

    mod.order_send = _order_send
    mod._captured_requests = captured_requests
    return mod


def _connected_adapter(monkeypatch, mt5_module, *, environment="demo"):
    monkeypatch.setitem(sys.modules, "MetaTrader5", mt5_module)
    adapter = MT5Adapter(login=1, password="pw", server="srv", environment=environment)
    adapter.connect()
    return adapter


# --- connection --------------------------------------------------------------


def test_connect_stores_handle_and_disconnect_clears_it(monkeypatch):
    adapter = _connected_adapter(monkeypatch, _fake_mt5_module())
    assert adapter._mt5 is not None
    adapter.disconnect()
    assert adapter._mt5 is None


def test_methods_raise_broker_connection_error_before_connect():
    adapter = MT5Adapter()
    with pytest.raises(BrokerConnectionError, match="call connect"):
        adapter.get_account()


# --- account -------------------------------------------------------------


def test_get_account_maps_fields(monkeypatch):
    adapter = _connected_adapter(monkeypatch, _fake_mt5_module())
    account = adapter.get_account()
    assert account.login == 12345
    assert account.currency == "EUR"
    assert account.balance == 2000.0
    assert account.equity == 2010.0


def test_get_balance_and_get_equity_delegate_to_get_account(monkeypatch):
    adapter = _connected_adapter(monkeypatch, _fake_mt5_module())
    assert adapter.get_balance() == 2000.0
    assert adapter.get_equity() == 2010.0


def test_get_account_raises_when_account_info_is_none(monkeypatch):
    mod = _fake_mt5_module()
    mod.account_info = lambda: None
    adapter = _connected_adapter(monkeypatch, mod)
    with pytest.raises(BrokerConnectionError, match="account_info"):
        adapter.get_account()


# --- positions -----------------------------------------------------------


def test_get_positions_maps_buy_and_sell_sides(monkeypatch):
    buy = _position(ticket=1, type=0)
    sell = _position(ticket=2, type=1, sl=None, tp=None)
    adapter = _connected_adapter(monkeypatch, _fake_mt5_module(positions=(buy, sell)))
    positions = adapter.get_positions()
    assert len(positions) == 2
    assert positions[0].side == OrderSide.BUY
    assert positions[0].position_id == "1"
    assert positions[1].side == OrderSide.SELL
    assert positions[1].stop_loss is None
    assert positions[1].take_profit is None


def test_get_positions_returns_empty_list_when_none_open(monkeypatch):
    adapter = _connected_adapter(monkeypatch, _fake_mt5_module(positions=()))
    assert adapter.get_positions() == []


# --- symbol info / quotes -------------------------------------------------


def test_get_symbol_info_maps_fields(monkeypatch):
    adapter = _connected_adapter(monkeypatch, _fake_mt5_module())
    info = adapter.get_symbol_info("EURUSD")
    assert info.symbol == "EURUSD"
    assert info.contract_size == 100000.0
    assert info.volume_step == 0.01


def test_get_symbol_info_raises_when_none(monkeypatch):
    mod = _fake_mt5_module()
    mod.symbol_info = lambda symbol: None
    adapter = _connected_adapter(monkeypatch, mod)
    with pytest.raises(BrokerConnectionError, match="symbol_info"):
        adapter.get_symbol_info("EURUSD")


def test_get_quote_maps_fields(monkeypatch):
    adapter = _connected_adapter(monkeypatch, _fake_mt5_module())
    quote = adapter.get_quote("EURUSD")
    assert quote.bid == 1.1000
    assert quote.ask == 1.1002
    assert quote.timestamp == datetime.fromtimestamp(1704067200, tz=timezone.utc)


def test_get_quote_raises_when_none(monkeypatch):
    mod = _fake_mt5_module()
    mod.symbol_info_tick = lambda symbol: None
    adapter = _connected_adapter(monkeypatch, mod)
    with pytest.raises(BrokerConnectionError, match="symbol_info_tick"):
        adapter.get_quote("EURUSD")


# --- place_order -----------------------------------------------------------


def test_place_order_market_buy_success(monkeypatch):
    adapter = _connected_adapter(monkeypatch, _fake_mt5_module())
    result = adapter.place_order(OrderRequest(symbol="EURUSD", side=OrderSide.BUY, order_type=OrderType.MARKET, volume=0.1))
    assert result.success is True
    assert result.status == OrderStatus.FILLED
    assert result.order_id == "999"


def test_place_order_rejected_when_retcode_not_done(monkeypatch):
    adapter = _connected_adapter(monkeypatch, _fake_mt5_module(order_send_retcode=10004))
    result = adapter.place_order(OrderRequest(symbol="EURUSD", side=OrderSide.BUY, order_type=OrderType.MARKET, volume=0.1))
    assert result.success is False
    assert result.status == OrderStatus.REJECTED


def test_place_order_builds_pending_limit_request_with_price(monkeypatch):
    mod = _fake_mt5_module()
    adapter = _connected_adapter(monkeypatch, mod)
    adapter.place_order(
        OrderRequest(symbol="EURUSD", side=OrderSide.SELL, order_type=OrderType.LIMIT, volume=0.2, price=1.2000,
                     stop_loss=1.2100, take_profit=1.1800)
    )
    request = mod._captured_requests[-1]
    assert request["type"] == mod.ORDER_TYPE_SELL_LIMIT
    assert request["action"] == mod.TRADE_ACTION_PENDING
    assert request["price"] == 1.2000
    assert request["sl"] == 1.2100
    assert request["tp"] == 1.1800


def test_place_order_maps_client_order_id_to_a_stable_deterministic_magic(monkeypatch):
    mod = _fake_mt5_module()
    adapter = _connected_adapter(monkeypatch, mod)
    order = OrderRequest(symbol="EURUSD", side=OrderSide.BUY, order_type=OrderType.MARKET, volume=0.1, client_order_id="abc-123")
    adapter.place_order(order)
    magic = mod._captured_requests[-1]["magic"]
    assert magic == _client_order_id_to_magic("abc-123")
    # same key -> same magic every call (unlike Python's own hash())
    assert _client_order_id_to_magic("abc-123") == _client_order_id_to_magic("abc-123")


def test_place_order_raises_after_emergency_stop(monkeypatch):
    adapter = _connected_adapter(monkeypatch, _fake_mt5_module())
    adapter.emergency_stop()
    with pytest.raises(EmergencyStopActive):
        adapter.place_order(OrderRequest(symbol="EURUSD", side=OrderSide.BUY, order_type=OrderType.MARKET, volume=0.1))


def test_place_order_allowed_again_after_reset_emergency_stop(monkeypatch):
    adapter = _connected_adapter(monkeypatch, _fake_mt5_module())
    adapter.emergency_stop()
    adapter.reset_emergency_stop()
    result = adapter.place_order(OrderRequest(symbol="EURUSD", side=OrderSide.BUY, order_type=OrderType.MARKET, volume=0.1))
    assert result.success is True


def test_place_order_blocked_on_live_environment_without_both_flags(monkeypatch):
    monkeypatch.delenv("LIVE_TRADING", raising=False)
    monkeypatch.delenv("LIVE_CONFIRMATION", raising=False)
    adapter = _connected_adapter(monkeypatch, _fake_mt5_module(), environment="live")
    with pytest.raises(LiveTradingBlockedError):
        adapter.place_order(OrderRequest(symbol="EURUSD", side=OrderSide.BUY, order_type=OrderType.MARKET, volume=0.1))


def test_place_order_allowed_on_live_environment_with_both_flags_true(monkeypatch):
    monkeypatch.setenv("LIVE_TRADING", "true")
    monkeypatch.setenv("LIVE_CONFIRMATION", "true")
    adapter = _connected_adapter(monkeypatch, _fake_mt5_module(), environment="live")
    result = adapter.place_order(OrderRequest(symbol="EURUSD", side=OrderSide.BUY, order_type=OrderType.MARKET, volume=0.1))
    assert result.success is True


def test_place_order_always_allowed_on_demo_regardless_of_live_flags(monkeypatch):
    monkeypatch.delenv("LIVE_TRADING", raising=False)
    monkeypatch.delenv("LIVE_CONFIRMATION", raising=False)
    adapter = _connected_adapter(monkeypatch, _fake_mt5_module(), environment="demo")
    result = adapter.place_order(OrderRequest(symbol="EURUSD", side=OrderSide.BUY, order_type=OrderType.MARKET, volume=0.1))
    assert result.success is True


# --- modify / cancel / close ------------------------------------------------


def test_modify_order_sets_sl_and_tp(monkeypatch):
    mod = _fake_mt5_module()
    adapter = _connected_adapter(monkeypatch, mod)
    adapter.modify_order("42", stop_loss=1.05, take_profit=1.20)
    request = mod._captured_requests[-1]
    assert request["action"] == mod.TRADE_ACTION_SLTP
    assert request["position"] == 42
    assert request["sl"] == 1.05
    assert request["tp"] == 1.20


def test_modify_order_with_price_uses_modify_action(monkeypatch):
    mod = _fake_mt5_module()
    adapter = _connected_adapter(monkeypatch, mod)
    adapter.modify_order("42", price=1.15)
    request = mod._captured_requests[-1]
    assert request["action"] == mod.TRADE_ACTION_MODIFY
    assert request["price"] == 1.15


def test_modify_order_respects_emergency_stop(monkeypatch):
    adapter = _connected_adapter(monkeypatch, _fake_mt5_module())
    adapter.emergency_stop()
    with pytest.raises(EmergencyStopActive):
        adapter.modify_order("42", stop_loss=1.05)


def test_cancel_order_sends_remove_action(monkeypatch):
    mod = _fake_mt5_module()
    adapter = _connected_adapter(monkeypatch, mod)
    adapter.cancel_order("77")
    request = mod._captured_requests[-1]
    assert request["action"] == mod.TRADE_ACTION_REMOVE
    assert request["order"] == 77


def test_close_position_uses_opposite_side_and_full_volume(monkeypatch):
    position = _position(ticket=555, type=0, volume=0.3)
    mod = _fake_mt5_module(positions=(position,))
    adapter = _connected_adapter(monkeypatch, mod)
    result = adapter.close_position("555")
    request = mod._captured_requests[-1]
    assert request["type"] == mod.ORDER_TYPE_SELL  # was a buy (type=0) -> close with a sell
    assert request["volume"] == 0.3
    assert request["position"] == 555
    assert result.success is True


def test_close_position_respects_partial_volume_override(monkeypatch):
    position = _position(ticket=555, type=1, volume=1.0)  # a sell position -> closes with a buy
    mod = _fake_mt5_module(positions=(position,))
    adapter = _connected_adapter(monkeypatch, mod)
    adapter.close_position("555", volume=0.25)
    request = mod._captured_requests[-1]
    assert request["type"] == mod.ORDER_TYPE_BUY
    assert request["volume"] == 0.25


def test_close_position_returns_failure_when_position_not_found(monkeypatch):
    adapter = _connected_adapter(monkeypatch, _fake_mt5_module(positions=()))
    result = adapter.close_position("999")
    assert result.success is False
    assert result.status == OrderStatus.UNKNOWN


def test_close_position_respects_emergency_stop(monkeypatch):
    position = _position(ticket=555, type=0, volume=0.3)
    adapter = _connected_adapter(monkeypatch, _fake_mt5_module(positions=(position,)))
    adapter.emergency_stop()
    with pytest.raises(EmergencyStopActive):
        adapter.close_position("555")


# --- order status ------------------------------------------------------------


def test_get_order_status_pending_when_still_open(monkeypatch):
    order_stub = types.SimpleNamespace(ticket=1)
    adapter = _connected_adapter(monkeypatch, _fake_mt5_module(orders=(order_stub,)))
    assert adapter.get_order_status("1") == OrderStatus.PENDING


def test_get_order_status_filled_when_in_history_and_not_pending(monkeypatch):
    history_stub = types.SimpleNamespace(ticket=1)
    adapter = _connected_adapter(monkeypatch, _fake_mt5_module(orders=(), history=(history_stub,)))
    assert adapter.get_order_status("1") == OrderStatus.FILLED


def test_get_order_status_unknown_when_neither_pending_nor_in_history(monkeypatch):
    adapter = _connected_adapter(monkeypatch, _fake_mt5_module(orders=(), history=()))
    assert adapter.get_order_status("1") == OrderStatus.UNKNOWN


# --- build_from_config -------------------------------------------------------


def test_build_from_config_raises_when_mt5_disabled():
    cfg = BrokersConfig(brokers={"mt5": BrokerEntryConfig(enabled=False)}, active_broker=None)
    from src.data.mt5_loader import MT5UnavailableError

    with pytest.raises(MT5UnavailableError, match="enabled is false"):
        build_from_config(cfg)


def test_build_from_config_reads_credentials_from_env(monkeypatch):
    monkeypatch.setenv("MT5_LOGIN", "555111")
    monkeypatch.setenv("MT5_PASSWORD", "secret")
    monkeypatch.setenv("MT5_SERVER", "Broker-Demo")
    cfg = BrokersConfig(
        brokers={"mt5": BrokerEntryConfig(enabled=True, environment="demo", terminal_path_env="MT5_TERMINAL_PATH")},
        active_broker="mt5",
    )
    adapter = build_from_config(cfg)
    assert adapter.login == 555111
    assert adapter.password == "secret"
    assert adapter.server == "Broker-Demo"
    assert adapter.environment == "demo"
