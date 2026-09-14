"""Mocked unit tests for the generic REST/WebSocket broker adapter
(CLAUDE.md Section 23): no real broker exists to connect to (the whole
point of "generic"), so every test stubs `requests.Session` with a fake
that answers from a canned response table, and the WebSocket half with a
fake async connector. This pins down GenericRestAdapter's own logic
(request building, response mapping, retry/rate-limit/reconciliation
behavior) -- it says nothing about how a specific real broker's API
actually looks, which isn't known yet.
"""
from __future__ import annotations

import asyncio
import json

import pytest
import requests

from src.brokers import generic_rest_adapter as gra_module
from src.brokers.base import (
    BrokerConnectionError,
    BrokerRejectionError,
    DuplicateOrderError,
    EmergencyStopActive,
    LiveTradingBlockedError,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from src.brokers.generic_rest_adapter import GenericRestAdapter, build_from_config
from src.core.config import BrokerEntryConfig, BrokersConfig


class _FakeResponse:
    def __init__(self, status_code, json_data=None):
        self.status_code = status_code
        self._json_data = json_data if json_data is not None else {}
        self.text = json.dumps(self._json_data)

    def json(self):
        return self._json_data


class _FakeSession:
    """responder(method, path, json_body) -> _FakeResponse; every call
    is recorded in .calls for assertions."""

    def __init__(self, responder):
        self.responder = responder
        self.calls: list[dict] = []
        self.closed = False

    def request(self, method, url, *, json=None, headers=None, timeout=None):
        path = url.split("://", 1)[-1].split("/", 1)[-1]
        path = "/" + path if not path.startswith("/") else path
        self.calls.append({"method": method, "path": path, "json": json, "headers": headers, "timeout": timeout})
        return self.responder(method, path, json)

    def close(self):
        self.closed = True


_DEFAULT_ACCOUNT = {
    "login": 12345, "name": "Test", "server": "demo-1", "currency": "EUR",
    "leverage": 30, "balance": 2000.0, "equity": 2010.0, "margin": 50.0, "margin_free": 1960.0,
}


def _table_responder(table: dict[tuple[str, str], _FakeResponse]):
    def _respond(method, path, json_body):
        key = (method, path)
        if key not in table:
            raise AssertionError(f"No fake response registered for {key}; registered: {list(table)}")
        return table[key]

    return _respond


def _make_adapter(monkeypatch, table: dict, *, connect=True, **kwargs):
    table = {("GET", "/account"): _FakeResponse(200, _DEFAULT_ACCOUNT), **table}
    session = _FakeSession(_table_responder(table))
    monkeypatch.setattr(gra_module.requests, "Session", lambda: session)
    adapter = GenericRestAdapter(
        api_base_url="http://fake-broker.test", retry_base_delay_seconds=0.001, requests_per_second=10_000,
        **kwargs,
    )
    if connect:
        adapter.connect()
    return adapter, session


# --- connection --------------------------------------------------------------


def test_connect_calls_account_endpoint_and_stores_session(monkeypatch):
    adapter, session = _make_adapter(monkeypatch, {})
    assert adapter._session is session
    assert session.calls[0]["method"] == "GET"
    assert session.calls[0]["path"] == "/account"


def test_disconnect_closes_session(monkeypatch):
    adapter, session = _make_adapter(monkeypatch, {})
    adapter.disconnect()
    assert session.closed is True
    assert adapter._session is None


def test_methods_raise_broker_connection_error_before_connect():
    adapter = GenericRestAdapter(api_base_url="http://fake-broker.test")
    with pytest.raises(BrokerConnectionError, match="call connect"):
        adapter.get_account()


def test_connect_raises_broker_connection_error_when_account_endpoint_keeps_failing(monkeypatch):
    def _always_500(method, path, json_body):
        return _FakeResponse(500)

    session = _FakeSession(_always_500)
    monkeypatch.setattr(gra_module.requests, "Session", lambda: session)
    adapter = GenericRestAdapter(api_base_url="http://fake-broker.test", max_retries=2, retry_base_delay_seconds=0.001)
    with pytest.raises(BrokerConnectionError, match="failed after 2 retries"):
        adapter.connect()


def test_request_retries_on_5xx_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def _flaky_account(method, path, json_body):
        calls["n"] += 1
        if calls["n"] < 3:
            return _FakeResponse(500)
        return _FakeResponse(200, _DEFAULT_ACCOUNT)

    session = _FakeSession(_flaky_account)
    monkeypatch.setattr(gra_module.requests, "Session", lambda: session)
    adapter = GenericRestAdapter(api_base_url="http://fake-broker.test", max_retries=5, retry_base_delay_seconds=0.001)
    adapter.connect()
    assert calls["n"] == 3


# --- account / positions / market data ---------------------------------------


def test_get_account_maps_fields(monkeypatch):
    adapter, _ = _make_adapter(monkeypatch, {})
    account = adapter.get_account()
    assert account.login == 12345
    assert account.currency == "EUR"
    assert account.balance == 2000.0


def test_get_balance_and_get_equity_delegate(monkeypatch):
    adapter, _ = _make_adapter(monkeypatch, {})
    assert adapter.get_balance() == 2000.0
    assert adapter.get_equity() == 2010.0


def test_get_positions_maps_a_list(monkeypatch):
    positions_payload = [
        {"position_id": "1", "symbol": "EURUSD", "side": "BUY", "volume": 0.1, "entry_price": 1.10,
         "current_price": 1.11, "stop_loss": 1.09, "take_profit": 1.15, "profit": 1.0, "open_time": "2024-01-01T00:00:00Z"},
        {"position_id": "2", "symbol": "GBPUSD", "side": "SELL", "volume": 0.2, "entry_price": 1.30,
         "current_price": 1.29, "stop_loss": None, "take_profit": None, "profit": 2.0, "open_time": "2024-01-02T00:00:00Z"},
    ]
    adapter, _ = _make_adapter(monkeypatch, {("GET", "/positions"): _FakeResponse(200, positions_payload)})
    positions = adapter.get_positions()
    assert len(positions) == 2
    assert positions[0].side == OrderSide.BUY
    assert positions[1].side == OrderSide.SELL
    assert positions[1].stop_loss is None


def test_get_symbol_info_maps_fields(monkeypatch):
    payload = {"digits": 5, "point": 0.00001, "contract_size": 100000.0, "volume_min": 0.01,
               "volume_max": 100.0, "volume_step": 0.01, "tick_value": 1.0, "tick_size": 0.00001, "spread": 15}
    adapter, _ = _make_adapter(monkeypatch, {("GET", "/symbols/EURUSD"): _FakeResponse(200, payload)})
    info = adapter.get_symbol_info("EURUSD")
    assert info.symbol == "EURUSD"
    assert info.contract_size == 100000.0


def test_get_quote_returns_fresh_quote(monkeypatch):
    from datetime import datetime, timezone

    now_iso = datetime.now(timezone.utc).isoformat()
    payload = {"bid": 1.1000, "ask": 1.1002, "timestamp": now_iso}
    adapter, _ = _make_adapter(monkeypatch, {("GET", "/quotes/EURUSD"): _FakeResponse(200, payload)})
    quote = adapter.get_quote("EURUSD")
    assert quote.bid == 1.1000
    assert quote.ask == 1.1002


def test_get_quote_raises_stale_price_error_when_too_old(monkeypatch):
    payload = {"bid": 1.1000, "ask": 1.1002, "timestamp": "2000-01-01T00:00:00Z"}
    adapter, _ = _make_adapter(
        monkeypatch, {("GET", "/quotes/EURUSD"): _FakeResponse(200, payload)}, stale_quote_max_age_seconds=5.0,
    )
    with pytest.raises(gra_module.StalePriceError):
        adapter.get_quote("EURUSD")


# --- place_order ---------------------------------------------------------------


def test_place_order_market_buy_success(monkeypatch):
    response = {"order_id": "999", "status": "FILLED", "filled_volume": 0.1, "message": "ok"}
    adapter, session = _make_adapter(monkeypatch, {("POST", "/orders"): _FakeResponse(200, response)})
    result = adapter.place_order(OrderRequest(symbol="EURUSD", side=OrderSide.BUY, order_type=OrderType.MARKET, volume=0.1))
    assert result.success is True
    assert result.status == OrderStatus.FILLED
    assert result.order_id == "999"

    body = session.calls[-1]["json"]
    assert body["symbol"] == "EURUSD"
    assert body["side"] == "BUY"
    assert body["order_type"] == "MARKET"
    assert body["volume"] == 0.1


def test_place_order_partial_fill_downgrades_status(monkeypatch):
    response = {"order_id": "1", "status": "FILLED", "filled_volume": 0.05}
    adapter, _ = _make_adapter(monkeypatch, {("POST", "/orders"): _FakeResponse(200, response)})
    result = adapter.place_order(OrderRequest(symbol="EURUSD", side=OrderSide.BUY, order_type=OrderType.MARKET, volume=0.1))
    assert result.status == OrderStatus.PARTIALLY_FILLED
    assert result.success is True


def test_place_order_rejected_status_maps_to_failure(monkeypatch):
    response = {"order_id": None, "status": "REJECTED", "message": "insufficient margin"}
    adapter, _ = _make_adapter(monkeypatch, {("POST", "/orders"): _FakeResponse(200, response)})
    result = adapter.place_order(OrderRequest(symbol="EURUSD", side=OrderSide.BUY, order_type=OrderType.MARKET, volume=0.1))
    assert result.success is False
    assert result.status == OrderStatus.REJECTED


def test_place_order_duplicate_client_order_id_refused_without_a_second_http_call(monkeypatch):
    response = {"order_id": "1", "status": "FILLED", "filled_volume": 0.1}
    adapter, session = _make_adapter(monkeypatch, {("POST", "/orders"): _FakeResponse(200, response)})
    order = OrderRequest(symbol="EURUSD", side=OrderSide.BUY, order_type=OrderType.MARKET, volume=0.1, client_order_id="abc")
    adapter.place_order(order)
    n_calls_after_first = len(session.calls)
    with pytest.raises(DuplicateOrderError):
        adapter.place_order(order)
    assert len(session.calls) == n_calls_after_first  # refused before any HTTP call


def test_place_order_raises_after_emergency_stop(monkeypatch):
    adapter, _ = _make_adapter(monkeypatch, {})
    adapter.emergency_stop()
    with pytest.raises(EmergencyStopActive):
        adapter.place_order(OrderRequest(symbol="EURUSD", side=OrderSide.BUY, order_type=OrderType.MARKET, volume=0.1))


def test_place_order_blocked_on_live_environment_without_both_flags(monkeypatch):
    monkeypatch.delenv("LIVE_TRADING", raising=False)
    monkeypatch.delenv("LIVE_CONFIRMATION", raising=False)
    adapter, _ = _make_adapter(monkeypatch, {}, environment="live")
    with pytest.raises(LiveTradingBlockedError):
        adapter.place_order(OrderRequest(symbol="EURUSD", side=OrderSide.BUY, order_type=OrderType.MARKET, volume=0.1))


def test_place_order_allowed_on_live_environment_with_both_flags_true(monkeypatch):
    monkeypatch.setenv("LIVE_TRADING", "true")
    monkeypatch.setenv("LIVE_CONFIRMATION", "true")
    response = {"order_id": "1", "status": "FILLED", "filled_volume": 0.1}
    adapter, _ = _make_adapter(monkeypatch, {("POST", "/orders"): _FakeResponse(200, response)}, environment="live")
    result = adapter.place_order(OrderRequest(symbol="EURUSD", side=OrderSide.BUY, order_type=OrderType.MARKET, volume=0.1))
    assert result.success is True


def test_place_order_always_allowed_on_demo_regardless_of_live_flags(monkeypatch):
    monkeypatch.delenv("LIVE_TRADING", raising=False)
    monkeypatch.delenv("LIVE_CONFIRMATION", raising=False)
    response = {"order_id": "1", "status": "FILLED", "filled_volume": 0.1}
    adapter, _ = _make_adapter(monkeypatch, {("POST", "/orders"): _FakeResponse(200, response)}, environment="demo")
    result = adapter.place_order(OrderRequest(symbol="EURUSD", side=OrderSide.BUY, order_type=OrderType.MARKET, volume=0.1))
    assert result.success is True


def test_place_order_4xx_raises_broker_rejection_error(monkeypatch):
    adapter, _ = _make_adapter(monkeypatch, {("POST", "/orders"): _FakeResponse(422, {"message": "bad request"})})
    with pytest.raises(BrokerRejectionError):
        adapter.place_order(OrderRequest(symbol="EURUSD", side=OrderSide.BUY, order_type=OrderType.MARKET, volume=0.1))


# --- modify / cancel / close ------------------------------------------------


def test_modify_order_sends_only_provided_fields(monkeypatch):
    adapter, session = _make_adapter(monkeypatch, {("PATCH", "/orders/42"): _FakeResponse(200, {"status": "PENDING"})})
    adapter.modify_order("42", stop_loss=1.05)
    body = session.calls[-1]["json"]
    assert body == {"stop_loss": 1.05}


def test_cancel_order_sends_delete(monkeypatch):
    adapter, session = _make_adapter(monkeypatch, {("DELETE", "/orders/42"): _FakeResponse(200, {"status": "CANCELLED"})})
    result = adapter.cancel_order("42")
    assert session.calls[-1]["method"] == "DELETE"
    assert result.status == OrderStatus.CANCELLED


def test_close_position_sends_post_with_volume(monkeypatch):
    adapter, session = _make_adapter(
        monkeypatch, {("POST", "/positions/555/close"): _FakeResponse(200, {"order_id": "1", "status": "FILLED", "filled_volume": 0.05})},
    )
    adapter.close_position("555", volume=0.05)
    assert session.calls[-1]["path"] == "/positions/555/close"
    assert session.calls[-1]["json"] == {"volume": 0.05}


def test_modify_cancel_close_respect_emergency_stop(monkeypatch):
    adapter, _ = _make_adapter(monkeypatch, {})
    adapter.emergency_stop()
    with pytest.raises(EmergencyStopActive):
        adapter.modify_order("1", stop_loss=1.0)
    with pytest.raises(EmergencyStopActive):
        adapter.cancel_order("1")
    with pytest.raises(EmergencyStopActive):
        adapter.close_position("1")


def test_get_order_status_maps_known_value(monkeypatch):
    adapter, _ = _make_adapter(monkeypatch, {("GET", "/orders/1"): _FakeResponse(200, {"status": "PENDING"})})
    assert adapter.get_order_status("1") == OrderStatus.PENDING


def test_get_order_status_falls_back_to_unknown_for_unrecognized_value(monkeypatch):
    adapter, _ = _make_adapter(monkeypatch, {("GET", "/orders/1"): _FakeResponse(200, {"status": "SOMETHING_WEIRD"})})
    assert adapter.get_order_status("1") == OrderStatus.UNKNOWN


# --- reconciliation ------------------------------------------------------------


def test_reconcile_compares_local_against_get_positions(monkeypatch):
    broker_position_payload = [
        {"position_id": "1", "symbol": "EURUSD", "side": "BUY", "volume": 0.1, "entry_price": 1.10,
         "current_price": 1.11, "stop_loss": None, "take_profit": None, "profit": 1.0, "open_time": "2024-01-01T00:00:00Z"},
    ]
    adapter, _ = _make_adapter(monkeypatch, {("GET", "/positions"): _FakeResponse(200, broker_position_payload)})
    report = adapter.reconcile([])
    assert len(report.missing_locally) == 1
    assert report.missing_locally[0].position_id == "1"


# --- WebSocket quote stream ----------------------------------------------------


class _FakeWebSocket:
    def __init__(self, messages):
        self._messages = list(messages)
        self.sent: list[str] = []

    async def send(self, message):
        self.sent.append(message)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._messages:
            raise StopAsyncIteration
        return self._messages.pop(0)


class _FakeConnector:
    def __init__(self, messages):
        self.ws = _FakeWebSocket(messages)
        self.requested_url = None

    def __call__(self, url):
        self.requested_url = url
        return self

    async def __aenter__(self):
        return self.ws

    async def __aexit__(self, *exc_info):
        return False


async def _collect(agen, limit):
    out = []
    async for item in agen:
        out.append(item)
        if len(out) >= limit:
            break
    return out


def test_stream_quotes_yields_parsed_quotes_and_sends_subscription():
    messages = [
        json.dumps({"symbol": "EURUSD", "bid": 1.10, "ask": 1.1002, "timestamp": "2024-01-01T00:00:00Z"}),
        json.dumps({"symbol": "EURUSD", "bid": 1.11, "ask": 1.1102, "timestamp": "2024-01-01T00:00:01Z"}),
    ]
    connector = _FakeConnector(messages)
    adapter = GenericRestAdapter(api_base_url="http://fake-broker.test", websocket_url="ws://fake-broker.test/stream")

    quotes = asyncio.run(_collect(adapter.stream_quotes(["EURUSD"], connector=connector), 2))

    assert [q.bid for q in quotes] == [1.10, 1.11]
    assert connector.requested_url == "ws://fake-broker.test/stream"
    assert json.loads(connector.ws.sent[0]) == {"action": "subscribe", "symbols": ["EURUSD"]}


def test_stream_quotes_skips_non_quote_messages():
    messages = [
        json.dumps({"event": "subscribed"}),  # not a quote -- must be skipped, not fabricated into one
        json.dumps({"symbol": "EURUSD", "bid": 1.10, "ask": 1.1002, "timestamp": "2024-01-01T00:00:00Z"}),
    ]
    connector = _FakeConnector(messages)
    adapter = GenericRestAdapter(api_base_url="http://fake-broker.test", websocket_url="ws://fake-broker.test/stream")

    quotes = asyncio.run(_collect(adapter.stream_quotes(["EURUSD"], connector=connector), 1))
    assert len(quotes) == 1
    assert quotes[0].symbol == "EURUSD"


def test_stream_quotes_raises_when_websocket_url_not_configured():
    adapter = GenericRestAdapter(api_base_url="http://fake-broker.test")

    async def _run():
        async for _ in adapter.stream_quotes(["EURUSD"], connector=_FakeConnector([])):
            pass

    with pytest.raises(BrokerConnectionError, match="websocket_url"):
        asyncio.run(_run())


# --- build_from_config ----------------------------------------------------------


def test_build_from_config_raises_when_disabled():
    cfg = BrokersConfig(brokers={"generic_rest": BrokerEntryConfig(enabled=False)}, active_broker=None)
    with pytest.raises(BrokerConnectionError, match="enabled is false"):
        build_from_config(cfg)


def test_build_from_config_raises_when_api_base_url_missing():
    cfg = BrokersConfig(
        brokers={"generic_rest": BrokerEntryConfig(enabled=True, api_base_url=None)}, active_broker="generic_rest",
    )
    with pytest.raises(BrokerConnectionError, match="api_base_url"):
        build_from_config(cfg)


def test_build_from_config_reads_env_and_yaml_fields(monkeypatch):
    monkeypatch.setenv("BROKER_API_KEY", "key123")
    monkeypatch.setenv("BROKER_API_SECRET", "secret456")
    monkeypatch.setenv("BROKER_ACCOUNT_ID", "acct789")
    cfg = BrokersConfig(
        brokers={
            "generic_rest": BrokerEntryConfig(
                enabled=True, environment="demo", api_base_url="https://api.example.test",
                websocket_url="wss://api.example.test/stream",
                api_key_env="BROKER_API_KEY", api_secret_env="BROKER_API_SECRET", account_id_env="BROKER_ACCOUNT_ID",
                rate_limit={"requests_per_second": 8}, timeout_seconds=20, max_retries=5,
                retry_base_delay_seconds=2.0, stale_quote_max_age_seconds=10.0,
            )
        },
        active_broker="generic_rest",
    )
    adapter = build_from_config(cfg)
    assert adapter.api_base_url == "https://api.example.test"
    assert adapter.api_key == "key123"
    assert adapter.api_secret == "secret456"
    assert adapter.account_id == "acct789"
    assert adapter.websocket_url == "wss://api.example.test/stream"
    assert adapter.timeout_seconds == 20
    assert adapter.max_retries == 5
    assert adapter.retry_base_delay_seconds == 2.0
    assert adapter.stale_quote_max_age_seconds == 10.0
