"""Generic REST/WebSocket broker adapter (CLAUDE.md Sections 23, 24, 25).

No real broker is named in the brief, so this adapter defines its own
small, self-describing REST convention (GET /account, /positions,
/symbols/{symbol}, /quotes/{symbol}; POST/PATCH/DELETE /orders...;
POST /positions/{id}/close) rather than guessing a specific broker's
actual schema — plugging in a real broker later means translating its
responses onto the same `AccountInfo`/`Position`/`SymbolInfo`/`Quote`/
`OrderResult` dataclasses this adapter already produces, at which point
this file's `_to_*` methods are exactly what needs adjusting, and
strategy/risk/execution code (which only sees `BrokerAdapter`) needs no
changes at all.

Section 25's API-safety list, all present:
- **Rate limiting** — `src.brokers.resilience.RateLimiter`, applied to
  every request.
- **Retry with exponential backoff** — `retry_with_backoff()`, on
  connection errors/timeouts/5xx.
- **Timeout** — `requests`' own `timeout=` on every call.
- **Connection recovery** — a fresh `requests.Session` is created on
  `connect()`; retries reuse it, so a transient failure doesn't need a
  brand-new session to recover.
- **Duplicate-order protection / idempotency** — a `client_order_id` is
  refused if already submitted in-process, and passed to the broker too
  so it can dedupe server-side "where supported" (Section 25's own
  wording — a generic adapter can't assume every broker honors it).
- **Order-state reconciliation** — `reconcile()` /
  `src.brokers.base.reconcile_positions()`.
- **Broker rejection handling** — HTTP 4xx raises `BrokerRejectionError`
  (distinct from `BrokerConnectionError`, which means "couldn't trust the
  broker's response at all").
- **Partial-fill handling** — `_to_order_result()` downgrades a
  "FILLED" status to `PARTIALLY_FILLED` when `filled_volume` is less
  than what was requested.
- **Stale-price detection** — `get_quote()` raises `StalePriceError` if
  the quote is older than `stale_quote_max_age_seconds`.

The WebSocket half (`stream_quotes()`) is a minimal, honestly-scoped
addition: it proves the adapter can open, subscribe on, and consume a
streaming connection generically (via an injectable connector, default
`websockets.connect`), but it is **not** wired into `get_quote()` or
anywhere else yet — turning a quote stream into what paper trading
(Phase 14) consumes is that phase's job, not this one's.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import AsyncIterator, Callable

import requests

from src.brokers.base import (
    AccountInfo,
    BrokerAdapter,
    BrokerConnectionError,
    BrokerRejectionError,
    DuplicateOrderError,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    Position,
    Quote,
    StalePriceError,
    SymbolInfo,
    reconcile_positions,
    ReconciliationReport,
)
from src.brokers.resilience import RateLimiter, retry_with_backoff
from src.core.config import BrokersConfig, load_brokers


def _parse_timestamp(value) -> datetime:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


class GenericRestAdapter(BrokerAdapter):
    def __init__(
        self,
        *,
        api_base_url: str,
        api_key: str | None = None,
        api_secret: str | None = None,
        account_id: str | None = None,
        websocket_url: str | None = None,
        environment: str = "demo",
        requests_per_second: float = 5.0,
        timeout_seconds: float = 10.0,
        max_retries: int = 3,
        retry_base_delay_seconds: float = 1.0,
        stale_quote_max_age_seconds: float = 5.0,
    ) -> None:
        super().__init__(environment=environment)
        self.api_base_url = api_base_url.rstrip("/")
        self.api_key = api_key
        self.api_secret = api_secret
        self.account_id = account_id
        self.websocket_url = websocket_url
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_base_delay_seconds = retry_base_delay_seconds
        self.stale_quote_max_age_seconds = stale_quote_max_age_seconds
        self._rate_limiter = RateLimiter(requests_per_second)
        self._session: requests.Session | None = None
        self._seen_client_order_ids: set[str] = set()

    # -- connection ---------------------------------------------------------

    def connect(self) -> None:
        self._session = requests.Session()
        self._request("GET", "/account")  # verify reachability before declaring "connected"

    def disconnect(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

    def _require_connection(self) -> requests.Session:
        if self._session is None:
            raise BrokerConnectionError("Not connected — call connect() first.")
        return self._session

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        if self.api_secret:
            headers["X-API-Secret"] = self.api_secret
        return headers

    def _request(self, method: str, path: str, *, json_body: dict | None = None):
        session = self._require_connection()
        self._rate_limiter.acquire()

        def _send() -> requests.Response:
            response = session.request(
                method, f"{self.api_base_url}{path}", json=json_body,
                headers=self._headers(), timeout=self.timeout_seconds,
            )
            if response.status_code >= 500:
                raise requests.exceptions.ConnectionError(f"{method} {path} -> HTTP {response.status_code}")
            return response

        try:
            response = retry_with_backoff(
                _send, max_retries=self.max_retries, base_delay_seconds=self.retry_base_delay_seconds,
                retriable_exceptions=(requests.exceptions.ConnectionError, requests.exceptions.Timeout),
            )
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            raise BrokerConnectionError(
                f"{method} {path} failed after {self.max_retries} retries: {exc}"
            ) from exc

        if 400 <= response.status_code < 500:
            raise BrokerRejectionError(f"{method} {path} rejected: HTTP {response.status_code} {response.text}")

        return response.json()

    # -- account / positions / market data -----------------------------------

    def get_account(self) -> AccountInfo:
        data = self._request("GET", "/account")
        return AccountInfo(
            login=int(data["login"]), name=str(data["name"]), server=str(data["server"]),
            currency=str(data["currency"]), leverage=float(data["leverage"]), balance=float(data["balance"]),
            equity=float(data["equity"]), margin=float(data["margin"]), margin_free=float(data["margin_free"]),
        )

    def get_balance(self) -> float:
        return self.get_account().balance

    def get_equity(self) -> float:
        return self.get_account().equity

    def get_positions(self) -> list[Position]:
        data = self._request("GET", "/positions")
        return [self._to_position(p) for p in data]

    @staticmethod
    def _to_position(p: dict) -> Position:
        return Position(
            position_id=str(p["position_id"]), symbol=p["symbol"], side=OrderSide(p["side"]),
            volume=float(p["volume"]), entry_price=float(p["entry_price"]),
            current_price=float(p["current_price"]),
            stop_loss=float(p["stop_loss"]) if p.get("stop_loss") is not None else None,
            take_profit=float(p["take_profit"]) if p.get("take_profit") is not None else None,
            profit=float(p["profit"]), open_time=_parse_timestamp(p["open_time"]),
        )

    def get_symbol_info(self, symbol: str) -> SymbolInfo:
        data = self._request("GET", f"/symbols/{symbol}")
        return SymbolInfo(
            symbol=symbol, digits=int(data["digits"]), point=float(data["point"]),
            contract_size=float(data["contract_size"]), volume_min=float(data["volume_min"]),
            volume_max=float(data["volume_max"]), volume_step=float(data["volume_step"]),
            tick_value=float(data["tick_value"]), tick_size=float(data["tick_size"]), spread=float(data["spread"]),
        )

    def get_quote(self, symbol: str) -> Quote:
        data = self._request("GET", f"/quotes/{symbol}")
        quote = Quote(symbol=symbol, bid=float(data["bid"]), ask=float(data["ask"]),
                       timestamp=_parse_timestamp(data["timestamp"]))
        age_seconds = (datetime.now(timezone.utc) - quote.timestamp).total_seconds()
        if age_seconds > self.stale_quote_max_age_seconds:
            raise StalePriceError(
                f"Quote for {symbol} is {age_seconds:.1f}s old (max {self.stale_quote_max_age_seconds}s) — "
                "refusing to trust it."
            )
        return quote

    # -- orders ---------------------------------------------------------------

    def place_order(self, order: OrderRequest) -> OrderResult:
        self._ensure_order_allowed()
        if order.client_order_id:
            if order.client_order_id in self._seen_client_order_ids:
                raise DuplicateOrderError(f"client_order_id {order.client_order_id!r} was already submitted.")
            self._seen_client_order_ids.add(order.client_order_id)

        body = {
            "symbol": order.symbol, "side": order.side.value, "order_type": order.order_type.value,
            "volume": order.volume, "price": order.price, "stop_loss": order.stop_loss,
            "take_profit": order.take_profit, "client_order_id": order.client_order_id,
            "comment": order.comment,
        }
        data = self._request("POST", "/orders", json_body=body)
        return self._to_order_result(data, requested_volume=order.volume)

    @staticmethod
    def _to_order_result(data: dict, *, requested_volume: float | None = None) -> OrderResult:
        status_raw = data.get("status", "UNKNOWN")
        filled_volume = data.get("filled_volume")
        if status_raw == "FILLED" and requested_volume is not None and filled_volume is not None and filled_volume < requested_volume:
            status_raw = "PARTIALLY_FILLED"
        try:
            status = OrderStatus(status_raw)
        except ValueError:
            status = OrderStatus.UNKNOWN
        order_id = data.get("order_id")
        return OrderResult(
            success=status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED),
            order_id=str(order_id) if order_id is not None else None,
            status=status, message=str(data.get("message", "")), retcode=data.get("retcode"), raw=data,
        )

    def modify_order(
        self, order_id: str, *, stop_loss: float | None = None, take_profit: float | None = None,
        price: float | None = None,
    ) -> OrderResult:
        self._ensure_order_allowed()
        body = {k: v for k, v in {"stop_loss": stop_loss, "take_profit": take_profit, "price": price}.items() if v is not None}
        data = self._request("PATCH", f"/orders/{order_id}", json_body=body)
        return self._to_order_result(data)

    def cancel_order(self, order_id: str) -> OrderResult:
        self._ensure_order_allowed()
        data = self._request("DELETE", f"/orders/{order_id}")
        return self._to_order_result(data)

    def close_position(self, position_id: str, *, volume: float | None = None) -> OrderResult:
        self._ensure_order_allowed()
        body = {"volume": volume} if volume is not None else {}
        data = self._request("POST", f"/positions/{position_id}/close", json_body=body)
        return self._to_order_result(data)

    def get_order_status(self, order_id: str) -> OrderStatus:
        data = self._request("GET", f"/orders/{order_id}")
        try:
            return OrderStatus(data.get("status", "UNKNOWN"))
        except ValueError:
            return OrderStatus.UNKNOWN

    # -- reconciliation (Section 25) -------------------------------------------

    def reconcile(self, local_positions: list[Position]) -> ReconciliationReport:
        return reconcile_positions(local_positions, self.get_positions())

    # -- WebSocket quote stream (Section 23) -----------------------------------

    async def stream_quotes(
        self, symbols: list[str], *, connector: Callable[[str], object] | None = None,
    ) -> AsyncIterator[Quote]:
        import json as json_module

        if connector is None:
            import websockets

            connector = websockets.connect
        if not self.websocket_url:
            raise BrokerConnectionError("websocket_url is not configured — cannot stream quotes.")

        async with connector(self.websocket_url) as ws:
            await ws.send(json_module.dumps({"action": "subscribe", "symbols": symbols}))
            async for raw_message in ws:
                data = json_module.loads(raw_message)
                if "symbol" not in data or "bid" not in data or "ask" not in data:
                    continue  # not a quote message (e.g. a subscription ack) -- skip, don't fabricate a Quote
                timestamp = _parse_timestamp(data["timestamp"]) if data.get("timestamp") else datetime.now(timezone.utc)
                yield Quote(symbol=data["symbol"], bid=float(data["bid"]), ask=float(data["ask"]), timestamp=timestamp)


def build_from_config(brokers_cfg: BrokersConfig | None = None) -> GenericRestAdapter:
    """Construct a GenericRestAdapter from config/brokers.yaml's
    `generic_rest` block plus .env — never hard-coded credentials
    (CLAUDE.md Sections 24, 42)."""
    import os

    brokers_cfg = brokers_cfg or load_brokers()
    cfg = brokers_cfg.brokers.get("generic_rest")
    if cfg is None or not cfg.enabled:
        raise BrokerConnectionError("config/brokers.yaml: brokers.generic_rest.enabled is false — nothing to connect to.")
    if not getattr(cfg, "api_base_url", None):
        raise BrokerConnectionError("config/brokers.yaml: brokers.generic_rest.api_base_url is not set.")

    api_key_env = getattr(cfg, "api_key_env", None) or "BROKER_API_KEY"
    api_secret_env = getattr(cfg, "api_secret_env", None) or "BROKER_API_SECRET"
    account_id_env = getattr(cfg, "account_id_env", None) or "BROKER_ACCOUNT_ID"
    rate_limit = getattr(cfg, "rate_limit", None) or {}
    requests_per_second = rate_limit.get("requests_per_second", 5.0) if isinstance(rate_limit, dict) else 5.0

    return GenericRestAdapter(
        api_base_url=cfg.api_base_url,
        api_key=os.getenv(api_key_env) or None,
        api_secret=os.getenv(api_secret_env) or None,
        account_id=os.getenv(account_id_env) or None,
        websocket_url=getattr(cfg, "websocket_url", None),
        environment=cfg.environment,
        requests_per_second=requests_per_second,
        timeout_seconds=getattr(cfg, "timeout_seconds", 10.0),
        max_retries=getattr(cfg, "max_retries", 3),
        retry_base_delay_seconds=getattr(cfg, "retry_base_delay_seconds", 1.0),
        stale_quote_max_age_seconds=getattr(cfg, "stale_quote_max_age_seconds", 5.0),
    )
