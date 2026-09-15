"""MT5 broker adapter (CLAUDE.md Sections 21, 22, 27).

⚠️ Environment constraint (Section 22): the `MetaTrader5` Python package
only works on Windows, connected to a running MT5 terminal — it cannot
be installed or imported here. This module imports it lazily (reusing
`src.data.mt5_loader`'s connect()/MT5UnavailableError) so it stays
importable and unit-testable — with every `MetaTrader5` call
mocked/stubbed — on Linux/macOS. Actually connecting, authenticating, and
placing paper/live orders only happens when this runs on Windows with a
real terminal; nothing here has been (or can be) exercised against a
real MT5 account from this sandbox.

Field/kwarg names below (account_info/symbol_info/positions_get/
order_send/... and their attributes) follow the MetaTrader5 package's
documented API surface as closely as this environment allows to verify
without the package installed; treat them as a best-effort mapping to
confirm against a real terminal before live use, not a guarantee.

Default mode is always paper/dry-run: `place_order`/`modify_order`/
`cancel_order`/`close_position` refuse to run against a `environment=
"live"` connection unless both `LIVE_TRADING=true` and
`LIVE_CONFIRMATION=true` (enforced by the base class — see
BrokerAdapter._ensure_order_allowed()). A "demo" connection is never
gated by those flags since no real money is at risk.
"""
from __future__ import annotations

import zlib
from datetime import datetime, timezone

from src.brokers.base import (
    AccountInfo,
    BrokerAdapter,
    BrokerConnectionError,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Quote,
    SymbolInfo,
)
from src.core.config import BrokersConfig, load_brokers
from src.data.mt5_loader import MT5UnavailableError

_ORDER_TYPE_ATTRS: dict[tuple[OrderSide, OrderType], str] = {
    (OrderSide.BUY, OrderType.MARKET): "ORDER_TYPE_BUY",
    (OrderSide.SELL, OrderType.MARKET): "ORDER_TYPE_SELL",
    (OrderSide.BUY, OrderType.LIMIT): "ORDER_TYPE_BUY_LIMIT",
    (OrderSide.SELL, OrderType.LIMIT): "ORDER_TYPE_SELL_LIMIT",
    (OrderSide.BUY, OrderType.STOP): "ORDER_TYPE_BUY_STOP",
    (OrderSide.SELL, OrderType.STOP): "ORDER_TYPE_SELL_STOP",
}

# MT5's own position "type" field: 0 = buy, 1 = sell.
_POSITION_TYPE_TO_SIDE = {0: OrderSide.BUY, 1: OrderSide.SELL}


def _client_order_id_to_magic(client_order_id: str) -> int:
    """MT5's "magic" number is a plain int; map a caller's string
    idempotency key onto one deterministically (same key -> same magic
    every time, unlike Python's own hash())."""
    return zlib.crc32(client_order_id.encode("utf-8")) & 0x7FFFFFFF


class MT5Adapter(BrokerAdapter):
    def __init__(
        self,
        *,
        login: int | None = None,
        password: str | None = None,
        server: str | None = None,
        path: str | None = None,
        environment: str = "demo",
    ) -> None:
        super().__init__(environment=environment)
        self.login = login
        self.password = password
        self.server = server
        self.path = path
        self._mt5 = None

    # -- connection -----------------------------------------------------

    def connect(self) -> None:
        from src.data import mt5_loader

        self._mt5 = mt5_loader.connect(login=self.login, password=self.password, server=self.server, path=self.path)

    def disconnect(self) -> None:
        if self._mt5 is not None:
            self._mt5.shutdown()
            self._mt5 = None

    def _require_connection(self):
        if self._mt5 is None:
            raise BrokerConnectionError("Not connected to MT5 — call connect() first.")
        return self._mt5

    # -- account / positions / market data -------------------------------

    def get_account(self) -> AccountInfo:
        mt5 = self._require_connection()
        info = mt5.account_info()
        if info is None:
            raise BrokerConnectionError(f"MT5 account_info() returned None: {mt5.last_error()}")
        return AccountInfo(
            login=int(info.login), name=str(info.name), server=str(info.server), currency=str(info.currency),
            leverage=float(info.leverage), balance=float(info.balance), equity=float(info.equity),
            margin=float(info.margin), margin_free=float(info.margin_free),
        )

    def get_balance(self) -> float:
        return self.get_account().balance

    def get_equity(self) -> float:
        return self.get_account().equity

    def get_positions(self) -> list[Position]:
        mt5 = self._require_connection()
        raw_positions = mt5.positions_get() or ()
        return [self._to_position(p) for p in raw_positions]

    @staticmethod
    def _to_position(p) -> Position:
        return Position(
            position_id=str(p.ticket),
            symbol=p.symbol,
            side=_POSITION_TYPE_TO_SIDE.get(p.type, OrderSide.BUY),
            volume=float(p.volume),
            entry_price=float(p.price_open),
            current_price=float(p.price_current),
            stop_loss=float(p.sl) if p.sl else None,
            take_profit=float(p.tp) if p.tp else None,
            profit=float(p.profit),
            open_time=datetime.fromtimestamp(p.time, tz=timezone.utc),
        )

    def get_symbol_info(self, symbol: str) -> SymbolInfo:
        mt5 = self._require_connection()
        info = mt5.symbol_info(symbol)
        if info is None:
            raise BrokerConnectionError(f"MT5 symbol_info({symbol!r}) returned None: {mt5.last_error()}")
        return SymbolInfo(
            symbol=symbol, digits=int(info.digits), point=float(info.point),
            contract_size=float(info.trade_contract_size), volume_min=float(info.volume_min),
            volume_max=float(info.volume_max), volume_step=float(info.volume_step),
            tick_value=float(info.trade_tick_value), tick_size=float(info.trade_tick_size),
            spread=float(info.spread),
        )

    def get_quote(self, symbol: str) -> Quote:
        mt5 = self._require_connection()
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise BrokerConnectionError(f"MT5 symbol_info_tick({symbol!r}) returned None: {mt5.last_error()}")
        return Quote(
            symbol=symbol, bid=float(tick.bid), ask=float(tick.ask),
            timestamp=datetime.fromtimestamp(tick.time, tz=timezone.utc),
        )

    # -- orders -----------------------------------------------------------

    def _build_request(self, order: OrderRequest) -> dict:
        mt5 = self._require_connection()
        order_type_attr = _ORDER_TYPE_ATTRS.get((order.side, order.order_type))
        if order_type_attr is None:
            raise ValueError(f"Unsupported side/order_type combination: {order.side}/{order.order_type}")

        request: dict = {
            "action": mt5.TRADE_ACTION_DEAL if order.order_type == OrderType.MARKET else mt5.TRADE_ACTION_PENDING,
            "symbol": order.symbol,
            "volume": order.volume,
            "type": getattr(mt5, order_type_attr),
            "sl": order.stop_loss or 0.0,
            "tp": order.take_profit or 0.0,
            "comment": order.comment or "",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        if order.price is not None:
            request["price"] = order.price
        if order.client_order_id:
            request["magic"] = _client_order_id_to_magic(order.client_order_id)
        return request

    def _to_order_result(self, result) -> OrderResult:
        mt5 = self._require_connection()
        success = result.retcode == mt5.TRADE_RETCODE_DONE
        order_id = getattr(result, "order", None)
        return OrderResult(
            success=success,
            order_id=str(order_id) if order_id else None,
            status=OrderStatus.FILLED if success else OrderStatus.REJECTED,
            message=str(getattr(result, "comment", "")),
            retcode=int(result.retcode),
            raw=result._asdict() if hasattr(result, "_asdict") else {},
        )

    def place_order(self, order: OrderRequest) -> OrderResult:
        self._ensure_order_allowed()
        mt5 = self._require_connection()
        result = mt5.order_send(self._build_request(order))
        return self._to_order_result(result)

    def modify_order(
        self, order_id: str, *, stop_loss: float | None = None, take_profit: float | None = None,
        price: float | None = None,
    ) -> OrderResult:
        self._ensure_order_allowed()
        mt5 = self._require_connection()
        request: dict = {"action": mt5.TRADE_ACTION_SLTP, "position": int(order_id)}
        if stop_loss is not None:
            request["sl"] = stop_loss
        if take_profit is not None:
            request["tp"] = take_profit
        if price is not None:
            request["price"] = price
            request["action"] = mt5.TRADE_ACTION_MODIFY
        result = mt5.order_send(request)
        return self._to_order_result(result)

    def cancel_order(self, order_id: str) -> OrderResult:
        self._ensure_order_allowed()
        mt5 = self._require_connection()
        result = mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": int(order_id)})
        return self._to_order_result(result)

    def close_position(self, position_id: str, *, volume: float | None = None) -> OrderResult:
        self._ensure_order_allowed()
        mt5 = self._require_connection()
        positions = mt5.positions_get(ticket=int(position_id)) or ()
        if not positions:
            return OrderResult(
                success=False, order_id=None, status=OrderStatus.UNKNOWN,
                message=f"No open position with id {position_id}", retcode=None,
            )
        position = positions[0]
        closing_side = OrderSide.SELL if position.type == 0 else OrderSide.BUY
        close_order = OrderRequest(
            symbol=position.symbol, side=closing_side, order_type=OrderType.MARKET,
            volume=volume if volume is not None else float(position.volume), comment="close_position",
        )
        request = self._build_request(close_order)
        request["position"] = int(position_id)
        result = mt5.order_send(request)
        return self._to_order_result(result)

    def get_order_status(self, order_id: str) -> OrderStatus:
        mt5 = self._require_connection()
        pending = mt5.orders_get(ticket=int(order_id)) if hasattr(mt5, "orders_get") else None
        if pending:
            return OrderStatus.PENDING
        history = mt5.history_orders_get(ticket=int(order_id)) if hasattr(mt5, "history_orders_get") else None
        if history:
            return OrderStatus.FILLED
        return OrderStatus.UNKNOWN


def build_from_config(brokers_cfg: BrokersConfig | None = None) -> MT5Adapter:
    """Construct an MT5Adapter from config/brokers.yaml + .env — never
    from hard-coded credentials (CLAUDE.md Sections 22, 24, 42)."""
    import os

    brokers_cfg = brokers_cfg or load_brokers()
    mt5_cfg = brokers_cfg.brokers.get("mt5")
    if mt5_cfg is None or not mt5_cfg.enabled:
        raise MT5UnavailableError("config/brokers.yaml: brokers.mt5.enabled is false — nothing to connect to.")

    terminal_path_env = getattr(mt5_cfg, "terminal_path_env", None) or "MT5_TERMINAL_PATH"
    login = os.getenv("MT5_LOGIN")
    return MT5Adapter(
        login=int(login) if login else None,
        password=os.getenv("MT5_PASSWORD") or None,
        server=os.getenv("MT5_SERVER") or None,
        path=os.getenv(terminal_path_env) or None,
        environment=mt5_cfg.environment,
    )
