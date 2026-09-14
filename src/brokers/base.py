"""Abstract broker adapter (CLAUDE.md Section 21).

Strategy, risk and execution code must never talk to a broker's SDK/API
directly — everything routes through this interface, so a broker can be
swapped (or a second one added, Phase 13) without touching strategy
logic. `MT5Adapter` (Phase 12) is the first implementation; the generic
REST/WebSocket adapter (Phase 13) implements the same interface.

Two safety mechanisms live here, shared by every adapter (CLAUDE.md
Section 27):
- `emergency_stop()` / `reset_emergency_stop()` — an in-process halt flag
  every order-sending method must check before doing anything.
- `_ensure_live_trading_allowed()` — when an adapter's `environment` is
  "live" (not "demo"), this refuses to proceed unless *both*
  `LIVE_TRADING=true` and `LIVE_CONFIRMATION=true` are set. A demo
  connection is never gated by this (it isn't real money), matching
  brokers.yaml's `environment: demo|live` field.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from src.core.config import is_live_trading_enabled


class BrokerError(Exception):
    """Base class for every broker-adapter error."""


class BrokerConnectionError(BrokerError):
    """Not connected, or the broker rejected/lost the connection."""


class EmergencyStopActive(BrokerError):
    """emergency_stop() has been called — refusing to send a new order."""


class LiveTradingBlockedError(BrokerError):
    """environment == 'live' but LIVE_TRADING/LIVE_CONFIRMATION aren't
    both true (CLAUDE.md Section 27) — refusing to send a real order."""


class BrokerRejectionError(BrokerError):
    """The broker actively rejected a request (e.g. HTTP 4xx) — as
    opposed to BrokerConnectionError, which is "couldn't reach/trust the
    broker at all" (CLAUDE.md Section 25's "broker rejection handling")."""


class DuplicateOrderError(BrokerError):
    """A client_order_id has already been submitted — refusing to send
    it twice (CLAUDE.md Section 25's "duplicate order protection")."""


class StalePriceError(BrokerError):
    """A quote is older than the configured freshness threshold — acting
    on it would risk trading on a price that's no longer real (CLAUDE.md
    Section 25's "stale-price detection")."""


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


@dataclass
class AccountInfo:
    login: int
    name: str
    server: str
    currency: str
    leverage: float
    balance: float
    equity: float
    margin: float
    margin_free: float


@dataclass
class Position:
    position_id: str
    symbol: str
    side: OrderSide
    volume: float
    entry_price: float
    current_price: float
    stop_loss: float | None
    take_profit: float | None
    profit: float
    open_time: datetime


@dataclass
class SymbolInfo:
    symbol: str
    digits: int
    point: float
    contract_size: float
    volume_min: float
    volume_max: float
    volume_step: float
    tick_value: float
    tick_size: float
    spread: float


@dataclass
class Quote:
    symbol: str
    bid: float
    ask: float
    timestamp: datetime


@dataclass
class OrderRequest:
    symbol: str
    side: OrderSide
    order_type: OrderType
    volume: float
    price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    # Caller-assigned idempotency key (CLAUDE.md Section 25's "duplicate
    # order protection"). This adapter only carries it through to the
    # broker (e.g. as an MT5 "magic" number) -- deduplicating on it is an
    # execution-layer concern (Phase 14/25), not this adapter's.
    client_order_id: str | None = None
    comment: str = ""


@dataclass
class OrderResult:
    success: bool
    order_id: str | None
    status: OrderStatus
    message: str
    retcode: int | None = None
    raw: dict = field(default_factory=dict)  # untouched broker response, for debugging/reconciliation


class BrokerAdapter(ABC):
    def __init__(self, *, environment: str = "demo") -> None:
        self.environment = environment
        self._halted = False

    def emergency_stop(self) -> None:
        """Block all new orders instantly (CLAUDE.md Section 27)."""
        self._halted = True

    def reset_emergency_stop(self) -> None:
        self._halted = False

    def _ensure_order_allowed(self) -> None:
        """Every place_order/modify_order/cancel_order/close_position
        implementation must call this first."""
        if self._halted:
            raise EmergencyStopActive("emergency_stop() has been called — no new orders are being sent.")
        if self.environment == "live" and not is_live_trading_enabled():
            raise LiveTradingBlockedError(
                "environment='live' but LIVE_TRADING and LIVE_CONFIRMATION are not both 'true' — "
                "refusing to send a real order (CLAUDE.md Section 27)."
            )

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def get_account(self) -> AccountInfo: ...

    @abstractmethod
    def get_balance(self) -> float: ...

    @abstractmethod
    def get_equity(self) -> float: ...

    @abstractmethod
    def get_positions(self) -> list[Position]: ...

    @abstractmethod
    def get_symbol_info(self, symbol: str) -> SymbolInfo: ...

    @abstractmethod
    def get_quote(self, symbol: str) -> Quote: ...

    @abstractmethod
    def place_order(self, order: OrderRequest) -> OrderResult: ...

    @abstractmethod
    def modify_order(
        self, order_id: str, *, stop_loss: float | None = None, take_profit: float | None = None,
        price: float | None = None,
    ) -> OrderResult: ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> OrderResult: ...

    @abstractmethod
    def close_position(self, position_id: str, *, volume: float | None = None) -> OrderResult: ...

    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderStatus: ...


@dataclass
class ReconciliationReport:
    """CLAUDE.md Section 25: "reconcile local positions with broker
    positions regularly." Broker-agnostic — works from two plain
    `Position` lists, so any adapter's `get_positions()` output can be
    compared against whatever an execution layer (Phase 14) believes
    locally, without either side depending on the other's internals."""

    missing_locally: list[Position] = field(default_factory=list)  # broker has it, local bookkeeping doesn't
    missing_at_broker: list[Position] = field(default_factory=list)  # local bookkeeping has it, broker doesn't
    mismatched: list[tuple[Position, Position]] = field(default_factory=list)  # same id, different volume/side/sl/tp

    @property
    def is_clean(self) -> bool:
        return not (self.missing_locally or self.missing_at_broker or self.mismatched)


def reconcile_positions(local_positions: list[Position], broker_positions: list[Position]) -> ReconciliationReport:
    local_by_id = {p.position_id: p for p in local_positions}
    broker_by_id = {p.position_id: p for p in broker_positions}

    missing_locally = [p for pid, p in broker_by_id.items() if pid not in local_by_id]
    missing_at_broker = [p for pid, p in local_by_id.items() if pid not in broker_by_id]
    mismatched = [
        (local_by_id[pid], broker_by_id[pid])
        for pid in local_by_id.keys() & broker_by_id.keys()
        if (local_by_id[pid].volume, local_by_id[pid].side) != (broker_by_id[pid].volume, broker_by_id[pid].side)
    ]
    return ReconciliationReport(missing_locally=missing_locally, missing_at_broker=missing_at_broker, mismatched=mismatched)
