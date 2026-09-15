"""Position Manager (CLAUDE.md Sections 26, 28).

Owns exactly one open virtual position at a time per paper-trading
session (one strategy/symbol/timeframe combination — matching every
other engine's scope, and matching how the vectorbt/Backtrader engines
already treat a direction change as "close, don't stack"). Detects
stop-loss/take-profit touches from a bar's high/low (an intrabar
approximation every bar-level backtest in this codebase already makes —
Phase 6/7 are no more precise), and produces `PaperTrade` records with
exactly the fields CLAUDE.md Section 28's trade log specifies.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

import pandas as pd

from src.execution.simulator import Fill
from src.strategies.base import SignalDirection


@dataclass
class OpenPosition:
    strategy: str
    symbol: str
    timeframe: str
    direction: SignalDirection
    entry_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit: float
    size: float
    risk_amount: float
    entry_commission: float
    entry_spread: float
    slippage_pct: float


@dataclass
class PaperTrade:
    trade_id: str
    strategy: str
    symbol: str
    timeframe: str
    direction: SignalDirection
    entry_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit: float
    size: float
    risk_amount: float
    spread: float
    slippage_pct: float
    exit_time: pd.Timestamp
    exit_price: float
    pnl: float
    r_multiple: float | None
    reason: str
    status: str = "closed"


class PositionManager:
    @staticmethod
    def open(
        *, strategy: str, symbol: str, timeframe: str, direction: SignalDirection, entry_time: pd.Timestamp,
        stop_loss: float, take_profit: float, size: float, risk_amount: float, fill: Fill,
    ) -> OpenPosition:
        return OpenPosition(
            strategy=strategy, symbol=symbol, timeframe=timeframe, direction=direction, entry_time=entry_time,
            entry_price=fill.price, stop_loss=stop_loss, take_profit=take_profit, size=size, risk_amount=risk_amount,
            entry_commission=fill.commission, entry_spread=fill.spread, slippage_pct=fill.slippage_pct,
        )

    @staticmethod
    def unrealized_pnl(position: OpenPosition, current_price: float) -> float:
        sign = 1.0 if position.direction == SignalDirection.LONG else -1.0
        return sign * (current_price - position.entry_price) * position.size - position.entry_commission

    @staticmethod
    def check_stop_touch(position: OpenPosition, bar: pd.Series) -> tuple[float, str] | None:
        """A conservative, stated assumption: if both stop-loss and
        take-profit are inside the same bar's high/low range, the stop
        loss is assumed to have been touched first (the worse outcome
        for the trader) -- there's no intrabar tick data to know the
        true order, and assuming the better outcome would be optimistic,
        not realistic (CLAUDE.md Section 11)."""
        if position.direction == SignalDirection.LONG:
            hit_stop = bar["low"] <= position.stop_loss
            hit_target = bar["high"] >= position.take_profit
        else:
            hit_stop = bar["high"] >= position.stop_loss
            hit_target = bar["low"] <= position.take_profit
        if hit_stop:
            return position.stop_loss, "stop_loss"
        if hit_target:
            return position.take_profit, "take_profit"
        return None

    @staticmethod
    def close(position: OpenPosition, *, exit_time: pd.Timestamp, fill: Fill, reason: str) -> PaperTrade:
        sign = 1.0 if position.direction == SignalDirection.LONG else -1.0
        gross_pnl = sign * (fill.price - position.entry_price) * position.size
        pnl = gross_pnl - position.entry_commission - fill.commission
        r_multiple = pnl / position.risk_amount if position.risk_amount else None
        return PaperTrade(
            trade_id=str(uuid.uuid4()), strategy=position.strategy, symbol=position.symbol, timeframe=position.timeframe,
            direction=position.direction, entry_time=position.entry_time, entry_price=position.entry_price,
            stop_loss=position.stop_loss, take_profit=position.take_profit, size=position.size,
            risk_amount=position.risk_amount, spread=position.entry_spread, slippage_pct=position.slippage_pct,
            exit_time=exit_time, exit_price=fill.price, pnl=pnl, r_multiple=r_multiple, reason=reason,
        )
