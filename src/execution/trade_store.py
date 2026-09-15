"""Read/write access to `paper_sessions`/`paper_trades` (CLAUDE.md
Sections 26, 28, 29). Same shape/conventions as
src.backtest.experiment_store, src.walkforward.window_store, etc.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime

from src.core.db import Database, get_default_database
from src.execution.models import PaperSession, PaperTradeRow
from src.execution.position_manager import PaperTrade


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


@dataclass
class PaperSessionRecord:
    run_id: str
    strategy: str
    symbol: str
    timeframe: str
    scenario: str
    initial_capital: float
    final_equity: float
    total_trades: int
    win_rate: float | None
    total_pnl: float
    kill_switch_triggered: bool
    still_open: dict | None = None
    code_version: str = "unknown"
    python_version: str = "unknown"
    library_versions: dict = field(default_factory=dict)
    created_at: datetime | None = None


def save_session(record: PaperSessionRecord, trades: list[PaperTrade], db: Database | None = None) -> None:
    database = db or get_default_database()
    with database.session() as session:
        session.add(
            PaperSession(
                run_id=record.run_id, strategy=record.strategy, symbol=record.symbol, timeframe=record.timeframe,
                scenario=record.scenario, initial_capital=record.initial_capital, final_equity=record.final_equity,
                total_trades=record.total_trades, win_rate=record.win_rate, total_pnl=record.total_pnl,
                kill_switch_triggered=record.kill_switch_triggered,
                still_open_json=json.dumps(_json_safe(record.still_open)) if record.still_open else None,
                code_version=record.code_version, python_version=record.python_version,
                library_versions_json=json.dumps(record.library_versions),
            )
        )
        for trade in trades:
            session.add(
                PaperTradeRow(
                    trade_id=trade.trade_id, run_id=record.run_id, strategy=trade.strategy, symbol=trade.symbol,
                    timeframe=trade.timeframe, direction=trade.direction.value,
                    entry_time=trade.entry_time.to_pydatetime() if hasattr(trade.entry_time, "to_pydatetime") else trade.entry_time,
                    entry_price=trade.entry_price, stop_loss=trade.stop_loss, take_profit=trade.take_profit,
                    size=trade.size, risk_amount=trade.risk_amount, spread=trade.spread, slippage_pct=trade.slippage_pct,
                    exit_time=trade.exit_time.to_pydatetime() if hasattr(trade.exit_time, "to_pydatetime") else trade.exit_time,
                    exit_price=trade.exit_price, pnl=trade.pnl, r_multiple=trade.r_multiple, reason=trade.reason,
                )
            )
        session.commit()


def _session_to_record(row: PaperSession) -> PaperSessionRecord:
    return PaperSessionRecord(
        run_id=row.run_id, strategy=row.strategy, symbol=row.symbol, timeframe=row.timeframe, scenario=row.scenario,
        initial_capital=row.initial_capital, final_equity=row.final_equity, total_trades=row.total_trades,
        win_rate=row.win_rate, total_pnl=row.total_pnl, kill_switch_triggered=row.kill_switch_triggered,
        still_open=json.loads(row.still_open_json) if row.still_open_json else None,
        code_version=row.code_version, python_version=row.python_version,
        library_versions=json.loads(row.library_versions_json), created_at=row.created_at,
    )


def get_session(run_id: str, db: Database | None = None) -> PaperSessionRecord | None:
    database = db or get_default_database()
    with database.session() as session:
        row = session.query(PaperSession).filter_by(run_id=run_id).one_or_none()
        return _session_to_record(row) if row else None


def list_sessions(
    *, strategy: str | None = None, symbol: str | None = None, timeframe: str | None = None,
    db: Database | None = None,
) -> list[PaperSessionRecord]:
    database = db or get_default_database()
    with database.session() as session:
        query = session.query(PaperSession)
        if strategy is not None:
            query = query.filter_by(strategy=strategy)
        if symbol is not None:
            query = query.filter_by(symbol=symbol)
        if timeframe is not None:
            query = query.filter_by(timeframe=timeframe)
        return [_session_to_record(row) for row in query.order_by(PaperSession.created_at.desc()).all()]


@dataclass
class PaperTradeRecord:
    trade_id: str
    run_id: str
    strategy: str
    symbol: str
    timeframe: str
    direction: str
    entry_time: datetime
    entry_price: float
    stop_loss: float
    take_profit: float
    size: float
    risk_amount: float
    spread: float
    slippage_pct: float
    exit_time: datetime
    exit_price: float
    pnl: float
    r_multiple: float | None
    reason: str


def _trade_to_record(row: PaperTradeRow) -> PaperTradeRecord:
    return PaperTradeRecord(
        trade_id=row.trade_id, run_id=row.run_id, strategy=row.strategy, symbol=row.symbol, timeframe=row.timeframe,
        direction=row.direction, entry_time=row.entry_time, entry_price=row.entry_price, stop_loss=row.stop_loss,
        take_profit=row.take_profit, size=row.size, risk_amount=row.risk_amount, spread=row.spread,
        slippage_pct=row.slippage_pct, exit_time=row.exit_time, exit_price=row.exit_price, pnl=row.pnl,
        r_multiple=row.r_multiple, reason=row.reason,
    )


def list_trades(run_id: str, db: Database | None = None) -> list[PaperTradeRecord]:
    database = db or get_default_database()
    with database.session() as session:
        rows = session.query(PaperTradeRow).filter_by(run_id=run_id).order_by(PaperTradeRow.entry_time.asc()).all()
        return [_trade_to_record(row) for row in rows]
