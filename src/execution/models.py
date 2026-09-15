"""Paper trading persistence models (CLAUDE.md Sections 26, 28, 29, 36).

`PaperTradeRow` holds exactly CLAUDE.md Section 28's trade-log field
list (timestamp, symbol, strategy, signal, entry, SL, TP, size, risk,
spread, slippage, exit, PnL, R multiple, reason) — this is the first
phase anything actually gets logged at that per-trade granularity;
Phases 6-10's `experiments` table only ever stored aggregate metrics.
`PaperSession` is one row per run_paper_trading_session() call, grouping
its trades and summarizing the outcome (mirrors the run_id pattern in
walkforward_windows/montecarlo_runs/portfolio_runs).
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.core.db import Base


class PaperSession(Base):
    __tablename__ = "paper_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, unique=True, index=True)

    strategy: Mapped[str] = mapped_column(String, index=True)
    symbol: Mapped[str] = mapped_column(String, index=True)
    timeframe: Mapped[str] = mapped_column(String, index=True)
    scenario: Mapped[str] = mapped_column(String)

    initial_capital: Mapped[float] = mapped_column(Float)
    final_equity: Mapped[float] = mapped_column(Float)
    total_trades: Mapped[int] = mapped_column(Integer)
    win_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_pnl: Mapped[float] = mapped_column(Float)
    kill_switch_triggered: Mapped[bool] = mapped_column(Boolean)
    still_open_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    code_version: Mapped[str] = mapped_column(String)
    python_version: Mapped[str] = mapped_column(String)
    library_versions_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class PaperTradeRow(Base):
    __tablename__ = "paper_trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_id: Mapped[str] = mapped_column(String, unique=True, index=True)
    run_id: Mapped[str] = mapped_column(String, index=True)

    strategy: Mapped[str] = mapped_column(String, index=True)
    symbol: Mapped[str] = mapped_column(String, index=True)
    timeframe: Mapped[str] = mapped_column(String)
    direction: Mapped[str] = mapped_column(String)  # the trade log's "signal"

    entry_time: Mapped[datetime] = mapped_column(DateTime)  # the trade log's "timestamp"
    entry_price: Mapped[float] = mapped_column(Float)  # "entry"
    stop_loss: Mapped[float] = mapped_column(Float)  # "SL"
    take_profit: Mapped[float] = mapped_column(Float)  # "TP"
    size: Mapped[float] = mapped_column(Float)  # "size"
    risk_amount: Mapped[float] = mapped_column(Float)  # "risk"
    spread: Mapped[float] = mapped_column(Float)  # "spread"
    slippage_pct: Mapped[float] = mapped_column(Float)  # "slippage"

    exit_time: Mapped[datetime] = mapped_column(DateTime)
    exit_price: Mapped[float] = mapped_column(Float)  # "exit"
    pnl: Mapped[float] = mapped_column(Float)  # "PnL"
    r_multiple: Mapped[float | None] = mapped_column(Float, nullable=True)  # "R multiple"
    reason: Mapped[str] = mapped_column(String)  # "reason"

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
