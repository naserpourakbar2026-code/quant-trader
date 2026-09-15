"""Walk-forward window persistence model (CLAUDE.md Sections 15, 29, 36).

One row per window of one walk-forward run: training/validation/OOS date
ranges, the parameters chosen on training, the post-hoc stability score,
and both validation and out-of-sample metrics — kept separate from
`experiments` (src.backtest.models) because a WFA window is a genuinely
different kind of record (three sub-periods and a pass/fail verdict, not
one backtest run).
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.core.db import Base


class WalkForwardWindow(Base):
    __tablename__ = "walkforward_windows"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, index=True)  # groups all windows from one run_walk_forward() call
    window_index: Mapped[int] = mapped_column(Integer)

    strategy: Mapped[str] = mapped_column(String, index=True)
    symbol: Mapped[str] = mapped_column(String, index=True)
    timeframe: Mapped[str] = mapped_column(String, index=True)
    scenario: Mapped[str] = mapped_column(String)

    train_start: Mapped[datetime] = mapped_column(DateTime)
    train_end: Mapped[datetime] = mapped_column(DateTime)
    validation_start: Mapped[datetime] = mapped_column(DateTime)
    validation_end: Mapped[datetime] = mapped_column(DateTime)
    oos_start: Mapped[datetime] = mapped_column(DateTime)
    oos_end: Mapped[datetime] = mapped_column(DateTime)

    best_params_json: Mapped[str] = mapped_column(Text)
    train_objective: Mapped[float] = mapped_column(Float)
    stability_score: Mapped[float] = mapped_column(Float)
    validation_metrics_json: Mapped[str] = mapped_column(Text)
    oos_metrics_json: Mapped[str] = mapped_column(Text)
    oos_objective: Mapped[float] = mapped_column(Float)
    oos_passed: Mapped[bool] = mapped_column(Boolean)

    code_version: Mapped[str] = mapped_column(String)
    random_seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
