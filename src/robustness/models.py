"""Robustness evaluation persistence model (CLAUDE.md Sections 18, 31,
32, 36, 43). One row per run_robustness_evaluation() call: the final
score/status and the key summary numbers behind it (OOS pass rate, MC
probability of ruin, cost-fragility, the parameters evaluated) — full
reproducibility metadata included, same as every other phase.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.core.db import Base


class RobustnessEvaluationRow(Base):
    __tablename__ = "robustness_evaluations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, unique=True, index=True)

    strategy: Mapped[str] = mapped_column(String, index=True)
    symbol: Mapped[str] = mapped_column(String, index=True)
    timeframe: Mapped[str] = mapped_column(String, index=True)

    status: Mapped[str] = mapped_column(String, index=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    sub_scores_json: Mapped[str] = mapped_column(Text)
    notes_json: Mapped[str] = mapped_column(Text)

    oos_pass_rate: Mapped[float] = mapped_column(Float)
    probability_of_ruin: Mapped[float] = mapped_column(Float)
    is_cost_fragile: Mapped[bool] = mapped_column(Boolean)
    n_trades: Mapped[int] = mapped_column(Integer)
    best_params_json: Mapped[str] = mapped_column(Text)

    walkforward_run_id: Mapped[str] = mapped_column(String)
    montecarlo_run_id: Mapped[str] = mapped_column(String)

    code_version: Mapped[str] = mapped_column(String)
    python_version: Mapped[str] = mapped_column(String)
    library_versions_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
