"""Experiment persistence model (CLAUDE.md Sections 12, 29, 36).

Every vectorbt screening run gets a unique ID and is persisted with:
experiment_id, strategy, symbol, timeframe, parameters, date_range,
metrics, data_version, code_version — plus the rest of Section 36's
reproducibility fields (Python/library versions, random seed).
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.core.db import Base


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(String, unique=True, index=True)

    strategy: Mapped[str] = mapped_column(String, index=True)
    symbol: Mapped[str] = mapped_column(String, index=True)
    timeframe: Mapped[str] = mapped_column(String, index=True)
    scenario: Mapped[str] = mapped_column(String)

    parameters_json: Mapped[str] = mapped_column(Text)
    date_range_start: Mapped[datetime] = mapped_column(DateTime)
    date_range_end: Mapped[datetime] = mapped_column(DateTime)
    metrics_json: Mapped[str] = mapped_column(Text)

    data_version: Mapped[str] = mapped_column(String)
    code_version: Mapped[str] = mapped_column(String)
    python_version: Mapped[str] = mapped_column(String)
    library_versions_json: Mapped[str] = mapped_column(Text)
    random_seed: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
