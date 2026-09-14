"""Kill switch persistence model (CLAUDE.md Sections 7, 27, 28, 29).

An append-only audit log, never a mutable singleton row: every trip and
every reset is its own event, so "is the kill switch currently
triggered" is answered by "what was the most recent event", and the
full history stays inspectable (Section 28's "System log ... kill
switch events") rather than being overwritten. This is the durable half
of the kill switch — it persists across process restarts and across
every RiskEngine instance/paper-trading session that shares the same
database, which is the actual point of a *hard* kill switch (Section 7):
one session tripping it must stop every other session too, not just
itself.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.core.db import Base


class KillSwitchEvent(Base):
    __tablename__ = "kill_switch_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(String)  # "trip" | "reset"
    reason: Mapped[str] = mapped_column(Text)
    equity: Mapped[float | None] = mapped_column(Float, nullable=True)
    drawdown: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
