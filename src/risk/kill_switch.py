"""The durable half of the hard risk kill switch (CLAUDE.md Sections 7,
27, 28). `RiskEngine`'s own `kill_switch_triggered` flag (src.risk.engine)
is per-instance and drawdown-based; this is the cross-session, persisted
record of every trip and reset, so "stop new trades" (Section 7) and
"block all new orders instantly" (Section 27's emergency_stop()) actually
mean something beyond the single process that first noticed the
drawdown breach.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.core.db import Database, get_default_database
from src.core.logging import get_system_logger
from src.risk.models import KillSwitchEvent


@dataclass
class KillSwitchEventRecord:
    event_type: str
    reason: str
    equity: float | None
    drawdown: float | None
    created_at: datetime | None = None


class KillSwitch:
    def __init__(self, db: Database | None = None) -> None:
        self.db = db

    def is_triggered(self) -> bool:
        database = self.db or get_default_database()
        with database.session() as session:
            latest = session.query(KillSwitchEvent).order_by(KillSwitchEvent.id.desc()).first()
            return latest is not None and latest.event_type == "trip"

    def trip(
        self, reason: str, *, equity: float | None = None, drawdown: float | None = None, brokers: list | None = None,
    ) -> None:
        """Record a trip event and, per Section 27, call emergency_stop()
        on every broker adapter the caller cares about — `brokers` is
        empty by default because paper trading (Phase 14) has no real
        broker connection to halt; a live-execution caller would pass
        its active BrokerAdapter instances here."""
        database = self.db or get_default_database()
        with database.session() as session:
            session.add(KillSwitchEvent(event_type="trip", reason=reason, equity=equity, drawdown=drawdown))
            session.commit()
        get_system_logger().error(f"KILL SWITCH TRIPPED: {reason}")
        for broker in brokers or []:
            broker.emergency_stop()

    def reset(self, note: str, *, brokers: list | None = None, reset_brokers: bool = True) -> None:
        """Never automatic — a human decision, always with a note
        explaining why, recorded in the same audit trail as the trip
        itself."""
        database = self.db or get_default_database()
        with database.session() as session:
            session.add(KillSwitchEvent(event_type="reset", reason=note))
            session.commit()
        get_system_logger().warning(f"KILL SWITCH RESET: {note}")
        if reset_brokers:
            for broker in brokers or []:
                broker.reset_emergency_stop()

    def history(self, *, limit: int = 50) -> list[KillSwitchEventRecord]:
        database = self.db or get_default_database()
        with database.session() as session:
            rows = session.query(KillSwitchEvent).order_by(KillSwitchEvent.id.desc()).limit(limit).all()
            return [
                KillSwitchEventRecord(
                    event_type=row.event_type, reason=row.reason, equity=row.equity, drawdown=row.drawdown,
                    created_at=row.created_at,
                )
                for row in rows
            ]
