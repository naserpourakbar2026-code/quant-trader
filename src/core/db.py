"""SQLAlchemy engine/session setup (CLAUDE.md Section 29).

SQLite initially, via SQLAlchemy — chosen specifically so a later move to
PostgreSQL only requires changing DATABASE_URL, not the ORM code.

Not a module-level singleton by default: `Database` is a small class you
construct explicitly (production code uses `get_default_database()`;
tests construct their own `Database("sqlite:///:memory:")`) so tests never
touch the real project database.
"""
from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.core.config import PROJECT_ROOT


class Base(DeclarativeBase):
    pass


def default_database_url() -> str:
    db_path = PROJECT_ROOT / "data" / "quant_trader.db"
    return f"sqlite:///{db_path}"


def get_database_url() -> str:
    return os.getenv("DATABASE_URL") or default_database_url()


class Database:
    def __init__(self, url: str | None = None):
        self.url = url or get_database_url()
        if self.url.startswith("sqlite:///"):
            db_file = self.url.removeprefix("sqlite:///")
            if db_file and db_file != ":memory:":
                Path(db_file).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(self.url, future=True)
        self._session_factory = sessionmaker(bind=self.engine, future=True)

    def init_db(self) -> None:
        """Create every table registered on Base.metadata that doesn't exist yet."""
        Base.metadata.create_all(self.engine)

    def session(self) -> Session:
        return self._session_factory()


_default_db: Database | None = None


def get_default_database() -> Database:
    global _default_db
    if _default_db is None:
        _default_db = Database()
        _default_db.init_db()
    return _default_db
