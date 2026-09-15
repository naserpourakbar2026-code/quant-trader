"""Logging setup.

Two separate sinks per CLAUDE.md Section 28:
- trade log: trade lifecycle events (populated starting with the backtest/
  execution phases)
- system log: connections, errors, retries, risk violations, kill switch

Both are loguru sinks writing under the `logs/` directory. Call
`configure_logging()` once at process start (main.py does this).
"""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from src.core.config import PROJECT_ROOT, load_settings

_configured = False


def configure_logging(level: str | None = None) -> None:
    global _configured
    if _configured:
        return

    settings = load_settings()
    log_level = level or settings.logging.level

    system_log_path = PROJECT_ROOT / settings.logging.system_log_file
    trade_log_path = PROJECT_ROOT / settings.logging.trade_log_file
    for path in (system_log_path, trade_log_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(sys.stderr, level=log_level)
    logger.add(
        system_log_path,
        level=log_level,
        rotation="10 MB",
        retention="30 days",
        filter=lambda record: record["extra"].get("channel") != "trade",
    )
    logger.add(
        trade_log_path,
        level="INFO",
        rotation="10 MB",
        retention="1 year",
        filter=lambda record: record["extra"].get("channel") == "trade",
    )

    _configured = True


def get_trade_logger():
    """Logger bound to the trade channel (routes to logs/trades.log)."""
    return logger.bind(channel="trade")


def get_system_logger():
    """Logger bound to the system channel (routes to logs/system.log)."""
    return logger.bind(channel="system")
