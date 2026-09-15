"""MetaTrader 5 historical data loader (CLAUDE.md Section 2, source A).

⚠️ Environment constraint (CLAUDE.md Section 22): the `MetaTrader5` Python
package only works on Windows, connected to a running MT5 terminal. It is
imported lazily here so this module — and everything that imports it — can
be loaded and unit-tested (with MT5 calls mocked) on Linux/macOS. Only
*calling* these functions requires Windows + a running terminal.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.data.schema import standardize_mt5_rates

TIMEFRAME_ATTRS: dict[str, str] = {
    "M5": "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1",
    "H4": "TIMEFRAME_H4",
}


class MT5UnavailableError(RuntimeError):
    """Raised when the MetaTrader5 package/terminal cannot be used here."""


def _import_mt5():
    try:
        import MetaTrader5 as mt5  # type: ignore[import-not-found]
    except ImportError as exc:
        raise MT5UnavailableError(
            "MetaTrader5 package is not installed/importable. It only runs on "
            "Windows with a running MT5 terminal (see CLAUDE.md Section 22). "
            "Install requirements-windows.txt on a Windows machine to use this."
        ) from exc
    return mt5


def connect(
    login: int | None = None,
    password: str | None = None,
    server: str | None = None,
    path: str | None = None,
):
    """Initialize the MT5 terminal connection. Raises MT5UnavailableError on failure."""
    mt5 = _import_mt5()
    kwargs: dict = {"login": login, "password": password, "server": server}
    if path:
        kwargs["path"] = path
    initialized = mt5.initialize(**kwargs)
    if not initialized:
        raise MT5UnavailableError(f"MT5 initialize() failed: {mt5.last_error()}")
    return mt5


def disconnect() -> None:
    mt5 = _import_mt5()
    mt5.shutdown()


def fetch_rates(symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch historical candles for [start, end] and standardize them.

    Requires an active connection (call `connect()` first).
    """
    mt5 = _import_mt5()
    if timeframe not in TIMEFRAME_ATTRS:
        raise ValueError(f"Unsupported timeframe {timeframe!r}; expected one of {sorted(TIMEFRAME_ATTRS)}")

    tf_const = getattr(mt5, TIMEFRAME_ATTRS[timeframe])
    rates = mt5.copy_rates_range(symbol, tf_const, start, end)
    if rates is None:
        raise MT5UnavailableError(
            f"copy_rates_range returned None for {symbol}/{timeframe}: {mt5.last_error()}"
        )

    raw = pd.DataFrame(rates)
    if raw.empty:
        return raw  # nothing in range; caller decides how to report this
    return standardize_mt5_rates(raw, symbol=symbol, timeframe=timeframe)
