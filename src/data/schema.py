"""Standardized internal OHLCV schema (CLAUDE.md Section 2).

Every data source (CSV, MT5) is mapped onto this same schema before
anything downstream (validation, features, strategies, backtests) sees it,
so the rest of the framework never needs to know where a candle came from.
"""
from __future__ import annotations

import pandas as pd

STANDARD_COLUMNS = [
    "timestamp",
    "symbol",
    "timeframe",
    "open",
    "high",
    "low",
    "close",
    "tick_volume",
    "spread",
    "real_volume",
]

# Minimum fields a CSV source must provide (CLAUDE.md Section 2).
REQUIRED_CSV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

# Fields MT5's copy_rates_range()/copy_rates_from() return.
REQUIRED_MT5_COLUMNS = [
    "time",
    "open",
    "high",
    "low",
    "close",
    "tick_volume",
    "spread",
    "real_volume",
]


def standardize_csv_rows(df: pd.DataFrame, *, symbol: str, timeframe: str) -> pd.DataFrame:
    """Map a minimal CSV onto the standardized internal schema.

    Required columns: timestamp, open, high, low, close, volume.
    Optional columns (tick_volume, spread, real_volume) are used verbatim
    if present; if absent they become NaN rather than a fabricated value —
    this is ingestion, not repair (that is Phase 3's job, and it only
    detects, never silently fixes).
    """
    missing = [c for c in REQUIRED_CSV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns {missing}; found {list(df.columns)}")

    out = pd.DataFrame(index=df.index)
    out["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    out["symbol"] = symbol
    out["timeframe"] = timeframe
    out["open"] = df["open"].astype(float)
    out["high"] = df["high"].astype(float)
    out["low"] = df["low"].astype(float)
    out["close"] = df["close"].astype(float)
    out["tick_volume"] = (
        df["tick_volume"].astype(float) if "tick_volume" in df.columns else df["volume"].astype(float)
    )
    out["spread"] = df["spread"].astype(float) if "spread" in df.columns else float("nan")
    out["real_volume"] = df["real_volume"].astype(float) if "real_volume" in df.columns else float("nan")

    return out[STANDARD_COLUMNS].sort_values("timestamp").reset_index(drop=True)


def standardize_mt5_rates(df: pd.DataFrame, *, symbol: str, timeframe: str) -> pd.DataFrame:
    """Map MT5 copy_rates_*() output onto the standardized internal schema."""
    missing = [c for c in REQUIRED_MT5_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"MT5 rates missing expected columns {missing}; found {list(df.columns)}")

    out = pd.DataFrame(index=df.index)
    out["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
    out["symbol"] = symbol
    out["timeframe"] = timeframe
    out["open"] = df["open"].astype(float)
    out["high"] = df["high"].astype(float)
    out["low"] = df["low"].astype(float)
    out["close"] = df["close"].astype(float)
    out["tick_volume"] = df["tick_volume"].astype(float)
    out["spread"] = df["spread"].astype(float)
    out["real_volume"] = df["real_volume"].astype(float)

    return out[STANDARD_COLUMNS].sort_values("timestamp").reset_index(drop=True)
