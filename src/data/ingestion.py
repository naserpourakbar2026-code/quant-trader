"""Phase 2: data ingestion orchestration.

Dispatches to the CSV or MT5 loader based on config/settings.yaml
(`data.source`), writes standardized candles to data/processed/, and
reports what happened per symbol/timeframe. Never fabricates data for a
source that is missing or unreachable — it reports "missing"/"error"
instead.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.core.config import PROJECT_ROOT, load_settings
from src.core.logging import get_system_logger
from src.data.csv_loader import load_csv
from src.data.mt5_loader import MT5UnavailableError


@dataclass
class IngestionResult:
    symbol: str
    timeframe: str
    status: str  # "ok" | "missing" | "error"
    rows: int
    message: str = ""


def processed_csv_path(symbol: str, timeframe: str, processed_dir: Path | None = None) -> Path:
    directory = processed_dir if processed_dir is not None else (PROJECT_ROOT / load_settings().paths.data_processed)
    return Path(directory) / f"{symbol}_{timeframe}.csv"


def ingest_one(
    symbol: str,
    timeframe: str,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
) -> IngestionResult:
    settings = load_settings()
    logger = get_system_logger()
    source = settings.data.source

    try:
        if source == "csv":
            df = load_csv(symbol, timeframe, raw_dir=raw_dir)
        elif source == "mt5":
            if start is None or end is None:
                raise ValueError("MT5 source requires both --start and --end dates for download-data")
            from src.data import mt5_loader  # lazy: keeps MT5 import optional

            mt5_loader.connect()
            try:
                df = mt5_loader.fetch_rates(symbol, timeframe, start, end)
            finally:
                mt5_loader.disconnect()
        else:
            raise ValueError(f"Unsupported data source {source!r}")
    except FileNotFoundError as exc:
        logger.warning(str(exc))
        return IngestionResult(symbol, timeframe, status="missing", rows=0, message=str(exc))
    except (MT5UnavailableError, ValueError) as exc:
        logger.warning(str(exc))
        return IngestionResult(symbol, timeframe, status="error", rows=0, message=str(exc))

    if df.empty:
        message = f"No candles returned for {symbol}/{timeframe} in the requested range."
        logger.warning(message)
        return IngestionResult(symbol, timeframe, status="missing", rows=0, message=message)

    out_path = processed_csv_path(symbol, timeframe, processed_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Ingested {len(df)} rows for {symbol}/{timeframe} -> {out_path}")
    return IngestionResult(symbol, timeframe, status="ok", rows=len(df))


def data_version_for(symbol: str, timeframe: str, processed_dir: Path | None = None) -> str:
    """A short, stable fingerprint of the processed CSV backing this
    symbol/timeframe (CLAUDE.md Section 36) — lets an experiment record
    detect if its underlying data has since changed. "unknown" if the
    file doesn't exist (never fabricated)."""
    path = processed_csv_path(symbol, timeframe, processed_dir)
    if not path.exists():
        return "unknown"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest[:16]


def run_download(
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
) -> list[IngestionResult]:
    settings = load_settings()
    syms = symbols or settings.data.symbols
    tfs = timeframes or settings.data.timeframes
    return [
        ingest_one(s, t, start=start, end=end, raw_dir=raw_dir, processed_dir=processed_dir)
        for s in syms
        for t in tfs
    ]
