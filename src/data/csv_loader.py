"""CSV historical data loader (CLAUDE.md Section 2, source B).

Raw CSV files are expected under data/raw/, named `{SYMBOL}_{TIMEFRAME}.csv`
(e.g. `EURUSD_H1.csv`), with at minimum: timestamp, open, high, low, close,
volume. This module never fetches or fabricates data — it only reads what
is already on disk.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.core.config import PROJECT_ROOT, load_settings
from src.data.schema import standardize_csv_rows


def raw_csv_path(symbol: str, timeframe: str, raw_dir: Path | None = None) -> Path:
    directory = raw_dir if raw_dir is not None else (PROJECT_ROOT / load_settings().paths.data_raw)
    return Path(directory) / f"{symbol}_{timeframe}.csv"


def load_csv(symbol: str, timeframe: str, raw_dir: Path | None = None) -> pd.DataFrame:
    """Read and standardize one raw CSV file. Raises FileNotFoundError if absent."""
    path = raw_csv_path(symbol, timeframe, raw_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"No raw CSV found for {symbol}/{timeframe} at {path}. "
            "Place a CSV there with at least: timestamp,open,high,low,close,volume"
        )
    df = pd.read_csv(path)
    return standardize_csv_rows(df, symbol=symbol, timeframe=timeframe)
