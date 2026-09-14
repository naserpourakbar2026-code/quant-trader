"""Phase 3: data quality validation (CLAUDE.md Section 3).

Detects — and only ever detects — issues in standardized candle data:
duplicate timestamps/rows, missing candles, out-of-order timestamps,
impossible OHLC values, negative/zero prices, abnormal spreads, timezone
inconsistencies, weekend data. Nothing here modifies or "fixes" the data;
that is explicitly out of scope by design.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.config import ValidationConfig, load_settings

TIMEFRAME_MINUTES: dict[str, int] = {"M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240}


@dataclass
class DataQualityReport:
    symbol: str
    timeframe: str
    rows: int
    duplicate_timestamps: int
    duplicate_rows: int
    out_of_order: int
    missing_candles: int
    invalid_candles: int
    negative_or_zero_prices: int
    abnormal_spreads: int
    weekend_candles: int
    timezone_inconsistencies: int
    avg_spread: float | None
    max_spread: float | None
    date_range_start: pd.Timestamp | None
    date_range_end: pd.Timestamp | None

    @staticmethod
    def _fmt_num(value: float | None) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "n/a"
        return f"{value:.5f}"

    def _fmt_range(self) -> str:
        if self.date_range_start is None or self.date_range_end is None:
            return "n/a"
        return f"{self.date_range_start.date()} to {self.date_range_end.date()}"

    def to_text(self) -> str:
        return "\n".join(
            [
                "DATA QUALITY REPORT",
                f"Symbol: {self.symbol} | Timeframe: {self.timeframe}",
                f"Rows: {self.rows} | Missing candles: {self.missing_candles} | "
                f"Duplicates: {self.duplicate_timestamps}",
                f"Invalid candles: {self.invalid_candles} | Avg spread: {self._fmt_num(self.avg_spread)} | "
                f"Max spread: {self._fmt_num(self.max_spread)} | Date range: {self._fmt_range()}",
                f"Out-of-order timestamps: {self.out_of_order} | Duplicate rows: {self.duplicate_rows}",
                f"Negative/zero prices: {self.negative_or_zero_prices} | "
                f"Abnormal spreads: {self.abnormal_spreads}",
                f"Weekend candles: {self.weekend_candles} | "
                f"Timezone inconsistencies: {self.timezone_inconsistencies}",
            ]
        )


def _weekend_closed_mask(index: pd.DatetimeIndex, cfg: ValidationConfig) -> np.ndarray:
    """Vectorized: True where a timestamp falls inside the configured
    weekend market-closure window (broker/calendar specific)."""
    dow = index.dayofweek
    hour = index.hour
    closed = (dow == cfg.weekend_close_day) & (hour >= cfg.weekend_close_hour)
    closed |= (dow == cfg.weekend_open_day) & (hour < cfg.weekend_open_hour)
    # any full day strictly between the close day and open day is closed too
    if cfg.weekend_close_day < cfg.weekend_open_day:
        closed |= (dow > cfg.weekend_close_day) & (dow < cfg.weekend_open_day)
    return closed


def check_timezone_consistency(raw_timestamps: pd.Series) -> int:
    """Count raw timestamp strings whose UTC-offset notation disagrees with
    the majority representation in the column (mixing 'Z', explicit
    offsets, and naive/no-offset strings is a real-world broker export bug).
    Must run on the *raw* string column — parsing to tz-aware timestamps
    normalizes this away, which is exactly the silent "repair" Section 3
    forbids happening unnoticed.
    """
    strs = raw_timestamps.astype(str)
    has_z = strs.str.endswith("Z")
    has_offset = strs.str.contains(r"[+-]\d{2}:?\d{2}$", regex=True) & ~has_z
    naive = ~has_z & ~has_offset
    majority = max(int(has_z.sum()), int(has_offset.sum()), int(naive.sum()))
    return int(len(strs) - majority)


def validate_candles(df: pd.DataFrame, *, timezone_inconsistencies: int = 0) -> DataQualityReport:
    """Run all detection checks on one symbol/timeframe's standardized
    candles (see src/data/schema.py for the expected columns)."""
    if df.empty:
        raise ValueError("Cannot validate an empty DataFrame")

    settings = load_settings()
    cfg = settings.validation

    symbol = str(df["symbol"].iloc[0])
    timeframe = str(df["timeframe"].iloc[0])
    rows = len(df)

    duplicate_timestamps = int(df["timestamp"].duplicated().sum())
    duplicate_rows = int(df.duplicated().sum())

    sorted_df = df.sort_values("timestamp")
    ts = pd.DatetimeIndex(sorted_df["timestamp"])
    out_of_order = int((df["timestamp"].diff().dropna() < pd.Timedelta(0)).sum())

    ohlc = sorted_df[["open", "high", "low", "close"]]
    invalid_mask = (
        ohlc.isna().any(axis=1)
        | (sorted_df["high"] < sorted_df[["open", "close", "low"]].max(axis=1))
        | (sorted_df["low"] > sorted_df[["open", "close", "high"]].min(axis=1))
    )
    invalid_candles = int(invalid_mask.sum())
    negative_or_zero_prices = int((ohlc <= 0).any(axis=1).sum())

    spread = sorted_df["spread"].dropna()
    if len(spread) > 1 and spread.std() > 0:
        z_scores = (spread - spread.mean()).abs() / spread.std()
        abnormal_spreads = int((z_scores > cfg.spread_outlier_zscore).sum())
    else:
        abnormal_spreads = 0
    avg_spread = float(spread.mean()) if len(spread) else None
    max_spread = float(spread.max()) if len(spread) else None

    weekend_candles = int(_weekend_closed_mask(ts, cfg).sum())

    missing_candles = 0
    step_minutes = TIMEFRAME_MINUTES.get(timeframe)
    if step_minutes and len(ts) > 1:
        step = pd.Timedelta(minutes=step_minutes)
        full_range = pd.date_range(ts[0], ts[-1], freq=step)
        expected_open = full_range[~_weekend_closed_mask(full_range, cfg)]
        missing_candles = int(len(expected_open) - expected_open.isin(ts).sum())

    return DataQualityReport(
        symbol=symbol,
        timeframe=timeframe,
        rows=rows,
        duplicate_timestamps=duplicate_timestamps,
        duplicate_rows=duplicate_rows,
        out_of_order=out_of_order,
        missing_candles=missing_candles,
        invalid_candles=invalid_candles,
        negative_or_zero_prices=negative_or_zero_prices,
        abnormal_spreads=abnormal_spreads,
        weekend_candles=weekend_candles,
        timezone_inconsistencies=timezone_inconsistencies,
        avg_spread=avg_spread,
        max_spread=max_spread,
        date_range_start=ts[0] if len(ts) else None,
        date_range_end=ts[-1] if len(ts) else None,
    )


def validate_symbol_timeframe(symbol: str, timeframe: str, raw_dir: Path | None = None) -> DataQualityReport:
    """Load one symbol/timeframe's raw CSV and validate it. Raises
    FileNotFoundError (via csv_loader) if no raw data exists yet."""
    from src.data.csv_loader import load_csv, raw_csv_path

    standardized = load_csv(symbol, timeframe, raw_dir=raw_dir)
    raw_df = pd.read_csv(raw_csv_path(symbol, timeframe, raw_dir))
    tz_inconsistencies = check_timezone_consistency(raw_df["timestamp"])
    return validate_candles(standardized, timezone_inconsistencies=tz_inconsistencies)


@dataclass
class ValidationSkip:
    symbol: str
    timeframe: str
    reason: str


def run_validation(
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    raw_dir: Path | None = None,
) -> tuple[list[DataQualityReport], list[ValidationSkip]]:
    """Validate every requested symbol/timeframe. A symbol/timeframe with
    no raw data yet is skipped (reported, not treated as a validation
    finding) rather than failing the whole run."""
    settings = load_settings()
    syms = symbols or settings.data.symbols
    tfs = timeframes or settings.data.timeframes

    reports: list[DataQualityReport] = []
    skipped: list[ValidationSkip] = []
    for symbol in syms:
        for timeframe in tfs:
            try:
                reports.append(validate_symbol_timeframe(symbol, timeframe, raw_dir=raw_dir))
            except FileNotFoundError as exc:
                skipped.append(ValidationSkip(symbol, timeframe, str(exc)))
    return reports, skipped
