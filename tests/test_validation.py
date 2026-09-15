import pandas as pd
import pytest

from src.data.schema import STANDARD_COLUMNS
from src.data.validation import (
    check_timezone_consistency,
    run_validation,
    validate_candles,
    validate_symbol_timeframe,
)


def _base_df(n=5, start="2024-01-02T00:00:00Z", freq="1h", symbol="EURUSD", timeframe="H1"):
    """2024-01-02 is a Tuesday — safely inside the trading week."""
    ts = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": symbol,
            "timeframe": timeframe,
            "open": 1.10,
            "high": 1.12,
            "low": 1.09,
            "close": 1.11,
            "tick_volume": 100.0,
            "spread": 2.0,
            "real_volume": float("nan"),
        }
    )[STANDARD_COLUMNS]


def test_validate_candles_detects_duplicate_rows_and_timestamps():
    df = pd.concat([_base_df(n=5), _base_df(n=5).iloc[[0]]], ignore_index=True)
    report = validate_candles(df)
    assert report.duplicate_timestamps == 1
    assert report.duplicate_rows == 1


def test_validate_candles_detects_duplicate_timestamp_without_duplicate_row():
    df = _base_df(n=5)
    dup_row = df.iloc[[0]].copy()
    dup_row["close"] = 9.99  # same timestamp, different value: not a full duplicate row
    df2 = pd.concat([df, dup_row], ignore_index=True)
    report = validate_candles(df2)
    assert report.duplicate_timestamps == 1
    assert report.duplicate_rows == 0


def test_validate_candles_detects_invalid_ohlc_and_nonpositive_price():
    df = _base_df(n=3)
    df.loc[0, "high"] = 1.0  # high below open/close/low -> impossible candle
    df.loc[1, "close"] = -1.0  # negative price
    report = validate_candles(df)
    assert report.invalid_candles == 2
    assert report.negative_or_zero_prices == 1


def test_validate_candles_detects_missing_candles_within_weekday():
    df = _base_df(n=2)
    df.loc[1, "timestamp"] = pd.Timestamp("2024-01-02T03:00:00Z")  # skips 01:00 and 02:00
    report = validate_candles(df)
    assert report.missing_candles == 2


def test_validate_candles_no_missing_when_contiguous():
    df = _base_df(n=10)
    report = validate_candles(df)
    assert report.missing_candles == 0


def test_validate_candles_flags_weekend_candles():
    df = _base_df(n=1, start="2024-01-06T12:00:00Z")  # Saturday
    report = validate_candles(df)
    assert report.weekend_candles == 1


def test_validate_candles_detects_abnormal_spread():
    df = _base_df(n=30)
    df.loc[df.index[-1], "spread"] = 50.0  # 29 candles at spread=2.0, one at 50.0
    report = validate_candles(df)
    assert report.abnormal_spreads == 1
    assert report.max_spread == 50.0
    assert report.avg_spread == pytest.approx((2.0 * 29 + 50.0) / 30)


def test_validate_candles_rejects_empty_dataframe():
    with pytest.raises(ValueError, match="empty"):
        validate_candles(_base_df(n=0))


def test_check_timezone_consistency_flags_minority_format():
    raw = pd.Series(
        ["2024-01-01T00:00:00Z"] * 8
        + ["2024-01-01T01:00:00+00:00"]
        + ["2024-01-01 02:00:00"]
    )
    assert check_timezone_consistency(raw) == 2


def test_check_timezone_consistency_all_consistent_returns_zero():
    raw = pd.Series(["2024-01-01T00:00:00Z"] * 5)
    assert check_timezone_consistency(raw) == 0


def test_validate_symbol_timeframe_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_symbol_timeframe("EURUSD", "H1", raw_dir=tmp_path)


def test_run_validation_skips_missing_and_reports_present(tmp_path):
    (tmp_path / "EURUSD_H1.csv").write_text(
        "timestamp,open,high,low,close,volume\n"
        "2024-01-02T00:00:00Z,1.10,1.12,1.09,1.11,100\n"
        "2024-01-02T01:00:00Z,1.11,1.13,1.10,1.12,120\n"
    )

    reports, skipped = run_validation(symbols=["EURUSD", "GBPUSD"], timeframes=["H1"], raw_dir=tmp_path)

    assert len(reports) == 1
    assert reports[0].symbol == "EURUSD"
    assert len(skipped) == 1
    assert skipped[0].symbol == "GBPUSD"


def test_to_text_matches_brief_template_format():
    report = validate_candles(_base_df(n=3))
    text = report.to_text()
    assert text.startswith("DATA QUALITY REPORT")
    assert "Symbol: EURUSD | Timeframe: H1" in text
    assert "Rows: 3" in text
