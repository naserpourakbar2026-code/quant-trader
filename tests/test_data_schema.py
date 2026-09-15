import math

import pandas as pd
import pytest

from src.data.schema import STANDARD_COLUMNS, standardize_csv_rows, standardize_mt5_rates


def test_standardize_csv_rows_maps_minimum_columns_and_fills_nan_for_optional():
    df = pd.DataFrame(
        {
            "timestamp": ["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"],
            "open": [1.1, 1.2],
            "high": [1.15, 1.25],
            "low": [1.05, 1.15],
            "close": [1.12, 1.22],
            "volume": [100, 150],
        }
    )
    out = standardize_csv_rows(df, symbol="EURUSD", timeframe="H1")

    assert list(out.columns) == STANDARD_COLUMNS
    assert out["symbol"].unique().tolist() == ["EURUSD"]
    assert out["timeframe"].unique().tolist() == ["H1"]
    assert out["tick_volume"].tolist() == [100.0, 150.0]
    assert all(math.isnan(v) for v in out["spread"])
    assert all(math.isnan(v) for v in out["real_volume"])


def test_standardize_csv_rows_uses_optional_columns_when_present():
    df = pd.DataFrame(
        {
            "timestamp": ["2024-01-01T00:00:00Z"],
            "open": [1.1],
            "high": [1.15],
            "low": [1.05],
            "close": [1.12],
            "volume": [100],
            "tick_volume": [123],
            "spread": [2],
            "real_volume": [999],
        }
    )
    out = standardize_csv_rows(df, symbol="EURUSD", timeframe="H1")
    assert out.loc[0, "tick_volume"] == 123.0
    assert out.loc[0, "spread"] == 2.0
    assert out.loc[0, "real_volume"] == 999.0


def test_standardize_csv_rows_rejects_missing_required_column():
    df = pd.DataFrame({"timestamp": ["2024-01-01T00:00:00Z"], "open": [1.1]})
    with pytest.raises(ValueError, match="missing required columns"):
        standardize_csv_rows(df, symbol="EURUSD", timeframe="H1")


def test_standardize_mt5_rates_maps_epoch_time_and_columns():
    df = pd.DataFrame(
        {
            "time": [1704067200],  # 2024-01-01T00:00:00Z
            "open": [1.1],
            "high": [1.15],
            "low": [1.05],
            "close": [1.12],
            "tick_volume": [100],
            "spread": [2],
            "real_volume": [500],
        }
    )
    out = standardize_mt5_rates(df, symbol="EURUSD", timeframe="H1")
    assert list(out.columns) == STANDARD_COLUMNS
    assert out.loc[0, "timestamp"] == pd.Timestamp("2024-01-01T00:00:00Z")


def test_standardize_mt5_rates_rejects_missing_column():
    df = pd.DataFrame({"time": [1704067200], "open": [1.1]})
    with pytest.raises(ValueError, match="missing expected columns"):
        standardize_mt5_rates(df, symbol="EURUSD", timeframe="H1")
