import pytest

from src.data.csv_loader import load_csv, raw_csv_path


def test_load_csv_reads_and_standardizes(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "EURUSD_H1.csv").write_text(
        "timestamp,open,high,low,close,volume\n"
        "2024-01-01T00:00:00Z,1.10,1.12,1.09,1.11,100\n"
        "2024-01-01T01:00:00Z,1.11,1.13,1.10,1.12,120\n"
    )

    df = load_csv("EURUSD", "H1", raw_dir=raw_dir)

    assert len(df) == 2
    assert df["symbol"].unique().tolist() == ["EURUSD"]
    assert df["timeframe"].unique().tolist() == ["H1"]
    assert df["close"].tolist() == [1.11, 1.12]


def test_load_csv_missing_file_raises_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="No raw CSV found"):
        load_csv("EURUSD", "H1", raw_dir=tmp_path)


def test_raw_csv_path_uses_naming_convention(tmp_path):
    path = raw_csv_path("GBPUSD", "M15", raw_dir=tmp_path)
    assert path == tmp_path / "GBPUSD_M15.csv"
