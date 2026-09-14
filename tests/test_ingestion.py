import pandas as pd

from src.data.ingestion import ingest_one, processed_csv_path, run_download


def _write_raw_csv(raw_dir, symbol, timeframe, rows=2):
    raw_dir.mkdir(parents=True, exist_ok=True)
    lines = ["timestamp,open,high,low,close,volume"]
    for i in range(rows):
        lines.append(f"2024-01-0{i + 1}T00:00:00Z,1.1{i},1.2{i},1.0{i},1.15{i},{100 + i}")
    (raw_dir / f"{symbol}_{timeframe}.csv").write_text("\n".join(lines) + "\n")


def test_ingest_one_writes_processed_csv_for_csv_source(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    _write_raw_csv(raw_dir, "EURUSD", "H1", rows=3)

    result = ingest_one("EURUSD", "H1", raw_dir=raw_dir, processed_dir=processed_dir)

    assert result.status == "ok"
    assert result.rows == 3
    out_path = processed_csv_path("EURUSD", "H1", processed_dir=processed_dir)
    assert out_path.exists()
    df = pd.read_csv(out_path)
    assert len(df) == 3


def test_ingest_one_reports_missing_when_raw_file_absent(tmp_path):
    result = ingest_one("EURUSD", "H1", raw_dir=tmp_path / "raw", processed_dir=tmp_path / "processed")
    assert result.status == "missing"
    assert result.rows == 0
    assert "No raw CSV found" in result.message


def test_run_download_iterates_requested_symbols_and_timeframes(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    _write_raw_csv(raw_dir, "EURUSD", "H1")
    _write_raw_csv(raw_dir, "GBPUSD", "H1")

    results = run_download(
        symbols=["EURUSD", "GBPUSD"],
        timeframes=["H1"],
        raw_dir=raw_dir,
        processed_dir=processed_dir,
    )

    assert len(results) == 2
    assert all(r.status == "ok" for r in results)
