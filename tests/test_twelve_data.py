"""Mocked unit tests for the Twelve Data historical data source
(CLAUDE.md Section 2, source C). No real network call is ever made here
-- every test stubs a fake `requests`-like session, mirroring the pattern
already used for the generic REST broker adapter (tests/test_generic_rest_adapter.py).
"""
from __future__ import annotations

import pytest
import requests

from src.data.schema import STANDARD_COLUMNS
from src.data.sources.twelve_data import (
    TwelveDataAuthError,
    TwelveDataRateLimitError,
    TwelveDataSymbolError,
    fetch_ohlcv,
)


class _FakeResponse:
    def __init__(self, status_code: int, json_data: dict | None = None):
        self.status_code = status_code
        self._json_data = json_data if json_data is not None else {}
        self.text = str(self._json_data)

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")


class _FakeSession:
    """responses: list of _FakeResponse, returned in order across calls
    (one per attempt -- lets a test script "429 then 200")."""

    def __init__(self, responses: list[_FakeResponse]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def get(self, url, *, params=None, timeout=None):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return self._responses[len(self.calls) - 1]


class _NoSleepLimiter:
    """A rate limiter double that never actually sleeps, so tests run
    fast, but still records how many times it was asked to acquire."""

    def __init__(self):
        self.acquire_calls = 0

    def acquire(self):
        self.acquire_calls += 1


_SUCCESS_PAYLOAD = {
    "status": "ok",
    "values": [
        {"datetime": "2024-01-01 00:00:00", "open": "1.10", "high": "1.12", "low": "1.09", "close": "1.11"},
        {"datetime": "2024-01-01 01:00:00", "open": "1.11", "high": "1.13", "low": "1.10", "close": "1.12"},
    ],
}


@pytest.fixture(autouse=True)
def _no_real_api_key(monkeypatch):
    monkeypatch.delenv("TWELVE_DATA_API_KEY", raising=False)


def test_fetch_ohlcv_raises_auth_error_when_api_key_missing():
    with pytest.raises(TwelveDataAuthError, match="TWELVE_DATA_API_KEY"):
        fetch_ohlcv("EURUSD", "H1")


def test_fetch_ohlcv_rejects_unsupported_timeframe(monkeypatch):
    monkeypatch.setenv("TWELVE_DATA_API_KEY", "fake-key")
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        fetch_ohlcv("EURUSD", "W1")


def test_fetch_ohlcv_rejects_malformed_symbol(monkeypatch):
    monkeypatch.setenv("TWELVE_DATA_API_KEY", "fake-key")
    with pytest.raises(TwelveDataSymbolError, match="EU"):
        fetch_ohlcv("EU", "H1")


def test_fetch_ohlcv_maps_successful_response_to_standard_schema(monkeypatch):
    monkeypatch.setenv("TWELVE_DATA_API_KEY", "fake-key")
    session = _FakeSession([_FakeResponse(200, _SUCCESS_PAYLOAD)])
    limiter = _NoSleepLimiter()

    df = fetch_ohlcv("EURUSD", "H1", session=session, rate_limiter=limiter)

    assert list(df.columns) == STANDARD_COLUMNS
    assert len(df) == 2
    assert df.loc[0, "symbol"] == "EURUSD"
    assert df.loc[0, "timeframe"] == "H1"
    assert df.loc[0, "close"] == 1.11
    # forex has no real volume on Twelve Data -- NaN, never fabricated
    assert df["tick_volume"].isna().all()
    assert df["real_volume"].isna().all()
    assert df["spread"].isna().all()
    assert limiter.acquire_calls == 1
    assert session.calls[0]["params"]["symbol"] == "EUR/USD"
    assert session.calls[0]["params"]["apikey"] == "fake-key"


def test_fetch_ohlcv_keeps_real_volume_when_the_api_provides_it(monkeypatch):
    monkeypatch.setenv("TWELVE_DATA_API_KEY", "fake-key")
    payload = {
        "status": "ok",
        "values": [
            {"datetime": "2024-01-01 00:00:00", "open": "1.1", "high": "1.2", "low": "1.0", "close": "1.15", "volume": "1234"},
        ],
    }
    session = _FakeSession([_FakeResponse(200, payload)])

    df = fetch_ohlcv("EURUSD", "H1", session=session, rate_limiter=_NoSleepLimiter())

    assert df.loc[0, "tick_volume"] == 1234.0
    assert df.loc[0, "real_volume"] == 1234.0


def test_fetch_ohlcv_empty_result_returns_standard_empty_frame(monkeypatch):
    monkeypatch.setenv("TWELVE_DATA_API_KEY", "fake-key")
    session = _FakeSession([_FakeResponse(200, {"status": "ok", "values": []})])

    df = fetch_ohlcv("EURUSD", "H1", session=session, rate_limiter=_NoSleepLimiter())

    assert df.empty
    assert list(df.columns) == STANDARD_COLUMNS


def test_fetch_ohlcv_raises_auth_error_on_http_401(monkeypatch):
    monkeypatch.setenv("TWELVE_DATA_API_KEY", "wrong-key")
    session = _FakeSession([_FakeResponse(401, {"status": "error", "code": 401, "message": "Invalid API Key"})])

    with pytest.raises(TwelveDataAuthError):
        fetch_ohlcv("EURUSD", "H1", session=session, rate_limiter=_NoSleepLimiter())


def test_fetch_ohlcv_raises_symbol_error_from_api_payload(monkeypatch):
    monkeypatch.setenv("TWELVE_DATA_API_KEY", "fake-key")
    payload = {"status": "error", "code": 400, "message": "**symbol** parameter is missing or invalid"}
    session = _FakeSession([_FakeResponse(200, payload)])

    with pytest.raises(TwelveDataSymbolError):
        fetch_ohlcv("EURUSD", "H1", session=session, rate_limiter=_NoSleepLimiter())


def test_fetch_ohlcv_retries_on_rate_limit_then_succeeds(monkeypatch):
    monkeypatch.setenv("TWELVE_DATA_API_KEY", "fake-key")
    session = _FakeSession([_FakeResponse(429, {"status": "error", "code": 429}), _FakeResponse(200, _SUCCESS_PAYLOAD)])
    limiter = _NoSleepLimiter()

    df = fetch_ohlcv(
        "EURUSD", "H1", session=session, rate_limiter=limiter,
        max_retries=3, retry_base_delay_seconds=0.0,
    )

    assert len(df) == 2
    assert len(session.calls) == 2
    assert limiter.acquire_calls == 2  # rate limiter re-acquired on every attempt


def test_fetch_ohlcv_gives_up_after_max_retries_on_persistent_rate_limit(monkeypatch):
    monkeypatch.setenv("TWELVE_DATA_API_KEY", "fake-key")
    session = _FakeSession([_FakeResponse(429, {"status": "error", "code": 429}) for _ in range(10)])

    with pytest.raises(TwelveDataRateLimitError):
        fetch_ohlcv(
            "EURUSD", "H1", session=session, rate_limiter=_NoSleepLimiter(),
            max_retries=2, retry_base_delay_seconds=0.0,
        )

    assert len(session.calls) == 3  # initial attempt + 2 retries
