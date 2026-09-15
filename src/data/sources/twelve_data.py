"""Twelve Data historical OHLCV loader (CLAUDE.md Section 2, source C).

A third data source alongside CSV (Section 2.B) and MT5 (Section 2.A,
Windows-only): https://twelvedata.com's REST `/time_series` endpoint,
covering forex/CFD pairs without a running MT5 terminal.

Rate limiting and retry-with-backoff reuse the same primitives already
built for the broker adapters (src.brokers.resilience, CLAUDE.md Section
25) instead of re-implementing them here.
"""
from __future__ import annotations

import os

import pandas as pd
import requests

from src.brokers.resilience import RateLimiter, retry_with_backoff
from src.data.schema import STANDARD_COLUMNS

API_BASE_URL = "https://api.twelvedata.com/time_series"

TIMEFRAME_INTERVALS: dict[str, str] = {
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
}


class TwelveDataError(RuntimeError):
    """Base error for anything that goes wrong talking to Twelve Data."""


class TwelveDataAuthError(TwelveDataError):
    """Missing or rejected API key. Never retriable."""


class TwelveDataRateLimitError(TwelveDataError):
    """Free-plan credit/rate limit hit (CLAUDE.md Section 25). Retriable."""


class TwelveDataSymbolError(TwelveDataError):
    """Symbol or interval Twelve Data doesn't recognize. Never retriable."""


def _api_key(env_var: str) -> str:
    key = os.environ.get(env_var, "").strip()
    if not key:
        raise TwelveDataAuthError(
            f"{env_var} is not set. Put a real Twelve Data API key in .env (see .env.example) "
            "-- never hard-code it in source or config files."
        )
    return key


def _to_twelvedata_symbol(symbol: str) -> str:
    """'EURUSD' -> 'EUR/USD' (Twelve Data's forex pair format)."""
    symbol = symbol.upper()
    if "/" in symbol:
        return symbol
    if len(symbol) != 6:
        raise TwelveDataSymbolError(
            f"Cannot map {symbol!r} to a Twelve Data forex pair (expected 6 letters, e.g. EURUSD)."
        )
    return f"{symbol[:3]}/{symbol[3:]}"


def _raise_for_error_payload(payload: dict) -> None:
    if not isinstance(payload, dict) or payload.get("status") != "error":
        return
    code = payload.get("code")
    message = str(payload.get("message", "Twelve Data returned an error"))
    lowered = message.lower()
    if code == 401 or "api key" in lowered:
        raise TwelveDataAuthError(message)
    if code == 429 or "credit" in lowered or "rate limit" in lowered:
        raise TwelveDataRateLimitError(message)
    if code in (400, 404) or "symbol" in lowered:
        raise TwelveDataSymbolError(message)
    raise TwelveDataError(message)


def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    *,
    start: str | None = None,
    end: str | None = None,
    output_size: int = 5000,
    api_key_env: str = "TWELVE_DATA_API_KEY",
    requests_per_second: float = 1.0,
    timeout_seconds: float = 10.0,
    max_retries: int = 3,
    retry_base_delay_seconds: float = 1.0,
    rate_limiter: RateLimiter | None = None,
    session=None,
) -> pd.DataFrame:
    """Fetch historical candles for one symbol/timeframe from Twelve Data
    and map them onto the standardized internal schema.

    Forex pairs have no real trading volume on Twelve Data (OTC market) --
    tick_volume/real_volume come back as NaN rather than a fabricated
    number (CLAUDE.md Section 3: detect, never repair).
    """
    if timeframe not in TIMEFRAME_INTERVALS:
        raise ValueError(f"Unsupported timeframe {timeframe!r}; expected one of {sorted(TIMEFRAME_INTERVALS)}")

    api_key = _api_key(api_key_env)
    td_symbol = _to_twelvedata_symbol(symbol)
    http = session or requests
    limiter = rate_limiter or RateLimiter(requests_per_second)

    params = {
        "symbol": td_symbol,
        "interval": TIMEFRAME_INTERVALS[timeframe],
        "apikey": api_key,
        "timezone": "UTC",
        "order": "ASC",
        "outputsize": output_size,
    }
    if start:
        params["start_date"] = start
    if end:
        params["end_date"] = end

    def _call() -> dict:
        limiter.acquire()
        response = http.get(API_BASE_URL, params=params, timeout=timeout_seconds)
        if response.status_code == 401:
            raise TwelveDataAuthError("Twelve Data rejected the API key (HTTP 401).")
        if response.status_code == 429:
            raise TwelveDataRateLimitError("Twelve Data rate limit hit (HTTP 429).")
        response.raise_for_status()
        payload = response.json()
        _raise_for_error_payload(payload)
        return payload

    payload = retry_with_backoff(
        _call,
        max_retries=max_retries,
        base_delay_seconds=retry_base_delay_seconds,
        retriable_exceptions=(
            TwelveDataRateLimitError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ),
    )

    values = payload.get("values") or []
    if not values:
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    raw = pd.DataFrame(values)
    out = pd.DataFrame(index=raw.index)
    out["timestamp"] = pd.to_datetime(raw["datetime"], utc=True)
    out["symbol"] = symbol
    out["timeframe"] = timeframe
    out["open"] = raw["open"].astype(float)
    out["high"] = raw["high"].astype(float)
    out["low"] = raw["low"].astype(float)
    out["close"] = raw["close"].astype(float)
    has_volume = "volume" in raw.columns
    out["tick_volume"] = raw["volume"].astype(float) if has_volume else float("nan")
    out["spread"] = float("nan")  # Twelve Data does not provide spread
    out["real_volume"] = raw["volume"].astype(float) if has_volume else float("nan")

    return out[STANDARD_COLUMNS].sort_values("timestamp").reset_index(drop=True)
