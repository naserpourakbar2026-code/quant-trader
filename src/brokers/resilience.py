"""Broker-agnostic API safety primitives (CLAUDE.md Section 25): rate
limiting and retry-with-backoff. Kept separate from any one adapter so
both the generic REST adapter (Phase 13) and any future adapter can
share them, and so they're testable without a network call in sight —
every timing source is injectable.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, TypeVar

T = TypeVar("T")


class RateLimiter:
    """Enforces a minimum spacing between calls (requests_per_second from
    brokers.yaml's `rate_limit` block) — a simple, easy-to-reason-about
    limiter, not a token bucket with burst allowance; that's a deliberate
    "basic" choice, not an oversight."""

    def __init__(self, requests_per_second: float, *, time_fn: Callable[[], float] = time.monotonic,
                 sleep_fn: Callable[[float], None] = time.sleep) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be > 0")
        self._min_interval = 1.0 / requests_per_second
        self._time_fn = time_fn
        self._sleep_fn = sleep_fn
        self._last_call: float | None = None

    def acquire(self) -> None:
        now = self._time_fn()
        if self._last_call is not None:
            elapsed = now - self._last_call
            remaining = self._min_interval - elapsed
            if remaining > 0:
                self._sleep_fn(remaining)
                now = self._time_fn()
        self._last_call = now


@dataclass
class RetryConfig:
    max_retries: int = 3
    base_delay_seconds: float = 1.0


def retry_with_backoff(
    func: Callable[[], T],
    *,
    max_retries: int,
    base_delay_seconds: float,
    retriable_exceptions: tuple[type[Exception], ...],
    sleep_fn: Callable[[float], None] = time.sleep,
) -> T:
    """Calls `func()`, retrying on any of `retriable_exceptions` with
    exponential backoff (base_delay * 2**attempt) up to `max_retries`
    times. The last failure re-raises rather than swallowing it — a
    caller that keeps failing after every retry needs to know, not get a
    silent None back."""
    attempt = 0
    while True:
        try:
            return func()
        except retriable_exceptions:
            if attempt >= max_retries:
                raise
            sleep_fn(base_delay_seconds * (2 ** attempt))
            attempt += 1
