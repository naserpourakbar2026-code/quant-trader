import pytest

from src.brokers.resilience import RateLimiter, retry_with_backoff


class _FakeClock:
    def __init__(self, start=0.0):
        self.now = start
        self.sleeps: list[float] = []

    def time_fn(self) -> float:
        return self.now

    def sleep_fn(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_rate_limiter_sleeps_to_fill_the_minimum_interval():
    clock = _FakeClock()
    limiter = RateLimiter(10.0, time_fn=clock.time_fn, sleep_fn=clock.sleep_fn)  # min interval 0.1s
    limiter.acquire()
    clock.now += 0.02  # only 20ms passed
    limiter.acquire()
    assert clock.sleeps == [pytest.approx(0.08)]


def test_rate_limiter_does_not_sleep_when_enough_time_already_passed():
    clock = _FakeClock()
    limiter = RateLimiter(10.0, time_fn=clock.time_fn, sleep_fn=clock.sleep_fn)
    limiter.acquire()
    clock.now += 1.0
    limiter.acquire()
    assert clock.sleeps == []


def test_rate_limiter_rejects_non_positive_rate():
    with pytest.raises(ValueError):
        RateLimiter(0)
    with pytest.raises(ValueError):
        RateLimiter(-1)


def test_retry_with_backoff_returns_immediately_on_first_success():
    calls = []

    def func():
        calls.append(1)
        return "ok"

    sleeps = []
    result = retry_with_backoff(func, max_retries=3, base_delay_seconds=1.0, retriable_exceptions=(ValueError,), sleep_fn=sleeps.append)
    assert result == "ok"
    assert len(calls) == 1
    assert sleeps == []


def test_retry_with_backoff_retries_then_succeeds_with_exponential_delays():
    attempts = {"n": 0}

    def func():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError("boom")
        return "ok"

    sleeps = []
    result = retry_with_backoff(func, max_retries=5, base_delay_seconds=1.0, retriable_exceptions=(ConnectionError,), sleep_fn=sleeps.append)
    assert result == "ok"
    assert attempts["n"] == 3
    assert sleeps == [1.0, 2.0]  # base_delay * 2**0, base_delay * 2**1


def test_retry_with_backoff_raises_after_exhausting_max_retries():
    def func():
        raise ConnectionError("always fails")

    sleeps = []
    with pytest.raises(ConnectionError):
        retry_with_backoff(func, max_retries=2, base_delay_seconds=0.5, retriable_exceptions=(ConnectionError,), sleep_fn=sleeps.append)
    assert sleeps == [0.5, 1.0]  # 2 retries after the first attempt


def test_retry_with_backoff_does_not_retry_non_retriable_exceptions():
    calls = []

    def func():
        calls.append(1)
        raise ValueError("not retriable here")

    with pytest.raises(ValueError):
        retry_with_backoff(func, max_retries=5, base_delay_seconds=1.0, retriable_exceptions=(ConnectionError,), sleep_fn=lambda s: None)
    assert len(calls) == 1  # never retried
