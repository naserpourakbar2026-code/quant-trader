"""End-to-end sanity check: Phase 4's feature engine feeding each Phase 5
strategy family on the same realistic synthetic OHLCV series. Not a claim
about profitability (that's Phase 6+) — just that the whole pipeline runs
without error and only ever emits internally-consistent signals.
"""
import numpy as np
import pandas as pd
import pytest

from src.data.schema import STANDARD_COLUMNS
from src.features.engine import compute_features
from src.strategies import STRATEGY_REGISTRY, create_strategy
from src.strategies.base import SignalDirection


def _synthetic_candles(n=400, seed=0, symbol="EURUSD", timeframe="H1"):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-02T00:00:00Z", periods=n, freq="1h")
    # Three phases — sustained uptrend, choppy range, sustained downtrend —
    # so trend, range and breakout conditions all plausibly occur somewhere
    # in the series (a flat random walk rarely sustains a "confirmed trend
    # + fresh breakout" long enough for any of the three strategies to act).
    third = n // 3
    drift = np.concatenate([np.full(third, 0.0015), np.full(third, 0.0), np.full(n - 2 * third, -0.0015)])
    noise_std = np.concatenate([np.full(third, 0.0003), np.full(third, 0.0007), np.full(n - 2 * third, 0.0003)])
    close = 1.10 + np.cumsum(drift + rng.normal(0, 1, size=n) * noise_std)
    high = close + rng.uniform(0.0001, 0.0004, size=n)
    low = close - rng.uniform(0.0001, 0.0004, size=n)
    open_ = close + rng.normal(0, 0.0002, size=n)
    tick_volume = rng.uniform(50, 200, size=n)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": symbol,
            "timeframe": timeframe,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "tick_volume": tick_volume,
            "spread": 1.5,
            "real_volume": float("nan"),
        }
    )[STANDARD_COLUMNS]


@pytest.fixture(scope="module")
def feature_df():
    return compute_features(_synthetic_candles())


@pytest.mark.parametrize("family", sorted(STRATEGY_REGISTRY))
def test_strategy_generates_valid_signal_at_every_bar_past_warmup(family, feature_df):
    strat = create_strategy(family)
    warmup = 150  # past every indicator's warm-up window
    invalid_signals = []

    for i in range(warmup, len(feature_df)):
        window = feature_df.iloc[: i + 1]
        signal = strat.generate_signal(window)
        assert signal.strategy_name == family
        assert signal.direction in (SignalDirection.LONG, SignalDirection.SHORT, SignalDirection.FLAT)
        if not strat.validate_signal(signal, window):
            invalid_signals.append((i, signal))

    assert not invalid_signals, f"{family}: {len(invalid_signals)} invalid signal(s), e.g. {invalid_signals[:3]}"


@pytest.mark.parametrize("family", sorted(STRATEGY_REGISTRY))
def test_strategy_produces_at_least_one_actionable_signal_over_full_series(family, feature_df):
    """Not a profitability claim — just confirms the entry conditions are
    reachable at all on a series designed to contain trends, breakouts and
    ranges (an always-FLAT strategy would be a wiring bug, not caution)."""
    strat = create_strategy(family)
    warmup = 150
    directions = {
        strat.generate_signal(feature_df.iloc[: i + 1]).direction for i in range(warmup, len(feature_df))
    }
    assert directions & {SignalDirection.LONG, SignalDirection.SHORT}
