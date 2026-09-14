"""Guards the single-source-of-truth property BaseStrategy relies on:
generate_signal() (used live/paper, one bar at a time) must always agree
with generate_signals_vectorized() (used by the Phase 6 vectorbt research
engine, whole series at once) — they are supposed to be the same
computation viewed two ways, never two independent implementations of
"the same" rule that could quietly drift apart. This also implicitly
guards against look-ahead bias: since generate_signal(window) recomputes
generate_signals_vectorized() on a *truncated* window, agreement with the
full-series table only holds if every indicator is purely backward-looking.
"""
import math

import pandas as pd
import pytest

from src.features.engine import compute_features
from src.strategies import STRATEGY_REGISTRY, create_strategy
from tests.test_strategies_integration import _synthetic_candles


@pytest.fixture(scope="module")
def feature_df():
    return compute_features(_synthetic_candles(n=400, seed=1))


def _optional_float(value):
    return None if pd.isna(value) else float(value)


def _approx_or_both_none(a, b):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return a == pytest.approx(b)


@pytest.mark.parametrize("family", sorted(STRATEGY_REGISTRY))
def test_scalar_generate_signal_matches_vectorized_table(family, feature_df):
    strat = create_strategy(family)
    table = strat.generate_signals_vectorized(feature_df)

    checked = 0
    for i in range(150, len(feature_df), 7):  # sample every 7th bar past warmup
        window = feature_df.iloc[: i + 1]
        signal = strat.generate_signal(window)
        row = table.iloc[i]

        assert signal.direction.value == row["direction"], f"row {i}: direction mismatch"
        assert signal.confidence == pytest.approx(float(row["confidence"])), f"row {i}: confidence mismatch"
        assert _approx_or_both_none(signal.stop_loss, _optional_float(row["stop_loss"])), f"row {i}: stop_loss mismatch"
        assert _approx_or_both_none(
            signal.take_profit, _optional_float(row["take_profit"])
        ), f"row {i}: take_profit mismatch"
        checked += 1

    assert checked > 20
