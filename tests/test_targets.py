import pandas as pd
import pytest

from src.strategies import targets


def test_r_multiple_target_long_and_short():
    assert targets.r_multiple_target(100.0, stop_loss=98.0, direction="LONG", r_multiple=2.0) == pytest.approx(104.0)
    assert targets.r_multiple_target(100.0, stop_loss=102.0, direction="SHORT", r_multiple=2.0) == pytest.approx(96.0)


def test_atr_target_long_and_short():
    assert targets.atr_target(100.0, atr_value=2.0, direction="LONG", multiplier=3.0) == pytest.approx(106.0)
    assert targets.atr_target(100.0, atr_value=2.0, direction="SHORT", multiplier=3.0) == pytest.approx(94.0)


def test_mean_reversion_target_returns_mean_price():
    assert targets.mean_reversion_target(101.5) == pytest.approx(101.5)


def test_r_multiple_target_vectorized_matches_scalar_per_row():
    entry = pd.Series([100.0, 200.0])
    stop_loss = pd.Series([98.0, 210.0])
    direction = pd.Series(["LONG", "SHORT"])
    result = targets.r_multiple_target(entry, stop_loss, direction, r_multiple=2.0)
    expected = [
        targets.r_multiple_target(100.0, 98.0, "LONG", 2.0),
        targets.r_multiple_target(200.0, 210.0, "SHORT", 2.0),
    ]
    assert isinstance(result, pd.Series)
    assert result.tolist() == pytest.approx(expected)
