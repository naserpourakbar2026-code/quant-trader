import pandas as pd
import pytest

from src.strategies import stops


def test_atr_stop_long_and_short():
    assert stops.atr_stop(100.0, atr_value=2.0, direction="LONG", multiplier=2.0) == pytest.approx(96.0)
    assert stops.atr_stop(100.0, atr_value=2.0, direction="SHORT", multiplier=2.0) == pytest.approx(104.0)


def test_fixed_pct_stop_long_and_short():
    assert stops.fixed_pct_stop(100.0, direction="LONG", pct=0.01) == pytest.approx(99.0)
    assert stops.fixed_pct_stop(100.0, direction="SHORT", pct=0.01) == pytest.approx(101.0)


def test_swing_stop_uses_recent_window_extremes():
    df = pd.DataFrame({"high": [10, 12, 11, 15, 13], "low": [5, 6, 4, 8, 7]})
    assert stops.swing_stop(df, direction="LONG", lookback=3) == pytest.approx(4.0)
    assert stops.swing_stop(df, direction="SHORT", lookback=3) == pytest.approx(15.0)


def test_volatility_stop_scales_multiplier_by_regime():
    base = stops.atr_stop(100.0, atr_value=2.0, direction="LONG", multiplier=2.0)
    low = stops.volatility_stop(100.0, atr_value=2.0, direction="LONG", atr_multiplier=2.0, volatility_regime="low_vol")
    normal = stops.volatility_stop(
        100.0, atr_value=2.0, direction="LONG", atr_multiplier=2.0, volatility_regime="normal_vol"
    )
    high = stops.volatility_stop(
        100.0, atr_value=2.0, direction="LONG", atr_multiplier=2.0, volatility_regime="high_vol"
    )
    assert normal == pytest.approx(base)
    assert low > normal  # tighter stop (multiplier scaled down) -> closer to entry
    assert high < normal  # wider stop (multiplier scaled up) -> farther from entry


def test_volatility_stop_unknown_regime_defaults_to_normal_scale():
    result = stops.volatility_stop(100.0, atr_value=2.0, direction="LONG", atr_multiplier=2.0, volatility_regime="nan")
    assert result == pytest.approx(stops.atr_stop(100.0, 2.0, "LONG", 2.0))


def test_atr_stop_vectorized_matches_scalar_per_row():
    entry = pd.Series([100.0, 200.0, 50.0])
    atr = pd.Series([2.0, 5.0, 1.0])
    direction = pd.Series(["LONG", "SHORT", "LONG"])
    result = stops.atr_stop(entry, atr, direction, multiplier=2.0)
    expected = [
        stops.atr_stop(100.0, 2.0, "LONG", 2.0),
        stops.atr_stop(200.0, 5.0, "SHORT", 2.0),
        stops.atr_stop(50.0, 1.0, "LONG", 2.0),
    ]
    assert isinstance(result, pd.Series)
    assert result.tolist() == pytest.approx(expected)


def test_volatility_stop_vectorized_matches_scalar_per_row():
    entry = pd.Series([100.0, 100.0])
    atr = pd.Series([2.0, 2.0])
    direction = pd.Series(["LONG", "LONG"])
    regime = pd.Series(["low_vol", "high_vol"])
    result = stops.volatility_stop(entry, atr, direction, atr_multiplier=2.0, volatility_regime=regime)
    expected = [
        stops.volatility_stop(100.0, 2.0, "LONG", 2.0, "low_vol"),
        stops.volatility_stop(100.0, 2.0, "LONG", 2.0, "high_vol"),
    ]
    assert result.tolist() == pytest.approx(expected)


def test_swing_stop_series_matches_scalar_swing_stop_at_every_row():
    df = pd.DataFrame({"high": [10, 12, 11, 15, 13, 14, 9, 16], "low": [5, 6, 4, 8, 7, 9, 3, 10]})
    direction = pd.Series(["LONG", "SHORT"] * 4)
    lookback = 3

    result = stops.swing_stop_series(df, direction, lookback=lookback)

    for i in range(lookback - 1, len(df)):
        expected = stops.swing_stop(df.iloc[: i + 1], direction.iloc[i], lookback=lookback)
        assert result.iloc[i] == pytest.approx(expected)
    assert result.iloc[: lookback - 1].isna().all()
