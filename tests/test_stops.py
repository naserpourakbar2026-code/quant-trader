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
