import math

import pandas as pd
import pytest

from src.features import indicators as ind


def test_ema_constant_series_equals_constant_after_warmup():
    s = pd.Series([5.0] * 15)
    result = ind.ema(s, 10)
    assert result.iloc[:9].isna().all()
    assert (result.iloc[9:] == 5.0).all()


def test_ema_rejects_nonpositive_period():
    with pytest.raises(ValueError):
        ind.ema(pd.Series([1.0, 2.0]), 0)


def test_true_range_uses_high_low_and_prior_close():
    df = pd.DataFrame({"high": [101.0, 105.0], "low": [99.0, 100.0], "close": [100.0, 101.0]})
    tr = ind.true_range(df)
    assert tr.iloc[0] == pytest.approx(2.0)  # no prior close: just high-low
    assert tr.iloc[1] == pytest.approx(5.0)  # high(105) - prior_close(100)


def test_atr_of_constant_true_range_converges_to_that_constant():
    df = pd.DataFrame({"high": [101.0] * 10, "low": [99.0] * 10, "close": [100.0] * 10})
    result = ind.atr(df, 3)
    assert result.iloc[:2].isna().all()
    assert result.iloc[2:].tolist() == pytest.approx([2.0] * 8)


def test_donchian_channel_rolling_extremes():
    df = pd.DataFrame({"high": [1.0, 2.0, 3.0, 4.0, 5.0], "low": [0.0, 1.0, 2.0, 3.0, 4.0]})
    out = ind.donchian_channel(df, 3)
    assert out["donchian_upper"].iloc[:2].isna().all()
    assert out.loc[2, "donchian_upper"] == 3.0
    assert out.loc[2, "donchian_lower"] == 0.0
    assert out.loc[4, "donchian_upper"] == 5.0
    assert out.loc[4, "donchian_lower"] == 2.0
    assert out.loc[4, "donchian_middle"] == 3.5


def test_bollinger_bands_constant_series_has_zero_width():
    bb = ind.bollinger_bands(pd.Series([5.0] * 10), 4, 2.0)
    tail = bb.iloc[4:]
    assert (tail["bb_upper"] == 5.0).all()
    assert (tail["bb_lower"] == 5.0).all()
    assert (tail["bb_width_pct"] == 0.0).all()


def test_rsi_pure_uptrend_is_100_pure_downtrend_is_0():
    up = pd.Series(range(1, 20)).astype(float)
    down = pd.Series(range(20, 1, -1)).astype(float)
    assert ind.rsi(up, 5).iloc[-1] == pytest.approx(100.0)
    assert ind.rsi(down, 5).iloc[-1] == pytest.approx(0.0)


def test_rsi_rejects_nonpositive_period():
    with pytest.raises(ValueError):
        ind.rsi(pd.Series([1.0, 2.0]), 0)


def test_rate_of_change_known_values():
    s = pd.Series([10.0, 20.0, 30.0, 40.0])
    roc = ind.rate_of_change(s, 2)
    assert roc.iloc[:2].isna().all()
    assert roc.iloc[2] == pytest.approx(200.0)  # (30-10)/10 * 100
    assert roc.iloc[3] == pytest.approx(100.0)  # (40-20)/20 * 100


def test_distance_from_ma_pct_and_atr():
    price = pd.Series([110.0])
    ma = pd.Series([100.0])
    atr_series = pd.Series([5.0])
    assert ind.distance_from_ma_pct(price, ma).iloc[0] == pytest.approx(10.0)
    assert ind.distance_from_ma_atr(price, ma, atr_series).iloc[0] == pytest.approx(2.0)


def test_distance_from_ma_pct_handles_zero_ma():
    result = ind.distance_from_ma_pct(pd.Series([1.0]), pd.Series([0.0]))
    assert math.isnan(result.iloc[0])
