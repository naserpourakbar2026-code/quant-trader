import numpy as np
import pandas as pd
import pytest

from src.data.schema import STANDARD_COLUMNS
from src.features.engine import FeatureParams, compute_features

EXPECTED_NEW_COLUMNS = [
    "ema_fast",
    "ema_slow",
    "atr",
    "donchian_upper",
    "donchian_lower",
    "donchian_middle",
    "bb_upper",
    "bb_middle",
    "bb_lower",
    "bb_width_pct",
    "rsi",
    "momentum_pct",
    "distance_from_ema_slow_pct",
    "distance_from_ema_slow_atr",
    "breakout_strength",
    "trend_regime",
    "volatility_regime",
    "is_range",
]


def _synthetic_candles(n=150, symbol="EURUSD", timeframe="H1", seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-02T00:00:00Z", periods=n, freq="1h")
    close = 1.10 + np.cumsum(rng.normal(0, 0.0005, size=n))
    high = close + rng.uniform(0.0001, 0.0005, size=n)
    low = close - rng.uniform(0.0001, 0.0005, size=n)
    open_ = close + rng.normal(0, 0.0002, size=n)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": symbol,
            "timeframe": timeframe,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "tick_volume": 100.0,
            "spread": 1.5,
            "real_volume": float("nan"),
        }
    )[STANDARD_COLUMNS]


SMALL_PARAMS = FeatureParams(
    ema_fast_period=3,
    ema_slow_period=8,
    atr_period=5,
    donchian_period=6,
    bollinger_period=6,
    bollinger_std=2.0,
    rsi_period=5,
    momentum_period=3,
    trend_slope_lookback=4,
    trend_strong_threshold=1.0,
    trend_weak_threshold=0.3,
    volatility_lookback=20,
    volatility_low_percentile=25.0,
    volatility_high_percentile=75.0,
    range_squeeze_lookback=6,
)


def test_compute_features_adds_all_expected_columns():
    df = _synthetic_candles(n=60)
    out = compute_features(df, params=SMALL_PARAMS)
    for col in EXPECTED_NEW_COLUMNS:
        assert col in out.columns, f"missing column {col}"
    assert len(out) == len(df)


def test_compute_features_preserves_original_standard_columns():
    df = _synthetic_candles(n=60)
    out = compute_features(df, params=SMALL_PARAMS)
    for col in STANDARD_COLUMNS:
        assert col in out.columns


def test_compute_features_sorts_by_timestamp():
    df = _synthetic_candles(n=20)
    shuffled = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
    out = compute_features(shuffled, params=SMALL_PARAMS)
    assert out["timestamp"].is_monotonic_increasing


def test_compute_features_rejects_empty_dataframe():
    df = _synthetic_candles(n=5).iloc[0:0]
    with pytest.raises(ValueError, match="empty"):
        compute_features(df, params=SMALL_PARAMS)


def test_compute_features_rejects_mixed_symbols():
    df1 = _synthetic_candles(n=10, symbol="EURUSD")
    df2 = _synthetic_candles(n=10, symbol="GBPUSD")
    mixed = pd.concat([df1, df2], ignore_index=True)
    with pytest.raises(ValueError, match="single symbol/timeframe"):
        compute_features(mixed, params=SMALL_PARAMS)


def test_compute_features_rejects_mixed_timeframes():
    df1 = _synthetic_candles(n=10, timeframe="H1")
    df2 = _synthetic_candles(n=10, timeframe="H4")
    mixed = pd.concat([df1, df2], ignore_index=True)
    with pytest.raises(ValueError, match="single symbol/timeframe"):
        compute_features(mixed, params=SMALL_PARAMS)


def test_feature_params_from_settings_matches_config():
    from src.core.config import load_settings

    params = FeatureParams.from_settings()
    cfg = load_settings().features
    assert params.ema_fast_period == cfg.ema_fast_period
    assert params.rsi_period == cfg.rsi_period


def test_breakout_strength_is_zero_inside_channel_and_signed_on_breakout():
    n = 30
    ts = pd.date_range("2024-01-02T00:00:00Z", periods=n, freq="1h")
    close = pd.Series([1.10] * (n - 1) + [1.50])  # sharp breakout on the last bar
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "open": close,
            "high": close + 0.001,
            "low": close - 0.001,
            "close": close,
            "tick_volume": 100.0,
            "spread": 1.0,
            "real_volume": float("nan"),
        }
    )[STANDARD_COLUMNS]

    out = compute_features(df, params=SMALL_PARAMS)
    assert out["breakout_strength"].iloc[-1] > 0
    # Well before the breakout, price is inside its own channel -> ~0
    assert out["breakout_strength"].iloc[15] == pytest.approx(0.0)
