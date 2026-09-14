"""Phase 4: feature engineering orchestration.

Combines src/features/indicators.py and src/features/regime.py into one
feature set for a single symbol/timeframe's standardized candles
(src/data/schema.py). Parameters come from config/settings.yaml's
`features:` block unless explicitly overridden — overriding is how Optuna
(Phase 8) will sweep these later, and how tests avoid depending on the
real config file.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.core.config import load_settings
from src.features import indicators as ind
from src.features import regime


@dataclass
class FeatureParams:
    ema_fast_period: int = 20
    ema_slow_period: int = 50
    atr_period: int = 14
    donchian_period: int = 20
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    rsi_period: int = 14
    momentum_period: int = 10
    trend_slope_lookback: int = 10
    trend_strong_threshold: float = 1.0
    trend_weak_threshold: float = 0.3
    volatility_lookback: int = 100
    volatility_low_percentile: float = 25.0
    volatility_high_percentile: float = 75.0
    range_squeeze_lookback: int = 20

    @classmethod
    def from_settings(cls) -> "FeatureParams":
        return cls(**load_settings().features.model_dump())


def compute_features(df: pd.DataFrame, params: FeatureParams | None = None) -> pd.DataFrame:
    """Compute the shared indicator/regime feature set for one
    symbol/timeframe's standardized candles.

    Raises ValueError on an empty DataFrame or one mixing more than one
    symbol/timeframe — every indicator here assumes a single contiguous,
    time-sorted series.
    """
    if df.empty:
        raise ValueError("Cannot compute features on an empty DataFrame")
    if df["symbol"].nunique() > 1 or df["timeframe"].nunique() > 1:
        raise ValueError("compute_features expects a single symbol/timeframe per call")

    p = params or FeatureParams.from_settings()
    out = df.sort_values("timestamp").reset_index(drop=True).copy()

    out["ema_fast"] = ind.ema(out["close"], p.ema_fast_period)
    out["ema_slow"] = ind.ema(out["close"], p.ema_slow_period)
    out["atr"] = ind.atr(out, p.atr_period)

    out = pd.concat([out, ind.donchian_channel(out, p.donchian_period)], axis=1)
    out = pd.concat([out, ind.bollinger_bands(out["close"], p.bollinger_period, p.bollinger_std)], axis=1)

    out["rsi"] = ind.rsi(out["close"], p.rsi_period)
    out["momentum_pct"] = ind.rate_of_change(out["close"], p.momentum_period)
    out["distance_from_ema_slow_pct"] = ind.distance_from_ma_pct(out["close"], out["ema_slow"])
    out["distance_from_ema_slow_atr"] = ind.distance_from_ma_atr(out["close"], out["ema_slow"], out["atr"])

    # Breakout strength: how many ATRs price is beyond the *prior* bar's
    # Donchian channel (using the prior bar avoids a candle "detecting" its
    # own breakout from its own high/low). Positive = upside breakout,
    # negative = downside breakout, 0 = inside the channel.
    prior_upper = out["donchian_upper"].shift(1)
    prior_lower = out["donchian_lower"].shift(1)
    safe_atr = out["atr"].replace(0, float("nan"))
    breakout_up = (out["close"] - prior_upper) / safe_atr
    breakout_down = (prior_lower - out["close"]) / safe_atr
    out["breakout_strength"] = breakout_up.where(
        breakout_up > 0, -breakout_down.where(breakout_down > 0, 0.0)
    )

    out["trend_regime"] = regime.classify_trend(
        out["ema_fast"],
        out["ema_slow"],
        out["atr"],
        slope_lookback=p.trend_slope_lookback,
        strong_threshold=p.trend_strong_threshold,
        weak_threshold=p.trend_weak_threshold,
    )
    out["volatility_regime"] = regime.classify_volatility(
        out["atr"],
        lookback=p.volatility_lookback,
        low_percentile=p.volatility_low_percentile,
        high_percentile=p.volatility_high_percentile,
    )
    out["is_range"] = regime.classify_range(
        out["trend_regime"], out["bb_width_pct"], squeeze_lookback=p.range_squeeze_lookback
    )

    return out
