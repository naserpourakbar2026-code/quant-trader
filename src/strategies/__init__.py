"""Strategy family registry (CLAUDE.md Section 5)."""
from __future__ import annotations

from src.strategies.base import BaseStrategy, PositionSizeResult, Signal, SignalDirection
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum import MomentumMultiFactorStrategy
from src.strategies.trend_following import TrendFollowingStrategy

STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
    "trend_following": TrendFollowingStrategy,
    "mean_reversion": MeanReversionStrategy,
    "momentum_multi_factor": MomentumMultiFactorStrategy,
}


def create_strategy(family: str, params: dict | None = None) -> BaseStrategy:
    try:
        strategy_cls = STRATEGY_REGISTRY[family]
    except KeyError:
        raise ValueError(f"Unknown strategy family {family!r}; expected one of {sorted(STRATEGY_REGISTRY)}") from None
    return strategy_cls(params)


__all__ = [
    "BaseStrategy",
    "Signal",
    "SignalDirection",
    "PositionSizeResult",
    "STRATEGY_REGISTRY",
    "create_strategy",
    "TrendFollowingStrategy",
    "MeanReversionStrategy",
    "MomentumMultiFactorStrategy",
]
