import pytest

from src.strategies import (
    STRATEGY_REGISTRY,
    MeanReversionStrategy,
    MomentumMultiFactorStrategy,
    TrendFollowingStrategy,
    create_strategy,
)


def test_registry_has_exactly_three_independent_families():
    assert set(STRATEGY_REGISTRY) == {"trend_following", "mean_reversion", "momentum_multi_factor"}


@pytest.mark.parametrize(
    "family,expected_cls",
    [
        ("trend_following", TrendFollowingStrategy),
        ("mean_reversion", MeanReversionStrategy),
        ("momentum_multi_factor", MomentumMultiFactorStrategy),
    ],
)
def test_create_strategy_returns_correct_type(family, expected_cls):
    strat = create_strategy(family)
    assert isinstance(strat, expected_cls)
    assert strat.strategy_name == family


def test_create_strategy_passes_through_params():
    strat = create_strategy("trend_following", params={"atr_stop_multiplier": 5.0})
    assert strat.params["atr_stop_multiplier"] == 5.0


def test_create_strategy_rejects_unknown_family():
    with pytest.raises(ValueError, match="Unknown strategy family"):
        create_strategy("not_a_real_family")
