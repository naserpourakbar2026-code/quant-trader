import pandas as pd
import pytest

from src.core.config import CooldownConfig, RiskConfig
from src.risk.correlated_exposure import CorrelatedExposureMonitor
from src.strategies.base import PositionSizeResult


def _risk_config(**overrides):
    defaults = dict(
        risk_per_trade=0.005, allowed_risk_levels=[0.005], max_daily_loss=0.03, max_weekly_loss=0.08,
        max_portfolio_drawdown=0.15, max_open_positions=5, max_correlated_exposure=0.02, max_leverage=30,
        max_margin_usage=0.5, daily_trade_limit=10,
        cooldown_after_consecutive_losses=CooldownConfig(losses=3, cooldown_bars=12),
    )
    defaults.update(overrides)
    return RiskConfig(**defaults)


def _size(risk_amount):
    return PositionSizeResult(units=1000.0, risk_amount=risk_amount, stop_distance=0.01)


def _correlation(pairs, symbols):
    """Builds a symmetric symbol x symbol correlation matrix; `pairs` is
    {(a, b): corr}; unset pairs default to 0.0, diagonal to 1.0."""
    frame = pd.DataFrame(0.0, index=symbols, columns=symbols)
    for s in symbols:
        frame.loc[s, s] = 1.0
    for (a, b), corr in pairs.items():
        frame.loc[a, b] = corr
        frame.loc[b, a] = corr
    return frame


def test_approves_when_no_other_open_positions():
    monitor = CorrelatedExposureMonitor(_risk_config(max_correlated_exposure=0.02))
    correlation = _correlation({}, ["EURUSD"])
    decision = monitor.check(
        symbol="EURUSD", position_size=_size(20.0), equity=2000.0, open_risk_by_symbol={}, correlation=correlation,
    )
    assert decision.approved is True  # 20 <= 2000*0.02=40


def test_rejects_when_correlated_symbol_pushes_over_the_limit():
    monitor = CorrelatedExposureMonitor(_risk_config(max_correlated_exposure=0.02))
    correlation = _correlation({("EURUSD", "GBPUSD"): 0.85}, ["EURUSD", "GBPUSD"])
    decision = monitor.check(
        symbol="EURUSD", position_size=_size(30.0), equity=2000.0,
        open_risk_by_symbol={"GBPUSD": 20.0}, correlation=correlation,
    )
    # correlated_risk = 30 + 20 = 50 > limit 40
    assert decision.approved is False
    assert "correlated exposure" in decision.reason


def test_approves_when_other_open_symbol_is_uncorrelated():
    monitor = CorrelatedExposureMonitor(_risk_config(max_correlated_exposure=0.02))
    correlation = _correlation({("EURUSD", "USDJPY"): 0.05}, ["EURUSD", "USDJPY"])
    decision = monitor.check(
        symbol="EURUSD", position_size=_size(30.0), equity=2000.0,
        open_risk_by_symbol={"USDJPY": 20.0}, correlation=correlation,
    )
    # USDJPY's risk isn't added (correlation 0.05 < default threshold 0.5) -> correlated_risk = 30 <= 40
    assert decision.approved is True


def test_negative_correlation_beyond_threshold_still_counts():
    """abs(corr) is what matters -- a strongly negatively correlated
    pair still represents concentrated risk to the same underlying
    driver, just in opposite directions."""
    monitor = CorrelatedExposureMonitor(_risk_config(max_correlated_exposure=0.02))
    correlation = _correlation({("EURUSD", "USDCHF"): -0.9}, ["EURUSD", "USDCHF"])
    decision = monitor.check(
        symbol="EURUSD", position_size=_size(30.0), equity=2000.0,
        open_risk_by_symbol={"USDCHF": 20.0}, correlation=correlation,
    )
    assert decision.approved is False


def test_own_symbol_never_double_counted_if_present_in_open_risk_map():
    monitor = CorrelatedExposureMonitor(_risk_config(max_correlated_exposure=0.02))
    correlation = _correlation({}, ["EURUSD"])
    decision = monitor.check(
        symbol="EURUSD", position_size=_size(20.0), equity=2000.0,
        open_risk_by_symbol={"EURUSD": 999.0}, correlation=correlation,
    )
    assert decision.approved is True  # the 999 entry for EURUSD itself is skipped, not summed in


def test_symbol_missing_from_correlation_matrix_is_treated_as_uncorrelated():
    monitor = CorrelatedExposureMonitor(_risk_config(max_correlated_exposure=0.02))
    correlation = _correlation({}, ["EURUSD"])  # GBPUSD not present at all
    decision = monitor.check(
        symbol="EURUSD", position_size=_size(30.0), equity=2000.0,
        open_risk_by_symbol={"GBPUSD": 20.0}, correlation=correlation,
    )
    assert decision.approved is True


def test_custom_correlation_threshold_is_respected():
    monitor = CorrelatedExposureMonitor(_risk_config(max_correlated_exposure=0.02))
    correlation = _correlation({("EURUSD", "GBPUSD"): 0.4}, ["EURUSD", "GBPUSD"])
    # below the default 0.5 threshold -> uncorrelated
    approved_default = monitor.check(
        symbol="EURUSD", position_size=_size(30.0), equity=2000.0,
        open_risk_by_symbol={"GBPUSD": 20.0}, correlation=correlation,
    ).approved
    # with a lower threshold, 0.4 now counts as correlated -> pushes over the limit
    approved_lower_threshold = monitor.check(
        symbol="EURUSD", position_size=_size(30.0), equity=2000.0,
        open_risk_by_symbol={"GBPUSD": 20.0}, correlation=correlation, correlation_threshold=0.3,
    ).approved
    assert approved_default is True
    assert approved_lower_threshold is False
