import pandas as pd
import pytest

from src.strategies.base import BaseStrategy, Signal, SignalDirection


class _DummyStrategy(BaseStrategy):
    """Minimal concrete subclass so we can exercise BaseStrategy's shared
    logic (calculate_position_size, validate_signal, _build_signal)
    without depending on any real strategy family's decision rules."""

    strategy_name = "dummy"

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        return self._build_signal(df, SignalDirection.LONG, confidence=0.5)

    def calculate_stop_loss(self, df: pd.DataFrame, direction: SignalDirection) -> float:
        last = df.iloc[-1]
        return float(last["close"]) - 1.0

    def calculate_take_profit(self, df: pd.DataFrame, direction: SignalDirection, stop_loss: float) -> float:
        last = df.iloc[-1]
        return float(last["close"]) + 2.0


def _one_row_df():
    return pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-02T00:00:00Z")],
            "symbol": ["EURUSD"],
            "timeframe": ["H1"],
            "close": [1.10],
        }
    )


def test_calculate_position_size_basic():
    strat = _DummyStrategy()
    result = strat.calculate_position_size(equity=2000.0, risk_pct=0.005, entry_price=1.10, stop_loss=1.08)
    assert result.risk_amount == pytest.approx(10.0)
    assert result.stop_distance == pytest.approx(0.02)
    assert result.units == pytest.approx(10.0 / 0.02)


def test_calculate_position_size_respects_pip_value_override():
    strat = _DummyStrategy()
    result = strat.calculate_position_size(
        equity=2000.0, risk_pct=0.005, entry_price=1.10, stop_loss=1.08, pip_value_per_unit=2.0
    )
    assert result.units == pytest.approx((10.0 / 0.02) / 2.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"equity": 0.0, "risk_pct": 0.005, "entry_price": 1.10, "stop_loss": 1.08},
        {"equity": 2000.0, "risk_pct": 0.0, "entry_price": 1.10, "stop_loss": 1.08},
        {"equity": 2000.0, "risk_pct": 0.005, "entry_price": 1.10, "stop_loss": 1.10},
        {"equity": 2000.0, "risk_pct": 0.005, "entry_price": 1.10, "stop_loss": 1.08, "pip_value_per_unit": 0.0},
    ],
)
def test_calculate_position_size_rejects_invalid_inputs(kwargs):
    strat = _DummyStrategy()
    with pytest.raises(ValueError):
        strat.calculate_position_size(**kwargs)


def test_validate_signal_flat_is_always_valid():
    strat = _DummyStrategy()
    signal = Signal(
        direction=SignalDirection.FLAT,
        entry_price=None,
        stop_loss=None,
        take_profit=None,
        confidence=0.0,
        strategy_name="dummy",
        timestamp=pd.Timestamp.now(tz="UTC"),
        symbol="EURUSD",
        timeframe="H1",
    )
    assert strat.validate_signal(signal, _one_row_df()) is True


def test_validate_signal_long_requires_correct_ordering():
    strat = _DummyStrategy()
    good = Signal(SignalDirection.LONG, 100.0, 98.0, 104.0, 0.5, "dummy", pd.Timestamp.now(tz="UTC"), "EURUSD", "H1")
    assert strat.validate_signal(good, _one_row_df()) is True

    bad = Signal(SignalDirection.LONG, 100.0, 104.0, 98.0, 0.5, "dummy", pd.Timestamp.now(tz="UTC"), "EURUSD", "H1")
    assert strat.validate_signal(bad, _one_row_df()) is False


def test_validate_signal_short_requires_correct_ordering():
    strat = _DummyStrategy()
    good = Signal(SignalDirection.SHORT, 100.0, 104.0, 96.0, 0.5, "dummy", pd.Timestamp.now(tz="UTC"), "EURUSD", "H1")
    assert strat.validate_signal(good, _one_row_df()) is True

    bad = Signal(SignalDirection.SHORT, 100.0, 96.0, 104.0, 0.5, "dummy", pd.Timestamp.now(tz="UTC"), "EURUSD", "H1")
    assert strat.validate_signal(bad, _one_row_df()) is False


def test_validate_signal_rejects_out_of_range_confidence():
    strat = _DummyStrategy()
    bad = Signal(SignalDirection.LONG, 100.0, 98.0, 104.0, 1.5, "dummy", pd.Timestamp.now(tz="UTC"), "EURUSD", "H1")
    assert strat.validate_signal(bad, _one_row_df()) is False


def test_validate_signal_rejects_nonpositive_entry_price():
    strat = _DummyStrategy()
    bad = Signal(SignalDirection.LONG, 0.0, 98.0, 104.0, 0.5, "dummy", pd.Timestamp.now(tz="UTC"), "EURUSD", "H1")
    assert strat.validate_signal(bad, _one_row_df()) is False


def test_build_signal_via_generate_signal_produces_valid_signal():
    strat = _DummyStrategy()
    df = _one_row_df()
    signal = strat.generate_signal(df)
    assert signal.direction == SignalDirection.LONG
    assert signal.entry_price == pytest.approx(1.10)
    assert signal.stop_loss == pytest.approx(0.10)
    assert signal.take_profit == pytest.approx(3.10)
    assert strat.validate_signal(signal, df) is True
