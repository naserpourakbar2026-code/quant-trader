"""Phase 7: Backtrader validation engine (CLAUDE.md Section 13).

After vectorbt (Phase 6) screens candidates cheaply, this engine re-checks
the survivors with genuine event-driven, next-bar execution: orders
placed in `next()` fill at the *following* bar's open (not the signal
bar's own close, unlike vectorbt's default), the broker can reject an
order for insufficient margin, and stop-loss/take-profit are real pending
child orders Backtrader checks bar-by-bar (not a fractional-distance
approximation). This is what catches "unrealistic results from vectorized
backtest assumptions" — the entry-condition logic itself is untouched:
it's read from the same generate_signals_vectorized() table Phase 6 and
live/paper trading (Phase 5) use, so only the *execution* realism differs
between engines, never "what the strategy decided".
"""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass

import backtrader as bt
import pandas as pd

from src.backtest.experiment_store import ExperimentRecord, save_experiment
from src.core.config import load_settings
from src.core.db import Database
from src.core.reproducibility import git_commit_hash, library_versions, python_version
from src.data.ingestion import data_version_for
from src.features.engine import FeatureParams, compute_features
from src.strategies import create_strategy

_DIRECTION_CODE = {"LONG": 1.0, "SHORT": -1.0, "FLAT": 0.0}


@dataclass
class ValidationResult:
    experiment_id: str
    strategy: str
    symbol: str
    timeframe: str
    scenario: str
    parameters: dict
    metrics: dict


class SignalPandasData(bt.feeds.PandasData):
    """A standard OHLCV feed plus three extra lines carrying a strategy's
    already-decided direction/stop_loss/take_profit per bar — Backtrader
    only ever *executes* against these, never re-derives them."""

    lines = ("direction", "stop_loss", "take_profit")
    params = (
        ("direction", "direction"),
        ("stop_loss", "stop_loss"),
        ("take_profit", "take_profit"),
    )


class SignalDrivenStrategy(bt.Strategy):
    """Places bracket orders (entry + real stop-loss + real take-profit
    child orders) off the precomputed direction/stop_loss/take_profit
    lines. Position size uses the strategy's own calculate_position_size()
    (Section 6/7-8) against the broker's *current* equity, so sizing
    reacts to drawdown/growth the same way it would live."""

    params = (("base_strategy", None), ("risk_pct", 0.005), ("pip_value_per_unit", 1.0))

    _PENDING_STATUSES = (bt.Order.Submitted, bt.Order.Accepted, bt.Order.Partial)

    def __init__(self):
        self.rejected_orders = 0
        self.margin_orders = 0
        self._pending_entry = None

    def _has_pending_entry(self) -> bool:
        return self._pending_entry is not None and self._pending_entry.status in self._PENDING_STATUSES

    def next(self):
        direction = self.data.direction[0]
        position_side = 1.0 if self.position.size > 0 else (-1.0 if self.position.size < 0 else 0.0)

        if position_side != 0.0 and direction != position_side:
            self.close()
            return

        # A bracket entry placed in next() fills at the *next* bar's open
        # (real next-bar execution, per Section 13) — so for at least one
        # bar after issuing it, self.position is still flat even though an
        # entry is already in flight. Without this guard, a signal that
        # stays nonzero for many consecutive bars (typical for trend
        # following) would re-issue a brand-new bracket order every single
        # bar it's still pending, stacking up dozens of duplicate orders
        # that each need their own margin until the account can't afford
        # any of them — this was caught by comparing against vectorbt's
        # trade count (~4-10 real signals) versus Backtrader initially
        # reporting 200+ margin rejections from one same series.
        if position_side != 0.0 or self._has_pending_entry():
            return

        if direction != 0.0:
            entry_price = float(self.data.close[0])
            stop_loss = float(self.data.stop_loss[0])
            take_profit = float(self.data.take_profit[0])
            if math.isnan(stop_loss) or math.isnan(take_profit):
                return

            sizing = self.p.base_strategy.calculate_position_size(
                equity=self.broker.getvalue(),
                risk_pct=self.p.risk_pct,
                entry_price=entry_price,
                stop_loss=stop_loss,
                pip_value_per_unit=self.p.pip_value_per_unit,
            )
            size = sizing.units
            if size <= 0:
                return

            # exectype must be forced to Market: buy_bracket()/sell_bracket()
            # default the *entry* leg to a Limit order at the signal bar's
            # close, which can sit unfilled for many bars (or never fill)
            # rather than executing at the next bar's open — the opposite
            # of the genuine next-bar market execution Section 13 wants,
            # and part of what caused the duplicate-order stacking bug
            # above before the pending-entry guard was added.
            if direction > 0:
                orders = self.buy_bracket(
                    size=size, exectype=bt.Order.Market, stopprice=stop_loss, limitprice=take_profit
                )
            else:
                orders = self.sell_bracket(
                    size=size, exectype=bt.Order.Market, stopprice=stop_loss, limitprice=take_profit
                )
            self._pending_entry = orders[0]

    def notify_order(self, order):
        if order.status == order.Rejected:
            self.rejected_orders += 1
        elif order.status == order.Margin:
            self.margin_orders += 1


def _build_feed_dataframe(feature_df: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    direction_numeric = table["direction"].map(_DIRECTION_CODE)
    feed = pd.DataFrame(
        {
            "open": feature_df["open"].to_numpy(),
            "high": feature_df["high"].to_numpy(),
            "low": feature_df["low"].to_numpy(),
            "close": feature_df["close"].to_numpy(),
            "volume": feature_df["tick_volume"].fillna(0).to_numpy(),
            "direction": direction_numeric.to_numpy(),
            "stop_loss": table["stop_loss"].to_numpy(),
            "take_profit": table["take_profit"].to_numpy(),
        },
        index=pd.DatetimeIndex(feature_df["timestamp"]),
    )
    return feed


def _extract_metrics(cerebro: "bt.Cerebro", strat, initial_capital: float) -> dict:
    final_value = cerebro.broker.getvalue()
    trade_analysis = strat.analyzers.ta.get_analysis()
    dd_analysis = strat.analyzers.dd.get_analysis()
    sharpe_analysis = strat.analyzers.sharpe.get_analysis()

    total_trades = trade_analysis.get("total", {}).get("closed", 0)
    won = trade_analysis.get("won", {}).get("total", 0)
    pnl = trade_analysis.get("pnl", {})
    won_pnl_total = trade_analysis.get("won", {}).get("pnl", {}).get("total", 0.0)
    lost_pnl_total = trade_analysis.get("lost", {}).get("pnl", {}).get("total", 0.0)

    metrics = {
        "total_return": (final_value / initial_capital) - 1.0,
        "sharpe_ratio": sharpe_analysis.get("sharperatio"),
        # Negative, matching vectorbt's sign convention (pf.max_drawdown())
        # -- Backtrader's own DrawDown analyzer reports a positive percentage.
        "max_drawdown": -((dd_analysis.get("max", {}).get("drawdown", 0.0) or 0.0) / 100.0),
        "trade_count": total_trades,
        "rejected_orders": strat.rejected_orders,
        "margin_orders": strat.margin_orders,
    }
    if total_trades > 0:
        metrics["win_rate"] = won / total_trades
        metrics["profit_factor"] = (won_pnl_total / abs(lost_pnl_total)) if lost_pnl_total else None
        metrics["expectancy"] = pnl.get("net", {}).get("average")
    else:
        metrics["win_rate"] = None
        metrics["profit_factor"] = None
        metrics["expectancy"] = None

    for key, value in metrics.items():
        if isinstance(value, float) and not math.isfinite(value):
            metrics[key] = None
    return metrics


def run_validation(
    strategy_family: str,
    symbol: str,
    timeframe: str,
    raw_df: pd.DataFrame,
    *,
    strategy_params: dict | None = None,
    feature_params: FeatureParams | None = None,
    scenario: str | None = None,
    initial_capital: float | None = None,
    persist: bool = True,
    db: Database | None = None,
) -> ValidationResult:
    """Event-driven validation of one strategy/symbol/timeframe/parameter
    combination. `raw_df` is standardized candle data for a single
    symbol/timeframe (src.data.schema columns)."""
    settings = load_settings()
    scenario = scenario or settings.execution.default_scenario
    if scenario not in settings.execution.costs:
        raise ValueError(f"Unknown execution scenario {scenario!r}; expected one of {list(settings.execution.costs)}")
    cost = settings.execution.costs[scenario]
    capital = initial_capital if initial_capital is not None else settings.account.initial_capital

    feature_df = compute_features(raw_df, feature_params)
    strat = create_strategy(strategy_family, strategy_params)
    table = strat.generate_signals_vectorized(feature_df)
    feed_df = _build_feed_dataframe(feature_df, table)

    # Real spread cost from the actual data (if present), folded into the
    # commission percentage — the same "% of price" dimension commission
    # already uses. Not a per-tick simulation (that needs live broker
    # quotes, Phase 12/13), but a genuine cost derived from real data
    # rather than assumed away.
    spread_pct = (raw_df["spread"] / raw_df["close"]).mean()
    spread_pct = float(spread_pct) if pd.notna(spread_pct) else 0.0
    effective_commission_pct = cost.commission_pct + spread_pct

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.adddata(SignalPandasData(dataname=feed_df))
    cerebro.addstrategy(
        SignalDrivenStrategy,
        base_strategy=strat,
        risk_pct=settings.risk.risk_per_trade,
    )
    cerebro.broker.setcash(capital)
    # NOTE: Backtrader's setcommission(), when `commtype` is left as its
    # default None, falls back to legacy CommissionInfo behavior that
    # infers stocklike=True whenever `margin` isn't explicitly set —
    # which makes it require the *full notional* (size * price) as cash
    # and silently ignores `leverage` entirely. That produced a ~90%
    # order-rejection rate here before this was caught: sizing is meant
    # to require only the margin (notional / leverage), not the full
    # notional. Passing commtype/stocklike/automargin explicitly is what
    # actually turns "leverage" into a real margin requirement
    # (get_margin(price) = price * automargin = price / leverage).
    cerebro.broker.setcommission(
        commission=effective_commission_pct,
        commtype=bt.CommInfoBase.COMM_PERC,
        percabs=True,
        stocklike=False,
        automargin=1.0 / settings.risk.max_leverage,
    )
    if cost.slippage_pct > 0:
        cerebro.broker.set_slippage_perc(perc=cost.slippage_pct)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="ta")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days)

    results = cerebro.run()
    strat_instance = results[0]

    metrics = _extract_metrics(cerebro, strat_instance, capital)
    experiment_id = str(uuid.uuid4())

    if persist:
        save_experiment(
            ExperimentRecord(
                experiment_id=experiment_id,
                engine="backtrader",
                strategy=strategy_family,
                symbol=symbol,
                timeframe=timeframe,
                scenario=scenario,
                parameters=dict(strat.params),
                date_range_start=feed_df.index[0].to_pydatetime(),
                date_range_end=feed_df.index[-1].to_pydatetime(),
                metrics=metrics,
                data_version=data_version_for(symbol, timeframe),
                code_version=git_commit_hash(),
                python_version=python_version(),
                library_versions=library_versions(["pandas", "numpy", "backtrader", "scipy"]),
            ),
            db=db,
        )

    return ValidationResult(
        experiment_id=experiment_id,
        strategy=strategy_family,
        symbol=symbol,
        timeframe=timeframe,
        scenario=scenario,
        parameters=dict(strat.params),
        metrics=metrics,
    )


def compare_with_screening(screening_metrics: dict, validation_metrics: dict) -> dict:
    """Side-by-side deltas between a vectorbt screening result and its
    Backtrader validation — the concrete mechanism for Section 13's
    "catches unrealistic results from vectorized backtest assumptions":
    a large gap here means the vectorized screen was overly optimistic
    (or, less commonly, pessimistic) about execution realism."""
    keys = ("total_return", "sharpe_ratio", "max_drawdown", "trade_count", "profit_factor")
    comparison = {}
    for key in keys:
        screen_value = screening_metrics.get(key)
        validate_value = validation_metrics.get(key)
        comparison[key] = {
            "screening": screen_value,
            "validation": validate_value,
            "delta": (validate_value - screen_value)
            if isinstance(screen_value, (int, float)) and isinstance(validate_value, (int, float))
            else None,
        }
    return comparison
