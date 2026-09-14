"""Phase 14: Paper trading (CLAUDE.md Section 26).

    Market Data -> Strategy -> Risk Engine -> Virtual Order
      -> Execution Simulator -> Position Manager -> PnL

Replays standardized historical candles bar by bar as if they were
arriving live (there is no live feed in this sandbox — that only exists
once a broker adapter is actually connected on Windows/production,
Phases 12-13). At each bar this loop:

1. Marks the (possibly still open) position to market and feeds the
   resulting equity to the Risk Engine (`src.risk.engine.RiskEngine`),
   which rolls day/week trackers and checks the drawdown kill switch —
   *before* anything else happens this bar.
2. Fills a signal-driven exit queued from the previous bar's close, or
   an intrabar stop-loss/take-profit touch, at this bar's own open/
   touch price via the Execution Simulator (spread/commission/slippage).
3. Fills a queued entry from the previous bar's close at this bar's
   open, but only after the Risk Engine approves it.
4. Calls `strategy.generate_signal()` on every bar up to and including
   this one (`BaseStrategy`'s single-bar view, Section 6) to decide
   what to do *next* bar — deliberately one bar's execution delay
   (Section 11), matching Backtrader's own next-bar-execution model
   (Phase 7) rather than filling at the same bar's close, which no real
   broker could ever do.

A strategy/symbol/timeframe carries at most one open position at a time
(same non-stacking convention as the vectorbt/Backtrader engines) —
signal reversal closes the old position and opens the new one at the
same bar's open. Any position still open when the data runs out is
reported as still open, mark-to-market, never force-closed — a real
paper-trading session doesn't end just because historical data ran out.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import pandas as pd

from src.core.config import load_settings
from src.core.reproducibility import git_commit_hash, library_versions, python_version
from src.execution.models import PaperSession  # noqa: F401 -- imported so Base.metadata registers it before init_db()
from src.execution.position_manager import OpenPosition, PaperTrade, PositionManager
from src.execution.simulator import ExecutionSimulator
from src.execution.trade_store import PaperSessionRecord, save_session
from src.features.engine import FeatureParams, compute_features
from src.risk.engine import RiskEngine
from src.strategies import create_strategy
from src.strategies.base import Signal, SignalDirection


@dataclass
class PaperTradingSessionResult:
    run_id: str
    strategy: str
    symbol: str
    timeframe: str
    scenario: str
    initial_capital: float
    final_equity: float
    closed_trades: list[PaperTrade] = field(default_factory=list)
    open_position: OpenPosition | None = None
    rejected_signals: list[tuple[Signal, str]] = field(default_factory=list)
    kill_switch_triggered: bool = False

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.closed_trades)

    @property
    def win_rate(self) -> float | None:
        if not self.closed_trades:
            return None
        return sum(1 for t in self.closed_trades if t.pnl > 0) / len(self.closed_trades)

    def to_text(self) -> str:
        win_rate_txt = f"{self.win_rate:.1%}" if self.win_rate is not None else "n/a"
        lines = [
            "PAPER TRADING SESSION REPORT",
            f"Strategy: {self.strategy} | Symbol: {self.symbol} | Timeframe: {self.timeframe} | Scenario: {self.scenario}",
            f"Initial capital: {self.initial_capital:.2f} | Final equity: {self.final_equity:.2f} "
            f"({(self.final_equity / self.initial_capital - 1):+.2%})",
            f"Closed trades: {len(self.closed_trades)} | Win rate: {win_rate_txt}",
            f"Total PnL: {self.total_pnl:+.2f}",
            f"Rejected signals (risk engine): {len(self.rejected_signals)}",
            f"Kill switch triggered: {self.kill_switch_triggered}",
        ]
        if self.open_position is not None:
            p = self.open_position
            lines.append(
                f"Still open: {p.direction.value} {p.size:.4f} units @ {p.entry_price:.5f} "
                f"(SL {p.stop_loss:.5f} / TP {p.take_profit:.5f}, opened {p.entry_time})"
            )
        for trade in self.closed_trades:
            r_txt = f"{trade.r_multiple:+.2f}R" if trade.r_multiple is not None else "n/a"
            lines.append(
                f"  {trade.entry_time} {trade.direction.value:5s} entry={trade.entry_price:.5f} "
                f"exit={trade.exit_price:.5f} pnl={trade.pnl:+.2f} ({r_txt}) reason={trade.reason}"
            )
        return "\n".join(lines)


def run_paper_trading_session(
    strategy_family: str,
    symbol: str,
    timeframe: str,
    raw_df: pd.DataFrame,
    *,
    strategy_params: dict | None = None,
    feature_params: FeatureParams | None = None,
    scenario: str | None = None,
    initial_capital: float | None = None,
    risk_config=None,
    persist: bool = True,
    db=None,
) -> PaperTradingSessionResult:
    settings = load_settings()
    scenario = scenario or settings.execution.default_scenario
    if scenario not in settings.execution.costs:
        raise ValueError(f"Unknown execution scenario {scenario!r}; expected one of {list(settings.execution.costs)}")
    capital = initial_capital if initial_capital is not None else settings.account.initial_capital
    risk_cfg = risk_config or settings.risk
    risk_pct = risk_cfg.risk_per_trade

    feature_df = compute_features(raw_df, feature_params)
    strategy = create_strategy(strategy_family, strategy_params)
    simulator = ExecutionSimulator(settings.execution.costs[scenario])
    risk_engine = RiskEngine(risk_cfg, capital)

    # generate_signals_vectorized() is purely causal (rolling windows,
    # never a future row) -- by the same "single source of truth"
    # argument every other engine relies on, reading row i of one
    # whole-series computation is exactly equivalent to calling
    # generate_signal() fresh on feature_df.iloc[:i+1] every bar, just
    # without redoing O(i) indicator work on every single iteration
    # (which would make this loop O(n^2) for no behavioral difference).
    signal_table = strategy.generate_signals_vectorized(feature_df).set_axis(feature_df.index)

    cash = capital
    open_position: OpenPosition | None = None
    pending_entry_signal: Signal | None = None
    pending_exit_reason: str | None = None
    closed_trades: list[PaperTrade] = []
    rejected_signals: list[tuple[Signal, str]] = []

    n = len(feature_df)
    for i in range(1, n):
        bar = feature_df.iloc[i]
        equity = cash + (PositionManager.unrealized_pnl(open_position, bar["close"]) if open_position else 0.0)
        risk_engine.update_equity(equity, bar["timestamp"])

        # (a) signal-driven exit queued from the previous bar's close, filled at this bar's open
        if open_position is not None and pending_exit_reason is not None:
            fill = simulator.simulate_exit(
                direction=open_position.direction, exit_price=bar["open"], bar_spread=bar["spread"], size=open_position.size,
            )
            trade = PositionManager.close(open_position, exit_time=bar["timestamp"], fill=fill, reason=pending_exit_reason)
            cash += trade.pnl
            risk_engine.record_trade_close(trade.pnl)
            closed_trades.append(trade)
            open_position = None
        pending_exit_reason = None

        # (b) intrabar stop-loss/take-profit touch on a still-open position
        if open_position is not None:
            touch = PositionManager.check_stop_touch(open_position, bar)
            if touch is not None:
                exit_price, reason = touch
                fill = simulator.simulate_exit(
                    direction=open_position.direction, exit_price=exit_price, bar_spread=bar["spread"], size=open_position.size,
                )
                trade = PositionManager.close(open_position, exit_time=bar["timestamp"], fill=fill, reason=reason)
                cash += trade.pnl
                risk_engine.record_trade_close(trade.pnl)
                closed_trades.append(trade)
                open_position = None

        # (c) entry queued from the previous bar's close, filled at this bar's open -- only while flat
        if open_position is None and pending_entry_signal is not None:
            signal = pending_entry_signal
            position_size = strategy.calculate_position_size(cash, risk_pct, bar["open"], signal.stop_loss)
            decision = risk_engine.evaluate(
                equity=cash, position_size=position_size, entry_price=bar["open"], open_positions_count=0,
            )
            if decision.approved:
                fill = simulator.simulate_entry(
                    direction=signal.direction, bar_open=bar["open"], bar_spread=bar["spread"], size=position_size.units,
                )
                open_position = PositionManager.open(
                    strategy=strategy_family, symbol=symbol, timeframe=timeframe, direction=signal.direction,
                    entry_time=bar["timestamp"], stop_loss=signal.stop_loss, take_profit=signal.take_profit,
                    size=position_size.units, risk_amount=position_size.risk_amount, fill=fill,
                )
                risk_engine.notify_order_opened()
            else:
                rejected_signals.append((signal, decision.reason))
        pending_entry_signal = None

        # (d) decide the next action from this bar's own close (reading
        # the precomputed signal_table row -- see the note above)
        row = signal_table.iloc[i]
        direction = SignalDirection(row["direction"])
        actionable = direction != SignalDirection.FLAT
        signal = Signal(
            direction=direction,
            entry_price=float(bar["close"]) if actionable else None,
            stop_loss=float(row["stop_loss"]) if actionable else None,
            take_profit=float(row["take_profit"]) if actionable else None,
            confidence=float(row["confidence"]),
            strategy_name=strategy.strategy_name,
            timestamp=bar["timestamp"], symbol=symbol, timeframe=timeframe,
        )
        history_so_far = feature_df.iloc[: i + 1]
        if open_position is not None and signal.is_actionable() and signal.direction != open_position.direction:
            pending_exit_reason = "signal_reversal"
            pending_entry_signal = signal
        elif open_position is not None and not signal.is_actionable():
            pending_exit_reason = "signal_flat"
        elif open_position is None and signal.is_actionable() and strategy.validate_signal(signal, history_so_far):
            pending_entry_signal = signal

    final_equity = cash + (
        PositionManager.unrealized_pnl(open_position, feature_df.iloc[-1]["close"]) if open_position else 0.0
    )

    result = PaperTradingSessionResult(
        run_id=str(uuid.uuid4()), strategy=strategy_family, symbol=symbol, timeframe=timeframe, scenario=scenario,
        initial_capital=capital, final_equity=final_equity, closed_trades=closed_trades, open_position=open_position,
        rejected_signals=rejected_signals, kill_switch_triggered=risk_engine.kill_switch_triggered,
    )

    if persist:
        still_open = None
        if result.open_position is not None:
            p = result.open_position
            still_open = {
                "direction": p.direction.value, "entry_time": str(p.entry_time), "entry_price": p.entry_price,
                "stop_loss": p.stop_loss, "take_profit": p.take_profit, "size": p.size,
            }
        save_session(
            PaperSessionRecord(
                run_id=result.run_id, strategy=result.strategy, symbol=result.symbol, timeframe=result.timeframe,
                scenario=result.scenario, initial_capital=result.initial_capital, final_equity=result.final_equity,
                total_trades=len(result.closed_trades), win_rate=result.win_rate, total_pnl=result.total_pnl,
                kill_switch_triggered=result.kill_switch_triggered, still_open=still_open,
                code_version=git_commit_hash(), python_version=python_version(),
                library_versions=library_versions(["pandas", "numpy"]),
            ),
            result.closed_trades,
            db=db,
        )

    return result
