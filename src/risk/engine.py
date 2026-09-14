"""Risk Engine (CLAUDE.md Section 7): the gate between a strategy's
signal and a virtual/real order. Paper trading (Phase 14) is the first
consumer; Phase 15 formalizes the hard drawdown kill switch as its own
named, more thoroughly tested concept and wires the same checks into
live execution — the checks themselves already have to exist here
because paper trading's own pipeline (Section 26) requires a Risk Engine
step, not just an execution simulator.

Deliberately excluded here, and why: `max_correlated_exposure` (Section
7) is a cross-*symbol* concept, but this engine's caller
(run_paper_trading_session) is scoped to one strategy/symbol/timeframe —
same as every other engine in this codebase (run_screening,
run_optimization, run_walk_forward, run_monte_carlo). There is nothing
else in scope to be "correlated" with inside a single session. A
portfolio-level exposure check belongs to a multi-symbol orchestrator
that doesn't exist yet, reusing Phase 11's correlation machinery when it
does — implementing a same-symbol-only stand-in here would just be
dead code today (this engine already never lets a symbol carry a second
concurrent position).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.core.config import RiskConfig
from src.strategies.base import PositionSizeResult


@dataclass
class RiskDecision:
    approved: bool
    reason: str


class RiskEngine:
    def __init__(self, cfg: RiskConfig, initial_capital: float) -> None:
        self.cfg = cfg
        self.peak_equity = initial_capital
        self.day_start_equity = initial_capital
        self.week_start_equity = initial_capital
        self._current_day = None
        self._current_week = None
        self.trades_opened_today = 0
        self.consecutive_losses = 0
        self.cooldown_remaining_bars = 0
        self.kill_switch_triggered = False

    def update_equity(self, equity: float, timestamp: pd.Timestamp) -> None:
        """Call once per bar with the current mark-to-market equity
        (realized + unrealized) *before* evaluating any new order this
        bar — rolls day/week trackers forward on a calendar boundary,
        updates the running peak for drawdown, and ticks any active
        cooldown down by one bar."""
        day = timestamp.date()
        week = timestamp.isocalendar()[:2]
        if self._current_day != day:
            self._current_day = day
            self.day_start_equity = equity
            self.trades_opened_today = 0
        if self._current_week != week:
            self._current_week = week
            self.week_start_equity = equity

        self.peak_equity = max(self.peak_equity, equity)
        drawdown = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        if drawdown >= self.cfg.max_portfolio_drawdown:
            self.kill_switch_triggered = True  # CLAUDE.md Section 7's hard kill switch: never resets itself

        if self.cooldown_remaining_bars > 0:
            self.cooldown_remaining_bars -= 1

    def record_trade_close(self, pnl: float) -> None:
        """Call once per closed trade — tracks the consecutive-loss
        cooldown (Section 7)."""
        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.cfg.cooldown_after_consecutive_losses.losses:
                self.cooldown_remaining_bars = self.cfg.cooldown_after_consecutive_losses.cooldown_bars
        else:
            self.consecutive_losses = 0

    def evaluate(
        self, *, equity: float, position_size: PositionSizeResult, entry_price: float, open_positions_count: int,
    ) -> RiskDecision:
        """Everything a strategy's already-sized signal must clear
        before it becomes a virtual order. `position_size` is the
        strategy's own calculate_position_size() result (Section 6) —
        this engine checks limits/margin against it, it never computes
        sizing itself, so there is exactly one formula for "how big"
        (BaseStrategy.calculate_position_size) and no risk of the two
        drifting apart."""
        if self.kill_switch_triggered:
            return RiskDecision(False, "kill switch active: max portfolio drawdown breached")
        if self.cooldown_remaining_bars > 0:
            return RiskDecision(False, f"cooldown active for {self.cooldown_remaining_bars} more bar(s) after consecutive losses")
        if self.trades_opened_today >= self.cfg.daily_trade_limit:
            return RiskDecision(False, "daily trade limit reached")

        day_return = (equity - self.day_start_equity) / self.day_start_equity if self.day_start_equity > 0 else 0.0
        if day_return <= -self.cfg.max_daily_loss:
            return RiskDecision(False, "max daily loss reached")

        week_return = (equity - self.week_start_equity) / self.week_start_equity if self.week_start_equity > 0 else 0.0
        if week_return <= -self.cfg.max_weekly_loss:
            return RiskDecision(False, "max weekly loss reached")

        if open_positions_count >= self.cfg.max_open_positions:
            return RiskDecision(False, "max open positions reached")

        # Margin check is an approximation (Section 8): true margin needs
        # broker-specific contract size/leverage, only available once a
        # broker adapter is connected (Phase 12/13). This treats notional
        # exposure / max_leverage as required margin, same simplification
        # BaseStrategy.calculate_position_size already documents.
        notional = position_size.units * entry_price
        required_margin = notional / self.cfg.max_leverage
        available_margin = equity * self.cfg.max_margin_usage
        if required_margin > available_margin:
            return RiskDecision(False, "insufficient margin under max_leverage/max_margin_usage limits")

        return RiskDecision(True, "approved")

    def notify_order_opened(self) -> None:
        self.trades_opened_today += 1
