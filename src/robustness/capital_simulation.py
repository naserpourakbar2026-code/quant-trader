"""Capital Simulation across risk levels (CLAUDE.md Section 33): replay
a strategy's own observed closed trades — expressed as R-multiples
(pnl / risk_amount), which are risk-level-independent — compounded at
each of the configured risk levels (settings.risk.allowed_risk_levels,
not a new hard-coded list) to see how €2,000 (or whatever
initial_capital is passed) would actually have behaved at 0.25% vs 1%
risk per trade, never assuming historical returns repeat (Section 33's
own words) — reusing the same bootstrap machinery Phase 10's Monte
Carlo already validated for probability-of-ruin estimation, just fed a
risk-scaled R-multiple series instead of vectorbt's raw per-trade
returns.

Why R-multiples: for fixed-fractional position sizing (every other risk
calculation in this codebase — BaseStrategy.calculate_position_size,
RiskEngine's margin check — already assumes this), a trade's return on
equity is `r_multiple * risk_pct` by construction (risk_amount = equity
* risk_pct, pnl = r_multiple * risk_amount). So replaying the *same*
sequence of R-multiples at a *different* risk_pct is exactly "what if
I'd sized this trade differently" — not a new assumption, just the
existing sizing formula run backwards.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.execution.trade_store import PaperTradeRecord
from src.montecarlo.mc_engine import simulate


@dataclass
class RiskLevelSimulation:
    risk_pct: float
    final_equity: float
    max_drawdown_eur: float
    max_losing_streak: int
    probability_of_ruin: float
    avg_monthly_return: float | None
    worst_monthly_return: float | None
    best_monthly_return: float | None
    avg_margin_utilization: float
    worst_margin_utilization: float


@dataclass
class CapitalSimulationResult:
    strategy: str
    symbol: str
    timeframe: str
    initial_capital: float
    n_trades: int
    risk_levels: list[RiskLevelSimulation] = field(default_factory=list)

    def to_text(self) -> str:
        lines = [
            "CAPITAL SIMULATION",
            f"Strategy: {self.strategy} | Symbol: {self.symbol} | Timeframe: {self.timeframe} | "
            f"Initial capital: {self.initial_capital} | Trades replayed: {self.n_trades}",
            "",
        ]
        for r in self.risk_levels:
            lines.append(
                f"  risk={r.risk_pct:.2%}  final_equity={r.final_equity:,.2f}  "
                f"max_dd={r.max_drawdown_eur:,.2f}  max_losing_streak={r.max_losing_streak}  "
                f"P(ruin)={r.probability_of_ruin:.2%}  avg_margin_util={r.avg_margin_utilization:.1%}  "
                f"worst_margin_util={r.worst_margin_utilization:.1%}"
            )
        return "\n".join(lines)


def _stop_distance_pct(trade: PaperTradeRecord) -> float | None:
    if not trade.entry_price:
        return None
    return abs(trade.entry_price - trade.stop_loss) / trade.entry_price


def run_capital_simulation(
    trades: list[PaperTradeRecord],
    *,
    strategy: str,
    symbol: str,
    timeframe: str,
    initial_capital: float,
    risk_levels: list[float],
    max_leverage: float,
    n_simulations: int = 1000,
    ruin_threshold: float = 0.5,
    seed: int | None = None,
) -> CapitalSimulationResult:
    ordered = sorted((t for t in trades if t.risk_amount), key=lambda t: t.exit_time)
    r_multiples = np.array([t.pnl / t.risk_amount for t in ordered])
    stop_pcts = np.array([_stop_distance_pct(t) for t in ordered], dtype=float)
    exit_times = pd.DatetimeIndex([t.exit_time for t in ordered]) if ordered else pd.DatetimeIndex([])

    results = []
    for risk_pct in risk_levels:
        if len(r_multiples) == 0:
            results.append(RiskLevelSimulation(
                risk_pct=risk_pct, final_equity=initial_capital, max_drawdown_eur=0.0, max_losing_streak=0,
                probability_of_ruin=0.0, avg_monthly_return=None, worst_monthly_return=None,
                best_monthly_return=None, avg_margin_utilization=0.0, worst_margin_utilization=0.0,
            ))
            continue

        scaled_returns = r_multiples * risk_pct
        equity = initial_capital * np.cumprod(1.0 + scaled_returns)
        running_max = np.maximum.accumulate(np.concatenate([[initial_capital], equity]))[1:]
        drawdown_eur = running_max - equity
        max_drawdown_eur = float(drawdown_eur.max())

        losing = scaled_returns < 0
        streak = 0
        max_streak = 0
        for is_loss in losing:
            streak = streak + 1 if is_loss else 0
            max_streak = max(max_streak, streak)

        margin_utilization = np.where(
            np.isfinite(stop_pcts) & (stop_pcts > 0), risk_pct / (stop_pcts * max_leverage), np.nan,
        )
        valid_margin = margin_utilization[np.isfinite(margin_utilization)]

        monthly_returns = None
        if len(exit_times) > 0:
            equity_series = pd.Series(equity, index=exit_times)
            monthly = equity_series.resample("ME").last().ffill()
            monthly_returns = monthly.pct_change().dropna()

        sim = simulate(
            scaled_returns, n_simulations=n_simulations, initial_capital=initial_capital,
            ruin_threshold=ruin_threshold, execution_noise_std=0.0, seed=seed,
        )

        results.append(RiskLevelSimulation(
            risk_pct=risk_pct,
            final_equity=float(equity[-1]),
            max_drawdown_eur=max_drawdown_eur,
            max_losing_streak=max_streak,
            probability_of_ruin=float(sim["ruin"].mean()),
            avg_monthly_return=float(monthly_returns.mean()) if monthly_returns is not None and len(monthly_returns) else None,
            worst_monthly_return=float(monthly_returns.min()) if monthly_returns is not None and len(monthly_returns) else None,
            best_monthly_return=float(monthly_returns.max()) if monthly_returns is not None and len(monthly_returns) else None,
            avg_margin_utilization=float(valid_margin.mean()) if len(valid_margin) else 0.0,
            worst_margin_utilization=float(valid_margin.max()) if len(valid_margin) else 0.0,
        ))

    return CapitalSimulationResult(
        strategy=strategy, symbol=symbol, timeframe=timeframe, initial_capital=initial_capital,
        n_trades=len(ordered), risk_levels=results,
    )
