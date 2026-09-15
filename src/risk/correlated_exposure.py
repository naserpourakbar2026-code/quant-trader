"""Correlated exposure check (CLAUDE.md Section 7's `max_correlated_exposure`)
— deliberately a standalone utility, not part of `RiskEngine` (see that
module's docstring for why): every engine in this codebase, RiskEngine
included, is scoped to one symbol per call, so there is nothing to be
"correlated" with inside a single-symbol session. This is for a
multi-symbol caller — a live/paper execution orchestrator running
several strategy/symbol combinations at once — that doesn't exist yet
in this codebase. It reuses the same *shape* of correlation matrix
Phase 11's `src.portfolio.portfolio_engine.correlation_matrix()`
produces (a symbol-keyed, symmetric DataFrame), so whenever such an
orchestrator is built, wiring the two together is a matter of computing
one and passing it in here, not inventing new plumbing.

`max_correlated_exposure` is read the same way `risk_per_trade` already
is: as a fraction of equity, applied to *risk amount* (dollars actually
at risk, i.e. `PositionSizeResult.risk_amount`), not raw notional --
notional scales with leverage and would make the same percentage
figure mean wildly different things account to account; risk amount is
what CLAUDE.md Section 7 elsewhere consistently sizes against.
"""
from __future__ import annotations

import pandas as pd

from src.core.config import RiskConfig
from src.risk.engine import RiskDecision
from src.strategies.base import PositionSizeResult


class CorrelatedExposureMonitor:
    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg

    @staticmethod
    def _correlation(correlation: pd.DataFrame, a: str, b: str) -> float | None:
        if a not in correlation.index or b not in correlation.columns:
            return None
        value = correlation.loc[a, b]
        return float(value) if pd.notna(value) else None

    def check(
        self,
        *,
        symbol: str,
        position_size: PositionSizeResult,
        equity: float,
        open_risk_by_symbol: dict[str, float],
        correlation: pd.DataFrame,
        correlation_threshold: float = 0.5,
    ) -> RiskDecision:
        """`open_risk_by_symbol` is every other currently-open position's
        risk_amount, keyed by symbol (never including `symbol` itself —
        a symbol's own existing exposure is a max_open_positions/
        stacking concern, not a *correlated*-exposure one). A symbol
        missing from `correlation` (never screened together, e.g.) is
        treated as uncorrelated, not as a reason to reject — silence in
        the data isn't evidence of safety, but this monitor's job is
        Section 7's specific correlated-exposure cap, not a substitute
        for the trader having a real correlation estimate."""
        correlated_risk = position_size.risk_amount
        for other_symbol, other_risk in open_risk_by_symbol.items():
            if other_symbol == symbol:
                continue
            corr = self._correlation(correlation, symbol, other_symbol)
            if corr is not None and abs(corr) >= correlation_threshold:
                correlated_risk += other_risk

        limit = equity * self.cfg.max_correlated_exposure
        if correlated_risk > limit:
            return RiskDecision(
                False,
                f"correlated exposure {correlated_risk:.2f} exceeds max_correlated_exposure limit "
                f"{limit:.2f} ({self.cfg.max_correlated_exposure:.2%} of equity)",
            )
        return RiskDecision(True, "approved")
