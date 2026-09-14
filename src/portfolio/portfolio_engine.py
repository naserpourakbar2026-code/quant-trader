"""Phase 11: Portfolio engine (CLAUDE.md Section 20).

"Don't assume one strategy is optimal." Screens many strategy x symbol x
timeframe combinations with the vectorbt engine (Phase 6), keeps only the
statistically-usable, best-ranked survivors (Section 12's "avoid wasting
compute" applies to a correlation matrix too), computes their pairwise
return correlation, and tests whether a couple of *basic* allocation
schemes (equal weight; inverse volatility) improve Sharpe, Sortino,
drawdown and return consistency versus the single best component alone.

This is deliberately not a portfolio optimizer (no mean-variance/Markowitz
solve, no leverage or margin-aware sizing) — Section 20 asks for "basic
allocation" to test whether diversification helps at all, not a final
capital-allocation system. Combining per-bar return series across
different timeframes is not attempted; components are grouped by
timeframe (bars from an H1 strategy and an H4 strategy are not the same
unit of time) and combined only within a group.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.backtest.vectorbt_engine import filter_top_candidates, get_bar_returns, run_screening
from src.core.config import PortfolioConfig, load_settings
from src.core.db import Database
from src.core.reproducibility import git_commit_hash, library_versions, python_version
from src.portfolio.run_store import PortfolioRecord, save_run

_SECONDS_PER_YEAR = 365.25 * 24 * 3600


def component_label(strategy: str, symbol: str, timeframe: str) -> str:
    return f"{strategy}|{symbol}|{timeframe}"


@dataclass
class PortfolioComponent:
    strategy: str
    symbol: str
    timeframe: str
    scenario: str
    metrics: dict
    returns: pd.Series  # per-bar returns, indexed by timestamp

    @property
    def label(self) -> str:
        return component_label(self.strategy, self.symbol, self.timeframe)


def build_component(
    strategy_family: str,
    symbol: str,
    timeframe: str,
    raw_df: pd.DataFrame,
    *,
    scenario: str | None = None,
    strategy_params: dict | None = None,
) -> PortfolioComponent:
    """Screen one combination (Phase 6) and pair its metrics with its
    per-bar return series in one PortfolioComponent."""
    screening = run_screening(
        strategy_family, symbol, timeframe, raw_df,
        strategy_params=strategy_params, scenario=scenario, persist=False,
    )
    returns = get_bar_returns(
        strategy_family, raw_df, strategy_params=strategy_params, scenario=screening.scenario,
    )
    return PortfolioComponent(
        strategy=strategy_family,
        symbol=symbol,
        timeframe=timeframe,
        scenario=screening.scenario,
        metrics=screening.metrics,
        returns=returns,
    )


def _annualization_factor(index: pd.DatetimeIndex) -> float:
    """Bars-per-year, inferred from the actual median spacing between
    bars in `index` — never a hard-coded per-timeframe constant, since
    the data itself already says how far apart its bars are."""
    if len(index) < 2:
        return 1.0
    deltas = np.diff(index.values).astype("timedelta64[s]").astype(float)
    deltas = deltas[deltas > 0]
    if len(deltas) == 0:
        return 1.0
    median_seconds = float(np.median(deltas))
    return _SECONDS_PER_YEAR / median_seconds


def evaluate_returns(returns: pd.Series) -> dict:
    """Sharpe/Sortino/max-drawdown/total-return/consistency computed
    directly from a per-bar return series — used for the *combined*
    portfolio series, which (being a weighted sum of several components'
    returns) is no longer a single vectorbt Portfolio object we could
    call .sharpe_ratio() on."""
    if returns.empty:
        return {
            "total_return": None, "sharpe_ratio": None, "sortino_ratio": None,
            "max_drawdown": None, "positive_month_ratio": None,
        }

    ann_factor = _annualization_factor(returns.index)
    mean = float(returns.mean())
    std = float(returns.std(ddof=0))
    sharpe = (mean / std) * np.sqrt(ann_factor) if std > 0 else None
    if sharpe is not None and not np.isfinite(sharpe):
        sharpe = None

    downside = returns[returns < 0]
    downside_std = float(downside.std(ddof=0)) if len(downside) > 0 else 0.0
    sortino = (mean / downside_std) * np.sqrt(ann_factor) if downside_std > 0 else None
    if sortino is not None and not np.isfinite(sortino):
        sortino = None

    cum_value = (1.0 + returns).cumprod()
    running_max = cum_value.cummax()
    drawdown = (running_max - cum_value) / running_max
    # Negative, matching vectorbt's own Portfolio.max_drawdown() sign
    # convention (component metrics come from that) -- so the two are
    # directly comparable in compare_to_best_component() below.
    max_drawdown = -float(drawdown.max())

    total_return = float(cum_value.iloc[-1] - 1.0)

    monthly = (1.0 + returns).resample("ME").prod() - 1.0
    positive_month_ratio = float((monthly > 0).mean()) if len(monthly) > 0 else None

    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "positive_month_ratio": positive_month_ratio,
    }


def align_returns(components: list[PortfolioComponent]) -> pd.DataFrame:
    """Outer-join every component's per-bar returns into one DataFrame,
    filling gaps with 0.0 (a bar where a symbol's data doesn't cover, or
    the strategy has an open/flat state contributing no return) — a
    simplification stated here, not hidden: it will understate real
    correlation/volatility if components' date ranges barely overlap, so
    treat this as directional, not a certified risk model."""
    series = {c.label: c.returns for c in components}
    frame = pd.concat(series, axis=1, join="outer")
    return frame.fillna(0.0)


def correlation_matrix(returns_frame: pd.DataFrame) -> pd.DataFrame:
    return returns_frame.corr()


def allocate_equal_weight(components: list[PortfolioComponent]) -> dict[str, float]:
    n = len(components)
    return {c.label: 1.0 / n for c in components}


def allocate_inverse_volatility(components: list[PortfolioComponent]) -> dict[str, float]:
    """Lower-volatility components get more weight — still a basic,
    justifiable scheme (no correlation/covariance solve), matching
    CLAUDE.md Section 20's "basic allocation" ask. Falls back to equal
    weight if any component has zero/undefined volatility (inverse-vol
    weighting is undefined in that case)."""
    stds = {c.label: float(c.returns.std(ddof=0)) for c in components}
    if any(s <= 0 or not np.isfinite(s) for s in stds.values()):
        return allocate_equal_weight(components)
    inv = {label: 1.0 / s for label, s in stds.items()}
    total = sum(inv.values())
    return {label: w / total for label, w in inv.items()}


_ALLOCATORS = {
    "equal_weight": allocate_equal_weight,
    "inverse_volatility": allocate_inverse_volatility,
}


def combine(returns_frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    weight_series = pd.Series(weights)
    return returns_frame[weight_series.index].mul(weight_series, axis=1).sum(axis=1)


def compare_to_best_component(portfolio_metrics: dict, best_metrics: dict) -> dict:
    """CLAUDE.md Section 20: "test whether diversification improves
    Sharpe, Sortino, drawdown, and return consistency" — one boolean per
    metric, comparing the combined portfolio against the single best
    component alone. max_drawdown follows vectorbt's own sign convention
    (zero or negative -- less negative is a smaller, better drawdown),
    so "improved" is "higher value" for every metric here, drawdown
    included."""

    def _improved(portfolio_value, best_value) -> bool | None:
        if portfolio_value is None or best_value is None:
            return None
        return bool(portfolio_value > best_value)  # numpy bool_ isn't JSON-serializable

    return {
        "sharpe_improved": _improved(portfolio_metrics.get("sharpe_ratio"), best_metrics.get("sharpe_ratio")),
        "sortino_improved": _improved(portfolio_metrics.get("sortino_ratio"), best_metrics.get("sortino_ratio")),
        "drawdown_improved": _improved(portfolio_metrics.get("max_drawdown"), best_metrics.get("max_drawdown")),
        "consistency_improved": _improved(
            portfolio_metrics.get("positive_month_ratio"), best_metrics.get("positive_month_ratio")
        ),
    }


@dataclass
class PortfolioResult:
    run_id: str
    scenario: str
    components: list[PortfolioComponent]
    correlation: pd.DataFrame
    allocations: dict[str, dict] = field(default_factory=dict)  # method -> {weights, metrics, verdict}
    best_component_label: str = ""

    def to_text(self) -> str:
        lines = [
            "PORTFOLIO ANALYSIS REPORT",
            f"Scenario: {self.scenario} | Components: {len(self.components)}",
            "",
            "Individual components (ranked by Sharpe):",
        ]
        for c in sorted(self.components, key=lambda c: c.metrics.get("sharpe_ratio") or float("-inf"), reverse=True):
            lines.append(
                f"  {c.label:45s} return={c.metrics.get('total_return'):+.2%} "
                f"sharpe={c.metrics.get('sharpe_ratio')} sortino={c.metrics.get('sortino_ratio')} "
                f"maxdd={c.metrics.get('max_drawdown')} trades={c.metrics.get('trade_count')}"
            )
        lines.append(f"\nBest single component: {self.best_component_label}")

        if len(self.components) > 1:
            lines.append("\nPairwise return correlation:")
            lines.append(self.correlation.round(2).to_string())

        for method, outcome in self.allocations.items():
            lines.append(f"\nAllocation: {method}")
            for label, weight in outcome["weights"].items():
                lines.append(f"  {label:45s} weight={weight:.2%}")
            m = outcome["metrics"]
            lines.append(
                f"  portfolio: return={m.get('total_return')} sharpe={m.get('sharpe_ratio')} "
                f"sortino={m.get('sortino_ratio')} maxdd={m.get('max_drawdown')} "
                f"positive_month_ratio={m.get('positive_month_ratio')}"
            )
            verdict = outcome["verdict"]
            improved = [k.replace("_improved", "") for k, v in verdict.items() if v is True]
            not_improved = [k.replace("_improved", "") for k, v in verdict.items() if v is False]
            lines.append(
                f"  vs. best component: improved={improved or 'none'} not_improved={not_improved or 'none'} "
                "(CLAUDE.md Section 20 — diversification helping on paper is not a guarantee going forward)"
            )
        return "\n".join(lines)


def run_portfolio_analysis(
    components_data: list[tuple[str, str, str, pd.DataFrame]],
    *,
    scenario: str | None = None,
    top_n: int | None = None,
    min_trades: int | None = None,
    allocation_methods: list[str] | None = None,
    persist: bool = True,
    db: Database | None = None,
) -> PortfolioResult:
    """Screen every (strategy, symbol, timeframe, raw_df) in
    `components_data`, keep the top `top_n` reliable survivors (by
    Sharpe, min_trades floor), and evaluate `allocation_methods` for each
    timeframe group with more than one surviving component (components on
    different timeframes are never combined — see module docstring).
    """
    if not components_data:
        raise ValueError("components_data must not be empty")

    settings = load_settings()
    cfg: PortfolioConfig = settings.portfolio
    scenario = scenario or settings.execution.default_scenario
    top_n = top_n if top_n is not None else cfg.top_n
    min_trades = min_trades if min_trades is not None else cfg.min_trades
    allocation_methods = allocation_methods if allocation_methods is not None else cfg.allocation_methods

    built = [
        build_component(strategy, symbol, timeframe, raw_df, scenario=scenario)
        for strategy, symbol, timeframe, raw_df in components_data
    ]

    class _Ranked:
        """Adapter so filter_top_candidates() (built for ScreeningResult)
        can rank PortfolioComponents too, without duplicating its logic."""

        def __init__(self, component: PortfolioComponent):
            self.component = component
            self.metrics = component.metrics

    ranked = filter_top_candidates([_Ranked(c) for c in built], top_n=top_n, min_trades=min_trades)
    survivors = [r.component for r in ranked]
    if not survivors:
        raise ValueError(
            f"No component had >= {min_trades} trades — nothing statistically usable to build a portfolio from."
        )

    best_component = max(survivors, key=lambda c: c.metrics.get("sharpe_ratio") or float("-inf"))

    by_timeframe: dict[str, list[PortfolioComponent]] = {}
    for c in survivors:
        by_timeframe.setdefault(c.timeframe, []).append(c)
    group = max(by_timeframe.values(), key=len)  # the largest same-timeframe group is what gets combined

    correlation = correlation_matrix(align_returns(group)) if len(group) > 1 else pd.DataFrame()

    allocations: dict[str, dict] = {}
    if len(group) > 1:
        returns_frame = align_returns(group)
        for method in allocation_methods:
            weights = _ALLOCATORS[method](group)
            combined_returns = combine(returns_frame, weights)
            metrics = evaluate_returns(combined_returns)
            verdict = compare_to_best_component(metrics, best_component.metrics)
            allocations[method] = {"weights": weights, "metrics": metrics, "verdict": verdict}

    result = PortfolioResult(
        run_id=str(uuid.uuid4()),
        scenario=scenario,
        components=survivors,
        correlation=correlation,
        allocations=allocations,
        best_component_label=best_component.label,
    )

    if persist:
        save_run(
            PortfolioRecord(
                run_id=result.run_id,
                scenario=result.scenario,
                n_components=len(result.components),
                components=[
                    {
                        "strategy": c.strategy, "symbol": c.symbol, "timeframe": c.timeframe,
                        "scenario": c.scenario, "metrics": c.metrics,
                    }
                    for c in result.components
                ],
                correlation=result.correlation.to_dict(),
                allocations={
                    method: {"weights": o["weights"], "metrics": o["metrics"], "verdict": o["verdict"]}
                    for method, o in result.allocations.items()
                },
                best_component_label=result.best_component_label,
                code_version=git_commit_hash(),
                python_version=python_version(),
                library_versions=library_versions(["pandas", "numpy", "vectorbt"]),
            ),
            db=db,
        )

    return result
