"""Phase 6: vectorbt research engine (CLAUDE.md Section 12).

Fast, vectorized parameter/symbol/timeframe screening — a coarse filter
("avoid wasting compute on obviously poor parameter regions"), not the
final word. Backtrader (Phase 7) re-validates the survivors with
realistic, event-driven execution (order rejection, margin, partial
fills); this engine only measures the raw entry/exit timing edge under
cost assumptions, using each strategy's own generate_signals_vectorized()
table (direction/confidence/stop_loss/take_profit) so the screening logic
can never drift from what generate_signal() would decide live.

Every run is persisted as an Experiment (src.backtest.experiment_store)
with a unique ID and full reproducibility metadata (Section 36).

get_trade_returns() exposes the same run's per-trade returns without
persisting anything — the raw material Phase 10's Monte Carlo engine
(src.montecarlo.mc_engine) resamples from. get_bar_returns() exposes the
same run's per-bar return series — the raw material Phase 11's portfolio
engine (src.portfolio.portfolio_engine) aligns across combinations.
"""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
import vectorbt as vbt

from src.backtest.experiment_store import ExperimentRecord, save_experiment
from src.core.config import load_settings
from src.core.db import Database
from src.core.reproducibility import git_commit_hash, library_versions, python_version
from src.data.ingestion import data_version_for
from src.features.engine import FeatureParams, compute_features
from src.strategies import create_strategy


@dataclass
class ScreeningResult:
    experiment_id: str
    strategy: str
    symbol: str
    timeframe: str
    scenario: str
    parameters: dict
    metrics: dict


def _direction_transitions(direction: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Convert a per-bar "desired direction" state into vectorbt entry/exit
    signals: entering when the state first becomes LONG/SHORT, exiting
    when it stops being that direction (to FLAT or straight to the
    opposite side)."""
    is_long = direction == "LONG"
    is_short = direction == "SHORT"
    long_entries = is_long & ~is_long.shift(1, fill_value=False)
    long_exits = ~is_long & is_long.shift(1, fill_value=False)
    short_entries = is_short & ~is_short.shift(1, fill_value=False)
    short_exits = ~is_short & is_short.shift(1, fill_value=False)
    return long_entries, long_exits, short_entries, short_exits


def _extract_metrics(pf: "vbt.Portfolio") -> dict:
    trade_count = int(pf.trades.count())
    metrics = {
        "total_return": float(pf.total_return()),
        "sharpe_ratio": float(pf.sharpe_ratio()),
        "sortino_ratio": float(pf.sortino_ratio()),
        "max_drawdown": float(pf.max_drawdown()),
        "trade_count": trade_count,
    }
    if trade_count > 0:
        metrics["win_rate"] = float(pf.trades.win_rate())
        metrics["profit_factor"] = float(pf.trades.profit_factor())
        metrics["expectancy"] = float(pf.trades.expectancy())
    else:
        metrics["win_rate"] = None
        metrics["profit_factor"] = None
        metrics["expectancy"] = None
    for key, value in metrics.items():
        if isinstance(value, float) and not math.isfinite(value):
            metrics[key] = None
    return metrics


def _build_portfolio(
    strategy_family: str,
    raw_df: pd.DataFrame,
    *,
    strategy_params: dict | None = None,
    feature_params: FeatureParams | None = None,
    scenario: str | None = None,
    initial_capital: float | None = None,
    warmup_df: pd.DataFrame | None = None,
) -> tuple["vbt.Portfolio", pd.DataFrame, str, object]:
    """Shared signal-to-portfolio construction used by both run_screening()
    (aggregate metrics) and get_trade_returns() (per-trade returns, for
    Phase 10 Monte Carlo resampling) — one place builds the vectorbt
    Portfolio so both consumers see the exact same trades.

    Returns (pf, indexed_feature_df_scored_range, scenario, strategy).
    """
    settings = load_settings()
    scenario = scenario or settings.execution.default_scenario
    if scenario not in settings.execution.costs:
        raise ValueError(f"Unknown execution scenario {scenario!r}; expected one of {list(settings.execution.costs)}")
    cost = settings.execution.costs[scenario]
    capital = initial_capital if initial_capital is not None else settings.account.initial_capital

    has_warmup = warmup_df is not None and not warmup_df.empty
    combined_raw = pd.concat([warmup_df, raw_df], ignore_index=True) if has_warmup else raw_df

    feature_df = compute_features(combined_raw, feature_params)
    indexed = feature_df.set_index("timestamp")

    strat = create_strategy(strategy_family, strategy_params)
    table = strat.generate_signals_vectorized(feature_df).set_axis(indexed.index)

    if has_warmup:
        cutoff = len(warmup_df)
        indexed = indexed.iloc[cutoff:]
        table = table.iloc[cutoff:]

    close = indexed["close"]
    long_entries, long_exits, short_entries, short_exits = _direction_transitions(table["direction"])

    sl_pct = ((close - table["stop_loss"]).abs() / close).replace([np.inf, -np.inf], np.nan)
    tp_pct = ((table["take_profit"] - close).abs() / close).replace([np.inf, -np.inf], np.nan)

    pf = vbt.Portfolio.from_signals(
        close,
        entries=long_entries,
        exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
        sl_stop=sl_pct,
        tp_stop=tp_pct,
        init_cash=capital,
        fees=cost.commission_pct,
        slippage=cost.slippage_pct,
    )
    return pf, indexed, scenario, strat


def run_screening(
    strategy_family: str,
    symbol: str,
    timeframe: str,
    raw_df: pd.DataFrame,
    *,
    strategy_params: dict | None = None,
    feature_params: FeatureParams | None = None,
    scenario: str | None = None,
    initial_capital: float | None = None,
    warmup_df: pd.DataFrame | None = None,
    persist: bool = True,
    db: Database | None = None,
) -> ScreeningResult:
    """Screen one strategy/symbol/timeframe/parameter combination.

    `raw_df` is standardized candle data (src.data.schema columns) for a
    single symbol/timeframe — e.g. from src.data.csv_loader.load_csv().

    `warmup_df`, if given, is *earlier* standardized candles prepended
    only so every indicator has proper history (no artificial NaN
    warm-up gap) — used by Phase 9's walk-forward engine so a
    validation/OOS window's indicators aren't computed in a vacuum.
    Entries/exits/metrics are still computed only over `raw_df`'s own
    rows; `warmup_df` never contributes a trade or a metric.
    """
    pf, indexed, scenario, strat = _build_portfolio(
        strategy_family,
        raw_df,
        strategy_params=strategy_params,
        feature_params=feature_params,
        scenario=scenario,
        initial_capital=initial_capital,
        warmup_df=warmup_df,
    )

    metrics = _extract_metrics(pf)
    experiment_id = str(uuid.uuid4())

    if persist:
        save_experiment(
            ExperimentRecord(
                experiment_id=experiment_id,
                engine="vectorbt",
                strategy=strategy_family,
                symbol=symbol,
                timeframe=timeframe,
                scenario=scenario,
                parameters=dict(strat.params),
                date_range_start=indexed.index[0].to_pydatetime(),
                date_range_end=indexed.index[-1].to_pydatetime(),
                metrics=metrics,
                data_version=data_version_for(symbol, timeframe),
                code_version=git_commit_hash(),
                python_version=python_version(),
                library_versions=library_versions(),
            ),
            db=db,
        )

    return ScreeningResult(
        experiment_id=experiment_id,
        strategy=strategy_family,
        symbol=symbol,
        timeframe=timeframe,
        scenario=scenario,
        parameters=dict(strat.params),
        metrics=metrics,
    )


def get_trade_returns(
    strategy_family: str,
    raw_df: pd.DataFrame,
    *,
    strategy_params: dict | None = None,
    feature_params: FeatureParams | None = None,
    scenario: str | None = None,
    initial_capital: float | None = None,
    warmup_df: pd.DataFrame | None = None,
) -> np.ndarray:
    """Per-trade fractional returns from the same vectorbt Portfolio
    run_screening() would build (same signals, same cost scenario) — the
    empirical distribution Phase 10's Monte Carlo engine resamples from
    (CLAUDE.md Section 17). Not persisted as an Experiment: this is a raw
    input to Monte Carlo, not a result in its own right.
    """
    pf, _indexed, _scenario, _strat = _build_portfolio(
        strategy_family,
        raw_df,
        strategy_params=strategy_params,
        feature_params=feature_params,
        scenario=scenario,
        initial_capital=initial_capital,
        warmup_df=warmup_df,
    )
    return pf.trades.returns.values


def get_bar_returns(
    strategy_family: str,
    raw_df: pd.DataFrame,
    *,
    strategy_params: dict | None = None,
    feature_params: FeatureParams | None = None,
    scenario: str | None = None,
    initial_capital: float | None = None,
    warmup_df: pd.DataFrame | None = None,
) -> pd.Series:
    """Per-bar (not per-trade) portfolio returns, indexed by timestamp,
    from the same vectorbt Portfolio run_screening() would build — the
    time series Phase 11's portfolio engine (src.portfolio.portfolio_engine)
    aligns across strategy/symbol/timeframe combinations to compute
    correlation and combined-portfolio metrics. Not persisted on its own.
    """
    pf, _indexed, _scenario, _strat = _build_portfolio(
        strategy_family,
        raw_df,
        strategy_params=strategy_params,
        feature_params=feature_params,
        scenario=scenario,
        initial_capital=initial_capital,
        warmup_df=warmup_df,
    )
    return pf.returns()


def screen_parameter_grid(
    strategy_family: str,
    symbol: str,
    timeframe: str,
    raw_df: pd.DataFrame,
    param_grid: dict[str, list],
    **kwargs,
) -> list[ScreeningResult]:
    """Run run_screening() once per combination in `param_grid` (a dict of
    param_name -> list of values to try), e.g.
    {"atr_stop_multiplier": [1.5, 2.0, 2.5], "take_profit_r_multiple": [1.5, 2.0, 3.0]}.
    Only parameters with a stated economic justification belong in a grid
    (CLAUDE.md Section 14) — this function doesn't police that; the
    caller decides what's worth sweeping.
    """
    if not param_grid:
        return [run_screening(strategy_family, symbol, timeframe, raw_df, **kwargs)]

    names = list(param_grid.keys())
    results = []
    for combo in product(*param_grid.values()):
        params = dict(zip(names, combo))
        results.append(
            run_screening(strategy_family, symbol, timeframe, raw_df, strategy_params=params, **kwargs)
        )
    return results


def filter_top_candidates(
    results: list[ScreeningResult],
    *,
    top_n: int = 5,
    min_trades: int = 10,
) -> list[ScreeningResult]:
    """Rank by Sharpe ratio, excluding statistically unreliable results
    (too few trades) — the concrete mechanism behind Section 12's "avoid
    wasting compute on obviously poor parameter regions": only the top
    survivors here are worth the cost of Backtrader validation (Phase 7).
    """
    reliable = [r for r in results if (r.metrics.get("trade_count") or 0) >= min_trades]
    ranked = sorted(reliable, key=lambda r: r.metrics.get("sharpe_ratio") or float("-inf"), reverse=True)
    return ranked[:top_n]
