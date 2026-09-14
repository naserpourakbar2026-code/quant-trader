"""Chart rendering (CLAUDE.md Section 30): Equity Curve, Drawdown,
Monthly Returns, Rolling Sharpe, Trade Distribution, Parameter Heatmap,
Walk-Forward Performance, Monte Carlo Distribution, Strategy Correlation.

Every chart is rendered with matplotlib (Agg backend, no display needed)
straight to a base64 PNG data URI — the user's explicit preference (see
CLAUDE.md Section 30's note) is one self-contained HTML file openable
directly in a browser, no server and no external JS/network fetch, which
an embedded `<img>` satisfies and a JS charting library would not.

Every function returns `None` (never a fabricated or empty-looking
chart) when there isn't enough real data to draw something meaningful —
the HTML renderer is expected to show an explanatory message instead.
"""
from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_FIGSIZE = (8, 4)


def _fig_to_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_equity_series(trades: list, initial_capital: float) -> pd.Series:
    """Cumulative equity after each closed trade, in exit-time order --
    the shared basis for the equity/drawdown/monthly-return charts. An
    empty `trades` list yields a single-point series at `initial_capital`
    (nothing to chart, not a crash)."""
    ordered = sorted(trades, key=lambda t: t.exit_time)
    if not ordered:
        return pd.Series([initial_capital], index=[pd.Timestamp.now(tz="UTC")])
    index = [t.exit_time for t in ordered]
    equity = initial_capital + np.cumsum([t.pnl for t in ordered])
    return pd.Series(equity, index=pd.DatetimeIndex(index))


def equity_curve_chart(equity: pd.Series) -> str | None:
    if len(equity) < 2:
        return None
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.plot(equity.index, equity.values, color="#2563eb")
    ax.set_title("Equity Curve")
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    return _fig_to_data_uri(fig)


def drawdown_chart(equity: pd.Series) -> str | None:
    if len(equity) < 2:
        return None
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.fill_between(drawdown.index, drawdown.values * 100, 0, color="#dc2626", alpha=0.4)
    ax.plot(drawdown.index, drawdown.values * 100, color="#dc2626")
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    return _fig_to_data_uri(fig)


def monthly_returns_chart(equity: pd.Series) -> str | None:
    if len(equity) < 2:
        return None
    monthly = equity.resample("ME").last().ffill()
    monthly_returns = monthly.pct_change().dropna()
    if monthly_returns.empty:
        return None
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    colors = ["#16a34a" if v >= 0 else "#dc2626" for v in monthly_returns.values]
    ax.bar(monthly_returns.index.strftime("%Y-%m"), monthly_returns.values * 100, color=colors)
    ax.set_title("Monthly Returns")
    ax.set_ylabel("Return (%)")
    ax.axhline(0, color="black", linewidth=0.8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(alpha=0.3, axis="y")
    return _fig_to_data_uri(fig)


def rolling_sharpe_chart(trades: list, *, window: int = 20) -> str | None:
    """Rolling mean/std of each trade's R-multiple (not raw PnL, which
    isn't scale-invariant) over a trailing window of trades -- an
    expectancy-stability signal, not an annualized Sharpe ratio (trades
    aren't evenly spaced in time, so "annualizing" a per-trade figure
    would be a fabricated precision)."""
    ordered = sorted(trades, key=lambda t: t.exit_time)
    r_multiples = pd.Series([t.r_multiple for t in ordered if t.r_multiple is not None])
    if len(r_multiples) < window:
        return None
    rolling_mean = r_multiples.rolling(window).mean()
    rolling_std = r_multiples.rolling(window).std()
    rolling_sharpe = (rolling_mean / rolling_std).dropna()
    if rolling_sharpe.empty:
        return None
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.plot(range(len(rolling_sharpe)), rolling_sharpe.values, color="#7c3aed")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Rolling Sharpe (R-multiple, {window}-trade window)")
    ax.set_xlabel("Trade #")
    ax.grid(alpha=0.3)
    return _fig_to_data_uri(fig)


def trade_distribution_chart(trades: list) -> str | None:
    r_multiples = [t.r_multiple for t in trades if t.r_multiple is not None]
    if len(r_multiples) < 2:
        return None
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.hist(r_multiples, bins=min(30, max(5, len(r_multiples) // 3)), color="#0891b2", edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Trade Distribution (R-multiple)")
    ax.set_xlabel("R-multiple")
    ax.set_ylabel("Trade count")
    ax.grid(alpha=0.3, axis="y")
    return _fig_to_data_uri(fig)


def parameter_heatmap_chart(experiments: list) -> tuple[str, str] | None:
    """Needs a group of vectorbt experiments (same strategy/symbol/
    timeframe) that vary in exactly two parameters -- e.g. the output of
    src.backtest.vectorbt_engine.screen_parameter_grid(). Returns
    (data_uri, description) for the largest such group found, or None
    if no group of >= 4 rows varying in exactly two parameters exists."""
    vectorbt_rows = [e for e in experiments if e.engine == "vectorbt"]
    groups: dict[tuple, list] = {}
    for e in vectorbt_rows:
        groups.setdefault((e.strategy, e.symbol, e.timeframe), []).append(e)

    best_group, best_params = None, None
    for key, rows in groups.items():
        if len(rows) < 4:
            continue
        candidate_keys = set().union(*(set(row.parameters) for row in rows))
        varying = sorted(k for k in candidate_keys if len({row.parameters.get(k) for row in rows}) > 1)
        if len(varying) == 2 and (best_group is None or len(rows) > len(best_group)):
            best_group, best_params = rows, varying

    if best_group is None:
        return None

    p1, p2 = best_params
    frame = pd.DataFrame(
        [{p1: row.parameters[p1], p2: row.parameters[p2], "sharpe": row.metrics.get("sharpe_ratio")} for row in best_group]
    )
    pivot = frame.pivot_table(index=p1, columns=p2, values="sharpe", aggfunc="mean")
    if pivot.empty:
        return None

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel(p2)
    ax.set_ylabel(p1)
    ax.set_title(f"Parameter Heatmap (Sharpe): {best_group[0].strategy}/{best_group[0].symbol}/{best_group[0].timeframe}")
    fig.colorbar(im, ax=ax, label="Sharpe ratio")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.values[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8)
    description = f"{p1} x {p2}, {len(best_group)} runs"
    return _fig_to_data_uri(fig), description


def walk_forward_chart(windows: list) -> tuple[str, str] | None:
    """Charts the single run_id with the most windows (mixing different
    runs' window_index on one axis would be misleading)."""
    if not windows:
        return None
    by_run: dict[str, list] = {}
    for w in windows:
        by_run.setdefault(w.run_id, []).append(w)
    run_id, run_windows = max(by_run.items(), key=lambda kv: len(kv[1]))
    run_windows = sorted(run_windows, key=lambda w: w.window_index)

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    colors = ["#16a34a" if w.oos_passed else "#dc2626" for w in run_windows]
    ax.bar([w.window_index for w in run_windows], [w.oos_objective for w in run_windows], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Walk-Forward Performance: {run_windows[0].strategy}/{run_windows[0].symbol}/{run_windows[0].timeframe}")
    ax.set_xlabel("Window")
    ax.set_ylabel("OOS objective")
    ax.grid(alpha=0.3, axis="y")
    description = f"{len(run_windows)} windows, {sum(w.oos_passed for w in run_windows)} passed"
    return _fig_to_data_uri(fig), description


def monte_carlo_distribution_chart(run) -> str | None:
    histogram = run.return_histogram
    if not histogram or not histogram.get("counts"):
        return None
    bin_edges = np.array(histogram["bin_edges"])
    counts = np.array(histogram["counts"])
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    widths = np.diff(bin_edges)

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.bar(centers * 100, counts, width=widths * 100, color="#0891b2", edgecolor="white")
    ax.axvline(run.median_return * 100, color="black", linewidth=1.2, label="median")
    ax.axvline(run.p5_return * 100, color="#dc2626", linewidth=1.0, linestyle="--", label="5th pct")
    ax.axvline(run.p95_return * 100, color="#16a34a", linewidth=1.0, linestyle="--", label="95th pct")
    ax.set_title(f"Monte Carlo Distribution: {run.strategy}/{run.symbol}/{run.timeframe} ({run.n_simulations} sims)")
    ax.set_xlabel("Simulated total return (%)")
    ax.set_ylabel("Simulation count")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    return _fig_to_data_uri(fig)


def correlation_heatmap_chart(portfolio_run) -> str | None:
    correlation = portfolio_run.correlation
    if not correlation or len(correlation) < 2:
        return None
    frame = pd.DataFrame(correlation)
    labels = list(frame.columns)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(frame.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("Strategy Correlation")
    fig.colorbar(im, ax=ax, label="correlation")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{frame.values[i, j]:.2f}", ha="center", va="center", fontsize=6)
    return _fig_to_data_uri(fig)
