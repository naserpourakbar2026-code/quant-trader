"""Self-contained HTML report (CLAUDE.md Section 30's explicit user
preference): one pre-rendered file, openable directly in a browser, no
server or notebook needed to see results. Every chart is an embedded
base64 PNG (src.reporting.charts) — no external JS, no network fetch.

Every section says "no data yet" rather than draw an empty/fabricated
chart when a phase hasn't been run against real data — this framework's
whole premise is never claiming a result that isn't there.
"""
from __future__ import annotations

import html
from datetime import datetime, timezone

from src.reporting import charts
from src.reporting.data import ReportData


def _table(rows: list[dict], *, empty_message: str) -> str:
    if not rows:
        return f'<p class="muted">{html.escape(empty_message)}</p>'
    columns = list(rows[0].keys())
    header = "".join(f"<th>{html.escape(str(c))}</th>" for c in columns)
    body = "\n".join(
        "<tr>" + "".join(f"<td>{html.escape(str(row.get(c, '')))}</td>" for c in columns) + "</tr>"
        for row in rows
    )
    return f'<div class="table-wrap"><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>'


def _chart_or_message(data_uri: str | None, *, alt: str, empty_message: str) -> str:
    if not data_uri:
        return f'<p class="muted">{html.escape(empty_message)}</p>'
    return f'<img src="{data_uri}" alt="{html.escape(alt)}" class="chart">'


def _fmt_pct(value) -> str:
    return f"{value:.2%}" if isinstance(value, (int, float)) else str(value)


def _fmt(value, spec: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        try:
            return format(value, spec or ".4g")
        except (ValueError, TypeError):
            return str(value)
    return str(value)


def _experiments_section(data: ReportData) -> str:
    rows = [
        {
            "engine": e.engine, "strategy": e.strategy, "symbol": e.symbol, "timeframe": e.timeframe,
            "scenario": e.scenario, "return": _fmt_pct(e.metrics.get("total_return")),
            "sharpe": _fmt(e.metrics.get("sharpe_ratio")), "sortino": _fmt(e.metrics.get("sortino_ratio")),
            "max_dd": _fmt_pct(e.metrics.get("max_drawdown")), "trades": e.metrics.get("trade_count"),
            "created_at": e.date_range_end,
        }
        for e in data.experiments[:100]
    ]
    heatmap = charts.parameter_heatmap_chart(data.experiments)
    heatmap_html = (
        f'<img src="{heatmap[0]}" alt="parameter heatmap" class="chart"><p class="caption">{html.escape(heatmap[1])}</p>'
        if heatmap else '<p class="muted">No group of >= 4 same-strategy/symbol/timeframe runs varying in exactly '
                         "two parameters yet — run `python main.py backtest` over a parameter grid to populate this.</p>"
    )
    return f"""
    <section id="experiments">
      <h2>Screening &amp; Validation Runs (Phases 6-8)</h2>
      <p class="muted">{len(data.experiments)} total experiment(s) recorded, showing up to 100 most recent.</p>
      {_table(rows, empty_message="No experiments yet — run `python main.py backtest` or `optimize`.")}
      <h3>Parameter Heatmap</h3>
      {heatmap_html}
    </section>"""


def _walkforward_section(data: ReportData) -> str:
    rows = [
        {
            "run_id": w.run_id[:8], "window": w.window_index, "strategy": w.strategy, "symbol": w.symbol,
            "timeframe": w.timeframe, "oos_objective": _fmt(w.oos_objective), "oos_passed": w.oos_passed,
            "stability_score": _fmt(w.stability_score),
        }
        for w in data.walkforward_windows[:100]
    ]
    chart = charts.walk_forward_chart(data.walkforward_windows)
    chart_html = (
        f'<img src="{chart[0]}" alt="walk-forward performance" class="chart"><p class="caption">{html.escape(chart[1])}</p>'
        if chart else '<p class="muted">No walk-forward runs yet — run `python main.py walk-forward`.</p>'
    )
    return f"""
    <section id="walkforward">
      <h2>Walk-Forward Analysis (Phase 9)</h2>
      {_table(rows, empty_message="No walk-forward windows yet — run `python main.py walk-forward`.")}
      <h3>Walk-Forward Performance</h3>
      {chart_html}
    </section>"""


def _montecarlo_section(data: ReportData) -> str:
    rows = [
        {
            "strategy": r.strategy, "symbol": r.symbol, "timeframe": r.timeframe, "sims": r.n_simulations,
            "median_return": _fmt_pct(r.median_return), "p5_return": _fmt_pct(r.p5_return),
            "p95_return": _fmt_pct(r.p95_return), "p_ruin": _fmt_pct(r.probability_of_ruin),
            "is_fragile": r.is_fragile,
        }
        for r in data.montecarlo_runs[:100]
    ]
    chart_html = "".join(
        f'<img src="{uri}" alt="monte carlo distribution" class="chart">'
        for uri in (charts.monte_carlo_distribution_chart(r) for r in data.montecarlo_runs[:3])
        if uri
    ) or '<p class="muted">No Monte Carlo runs yet — run `python main.py monte-carlo`.</p>'
    return f"""
    <section id="montecarlo">
      <h2>Monte Carlo Analysis (Phase 10)</h2>
      {_table(rows, empty_message="No Monte Carlo runs yet — run `python main.py monte-carlo`.")}
      <h3>Monte Carlo Distribution (most recent runs)</h3>
      {chart_html}
    </section>"""


def _portfolio_section(data: ReportData) -> str:
    rows = [
        {
            "run_id": p.run_id[:8], "scenario": p.scenario, "n_components": p.n_components,
            "best_component": p.best_component_label, "allocation_methods": ", ".join(p.allocations),
        }
        for p in data.portfolio_runs[:50]
    ]
    chart = charts.correlation_heatmap_chart(data.portfolio_runs[0]) if data.portfolio_runs else None
    chart_html = _chart_or_message(chart, alt="strategy correlation", empty_message="No portfolio run with 2+ correlated components yet.")
    return f"""
    <section id="portfolio">
      <h2>Portfolio Analysis (Phase 11)</h2>
      {_table(rows, empty_message="No portfolio runs yet — run `python main.py portfolio`.")}
      <h3>Strategy Correlation (most recent run)</h3>
      {chart_html}
    </section>"""


def _paper_trading_section(data: ReportData) -> str:
    rows = [
        {
            "run_id": s.run_id[:8], "strategy": s.strategy, "symbol": s.symbol, "timeframe": s.timeframe,
            "initial_capital": _fmt(s.initial_capital, ".2f"), "final_equity": _fmt(s.final_equity, ".2f"),
            "total_trades": s.total_trades, "win_rate": _fmt_pct(s.win_rate) if s.win_rate is not None else "n/a",
            "total_pnl": _fmt(s.total_pnl, "+.2f"), "kill_switch": s.kill_switch_triggered,
        }
        for s in data.paper_sessions[:100]
    ]

    sessions_by_run_id = {s.run_id: s for s in data.paper_sessions}
    chart_sections = []
    for run_id, trades in data.paper_trades_by_run.items():
        session = sessions_by_run_id.get(run_id)
        if session is None or not trades:
            continue
        equity = charts.build_equity_series(trades, session.initial_capital)
        pieces = {
            "Equity Curve": charts.equity_curve_chart(equity),
            "Drawdown": charts.drawdown_chart(equity),
            "Monthly Returns": charts.monthly_returns_chart(equity),
            "Rolling Sharpe": charts.rolling_sharpe_chart(trades),
            "Trade Distribution": charts.trade_distribution_chart(trades),
        }
        imgs = "".join(f'<figure><img src="{uri}" class="chart"><figcaption>{label}</figcaption></figure>' for label, uri in pieces.items() if uri)
        if imgs:
            chart_sections.append(
                f'<h3>{html.escape(session.strategy)} / {html.escape(session.symbol)} / '
                f'{html.escape(session.timeframe)} (run {session.run_id[:8]})</h3><div class="chart-grid">{imgs}</div>'
            )

    charts_html = "".join(chart_sections) or '<p class="muted">No paper-trading sessions with closed trades yet — run `python main.py paper-trade`.</p>'
    return f"""
    <section id="paper-trading">
      <h2>Paper Trading (Phase 14)</h2>
      {_table(rows, empty_message="No paper-trading sessions yet — run `python main.py paper-trade`.")}
      {charts_html}
    </section>"""


def _kill_switch_section(data: ReportData) -> str:
    rows = [
        {"event": e.event_type, "reason": e.reason, "equity": _fmt(e.equity, ".2f"), "drawdown": _fmt_pct(e.drawdown) if e.drawdown is not None else "n/a", "created_at": e.created_at}
        for e in data.kill_switch_events
    ]
    return f"""
    <section id="kill-switch">
      <h2>Risk &amp; Kill Switch (Phase 15)</h2>
      {_table(rows, empty_message="No kill switch events recorded — the switch has never tripped.")}
    </section>"""


def render_html_report(data: ReportData) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections = "\n".join([
        _experiments_section(data),
        _walkforward_section(data),
        _montecarlo_section(data),
        _portfolio_section(data),
        _paper_trading_section(data),
        _kill_switch_section(data),
    ])
    nav_items = [
        ("experiments", "Screening"), ("walkforward", "Walk-Forward"), ("montecarlo", "Monte Carlo"),
        ("portfolio", "Portfolio"), ("paper-trading", "Paper Trading"), ("kill-switch", "Kill Switch"),
    ]
    nav = "".join(f'<a href="#{anchor}">{label}</a>' for anchor, label in nav_items)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>quant-trader — results report</title>
<style>
  :root {{
    --bg: #f7f7f8; --fg: #1a1a1a; --muted: #6b7280; --card: #ffffff;
    --border: #e5e7eb; --accent: #2563eb;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg: #111318; --fg: #e5e7eb; --muted: #9ca3af; --card: #1a1d24; --border: #2a2e37; --accent: #60a5fa; }}
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 24px 16px 48px; background: var(--bg); color: var(--fg);
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
  .wrap {{ max-width: 1000px; margin: 0 auto; }}
  h1 {{ font-size: 1.5rem; margin-bottom: 4px; }}
  h2 {{ font-size: 1.2rem; margin-top: 0; }}
  h3 {{ font-size: 1rem; color: var(--muted); }}
  .subtitle {{ color: var(--muted); margin-top: 0; margin-bottom: 20px; }}
  .muted {{ color: var(--muted); font-size: 0.9rem; }}
  .caption {{ color: var(--muted); font-size: 0.8rem; margin-top: -8px; }}
  nav {{ margin-bottom: 20px; display: flex; gap: 12px; flex-wrap: wrap; }}
  nav a {{ color: var(--accent); text-decoration: none; font-size: 0.9rem; }}
  section {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px;
             padding: 20px; margin-bottom: 20px; overflow-x: auto; }}
  .table-wrap {{ overflow-x: auto; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--border); white-space: nowrap; }}
  .chart {{ max-width: 100%; border-radius: 8px; margin: 8px 0; }}
  .chart-grid {{ display: flex; flex-wrap: wrap; gap: 16px; }}
  .chart-grid figure {{ margin: 0; flex: 1 1 380px; }}
  figcaption {{ text-align: center; color: var(--muted); font-size: 0.85rem; }}
  footer {{ color: var(--muted); font-size: 0.8rem; text-align: center; margin-top: 24px; }}
</style>
</head>
<body>
  <div class="wrap">
    <h1>quant-trader — results report</h1>
    <p class="subtitle">Static snapshot of every persisted engine result — open this file directly, no server needed.
    Never a claim of guaranteed profitability (CLAUDE.md Section 44).</p>
    <nav>{nav}</nav>
    {sections}
    <footer>Generated {generated_at} by src.reporting.report_engine (CLAUDE.md Section 30, Phase 16)</footer>
  </div>
</body>
</html>
"""
