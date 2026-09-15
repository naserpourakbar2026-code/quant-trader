"""CLI entry point for the quant-trader framework.

Commands are added phase by phase (see CLAUDE.md Section 40). Commands not
yet implemented say so explicitly rather than silently doing nothing or
faking a result.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime

from src.backtest.backtrader_engine import compare_with_screening, run_validation as run_backtrader_validation
from src.backtest.vectorbt_engine import run_screening
from src.core.config import PROJECT_ROOT, is_live_trading_enabled, load_brokers, load_settings, load_strategies
from src.core.logging import configure_logging, get_system_logger
from src.data.csv_loader import load_csv
from src.data.ingestion import run_download
from src.data.validation import run_validation
from src.execution.paper_trading import run_paper_trading_session
from src.montecarlo.mc_engine import run_monte_carlo
from src.optimization.optuna_engine import assess_parameter_stability, run_optimization
from src.portfolio.portfolio_engine import run_portfolio_analysis
from src.reporting.report_engine import run_report
from src.robustness.evaluation import run_robustness_evaluation
from src.risk.kill_switch import KillSwitch
from src.walkforward.wfa_engine import run_walk_forward

# command -> (implemented, scheduled phase)
COMMAND_PHASES: dict[str, tuple[bool, int]] = {
    "live": (False, 14),
}


def cmd_info(_args: argparse.Namespace) -> int:
    """Implemented now: validates the environment and prints a summary."""
    settings = load_settings()
    strategies = load_strategies()
    brokers = load_brokers()

    print("quant-trader — environment info")
    print(f"  initial capital       : {settings.account.initial_capital} {settings.account.currency}")
    print(f"  risk per trade         : {settings.risk.risk_per_trade:.4%}")
    print(f"  symbols                : {', '.join(settings.data.symbols)}")
    print(f"  timeframes             : {', '.join(settings.data.timeframes)}")
    print(f"  data source            : {settings.data.source}")
    print(f"  registered strategies  : {len(strategies.strategies)}")
    print(f"  active broker          : {brokers.active_broker or '(none configured)'}")
    print(f"  live trading enabled   : {is_live_trading_enabled()} "
          f"(requires LIVE_TRADING=true AND LIVE_CONFIRMATION=true)")
    print("  kill switch            : see `python main.py kill-switch` for status/history/reset")
    return 0


def cmd_download_data(args: argparse.Namespace) -> int:
    """Ingest historical data (Phase 2): CSV files from data/raw/, or MT5
    (Windows-only, requires --start/--end and a running terminal).
    """
    symbols = [args.symbol] if args.symbol else None
    timeframes = [args.timeframe] if args.timeframe else None
    start = datetime.fromisoformat(args.start) if args.start else None
    end = datetime.fromisoformat(args.end) if args.end else None

    results = run_download(symbols=symbols, timeframes=timeframes, start=start, end=end, source=args.source)
    ok = [r for r in results if r.status == "ok"]
    missing = [r for r in results if r.status == "missing"]
    errors = [r for r in results if r.status == "error"]

    print(f"Ingested {len(ok)}/{len(results)} symbol/timeframe combinations.")
    for r in ok:
        print(f"  OK      {r.symbol:8s} {r.timeframe:4s}  {r.rows} rows")
    for r in missing:
        print(f"  MISSING {r.symbol:8s} {r.timeframe:4s}  {r.message}")
    for r in errors:
        print(f"  ERROR   {r.symbol:8s} {r.timeframe:4s}  {r.message}")

    if missing and not ok and not errors:
        settings = load_settings()
        print(
            f"\nNo raw data found yet. For source={settings.data.source!r}, place CSV files under "
            f"{settings.paths.data_raw}/<SYMBOL>_<TIMEFRAME>.csv with at least "
            "timestamp,open,high,low,close,volume columns."
        )

    return 0 if not errors else 1


def cmd_validate_data(args: argparse.Namespace) -> int:
    """Run the Phase 3 data-quality checks and print a report per
    symbol/timeframe. Detection only — never repairs anything found."""
    symbols = [args.symbol] if args.symbol else None
    timeframes = [args.timeframe] if args.timeframe else None

    reports, skipped = run_validation(symbols=symbols, timeframes=timeframes)

    for report in reports:
        print(report.to_text())
        print()

    if skipped:
        print(f"Skipped {len(skipped)} symbol/timeframe combination(s) with no raw data:")
        for skip in skipped:
            print(f"  {skip.symbol:8s} {skip.timeframe:4s}  {skip.reason}")

    if not reports and not skipped:
        print("Nothing to validate.")

    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    """Screen with vectorbt (Phase 6) then validate with Backtrader
    (Phase 7) for every enabled strategy x symbol x timeframe combination
    from config/strategies.yaml (optionally filtered), printing both
    engines' key metrics side by side. Persists an Experiment row per
    engine per combination."""
    strategies_cfg = load_strategies()
    settings = load_settings()
    scenario = args.scenario or settings.execution.default_scenario

    combos: list[tuple[str, str, str]] = []
    for name, definition in strategies_cfg.strategies.items():
        if not definition.enabled:
            continue
        if args.strategy and name != args.strategy:
            continue
        symbols = [args.symbol] if args.symbol else definition.symbols
        timeframes = [args.timeframe] if args.timeframe else definition.timeframes
        combos.extend((name, symbol, timeframe) for symbol in symbols for timeframe in timeframes)

    if not combos:
        print("No enabled strategy/symbol/timeframe combination matched.")
        return 0

    logger = get_system_logger()
    ok, missing, errors = [], [], []
    for family, symbol, timeframe in combos:
        try:
            raw_df = load_csv(symbol, timeframe)
        except FileNotFoundError as exc:
            missing.append((family, symbol, timeframe, str(exc)))
            continue
        try:
            screening = run_screening(family, symbol, timeframe, raw_df, scenario=scenario)
            validation = run_backtrader_validation(family, symbol, timeframe, raw_df, scenario=scenario)
        except Exception as exc:  # keep going across the rest of the batch
            logger.warning(f"Backtest failed for {family}/{symbol}/{timeframe}: {exc}")
            errors.append((family, symbol, timeframe, str(exc)))
            continue
        ok.append((family, symbol, timeframe, screening, validation))

    print(f"Backtested {len(ok)}/{len(combos)} combination(s) (scenario={scenario!r}).")
    for family, symbol, timeframe, screening, validation in ok:
        comparison = compare_with_screening(screening.metrics, validation.metrics)
        print(f"\n{family} {symbol} {timeframe}")
        print(
            f"  vectorbt  : return={screening.metrics['total_return']:.4%} "
            f"sharpe={screening.metrics['sharpe_ratio']} trades={screening.metrics['trade_count']}"
        )
        print(
            f"  backtrader: return={validation.metrics['total_return']:.4%} "
            f"sharpe={validation.metrics['sharpe_ratio']} trades={validation.metrics['trade_count']} "
            f"rejected={validation.metrics['rejected_orders']} margin={validation.metrics['margin_orders']}"
        )
        return_delta = comparison["total_return"]["delta"]
        if return_delta is not None:
            print(f"  return delta (backtrader - vectorbt): {return_delta:+.4%}")

    if missing:
        print(f"\nSkipped {len(missing)} combination(s) with no raw data:")
        for family, symbol, timeframe, message in missing:
            print(f"  {family:24s} {symbol:8s} {timeframe:4s}  {message}")

    if errors:
        print(f"\n{len(errors)} combination(s) failed:")
        for family, symbol, timeframe, message in errors:
            print(f"  {family:24s} {symbol:8s} {timeframe:4s}  {message}")

    return 0 if not errors else 1


def cmd_optimize(args: argparse.Namespace) -> int:
    """Optuna search over one strategy's configured optimization_space
    for one symbol/timeframe (Phase 8), then a post-hoc parameter
    stability check (Section 16). Deliberately scoped to a single,
    explicit combination rather than a broad default sweep — a search
    with dozens of trials per combination is not something to run
    accidentally across every symbol/timeframe."""
    strategies_cfg = load_strategies()
    definition = strategies_cfg.strategies.get(args.strategy)
    if definition is None:
        print(f"Unknown strategy {args.strategy!r}; expected one of {sorted(strategies_cfg.strategies)}", file=sys.stderr)
        return 1
    if not definition.optimization_space:
        print(f"Strategy {args.strategy!r} has no optimization_space configured in strategies.yaml.", file=sys.stderr)
        return 1

    try:
        raw_df = load_csv(args.symbol, args.timeframe)
    except FileNotFoundError as exc:
        print(str(exc))
        return 0

    result = run_optimization(
        args.strategy,
        args.symbol,
        args.timeframe,
        raw_df,
        definition.optimization_space,
        n_trials=args.trials,
        scenario=args.scenario,
        seed=args.seed,
    )
    print(f"Optimized {args.strategy} {args.symbol} {args.timeframe} over {result.n_trials} trials "
          f"(scenario={result.scenario!r}).")
    print(f"  best params    : {result.best_params}")
    print(f"  best objective : {result.best_objective:.4f}")
    print(
        f"  metrics        : return={result.best_metrics['total_return']:.4%} "
        f"sharpe={result.best_metrics['sharpe_ratio']} trades={result.best_metrics['trade_count']}"
    )

    stability = assess_parameter_stability(
        args.strategy, args.symbol, args.timeframe, raw_df, result.best_params,
        definition.optimization_space, scenario=result.scenario,
    )
    print(
        f"  stability score: {stability.score:.2f} "
        "(0=isolated spike, 1=stable region — CLAUDE.md Section 16; never trust an isolated spike alone)"
    )

    return 0


def cmd_walk_forward(args: argparse.Namespace) -> int:
    """Rolling walk-forward analysis (Phase 9, CLAUDE.md Section 15):
    optimize on training only, validate, then test on out-of-sample data
    the optimizer never saw, moving the window forward and repeating.
    Deliberately scoped to a single, explicit combination — this runs a
    full Optuna search per window, so it is strictly more expensive than
    `optimize` and should never fire across every symbol/timeframe by
    accident."""
    strategies_cfg = load_strategies()
    definition = strategies_cfg.strategies.get(args.strategy)
    if definition is None:
        print(f"Unknown strategy {args.strategy!r}; expected one of {sorted(strategies_cfg.strategies)}", file=sys.stderr)
        return 1
    if not definition.optimization_space:
        print(f"Strategy {args.strategy!r} has no optimization_space configured in strategies.yaml.", file=sys.stderr)
        return 1

    try:
        raw_df = load_csv(args.symbol, args.timeframe)
    except FileNotFoundError as exc:
        print(str(exc))
        return 0

    report = run_walk_forward(
        args.strategy,
        args.symbol,
        args.timeframe,
        raw_df,
        definition.optimization_space,
        train_pct=args.train_pct,
        validation_pct=args.validation_pct,
        oos_pct=args.oos_pct,
        window_bars=args.window_bars,
        step_bars=args.step_bars,
        n_trials=args.trials,
        scenario=args.scenario,
        seed=args.seed,
    )
    print(report.to_text())
    return 0


def cmd_monte_carlo(args: argparse.Namespace) -> int:
    """Bootstrap-resample a strategy's observed trade returns thousands
    of times to estimate an outcome distribution (Phase 10, CLAUDE.md
    Section 17). Deliberately scoped to a single, explicit combination
    with fixed parameters — this evaluates a strategy already chosen
    elsewhere (e.g. via `optimize`/`walk-forward`), it does not search
    for one."""
    strategies_cfg = load_strategies()
    definition = strategies_cfg.strategies.get(args.strategy)
    if definition is None:
        print(f"Unknown strategy {args.strategy!r}; expected one of {sorted(strategies_cfg.strategies)}", file=sys.stderr)
        return 1

    try:
        raw_df = load_csv(args.symbol, args.timeframe)
    except FileNotFoundError as exc:
        print(str(exc))
        return 0

    try:
        result = run_monte_carlo(
            args.strategy,
            args.symbol,
            args.timeframe,
            raw_df,
            scenario=args.scenario,
            n_simulations=args.simulations,
            seed=args.seed,
        )
    except ValueError as exc:
        print(str(exc))
        return 0

    print(result.to_text())
    return 0


def cmd_portfolio(args: argparse.Namespace) -> int:
    """Screen every enabled strategy x symbol x timeframe combination
    (optionally filtered), keep the statistically-usable top survivors,
    and test whether combining them (equal-weight / inverse-volatility)
    improves Sharpe, Sortino, drawdown, and return consistency versus the
    single best combination alone (Phase 11, CLAUDE.md Section 20)."""
    strategies_cfg = load_strategies()
    settings = load_settings()
    scenario = args.scenario or settings.execution.default_scenario

    combos: list[tuple[str, str, str]] = []
    for name, definition in strategies_cfg.strategies.items():
        if not definition.enabled:
            continue
        if args.strategy and name != args.strategy:
            continue
        symbols = [args.symbol] if args.symbol else definition.symbols
        timeframes = [args.timeframe] if args.timeframe else definition.timeframes
        combos.extend((name, symbol, timeframe) for symbol in symbols for timeframe in timeframes)

    if not combos:
        print("No enabled strategy/symbol/timeframe combination matched.")
        return 0

    components_data = []
    missing = []
    for family, symbol, timeframe in combos:
        try:
            components_data.append((family, symbol, timeframe, load_csv(symbol, timeframe)))
        except FileNotFoundError as exc:
            missing.append((family, symbol, timeframe, str(exc)))

    if missing:
        print(f"Skipped {len(missing)}/{len(combos)} combination(s) with no raw data:")
        for family, symbol, timeframe, message in missing:
            print(f"  {family:24s} {symbol:8s} {timeframe:4s}  {message}")

    if not components_data:
        print("No raw data available for any matched combination.")
        return 0

    try:
        result = run_portfolio_analysis(
            components_data, scenario=scenario, top_n=args.top_n, min_trades=args.min_trades,
        )
    except ValueError as exc:
        print(str(exc))
        return 0

    print(f"\nScreened {len(components_data)} combination(s).")
    print(result.to_text())
    return 0


def cmd_paper_trade(args: argparse.Namespace) -> int:
    """Replay historical candles bar-by-bar through the full Market Data
    -> Strategy -> Risk Engine -> Virtual Order -> Execution Simulator
    -> Position Manager -> PnL pipeline (Phase 14, CLAUDE.md Section 26).
    There is no live feed in this sandbox; this is always a simulated
    session over historical data, never a real order."""
    strategies_cfg = load_strategies()
    definition = strategies_cfg.strategies.get(args.strategy)
    if definition is None:
        print(f"Unknown strategy {args.strategy!r}; expected one of {sorted(strategies_cfg.strategies)}", file=sys.stderr)
        return 1

    try:
        raw_df = load_csv(args.symbol, args.timeframe)
    except FileNotFoundError as exc:
        print(str(exc))
        return 0

    try:
        result = run_paper_trading_session(
            args.strategy, args.symbol, args.timeframe, raw_df, scenario=args.scenario, initial_capital=args.capital,
        )
    except ValueError as exc:
        print(str(exc))
        return 0

    print(result.to_text())
    return 0


def cmd_kill_switch(args: argparse.Namespace) -> int:
    """Show the durable kill switch's current status/history, or record
    a reset (Phase 15, CLAUDE.md Sections 7, 27). A reset is always an
    explicit human action with a required note — never automatic."""
    switch = KillSwitch()
    if args.reset:
        switch.reset(args.reset)
        print(f"Kill switch reset recorded: {args.reset!r}")

    triggered = switch.is_triggered()
    print(f"Kill switch status: {'TRIGGERED' if triggered else 'clear'}")

    events = switch.history(limit=args.history)
    if events:
        print(f"\nLast {len(events)} event(s) (newest first):")
        for event in events:
            extra = ""
            if event.drawdown is not None:
                extra = f" (equity={event.equity:.2f}, drawdown={event.drawdown:.2%})"
            print(f"  {event.created_at} {event.event_type:5s} {event.reason}{extra}")
    return 0


def cmd_report(_args: argparse.Namespace) -> int:
    """Generate the self-contained HTML report plus CSV/JSON exports
    from every persisted engine's results so far (Phase 16, CLAUDE.md
    Section 30). Opening reports/report.html in a browser is enough to
    view the outcome — no server, no notebook, no further command."""
    result = run_report()
    for path in [result.html_path, *result.csv_paths, result.json_path]:
        print(f"Wrote {path.relative_to(PROJECT_ROOT)}")
    print("\nOpen the HTML file directly in a browser to view the results.")
    return 0


def cmd_robustness(args: argparse.Namespace) -> int:
    """Final robustness evaluation (Phase 18, CLAUDE.md Sections 18,
    31-36, 43-44): walk-forward + Monte Carlo + a transaction-cost
    stress test + a capital simulation across every configured risk
    level, combined into one 0-100 score and a PASS/FAIL/COST FRAGILE/
    OVERFIT/INSUFFICIENT DATA/NOT ROBUST verdict. Never the strategy
    with the highest backtest profit alone — the strongest evidenced
    combination of profitability, risk-adjusted return, statistical
    reliability, out-of-sample performance, parameter stability, and
    cost/randomization robustness. Deliberately scoped to one explicit
    combination — it runs a full walk-forward search internally, so it
    is at least as expensive as `walk-forward` alone."""
    strategies_cfg = load_strategies()
    definition = strategies_cfg.strategies.get(args.strategy)
    if definition is None:
        print(f"Unknown strategy {args.strategy!r}; expected one of {sorted(strategies_cfg.strategies)}", file=sys.stderr)
        return 1
    if not definition.optimization_space:
        print(f"Strategy {args.strategy!r} has no optimization_space configured in strategies.yaml.", file=sys.stderr)
        return 1

    try:
        raw_df = load_csv(args.symbol, args.timeframe)
    except FileNotFoundError as exc:
        print(str(exc))
        return 0

    result = run_robustness_evaluation(
        args.strategy, args.symbol, args.timeframe, raw_df, definition.optimization_space,
        n_trials=args.trials, mc_simulations=args.simulations, scenario=args.scenario, seed=args.seed,
    )
    print(result.to_text())
    return 0


def _not_implemented(name: str, phase: int) -> int:
    logger = get_system_logger()
    message = (
        f"`{name}` is not implemented yet — scheduled for Phase {phase} "
        "(see CLAUDE.md Section 40). Refusing to run rather than fake a result."
    )
    logger.warning(message)
    print(message, file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quant-trader", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("info", help="Validate config and print environment summary").set_defaults(func=cmd_info)

    download_parser = subparsers.add_parser(
        "download-data", help="Ingest historical data into data/processed/ (Phase 2)"
    )
    download_parser.add_argument("--symbol", help="Limit to one symbol (default: all configured symbols)")
    download_parser.add_argument("--timeframe", help="Limit to one timeframe (default: all configured timeframes)")
    download_parser.add_argument("--start", help="ISO date, required for mt5 source (e.g. 2024-01-01)")
    download_parser.add_argument("--end", help="ISO date, required for mt5 source")
    download_parser.add_argument(
        "--source", choices=["csv", "mt5", "twelvedata"],
        help="Override data.source from settings.yaml for this run",
    )
    download_parser.set_defaults(func=cmd_download_data)

    validate_parser = subparsers.add_parser(
        "validate-data", help="Run data-quality checks and print a report (Phase 3)"
    )
    validate_parser.add_argument("--symbol", help="Limit to one symbol (default: all configured symbols)")
    validate_parser.add_argument("--timeframe", help="Limit to one timeframe (default: all configured timeframes)")
    validate_parser.set_defaults(func=cmd_validate_data)

    backtest_parser = subparsers.add_parser(
        "backtest", help="vectorbt screen + Backtrader validation for enabled strategies (Phases 6-7)"
    )
    backtest_parser.add_argument("--strategy", help="Limit to one strategy family (default: all enabled)")
    backtest_parser.add_argument("--symbol", help="Limit to one symbol (default: each strategy's configured list)")
    backtest_parser.add_argument(
        "--timeframe", help="Limit to one timeframe (default: each strategy's configured list)"
    )
    backtest_parser.add_argument("--scenario", help="optimistic|realistic|stress (default: execution.default_scenario)")
    backtest_parser.set_defaults(func=cmd_backtest)

    optimize_parser = subparsers.add_parser(
        "optimize", help="Optuna search + parameter stability check for one strategy/symbol/timeframe (Phase 8)"
    )
    optimize_parser.add_argument("--strategy", required=True, help="Strategy family (see config/strategies.yaml)")
    optimize_parser.add_argument("--symbol", required=True, help="Symbol to optimize against")
    optimize_parser.add_argument("--timeframe", required=True, help="Timeframe to optimize against")
    optimize_parser.add_argument("--trials", type=int, help="Number of Optuna trials (default: optimization.n_trials)")
    optimize_parser.add_argument("--scenario", help="optimistic|realistic|stress (default: execution.default_scenario)")
    optimize_parser.add_argument("--seed", type=int, help="Random seed override (default: optimization.random_seed)")
    optimize_parser.set_defaults(func=cmd_optimize)

    wf_parser = subparsers.add_parser(
        "walk-forward", help="Rolling train/validate/OOS walk-forward analysis (Phase 9)"
    )
    wf_parser.add_argument("--strategy", required=True, help="Strategy family (see config/strategies.yaml)")
    wf_parser.add_argument("--symbol", required=True, help="Symbol to analyze")
    wf_parser.add_argument("--timeframe", required=True, help="Timeframe to analyze")
    wf_parser.add_argument("--trials", type=int, help="Optuna trials per window (default: optimization.n_trials)")
    wf_parser.add_argument("--train-pct", type=float, default=0.6, dest="train_pct")
    wf_parser.add_argument("--validation-pct", type=float, default=0.2, dest="validation_pct")
    wf_parser.add_argument("--oos-pct", type=float, default=0.2, dest="oos_pct")
    wf_parser.add_argument(
        "--window-bars", type=int, dest="window_bars", help="Bars per window (default: the whole series, one window)"
    )
    wf_parser.add_argument("--step-bars", type=int, dest="step_bars", help="Bars to advance per window (default: OOS length)")
    wf_parser.add_argument("--scenario", help="optimistic|realistic|stress (default: execution.default_scenario)")
    wf_parser.add_argument("--seed", type=int, help="Random seed override (default: optimization.random_seed)")
    wf_parser.set_defaults(func=cmd_walk_forward)

    mc_parser = subparsers.add_parser(
        "monte-carlo", help="Bootstrap-resample observed trade returns to estimate an outcome distribution (Phase 10)"
    )
    mc_parser.add_argument("--strategy", required=True, help="Strategy family (see config/strategies.yaml)")
    mc_parser.add_argument("--symbol", required=True, help="Symbol to analyze")
    mc_parser.add_argument("--timeframe", required=True, help="Timeframe to analyze")
    mc_parser.add_argument("--simulations", type=int, help="Number of simulations (default: montecarlo.n_simulations)")
    mc_parser.add_argument("--scenario", help="optimistic|realistic|stress (default: execution.default_scenario)")
    mc_parser.add_argument("--seed", type=int, help="Random seed override (default: montecarlo.random_seed)")
    mc_parser.set_defaults(func=cmd_monte_carlo)

    portfolio_parser = subparsers.add_parser(
        "portfolio", help="Screen combinations + test basic allocation across them (Phase 11)"
    )
    portfolio_parser.add_argument("--strategy", help="Limit to one strategy family (default: all enabled)")
    portfolio_parser.add_argument("--symbol", help="Limit to one symbol (default: each strategy's configured list)")
    portfolio_parser.add_argument(
        "--timeframe", help="Limit to one timeframe (default: each strategy's configured list)"
    )
    portfolio_parser.add_argument("--scenario", help="optimistic|realistic|stress (default: execution.default_scenario)")
    portfolio_parser.add_argument(
        "--top-n", type=int, dest="top_n", help="Max combinations entering the allocation step (default: portfolio.top_n)"
    )
    portfolio_parser.add_argument(
        "--min-trades", type=int, dest="min_trades", help="Exclude combinations with fewer trades (default: portfolio.min_trades)"
    )
    portfolio_parser.set_defaults(func=cmd_portfolio)

    paper_trade_parser = subparsers.add_parser(
        "paper-trade", help="Simulate a strategy bar-by-bar over historical data (Phase 14)"
    )
    paper_trade_parser.add_argument("--strategy", required=True, help="Strategy family (see config/strategies.yaml)")
    paper_trade_parser.add_argument("--symbol", required=True, help="Symbol to simulate")
    paper_trade_parser.add_argument("--timeframe", required=True, help="Timeframe to simulate")
    paper_trade_parser.add_argument("--scenario", help="optimistic|realistic|stress (default: execution.default_scenario)")
    paper_trade_parser.add_argument("--capital", type=float, help="Initial capital override (default: account.initial_capital)")
    paper_trade_parser.set_defaults(func=cmd_paper_trade)

    kill_switch_parser = subparsers.add_parser(
        "kill-switch", help="Show/reset the durable risk kill switch (Phase 15)"
    )
    kill_switch_parser.add_argument(
        "--reset", help="Record a reset with this note (a human decision — never automatic)"
    )
    kill_switch_parser.add_argument("--history", type=int, default=10, help="Number of past events to show (default: 10)")
    kill_switch_parser.set_defaults(func=cmd_kill_switch)

    report_parser = subparsers.add_parser(
        "report", help="Generate the self-contained HTML report + CSV/JSON exports (Phase 16)"
    )
    report_parser.set_defaults(func=cmd_report)

    robustness_parser = subparsers.add_parser(
        "robustness", help="Final robustness evaluation: score + PASS/FAIL/... verdict (Phase 18)"
    )
    robustness_parser.add_argument("--strategy", required=True, help="Strategy family (see config/strategies.yaml)")
    robustness_parser.add_argument("--symbol", required=True, help="Symbol to evaluate")
    robustness_parser.add_argument("--timeframe", required=True, help="Timeframe to evaluate")
    robustness_parser.add_argument("--trials", type=int, help="Optuna trials per walk-forward window (default: optimization.n_trials)")
    robustness_parser.add_argument("--simulations", type=int, help="Monte Carlo simulations (default: montecarlo.n_simulations)")
    robustness_parser.add_argument("--scenario", help="optimistic|realistic|stress (default: execution.default_scenario)")
    robustness_parser.add_argument("--seed", type=int, help="Random seed override")
    robustness_parser.set_defaults(func=cmd_robustness)

    for name, (_implemented, phase) in COMMAND_PHASES.items():
        sub = subparsers.add_parser(name, help=f"(pending — Phase {phase})")
        sub.set_defaults(func=lambda args, n=name, p=phase: _not_implemented(n, p))

    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
