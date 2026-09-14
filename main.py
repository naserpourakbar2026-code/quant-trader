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
from src.core.config import is_live_trading_enabled, load_brokers, load_settings, load_strategies
from src.core.logging import configure_logging, get_system_logger
from src.data.csv_loader import load_csv
from src.data.ingestion import run_download
from src.data.validation import run_validation

# command -> (implemented, scheduled phase)
COMMAND_PHASES: dict[str, tuple[bool, int]] = {
    "optimize": (False, 8),
    "walk-forward": (False, 9),
    "monte-carlo": (False, 10),
    "portfolio": (False, 11),
    "paper-trade": (False, 14),
    "live": (False, 14),
    "report": (False, 16),
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
    return 0


def cmd_download_data(args: argparse.Namespace) -> int:
    """Ingest historical data (Phase 2): CSV files from data/raw/, or MT5
    (Windows-only, requires --start/--end and a running terminal).
    """
    symbols = [args.symbol] if args.symbol else None
    timeframes = [args.timeframe] if args.timeframe else None
    start = datetime.fromisoformat(args.start) if args.start else None
    end = datetime.fromisoformat(args.end) if args.end else None

    results = run_download(symbols=symbols, timeframes=timeframes, start=start, end=end)
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
