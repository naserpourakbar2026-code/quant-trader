"""CLI entry point for the quant-trader framework.

Commands are added phase by phase (see CLAUDE.md Section 40). Commands not
yet implemented say so explicitly rather than silently doing nothing or
faking a result.
"""
from __future__ import annotations

import argparse
import sys

from src.core.config import is_live_trading_enabled, load_brokers, load_settings, load_strategies
from src.core.logging import configure_logging, get_system_logger

# command -> (implemented, scheduled phase)
COMMAND_PHASES: dict[str, tuple[bool, int]] = {
    "download-data": (False, 2),
    "validate-data": (False, 3),
    "backtest": (False, 7),
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
