"""Phase 17: full integration tests (CLAUDE.md Section 40).

Every previous phase already has its own unit tests against a store it
constructs and tears down in isolation. What none of them prove is that
the *whole* pipeline — Data -> Validation -> Features -> Strategies ->
Backtesting -> Optimization -> Walk-Forward -> Monte Carlo -> Portfolio
-> Paper Trading -> Reporting — actually works end to end, driven the
way a real user drives it: through the CLI, against real files on disk
and one real (if temporary) database, each phase's output feeding the
next. This file also ties together two things that were only ever
tested in isolation from each other: a durable kill switch trip
(Phase 15) actually disabling a broker adapter (Phase 12/13) it was
never exercised against before.

Every fixture that touches the real project's data/raw or reports/
directories cleans up after itself unconditionally (yield + finally),
exactly like the manual CLI smoke tests run throughout this project's
development -- this file just makes that discipline automatic and
repeatable.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

import src.core.db as db_module
from main import build_parser
from src.brokers.base import BrokerAdapter, BrokerConnectionError, EmergencyStopActive, OrderRequest, OrderSide, OrderType
from src.brokers.generic_rest_adapter import GenericRestAdapter
from src.brokers.mt5_adapter import MT5Adapter
from src.core.config import PROJECT_ROOT
from src.core.db import Database
from src.data.csv_loader import raw_csv_path
from src.risk.kill_switch import KillSwitch


def _write_synthetic_csv(symbol: str, timeframe: str, *, n: int = 1500, seed: int = 1) -> None:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2022-01-01", periods=n, freq="1h")
    third = n // 3
    drift = np.concatenate([np.full(third, 0.0008), np.full(third, 0.0), np.full(n - 2 * third, -0.0008)])
    noise_std = np.concatenate([np.full(third, 0.0003), np.full(third, 0.0006), np.full(n - 2 * third, 0.0003)])
    close = 1.10 + np.cumsum(drift + rng.normal(0, 1, size=n) * noise_std)
    high = close + rng.uniform(0.0001, 0.0004, size=n)
    low = close - rng.uniform(0.0001, 0.0004, size=n)
    open_ = close + rng.normal(0, 0.0002, size=n)
    volume = rng.uniform(50, 200, size=n)
    df = pd.DataFrame(
        {"timestamp": ts.astype(str), "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )
    path = raw_csv_path(symbol, timeframe)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


@pytest.fixture()
def real_raw_csv():
    """Real CLI commands read from the fixed, configured data/raw/
    directory with no override flag -- so a genuine CLI-level
    integration test needs a real file there, cleaned up unconditionally."""
    symbols_and_timeframes = [("EURUSD", "H1"), ("GBPUSD", "H1")]
    for seed, (symbol, timeframe) in enumerate(symbols_and_timeframes, start=1):
        _write_synthetic_csv(symbol, timeframe, seed=seed)
    try:
        yield
    finally:
        for symbol, timeframe in symbols_and_timeframes:
            raw_csv_path(symbol, timeframe).unlink(missing_ok=True)


@pytest.fixture()
def isolated_default_db(monkeypatch, tmp_path):
    """Redirects get_default_database() (what every real CLI command
    uses, since none expose a --db override) to a throwaway sqlite file
    for the duration of one test, then restores whatever the process-
    wide singleton was before -- so this is safe regardless of what
    other test modules have or haven't already touched."""
    db_path = tmp_path / "integration.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    original = db_module._default_db
    db_module._default_db = None
    try:
        yield db_path
    finally:
        db_module._default_db = original


@pytest.fixture()
def clean_report_output():
    """python main.py report always writes under the real project's
    reports/ directory (no --output-dir flag) -- clean up whatever it
    produces regardless of test outcome."""
    try:
        yield
    finally:
        (PROJECT_ROOT / "reports" / "report.html").unlink(missing_ok=True)
        exports_dir = PROJECT_ROOT / "reports" / "exports"
        if exports_dir.exists():
            for f in exports_dir.iterdir():
                f.unlink()
            exports_dir.rmdir()


# --- the full CLI-driven pipeline --------------------------------------------


def test_full_cli_pipeline_end_to_end(isolated_default_db, real_raw_csv, clean_report_output, capsys):
    parser = build_parser()

    def run(*argv):
        args = parser.parse_args(list(argv))
        exit_code = args.func(args)
        return exit_code, capsys.readouterr().out

    code, out = run("info")
    assert code == 0
    assert "quant-trader" in out

    code, out = run("validate-data", "--symbol", "EURUSD", "--timeframe", "H1")
    assert code == 0
    assert "DATA QUALITY REPORT" in out

    code, out = run("backtest", "--strategy", "trend_following", "--symbol", "EURUSD", "--timeframe", "H1")
    assert code == 0
    assert "Backtested 1/1" in out

    code, out = run(
        "optimize", "--strategy", "trend_following", "--symbol", "EURUSD", "--timeframe", "H1", "--trials", "3",
    )
    assert code == 0
    assert "best objective" in out

    code, out = run(
        "walk-forward", "--strategy", "trend_following", "--symbol", "EURUSD", "--timeframe", "H1", "--trials", "2",
    )
    assert code == 0
    assert "WALK-FORWARD ANALYSIS REPORT" in out

    code, out = run(
        "monte-carlo", "--strategy", "trend_following", "--symbol", "EURUSD", "--timeframe", "H1", "--simulations", "100",
    )
    assert code == 0
    assert "MONTE CARLO ANALYSIS REPORT" in out

    code, out = run("portfolio", "--symbol", "EURUSD", "--timeframe", "H1", "--min-trades", "5")
    assert code == 0
    assert "PORTFOLIO ANALYSIS REPORT" in out

    code, out = run("paper-trade", "--strategy", "trend_following", "--symbol", "EURUSD", "--timeframe", "H1")
    assert code == 0
    assert "PAPER TRADING SESSION REPORT" in out

    code, out = run(
        "robustness", "--strategy", "trend_following", "--symbol", "EURUSD", "--timeframe", "H1",
        "--trials", "3", "--simulations", "100",
    )
    assert code == 0
    assert "FINAL ROBUSTNESS EVALUATION" in out

    code, out = run("kill-switch")
    assert code == 0
    assert "clear" in out  # this benign synthetic run shouldn't have tripped it

    code, out = run("report")
    assert code == 0

    html_path = PROJECT_ROOT / "reports" / "report.html"
    assert html_path.exists()
    html = html_path.read_text()
    # every phase actually ran and the report reflects it -- none of the
    # "no data yet" placeholders should be present for these sections
    assert "No experiments yet" not in html
    assert "No walk-forward runs yet" not in html
    assert "No Monte Carlo runs yet" not in html
    assert "No portfolio runs yet" not in html
    assert "No paper-trading sessions yet" not in html
    assert "No robustness evaluations yet" not in html
    assert "trend_following" in html
    assert "EURUSD" in html

    # and the underlying persistence agrees with what the CLI reported
    from src.backtest.experiment_store import list_experiments
    from src.execution.trade_store import list_sessions
    from src.montecarlo.run_store import list_runs as list_mc_runs
    from src.portfolio.run_store import list_runs as list_portfolio_runs
    from src.robustness.run_store import list_evaluations
    from src.walkforward.window_store import list_windows

    engines_seen = {e.engine for e in list_experiments()}
    assert {"vectorbt", "backtrader", "optuna"} <= engines_seen
    assert len(list_windows()) >= 2  # one from `walk-forward`, one more from `robustness`'s internal walk-forward
    assert len(list_mc_runs()) == 2  # one from `monte-carlo`, one more from `robustness`'s internal Monte Carlo
    assert len(list_portfolio_runs()) == 1
    assert len(list_sessions()) == 1
    assert len(list_evaluations()) == 1


def test_cli_pipeline_reports_missing_data_honestly_without_any_csv(isolated_default_db, clean_report_output, capsys):
    """The mirror image of the happy path: with no raw data placed at
    all, every phase must say so and refuse to fabricate a result --
    and the final report must say "no data yet" for all of them, not
    silently produce empty-looking charts."""
    parser = build_parser()

    def run(*argv):
        args = parser.parse_args(list(argv))
        return args.func(args), capsys.readouterr().out

    code, out = run("backtest", "--strategy", "trend_following", "--symbol", "EURUSD", "--timeframe", "H1")
    assert code == 0
    assert "Skipped" in out

    code, out = run("report")
    assert code == 0
    html = (PROJECT_ROOT / "reports" / "report.html").read_text()
    assert "No experiments yet" in html
    assert "No paper-trading sessions yet" in html
    assert "No robustness evaluations yet" in html
    assert "NO ROBUST STRATEGY FOUND" in html


# --- broker adapter interface conformance ------------------------------------


@pytest.mark.parametrize("adapter_cls", [MT5Adapter, GenericRestAdapter])
def test_broker_adapters_implement_the_full_abstract_interface(adapter_cls):
    """Section 21's whole point: every adapter honors the same contract.
    A missing method would make instantiation itself fail (Python
    enforces this for ABCs) -- this test makes that guarantee explicit
    and named, rather than an incidental side effect of other tests
    happening to construct one."""
    assert not inspect.isabstract(adapter_cls)
    for method_name in BrokerAdapter.__abstractmethods__:
        assert callable(getattr(adapter_cls, method_name))


# --- kill switch x broker adapter cross-wiring (never tested together before) --


def _order():
    return OrderRequest(symbol="EURUSD", side=OrderSide.BUY, order_type=OrderType.MARKET, volume=0.1)


@pytest.fixture()
def db():
    database = Database(url="sqlite:///:memory:")
    database.init_db()
    return database


def test_kill_switch_trip_blocks_a_generic_rest_adapter(db):
    adapter = GenericRestAdapter(api_base_url="http://fake-broker.test")
    switch = KillSwitch(db=db)
    switch.trip("integration test breach", brokers=[adapter])
    with pytest.raises(EmergencyStopActive):
        adapter.place_order(_order())


def test_kill_switch_trip_blocks_an_mt5_adapter(db):
    adapter = MT5Adapter()
    switch = KillSwitch(db=db)
    switch.trip("integration test breach", brokers=[adapter])
    with pytest.raises(EmergencyStopActive):
        adapter.place_order(_order())


def test_kill_switch_reset_re_enables_broker_after_being_tripped(db):
    adapter = GenericRestAdapter(api_base_url="http://fake-broker.test")
    switch = KillSwitch(db=db)
    switch.trip("integration test breach", brokers=[adapter])
    assert adapter._halted is True

    switch.reset("reviewed, resuming", brokers=[adapter])
    assert adapter._halted is False
    # still raises BrokerConnectionError (never connected) -- but crucially
    # NOT EmergencyStopActive anymore, proving the halt itself was lifted
    with pytest.raises(BrokerConnectionError):
        adapter.place_order(_order())
