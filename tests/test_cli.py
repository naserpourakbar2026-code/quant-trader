import pytest

import main
from main import (
    build_parser,
    cmd_backtest,
    cmd_download_data,
    cmd_info,
    cmd_kill_switch,
    cmd_monte_carlo,
    cmd_optimize,
    cmd_paper_trade,
    cmd_portfolio,
    cmd_report,
    cmd_robustness,
    cmd_validate_data,
    cmd_walk_forward,
)


def test_info_command_runs_and_returns_zero(capsys):
    parser = build_parser()
    args = parser.parse_args(["info"])
    assert args.func is cmd_info
    exit_code = args.func(args)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "quant-trader" in out


def test_pending_commands_return_nonzero_and_do_not_pretend_to_run():
    parser = build_parser()
    for name in ["live"]:
        args = parser.parse_args([name])
        exit_code = args.func(args)
        assert exit_code != 0


def test_backtest_command_reports_missing_when_no_raw_file(capsys):
    """No CSV placed yet under data/raw for this symbol/timeframe -> the
    combination is skipped (reported, not fabricated), exit code 0."""
    parser = build_parser()
    args = parser.parse_args(["backtest", "--strategy", "trend_following", "--symbol", "EURUSD", "--timeframe", "H1"])
    assert args.func is cmd_backtest
    exit_code = args.func(args)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Skipped" in out
    assert "Backtested 0/1" in out


def test_backtest_command_no_matching_strategy_reports_and_exits_zero(capsys):
    parser = build_parser()
    args = parser.parse_args(["backtest", "--strategy", "not_a_real_strategy"])
    exit_code = args.func(args)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "No enabled strategy" in out


def test_download_data_command_reports_missing_when_no_raw_file(capsys):
    """No CSV placed yet under data/raw for this symbol/timeframe -> reported
    as missing, not fabricated, exit code 0 (missing is not an error)."""
    parser = build_parser()
    args = parser.parse_args(["download-data", "--symbol", "EURUSD", "--timeframe", "H1"])
    assert args.func is cmd_download_data
    exit_code = args.func(args)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "MISSING" in out


def test_validate_data_command_reports_skip_when_no_raw_file(capsys):
    """No CSV placed yet -> reported as skipped, exit code 0."""
    parser = build_parser()
    args = parser.parse_args(["validate-data", "--symbol", "EURUSD", "--timeframe", "H1"])
    assert args.func is cmd_validate_data
    exit_code = args.func(args)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Skipped" in out


def test_optimize_command_reports_missing_when_no_raw_file(capsys):
    parser = build_parser()
    args = parser.parse_args(["optimize", "--strategy", "trend_following", "--symbol", "EURUSD", "--timeframe", "H1"])
    assert args.func is cmd_optimize
    exit_code = args.func(args)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "No raw CSV found" in out


def test_optimize_command_rejects_unknown_strategy(capsys):
    parser = build_parser()
    args = parser.parse_args(["optimize", "--strategy", "not_a_real_strategy", "--symbol", "EURUSD", "--timeframe", "H1"])
    exit_code = args.func(args)
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "Unknown strategy" in err


def test_optimize_command_requires_strategy_symbol_timeframe():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["optimize", "--strategy", "trend_following"])


def test_walk_forward_command_reports_missing_when_no_raw_file(capsys):
    parser = build_parser()
    args = parser.parse_args(
        ["walk-forward", "--strategy", "trend_following", "--symbol", "EURUSD", "--timeframe", "H1"]
    )
    assert args.func is cmd_walk_forward
    exit_code = args.func(args)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "No raw CSV found" in out


def test_walk_forward_command_rejects_unknown_strategy(capsys):
    parser = build_parser()
    args = parser.parse_args(
        ["walk-forward", "--strategy", "not_a_real_strategy", "--symbol", "EURUSD", "--timeframe", "H1"]
    )
    exit_code = args.func(args)
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "Unknown strategy" in err


def test_walk_forward_command_requires_strategy_symbol_timeframe():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["walk-forward", "--strategy", "trend_following"])


def test_monte_carlo_command_reports_missing_when_no_raw_file(capsys):
    parser = build_parser()
    args = parser.parse_args(
        ["monte-carlo", "--strategy", "trend_following", "--symbol", "EURUSD", "--timeframe", "H1"]
    )
    assert args.func is cmd_monte_carlo
    exit_code = args.func(args)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "No raw CSV found" in out


def test_monte_carlo_command_rejects_unknown_strategy(capsys):
    parser = build_parser()
    args = parser.parse_args(
        ["monte-carlo", "--strategy", "not_a_real_strategy", "--symbol", "EURUSD", "--timeframe", "H1"]
    )
    exit_code = args.func(args)
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "Unknown strategy" in err


def test_monte_carlo_command_requires_strategy_symbol_timeframe():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["monte-carlo", "--strategy", "trend_following"])


def test_portfolio_command_reports_missing_when_no_raw_file(capsys):
    """No CSV placed yet under data/raw for this symbol/timeframe -> the
    combination is skipped (reported, not fabricated), exit code 0."""
    parser = build_parser()
    args = parser.parse_args(
        ["portfolio", "--strategy", "trend_following", "--symbol", "EURUSD", "--timeframe", "H1"]
    )
    assert args.func is cmd_portfolio
    exit_code = args.func(args)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Skipped" in out
    assert "No raw data available" in out


def test_portfolio_command_no_matching_strategy_reports_and_exits_zero(capsys):
    parser = build_parser()
    args = parser.parse_args(["portfolio", "--strategy", "not_a_real_strategy"])
    exit_code = args.func(args)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "No enabled strategy" in out


def test_paper_trade_command_reports_missing_when_no_raw_file(capsys):
    parser = build_parser()
    args = parser.parse_args(
        ["paper-trade", "--strategy", "trend_following", "--symbol", "EURUSD", "--timeframe", "H1"]
    )
    assert args.func is cmd_paper_trade
    exit_code = args.func(args)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "No raw CSV found" in out


def test_paper_trade_command_rejects_unknown_strategy(capsys):
    parser = build_parser()
    args = parser.parse_args(
        ["paper-trade", "--strategy", "not_a_real_strategy", "--symbol", "EURUSD", "--timeframe", "H1"]
    )
    exit_code = args.func(args)
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "Unknown strategy" in err


def test_paper_trade_command_requires_strategy_symbol_timeframe():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["paper-trade", "--strategy", "trend_following"])


class _FakeKillSwitchEvent:
    def __init__(self, event_type, reason, equity=None, drawdown=None, created_at="2024-01-01T00:00:00Z"):
        self.event_type = event_type
        self.reason = reason
        self.equity = equity
        self.drawdown = drawdown
        self.created_at = created_at


class _FakeKillSwitch:
    """Stands in for src.risk.kill_switch.KillSwitch so this CLI test
    never touches the real project database -- cmd_kill_switch's own
    argument-handling/formatting logic is what's under test here, not
    KillSwitch itself (see tests/test_kill_switch.py for that)."""

    instances: list["_FakeKillSwitch"] = []

    def __init__(self, db=None):
        self.reset_calls: list[str] = []
        self._triggered = True
        _FakeKillSwitch.instances.append(self)

    def is_triggered(self):
        return self._triggered

    def reset(self, note, **kwargs):
        self.reset_calls.append(note)
        self._triggered = False

    def history(self, *, limit=10):
        return [_FakeKillSwitchEvent("trip", "max_portfolio_drawdown breached", equity=1800.0, drawdown=0.15)][:limit]


@pytest.fixture()
def fake_kill_switch(monkeypatch):
    _FakeKillSwitch.instances = []
    monkeypatch.setattr(main, "KillSwitch", _FakeKillSwitch)
    return _FakeKillSwitch


def test_kill_switch_command_shows_status_and_history(fake_kill_switch, capsys):
    parser = build_parser()
    args = parser.parse_args(["kill-switch"])
    assert args.func is cmd_kill_switch
    exit_code = args.func(args)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "TRIGGERED" in out
    assert "max_portfolio_drawdown breached" in out
    assert fake_kill_switch.instances[0].reset_calls == []


def test_kill_switch_command_records_a_reset_when_flag_given(fake_kill_switch, capsys):
    parser = build_parser()
    args = parser.parse_args(["kill-switch", "--reset", "reviewed manually, resuming"])
    exit_code = args.func(args)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "reset recorded" in out
    assert "clear" in out  # reset() flips the fake's internal state before the status line prints
    assert fake_kill_switch.instances[0].reset_calls == ["reviewed manually, resuming"]


def test_kill_switch_command_respects_history_limit(fake_kill_switch, capsys):
    parser = build_parser()
    args = parser.parse_args(["kill-switch", "--history", "0"])
    exit_code = args.func(args)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "max_portfolio_drawdown breached" not in out


def test_report_command_writes_files_and_prints_their_paths(monkeypatch, tmp_path, capsys):
    """Stubs run_report() itself so this test never touches the real
    project database (src.reporting.report_engine.run_report has its
    own thorough tests against an explicit in-memory db) -- this only
    checks cmd_report's own glue: does it call run_report() and print
    every path it returns."""
    from src.reporting.report_engine import ReportResult

    html_path = tmp_path / "reports" / "report.html"
    csv_path = tmp_path / "reports" / "exports" / "experiments.csv"
    json_path = tmp_path / "reports" / "exports" / "report.json"
    for p in (html_path, csv_path, json_path):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")

    fake_result = ReportResult(data=None, html_path=html_path, csv_paths=[csv_path], json_path=json_path)
    monkeypatch.setattr(main, "run_report", lambda: fake_result)
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)

    parser = build_parser()
    args = parser.parse_args(["report"])
    assert args.func is cmd_report
    exit_code = args.func(args)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "reports/report.html" in out
    assert "reports/exports/experiments.csv" in out
    assert "reports/exports/report.json" in out
    assert "Open the HTML file directly" in out


def test_robustness_command_reports_missing_when_no_raw_file(capsys):
    parser = build_parser()
    args = parser.parse_args(
        ["robustness", "--strategy", "trend_following", "--symbol", "EURUSD", "--timeframe", "H1"]
    )
    assert args.func is cmd_robustness
    exit_code = args.func(args)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "No raw CSV found" in out


def test_robustness_command_rejects_unknown_strategy(capsys):
    parser = build_parser()
    args = parser.parse_args(
        ["robustness", "--strategy", "not_a_real_strategy", "--symbol", "EURUSD", "--timeframe", "H1"]
    )
    exit_code = args.func(args)
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "Unknown strategy" in err


def test_robustness_command_requires_strategy_symbol_timeframe():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["robustness", "--strategy", "trend_following"])
