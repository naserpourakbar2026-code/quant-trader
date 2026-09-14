import pytest

from main import (
    build_parser,
    cmd_backtest,
    cmd_download_data,
    cmd_info,
    cmd_monte_carlo,
    cmd_optimize,
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
    for name in ["live", "report"]:
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
