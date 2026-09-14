from main import build_parser, cmd_download_data, cmd_info


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
    for name in ["backtest", "live", "report"]:
        args = parser.parse_args([name])
        exit_code = args.func(args)
        assert exit_code != 0


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
