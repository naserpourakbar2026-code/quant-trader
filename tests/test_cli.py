from main import build_parser, cmd_info


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
    for name in ["download-data", "backtest", "live", "report"]:
        args = parser.parse_args([name])
        exit_code = args.func(args)
        assert exit_code != 0
