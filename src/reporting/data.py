"""Report data gathering (CLAUDE.md Section 30): pulls every persisted
engine's results into one place for the HTML/CSV/JSON renderers, without
any renderer needing to know how any individual engine's store works.
Every list here can legitimately be empty — nothing here fabricates
data when a phase hasn't been run against real data yet; the renderers
are expected to say "no data yet" rather than draw an empty chart as if
it meant something.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.backtest.experiment_store import ExperimentRecord, list_experiments
from src.core.db import Database
from src.execution.trade_store import PaperSessionRecord, PaperTradeRecord, list_sessions, list_trades
from src.montecarlo.run_store import MonteCarloRecord
from src.montecarlo.run_store import list_runs as list_montecarlo_runs
from src.portfolio.run_store import PortfolioRecord
from src.portfolio.run_store import list_runs as list_portfolio_runs
from src.risk.kill_switch import KillSwitch, KillSwitchEventRecord
from src.robustness.run_store import RobustnessEvaluationRecord
from src.robustness.run_store import list_evaluations as list_robustness_evaluations
from src.walkforward.window_store import WindowRecord, list_windows


@dataclass
class ReportData:
    experiments: list[ExperimentRecord] = field(default_factory=list)
    walkforward_windows: list[WindowRecord] = field(default_factory=list)
    montecarlo_runs: list[MonteCarloRecord] = field(default_factory=list)
    portfolio_runs: list[PortfolioRecord] = field(default_factory=list)
    paper_sessions: list[PaperSessionRecord] = field(default_factory=list)
    paper_trades_by_run: dict[str, list[PaperTradeRecord]] = field(default_factory=dict)
    kill_switch_events: list[KillSwitchEventRecord] = field(default_factory=list)
    robustness_evaluations: list[RobustnessEvaluationRecord] = field(default_factory=list)


def gather_report_data(db: Database | None = None, *, max_paper_sessions: int = 5) -> ReportData:
    """`max_paper_sessions` caps how many of the most recent paper
    sessions get their own chart set in the HTML report — unbounded
    would make the page grow without limit as more sessions accumulate;
    every session is still listed in the summary table and the CSV/JSON
    exports regardless of this cap."""
    paper_sessions_all = list_sessions(db=db)
    charted_sessions = paper_sessions_all[:max_paper_sessions]

    return ReportData(
        experiments=list_experiments(db=db),
        walkforward_windows=list_windows(db=db),
        montecarlo_runs=list_montecarlo_runs(db=db),
        portfolio_runs=list_portfolio_runs(db=db),
        paper_sessions=paper_sessions_all,
        paper_trades_by_run={s.run_id: list_trades(s.run_id, db=db) for s in charted_sessions},
        kill_switch_events=KillSwitch(db=db).history(limit=50),
        robustness_evaluations=list_robustness_evaluations(db=db),
    )
