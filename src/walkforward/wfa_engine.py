"""Phase 9: Walk-forward analysis (CLAUDE.md Section 15, and the
out-of-sample requirement in Section 18).

Each window is split Training 60% / Validation 20% / Out-of-Sample 20%
(configurable): optimize on training only (Phase 8's Optuna engine),
assess the chosen parameter region's stability (Section 16), validate
the frozen parameters on the validation slice, then test on
out-of-sample data the optimizer never saw. The window then moves
forward and repeats ("rolling windows").

Never optimizes using OOS data — enforced by construction: `run_optimization`
only ever receives the training slice; validation/OOS data reaches the
engine only through `run_screening`'s read-only scoring path, and never
back into the optimizer for this or any later window.

Indicator continuity: each slice is screened with all *earlier* rows (from
the start of `raw_df`) passed as `warmup_df` (src.backtest.vectorbt_engine),
so a window's indicators have real history instead of a NaN warm-up gap —
without ever scoring a trade outside that window's own rows.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace

import pandas as pd

from src.backtest.vectorbt_engine import run_screening
from src.core.config import ParamSpec, load_settings
from src.core.db import Database
from src.core.reproducibility import git_commit_hash
from src.features.engine import FeatureParams
from src.optimization.optuna_engine import (
    assess_parameter_stability,
    composite_objective,
    run_optimization,
    split_params,
)
from src.walkforward.window_store import WindowRecord, save_window


@dataclass
class WindowBounds:
    index: int
    train: tuple[int, int]
    validation: tuple[int, int]
    oos: tuple[int, int]


def generate_windows(
    n_rows: int,
    *,
    train_pct: float = 0.6,
    validation_pct: float = 0.2,
    oos_pct: float = 0.2,
    window_bars: int | None = None,
    step_bars: int | None = None,
) -> list[WindowBounds]:
    """Row-index bounds for each window. `window_bars=None` means one
    window spanning the whole series (Section 15's basic 60/20/20 split);
    a smaller `window_bars` with `step_bars` (default: the OOS length)
    produces multiple rolling windows advancing across the series."""
    total_pct = train_pct + validation_pct + oos_pct
    if abs(total_pct - 1.0) > 1e-6:
        raise ValueError(f"train_pct + validation_pct + oos_pct must sum to 1.0, got {total_pct}")

    window_bars = window_bars or n_rows
    if window_bars > n_rows:
        raise ValueError(f"window_bars ({window_bars}) exceeds available rows ({n_rows})")

    train_bars = int(window_bars * train_pct)
    validation_bars = int(window_bars * validation_pct)
    oos_bars = window_bars - train_bars - validation_bars
    if train_bars == 0 or validation_bars == 0 or oos_bars == 0:
        raise ValueError(
            f"window_bars ({window_bars}) is too small to split into non-empty "
            f"train/validation/oos segments at {train_pct}/{validation_pct}/{oos_pct}"
        )
    step_bars = step_bars or oos_bars

    windows = []
    start = 0
    index = 0
    while start + window_bars <= n_rows:
        train_end = start + train_bars
        validation_end = train_end + validation_bars
        oos_end = start + window_bars
        windows.append(
            WindowBounds(
                index=index,
                train=(start, train_end),
                validation=(train_end, validation_end),
                oos=(validation_end, oos_end),
            )
        )
        start += step_bars
        index += 1

    if not windows:
        raise ValueError("No window fits in the given data with these settings")
    return windows


@dataclass
class WalkForwardWindowResult:
    index: int
    train_range: tuple[pd.Timestamp, pd.Timestamp]
    validation_range: tuple[pd.Timestamp, pd.Timestamp]
    oos_range: tuple[pd.Timestamp, pd.Timestamp]
    best_params: dict
    train_objective: float
    stability_score: float
    validation_metrics: dict
    oos_metrics: dict
    oos_objective: float
    oos_passed: bool


@dataclass
class WalkForwardReport:
    run_id: str
    strategy: str
    symbol: str
    timeframe: str
    windows: list[WalkForwardWindowResult] = field(default_factory=list)

    @property
    def oos_pass_rate(self) -> float:
        if not self.windows:
            return 0.0
        return sum(1 for w in self.windows if w.oos_passed) / len(self.windows)

    def parameter_consistency(self) -> dict[str, float]:
        """Coefficient of variation (std/|mean|) of each tunable
        parameter's chosen value across windows — low means the search
        kept landing on similar values window after window (CLAUDE.md
        Section 16's spirit, extended across time instead of just across
        one window's local neighborhood). Empty if fewer than 2 windows."""
        if len(self.windows) < 2:
            return {}
        names = self.windows[0].best_params.keys()
        result = {}
        for name in names:
            values = pd.Series([w.best_params[name] for w in self.windows], dtype=float)
            mean = values.mean()
            result[name] = float(values.std() / abs(mean)) if mean != 0 else float("inf")
        return result

    def to_text(self) -> str:
        lines = [
            "WALK-FORWARD ANALYSIS REPORT",
            f"Strategy: {self.strategy} | Symbol: {self.symbol} | Timeframe: {self.timeframe}",
            f"Windows: {len(self.windows)} | OOS pass rate: {self.oos_pass_rate:.0%}",
            "",
        ]
        for w in self.windows:
            verdict = "PASS" if w.oos_passed else "FAIL"
            lines.append(
                f"Window {w.index}: train {w.train_range[0].date()}..{w.train_range[1].date()} | "
                f"validation {w.validation_range[0].date()}..{w.validation_range[1].date()} | "
                f"OOS {w.oos_range[0].date()}..{w.oos_range[1].date()} -> {verdict}"
            )
            lines.append(
                f"  params={w.best_params} stability={w.stability_score:.2f} "
                f"oos_return={w.oos_metrics.get('total_return')} oos_trades={w.oos_metrics.get('trade_count')}"
            )
        consistency = self.parameter_consistency()
        if consistency:
            lines.append("")
            lines.append("Parameter consistency across windows (lower = more stable selection over time):")
            for name, cv in consistency.items():
                lines.append(f"  {name}: CV={cv:.2f}")
        return "\n".join(lines)


def run_walk_forward(
    strategy_family: str,
    symbol: str,
    timeframe: str,
    raw_df: pd.DataFrame,
    space: dict[str, ParamSpec],
    *,
    train_pct: float = 0.6,
    validation_pct: float = 0.2,
    oos_pct: float = 0.2,
    window_bars: int | None = None,
    step_bars: int | None = None,
    n_trials: int | None = None,
    scenario: str | None = None,
    seed: int | None = None,
    persist: bool = True,
    db: Database | None = None,
) -> WalkForwardReport:
    settings = load_settings()
    cfg = settings.optimization
    scenario = scenario or settings.execution.default_scenario
    code_version = git_commit_hash()

    raw_df = raw_df.reset_index(drop=True)
    windows = generate_windows(
        len(raw_df),
        train_pct=train_pct,
        validation_pct=validation_pct,
        oos_pct=oos_pct,
        window_bars=window_bars,
        step_bars=step_bars,
    )

    run_id = str(uuid.uuid4())
    results: list[WalkForwardWindowResult] = []

    for bounds in windows:
        train_start, train_end = bounds.train
        validation_start, validation_end = bounds.validation
        oos_start, oos_end = bounds.oos

        train_slice = raw_df.iloc[train_start:train_end].reset_index(drop=True)
        train_warmup = raw_df.iloc[0:train_start].reset_index(drop=True) if train_start > 0 else None

        opt_result = run_optimization(
            strategy_family,
            symbol,
            timeframe,
            train_slice,
            space,
            n_trials=n_trials,
            scenario=scenario,
            seed=seed,
            warmup_df=train_warmup,
            persist=False,
        )
        stability = assess_parameter_stability(
            strategy_family,
            symbol,
            timeframe,
            train_slice,
            opt_result.best_params,
            space,
            scenario=scenario,
            warmup_df=train_warmup,
        )

        feature_kwargs, strategy_kwargs = split_params(opt_result.best_params)
        feature_params = replace(FeatureParams.from_settings(), **feature_kwargs)

        validation_slice = raw_df.iloc[validation_start:validation_end].reset_index(drop=True)
        validation_warmup = raw_df.iloc[0:validation_start].reset_index(drop=True)
        validation_result = run_screening(
            strategy_family,
            symbol,
            timeframe,
            validation_slice,
            strategy_params=strategy_kwargs,
            feature_params=feature_params,
            scenario=scenario,
            warmup_df=validation_warmup,
            persist=False,
        )

        oos_slice = raw_df.iloc[oos_start:oos_end].reset_index(drop=True)
        oos_warmup = raw_df.iloc[0:oos_start].reset_index(drop=True)
        oos_result = run_screening(
            strategy_family,
            symbol,
            timeframe,
            oos_slice,
            strategy_params=strategy_kwargs,
            feature_params=feature_params,
            scenario=scenario,
            warmup_df=oos_warmup,
            persist=False,
        )

        oos_objective = composite_objective(oos_result.metrics, cfg)
        oos_passed = oos_objective > 0

        window_result = WalkForwardWindowResult(
            index=bounds.index,
            train_range=(train_slice["timestamp"].iloc[0], train_slice["timestamp"].iloc[-1]),
            validation_range=(validation_slice["timestamp"].iloc[0], validation_slice["timestamp"].iloc[-1]),
            oos_range=(oos_slice["timestamp"].iloc[0], oos_slice["timestamp"].iloc[-1]),
            best_params=opt_result.best_params,
            train_objective=opt_result.best_objective,
            stability_score=stability.score,
            validation_metrics=validation_result.metrics,
            oos_metrics=oos_result.metrics,
            oos_objective=oos_objective,
            oos_passed=oos_passed,
        )
        results.append(window_result)

        if persist:
            save_window(
                WindowRecord(
                    run_id=run_id,
                    window_index=bounds.index,
                    strategy=strategy_family,
                    symbol=symbol,
                    timeframe=timeframe,
                    scenario=scenario,
                    train_start=window_result.train_range[0].to_pydatetime(),
                    train_end=window_result.train_range[1].to_pydatetime(),
                    validation_start=window_result.validation_range[0].to_pydatetime(),
                    validation_end=window_result.validation_range[1].to_pydatetime(),
                    oos_start=window_result.oos_range[0].to_pydatetime(),
                    oos_end=window_result.oos_range[1].to_pydatetime(),
                    best_params=opt_result.best_params,
                    train_objective=opt_result.best_objective,
                    stability_score=stability.score,
                    validation_metrics=validation_result.metrics,
                    oos_metrics=oos_result.metrics,
                    oos_objective=oos_objective,
                    oos_passed=oos_passed,
                    code_version=code_version,
                    random_seed=seed if seed is not None else cfg.random_seed,
                ),
                db=db,
            )

    return WalkForwardReport(run_id=run_id, strategy=strategy_family, symbol=symbol, timeframe=timeframe, windows=results)
