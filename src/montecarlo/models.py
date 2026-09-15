"""Monte Carlo run persistence model (CLAUDE.md Sections 17, 29, 36).

One row per run_monte_carlo() call: the observed trade-return sample it
resampled from, the simulation-count/seed/ruin-threshold it used, and the
resulting outcome distribution summary — kept separate from `experiments`
(src.backtest.models) and `walkforward_windows` (src.walkforward.models)
because a Monte Carlo run is a distribution over many simulated equity
curves, not a single backtest or a single train/validation/OOS window.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.core.db import Base


class MonteCarloRun(Base):
    __tablename__ = "montecarlo_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, unique=True, index=True)

    strategy: Mapped[str] = mapped_column(String, index=True)
    symbol: Mapped[str] = mapped_column(String, index=True)
    timeframe: Mapped[str] = mapped_column(String, index=True)
    scenario: Mapped[str] = mapped_column(String)

    n_simulations: Mapped[int] = mapped_column(Integer)
    n_trades_observed: Mapped[int] = mapped_column(Integer)
    initial_capital: Mapped[float] = mapped_column(Float)
    ruin_threshold: Mapped[float] = mapped_column(Float)

    median_return: Mapped[float] = mapped_column(Float)
    p5_return: Mapped[float] = mapped_column(Float)
    p95_return: Mapped[float] = mapped_column(Float)
    worst_drawdown: Mapped[float] = mapped_column(Float)
    p95_drawdown: Mapped[float] = mapped_column(Float)
    median_losing_streak: Mapped[float] = mapped_column(Float)
    p95_losing_streak: Mapped[float] = mapped_column(Float)
    worst_losing_streak: Mapped[int] = mapped_column(Integer)
    probability_of_ruin: Mapped[float] = mapped_column(Float)
    probability_of_negative_return: Mapped[float] = mapped_column(Float)
    is_fragile: Mapped[bool] = mapped_column(Boolean)

    parameters_json: Mapped[str] = mapped_column(Text)
    # Histogram (bin_edges/counts), not the raw per-simulation array --
    # cheap enough to always persist, and it's what Phase 16's reporting
    # engine renders as the "Monte Carlo Distribution" chart (CLAUDE.md
    # Section 30) without needing to keep every simulation's return.
    return_histogram_json: Mapped[str] = mapped_column(Text)

    code_version: Mapped[str] = mapped_column(String)
    python_version: Mapped[str] = mapped_column(String)
    library_versions_json: Mapped[str] = mapped_column(Text)
    random_seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
