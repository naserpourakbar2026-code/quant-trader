"""Portfolio run persistence model (CLAUDE.md Sections 20, 29, 36).

One row per run_portfolio_analysis() call: which strategy/symbol/
timeframe combinations were screened, their pairwise return correlation,
and — per allocation method — the resulting weights and combined-
portfolio metrics versus the best single component. Kept separate from
`experiments`, `walkforward_windows` and `montecarlo_runs` because a
portfolio run is a comparison *across* combinations, not a property of
any one of them.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.core.db import Base


class PortfolioRun(Base):
    __tablename__ = "portfolio_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, unique=True, index=True)

    scenario: Mapped[str] = mapped_column(String)
    n_components: Mapped[int] = mapped_column(Integer)

    components_json: Mapped[str] = mapped_column(Text)  # list of {strategy, symbol, timeframe, scenario, metrics}
    correlation_json: Mapped[str] = mapped_column(Text)  # nested dict: component label -> component label -> corr
    allocations_json: Mapped[str] = mapped_column(Text)  # {method: {weights: {...}, metrics: {...}, verdict: {...}}}
    best_component_label: Mapped[str] = mapped_column(String)

    code_version: Mapped[str] = mapped_column(String)
    python_version: Mapped[str] = mapped_column(String)
    library_versions_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
