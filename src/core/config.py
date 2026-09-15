"""Configuration loading and validation.

Loads config/settings.yaml, config/strategies.yaml and config/brokers.yaml
into validated pydantic models. No trading-relevant value should be
hard-coded elsewhere in the codebase — add it to a YAML file and a field
here instead.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


class CooldownConfig(BaseModel):
    losses: int = Field(gt=0)
    cooldown_bars: int = Field(gt=0)


class RiskConfig(BaseModel):
    risk_per_trade: float = Field(gt=0, le=0.05)
    allowed_risk_levels: list[float]
    max_daily_loss: float = Field(gt=0, le=1)
    max_weekly_loss: float = Field(gt=0, le=1)
    max_portfolio_drawdown: float = Field(gt=0, le=1)
    max_open_positions: int = Field(gt=0)
    max_correlated_exposure: float = Field(gt=0, le=1)
    max_leverage: float = Field(gt=0)
    max_margin_usage: float = Field(gt=0, le=1)
    daily_trade_limit: int = Field(gt=0)
    cooldown_after_consecutive_losses: CooldownConfig

    @field_validator("risk_per_trade")
    @classmethod
    def risk_per_trade_must_be_allowed(cls, v: float, info: Any) -> float:
        # allowed_risk_levels may not be populated yet during this validator
        # (pydantic v2 validates fields in declaration order); re-checked
        # again below in AppConfig.
        return v


class AccountConfig(BaseModel):
    initial_capital: float = Field(gt=0)
    currency: str


class TwelveDataSourceConfig(BaseModel):
    """config/settings.yaml `data.twelvedata` block (CLAUDE.md Section 2,
    source C). The API key itself is never here — only the name of the
    .env variable holding it (CLAUDE.md Section 42)."""

    enabled: bool = False
    api_key_env: str = "TWELVE_DATA_API_KEY"
    rate_limit: dict = Field(default_factory=lambda: {"requests_per_second": 1.0})
    timeout_seconds: float = 10.0
    max_retries: int = 3
    retry_base_delay_seconds: float = 1.0
    output_size: int = 5000


class DataConfig(BaseModel):
    symbols: list[str]
    optional_symbols: list[str] = Field(default_factory=list)
    timeframes: list[str]
    source: str
    twelvedata: TwelveDataSourceConfig = Field(default_factory=TwelveDataSourceConfig)

    @field_validator("source")
    @classmethod
    def source_must_be_supported(cls, v: str) -> str:
        allowed = {"csv", "mt5", "twelvedata"}
        if v not in allowed:
            raise ValueError(f"data.source must be one of {allowed}, got {v!r}")
        return v


class CostScenario(BaseModel):
    """Illustrative transaction-cost assumptions for one execution scenario
    (CLAUDE.md Section 11) — NOT real broker figures. Real spread/commission/
    slippage are broker- and symbol-specific and only become known once a
    broker adapter is connected (Phase 12/13); until then these are stated
    placeholders a screening/backtest run uses so costs aren't silently
    zero, not a claim about any real broker's pricing."""

    commission_pct: float = Field(ge=0)
    slippage_pct: float = Field(ge=0)


class ExecutionConfig(BaseModel):
    scenarios: list[str]
    default_scenario: str
    costs: dict[str, CostScenario] = Field(default_factory=dict)

    @field_validator("default_scenario")
    @classmethod
    def default_scenario_in_scenarios(cls, v: str, info: Any) -> str:
        scenarios = info.data.get("scenarios", [])
        if scenarios and v not in scenarios:
            raise ValueError(f"default_scenario {v!r} not in scenarios {scenarios}")
        return v

    @model_validator(mode="after")
    def every_scenario_has_a_cost_profile(self) -> "ExecutionConfig":
        missing = [s for s in self.scenarios if s not in self.costs]
        if missing:
            raise ValueError(f"scenarios missing a cost profile in execution.costs: {missing}")
        return self


class ValidationConfig(BaseModel):
    weekend_close_day: int = Field(ge=0, le=6)
    weekend_close_hour: int = Field(ge=0, le=23)
    weekend_open_day: int = Field(ge=0, le=6)
    weekend_open_hour: int = Field(ge=0, le=23)
    spread_outlier_zscore: float = Field(gt=0)


class FeaturesConfig(BaseModel):
    ema_fast_period: int = Field(gt=0)
    ema_slow_period: int = Field(gt=0)
    atr_period: int = Field(gt=0)
    donchian_period: int = Field(gt=0)
    bollinger_period: int = Field(gt=0)
    bollinger_std: float = Field(gt=0)
    rsi_period: int = Field(gt=0)
    adx_period: int = Field(gt=0)
    momentum_period: int = Field(gt=0)
    trend_slope_lookback: int = Field(gt=0)
    trend_strong_threshold: float = Field(gt=0)
    trend_weak_threshold: float = Field(gt=0)
    volatility_lookback: int = Field(gt=0)
    volatility_low_percentile: float = Field(ge=0, le=100)
    volatility_high_percentile: float = Field(ge=0, le=100)
    range_squeeze_lookback: int = Field(gt=0)

    @model_validator(mode="after")
    def cross_field_ordering(self) -> "FeaturesConfig":
        # Cross-field checks in one place (not per-field validators) since
        # pydantic v2 validates fields in declaration order and these pairs
        # aren't declared in an order that would make per-field checks
        # reliably see both sides.
        if self.ema_slow_period <= self.ema_fast_period:
            raise ValueError(
                f"ema_slow_period ({self.ema_slow_period}) must be greater than "
                f"ema_fast_period ({self.ema_fast_period})"
            )
        if self.trend_strong_threshold <= self.trend_weak_threshold:
            raise ValueError(
                f"trend_strong_threshold ({self.trend_strong_threshold}) must be greater than "
                f"trend_weak_threshold ({self.trend_weak_threshold})"
            )
        if self.volatility_high_percentile <= self.volatility_low_percentile:
            raise ValueError(
                f"volatility_high_percentile ({self.volatility_high_percentile}) must be greater than "
                f"volatility_low_percentile ({self.volatility_low_percentile})"
            )
        return self


class OptimizationWeights(BaseModel):
    """Illustrative weighting for the composite objective (CLAUDE.md
    Section 14: "Profit Factor + Sharpe + Sortino + Expectancy, penalized
    by max drawdown, low trade count, and high parameter sensitivity").
    Not a claim these exact weights are optimal — a stated, adjustable
    default, same pattern as the cost-scenario placeholders."""

    profit_factor_cap: float = Field(gt=0)  # cap so a zero-losing-trade run doesn't get an infinite score
    drawdown_penalty: float = Field(ge=0)  # multiplier on |max_drawdown|
    trade_count_penalty: float = Field(ge=0)  # multiplier per trade short of min_trades


class StabilityConfig(BaseModel):
    neighbor_step_pct: float = Field(gt=0, le=1)  # float params: perturb by +/- this fraction
    neighbor_step_int: int = Field(gt=0)  # int params: perturb by +/- this many steps


class OptimizationConfig(BaseModel):
    n_trials: int = Field(gt=0)
    min_trades: int = Field(gt=0)
    random_seed: int
    weights: OptimizationWeights
    stability: StabilityConfig


class MonteCarloFragilityConfig(BaseModel):
    """Stated, adjustable thresholds for CLAUDE.md Section 17's "fragile
    under MC strategies are not robust" — not a claim these exact cutoffs
    are the only reasonable ones."""

    max_probability_of_ruin: float = Field(ge=0, le=1)
    max_probability_of_negative_return: float = Field(ge=0, le=1)


class MonteCarloConfig(BaseModel):
    n_simulations: int = Field(gt=0)
    random_seed: int
    ruin_threshold: float = Field(gt=0, lt=1)  # equity <= this fraction of initial capital counts as "ruin"
    execution_noise_std: float = Field(ge=0)  # extra per-trade return noise layered on resampled historical returns
    min_trades: int = Field(gt=0)  # below this, results are still reported but flagged unreliable
    fragility: MonteCarloFragilityConfig


class RobustnessWeights(BaseModel):
    """CLAUDE.md Section 31's weighting table for the Final Robustness
    Score — a stated, adjustable default (like every other weighting
    table in this codebase), validated to sum to 1.0 so no single
    factor can silently dominate by an arithmetic accident."""

    oos_performance: float = Field(ge=0, le=1)
    drawdown: float = Field(ge=0, le=1)
    profit_factor: float = Field(ge=0, le=1)
    sharpe_sortino: float = Field(ge=0, le=1)
    parameter_stability: float = Field(ge=0, le=1)
    monte_carlo_robustness: float = Field(ge=0, le=1)
    cost_sensitivity: float = Field(ge=0, le=1)
    trade_count_reliability: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "RobustnessWeights":
        total = (
            self.oos_performance + self.drawdown + self.profit_factor + self.sharpe_sortino
            + self.parameter_stability + self.monte_carlo_robustness + self.cost_sensitivity
            + self.trade_count_reliability
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"robustness.weights must sum to 1.0, got {total}")
        return self


class RobustnessConfig(BaseModel):
    """CLAUDE.md Sections 31-32: the Final Robustness Score and the
    minimum bar a strategy must clear to be selected as one of the
    top-3 "robust" candidates — never the strategy with the highest
    backtest profit alone (Section 44)."""

    min_pass_score: float = Field(ge=0, le=100)
    sharpe_reference: float = Field(gt=0)  # a Sharpe at/above this scores 100 on that sub-factor
    cost_stress_spread_multipliers: list[float]
    weights: RobustnessWeights

    @field_validator("cost_stress_spread_multipliers")
    @classmethod
    def multipliers_include_baseline(cls, v: list[float]) -> list[float]:
        if not v or min(v) < 1.0:
            raise ValueError("cost_stress_spread_multipliers must be non-empty and include a >=1.0 baseline")
        return sorted(v)


class PortfolioConfig(BaseModel):
    """CLAUDE.md Section 20: don't assume one strategy is optimal — screen
    many strategy x symbol x timeframe combinations, but only carry the
    statistically-usable survivors into the (deliberately basic)
    correlation/allocation step."""

    top_n: int = Field(gt=0)  # cap on how many screened combinations enter the portfolio step
    min_trades: int = Field(gt=0)  # combinations with fewer trades than this are excluded as unreliable
    allocation_methods: list[str]

    @field_validator("allocation_methods")
    @classmethod
    def allocation_methods_supported(cls, v: list[str]) -> list[str]:
        allowed = {"equal_weight", "inverse_volatility"}
        unknown = [m for m in v if m not in allowed]
        if unknown:
            raise ValueError(f"portfolio.allocation_methods has unsupported entries {unknown}; expected one of {allowed}")
        return v


class PathsConfig(BaseModel):
    data_raw: str
    data_processed: str
    data_cache: str
    reports: str
    logs: str


class LoggingConfig(BaseModel):
    level: str = "INFO"
    trade_log_file: str
    system_log_file: str


class AppConfig(BaseModel):
    account: AccountConfig
    risk: RiskConfig
    data: DataConfig
    execution: ExecutionConfig
    validation: ValidationConfig
    features: FeaturesConfig
    optimization: OptimizationConfig
    montecarlo: MonteCarloConfig
    portfolio: PortfolioConfig
    robustness: RobustnessConfig
    live_trading: bool
    paths: PathsConfig
    logging: LoggingConfig

    @field_validator("risk")
    @classmethod
    def risk_per_trade_allowed(cls, v: RiskConfig) -> RiskConfig:
        if v.allowed_risk_levels and v.risk_per_trade not in v.allowed_risk_levels:
            raise ValueError(
                f"risk_per_trade {v.risk_per_trade} not in allowed_risk_levels "
                f"{v.allowed_risk_levels}"
            )
        return v


class ParamSpec(BaseModel):
    """One tunable parameter's search range for Optuna (CLAUDE.md Section
    14) — every entry here must have a logical economic/trading
    justification; this is not an "optimize everything" grid."""

    type: str  # "int" | "float"
    low: float
    high: float
    step: float | None = None

    @field_validator("type")
    @classmethod
    def type_must_be_supported(cls, v: str) -> str:
        if v not in ("int", "float"):
            raise ValueError(f"ParamSpec.type must be 'int' or 'float', got {v!r}")
        return v

    @model_validator(mode="after")
    def low_less_than_high(self) -> "ParamSpec":
        if self.low >= self.high:
            raise ValueError(f"low ({self.low}) must be < high ({self.high})")
        return self


class StrategyDefinition(BaseModel):
    family: str
    enabled: bool = True
    symbols: list[str] = Field(default_factory=list)
    timeframes: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    optimization_space: dict[str, ParamSpec] = Field(default_factory=dict)


class StrategiesConfig(BaseModel):
    stop_loss_types: list[str]
    take_profit_ratios: list[float]
    strategies: dict[str, StrategyDefinition] = Field(default_factory=dict)


class BrokerEntryConfig(BaseModel):
    enabled: bool = False
    name: str | None = None
    environment: str = "demo"

    model_config = {"extra": "allow"}


class BrokersConfig(BaseModel):
    brokers: dict[str, BrokerEntryConfig]
    active_broker: str | None = None

    @field_validator("active_broker")
    @classmethod
    def active_broker_must_exist(cls, v: str | None, info: Any) -> str | None:
        if v is None:
            return v
        brokers = info.data.get("brokers", {})
        if v not in brokers:
            raise ValueError(f"active_broker {v!r} not defined in brokers.yaml")
        return v


def is_live_trading_enabled() -> bool:
    """Live trading requires BOTH env flags to be true. Never on by default."""
    live = os.getenv("LIVE_TRADING", "false").strip().lower() == "true"
    confirmed = os.getenv("LIVE_CONFIRMATION", "false").strip().lower() == "true"
    return live and confirmed


@lru_cache(maxsize=1)
def load_settings(config_dir: Path | None = None) -> AppConfig:
    directory = config_dir or CONFIG_DIR
    raw = _load_yaml(directory / "settings.yaml")
    return AppConfig(**raw)


@lru_cache(maxsize=1)
def load_strategies(config_dir: Path | None = None) -> StrategiesConfig:
    directory = config_dir or CONFIG_DIR
    raw = _load_yaml(directory / "strategies.yaml")
    return StrategiesConfig(**raw)


@lru_cache(maxsize=1)
def load_brokers(config_dir: Path | None = None) -> BrokersConfig:
    directory = config_dir or CONFIG_DIR
    raw = _load_yaml(directory / "brokers.yaml")
    return BrokersConfig(**raw)
