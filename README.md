# quant-trader

A production-oriented quantitative trading research and execution framework
for Forex CFD trading. **Not** a simple trading bot — the goal is to find
strategies with a realistic probability of remaining profitable
out-of-sample, not the highest historical backtest profit.

Full specification: see [`CLAUDE.md`](./CLAUDE.md). This project is built
phase by phase (see Section 40 there); progress so far:

| Phase | Scope                                          | Status  |
|-------|-------------------------------------------------|---------|
| 1     | Architecture + environment setup                | ✅ done |
| 2     | Data ingestion                                  | ✅ done |
| 3     | Data validation                                 | ✅ done |
| 4     | Feature engine                                  | ✅ done |
| 5     | Three strategies                                | ✅ done |
| 6     | vectorbt research engine                        | ✅ done |
| 7     | Backtrader validation engine                    | ✅ done |
| 8     | Optuna optimization                             | ✅ done |
| 9     | Walk-forward                                    | ✅ done |
| 10    | Monte Carlo                                     | ✅ done |
| 11    | Portfolio engine                                | ✅ done |
| 12    | MT5 adapter (code + mocked tests, Windows-only) | ✅ done |
| 13    | Generic broker API adapter                      | ✅ done |
| 14    | Paper trading                                   | ✅ done |
| 15    | Risk & kill switch                              | ✅ done |
| 16    | Reporting                                       | pending |
| 17    | Full integration tests                          | pending |
| 18    | Final robustness evaluation                     | pending |

## Requirements

- Python 3.11+
- Linux/macOS for research, backtesting, paper trading, and the generic
  broker adapter. The `MetaTrader5` package (used only by the MT5 broker
  adapter) is Windows-only and cannot run in this environment — see
  `requirements-windows.txt` and CLAUDE.md Section 22.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# On Windows, if you need the MT5 adapter:
# pip install -r requirements-windows.txt

cp .env.example .env   # then fill in real values; .env is gitignored
```

## Configuration

All trading-relevant values live in YAML, never hard-coded:

- `config/settings.yaml` — capital, risk limits, symbols/timeframes, execution
  scenarios, logging/paths.
- `config/strategies.yaml` — strategy registry and parameters for the three
  families below (Phase 5).
- `config/brokers.yaml` — non-secret broker connection settings. Secrets
  (API keys, MT5 credentials) go in `.env`, never in this file or in source.

Config is loaded and validated through pydantic models in
`src/core/config.py`.

## Database (Phase 6)

SQLite via SQLAlchemy (`src/core/db.py`), so a later move to PostgreSQL
only needs a `DATABASE_URL` change (CLAUDE.md Section 29). Defaults to
`data/quant_trader.db`; override via `.env`'s `DATABASE_URL`. Production
code calls `get_default_database()` (a lazily-created singleton); tests
construct their own `Database("sqlite:///:memory:")` so they never touch
the real project database. Currently holds `experiments`
(`src/backtest/models.py`) — one row per vectorbt screening or Backtrader
validation run, distinguished by an `engine` column ("vectorbt" |
"backtrader") — Optuna's best trial persists here too, with `engine:
"optuna"`. Also holds `walkforward_windows`
(`src/walkforward/models.py`) — one row per walk-forward window (Phase
9), a different kind of record (three sub-periods and a pass/fail
verdict) than a single backtest `Experiment`. And `montecarlo_runs`
(`src/montecarlo/models.py`) — one row per Monte Carlo run (Phase 10): the
observed trade count it resampled from, simulation count/seed, and the
resulting outcome-distribution summary. And `portfolio_runs`
(`src/portfolio/models.py`) — one row per portfolio analysis run (Phase
11): the screened components, their pairwise correlation, and each
allocation method's weights/metrics/verdict versus the best single
component. And `paper_sessions` / `paper_trades`
(`src/execution/models.py`, Phase 14) — one session summary row plus one
row per *closed* trade, the latter holding exactly CLAUDE.md Section
28's trade-log field list (timestamp, symbol, strategy, signal, entry,
SL, TP, size, risk, spread, slippage, exit, PnL, R multiple, reason) —
the first phase anything gets logged at that per-trade granularity
(Phases 6-11's `experiments` table only ever stored aggregate metrics).
And `kill_switch_events` (`src/risk/models.py`, Phase 15) — an
append-only audit log (never a mutable row) of every kill-switch trip
and reset, which is what makes the kill switch durable across sessions
and process restarts (Section 7's "hard" kill switch), not a fresh
in-memory flag every run forgets. More tables arrive with the phases
that need them.

## Data ingestion (Phase 2)

Two sources, dispatched by `data.source` in `config/settings.yaml`:

- **CSV** (default): place raw files under `data/raw/<SYMBOL>_<TIMEFRAME>.csv`
  with at least `timestamp,open,high,low,close,volume` columns (optional
  `tick_volume`, `spread`, `real_volume` are used if present). Nothing is
  fetched or fabricated — a missing file is reported as `MISSING`, not
  silently skipped or filled in.
- **MT5**: Windows-only (CLAUDE.md Section 22). `src/data/mt5_loader.py`
  imports `MetaTrader5` lazily so it stays importable on Linux/macOS; only
  calling it requires a running terminal. Use `--start`/`--end` with
  `download-data` once running on Windows.

Ingestion writes standardized candles (`timestamp, symbol, timeframe, open,
high, low, close, tick_volume, spread, real_volume`) to
`data/processed/<SYMBOL>_<TIMEFRAME>.csv`.

## Data validation (Phase 3)

`validate-data` runs the checks from CLAUDE.md Section 3 against the raw
CSV for each symbol/timeframe and prints a `DATA QUALITY REPORT`. It only
**detects** — it never repairs or drops anything:

- duplicate timestamps / duplicate rows
- out-of-order timestamps
- impossible OHLC values (e.g. high below open/close/low) and negative/zero prices
- abnormal spreads (z-score vs. `validation.spread_outlier_zscore` in `config/settings.yaml`)
- missing candles (expected-frequency gaps, excluding the configured weekend closure window)
- weekend candles (present but inside that closure window — often a broker/CFD-hours anomaly worth checking)
- timezone inconsistencies (mixed `Z` / explicit-offset / naive timestamp strings in the raw file)

The weekend-closure window and spread-outlier threshold are configurable
under `validation:` in `config/settings.yaml` — broker calendars differ.

## Feature engine (Phase 4)

`src/features/` is the shared indicator/regime library every strategy
family (Phase 5) will build on — a pure library, not a CLI command (the
brief's CLI list has none for it):

- `src/features/indicators.py`: EMA, ATR (Wilder), Donchian channel,
  Bollinger Bands, RSI (Wilder), rate of change, distance from a moving
  average (in % and in ATR units).
- `src/features/regime.py`: objective, threshold-based classification —
  `classify_trend` (no/weak/strong, from the ATR-normalized slope of the
  slow EMA, only counted when the fast EMA agrees with its direction),
  `classify_volatility` (low/normal/high, from the ATR's own trailing
  percentile rank), `classify_range` (a Bollinger-width squeeze while
  `no_trend`).
- `src/features/engine.py`: `compute_features(df)` runs all of the above
  on one symbol/timeframe's standardized candles and adds a
  `breakout_strength` column (signed distance, in ATR units, beyond the
  *prior* bar's Donchian channel).

All periods/thresholds live under `features:` in `config/settings.yaml`
(`FeatureParams.from_settings()`) and can be overridden per call — e.g. for
Optuna sweeps in Phase 8.

## Strategies (Phase 5)

`src/strategies/` implements three independent families (CLAUDE.md
Section 5), each on the standard interface from Section 6
(`generate_signal`, `calculate_stop_loss`, `calculate_take_profit`,
`calculate_position_size`, `validate_signal` — see `src/strategies/base.py`):

- **`trend_following.py`** — trades a Donchian-style breakout, but only in
  the direction of an already objectively-classified trend
  (`trend_regime` from Phase 4), with a volatility filter against
  breakouts during `low_vol`.
- **`mean_reversion.py`** — fades a Bollinger Band extreme confirmed by
  RSI, but refuses to fade the direction of a *strong* trend (Section 5B's
  "avoid aggressively trading against strong trends"); its take-profit is
  the reversion target (the Bollinger middle band) rather than a fixed
  R-multiple.
- **`momentum.py`** — a genuine multi-factor score (momentum, breakout
  strength, trend alignment), each with a stated justification; tick
  volume is used only as a capped ±20% confirmation multiplier, never a
  standalone signal, since retail-forex tick volume is a price-change
  count, not real traded volume.

Stop-loss types (`src/strategies/stops.py`: ATR, swing, fixed %,
volatility-scaled ATR) and take-profit types (`src/strategies/targets.py`:
R-multiple, ATR target, mean-reversion target) are implemented as pluggable
functions per CLAUDE.md Sections 9-10 — *which* one performs best is an
empirical question for Backtrader validation (Phase 7) and Optuna (Phase
8), not decided here. `create_strategy(family, params)` in
`src/strategies/__init__.py` instantiates any of the three by name from
`config/strategies.yaml`.

`calculate_position_size()`'s formula (equity × risk% ÷ stop distance)
assumes the traded pair's quote currency equals the account currency — a
stated simplification until real broker pip-value/contract-size
specifications are available from a connected broker adapter (Phase 12/13).

## vectorbt research engine (Phase 6)

`src/backtest/vectorbt_engine.py` — fast, vectorized screening across
strategy × symbol × timeframe × parameter combinations (CLAUDE.md Section
12). A coarse filter, not the final word: Backtrader (Phase 7) re-checks
survivors with realistic, event-driven execution.

- `run_screening(strategy_family, symbol, timeframe, raw_df, ...)` computes
  features, reads the strategy's `generate_signals_vectorized()` table
  (the *same* table `generate_signal()` reads from for live/paper trading
  — see Phase 5 above), converts it into vectorbt entries/exits plus
  fractional SL/TP, and runs `vbt.Portfolio.from_signals()` under one of
  three cost scenarios (`execution.costs` in `config/settings.yaml` —
  illustrative commission/slippage, not real broker figures, same caveat
  as position sizing in Phase 5).
- `screen_parameter_grid(...)` runs one screening per combination in a
  parameter grid (only parameters with a stated economic justification
  belong in a grid — Section 14).
- `filter_top_candidates(results, top_n, min_trades)` — the concrete
  "avoid wasting compute on obviously poor parameter regions" mechanism:
  ranks by Sharpe ratio after excluding statistically unreliable
  (too-few-trades) results, so only the survivors are worth Backtrader's
  greater expense.
- Every run is persisted as an `Experiment` row
  (`src/backtest/experiment_store.py`, SQLite via SQLAlchemy — Section
  29) with a unique ID, its parameters/metrics, and full reproducibility
  metadata (Section 36): git commit hash, data-file fingerprint, Python
  and library versions.

No new CLI command — Section 38's CLI list has none for screening;
`backtest`/`optimize` (Phases 7-8) will call into this engine directly.

⚠️ **Verified dependency combination**: `vectorbt` 1.1.0 requires
`pandas>=3.0.3` and `numpy>=2.4.6` in this environment's package index —
notably *not* the `pandas<3`/`numpy<2` range assumed in Phase 1.
`requirements.txt` now reflects what's actually verified working (the
full test suite passes against pandas 3.0.5 / numpy 2.4.6). Separately,
`vectorbt` 1.1.0 fails at import time against `plotly` 7.x (it references
a `scattermapbox` property `plotly` removed) — the `plotly<6.0` pin is
load-bearing, not cosmetic.

## Backtrader validation engine (Phase 7)

`src/backtest/backtrader_engine.py` — after vectorbt screens candidates
cheaply (Phase 6), this re-checks the survivors with genuine event-driven,
next-bar execution (CLAUDE.md Section 13): orders placed in `next()` fill
at the *following* bar's open rather than the signal bar's own close, the
broker can reject an order for insufficient margin, and stop-loss/
take-profit are real pending child orders Backtrader checks bar-by-bar —
not a fractional-distance approximation like the vectorbt screen uses.
The entry-condition logic itself is untouched: `run_validation()` reads
the *same* `generate_signals_vectorized()` table Phase 5/6 use, so only
execution realism differs between engines, never "what the strategy
decided" — `compare_with_screening(screening_metrics, validation_metrics)`
puts both side by side per Section 13's stated purpose ("catches
unrealistic results from vectorized backtest assumptions").

Costs: commission from `execution.costs` (same scenarios as Phase 6) plus
the real spread from the data's own `spread` column (averaged, folded
into the commission percentage), leverage-based margin via
`risk.max_leverage`. Persists as an `Experiment` row with `engine:
"backtrader"` so a screening result and its validation are queryable
side by side.

⚠️ **Two real bugs found and fixed while building this** (both now
pinned by regression tests):
1. Backtrader's `setcommission(leverage=...)`, when `commtype`/`margin`
   are left at their defaults, silently falls back to legacy
   stock-like accounting that requires the *full notional* as cash and
   ignores `leverage` entirely — this alone rejected ~90%+ of orders.
   Fixed by passing `commtype`/`stocklike=False`/`automargin=1/leverage`
   explicitly, which is what actually turns leverage into a real margin
   requirement.
2. `buy_bracket()`/`sell_bracket()` default their entry leg to a **Limit**
   order at the signal bar's price, not Market — it can sit unfilled for
   many bars (the opposite of "next-bar execution"), during which a
   signal that stays active would keep re-issuing brand-new duplicate
   entry orders every bar, each needing its own margin, until the account
   couldn't afford any of them. Fixed by forcing `exectype=bt.Order.Market`
   and guarding entries on a pending-order check.

## Optuna optimization (Phase 8)

`src/optimization/optuna_engine.py` — searches only the parameters listed
in each strategy's `optimization_space` (`config/strategies.yaml`): a
small, economically-justified set per CLAUDE.md Section 14 (breakout
period + stop/TP multipliers for trend-following; Bollinger period/
deviation + RSI thresholds for mean reversion; momentum period + entry
threshold + TP multiple for the multi-factor strategy) — never "dozens of
parameters at once". Each trial is evaluated via Phase 6's vectorbt
screen, so the search reads the exact same `generate_signals_vectorized()`
computation live/paper trading and Backtrader validation do.

- `composite_objective()` implements Section 14's own formula — Profit
  Factor + Sharpe + Sortino + Expectancy, penalized by `|max_drawdown|`
  and a shortfall below `optimization.min_trades` — with weights in
  `config/settings.yaml`'s `optimization:` block (stated, adjustable
  defaults, not a claim of optimality, same pattern as the cost-scenario
  placeholders in Phase 6).
- `run_optimization()` runs an Optuna TPE-sampler study (seeded for
  reproducibility — Section 36) and persists only the *best* trial as an
  `Experiment` with `engine: "optuna"` — individual trials aren't
  persisted (they'd flood the table for no benefit).
- `assess_parameter_stability()` — Section 16's "never select an isolated
  spike": after the search, perturbs each tunable parameter by one step
  and re-scores; a stable region's neighbors score nearly as well as the
  best point, an isolated spike collapses on its neighbors. Done once,
  post-hoc (checking every trial's neighborhood during the search itself
  would multiply the total cost by the neighborhood size for no benefit).
- `python main.py optimize` requires an explicit `--strategy --symbol
  --timeframe` (never a broad default sweep — a multi-trial search isn't
  something to run accidentally across every combination) and prints the
  best parameters, objective, key metrics, and the stability score.

## Walk-forward analysis (Phase 9)

`src/walkforward/wfa_engine.py` — CLAUDE.md Section 15's rolling
train/validate/out-of-sample discipline, plus Section 18's out-of-sample
requirement. Each window is split Training 60% / Validation 20% /
Out-of-Sample 20% (configurable): optimize on training only (Phase 8's
Optuna engine) → assess the chosen parameters' stability (Section 16) →
validate the frozen parameters on the validation slice → test on
out-of-sample data the optimizer never saw. The window then advances and
repeats ("rolling windows").

- **Never optimizes using OOS data** — enforced by construction, not by
  convention: `run_optimization()` only ever receives the training
  slice; validation/OOS data only ever reaches `run_screening()`'s
  read-only scoring path.
- **Indicator continuity without leakage**: each slice is screened with
  every *earlier* row (from the very start of the series) passed as
  `warmup_df` (a new parameter on `run_screening()`/`run_optimization()`/
  `assess_parameter_stability()`), so a window's indicators have real
  history instead of an artificial NaN warm-up gap — while trades and
  metrics are still scored only over that window's own rows, never the
  warmup context.
- A window's out-of-sample verdict is `composite_objective(oos_metrics) >
  0` (Phase 8's own objective, reused for consistency) — Section 18's
  "if a strategy fails OOS, reject it", made concrete.
- `WalkForwardReport.parameter_consistency()` extends Section 16's
  "never trust an isolated spike" across *time*: the coefficient of
  variation of each tunable parameter's chosen value across windows —
  low means the search keeps landing on similar values window after
  window, not scattering.
- Persists one `WalkForwardWindow` row per window
  (`src/walkforward/window_store.py`, its own table — a window is a
  different kind of record than a single backtest `Experiment`: three
  sub-periods and a pass/fail verdict, not one run).
- `python main.py walk-forward` requires an explicit `--strategy
  --symbol --timeframe` (strictly more expensive than `optimize` — a
  full Optuna search runs *per window*) and prints the full report:
  every window's date ranges, chosen parameters, stability score,
  OOS metrics and verdict, plus the aggregate OOS pass rate and
  parameter-consistency table.

## Monte Carlo analysis (Phase 10)

`src/montecarlo/mc_engine.py` — CLAUDE.md Section 17: a single historical
backtest is one draw from an unknown distribution of possible outcomes,
not a promise. This bootstrap-resamples a strategy's own observed
per-trade returns (from the vectorbt screening engine, Phase 6 —
`src.backtest.vectorbt_engine.get_trade_returns()`, a thin wrapper that
shares its Portfolio construction with `run_screening()` so the trades
Monte Carlo resamples from are exactly the trades screening/optimization
scored) thousands of times and reports the *spread* of outcomes, not a
single number.

- Each of `montecarlo.n_simulations` (default 1000) draws resamples
  `len(observed_trades)` trades **with replacement** — randomizing both
  trade order and trade-return composition (Section 17) — then adds
  independent Gaussian noise per resampled trade
  (`montecarlo.execution_noise_std`) to represent slippage/execution
  variation beyond the single fixed cost scenario already baked into the
  observed returns. The resampled sequence is compounded into a
  simulated equity curve starting from `initial_capital`.
- Reports exactly Section 17's list: median/5th/95th percentile return,
  worst drawdown, 95th-percentile drawdown, median/95th-percentile/worst
  losing streak, probability of ruin (equity ever falling to
  `montecarlo.ruin_threshold` — default 50% — of initial capital in a
  simulation), and probability of a negative return.
- `is_fragile` is a stated, adjustable rule (`montecarlo.fragility` in
  `config/settings.yaml`): probability of ruin or probability of a
  negative return exceeding a configured cutoff — Section 17's "fragile
  under Monte Carlo strategies are not robust", made concrete rather than
  left to eyeballing a chart.
- Too few observed trades (below `montecarlo.min_trades`) never blocks
  the run — the result is still computed, just marked
  `insufficient_data` and printed with an explicit warning, per Section
  18/31's "trade count" robustness factor.
- Persists one `MonteCarloRun` row per call (`src/montecarlo/run_store.py`)
  with full reproducibility metadata (Section 36): seed, code/library
  versions, the parameters evaluated.
- `python main.py monte-carlo` requires an explicit `--strategy --symbol
  --timeframe` (evaluates a strategy/parameter combination already
  chosen elsewhere — it does not search for one) and prints the full
  report.

## Portfolio engine (Phase 11)

`src/portfolio/portfolio_engine.py` — CLAUDE.md Section 20: "don't assume
one strategy is optimal." Screens strategy x symbol x timeframe
combinations (the same vectorbt screening engine as everywhere else),
then tests whether combining several of them actually helps versus just
running the single best one.

- Ranks every screened combination by Sharpe and keeps only the top
  `portfolio.top_n` combinations with at least `portfolio.min_trades`
  trades (reusing Phase 6's `filter_top_candidates()` — the same "avoid
  wasting compute on obviously poor regions" rationale, applied to
  correlation/allocation instead of parameter sweeps).
- Combinations on **different timeframes are never combined** — a bar of
  H1 and a bar of H4 are not the same unit of time. Survivors are grouped
  by timeframe and only the single largest group enters the
  correlation/allocation step.
- `get_bar_returns()` (a new function on `src.backtest.vectorbt_engine`,
  sharing its Portfolio construction with `run_screening()`) gives each
  survivor's per-bar return series; `align_returns()` outer-joins them on
  timestamp (missing bars filled with 0.0 — a stated simplification, not
  a certified risk model) and `correlation_matrix()` reports their
  pairwise correlation.
- Two **basic** allocation schemes (Section 20's own wording — this is
  not a mean-variance/Markowitz solver): `equal_weight` and
  `inverse_volatility` (lower-volatility components get more weight;
  falls back to equal weight if any component's volatility is exactly
  zero, where inverse-vol is undefined).
- `evaluate_returns()` computes Sharpe/Sortino/max-drawdown/return/
  positive-month-ratio directly from the combined return series
  (annualized using a factor inferred from the *actual* median bar
  spacing in the data — never a hard-coded per-timeframe constant), using
  the **same sign convention as vectorbt's own `max_drawdown()`** (zero
  or negative; less negative is a smaller, better drawdown) so
  `compare_to_best_component()` can compare portfolio vs. single-best
  metrics directly — exactly Section 20's ask: "test whether
  diversification improves Sharpe, Sortino, drawdown, and return
  consistency."
- `python main.py portfolio` (optionally filtered by `--strategy
  --symbol --timeframe`, `--top-n`, `--min-trades`) screens every enabled
  combination that matches, reports which had no raw data, and prints
  each survivor's individual metrics, the correlation matrix, and each
  allocation method's weights/metrics/improved-vs-not verdict.

## Broker adapters (Phase 12: MT5; Phase 13: generic REST/WebSocket)

`src/brokers/base.py` — CLAUDE.md Section 21's abstract `BrokerAdapter`:
`connect`/`disconnect`, `get_account`/`get_balance`/`get_equity`,
`get_positions`, `get_symbol_info`, `get_quote`, `place_order`/
`modify_order`/`cancel_order`/`close_position`, `get_order_status`.
Strategy/risk/execution code depends on this interface, never on a
specific broker's SDK — Phase 13's generic REST/WebSocket adapter
implements the same interface.

Two safety mechanisms live in the base class, shared by every adapter
(Section 27): `emergency_stop()`/`reset_emergency_stop()` (an in-process
halt flag checked before any order-sending call) and
`_ensure_order_allowed()`, which blocks `place_order`/`modify_order`/
`cancel_order`/`close_position` whenever `environment == "live"` unless
**both** `LIVE_TRADING=true` and `LIVE_CONFIRMATION=true` are set. A
`"demo"` connection is never gated by those two flags — no real money is
at risk there. `base.py` also holds `reconcile_positions()` (CLAUDE.md
Section 25's "reconcile local positions with broker positions
regularly") — broker-agnostic, comparing two plain `Position` lists by
id, volume and side, so any adapter's `get_positions()` output can be
checked against an execution layer's own bookkeeping (Phase 14) without
either depending on the other's internals.

`src/brokers/mt5_adapter.py` — `MT5Adapter`, the first implementation.
**Environment constraint (Section 22): the `MetaTrader5` Python package
only works on Windows with a running MT5 terminal — it cannot be
installed or imported here.** It's imported lazily (reusing
`src.data.mt5_loader.connect()`/`MT5UnavailableError` from Phase 2's data
ingestion), so this module stays importable and unit-testable on
Linux/macOS with every `MetaTrader5` call mocked — see
`tests/test_mt5_adapter.py`, which stubs `sys.modules["MetaTrader5"]`
with a fake module (namedtuples/namespaces + the real API's constants)
covering every adapter method: connection lifecycle, account/position/
symbol/quote mapping, order placement (market and pending, with a
caller-supplied `client_order_id` mapped deterministically to an MT5
magic number via CRC32 — never Python's own `hash()`, which isn't stable
across runs), modify/cancel/close, order status, the emergency-stop halt,
and the live/demo gating in both directions. **None of this proves the
real MT5 package behaves identically** — only a real terminal on Windows
can (Section 22) — these tests only pin down `MT5Adapter`'s own logic
against whatever `MetaTrader5` hands it back.

`build_from_config()` constructs an `MT5Adapter` from
`config/brokers.yaml`'s `brokers.mt5` block plus `.env` (`MT5_LOGIN`,
`MT5_PASSWORD`, `MT5_SERVER`, `MT5_TERMINAL_PATH`) — credentials never
hard-coded (Sections 22, 24, 42). `brokers.mt5.enabled` is `false` by
default; connecting for real requires enabling it, running on Windows
with `requirements-windows.txt` installed and a terminal open, and
filling in `.env`.

`src/brokers/generic_rest_adapter.py` — `GenericRestAdapter`, the second
implementation (Section 23). No real broker is named in the brief, so
this adapter defines its own small, self-describing REST convention
(`GET /account`, `/positions`, `/symbols/{symbol}`, `/quotes/{symbol}`;
`POST/PATCH/DELETE /orders...`; `POST /positions/{id}/close`) rather than
guessing a specific broker's actual schema — plugging in a real broker
later means translating its responses onto the same shared dataclasses
this adapter already produces; strategy/risk/execution code (which only
ever sees `BrokerAdapter`) needs no changes at all.

Section 25's full API-safety list is implemented, all in
`src/brokers/resilience.py` (broker-agnostic, so any future adapter can
reuse it) plus this adapter's own request path:
- **Rate limiting** — `RateLimiter` (a simple minimum-interval enforcer,
  not a token bucket — a deliberate "basic" choice), applied to every
  request.
- **Retry with exponential backoff** — `retry_with_backoff()` on
  connection errors/timeouts/5xx; every timing source is injectable, so
  tests never actually sleep.
- **Timeout**, **connection recovery** — `requests`' own `timeout=` on
  every call; a fresh `requests.Session` per `connect()`.
- **Duplicate-order protection / idempotency** — a `client_order_id`
  already submitted in-process is refused before any HTTP call, and
  passed to the broker too (Section 25's own "where supported" caveat —
  a generic adapter can't assume every broker honors it).
- **Order-state reconciliation** — `reconcile()`, wrapping
  `reconcile_positions()` above.
- **Broker rejection handling** — HTTP 4xx raises `BrokerRejectionError`,
  distinct from `BrokerConnectionError` ("couldn't trust the broker's
  response at all").
- **Partial-fill handling** — a "FILLED" response with
  `filled_volume` less than requested is downgraded to
  `PARTIALLY_FILLED`.
- **Stale-price detection** — `get_quote()` raises `StalePriceError` if
  the quote is older than `stale_quote_max_age_seconds`.

`stream_quotes()` is a minimal, honestly-scoped WebSocket addition
(Section 23's other half): it proves the adapter can open, subscribe on,
and consume a streaming connection generically (an injectable connector,
default `websockets.connect`) — it is **not** wired into `get_quote()` or
anywhere else. Phase 14's paper trading, as built, replays historical
candles rather than a broker's live feed (there is no live connection in
this sandbox); wiring a real quote stream into a live/paper session that
talks to an actual broker is future work, not something Phase 14 does.

`tests/test_generic_rest_adapter.py` (33 tests) stubs `requests.Session`
with a fake answering from a canned response table, and the WebSocket
half with a fake async connector — this pins down `GenericRestAdapter`'s
own request-building/response-mapping/retry/reconciliation logic, and
says nothing about how any specific real broker's API actually looks,
since none is chosen yet. `build_from_config()` wires
`config/brokers.yaml`'s `brokers.generic_rest` block plus `.env`
(`BROKER_API_KEY`/`BROKER_API_SECRET`/`BROKER_ACCOUNT_ID`) into an
adapter instance; `brokers.generic_rest.enabled` is `false` by default.

## Paper trading (Phase 14)

CLAUDE.md Section 26's pipeline, implemented exactly in this order:

```
Market Data -> Strategy -> Risk Engine -> Virtual Order
  -> Execution Simulator -> Position Manager -> PnL
```

`src/execution/paper_trading.py`'s `run_paper_trading_session()` replays
one strategy/symbol/timeframe's historical candles bar by bar as if they
were arriving live — there is no live broker feed in this sandbox (or
until Phases 12/13's adapters are actually connected in production), so
this is always a simulated session over historical data, never a real
order.

- **One bar of execution delay, deliberately** (Section 11): a signal
  decided from bar *i*'s close is only filled at bar *i+1*'s open —
  matching Backtrader's own next-bar-execution model (Phase 7), never
  the same bar's close a real broker could never fill at.
- **`src/risk/engine.py`'s `RiskEngine`** sits between every signal and
  every virtual order: the max-drawdown kill switch (checked every bar,
  never resets itself once tripped), the consecutive-loss cooldown,
  daily/weekly loss and trade-count limits, max open positions, and an
  approximate margin check (notional / `max_leverage` against equity *
  `max_margin_usage` — true margin needs broker-specific contract size,
  only available once a broker is connected, Section 8). It only
  *evaluates* a size `BaseStrategy.calculate_position_size()` already
  computed — sizing has exactly one formula in the whole codebase.
  `max_correlated_exposure` is deliberately not implemented here: it's a
  cross-*symbol* concept, but this session (like every other engine in
  this codebase) is scoped to one symbol — there's nothing else in scope
  to be correlated with. A portfolio-level exposure check belongs to a
  multi-symbol orchestrator that doesn't exist yet, reusing Phase 11's
  correlation machinery when it does.
- **`src/execution/simulator.py`'s `ExecutionSimulator`** applies
  slippage (always against the trader, direction-aware) and a commission
  that folds in the bar's own real `spread` value the same way
  `src.backtest.backtrader_engine` does, under the same
  `execution.costs[scenario]` every other engine uses.
- **`src/execution/position_manager.py`'s `PositionManager`** detects
  intrabar stop-loss/take-profit touches from a bar's high/low (assuming
  the stop was touched first if both would be in the same bar — the
  conservative, worse-for-the-trader assumption, since there's no
  intrabar tick data to know the true order) and produces `PaperTrade`
  records with exactly Section 28's trade-log fields.
- At most **one open position per session** at a time (matching the
  vectorbt/Backtrader engines' own non-stacking convention); a signal
  reversal closes the old position and opens the new one at the same
  bar's open. A position still open when the data runs out is reported
  as still open, mark-to-market — never force-closed into a synthetic
  trade that didn't happen.
- `python main.py paper-trade --strategy --symbol --timeframe
  [--scenario] [--capital]` prints the full session report: final
  equity, every closed trade, win rate, total PnL, rejected-signal count,
  and whether the kill switch tripped.

The main efficiency choice worth calling out: `generate_signals_vectorized()`
is computed **once** for the whole series up front (exactly the table
`run_screening()` reads) rather than re-calling `generate_signal()` on a
growing slice every bar — since that function is purely causal (rolling
windows, never a future row), reading row *i* of one whole-series
computation is provably identical to recomputing it fresh on
`df.iloc[:i+1]` every iteration, just without turning the loop from O(n)
into O(n²) for zero behavioral difference.

## Risk & kill switch (Phase 15)

Phase 14's `RiskEngine` already had to implement most of CLAUDE.md
Section 7's checks — paper trading's own pipeline requires a Risk Engine
step. Phase 15 formalizes the one piece that mattered most and wasn't
durable yet: the hard kill switch.

- **`src/risk/kill_switch.py`'s `KillSwitch`** is an append-only audit
  log (`kill_switch_events`, never a mutable singleton row) of every
  trip and reset. `RiskEngine` now reads it once at construction — a
  brand-new `RiskEngine` (a brand-new paper-trading session, potentially
  in a brand-new process) starts **already triggered** if *any* earlier
  session recorded a trip against the same database. That's what "hard"
  means in Section 7: one session's drawdown breach has to stop every
  other session too, not just itself. A trip logs to the system log
  (Section 28's "risk violation, kill switch events") and, if broker
  adapters were registered with the `RiskEngine`, calls their
  `emergency_stop()` (Section 27) — empty by default, since paper
  trading replays historical data and holds no real broker connection.
- **Reset is always an explicit human action** — `KillSwitch.reset()`
  requires a note explaining why, recorded in the same audit trail as
  the trip; nothing resets itself.
- **`python main.py kill-switch`** shows the current status and recent
  history; `--reset "<note>"` records a reset. `RiskEngine.evaluate()`
  also logs every rejection to the system log (Section 28's "risk
  violation").
- **`src/risk/correlated_exposure.py`'s `CorrelatedExposureMonitor`**
  implements Section 7's `max_correlated_exposure` — deliberately kept
  *outside* `RiskEngine`, since every engine in this codebase (including
  RiskEngine) is scoped to one symbol per call, and there's nothing to
  be "correlated" with inside a single-symbol session. It's a standalone
  utility for a multi-symbol orchestrator that doesn't exist yet: given
  each open position's risk amount by symbol and a symbol x symbol
  correlation matrix (the same shape Phase 11's
  `correlation_matrix()` produces), it caps combined risk (not raw
  notional — that scales with leverage and would make the same
  percentage mean different things account to account) across symbols
  correlated above a threshold at `max_correlated_exposure` of equity.

## CLI

```bash
python main.py info            # environment/config summary (implemented)
python main.py download-data   # ingest historical data (implemented, Phase 2)
python main.py download-data --symbol EURUSD --timeframe H1
python main.py validate-data   # data-quality report (implemented, Phase 3)
python main.py validate-data --symbol EURUSD --timeframe H1
python main.py backtest        # vectorbt screen + Backtrader validation (implemented, Phases 6-7)
python main.py backtest --strategy trend_following --symbol EURUSD --timeframe H1 --scenario stress
python main.py optimize --strategy trend_following --symbol EURUSD --timeframe H1  # implemented, Phase 8
python main.py optimize --strategy mean_reversion --symbol EURUSD --timeframe H1 --trials 100 --scenario stress
python main.py walk-forward --strategy trend_following --symbol EURUSD --timeframe H1  # implemented, Phase 9
python main.py walk-forward --strategy trend_following --symbol EURUSD --timeframe H1 --window-bars 600 --step-bars 300
python main.py monte-carlo --strategy trend_following --symbol EURUSD --timeframe H1  # implemented, Phase 10
python main.py monte-carlo --strategy trend_following --symbol EURUSD --timeframe H1 --simulations 5000 --scenario stress
python main.py portfolio        # implemented, Phase 11 -- all enabled combinations
python main.py portfolio --timeframe H1 --top-n 5 --min-trades 20
python main.py paper-trade --strategy trend_following --symbol EURUSD --timeframe H1  # implemented, Phase 14
python main.py paper-trade --strategy trend_following --symbol EURUSD --timeframe H1 --capital 5000 --scenario stress
python main.py kill-switch     # implemented, Phase 15 -- status + history
python main.py kill-switch --reset "reviewed manually, resuming"
python main.py live            # not yet implemented (requires LIVE_TRADING=true AND LIVE_CONFIRMATION=true)
python main.py report          # Phase 16
```

Commands not yet implemented exit with a non-zero status and say which
phase will add them, rather than silently doing nothing.

## Testing

```bash
pytest
```

## Live trading safety

Live orders are only ever sent if **both** `LIVE_TRADING=true` and
`LIVE_CONFIRMATION=true` are set in `.env`. Either being false/unset keeps
the system in paper/dry-run mode. See `src/core/config.py:is_live_trading_enabled`.
Every `BrokerAdapter` (Phase 12/13) additionally refuses to send an order
on an `environment="live"` connection under the same two-flag rule
(`_ensure_order_allowed()`), and the durable kill switch (Phase 15) can
call `emergency_stop()` on any registered adapter the moment a drawdown
breach is recorded — see "Risk & kill switch" above.

## Project structure

```
quant_trader/
├── config/            # settings.yaml, strategies.yaml, brokers.yaml
├── data/{raw,processed,cache}/
├── src/
│   ├── core/          # config loading, logging
│   ├── data/ features/ strategies/ risk/ backtest/ optimization/
│   ├── walkforward/ montecarlo/ portfolio/ execution/ brokers/ reporting/
├── tests/
├── notebooks/ reports/ logs/ scripts/
├── main.py
├── requirements.txt / requirements-windows.txt
├── .env.example
└── CLAUDE.md
```
