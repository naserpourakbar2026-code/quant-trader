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
| 8     | Optuna optimization                             | pending |
| 9     | Walk-forward                                    | pending |
| 10    | Monte Carlo                                     | pending |
| 11    | Portfolio engine                                | pending |
| 12    | MT5 adapter (code + mocked tests, Windows-only) | pending |
| 13    | Generic broker API adapter                      | pending |
| 14    | Paper trading                                   | pending |
| 15    | Risk & kill switch                              | pending |
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
"backtrader") so the two can be queried and compared side by side. More
tables arrive with the phases that need them (optimization/walk-forward/
Monte Carlo results, ...).

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

## CLI

```bash
python main.py info            # environment/config summary (implemented)
python main.py download-data   # ingest historical data (implemented, Phase 2)
python main.py download-data --symbol EURUSD --timeframe H1
python main.py validate-data   # data-quality report (implemented, Phase 3)
python main.py validate-data --symbol EURUSD --timeframe H1
python main.py backtest        # vectorbt screen + Backtrader validation (implemented, Phases 6-7)
python main.py backtest --strategy trend_following --symbol EURUSD --timeframe H1 --scenario stress
python main.py optimize        # Phase 8
python main.py walk-forward    # Phase 9
python main.py monte-carlo     # Phase 10
python main.py portfolio       # Phase 11
python main.py paper-trade     # Phase 14
python main.py live            # Phase 14 (requires LIVE_TRADING=true AND LIVE_CONFIRMATION=true)
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
