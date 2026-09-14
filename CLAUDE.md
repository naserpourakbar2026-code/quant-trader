# Project Brief: Forex CFD Quant Research & Execution Framework
### (Optimized for use as a Claude Code project prompt / CLAUDE.md)

## Role

You are acting as a senior Quant Developer, Quant Researcher, Algorithmic Trader,
Python Architect, and Risk Engineer.

## Objective

Build a complete, production-oriented **quantitative trading research and execution
framework** for Forex CFD trading. This is **not** a simple trading bot.

The pipeline the system must support:

```
Historical Data
  → Cleaning & Validation
  → Feature Engineering
  → Strategy Generation
  → Backtesting
  → Optimization
  → Walk-Forward Analysis
  → Monte Carlo Analysis
  → Robustness Testing
  → Portfolio Analysis
  → Paper Trading
  → Broker API Execution
```

**Primary goal:** discover strategies with a realistic probability of remaining
profitable out-of-sample — not strategies with the highest historical backtest
profit. **Never claim guaranteed profitability.**

---

## ⚠️ How to use this document with Claude Code

This brief describes the *entire* system, but Claude Code should **not** attempt
to build all of it in one uninterrupted run. Work phase by phase (see Section 18):

1. Save this file as `CLAUDE.md` at the project root so every session has context.
2. Give Claude Code one phase at a time (e.g. "Implement Phase 3 now").
3. After each phase: run the test suite, review the diff/output, `git commit`,
   then move to the next phase. Don't let multiple phases run unattended.
4. Use a TODO list (Claude Code's task tracking) to track phase progress across
   sessions, since context will not persist automatically between sessions.
5. Treat Section 41 (Claude Code operating rules) as always-active instructions,
   not just a one-time phase.

---

## 1. Technology Stack

Python 3.11+.

Core libraries:
- pandas, numpy, scipy
- **vectorbt** (open-source edition — confirm this, not vectorbt PRO, unless you have a license) for fast large-scale parameter research/screening
- **Backtrader** for event-driven validation and realistic execution simulation
- Optuna (parameter optimization)
- scikit-learn / statsmodels where statistically justified
- matplotlib, plotly (reporting)
- pydantic, PyYAML (config/validation)
- SQLAlchemy + SQLite (persistence)
- requests, websockets
- `MetaTrader5` Python package — **Windows-only** (wraps the MT5 terminal's native DLLs). It will not install or run inside a Linux/macOS dev sandbox. See Section 22 for how to handle this.
- pytest, loguru

Do not rely exclusively on vectorbt or Backtrader — the architecture must allow
swapping either engine later.

---

## 2. Data Sources

Support two historical-data sources:
- **A)** MetaTrader 5 (only usable from a Windows environment with a running MT5 terminal)
- **B)** CSV

Minimum OHLCV fields: `timestamp, open, high, low, close, volume`

Timeframes: M5, M15, M30, H1, H4

Markets: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD, EURJPY, GBPJPY
Optional: XAUUSD (keep results **separate** from Forex results — different volatility regime).

Standardized internal schema regardless of source:
```
timestamp, symbol, timeframe, open, high, low, close, tick_volume, spread, real_volume
```

---

## 3. Data Quality

Build a validation pipeline that **detects but never silently repairs**:
- duplicate timestamps / duplicate rows
- missing candles, out-of-order timestamps
- impossible OHLC values, negative or zero prices
- abnormal spreads
- timezone inconsistencies
- weekend data, broker-specific gaps

Generate a data-quality report per symbol/timeframe:
```
DATA QUALITY REPORT
Symbol: EURUSD | Timeframe: H1
Rows: ... | Missing candles: ... | Duplicates: ...
Invalid candles: ... | Avg spread: ... | Max spread: ... | Date range: ...
```

---

## 4. Project Structure

```
quant_trader/
├── config/
│   ├── settings.yaml
│   ├── strategies.yaml
│   └── brokers.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── cache/
├── src/
│   ├── data/
│   ├── features/
│   ├── strategies/
│   ├── risk/
│   ├── backtest/
│   ├── optimization/
│   ├── walkforward/
│   ├── montecarlo/
│   ├── portfolio/
│   ├── execution/
│   ├── brokers/
│   └── reporting/
├── tests/
├── notebooks/
├── reports/
├── logs/
├── scripts/
├── main.py
├── requirements.txt
├── .env.example
├── README.md
├── CLAUDE.md          ← this brief, kept in the repo
└── pyproject.toml
```

Keep research code separate from live execution code.

---

## 5. Strategies

Implement **at least three independent strategy families** (not randomly combined indicators):

**A. Trend Following / Breakout** — EMA trend structure, ATR, Donchian-style breakout,
momentum, volatility filter, market structure. Must classify Trend / No Trend / Weak
Trend / Strong Trend **objectively** (rule-based, not subjective).

**B. Mean Reversion** — Bollinger Bands, RSI, distance from moving average, ATR,
volatility regime, range detection. Must avoid aggressively trading against strong trends.

**C. Momentum / Multi-Factor** — only factors that can be statistically justified:
momentum, trend strength, volatility, price structure, breakout strength, volume/tick
volume if reliable. Don't add indicators just because they're popular.

---

## 6. Strategy Interface

Every strategy implements a standard interface:
```python
generate_signal()
calculate_stop_loss()
calculate_take_profit()
calculate_position_size()
validate_signal()
```
Returns structured signals: `LONG | SHORT | FLAT` with
`entry_price, stop_loss, take_profit, confidence, strategy_name, timestamp, symbol, timeframe`.

---

## 7. Risk Management

- Initial account: **€2,000**
- Default risk per trade: **0.5%** (configurable: 0.25% / 0.5% / 0.75% / 1.0%)
- Position sizing: percentage-risk based, ATR-based
- Limits: max daily loss, max weekly loss, max portfolio drawdown, max open positions,
  max correlated exposure, max leverage, max margin usage, daily trade limit,
  cooldown after consecutive losses
- **Hard risk kill switch**: if equity drawdown ≥ configured max → stop new trades

---

## 8. Leverage

Leverage is a broker/account constraint, **never** chosen to increase profit.
Position size derives from: Account Equity, Risk %, Stop Distance, Contract Size,
Tick/Pip Value, Broker Specifications. Then verify required margin vs. available
margin vs. max allowed exposure.

---

## 9–10. Stop Loss & Take Profit

- Stop loss types to implement/compare: ATR stop, swing-based stop, fixed % stop, volatility stop
- Take profit ratios to test: 1:1, 1:1.5, 1:2, 1:3, plus ATR target, trailing stop,
  break-even, partial exit
- Do not assume higher R:R is automatically better — test it.

---

## 11. Realistic Execution

Backtests must model: spread, commission, slippage, swap, execution delay.
Run three scenarios — **OPTIMISTIC / REALISTIC / STRESS**. A strategy is only
"robust" if it holds up under REALISTIC assumptions.

---

## 12. vectorbt Research Engine

Use vectorbt for rapid parameter sweeps, multi-symbol/multi-timeframe screening.
Avoid wasting compute on obviously poor parameter regions. Every experiment gets
a unique ID and is persisted with: `experiment_id, strategy, symbol, timeframe,
parameters, date_range, metrics, data_version, code_version`.

---

## 13. Backtrader Validation

After vectorbt screens candidates, validate the best ones with Backtrader
(event-driven, next-bar execution) modeling: spread, commission, slippage,
order rejection, insufficient margin, SL, TP. This catches unrealistic results
from vectorized backtest assumptions.

---

## 14. Optuna Optimization

Optimize only parameters with a logical economic/trading justification (EMA
periods, ATR period/multiplier, breakout period, RSI period/threshold, Bollinger
period/deviation, momentum period, stop/TP multipliers). Don't optimize dozens of
parameters at once.

Composite risk-adjusted objective (not raw net profit):
`Profit Factor + Sharpe + Sortino + Expectancy`, penalized by max drawdown, low
trade count, and high parameter sensitivity.

---

## 15. Walk-Forward Analysis

Rolling WFA: Training 60% / Validation 20% / Out-of-Sample 20% (also support
rolling windows). Per window: optimize on training → pick a robust parameter
region → validate → freeze parameters → test on unseen OOS data → move window
forward. **Never optimize using OOS data.** Generate a walk-forward report.

---

## 16. Parameter Stability

Never select a single "best" parameter — search for stable regions:
```
EMA 20 → poor   EMA 30 → good   EMA 35 → good
EMA 40 → good   EMA 45 → good   EMA 60 → poor
```
is more robust than an isolated spike (`EMA 37 → extraordinary`, neighbors
terrible). Build a Parameter Stability Score.

---

## 17. Monte Carlo

For top strategies, run ≥1000 simulations randomizing trade sequence, trade
returns, slippage, execution variation, losing streaks. Estimate: median return,
5th/95th percentile return, worst drawdown, 95% drawdown, max losing streak,
probability of ruin, probability of negative return. Fragile-under-MC strategies
are not "robust."

---

## 18. Out-of-Sample Requirement

High backtest return alone is never sufficient. Evaluate: OOS profitability,
Profit Factor, Drawdown, Sharpe, Sortino, Expectancy, trade count, parameter
stability, Monte Carlo results, cost sensitivity. **If a strategy fails OOS, reject it.**

---

## 19. Market Regime Analysis

Classify: Strong Trend / Weak Trend / Range / High Volatility / Low Volatility.
Report Return, Drawdown, PF, Sharpe, Trade Count per regime to identify where
each strategy works and where it fails.

---

## 20. Portfolio Engine

Don't assume one strategy is optimal. Analyze Strategy × Symbol × Timeframe
combinations, compute correlations, and test whether diversification improves
Sharpe, Sortino, drawdown, and return consistency. Implement basic allocation.

---

## 21. Broker Adapter Architecture

Strategy code must **never** directly depend on a broker. Define an abstract
`BrokerAdapter`:
```python
connect(); disconnect(); get_account(); get_balance(); get_equity()
get_positions(); get_symbol_info(); get_quote()
place_order(); modify_order(); cancel_order(); close_position(); get_order_status()
```

---

## 22. MT5 Broker Adapter — ⚠️ environment constraint

The official `MetaTrader5` Python package only functions on **Windows**, connected
to a running MT5 terminal — it cannot be installed or executed in a Linux/macOS
Claude Code sandbox. Recommended approach:

- Write the adapter code and its unit tests (with the MT5 calls mocked/stubbed)
  in this environment.
- Actually connect, authenticate, and place paper/live orders only when this
  code is run on a Windows machine with MT5 installed.
- Default mode is always **PAPER / DRY RUN**. Live trading requires an explicit
  `LIVE_TRADING=true` flag (never on by default).

Support: connection, account info, symbol info, quotes, market/pending orders,
SL/TP, position retrieval, order modification, position closing.

---

## 23. Generic REST / WebSocket Broker Adapter

An abstract adapter so a REST/WebSocket broker can be plugged in later:
Authentication, Account, Market Data, Orders, Positions, Risk Checks, Error
Handling, Rate Limiting, Retries, Connection Recovery. No broker hard-coded
into strategy logic.

---

## 24. Broker Configuration

`brokers.yaml`:
```yaml
broker:
  name:
  environment:
  api_base_url:
  websocket_url:
  account_id:
  rate_limit:
  commission:
  spread_model:
  slippage_model:
```
Secrets go in `.env` (never in source or `brokers.yaml` itself).

---

## 25. API Safety

Rate limiter, retry with exponential backoff, timeout, connection recovery,
duplicate-order protection, idempotency where supported, order-state
reconciliation, broker rejection handling, partial-fill handling, stale-price
detection. Reconcile local positions with broker positions regularly.

---

## 26. Paper Trading

```
Market Data → Strategy → Risk Engine → Virtual Order
  → Execution Simulator → Position Manager → PnL
```
Should behave as close to live execution as possible.

---

## 27. Live Trading Safety

Live mode requires **both** `LIVE_TRADING=true` **and** `LIVE_CONFIRMATION=true`.
If either is false → no real order is ever sent. Implement `emergency_stop()`
to block all new orders instantly.

---

## 28. Logging

**Trade log:** timestamp, symbol, strategy, signal, entry, SL, TP, size, risk,
spread, slippage, exit, PnL, R multiple, reason.

**System log:** API connection, errors, retries, order rejection, position
mismatch, risk violation, kill switch events.

---

## 29. Database

SQLite initially, via SQLAlchemy (so migration to PostgreSQL later is easy).
Store: candle metadata, experiments, backtests, optimization results,
walk-forward results, Monte Carlo results, signals, orders, positions, trades,
equity curve.

---

## 30. Reporting

Outputs: HTML reports, CSV exports, JSON results.
Charts: Equity Curve, Drawdown, Monthly Returns, Rolling Sharpe, Trade
Distribution, Parameter Heatmap, Walk-Forward Performance, Monte Carlo
Distribution, Strategy Correlation.

---

## 31. Final Robustness Score (0–100)

Suggested weighting (adjustable if justified):
| Factor | Weight |
|---|---|
| OOS performance | 20% |
| Drawdown | 20% |
| Profit Factor | 15% |
| Sharpe/Sortino | 15% |
| Parameter Stability | 10% |
| Monte Carlo robustness | 10% |
| Cost sensitivity | 5% |
| Trade count / statistical reliability | 5% |

No single metric should dominate the score.

---

## 32. Strategy Selection

Produce 🥇 Best / 🥈 Second / 🥉 Third robust strategy — **only** if they pass
minimum robustness criteria. If none passes: **"NO ROBUST STRATEGY FOUND"** is
an acceptable and expected outcome.

---

## 33. Capital Simulation

Simulate €2,000 initial capital across risk levels (0.25% / 0.5% / 0.75% / 1%).
Report: equity curve, max drawdown in EUR, max losing streak, risk of ruin,
margin utilization, average/worst/best monthly return. Never assume historical
returns repeat.

---

## 34. Transaction Cost Stress Test

Re-run with normal / 1.5x / 2x / 3x spread and varying slippage. If
profitability disappears with a small cost increase → mark **COST FRAGILE**.

---

## 35. Randomization Tests

Random trade order, randomized entry delay, increased spread/slippage,
parameter perturbation, slightly altered indicator periods. A robust strategy
should not collapse from small changes.

---

## 36. Reproducibility

Every experiment records: git commit hash (if available), data version,
configuration, parameters, random seed, Python version, library versions.
Results must be reproducible from these records.

---

## 37. Testing

pytest suite covering at minimum: data validation, indicator calculation,
signal generation, position sizing, risk limits, stop loss, take profit, order
creation, duplicate-order protection, broker connection, API retry, rate
limit, drawdown kill switch.

---

## 38. CLI

```
python main.py download-data
python main.py validate-data
python main.py backtest
python main.py optimize
python main.py walk-forward
python main.py monte-carlo
python main.py portfolio
python main.py paper-trade
python main.py live
python main.py report
```

---

## 39. Configuration

No hard-coded important values — use YAML:
```yaml
initial_capital: 2000
risk_per_trade: 0.005
max_drawdown: 0.15
max_daily_loss: 0.03
live_trading: false
symbols: [EURUSD, GBPUSD, USDJPY]
timeframes: [M15, H1, H4]
```

---

## 40. Development Process — Phased Delivery

**Do not generate the whole project in one uninterrupted response.** Work one
phase at a time, and stop for review between phases (see the Claude Code usage
note at the top of this document):

| Phase | Scope |
|---|---|
| 1 | Architecture + environment setup |
| 2 | Data ingestion |
| 3 | Data validation |
| 4 | Feature engine |
| 5 | Three strategies |
| 6 | vectorbt research engine |
| 7 | Backtrader validation engine |
| 8 | Optuna optimization |
| 9 | Walk-Forward |
| 10 | Monte Carlo |
| 11 | Portfolio engine |
| 12 | MT5 adapter (code + mocked tests only — see Section 22) |
| 13 | Generic broker API adapter |
| 14 | Paper trading |
| 15 | Risk & kill switch |
| 16 | Reporting |
| 17 | Full integration tests |
| 18 | Final robustness evaluation |

---

## 41. Claude Code Operating Rules (always active)

Before writing code:
1. Inspect the existing repository and files.
2. Do not overwrite useful existing work.
3. Reuse existing components where appropriate.
4. State a clear implementation plan before implementing (use a TODO list).

After each major phase:
1. Run the test suite.
2. Inspect and fix any errors.
3. Re-run tests.
4. Only proceed once the current phase is functional.
5. Commit the phase to git with a descriptive message.

Never leave important functionality as `TODO` / `pass` / `NotImplementedError` /
placeholder — **unless** it genuinely requires credentials or broker-specific
information that can't be invented. For broker credentials, build a safe
configuration interface but never fabricate credentials.

---

## 42. Security

Never expose or commit: API keys, passwords, broker credentials, account
numbers. Use `.env` + `.env.example`, and ensure `.env` is in `.gitignore`.

---

## 43. Final Deliverable

At project completion, provide:
1. Complete project structure
2. Installation instructions
3. Environment setup
4. Data acquisition instructions
5. Backtest / optimize / walk-forward / Monte Carlo / paper-trade commands
6. MT5 connection instructions (for use on a Windows machine)
7. Broker adapter instructions
8. Report generation instructions
9. Testing instructions

Plus a final results table:

| Strategy | Symbol | Timeframe | Net Return | CAGR | Profit Factor | Win Rate | Expectancy | Max DD | Sharpe | Sortino | Trades | OOS Return | MC DD | Robustness Score | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

Status ∈ `{PASS, FAIL, COST FRAGILE, OVERFIT, INSUFFICIENT DATA, NOT ROBUST}`

---

## 44. Final Rule

The objective is **not** "find the strategy with the highest backtest profit."

The objective is: *"Find the strategy with the strongest combination of
profitability, risk-adjusted return, statistical reliability, out-of-sample
performance, parameter stability, execution realism, and robustness."*

If the evidence does not support profitability, **say so**. Do not manufacture
a successful strategy, optimize until the backtest merely "looks good," use
future information, use OOS data during optimization, or claim guaranteed profits.

---

**Start with Phase 1: inspect the repository and existing environment, then
propose the architecture and implementation plan before writing any production code.**
