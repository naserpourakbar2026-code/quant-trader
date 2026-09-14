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
| 2     | Data ingestion                                  | pending |
| 3     | Data validation                                 | pending |
| 4     | Feature engine                                  | pending |
| 5     | Three strategies                                | pending |
| 6     | vectorbt research engine                        | pending |
| 7     | Backtrader validation engine                    | pending |
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
- `config/strategies.yaml` — strategy registry and parameters (populated in
  Phase 5).
- `config/brokers.yaml` — non-secret broker connection settings. Secrets
  (API keys, MT5 credentials) go in `.env`, never in this file or in source.

Config is loaded and validated through pydantic models in
`src/core/config.py`.

## CLI

```bash
python main.py info            # environment/config summary (implemented)
python main.py download-data   # Phase 2
python main.py validate-data   # Phase 3
python main.py backtest        # Phase 7
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
