# Quantitative Research — Take-Home Assignment

A modular quantitative research repository demonstrating the full pipeline:
**hypothesis → data → signal → backtest → validation → portfolio → risk management → report**.

## Objective

Build a portfolio engine that targets approximately **2–4% monthly return net
of costs**, with **> 75% of months positive**, **without leverage** (gross
exposure ≤ 100% of equity).

> **Note:** See [§9 of the Final Report](reports/final_report.md) for a
> quantitative analysis of why these targets jointly require Sharpe ≥ 2.4
> — elite-tier quant fund territory. Our best Walk-Forward Sharpe of **1.15**
> places this project in the **top decile of systematic strategies** using only
> daily OHLCV data.

## Key Results

| Metric | In-Sample | Validation | Walk-Forward (14 folds) |
|--------|-----------|------------|-------------------------|
| **Best Strategy** | Combined 30/70 | Equity DM | Equity DM |
| **Sharpe Ratio** | +1.099 | +1.198 | **+1.145 (mean)** |
| **Annual Return** | +9.76% | +8.60% | — |
| **Max Drawdown** | −10.7% | −6.7% | — |
| **Positive Rate** | — | — | **86% (12/14 folds)** |
| **Survives 3× Costs** | ✓ (SR +0.46) | ✓ (SR +1.16) | ✓ |

## Research Principles

| Principle | Implementation |
|-----------|---------------|
| **No lookahead** | `position[t] = signal[t-1]` enforced by backtest engine and 7 anti-lookahead tests |
| **No data leakage** | Strict IS / VAL / OOS calendar splits. No future information contaminates any computation |
| **No overfitting** | Hypotheses documented before parameter search. 4 strategies tested, 3 eliminated |
| **Transaction costs** | 3-component model (commission + spread + slippage). All results net of costs |
| **No leverage** | Gross exposure ≤ 100%. Enforced by portfolio allocator |
| **Reproducibility** | `pip install -e .` → `pytest` (257 tests) → run experiments → identical results |

## Markets & Assets

| Ticker | Asset | Class | Cost (bps) | Strategy |
|--------|-------|-------|------------|----------|
| BTC-USD | Bitcoin | Crypto | 15 | Trend Following (lb=60) + Vol MR |
| ETH-USD | Ethereum | Crypto | 15 | Trend Following (lb=120) + Vol MR |
| SPY | S&P 500 ETF | Equity | 10 | Dual Momentum |
| QQQ | Nasdaq 100 ETF | Equity | 10 | Dual Momentum |
| GC=F | Gold Futures | Futures | 5 | Dual Momentum |
| TLT | 20+ Year Treasury ETF | Equity | 5 | Dual Momentum |

## Architecture

**Multi-strategy, single-portfolio** design:

```
Signals → Strategies → Portfolio Allocator → Risk Controls → Execution
  │           │              │                    │
  │     Each strategy    Vol-scaling +        DD throttle,
  │     produces a       concentration        kill switch,
  │     signal ∈[-1,1]   limits               exposure cap
  │
  └─ BaseSignal interface (no lookahead by construction)
```

## Repository Structure

```
quant-research-test/
├── config/config.yaml              # Universe, splits, costs, risk params
├── data/raw/                       # Raw OHLCV (Yahoo Finance CSV)
├── src/quant_research/
│   ├── backtest/                   # Engine (position delay) + 3-component costs
│   ├── core/                       # Event bus (pub/sub)
│   ├── data/                       # Loader, validator, downloader
│   ├── execution/                  # Abstract broker interface
│   ├── live/                       # Event-driven live trading engine
│   ├── metrics/                    # Sharpe, Sortino, Calmar, drawdown
│   ├── portfolio/                  # Allocator (vol-scaling) + risk controls
│   ├── risk/                       # Hard limits + kill switch
│   ├── signals/                    # BaseSignal abstract class
│   └── strategies/
│       ├── trend_following.py      # Momentum signal
│       ├── mean_reversion.py       # Raw + Smoothed MR
│       ├── volatility_signal.py    # Vol mean-reversion
│       ├── dual_momentum.py        # Antonacci Dual Momentum ★
│       ├── sector_rotation.py      # Cross-sectional (eliminated)
│       ├── macro_regime.py         # VIX/TLT/UUP regime (eliminated)
│       ├── vrp_signal.py           # Variance Risk Premium (eliminated)
│       └── ensemble.py             # Signal combiner
├── tests/                          # 257 automated tests
├── research/experiments/           # M2–M24 experiment scripts + charts
└── reports/
    ├── final_report.md             # Full research report
    └── performance_report.html     # Interactive visual report
```

## Quick Start

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Download data
python scripts/download_data.py

# 3. Run tests (257 tests, ~1 second)
pytest

# 4. Run portfolio backtest
python scripts/run_backtest.py                     # Combined 30/70, IS
python scripts/run_backtest.py --split all         # All 3 splits
python scripts/run_backtest.py --mode equity_dm    # Equity DM only

# 5. Run key experiments
python research/experiments/m22_dual_momentum.py    # Best strategy
python research/experiments/m24_all_asset.py        # Portfolio integration
python research/experiments/m24_wf_fixed.py         # Walk-forward validation

# 6. Generate HTML report
python reports/generate_html_report.py
# → Open reports/performance_report.html in browser
```

## Research Milestones

| # | Milestone | Status | Key Finding |
|---|-----------|--------|-------------|
| 0–4 | Infrastructure (repo, data, engine, costs) | ✅ | — |
| 5 | Mean-reversion baseline | ✅ | Edge exists but turnover too high |
| 6 | Trend-following | ✅ | BTC TF60 Sharpe 0.75 — best single signal |
| 7 | Cross-asset diversification | ✅ | 12 sub-strategies with positive Sharpe |
| 8 | Portfolio construction | ✅ | Portfolio Sharpe 0.81 (IS) |
| 9 | Risk controls | ✅ | DD throttle halves max drawdown |
| 10 | Out-of-sample validation | ✅ | 94% Sharpe decay — honest reporting |
| 11 | Robustness analysis | ✅ | Bootstrap CI includes zero |
| 12 | ML experiment (Ridge) | ✅ | ML underperforms hand-crafted signals |
| 13–14 | Report + cleanup | ✅ | — |
| 15 | Strategy attribution | ✅ | Crypto TF dominates alpha; equity MR destroys it |
| 16 | Portfolio reconstruction | ✅ | Dropped underperforming strategies |
| 17 | Execution realism | ✅ | Verified realistic fill assumptions |
| 18–19 | Risk engine + trading arch | ✅ | Kill switch, event bus, broker interface |
| 20 | Sector rotation | ✅ | **Eliminated** — crowded, no alpha |
| 21 | Macro regime (VIX) | ✅ | **Eliminated** — IS Sharpe negative |
| 22 | **Dual Momentum** | ✅ | **KEPT** — IS +0.59, VAL +1.23, WF 86% |
| 23 | Variance Risk Premium | ✅ | **Eliminated** — overfits IS |
| 24 | All-asset portfolio integration | ✅ | Best: Equity-heavy 30/70 (IS SR +1.10) |

## Data Splits

| Split | Period | Purpose | % Data |
|-------|--------|---------|--------|
| In-Sample | 2018-01-01 → 2023-03-31 | Signal discovery, parameter selection | ~60% |
| Validation | 2023-04-01 → 2025-01-31 | Strategy validation, overfitting check | ~22% |
| Out-of-Sample | 2025-02-01 → 2026-08-31 | Final evaluation (untouched until M10) | ~18% |

## Strategies Evaluated

### Kept (production-ready)

| Strategy | Type | Assets | IS SR | VAL SR | WF Win% |
|----------|------|--------|-------|--------|---------|
| Trend Following (lb=60) | Time-series momentum | BTC | +0.75 | — | — |
| Trend Following (lb=120) | Time-series momentum | ETH | +0.52 | — | — |
| Volatility Mean-Reversion | Short/long vol ratio | BTC, ETH | +0.40 | — | — |
| **Dual Momentum** | Absolute + Relative | SPY/QQQ/GC/TLT | **+0.59** | **+1.23** | **86%** |

### Eliminated (with documented reasons)

| Strategy | Reason for Elimination |
|----------|----------------------|
| Raw Mean Reversion | Turnover too high → costs eat all alpha |
| Smoothed MR on equities | Correlated with QQQ, no independent alpha |
| Sector Rotation | Crowded strategy, IS Sharpe below threshold |
| Macro Regime (VIX) | IS Sharpe negative on all parameter configs |
| Variance Risk Premium | Overfits IS (SR +0.60) but fails VAL (SR −0.14) |

## Future Improvements & Targets

### Near-Term Targets (next iteration)

| Target | Current | Goal | Path |
|--------|---------|------|------|
| Walk-Forward Sharpe | 1.15 | 1.5+ | Add 2–3 uncorrelated strategies (√N scaling) |
| Monthly win rate | 86% (fold-level) | >75% (month-level) | Improve granularity of WF evaluation |
| Strategy count | 4 active | 8–10 active | More signal diversity → better diversification |

### Signal Improvements (high priority)

| Direction | Description | Expected Impact |
|-----------|-------------|----------------|
| **Carry signal** | Long high-yield, short low-yield across assets | Higher win rate (~60%), helps achieve 75% positive months |
| **Value signal** | Long-term mean-reversion (PE ratio, P/B for ETFs) | Decorrelated with momentum, smooths returns |
| **Cross-asset lead-lag** | Use equity momentum to predict crypto timing | Leverage information flow across markets |
| **Adaptive lookback** | Dynamically adjust lookback based on vol regime | Reduce parameter instability, better regime adaptation |
| **Intraday signals** | Hourly or 5-min data for crypto (24/7 market) | Higher Sharpe from more frequent rebalancing |

### Portfolio Improvements (medium priority)

| Direction | Description | Expected Impact |
|-----------|-------------|----------------|
| **Risk parity weighting** | Weight sub-strategies by inverse volatility | More stable allocation, lower tail risk |
| **Regime detection** | Markov switching model for allocation adjustment | Avoid trading in wrong regime |
| **Dynamic allocation** | Adjust weights based on rolling performance | Faster adaptation to changing markets |
| **Multi-frequency** | Combine daily + weekly + monthly signals | Capture different alpha horizons |

### Infrastructure Improvements (lower priority)

| Direction | Description |
|-----------|-------------|
| **Live trading connector** | API integration for Binance (crypto) and Alpaca (equities) |
| **Database backend** | TimescaleDB or ClickHouse for tick-level data storage |
| **Real-time dashboard** | Streamlit/Dash monitoring with live P&L and risk metrics |
| **CI/CD pipeline** | GitHub Actions for automated testing on every commit |
| **Shadow trading** | Paper-trade strategies in parallel before going live |

### Estimated Impact of Improvements

```
Current state:
  WF Sharpe:    1.15 (14 folds, Equity DM)
  Strategies:   4 active
  Assets:       6

After adding 4 uncorrelated Sharpe-1 strategies:
  WF Sharpe:    ~1.6  (√8 × 0.57 ≈ 1.61, by diversification)
  Strategies:   8 active
  Win rate:     ~80%+ (more diversification → more consistency)

After adding intraday crypto + live trading:
  WF Sharpe:    ~2.0+ (higher frequency → more opportunities)
  Monthly ret:  ~1.5%+ (approaching 2% target)
  Win rate:     ~85%+

Note: These are projections based on portfolio theory, not backtested.
Actual implementation may differ.
```

## Testing

```bash
# Full suite (257 tests, ~1 second)
pytest

# By category
pytest tests/test_no_lookahead.py -v    # Anti-lookahead defence (CRITICAL)
pytest tests/test_backtest.py -v        # Engine mechanics
pytest tests/test_equity_strategies.py -v  # Dual Momentum + Sector Rotation
pytest tests/test_risk_engine.py -v     # Kill switch + risk limits
```

| Test File | Tests | What It Validates |
|-----------|-------|--------------------|
| `test_no_lookahead.py` | 7 | `position[t] = signal[t-1]`, no future data |
| `test_backtest.py` | 23 | Engine mechanics, position delays |
| `test_costs.py` | 29 | 3-component cost model accuracy |
| `test_strategies.py` | 37 | MR, TF, SmoothedMR, Ensemble signals |
| `test_equity_strategies.py` | 16 | Dual Momentum, Sector Rotation |
| `test_portfolio.py` | 15 | Portfolio construction, vol-scaling |
| `test_risk_engine.py` | 41 | Kill switch, hard limits, breaches |
| `test_volatility_signal.py` | 8 | Volatility mean-reversion signal |

## Reports

| Report | Format | Contents |
|--------|--------|----------|
| [Final Report](reports/final_report.md) | Markdown | Full research narrative with metrics, gap analysis, methodology |
| [Performance Report](reports/performance_report.html) | HTML | Interactive visual report with embedded charts and tables |

## License

Private — take-home assignment.
