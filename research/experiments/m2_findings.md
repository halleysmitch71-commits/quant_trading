# Milestone 2 — Exploratory Research Findings

*Date: 2026-09-09 | Status: Observation only — no models fitted*

## Data Overview

| Ticker | Ann. Return | Ann. Volatility | Sharpe | Max DD | Skew | Kurtosis |
|--------|-------------|-----------------|--------|--------|------|----------|
| SPY | 16.8% | 19.1% | 0.88 | −33.7% | −0.29 | 13.2 |
| QQQ | 23.3% | 23.8% | 0.98 | −35.1% | −0.17 | 6.6 |
| BTC-USD | 49.9% | 63.5% | 0.79 | −81.5% | −0.29 | 8.4 |
| ETH-USD | 62.3% | 83.5% | 0.75 | −94.0% | −0.15 | 6.2 |
| GC=F | 17.0% | 17.4% | 0.98 | −25.1% | −0.69 | 8.5 |
| CL=F | −24.2% | 122.7% | −0.20 | −149.3% | −30.4 | 1161.0 |

## Key Observations

### 1. Diversification Potential is Strong

The correlation matrix shows excellent diversification:
- **SPY–QQQ: 0.94** → Very high; these are essentially the same trade
- **GC=F vs everything: 0.03–0.11** → Gold is nearly uncorrelated to all other assets
- **CL=F vs everything: 0.03–0.12** → Oil is also near-zero correlation
- **BTC/ETH vs equities: 0.31–0.34** → Moderate; crypto adds diversification
- **BTC–ETH: 0.82** → High; these move together

**Implication:** A portfolio that includes gold + crypto + equities should
benefit from meaningful diversification.  However, SPY and QQQ are largely
redundant — we may want to keep only one or accept the high correlation.

### 2. CL=F Data is Problematic

CL=F (Crude Oil) has extreme statistics:
- Kurtosis = 1,161 (caused by the April 2020 negative price event)
- Max drawdown = −149% (price went from positive to negative)
- Annualised volatility = 123%

**Decision:** CL=F may distort any volatility-scaled strategy.  We should
either (a) use it with extreme caution or (b) replace it with a cleaner
proxy (e.g., USO ETF or a different commodity).  We will revisit this in
Milestone 5.

### 3. Significant Lag-1 Autocorrelation

All assets show statistically significant lag-1 autocorrelation:

| Ticker | Lag-1 AC | Significant? | Pattern |
|--------|----------|--------------|---------|
| SPY | −0.134 | ✓✓ | Mean-reversion |
| QQQ | −0.122 | ✓✓ | Mean-reversion |
| BTC-USD | −0.046 | ✓ | Mean-reversion |
| ETH-USD | −0.045 | ✓ | Mean-reversion |
| GC=F | −0.029 | | Weak mean-reversion |
| CL=F | +0.287 | ✓✓ | Momentum |

**Implication:**
- Equities (SPY, QQQ) show strong daily mean-reversion → candidate for
  short-term mean-reversion signals.
- Crypto shows weaker mean-reversion → may work at longer timeframes.
- CL=F shows strong momentum → but data quality concerns apply.
- GC=F autocorrelation is negligible → may respond to trend-following
  at longer horizons.

### 4. Volatility Regimes are Clearly Visible

- **COVID crash (2020-03):** Volatility spiked to 80%+ for equities, 150%+ for crypto
- **Rate hike regime (2022):** Sustained elevated volatility for equities and crypto
- **Recent period:** QQQ and GC=F volatility has increased vs historical

**Implication:** Any strategy must be robust to volatility regime changes.
Volatility-scaling position sizes will be essential (Milestone 8).

### 5. Return Distributions are Fat-Tailed

All assets show excess kurtosis (>3), meaning extreme events happen more
often than a normal distribution predicts.  Equities also show negative
skewness (bigger losses than gains).

**Implication:** Risk management cannot rely on Gaussian assumptions.
Maximum drawdown controls and tail-risk awareness are necessary.

### 6. Percentage of Positive Days

All assets hover around 50–56% positive days.  This is typical — daily
returns are nearly a coin flip.  Any edge must come from asymmetric
gains (letting winners run) or reducing exposure during drawdowns.

## Preliminary Hypothesis Ideas

These are *observations*, not commitments.  Each will be formally tested
in Milestones 5–7.

| # | Hypothesis | Based On | Asset(s) |
|---|-----------|----------|----------|
| H1 | Daily mean-reversion in equities | Strong negative lag-1 AC (−0.13) | SPY, QQQ |
| H2 | Medium-term momentum (20–60 day) | Trend persistence in price charts | All |
| H3 | Volatility-mean-reversion | Vol clusters and mean-reverts | SPY, BTC-USD |
| H4 | Gold as a diversifier/hedge | Near-zero correlation + positive return | GC=F |
| H5 | Cross-asset momentum | Some assets lead/lag others | Cross-asset |

**None of these hypotheses have been tested yet.  We will formally evaluate
them starting in Milestone 5.**

## Charts Generated

All 9 charts saved to `research/experiments/m2_charts/`:

1. `01_price_history.png` — Normalised prices (base 100)
2. `02_return_distributions.png` — Daily return histograms
3. `03_rolling_volatility.png` — 21-day and 63-day rolling volatility
4. `04_drawdowns.png` — Drawdown curves with max DD
5. `05_correlation_matrix.png` — Return correlation heatmap
6. `06_rolling_corr_vs_spy.png` — 63-day rolling correlation vs SPY
7. `07_monthly_returns_heatmap.png` — Year × month return heatmaps
8. `08_regime_rolling_sharpe.png` — Rolling Sharpe with regime overlay
9. `09_autocorrelation.png` — Return autocorrelation (lags 1–20)
