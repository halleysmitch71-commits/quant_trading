# Milestone 6 — Second Strategy Research Findings

*Date: 2026-09-09 | Data: IN-SAMPLE ONLY (2018-01 → 2023-03)*

## Strategy: Time-Series Momentum (Trend-Following)

**Hypothesis:** Asset prices exhibit medium-term momentum (60-day lookback).
**Signal:** `signal[t] = clip(z / 2, -1, 1)` where `z = lookback_ret / (vol × √N)`
**Parameters:** `lookback=60, scale=2.0` (from literature, not optimised)

## Results: Mixed — Strategy Works on CRYPTO, Fails on EQUITIES

### Performance Summary

| Ticker | Sharpe | Ann. Return | Max DD | Turnover | % Pos Months |
|--------|--------|-------------|--------|----------|--------------|
| SPY | −0.531 | −4.9% | −27.0% | 93.8 | 42.9% |
| QQQ | −0.387 | −4.4% | −27.4% | 87.1 | 46.0% |
| **BTC-USD** | **+0.748** | **+25.9%** | −57.2% | 114.8 | 44.4% |
| **ETH-USD** | **+0.311** | **+14.6%** | −66.9% | 117.9 | **50.8%** |
| GC=F | −0.196 | −1.5% | −18.5% | 87.1 | 42.9% |

### Key Finding: Crypto Momentum is Strong

BTC-USD shows the strongest results:
- Sharpe = 0.748 (exceeds our 0.3 threshold)
- Ann. return = 25.9% (impressive even after 15 bps costs)
- This makes economic sense: crypto markets are less efficient, with
  stronger trend persistence due to retail-driven momentum

ETH-USD is marginal but passes Sharpe threshold.

### Key Finding: Equity Momentum Fails at 60-day Horizon

SPY and QQQ show negative Sharpe even at zero costs (SPY: −0.35).
This suggests equities in 2018-2023 were mean-reverting at this
horizon, not trending. The COVID crash and 2022 bear market
created whipsaw environments that hurt momentum signals.

### Head-to-Head: SPY Trend vs. Mean-Reversion

| Metric | Mean-Reversion | Trend-Following |
|--------|---------------|-----------------|
| Sharpe | −0.742 | −0.531 |
| Ann. Return | −12.0% | −4.9% |
| Max DD | −49.9% | −27.0% |
| **Total Turnover** | **1,063** | **94** |
| **Total Costs** | **106.3%** | **9.4%** |
| Avg Daily Turnover | 0.804 | 0.071 |

**Turnover reduced by 91%!**  Trend-following generates 11× less
turnover than mean-reversion. However, even at zero costs the equity
momentum signal is negative — the problem is not costs, it's
that equity momentum doesn't work at this horizon in this period.

### Lookback Sensitivity (SPY)

| Lookback | Sharpe | Turnover |
|----------|--------|----------|
| 20 | −0.526 | 165.3 |
| 40 | −0.529 | 110.9 |
| 60 | −0.531 | 93.8 |
| 90 | −0.371 | 74.2 |
| 120 | −0.341 | 62.2 |
| 180 | −0.083 | 42.3 |

Longer lookbacks perform better for SPY but none are positive.
Turnover decreases monotonically with lookback (as expected).

## Strategic Implications

We now have two strategies with complementary characteristics:

| Property | Mean-Reversion | Trend-Following |
|----------|---------------|-----------------|
| Best asset class | Equities (at zero cost) | Crypto |
| Turnover | Very high | Low |
| Regime | Works in ranging markets | Works in trending markets |
| Correlation | Negative autocorrelation | Positive autocorrelation |
| Expected correlation between strategies | **Negative** | **Negative** |

The negative correlation between MR and TF is the key insight for
portfolio construction (Milestone 8). Even if neither strategy is
great alone, combining them may produce a portfolio with better
risk-adjusted returns.

## Viable Strategy Components for Portfolio Construction

1. **BTC-USD Trend-Following** (Sharpe 0.75) → Keep
2. **ETH-USD Trend-Following** (Sharpe 0.31) → Marginal, keep for diversification
3. **SPY Mean-Reversion at zero cost** (Sharpe 0.48) → Need turnover reduction
4. **QQQ Mean-Reversion at zero cost** → Similar to SPY, consider including

## Next Steps

For Milestone 7, we should explore:
1. Cross-asset strategies (using one asset's signal to trade another)
2. A turnover-controlled version of mean-reversion
3. Combined MR+TF signals within a single asset
