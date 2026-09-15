# Milestone 5 — Baseline Strategy Research Findings

*Date: 2026-09-09 | Data: IN-SAMPLE ONLY (2018-01 → 2023-03)*

## Strategy: Daily Mean-Reversion

**Hypothesis:** Daily returns exhibit mean-reversion (negative lag-1 autocorrelation).  
**Signal:** `signal[t] = −z_score[t]` where `z = return_1d / rolling_vol(20)`  
**Parameter:** `vol_lookback = 20` (not optimised)  

## Result: FAIL (but with an important insight)

### Performance Summary (net of costs)

| Ticker | Ann. Return | Sharpe | Max DD | % Pos Months | Total Costs |
|--------|-------------|--------|--------|--------------|-------------|
| SPY | −11.97% | −0.742 | −49.9% | 25.4% | 106.3% of capital |
| QQQ | −7.52% | −0.380 | −40.4% | 31.7% | 107.0% |
| BTC-USD | −31.08% | −0.772 | −94.8% | 39.7% | 218.0% |
| ETH-USD | −25.07% | −0.474 | −91.0% | 41.3% | 228.1% |
| GC=F | −2.82% | −0.276 | −25.7% | 50.8% | 52.9% |

### Failure Criteria Assessment

| Criterion | SPY | QQQ | BTC | ETH | GC |
|-----------|-----|-----|-----|-----|-----|
| Sharpe ≥ 0.3 | ✗ | ✗ | ✗ | ✗ | ✗ |
| >50% positive months | ✗ | ✗ | ✗ | ✗ | ✓ |
| Survives 2× costs | ✗ | ✗ | ✗ | ✗ | ✗ |

**Verdict: Strategy REJECTED in its current form.**

## The Critical Insight: It's the Turnover

The cost sensitivity analysis reveals the core problem:

| Cost Multiple | Total bps | SPY Ann. Return | SPY Sharpe |
|---------------|-----------|-----------------|------------|
| 0× (no costs) | 0 | **+7.8%** | **+0.48** |
| 0.5× | 5 | −2.6% | −0.16 |
| 1× (base) | 10 | −12.0% | −0.74 |
| 2× | 20 | −28.1% | −1.74 |

**At zero costs, the mean-reversion signal IS profitable (Sharpe 0.48).**
The edge is real, but it's completely eaten by costs.

### Root cause: excessive turnover
- Total turnover = 1,063 (over 5 years)
- Average daily turnover = 0.80 (the strategy trades ~80% of its position every day)
- At 10 bps one-way, this costs 106% of initial capital over the period

## Research Implications

1. **The autocorrelation signal exists** — we're not imagining it.
2. **But the signal produces a continuous, high-frequency position change** — trading ~80% of the portfolio daily.
3. **To be viable, we need to either:**
   - **Reduce turnover** by using a wider lookback or adding a position-change threshold
   - **Use a lower-frequency signal** (weekly or monthly mean-reversion)
   - **Combine with a trend-following signal** (which trades less frequently)
4. **Crypto is even worse** because: higher volatility → larger z-scores → even more turnover, plus higher cost per trade.
5. **Gold (GC=F) is the closest to breaking even** because: lowest costs (5 bps) + lowest volatility.

## What We Learned (for the interview)

> "Our first strategy was a daily mean-reversion signal based on the observed
> negative autocorrelation in equity returns. At zero costs, it showed a
> Sharpe of 0.48. However, the signal generates extremely high turnover
> (~80% daily), which at 10 bps per trade completely destroys the edge.
>
> This taught us that **signal profitability before costs is necessary but
> not sufficient**. The key metric is **edge per unit of turnover**.
> We revised our approach by adding turnover controls and exploring
> lower-frequency signals."

## Next Steps

For Milestone 6 (second strategy), we should:
1. Add a **deadzone/threshold** to the mean-reversion signal to reduce turnover
2. Or pivot to a **momentum/trend-following** signal (lower turnover by nature)
3. Compare: reduced-turnover MR vs. momentum — which has better edge/turnover ratio?
