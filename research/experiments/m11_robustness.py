"""
Milestone 11 — Robustness and Parameter Stability Analysis
============================================================

After M10 showed 94% OOS Sharpe decay, this milestone investigates:

  1. Parameter stability — how sensitive is IS performance to params?
  2. Regime analysis — which market regimes drove the IS edge?
  3. Signal correlation — are sub-strategies truly diversified?
  4. Bootstrap significance — is the IS Sharpe statistically significant?
  5. Cost sensitivity at portfolio level — edge per unit of cost

This is a DIAGNOSTIC milestone.  No parameters are changed.

Usage:
    python research/experiments/m11_robustness.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from itertools import product

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant_research.backtest.costs import (
    EQUITY_COSTS, CRYPTO_COSTS, FUTURES_COSTS, scale_costs, CostModel,
)
from quant_research.backtest.engine import run_backtest
from quant_research.data.loader import load_config, load_universe, split_data
from quant_research.metrics.performance import (
    full_report, monthly_returns, drawdown_series, sharpe_ratio,
)
from quant_research.portfolio.allocator import SubStrategy, build_portfolio
from quant_research.strategies.mean_reversion_smooth import SmoothedMeanReversionSignal
from quant_research.strategies.trend_following import TrendFollowingSignal

CHART_DIR = PROJECT_ROOT / "research" / "experiments" / "m11_charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

PORTFOLIO_PARAMS = dict(
    vol_target=0.10, vol_lookback=63, max_gross_exposure=1.0,
    dd_threshold=-0.05, concentration_limit=0.40,
)


def get_sub_strategies():
    return [
        SubStrategy("BTC_TF60", "BTC-USD",
                     TrendFollowingSignal(lookback=60, scale=2.0),
                     CRYPTO_COSTS, weight=1.0),
        SubStrategy("ETH_TF120", "ETH-USD",
                     TrendFollowingSignal(lookback=120, scale=2.0),
                     CRYPTO_COSTS, weight=1.0),
        SubStrategy("QQQ_SMR10", "QQQ",
                     SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=10),
                     EQUITY_COSTS, weight=1.0),
        SubStrategy("SPY_SMR10", "SPY",
                     SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=10),
                     EQUITY_COSTS, weight=1.0),
        SubStrategy("GC_SMR5", "GC=F",
                     SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=5),
                     FUTURES_COSTS, weight=1.0),
    ]


def main():
    print("=" * 70)
    print("MILESTONE 11 — Robustness & Parameter Stability Analysis")
    print("=" * 70)

    config = load_config()
    universe = load_universe(config)

    is_universe = {}
    val_universe = {}
    for ticker, df in universe.items():
        splits = split_data(df, config)
        is_universe[ticker] = splits["in_sample"]
        if "validation" in splits:
            val_universe[ticker] = splits["validation"]

    # ═══════════════════════════════════════════════════════════════════
    # 1. PARAMETER STABILITY — Trend-Following lookback
    # ═══════════════════════════════════════════════════════════════════
    print("\n[1/5] Parameter stability: TF lookback sweep...")

    lookbacks = [20, 40, 60, 80, 100, 120, 150, 180]
    param_rows = []

    for lb in lookbacks:
        strategies = [
            SubStrategy("BTC_TF", "BTC-USD",
                         TrendFollowingSignal(lookback=lb, scale=2.0),
                         CRYPTO_COSTS, weight=1.0),
            SubStrategy("ETH_TF", "ETH-USD",
                         TrendFollowingSignal(lookback=lb, scale=2.0),
                         CRYPTO_COSTS, weight=1.0),
            SubStrategy("QQQ_SMR", "QQQ",
                         SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=10),
                         EQUITY_COSTS, weight=1.0),
            SubStrategy("SPY_SMR", "SPY",
                         SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=10),
                         EQUITY_COSTS, weight=1.0),
            SubStrategy("GC_SMR", "GC=F",
                         SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=5),
                         FUTURES_COSTS, weight=1.0),
        ]

        # In-sample
        r_is = build_portfolio(is_universe, strategies, **PORTFOLIO_PARAMS)
        rep_is = full_report(r_is.net_returns)

        # Validation
        r_val = build_portfolio(val_universe, strategies, **PORTFOLIO_PARAMS)
        rep_val = full_report(r_val.net_returns)

        param_rows.append({
            "tf_lookback": lb,
            "is_sharpe": rep_is["sharpe_ratio"],
            "val_sharpe": rep_val["sharpe_ratio"],
            "is_return": rep_is["annualised_return"],
            "val_return": rep_val["annualised_return"],
            "decay_pct": (1 - rep_val["sharpe_ratio"] / rep_is["sharpe_ratio"]) * 100
            if rep_is["sharpe_ratio"] != 0 else 0,
        })

    param_df = pd.DataFrame(param_rows)
    print(param_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ═══════════════════════════════════════════════════════════════════
    # 2. PARAMETER STABILITY — MR ema_halflife
    # ═══════════════════════════════════════════════════════════════════
    print("\n[2/5] Parameter stability: MR ema_halflife sweep...")

    halflives = [3, 5, 7, 10, 15, 20, 30]
    mr_rows = []

    for hl in halflives:
        strategies = [
            SubStrategy("BTC_TF", "BTC-USD",
                         TrendFollowingSignal(lookback=60, scale=2.0),
                         CRYPTO_COSTS, weight=1.0),
            SubStrategy("ETH_TF", "ETH-USD",
                         TrendFollowingSignal(lookback=120, scale=2.0),
                         CRYPTO_COSTS, weight=1.0),
            SubStrategy("QQQ_SMR", "QQQ",
                         SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=hl),
                         EQUITY_COSTS, weight=1.0),
            SubStrategy("SPY_SMR", "SPY",
                         SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=hl),
                         EQUITY_COSTS, weight=1.0),
            SubStrategy("GC_SMR", "GC=F",
                         SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=hl),
                         FUTURES_COSTS, weight=1.0),
        ]

        r_is = build_portfolio(is_universe, strategies, **PORTFOLIO_PARAMS)
        rep_is = full_report(r_is.net_returns)

        r_val = build_portfolio(val_universe, strategies, **PORTFOLIO_PARAMS)
        rep_val = full_report(r_val.net_returns)

        mr_rows.append({
            "ema_halflife": hl,
            "is_sharpe": rep_is["sharpe_ratio"],
            "val_sharpe": rep_val["sharpe_ratio"],
            "is_return": rep_is["annualised_return"],
            "val_return": rep_val["annualised_return"],
        })

    mr_df = pd.DataFrame(mr_rows)
    print(mr_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ═══════════════════════════════════════════════════════════════════
    # 3. REGIME ANALYSIS — yearly breakdown
    # ═══════════════════════════════════════════════════════════════════
    print("\n[3/5] Regime analysis: yearly performance breakdown...")

    # Run on full data
    full_result = build_portfolio(is_universe, get_sub_strategies(), **PORTFOLIO_PARAMS)

    yearly_groups = full_result.net_returns.groupby(full_result.net_returns.index.year)
    regime_rows = []
    for year, returns in yearly_groups:
        rep = full_report(returns)
        regime_rows.append({
            "year": year,
            "ann_return": rep["annualised_return"],
            "sharpe": rep["sharpe_ratio"],
            "max_dd": rep["max_drawdown"],
            "pct_pos_mo": rep["pct_positive_months"],
            "n_days": len(returns),
        })
    regime_df = pd.DataFrame(regime_rows)
    print(regime_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ═══════════════════════════════════════════════════════════════════
    # 4. BOOTSTRAP SIGNIFICANCE TEST
    # ═══════════════════════════════════════════════════════════════════
    print("\n[4/5] Bootstrap significance test (1000 resamples)...")

    is_returns = full_result.net_returns.values
    n_bootstrap = 1000
    rng = np.random.default_rng(42)
    bootstrap_sharpes = []

    for _ in range(n_bootstrap):
        # Resample daily returns with replacement (block bootstrap, block=21)
        block_size = 21
        n_blocks = len(is_returns) // block_size + 1
        block_starts = rng.integers(0, len(is_returns) - block_size, n_blocks)
        resampled = np.concatenate([
            is_returns[s:s + block_size] for s in block_starts
        ])[:len(is_returns)]
        resampled_sr = pd.Series(resampled)
        sr = sharpe_ratio(resampled_sr)
        bootstrap_sharpes.append(sr)

    bootstrap_sharpes = np.array(bootstrap_sharpes)
    ci_lower = np.percentile(bootstrap_sharpes, 2.5)
    ci_upper = np.percentile(bootstrap_sharpes, 97.5)
    pct_positive = (bootstrap_sharpes > 0).mean()

    print(f"  Original IS Sharpe: {sharpe_ratio(full_result.net_returns):.3f}")
    print(f"  Bootstrap mean:     {bootstrap_sharpes.mean():.3f}")
    print(f"  95% CI:             [{ci_lower:.3f}, {ci_upper:.3f}]")
    print(f"  P(Sharpe > 0):      {pct_positive:.1%}")

    if ci_lower > 0:
        print("  ✓ Statistically significant at 95% level (IS)")
    else:
        print("  ○ NOT statistically significant at 95% level")

    # ═══════════════════════════════════════════════════════════════════
    # 5. SIGNAL CORRELATION ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    print("\n[5/5] Signal correlation analysis...")

    signal_returns = {}
    strategies = get_sub_strategies()
    for ss in strategies:
        data = is_universe[ss.ticker]
        signal = ss.signal_obj.generate(data)
        prices = data["close"]
        result = run_backtest(prices=prices, signal=signal, cost_model=ss.cost_model)
        signal_returns[ss.name] = result.net_returns

    # Align and compute correlation
    ret_df = pd.DataFrame(signal_returns)
    ret_df = ret_df.dropna()
    corr = ret_df.corr()
    print("\nSub-strategy return correlation matrix:")
    print(corr.to_string(float_format=lambda x: f"{x:.3f}"))

    avg_corr = corr.values[np.triu_indices_from(corr.values, k=1)].mean()
    print(f"\nAverage pairwise correlation: {avg_corr:.3f}")

    # ═══════════════════════════════════════════════════════════════════
    # CHARTS
    # ═══════════════════════════════════════════════════════════════════
    print("\nGenerating charts...")

    # Chart 1: Parameter stability — TF lookback
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1, ax2 = axes

    ax1.plot(param_df["tf_lookback"], param_df["is_sharpe"], "b-o",
             label="In-Sample", linewidth=1.5)
    ax1.plot(param_df["tf_lookback"], param_df["val_sharpe"], "r-o",
             label="Validation", linewidth=1.5)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_title("TF Lookback Stability", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Lookback (days)")
    ax1.set_ylabel("Sharpe Ratio")
    ax1.legend()

    ax2.plot(mr_df["ema_halflife"], mr_df["is_sharpe"], "b-o",
             label="In-Sample", linewidth=1.5)
    ax2.plot(mr_df["ema_halflife"], mr_df["val_sharpe"], "r-o",
             label="Validation", linewidth=1.5)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("MR EMA Halflife Stability", fontsize=12, fontweight="bold")
    ax2.set_xlabel("EMA Halflife (days)")
    ax2.set_ylabel("Sharpe Ratio")
    ax2.legend()

    fig.suptitle("Parameter Stability: IS vs Validation", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "01_param_stability.png", dpi=150)
    plt.close(fig)
    print("  ✓ 01_param_stability.png")

    # Chart 2: Yearly regime breakdown
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["green" if s > 0 else "red" for s in regime_df["sharpe"]]
    ax.bar(regime_df["year"].astype(str), regime_df["sharpe"], color=colors, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("In-Sample Sharpe by Year", fontsize=14, fontweight="bold")
    ax.set_ylabel("Sharpe Ratio")
    for i, (_, row) in enumerate(regime_df.iterrows()):
        ax.text(i, row["sharpe"] + 0.05, f"{row['sharpe']:.2f}",
                ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "02_yearly_regime.png", dpi=150)
    plt.close(fig)
    print("  ✓ 02_yearly_regime.png")

    # Chart 3: Bootstrap histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(bootstrap_sharpes, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="red", linewidth=1.5, linestyle="--", label="Zero")
    ax.axvline(bootstrap_sharpes.mean(), color="green", linewidth=1.5,
               label=f"Mean = {bootstrap_sharpes.mean():.3f}")
    ax.axvline(ci_lower, color="orange", linewidth=1.0, linestyle=":",
               label=f"95% CI lower = {ci_lower:.3f}")
    ax.set_title("Bootstrap Sharpe Distribution (1000 resamples)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Sharpe Ratio")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHART_DIR / "03_bootstrap.png", dpi=150)
    plt.close(fig)
    print("  ✓ 03_bootstrap.png")

    # Chart 4: Correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr.index, fontsize=9)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center",
                    va="center", fontsize=9,
                    color="white" if abs(corr.values[i, j]) > 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Sub-Strategy Return Correlation", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "04_correlation.png", dpi=150)
    plt.close(fig)
    print("  ✓ 04_correlation.png")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ROBUSTNESS SUMMARY")
    print("=" * 70)
    print(f"\n  Parameter stability:")
    print(f"    TF lookback: IS Sharpe range [{param_df['is_sharpe'].min():.3f}, "
          f"{param_df['is_sharpe'].max():.3f}]")
    print(f"    TF lookback: VAL Sharpe range [{param_df['val_sharpe'].min():.3f}, "
          f"{param_df['val_sharpe'].max():.3f}]")
    print(f"    MR halflife: IS Sharpe range [{mr_df['is_sharpe'].min():.3f}, "
          f"{mr_df['is_sharpe'].max():.3f}]")

    print(f"\n  Bootstrap significance:")
    print(f"    IS Sharpe 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
    print(f"    P(Sharpe > 0): {pct_positive:.1%}")

    print(f"\n  Signal diversification:")
    print(f"    Avg pairwise correlation: {avg_corr:.3f}")

    best_year = regime_df.loc[regime_df["sharpe"].idxmax()]
    worst_year = regime_df.loc[regime_df["sharpe"].idxmin()]
    print(f"\n  Regime dependence:")
    print(f"    Best year:  {int(best_year['year'])} (Sharpe {best_year['sharpe']:.2f})")
    print(f"    Worst year: {int(worst_year['year'])} (Sharpe {worst_year['sharpe']:.2f})")

    print(f"\nCharts saved to: {CHART_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
