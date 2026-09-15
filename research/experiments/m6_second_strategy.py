"""
Milestone 6 — Second Strategy: Time-Series Momentum (Trend Following)
======================================================================

This script tests Hypothesis H2 from Milestone 2 on IN-SAMPLE data only.

Hypothesis:  Asset prices exhibit medium-term momentum.
Signal:      signal[t] = clip(z / 2, -1, 1)
             where z = lookback_return / (rolling_vol × √N)
Assets:      All 5 tradeable assets.

Also compares head-to-head with Milestone 5's mean-reversion baseline.

IMPORTANT:
  • IN-SAMPLE data ONLY (2018-01 → 2023-03).
  • lookback=60 chosen from literature, NOT optimised.

Usage:
    python research/experiments/m6_second_strategy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

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
    full_report, format_report, monthly_returns, drawdown_series,
)
from quant_research.strategies.mean_reversion import MeanReversionSignal
from quant_research.strategies.trend_following import TrendFollowingSignal

CHART_DIR = PROJECT_ROOT / "research" / "experiments" / "m6_charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

ASSET_COSTS = {
    "SPY": EQUITY_COSTS,
    "QQQ": EQUITY_COSTS,
    "BTC-USD": CRYPTO_COSTS,
    "ETH-USD": CRYPTO_COSTS,
    "GC=F": FUTURES_COSTS,
}


def run_single(ticker, data, cost_model, signal_obj):
    """Run a single backtest and return result + report."""
    signal = signal_obj.generate(data)
    prices = data["close"]
    result = run_backtest(prices=prices, signal=signal, cost_model=cost_model)
    report = full_report(result.net_returns, turnover=result.turnover, costs=result.costs)
    return result, report


def main():
    print("=" * 70)
    print("MILESTONE 6 — Second Strategy: Trend Following")
    print("=" * 70)

    config = load_config()
    universe = load_universe(config)

    # ── 1. Run trend-following on all assets ─────────────────────────────
    print("\n[1/5] Running Trend-Following (lookback=60) on IN-SAMPLE data...")
    tf_results = {}
    for ticker, cost_model in ASSET_COSTS.items():
        if ticker not in universe:
            continue
        splits = split_data(universe[ticker], config)
        is_data = splits["in_sample"]

        sig_obj = TrendFollowingSignal(lookback=60, scale=2.0)
        result, report = run_single(ticker, is_data, cost_model, sig_obj)
        tf_results[ticker] = {"result": result, "report": report}

        print(f"  {ticker}: Sharpe={report['sharpe_ratio']:.3f}, "
              f"Ann.Ret={report['annualised_return']:.2%}, "
              f"Turnover={report.get('total_turnover', 0):.1f}, "
              f"%Pos.Mo={report['pct_positive_months']:.1%}")

    # ── 2. Head-to-head comparison with mean-reversion ───────────────────
    print("\n[2/5] Head-to-head: Trend vs. Mean-Reversion (SPY)...")
    spy_splits = split_data(universe["SPY"], config)
    spy_is = spy_splits["in_sample"]

    mr_sig = MeanReversionSignal(vol_lookback=20)
    tf_sig = TrendFollowingSignal(lookback=60, scale=2.0)

    mr_result, mr_report = run_single("SPY", spy_is, EQUITY_COSTS, mr_sig)
    tf_result, tf_report = run_single("SPY", spy_is, EQUITY_COSTS, tf_sig)

    comparison = pd.DataFrame({
        "Metric": [
            "Ann. Return", "Ann. Vol", "Sharpe", "Sortino",
            "Max DD", "Win Rate", "% Pos Months",
            "Total Turnover", "Avg Daily Turnover", "Total Costs",
        ],
        "MeanReversion": [
            f"{mr_report['annualised_return']:.2%}",
            f"{mr_report['annualised_volatility']:.2%}",
            f"{mr_report['sharpe_ratio']:.3f}",
            f"{mr_report['sortino_ratio']:.3f}",
            f"{mr_report['max_drawdown']:.2%}",
            f"{mr_report['win_rate']:.1%}",
            f"{mr_report['pct_positive_months']:.1%}",
            f"{mr_report.get('total_turnover', 0):.1f}",
            f"{mr_report.get('avg_daily_turnover', 0):.4f}",
            f"{mr_report.get('total_costs', 0):.4f}",
        ],
        "TrendFollowing": [
            f"{tf_report['annualised_return']:.2%}",
            f"{tf_report['annualised_volatility']:.2%}",
            f"{tf_report['sharpe_ratio']:.3f}",
            f"{tf_report['sortino_ratio']:.3f}",
            f"{tf_report['max_drawdown']:.2%}",
            f"{tf_report['win_rate']:.1%}",
            f"{tf_report['pct_positive_months']:.1%}",
            f"{tf_report.get('total_turnover', 0):.1f}",
            f"{tf_report.get('avg_daily_turnover', 0):.4f}",
            f"{tf_report.get('total_costs', 0):.4f}",
        ],
    }).set_index("Metric")
    print("\n" + comparison.to_string())

    # ── 3. Lookback sensitivity ──────────────────────────────────────────
    print("\n[3/5] Lookback sensitivity (SPY, trend-following)...")
    lookbacks = [20, 40, 60, 90, 120, 180]
    sensitivity_rows = []
    for lb in lookbacks:
        sig = TrendFollowingSignal(lookback=lb, scale=2.0)
        result, report = run_single("SPY", spy_is, EQUITY_COSTS, sig)
        sensitivity_rows.append({
            "lookback": lb,
            "ann_return": report["annualised_return"],
            "sharpe": report["sharpe_ratio"],
            "pct_pos_months": report["pct_positive_months"],
            "total_turnover": report.get("total_turnover", 0),
            "total_costs": report.get("total_costs", 0),
        })
    sens_df = pd.DataFrame(sensitivity_rows)
    print(sens_df.to_string(index=False))

    # ── 4. Cost sensitivity (trend-following, SPY) ───────────────────────
    print("\n[4/5] Cost sensitivity (SPY, trend-following, lookback=60)...")
    cost_mults = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    cost_rows = []
    for mult in cost_mults:
        cm = scale_costs(EQUITY_COSTS, mult) if mult > 0 else CostModel()
        sig = TrendFollowingSignal(lookback=60)
        result, report = run_single("SPY", spy_is, cm, sig)
        cost_rows.append({
            "cost_multiple": mult,
            "total_bps": cm.total_bps,
            "ann_return": report["annualised_return"],
            "sharpe": report["sharpe_ratio"],
        })
    cost_df = pd.DataFrame(cost_rows)
    print(cost_df.to_string(index=False))

    # ── 5. Charts ────────────────────────────────────────────────────────
    print("\n[5/5] Generating charts...")

    # Equity curves: all assets, trend-following
    fig, ax = plt.subplots(figsize=(14, 7))
    for ticker, res in tf_results.items():
        eq = res["result"].equity / 100_000
        ax.plot(eq.index, eq.values, label=ticker, linewidth=1.3)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title("Trend-Following (60-day) — In-Sample Equity Curves", fontsize=14, fontweight="bold")
    ax.set_ylabel("Equity (multiple of initial)")
    ax.legend(framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(CHART_DIR / "01_tf_equity_curves.png", dpi=150)
    plt.close(fig)
    print("  ✓ 01_tf_equity_curves.png")

    # Head-to-head: MR vs TF on SPY
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax1 = axes[0]
    ax1.plot(mr_result.equity.index, mr_result.equity / 100_000,
             label="Mean-Reversion", color="red", linewidth=1.3)
    ax1.plot(tf_result.equity.index, tf_result.equity / 100_000,
             label="Trend-Following", color="blue", linewidth=1.3)
    ax1.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.set_title("SPY: Mean-Reversion vs. Trend-Following", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Equity (multiple)")
    ax1.legend(framealpha=0.9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax2 = axes[1]
    mr_dd = drawdown_series(mr_result.net_returns) * 100
    tf_dd = drawdown_series(tf_result.net_returns) * 100
    ax2.plot(mr_dd.index, mr_dd.values, label="MR Drawdown", color="red", alpha=0.7)
    ax2.plot(tf_dd.index, tf_dd.values, label="TF Drawdown", color="blue", alpha=0.7)
    ax2.set_title("Drawdown Comparison", fontsize=12)
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend(framealpha=0.9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig(CHART_DIR / "02_mr_vs_tf_spy.png", dpi=150)
    plt.close(fig)
    print("  ✓ 02_mr_vs_tf_spy.png")

    # Lookback sensitivity chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1, ax2 = axes
    ax1.bar([str(lb) for lb in sens_df["lookback"]], sens_df["sharpe"], color="steelblue")
    ax1.set_title("Sharpe Ratio by Lookback", fontsize=12)
    ax1.set_xlabel("Lookback (days)")
    ax1.set_ylabel("Sharpe Ratio")
    ax1.axhline(0, color="black", linewidth=0.8)

    ax2.bar([str(lb) for lb in sens_df["lookback"]], sens_df["total_turnover"], color="coral")
    ax2.set_title("Total Turnover by Lookback", fontsize=12)
    ax2.set_xlabel("Lookback (days)")
    ax2.set_ylabel("Total Turnover")

    fig.suptitle("Lookback Sensitivity — SPY Trend-Following", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "03_lookback_sensitivity.png", dpi=150)
    plt.close(fig)
    print("  ✓ 03_lookback_sensitivity.png")

    # ── Failure criteria ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FAILURE CRITERIA EVALUATION")
    print("=" * 70)
    for ticker, res in tf_results.items():
        r = res["report"]
        mr_turnover = 0
        if ticker == "SPY":
            mr_turnover = mr_report.get("total_turnover", 0)

        checks = []
        if r["sharpe_ratio"] >= 0.3:
            checks.append(f"  ✓ Sharpe = {r['sharpe_ratio']:.3f} ≥ 0.3")
        else:
            checks.append(f"  ✗ Sharpe = {r['sharpe_ratio']:.3f} < 0.3")

        if r["pct_positive_months"] > 0.50:
            checks.append(f"  ✓ %Pos Months = {r['pct_positive_months']:.1%} > 50%")
        else:
            checks.append(f"  ✗ %Pos Months = {r['pct_positive_months']:.1%} ≤ 50%")

        turnover = r.get("total_turnover", 0)
        checks.append(f"  ℹ Turnover = {turnover:.1f}")

        print(f"\n{ticker}:")
        for c in checks:
            print(c)

    # 2× cost robustness for SPY
    doubled_cm = scale_costs(EQUITY_COSTS, 2.0)
    _, doubled_report = run_single("SPY", spy_is, doubled_cm, TrendFollowingSignal(lookback=60))
    print(f"\nSPY at 2× costs (20 bps one-way):")
    print(f"  Sharpe = {doubled_report['sharpe_ratio']:.3f}, "
          f"Ann.Return = {doubled_report['annualised_return']:.2%}")

    print(f"\nCharts saved to: {CHART_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
