"""
Final Report Generator
=======================

Generates all charts and data for reports/final_report.md.
Runs every experiment one final time for reproducibility.

Usage:
    python research/experiments/generate_final_report.py
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

from quant_research.backtest.costs import EQUITY_COSTS, CRYPTO_COSTS, FUTURES_COSTS
from quant_research.backtest.engine import run_backtest
from quant_research.data.loader import load_config, load_universe, split_data
from quant_research.metrics.performance import (
    full_report, format_report, monthly_returns, drawdown_series,
)
from quant_research.portfolio.allocator import SubStrategy, build_portfolio
from quant_research.strategies.mean_reversion import MeanReversionSignal
from quant_research.strategies.mean_reversion_smooth import SmoothedMeanReversionSignal
from quant_research.strategies.trend_following import TrendFollowingSignal

CHART_DIR = PROJECT_ROOT / "reports" / "charts"
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
    print("Generating final report charts...")

    config = load_config()
    universe = load_universe(config)

    # Prepare splits
    split_universes = {}
    for split_name in ["in_sample", "validation", "out_of_sample"]:
        split_u = {}
        for ticker, df in universe.items():
            splits = split_data(df, config)
            if split_name in splits and len(splits[split_name]) > 0:
                split_u[ticker] = splits[split_name]
        split_universes[split_name] = split_u

    # Run portfolio on all splits
    results = {}
    for split_name, split_u in split_universes.items():
        try:
            result = build_portfolio(split_u, get_sub_strategies(), **PORTFOLIO_PARAMS)
            report = full_report(result.net_returns, costs=result.costs)
            results[split_name] = {"result": result, "report": report}
        except Exception as e:
            print(f"  {split_name}: FAILED — {e}")

    # ── Chart 1: Three-panel equity curves ───────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 7))
    split_colors = {"in_sample": "#2563EB", "validation": "#F59E0B", "out_of_sample": "#10B981"}
    split_labels = {"in_sample": "In-Sample (2018–2023)", "validation": "Validation (2023–2025)",
                    "out_of_sample": "Out-of-Sample (2025–2026)"}

    for split_name, data in results.items():
        eq = data["result"].equity / data["result"].initial_capital
        ax.plot(eq.index, eq.values, label=split_labels.get(split_name, split_name),
                linewidth=2.0, color=split_colors.get(split_name, "gray"))

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.fill_between(
        results["in_sample"]["result"].equity.index,
        0.8,
        (results["in_sample"]["result"].equity / results["in_sample"]["result"].initial_capital).values,
        alpha=0.05, color="#2563EB",
    )
    ax.set_title("Portfolio Equity Curves — All Data Splits", fontsize=16, fontweight="bold")
    ax.set_ylabel("Equity (multiple of initial capital)", fontsize=12)
    ax.legend(fontsize=12, framealpha=0.95, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "01_equity_all_splits.png", dpi=200)
    plt.close(fig)
    print("  ✓ 01_equity_all_splits.png")

    # ── Chart 2: Monthly returns heatmap style ───────────────────────────
    is_result = results["in_sample"]["result"]
    monthly = monthly_returns(is_result.net_returns)
    monthly_pct = monthly * 100

    # Create year × month pivot
    mo_df = pd.DataFrame({"year": monthly.index.year, "month": monthly.index.month,
                           "return": monthly_pct.values})
    pivot = mo_df.pivot(index="year", columns="month", values="return")
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(pivot.columns)]

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-5, vmax=5)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, fontsize=10)
    ax.set_yticklabels(pivot.index, fontsize=10)
    for i in range(pivot.values.shape[0]):
        for j in range(pivot.values.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                color = "white" if abs(v) > 3 else "black"
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Monthly Return (%)")
    ax.set_title("Monthly Returns Heatmap (In-Sample, %)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "02_monthly_heatmap.png", dpi=200)
    plt.close(fig)
    print("  ✓ 02_monthly_heatmap.png")

    # ── Chart 3: Drawdown comparison ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 5))
    for split_name, data in results.items():
        dd = drawdown_series(data["result"].net_returns) * 100
        ax.fill_between(dd.index, dd.values, 0, alpha=0.2,
                         color=split_colors.get(split_name, "gray"))
        ax.plot(dd.index, dd.values, linewidth=1.0,
                color=split_colors.get(split_name, "gray"),
                label=split_labels.get(split_name, split_name))
    ax.set_title("Portfolio Drawdown — All Splits", fontsize=14, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(fontsize=10, framealpha=0.95)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "03_drawdown_all.png", dpi=200)
    plt.close(fig)
    print("  ✓ 03_drawdown_all.png")

    # ── Chart 4: Gross exposure over time ────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 4))
    for split_name, data in results.items():
        exp = data["result"].gross_exposure * 100
        ax.plot(exp.index, exp.values, linewidth=0.8,
                color=split_colors.get(split_name, "gray"),
                label=split_labels.get(split_name, split_name))
    ax.axhline(100, color="red", linestyle="--", linewidth=1.5, label="100% cap")
    ax.set_title("Gross Exposure Over Time", fontsize=14, fontweight="bold")
    ax.set_ylabel("Exposure (%)")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=9, framealpha=0.95)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "04_exposure.png", dpi=200)
    plt.close(fig)
    print("  ✓ 04_exposure.png")

    # ── Print final summary data ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL REPORT DATA")
    print("=" * 70)
    for split_name, data in results.items():
        r = data["report"]
        print(f"\n  {split_name}:")
        print(f"    Sharpe:     {r['sharpe_ratio']:.3f}")
        print(f"    Ann.Return: {r['annualised_return']:.2%}")
        print(f"    Ann.Vol:    {r['annualised_volatility']:.2%}")
        print(f"    Max DD:     {r['max_drawdown']:.2%}")
        print(f"    Sortino:    {r['sortino_ratio']:.3f}")
        print(f"    Calmar:     {r['calmar_ratio']:.3f}")
        print(f"    %PosMo:     {r['pct_positive_months']:.1%}")
        print(f"    AvgMo:      {r['avg_monthly_return']:.2%}")

    print(f"\nAll charts saved to: {CHART_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
