"""
Milestone 8 — Portfolio Construction and Volatility Scaling
============================================================

Combines the viable sub-strategies from M7 into a single portfolio.

Sub-strategy menu (from M7):
  • BTC-USD × TF_60    (Sharpe +0.75)  — Primary alpha
  • ETH-USD × TF_120   (Sharpe +0.61)  — Crypto diversifier
  • QQQ × SmoothMR_h10 (Sharpe +0.23)  — Tech MR
  • SPY × SmoothMR_h10 (Sharpe +0.07)  — Equity MR
  • GC=F × SmoothMR_h5 (Sharpe +0.12)  — Gold diversifier

Portfolio features:
  • Equal-weight allocation across sub-strategies
  • Vol-scaled to 10% annualised target
  • Gross exposure capped at 100%
  • Per-asset transaction costs

IN-SAMPLE data only (2018-01 → 2023-03).

Usage:
    python research/experiments/m8_portfolio.py
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
    EQUITY_COSTS, CRYPTO_COSTS, FUTURES_COSTS, scale_costs,
)
from quant_research.data.loader import load_config, load_universe, split_data
from quant_research.metrics.performance import (
    full_report, format_report, monthly_returns, drawdown_series,
)
from quant_research.portfolio.allocator import SubStrategy, build_portfolio
from quant_research.strategies.mean_reversion_smooth import SmoothedMeanReversionSignal
from quant_research.strategies.trend_following import TrendFollowingSignal

CHART_DIR = PROJECT_ROOT / "research" / "experiments" / "m8_charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)


def get_sub_strategies() -> list[SubStrategy]:
    """Define the sub-strategy menu."""
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
    print("MILESTONE 8 — Portfolio Construction & Volatility Scaling")
    print("=" * 70)

    config = load_config()
    universe = load_universe(config)

    # Prepare in-sample data
    is_universe = {}
    for ticker, df in universe.items():
        splits = split_data(df, config)
        is_universe[ticker] = splits["in_sample"]

    sub_strategies = get_sub_strategies()

    # ── 1. Build the portfolio ───────────────────────────────────────────
    print("\n[1/5] Building portfolio (5 sub-strategies, vol_target=10%)...")
    result = build_portfolio(
        universe=is_universe,
        sub_strategies=sub_strategies,
        vol_target=0.10,
        vol_lookback=63,
        max_gross_exposure=1.0,
    )

    report = full_report(result.net_returns, costs=result.costs)
    print(format_report(report))

    # ── 2. Vol target sensitivity ────────────────────────────────────────
    print("\n[2/5] Vol target sensitivity...")
    vol_targets = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
    vol_rows = []
    for vt in vol_targets:
        r = build_portfolio(is_universe, sub_strategies, vol_target=vt)
        rep = full_report(r.net_returns, costs=r.costs)
        vol_rows.append({
            "vol_target": f"{vt:.0%}",
            "realised_vol": f"{rep['annualised_volatility']:.1%}",
            "ann_return": f"{rep['annualised_return']:.2%}",
            "sharpe": f"{rep['sharpe_ratio']:.3f}",
            "max_dd": f"{rep['max_drawdown']:.2%}",
            "avg_monthly": f"{rep['avg_monthly_return']:.2%}",
            "pct_pos_mo": f"{rep['pct_positive_months']:.1%}",
            "avg_exposure": f"{r.gross_exposure.mean():.2%}",
        })
    vol_df = pd.DataFrame(vol_rows)
    print(vol_df.to_string(index=False))

    # ── 3. Monthly returns analysis ──────────────────────────────────────
    print("\n[3/5] Monthly returns analysis...")
    monthly = monthly_returns(result.net_returns) * 100
    print(f"  Months total:    {len(monthly)}")
    print(f"  Positive months: {(monthly > 0).sum()} ({(monthly > 0).mean():.1%})")
    print(f"  Avg monthly:     {monthly.mean():.2f}%")
    print(f"  Median monthly:  {monthly.median():.2f}%")
    print(f"  Best month:      {monthly.max():.2f}%")
    print(f"  Worst month:     {monthly.min():.2f}%")

    # Check 2-4% monthly target
    in_target = ((monthly >= 2.0) & (monthly <= 4.0)).sum()
    print(f"  Months in 2-4% range: {in_target} ({in_target/len(monthly):.1%})")

    # ── 4. Exposure analysis ─────────────────────────────────────────────
    print("\n[4/5] Gross exposure analysis...")
    print(f"  Mean exposure:   {result.gross_exposure.mean():.2%}")
    print(f"  Max exposure:    {result.gross_exposure.max():.2%}")
    print(f"  Min exposure:    {result.gross_exposure.min():.2%}")
    print(f"  Exposure > 90%:  {(result.gross_exposure > 0.9).mean():.1%} of days")

    # ── 5. Charts ────────────────────────────────────────────────────────
    print("\n[5/5] Generating charts...")

    # Equity curve
    fig, ax = plt.subplots(figsize=(14, 7))
    eq = result.equity / result.initial_capital
    ax.plot(eq.index, eq.values, color="navy", linewidth=1.5)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.fill_between(eq.index, 1.0, eq.values, alpha=0.15,
                     where=eq.values >= 1.0, color="green")
    ax.fill_between(eq.index, 1.0, eq.values, alpha=0.15,
                     where=eq.values < 1.0, color="red")
    ax.set_title("Portfolio Equity Curve (In-Sample, Net of Costs)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Equity (multiple of initial)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(CHART_DIR / "01_portfolio_equity.png", dpi=150)
    plt.close(fig)
    print("  ✓ 01_portfolio_equity.png")

    # Monthly returns bar chart
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["green" if r > 0 else "red" for r in monthly.values]
    ax.bar(monthly.index, monthly.values, width=20, color=colors, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(2, color="blue", linewidth=0.8, linestyle="--", alpha=0.5, label="2% target")
    ax.axhline(4, color="blue", linewidth=0.8, linestyle="--", alpha=0.5, label="4% target")
    ax.set_title("Portfolio Monthly Returns (%)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Return (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHART_DIR / "02_monthly_returns.png", dpi=150)
    plt.close(fig)
    print("  ✓ 02_monthly_returns.png")

    # Drawdown
    fig, ax = plt.subplots(figsize=(14, 5))
    dd = drawdown_series(result.net_returns) * 100
    ax.fill_between(dd.index, dd.values, 0, color="red", alpha=0.3)
    ax.plot(dd.index, dd.values, color="red", linewidth=0.8)
    ax.set_title(f"Portfolio Drawdown (Max DD = {dd.min():.1f}%)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(CHART_DIR / "03_drawdown.png", dpi=150)
    plt.close(fig)
    print("  ✓ 03_drawdown.png")

    # Gross exposure
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(result.gross_exposure.index, result.gross_exposure.values * 100,
                     0, color="steelblue", alpha=0.3)
    ax.plot(result.gross_exposure.index, result.gross_exposure.values * 100,
            color="steelblue", linewidth=0.8)
    ax.axhline(100, color="red", linestyle="--", linewidth=1.0, label="100% cap")
    ax.set_title("Gross Exposure Over Time", fontsize=14, fontweight="bold")
    ax.set_ylabel("Gross Exposure (%)")
    ax.set_ylim(0, 110)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(CHART_DIR / "04_gross_exposure.png", dpi=150)
    plt.close(fig)
    print("  ✓ 04_gross_exposure.png")

    # Asset position breakdown
    fig, ax = plt.subplots(figsize=(14, 6))
    positions = result.asset_positions * 100
    for col in positions.columns:
        ax.plot(positions.index, positions[col].values, label=col, linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Asset Positions Over Time (%)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Position (%)")
    ax.legend(framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(CHART_DIR / "05_asset_positions.png", dpi=150)
    plt.close(fig)
    print("  ✓ 05_asset_positions.png")

    # ── Assignment target check ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ASSIGNMENT TARGET CHECK")
    print("=" * 70)
    avg_mo = report["avg_monthly_return"] * 100
    pct_pos = report["pct_positive_months"] * 100
    max_exp = result.gross_exposure.max()

    targets = [
        ("2-4% monthly return", f"{avg_mo:.2f}%", 2.0 <= avg_mo <= 4.0),
        (">75% positive months", f"{pct_pos:.1f}%", pct_pos > 75),
        ("Gross exposure ≤ 100%", f"{max_exp:.2%}", max_exp <= 1.0),
        ("No leverage", "Yes" if max_exp <= 1.0 else "No", max_exp <= 1.0),
    ]

    for target, actual, passed in targets:
        status = "✓" if passed else "✗"
        print(f"  {status} {target}: {actual}")

    print(f"\n  Sharpe Ratio: {report['sharpe_ratio']:.3f}")
    print(f"  Sortino Ratio: {report['sortino_ratio']:.3f}")
    print(f"  Max Drawdown: {report['max_drawdown']:.2%}")
    print(f"  Calmar Ratio: {report['calmar_ratio']:.3f}")

    print(f"\nCharts saved to: {CHART_DIR}/")
    print("=" * 70)
    print("NOTE: These are IN-SAMPLE results. Validation in Milestone 10.")
    print("=" * 70)


if __name__ == "__main__":
    main()
