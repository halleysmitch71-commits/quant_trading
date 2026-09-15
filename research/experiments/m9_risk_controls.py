"""
Milestone 9 — Risk Controls and Gross Exposure Constraint
==========================================================

Tests the impact of drawdown throttling and concentration limits
on portfolio performance.

Compares:
  1. No risk controls (M8 baseline)
  2. Drawdown throttle only
  3. Concentration limit only
  4. Both combined
  5. Higher vol target with risk controls (to close the monthly return gap)

IN-SAMPLE data only (2018-01 → 2023-03).

Usage:
    python research/experiments/m9_risk_controls.py
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
from quant_research.data.loader import load_config, load_universe, split_data
from quant_research.metrics.performance import (
    full_report, format_report, monthly_returns, drawdown_series,
)
from quant_research.portfolio.allocator import SubStrategy, build_portfolio
from quant_research.strategies.mean_reversion_smooth import SmoothedMeanReversionSignal
from quant_research.strategies.trend_following import TrendFollowingSignal

CHART_DIR = PROJECT_ROOT / "research" / "experiments" / "m9_charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)


def get_sub_strategies() -> list[SubStrategy]:
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


def run_config(is_universe, name, **kwargs):
    """Run portfolio with given config and return summary row."""
    result = build_portfolio(is_universe, get_sub_strategies(), **kwargs)
    report = full_report(result.net_returns, costs=result.costs)
    monthly = monthly_returns(result.net_returns)
    return {
        "config": name,
        "ann_return": report["annualised_return"],
        "ann_vol": report["annualised_volatility"],
        "sharpe": report["sharpe_ratio"],
        "sortino": report["sortino_ratio"],
        "max_dd": report["max_drawdown"],
        "avg_monthly": report["avg_monthly_return"],
        "pct_pos_mo": report["pct_positive_months"],
        "avg_exposure": result.gross_exposure.mean(),
        "max_exposure": result.gross_exposure.max(),
        "result": result,
        "report": report,
    }


def main():
    print("=" * 70)
    print("MILESTONE 9 — Risk Controls & Gross Exposure Constraint")
    print("=" * 70)

    config = load_config()
    universe = load_universe(config)

    is_universe = {}
    for ticker, df in universe.items():
        splits = split_data(df, config)
        is_universe[ticker] = splits["in_sample"]

    # ── 1. Compare risk control configurations ───────────────────────────
    print("\n[1/4] Comparing risk control configurations...")

    configs = [
        ("M8 Baseline (no risk)", dict(vol_target=0.10)),
        ("DD throttle (-5%)", dict(vol_target=0.10, dd_threshold=-0.05)),
        ("DD throttle (-3%)", dict(vol_target=0.10, dd_threshold=-0.03)),
        ("Concentration 40%", dict(vol_target=0.10, concentration_limit=0.40)),
        ("DD(-5%) + Conc(40%)", dict(vol_target=0.10, dd_threshold=-0.05, concentration_limit=0.40)),
        ("DD(-3%) + Conc(40%)", dict(vol_target=0.10, dd_threshold=-0.03, concentration_limit=0.40)),
    ]

    results_10 = []
    for name, kwargs in configs:
        row = run_config(is_universe, name, **kwargs)
        results_10.append(row)
        print(f"  {name:30s} Sharpe={row['sharpe']:.3f}  "
              f"MaxDD={row['max_dd']:.2%}  "
              f"%PosMo={row['pct_pos_mo']:.1%}  "
              f"AvgMo={row['avg_monthly']:.2%}")

    # ── 2. Scale up: higher vol targets with risk controls ───────────────
    print("\n[2/4] Scaling up with risk controls...")

    scale_configs = [
        ("Vol 15%, no risk", dict(vol_target=0.15)),
        ("Vol 15%, DD(-5%)+C(40%)", dict(vol_target=0.15, dd_threshold=-0.05, concentration_limit=0.40)),
        ("Vol 20%, no risk", dict(vol_target=0.20)),
        ("Vol 20%, DD(-5%)+C(40%)", dict(vol_target=0.20, dd_threshold=-0.05, concentration_limit=0.40)),
        ("Vol 25%, no risk", dict(vol_target=0.25)),
        ("Vol 25%, DD(-5%)+C(40%)", dict(vol_target=0.25, dd_threshold=-0.05, concentration_limit=0.40)),
        ("Vol 30%, no risk", dict(vol_target=0.30)),
        ("Vol 30%, DD(-5%)+C(40%)", dict(vol_target=0.30, dd_threshold=-0.05, concentration_limit=0.40)),
    ]

    results_scale = []
    for name, kwargs in scale_configs:
        row = run_config(is_universe, name, **kwargs)
        results_scale.append(row)
        print(f"  {name:30s} Sharpe={row['sharpe']:.3f}  "
              f"MaxDD={row['max_dd']:.2%}  "
              f"%PosMo={row['pct_pos_mo']:.1%}  "
              f"AvgMo={row['avg_monthly']:.2%}  "
              f"AvgExp={row['avg_exposure']:.1%}")

    # ── 3. Summary tables ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY: Risk Control Comparison (Vol Target = 10%)")
    print("=" * 70)
    df_10 = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ("result", "report")}
        for r in results_10
    ])
    df_10_fmt = df_10.copy()
    for col in ["ann_return", "ann_vol", "max_dd", "avg_monthly", "pct_pos_mo", "avg_exposure", "max_exposure"]:
        df_10_fmt[col] = df_10[col].apply(lambda x: f"{x:.2%}")
    for col in ["sharpe", "sortino"]:
        df_10_fmt[col] = df_10[col].apply(lambda x: f"{x:.3f}")
    print(df_10_fmt.to_string(index=False))

    print("\n" + "=" * 70)
    print("SUMMARY: Scaling Up with Risk Controls")
    print("=" * 70)
    df_scale = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ("result", "report")}
        for r in results_scale
    ])
    df_scale_fmt = df_scale.copy()
    for col in ["ann_return", "ann_vol", "max_dd", "avg_monthly", "pct_pos_mo", "avg_exposure", "max_exposure"]:
        df_scale_fmt[col] = df_scale[col].apply(lambda x: f"{x:.2%}")
    for col in ["sharpe", "sortino"]:
        df_scale_fmt[col] = df_scale[col].apply(lambda x: f"{x:.3f}")
    print(df_scale_fmt.to_string(index=False))

    # ── 4. Charts ────────────────────────────────────────────────────────
    print("\n[3/4] Generating charts...")

    # Chart 1: Equity curves — baseline vs risk-controlled (vol=10%)
    fig, ax = plt.subplots(figsize=(14, 7))
    for r in results_10:
        eq = r["result"].equity / r["result"].initial_capital
        ax.plot(eq.index, eq.values, label=r["config"], linewidth=1.2)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title("Risk Control Comparison — Vol Target 10%", fontsize=14, fontweight="bold")
    ax.set_ylabel("Equity (multiple)")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(CHART_DIR / "01_risk_comparison_10pct.png", dpi=150)
    plt.close(fig)
    print("  ✓ 01_risk_comparison_10pct.png")

    # Chart 2: Scaled-up equity curves
    fig, ax = plt.subplots(figsize=(14, 7))
    for r in results_scale:
        eq = r["result"].equity / r["result"].initial_capital
        style = "--" if "no risk" in r["config"] else "-"
        ax.plot(eq.index, eq.values, label=r["config"], linewidth=1.2, linestyle=style)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title("Scaling Up: Higher Vol Targets ± Risk Controls", fontsize=14, fontweight="bold")
    ax.set_ylabel("Equity (multiple)")
    ax.legend(fontsize=7, framealpha=0.9, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(CHART_DIR / "02_scale_up.png", dpi=150)
    plt.close(fig)
    print("  ✓ 02_scale_up.png")

    # Chart 3: Best configuration monthly returns
    best = max(results_scale, key=lambda r: r["sharpe"])
    monthly = monthly_returns(best["result"].net_returns) * 100
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["green" if r > 0 else "red" for r in monthly.values]
    ax.bar(monthly.index, monthly.values, width=20, color=colors, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(2, color="blue", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(4, color="blue", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title(f"Best Config: {best['config']} — Monthly Returns",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Return (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(CHART_DIR / "03_best_monthly.png", dpi=150)
    plt.close(fig)
    print("  ✓ 03_best_monthly.png")

    # ── 5. Assignment target check ───────────────────────────────────────
    print("\n[4/4] Assignment target check (best configuration)...")
    print(f"\n  Best config: {best['config']}")
    print(format_report(best["report"]))

    avg_mo = best["report"]["avg_monthly_return"] * 100
    pct_pos = best["report"]["pct_positive_months"] * 100
    max_exp = best["result"].gross_exposure.max()

    targets = [
        ("2-4% monthly return", f"{avg_mo:.2f}%", 2.0 <= avg_mo <= 4.0),
        (">75% positive months", f"{pct_pos:.1f}%", pct_pos > 75),
        ("Gross exposure ≤ 100%", f"{max_exp:.2%}", max_exp <= 1.0 + 1e-10),
        ("No leverage", "Yes" if max_exp <= 1.0 + 1e-10 else "No", max_exp <= 1.0 + 1e-10),
    ]

    for target, actual, passed in targets:
        status = "✓" if passed else "✗"
        print(f"  {status} {target}: {actual}")

    print(f"\nCharts saved to: {CHART_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
