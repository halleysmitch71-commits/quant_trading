"""
Milestone 7 — Cross-Asset / Diversified Strategies
====================================================

This script:
  1. Tests the smoothed (turnover-controlled) mean-reversion signal.
  2. Tests ensemble (MR+TF) blended signals per asset.
  3. Builds the complete "sub-strategy menu" for portfolio construction.
  4. Compares all viable sub-strategies head-to-head.

All tests use IN-SAMPLE data only (2018-01 → 2023-03).

Usage:
    python research/experiments/m7_diversified_strategies.py
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
from quant_research.metrics.performance import full_report, format_report
from quant_research.strategies.mean_reversion import MeanReversionSignal
from quant_research.strategies.mean_reversion_smooth import SmoothedMeanReversionSignal
from quant_research.strategies.trend_following import TrendFollowingSignal
from quant_research.strategies.ensemble import EnsembleSignal

CHART_DIR = PROJECT_ROOT / "research" / "experiments" / "m7_charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

ASSET_COSTS = {
    "SPY": EQUITY_COSTS,
    "QQQ": EQUITY_COSTS,
    "BTC-USD": CRYPTO_COSTS,
    "ETH-USD": CRYPTO_COSTS,
    "GC=F": FUTURES_COSTS,
}


def run_single(data, cost_model, signal_obj):
    """Run a single backtest and return result + report."""
    signal = signal_obj.generate(data)
    prices = data["close"]
    result = run_backtest(prices=prices, signal=signal, cost_model=cost_model)
    report = full_report(result.net_returns, turnover=result.turnover, costs=result.costs)
    return result, report


def main():
    print("=" * 70)
    print("MILESTONE 7 — Cross-Asset / Diversified Strategies")
    print("=" * 70)

    config = load_config()
    universe = load_universe(config)

    # Define signal variants
    signal_variants = {
        "RawMR": MeanReversionSignal(vol_lookback=20),
        "SmoothMR_h5": SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=5),
        "SmoothMR_h10": SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=10),
        "TF_60": TrendFollowingSignal(lookback=60, scale=2.0),
        "TF_120": TrendFollowingSignal(lookback=120, scale=2.0),
        "Ensemble_MR+TF": EnsembleSignal([
            SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=5),
            TrendFollowingSignal(lookback=60, scale=2.0),
        ]),
    }

    # ── 1. Full grid: all signals × all assets ───────────────────────────
    print("\n[1/4] Running all signal variants on all assets (IN-SAMPLE)...")
    all_results = {}
    rows = []

    for ticker, cost_model in ASSET_COSTS.items():
        if ticker not in universe:
            continue
        splits = split_data(universe[ticker], config)
        is_data = splits["in_sample"]

        for sig_name, sig_obj in signal_variants.items():
            result, report = run_single(is_data, cost_model, sig_obj)
            key = f"{ticker}|{sig_name}"
            all_results[key] = {"result": result, "report": report}

            rows.append({
                "ticker": ticker,
                "signal": sig_name,
                "sharpe": report["sharpe_ratio"],
                "ann_return": report["annualised_return"],
                "ann_vol": report["annualised_volatility"],
                "max_dd": report["max_drawdown"],
                "pct_pos_mo": report["pct_positive_months"],
                "turnover": report.get("total_turnover", 0),
                "costs": report.get("total_costs", 0),
            })

    grid_df = pd.DataFrame(rows)

    # Print pivot: Sharpe by ticker × signal
    print("\n── Sharpe Ratio Grid ──")
    sharpe_pivot = grid_df.pivot(index="ticker", columns="signal", values="sharpe")
    print(sharpe_pivot.to_string(float_format=lambda x: f"{x:.3f}"))

    # Print pivot: Turnover
    print("\n── Total Turnover Grid ──")
    to_pivot = grid_df.pivot(index="ticker", columns="signal", values="turnover")
    print(to_pivot.to_string(float_format=lambda x: f"{x:.0f}"))

    # ── 2. Smoothed MR: turnover improvement ────────────────────────────
    print("\n[2/4] Smoothed MR vs Raw MR (SPY)...")
    spy_rows = grid_df[grid_df["ticker"] == "SPY"]
    spy_compare = spy_rows[["signal", "sharpe", "ann_return", "turnover", "costs"]].copy()
    spy_compare = spy_compare.set_index("signal")
    print(spy_compare.to_string(float_format=lambda x: f"{x:.4f}"))

    # ── 3. Identify viable sub-strategies ────────────────────────────────
    print("\n[3/4] Viable sub-strategies (Sharpe > 0 net of costs)...")
    viable = grid_df[grid_df["sharpe"] > 0].sort_values("sharpe", ascending=False)
    if len(viable) > 0:
        print(viable[["ticker", "signal", "sharpe", "ann_return", "turnover", "pct_pos_mo"]].to_string(
            index=False, float_format=lambda x: f"{x:.3f}",
        ))
    else:
        print("  No sub-strategies with positive Sharpe.")

    print("\n  Also considering strategies with negative Sharpe but low turnover")
    print("  (may contribute diversification value in a portfolio):")
    low_to_candidates = grid_df[
        (grid_df["sharpe"] > -0.3) & (grid_df["turnover"] < 200)
    ].sort_values("sharpe", ascending=False)
    if len(low_to_candidates) > 0:
        print(low_to_candidates[["ticker", "signal", "sharpe", "ann_return", "turnover"]].to_string(
            index=False, float_format=lambda x: f"{x:.3f}",
        ))

    # ── 4. Charts ────────────────────────────────────────────────────────
    print("\n[4/4] Generating charts...")

    # Chart 1: SPY — all signal variants equity curves
    fig, ax = plt.subplots(figsize=(14, 7))
    spy_signals = ["RawMR", "SmoothMR_h5", "SmoothMR_h10", "TF_60", "Ensemble_MR+TF"]
    colors = {"RawMR": "red", "SmoothMR_h5": "orange", "SmoothMR_h10": "goldenrod",
              "TF_60": "blue", "Ensemble_MR+TF": "green"}
    for sig_name in spy_signals:
        key = f"SPY|{sig_name}"
        if key in all_results:
            eq = all_results[key]["result"].equity / 100_000
            ax.plot(eq.index, eq.values, label=sig_name, linewidth=1.3,
                    color=colors.get(sig_name, "gray"))
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title("SPY — All Signal Variants (In-Sample)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Equity (multiple)")
    ax.legend(framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(CHART_DIR / "01_spy_all_signals.png", dpi=150)
    plt.close(fig)
    print("  ✓ 01_spy_all_signals.png")

    # Chart 2: BTC — signal variants
    fig, ax = plt.subplots(figsize=(14, 7))
    btc_signals = ["RawMR", "SmoothMR_h5", "TF_60", "TF_120", "Ensemble_MR+TF"]
    colors_btc = {"RawMR": "red", "SmoothMR_h5": "orange",
                  "TF_60": "blue", "TF_120": "navy", "Ensemble_MR+TF": "green"}
    for sig_name in btc_signals:
        key = f"BTC-USD|{sig_name}"
        if key in all_results:
            eq = all_results[key]["result"].equity / 100_000
            ax.plot(eq.index, eq.values, label=sig_name, linewidth=1.3,
                    color=colors_btc.get(sig_name, "gray"))
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title("BTC-USD — All Signal Variants (In-Sample)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Equity (multiple)")
    ax.legend(framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(CHART_DIR / "02_btc_all_signals.png", dpi=150)
    plt.close(fig)
    print("  ✓ 02_btc_all_signals.png")

    # Chart 3: Sharpe heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot_data = sharpe_pivot.values
    im = ax.imshow(pivot_data, cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(sharpe_pivot.columns)))
    ax.set_yticks(range(len(sharpe_pivot.index)))
    ax.set_xticklabels(sharpe_pivot.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(sharpe_pivot.index)
    for i in range(pivot_data.shape[0]):
        for j in range(pivot_data.shape[1]):
            v = pivot_data[i, j]
            color = "white" if abs(v) > 0.5 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color=color)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Sharpe Ratio: Asset × Signal (In-Sample)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "03_sharpe_heatmap.png", dpi=150)
    plt.close(fig)
    print("  ✓ 03_sharpe_heatmap.png")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUB-STRATEGY MENU FOR PORTFOLIO CONSTRUCTION (Milestone 8)")
    print("=" * 70)

    menu_items = [
        ("BTC-USD", "TF_60", "Primary alpha source"),
        ("BTC-USD", "TF_120", "Longer-term crypto trend"),
        ("ETH-USD", "TF_60", "Diversifies BTC exposure"),
        ("SPY", "SmoothMR_h5", "Equity mean-reversion"),
        ("SPY", "SmoothMR_h10", "Lower turnover MR variant"),
        ("QQQ", "SmoothMR_h5", "Tech-heavy MR"),
        ("SPY", "Ensemble_MR+TF", "Blended equity signal"),
        ("GC=F", "TF_120", "Gold trend — decorrelation"),
    ]

    for ticker, sig_name, note in menu_items:
        key = f"{ticker}|{sig_name}"
        if key in all_results:
            r = all_results[key]["report"]
            status = "✓" if r["sharpe_ratio"] > 0 else "○"
            print(f"  {status} {ticker:10s} {sig_name:18s} "
                  f"Sharpe={r['sharpe_ratio']:+.3f}  "
                  f"TO={r.get('total_turnover', 0):6.0f}  "
                  f"— {note}")

    print(f"\nCharts saved to: {CHART_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
