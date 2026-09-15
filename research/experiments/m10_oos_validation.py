"""
Milestone 10 — Out-of-Sample and Walk-Forward Validation
==========================================================

The most critical milestone: does the in-sample edge survive on unseen data?

Data splits:
  • In-sample:       2018-01-01 → 2023-03-31  (~5.25 years)
  • Validation:      2023-04-01 → 2025-01-31  (~1.8 years)
  • Out-of-sample:   2025-02-01 → 2026-08-31  (~1.6 years)

Tests:
  1. Run SAME portfolio (same signals, same params) on validation data.
  2. Run on out-of-sample data.
  3. Walk-forward: retrain vol-scaling on rolling windows.
  4. Compare IS vs VAL vs OOS performance.

CRITICAL RULE: No parameters are changed based on validation results.
We use the exact same config from M8/M9 milestones.

Usage:
    python research/experiments/m10_oos_validation.py
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

CHART_DIR = PROJECT_ROOT / "research" / "experiments" / "m10_charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)


def get_sub_strategies() -> list[SubStrategy]:
    """Exact same sub-strategies from M8/M9.  No changes."""
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


PORTFOLIO_PARAMS = dict(
    vol_target=0.10,
    vol_lookback=63,
    max_gross_exposure=1.0,
    dd_threshold=-0.05,
    concentration_limit=0.40,
)


def run_on_split(universe, split_name):
    """Run the portfolio on a given data split."""
    result = build_portfolio(universe, get_sub_strategies(), **PORTFOLIO_PARAMS)
    report = full_report(result.net_returns, costs=result.costs)
    monthly = monthly_returns(result.net_returns)
    return result, report, monthly


def walk_forward(universe_full, config, window_years=3, step_months=6):
    """Walk-forward analysis: rolling train/test windows.

    For each step:
      1. Train period: past `window_years` years
      2. Test period: next `step_months` months
      3. Run portfolio on test period using vol-scaling learned from train

    The signals themselves have no trainable parameters — only the
    vol-scaling adapts to recent data (which it does naturally through
    the rolling window).
    """
    all_tickers = {"SPY", "QQQ", "BTC-USD", "ETH-USD", "GC=F"}
    # Find common date range across all assets
    common_dates = None
    for t in all_tickers:
        if t in universe_full:
            idx = universe_full[t].index
            if common_dates is None:
                common_dates = idx
            else:
                common_dates = common_dates.intersection(idx)
    common_dates = common_dates.sort_values()

    start = common_dates[0]
    end = common_dates[-1]

    # Generate walk-forward windows
    train_days = int(window_years * 252)
    step_days = int(step_months * 21)

    windows = []
    cursor = train_days
    while cursor + step_days <= len(common_dates):
        train_end_idx = cursor
        test_start_idx = cursor
        test_end_idx = min(cursor + step_days, len(common_dates))

        train_dates = common_dates[:train_end_idx]
        test_dates = common_dates[test_start_idx:test_end_idx]

        if len(test_dates) < 10:
            break

        windows.append((train_dates, test_dates))
        cursor += step_days

    # Run each window
    all_test_returns = []
    window_results = []

    for i, (train_dates, test_dates) in enumerate(windows):
        # Build universe for test period
        # The portfolio allocator will use the test period data
        # Vol-scaling will initially use the warmup from train period
        # We include train + test data so vol estimation has history
        all_dates = train_dates.append(test_dates).unique().sort_values()

        wf_universe = {}
        for t in all_tickers:
            if t in universe_full:
                mask = universe_full[t].index.isin(all_dates)
                wf_universe[t] = universe_full[t][mask]

        try:
            result = build_portfolio(wf_universe, get_sub_strategies(), **PORTFOLIO_PARAMS)

            # Extract only the test period returns
            test_mask = result.net_returns.index.isin(test_dates)
            test_returns = result.net_returns[test_mask]

            if len(test_returns) > 0:
                all_test_returns.append(test_returns)
                rep = full_report(test_returns)
                window_results.append({
                    "window": i + 1,
                    "test_start": test_dates[0].strftime("%Y-%m"),
                    "test_end": test_dates[-1].strftime("%Y-%m"),
                    "n_days": len(test_returns),
                    "total_return": (1 + test_returns).prod() - 1,
                    "sharpe": rep["sharpe_ratio"],
                    "max_dd": rep["max_drawdown"],
                })
        except Exception as e:
            print(f"  Window {i+1} failed: {e}")

    # Chain all test returns
    if all_test_returns:
        chained = pd.concat(all_test_returns)
        chained = chained[~chained.index.duplicated(keep='first')]
        chained = chained.sort_index()
    else:
        chained = pd.Series(dtype=float)

    return chained, pd.DataFrame(window_results)


def main():
    print("=" * 70)
    print("MILESTONE 10 — Out-of-Sample & Walk-Forward Validation")
    print("=" * 70)

    config = load_config()
    universe = load_universe(config)

    # Prepare data splits
    split_universes = {}
    for split_name in ["in_sample", "validation", "out_of_sample"]:
        split_u = {}
        for ticker, df in universe.items():
            splits = split_data(df, config)
            if split_name in splits and len(splits[split_name]) > 0:
                split_u[ticker] = splits[split_name]
        split_universes[split_name] = split_u

    # ── 1. Run on all three splits ───────────────────────────────────────
    print("\n[1/4] Running portfolio on all data splits...")
    print("       (Same params from M8/M9 — NO changes)")

    split_results = {}
    for split_name in ["in_sample", "validation", "out_of_sample"]:
        split_u = split_universes[split_name]
        if not split_u:
            print(f"  ⚠ {split_name}: No data available, skipping.")
            continue

        try:
            result, report, monthly = run_on_split(split_u, split_name)
            split_results[split_name] = {
                "result": result, "report": report, "monthly": monthly,
            }
            print(f"  {split_name:15s}: Sharpe={report['sharpe_ratio']:.3f}, "
                  f"Ann.Ret={report['annualised_return']:.2%}, "
                  f"MaxDD={report['max_drawdown']:.2%}, "
                  f"%PosMo={report['pct_positive_months']:.1%}, "
                  f"AvgMo={report['avg_monthly_return']:.2%}")
        except Exception as e:
            print(f"  {split_name:15s}: FAILED — {e}")

    # ── 2. Comparison table ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CROSS-SPLIT COMPARISON")
    print("=" * 70)

    comparison_rows = []
    for split_name, data in split_results.items():
        r = data["report"]
        comparison_rows.append({
            "Split": split_name,
            "Period": f"{data['result'].net_returns.index[0].strftime('%Y-%m')} → "
                      f"{data['result'].net_returns.index[-1].strftime('%Y-%m')}",
            "Days": r["n_trading_days"],
            "Ann.Return": f"{r['annualised_return']:.2%}",
            "Ann.Vol": f"{r['annualised_volatility']:.2%}",
            "Sharpe": f"{r['sharpe_ratio']:.3f}",
            "Sortino": f"{r['sortino_ratio']:.3f}",
            "MaxDD": f"{r['max_drawdown']:.2%}",
            "WinRate": f"{r['win_rate']:.1%}",
            "%PosMo": f"{r['pct_positive_months']:.1%}",
            "AvgMo": f"{r['avg_monthly_return']:.2%}",
        })
    comp_df = pd.DataFrame(comparison_rows).set_index("Split")
    print(comp_df.to_string())

    # ── 3. Walk-forward analysis ─────────────────────────────────────────
    print("\n[2/4] Walk-forward analysis (3-year window, 6-month steps)...")
    wf_returns, wf_windows = walk_forward(universe, config)

    if len(wf_windows) > 0:
        print("\nWalk-Forward Windows:")
        print(wf_windows.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

        wf_report = full_report(wf_returns)
        print(f"\nWalk-Forward Aggregate:")
        print(f"  Sharpe: {wf_report['sharpe_ratio']:.3f}")
        print(f"  Ann. Return: {wf_report['annualised_return']:.2%}")
        print(f"  Max DD: {wf_report['max_drawdown']:.2%}")
        print(f"  % Pos Months: {wf_report['pct_positive_months']:.1%}")

    # ── 4. Sharpe decay analysis ─────────────────────────────────────────
    print("\n[3/4] Sharpe decay analysis...")
    if "in_sample" in split_results and "validation" in split_results:
        is_sharpe = split_results["in_sample"]["report"]["sharpe_ratio"]
        val_sharpe = split_results["validation"]["report"]["sharpe_ratio"]
        decay = (is_sharpe - val_sharpe) / is_sharpe * 100 if is_sharpe != 0 else 0
        print(f"  IS Sharpe:  {is_sharpe:.3f}")
        print(f"  VAL Sharpe: {val_sharpe:.3f}")
        print(f"  Decay:      {decay:.1f}%")

        if decay < 30:
            print("  ✓ Healthy: <30% decay suggests genuine edge.")
        elif decay < 50:
            print("  ○ Moderate: 30-50% decay, some overfitting possible.")
        else:
            print("  ✗ Significant: >50% decay, likely overfitting.")

    if "out_of_sample" in split_results:
        oos_sharpe = split_results["out_of_sample"]["report"]["sharpe_ratio"]
        oos_decay = (is_sharpe - oos_sharpe) / is_sharpe * 100 if is_sharpe != 0 else 0
        print(f"  OOS Sharpe: {oos_sharpe:.3f}")
        print(f"  OOS Decay:  {oos_decay:.1f}%")

    # ── 5. Charts ────────────────────────────────────────────────────────
    print("\n[4/4] Generating charts...")

    # Chart 1: Equity curves for all splits
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = {"in_sample": "blue", "validation": "orange", "out_of_sample": "green"}
    for split_name, data in split_results.items():
        eq = data["result"].equity / data["result"].initial_capital
        ax.plot(eq.index, eq.values, label=split_name, linewidth=1.5,
                color=colors.get(split_name, "gray"))
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title("Portfolio Equity Curves by Data Split", fontsize=14, fontweight="bold")
    ax.set_ylabel("Equity (multiple)")
    ax.legend(framealpha=0.9, fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(CHART_DIR / "01_equity_by_split.png", dpi=150)
    plt.close(fig)
    print("  ✓ 01_equity_by_split.png")

    # Chart 2: Monthly returns for each split
    fig, axes = plt.subplots(len(split_results), 1,
                              figsize=(14, 4 * len(split_results)), sharex=False)
    if len(split_results) == 1:
        axes = [axes]
    for ax, (split_name, data) in zip(axes, split_results.items()):
        monthly = data["monthly"] * 100
        colors_bar = ["green" if r > 0 else "red" for r in monthly.values]
        ax.bar(monthly.index, monthly.values, width=20, color=colors_bar, alpha=0.7)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(2, color="blue", linewidth=0.8, linestyle="--", alpha=0.3)
        pct_pos = (monthly > 0).mean() * 100
        ax.set_title(f"{split_name} — Monthly Returns "
                     f"(avg={monthly.mean():.2f}%, pos={pct_pos:.0f}%)",
                     fontsize=12)
        ax.set_ylabel("Return (%)")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "02_monthly_by_split.png", dpi=150)
    plt.close(fig)
    print("  ✓ 02_monthly_by_split.png")

    # Chart 3: Walk-forward equity curve
    if len(wf_returns) > 0:
        fig, ax = plt.subplots(figsize=(14, 7))
        wf_eq = (1 + wf_returns).cumprod()
        ax.plot(wf_eq.index, wf_eq.values, color="purple", linewidth=1.5)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_title("Walk-Forward Equity Curve (3yr window, 6mo steps)",
                     fontsize=14, fontweight="bold")
        ax.set_ylabel("Equity (multiple)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.tight_layout()
        fig.savefig(CHART_DIR / "03_walk_forward.png", dpi=150)
        plt.close(fig)
        print("  ✓ 03_walk_forward.png")

    # Chart 4: Sharpe by split bar chart
    sharpe_data = {}
    for split_name, data in split_results.items():
        sharpe_data[split_name] = data["report"]["sharpe_ratio"]
    if len(wf_returns) > 0:
        sharpe_data["walk_forward"] = wf_report["sharpe_ratio"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(sharpe_data.keys(), sharpe_data.values(),
                  color=["blue", "orange", "green", "purple"][:len(sharpe_data)])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Sharpe Ratio by Data Split", fontsize=14, fontweight="bold")
    ax.set_ylabel("Sharpe Ratio")
    for bar, v in zip(bars, sharpe_data.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{v:.3f}", ha="center", fontsize=12)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "04_sharpe_by_split.png", dpi=150)
    plt.close(fig)
    print("  ✓ 04_sharpe_by_split.png")

    print(f"\nCharts saved to: {CHART_DIR}/")
    print("=" * 70)
    print("VALIDATION COMPLETE — These results are final.")
    print("=" * 70)


if __name__ == "__main__":
    main()
