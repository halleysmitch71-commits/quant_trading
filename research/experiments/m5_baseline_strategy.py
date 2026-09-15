"""
Milestone 5 — First Simple Baseline Strategy: Daily Mean-Reversion
===================================================================

This script tests Hypothesis H1 from Milestone 2 on IN-SAMPLE data only.

Hypothesis:  Daily equity returns exhibit mean-reversion.
Signal:      signal[t] = −z_score[t] = −return_1d[t] / rolling_vol[t, 20]
Assets:      SPY, QQQ (strongest negative autocorrelation)
             Also tested on: BTC-USD, ETH-USD, GC=F (comparison)

IMPORTANT:
  • We use IN-SAMPLE data ONLY (2018-01 → 2023-03).
  • Validation and out-of-sample data are NOT touched.
  • We use vol_lookback=20 without optimisation.
  • Transaction costs use the preset CostModel for each asset class.

Usage:
    python research/experiments/m5_baseline_strategy.py
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
    EQUITY_COSTS,
    CRYPTO_COSTS,
    FUTURES_COSTS,
    CostModel,
    cost_model_for_asset_class,
    scale_costs,
)
from quant_research.backtest.engine import run_backtest
from quant_research.data.loader import load_config, load_universe, split_data
from quant_research.metrics.performance import (
    full_report,
    format_report,
    monthly_returns,
    drawdown_series,
)
from quant_research.strategies.mean_reversion import MeanReversionSignal

CHART_DIR = PROJECT_ROOT / "research" / "experiments" / "m5_charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)


def run_strategy_on_asset(
    ticker: str,
    data: pd.DataFrame,
    cost_model: CostModel,
    vol_lookback: int = 20,
) -> dict:
    """Run the mean-reversion strategy on a single asset (in-sample)."""
    sig_obj = MeanReversionSignal(vol_lookback=vol_lookback)
    signal = sig_obj.generate(data)
    prices = data["close"]

    result = run_backtest(
        prices=prices,
        signal=signal,
        initial_capital=100_000,
        cost_model=cost_model,
    )

    report = full_report(
        result.net_returns,
        turnover=result.turnover,
        costs=result.costs,
    )
    report["ticker"] = ticker
    report["cost_model"] = cost_model

    return {
        "ticker": ticker,
        "result": result,
        "report": report,
        "signal_obj": sig_obj,
    }


def plot_equity_curves(results: dict[str, dict]):
    """Plot equity curves for all assets."""
    fig, ax = plt.subplots(figsize=(14, 7))

    for ticker, res in results.items():
        equity = res["result"].equity
        ax.plot(equity.index, equity.values / 100_000, label=ticker, linewidth=1.3)

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title("Mean-Reversion Strategy — In-Sample Equity Curves", fontsize=14, fontweight="bold")
    ax.set_ylabel("Equity (multiple of initial)")
    ax.legend(framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(CHART_DIR / "01_equity_curves.png", dpi=150)
    plt.close(fig)
    print("  ✓ 01_equity_curves.png")


def plot_drawdowns(results: dict[str, dict]):
    """Plot drawdown curves."""
    fig, ax = plt.subplots(figsize=(14, 5))

    for ticker, res in results.items():
        dd = drawdown_series(res["result"].net_returns) * 100
        ax.plot(dd.index, dd.values, label=ticker, linewidth=1.0, alpha=0.8)

    ax.set_title("Drawdown Curves (In-Sample)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(CHART_DIR / "02_drawdowns.png", dpi=150)
    plt.close(fig)
    print("  ✓ 02_drawdowns.png")


def plot_monthly_returns_bar(results: dict[str, dict]):
    """Plot monthly returns for the primary asset (SPY)."""
    if "SPY" not in results:
        return

    monthly = monthly_returns(results["SPY"]["result"].net_returns) * 100
    fig, ax = plt.subplots(figsize=(14, 5))

    colors = ["green" if r > 0 else "red" for r in monthly.values]
    ax.bar(monthly.index, monthly.values, width=20, color=colors, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("SPY Mean-Reversion — Monthly Returns (In-Sample)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Return (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(CHART_DIR / "03_monthly_returns_spy.png", dpi=150)
    plt.close(fig)
    print("  ✓ 03_monthly_returns_spy.png")


def cost_sensitivity_analysis(
    ticker: str,
    data: pd.DataFrame,
    base_cost: CostModel,
) -> pd.DataFrame:
    """Run strategy at different cost multiples."""
    multipliers = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    rows = []

    sig_obj = MeanReversionSignal(vol_lookback=20)
    signal = sig_obj.generate(data)
    prices = data["close"]

    for mult in multipliers:
        cm = scale_costs(base_cost, mult) if mult > 0 else CostModel()
        result = run_backtest(prices=prices, signal=signal, cost_model=cm)
        report = full_report(result.net_returns, turnover=result.turnover, costs=result.costs)
        rows.append({
            "cost_multiple": mult,
            "total_bps": cm.total_bps,
            "ann_return": report["annualised_return"],
            "sharpe": report["sharpe_ratio"],
            "pct_pos_months": report["pct_positive_months"],
            "total_costs": report.get("total_costs", 0),
        })

    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("MILESTONE 5 — Baseline Strategy: Daily Mean-Reversion")
    print("=" * 70)

    # Load data
    config = load_config()
    universe = load_universe(config)

    # Define which assets to test
    test_assets = {
        "SPY": ("equity", EQUITY_COSTS),
        "QQQ": ("equity", EQUITY_COSTS),
        "BTC-USD": ("crypto", CRYPTO_COSTS),
        "ETH-USD": ("crypto", CRYPTO_COSTS),
        "GC=F": ("futures", FUTURES_COSTS),
    }

    # Run on IN-SAMPLE only
    print("\n[1/4] Running strategy on IN-SAMPLE data (2018-01 → 2023-03)...")
    results = {}
    for ticker, (asset_class, cost_model) in test_assets.items():
        if ticker not in universe:
            print(f"  ⚠ {ticker} not in universe, skipping.")
            continue

        splits = split_data(universe[ticker], config)
        is_data = splits["in_sample"]

        if len(is_data) < 30:
            print(f"  ⚠ {ticker} has insufficient in-sample data, skipping.")
            continue

        res = run_strategy_on_asset(ticker, is_data, cost_model)
        results[ticker] = res
        print(f"  {ticker}: Sharpe={res['report']['sharpe_ratio']:.3f}, "
              f"Ann.Return={res['report']['annualised_return']:.2%}, "
              f"MaxDD={res['report']['max_drawdown']:.2%}, "
              f"%Pos.Months={res['report']['pct_positive_months']:.1%}")

    # Summary table
    print("\n" + "=" * 70)
    print("IN-SAMPLE RESULTS SUMMARY")
    print("=" * 70)
    summary_rows = []
    for ticker, res in results.items():
        r = res["report"]
        summary_rows.append({
            "Ticker": ticker,
            "Ann. Return": f"{r['annualised_return']:.2%}",
            "Ann. Vol": f"{r['annualised_volatility']:.2%}",
            "Sharpe": f"{r['sharpe_ratio']:.3f}",
            "Sortino": f"{r['sortino_ratio']:.3f}",
            "Max DD": f"{r['max_drawdown']:.2%}",
            "Win Rate": f"{r['win_rate']:.1%}",
            "% Pos Months": f"{r['pct_positive_months']:.1%}",
            "Avg Monthly": f"{r['avg_monthly_return']:.2%}",
            "Total Costs": f"{r.get('total_costs', 0):.4f}",
        })
    summary_df = pd.DataFrame(summary_rows).set_index("Ticker")
    print(summary_df.to_string())

    # Full report for primary asset
    if "SPY" in results:
        print("\n" + format_report(results["SPY"]["report"]))

    # Charts
    print("\n[2/4] Generating charts...")
    plot_equity_curves(results)
    plot_drawdowns(results)
    plot_monthly_returns_bar(results)

    # Cost sensitivity
    print("\n[3/4] Cost sensitivity analysis (SPY)...")
    if "SPY" in results:
        spy_splits = split_data(universe["SPY"], config)
        cost_sens = cost_sensitivity_analysis("SPY", spy_splits["in_sample"], EQUITY_COSTS)
        print(cost_sens.to_string(index=False))

    # Failure criteria check
    print("\n[4/4] Failure criteria evaluation...")
    print("-" * 50)
    for ticker, res in results.items():
        r = res["report"]
        checks = []

        # Criterion 1: Sharpe > 0.3
        if r["sharpe_ratio"] >= 0.3:
            checks.append(f"  ✓ Sharpe = {r['sharpe_ratio']:.3f} ≥ 0.3")
        else:
            checks.append(f"  ✗ Sharpe = {r['sharpe_ratio']:.3f} < 0.3  ← FAIL")

        # Criterion 2: > 50% positive months
        if r["pct_positive_months"] > 0.50:
            checks.append(f"  ✓ %Pos Months = {r['pct_positive_months']:.1%} > 50%")
        else:
            checks.append(f"  ✗ %Pos Months = {r['pct_positive_months']:.1%} ≤ 50%  ← FAIL")

        print(f"\n{ticker}:")
        for c in checks:
            print(c)

    # Cost robustness check for SPY
    if "SPY" in results:
        spy_splits = split_data(universe["SPY"], config)
        doubled_cm = scale_costs(EQUITY_COSTS, 2.0)
        doubled_res = run_strategy_on_asset("SPY", spy_splits["in_sample"], doubled_cm)
        dr = doubled_res["report"]
        print(f"\nSPY at 2× costs (20 bps one-way):")
        if dr["sharpe_ratio"] > 0:
            print(f"  ✓ Still profitable: Sharpe = {dr['sharpe_ratio']:.3f}")
        else:
            print(f"  ✗ Not profitable at 2× costs: Sharpe = {dr['sharpe_ratio']:.3f}")

    print(f"\nCharts saved to: {CHART_DIR}/")
    print("=" * 70)
    print("NOTE: These are IN-SAMPLE results only.")
    print("Validation will occur in Milestone 10.")
    print("=" * 70)


if __name__ == "__main__":
    main()
