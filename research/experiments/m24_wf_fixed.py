"""
M24-WF — Fixed Walk-Forward Validation
=========================================

Fixes the critical Walk-Forward bug: signals with long lookback
(DualMomentum lookback=252) need historical data BEYOND the test
window for warm-up. Previous WF only passed test-window data,
causing signals to produce NaN → zero returns.

Fix: For each fold, feed signal (warmup + test) data, build
portfolio on the full range, then evaluate PnL ONLY on the
test window.

Tests:
  1. Dual Momentum standalone (M22 — fixed WF)
  2. Crypto-only (baseline)
  3. All-Asset Equity-heavy 30/70 (M24 best IS config)
  4. Equity-only DM (M24 best VAL config)

Usage:
    python research/experiments/m24_wf_fixed.py
"""

from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant_research.backtest.costs import (
    CRYPTO_COSTS, EQUITY_COSTS, FUTURES_COSTS, scale_costs,
)
from quant_research.data.loader import load_raw, clean, load_config, split_data
from quant_research.metrics.performance import full_report, sharpe_ratio
from quant_research.portfolio.allocator import SubStrategy, build_portfolio
from quant_research.strategies.trend_following import TrendFollowingSignal
from quant_research.strategies.volatility_signal import VolatilityMeanReversionSignal
from quant_research.strategies.dual_momentum import DualMomentumSignal

CHART_DIR = PROJECT_ROOT / "research" / "experiments" / "m24_charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR = PROJECT_ROOT / "data" / "raw"

PP = dict(vol_target=0.10, vol_lookback=63, max_gross_exposure=1.0,
          dd_threshold=-0.05, concentration_limit=0.40)

# Warmup buffer: must be >= max(all signal lookbacks)
# DualMomentum=252, TrendFollowing=120, Vol=120, so 300 days is safe
WARMUP_BUFFER = 300


def load_and_clean(ticker, asset_class="equity"):
    df = load_raw(ticker, raw_dir=RAW_DIR)
    return clean(df, ticker=ticker, asset_class=asset_class)


def load_all():
    """Load all assets."""
    tickers = {
        "BTC-USD": "crypto", "ETH-USD": "crypto",
        "SPY": "equity", "QQQ": "equity",
        "GC=F": "futures", "TLT": "equity",
    }
    data = {}
    for ticker, ac in tickers.items():
        data[ticker] = load_and_clean(ticker, ac)
    return data


# ── Strategy builders (identical to M24) ─────────────────────────────

def crypto_strategies(weight=1.0):
    return [
        SubStrategy("BTC_TF60", "BTC-USD",
                     TrendFollowingSignal(lookback=60, scale=2.0),
                     CRYPTO_COSTS, weight=weight),
        SubStrategy("ETH_TF120", "ETH-USD",
                     TrendFollowingSignal(lookback=120, scale=2.0),
                     CRYPTO_COSTS, weight=weight),
        SubStrategy("BTC_VOL", "BTC-USD",
                     VolatilityMeanReversionSignal(short_window=10, long_window=60),
                     CRYPTO_COSTS, weight=weight),
        SubStrategy("ETH_VOL", "ETH-USD",
                     VolatilityMeanReversionSignal(short_window=10, long_window=60),
                     CRYPTO_COSTS, weight=weight),
    ]


def equity_dm_strategies(data_dict, weight=1.0):
    dm_tickers = ["SPY", "QQQ", "GC=F", "TLT"]
    dm_data = {t: data_dict[t] for t in dm_tickers if t in data_dict}
    cost_map = {"SPY": EQUITY_COSTS, "QQQ": EQUITY_COSTS,
                "GC=F": FUTURES_COSTS, "TLT": scale_costs(EQUITY_COSTS, 0.5)}
    strats = []
    for ticker in dm_tickers:
        if ticker not in dm_data:
            continue
        sig = DualMomentumSignal(dm_data, ticker, lookback=252, n_top=3)
        strats.append(SubStrategy(
            f"{ticker}_DM", ticker, sig, cost_map.get(ticker, EQUITY_COSTS),
            weight=weight,
        ))
    return strats


# ── Fixed Walk-Forward Engine ────────────────────────────────────────

def walk_forward_fixed(
    all_data: dict[str, pd.DataFrame],
    build_strats_fn,
    test_window: int = 126,
    warmup: int = WARMUP_BUFFER,
    n_folds: int = 15,
    label: str = "Strategy",
) -> list[dict]:
    """Run walk-forward with proper warm-up for long-lookback signals.

    For each fold:
      1. Define test period: [test_start, test_end]
      2. Slice data: [test_start - warmup, test_end] (expanded window)
      3. Build portfolio on expanded data (signals get warm-up history)
      4. Extract returns ONLY from [test_start, test_end]
      5. Compute metrics on test-only returns

    Parameters
    ----------
    all_data : dict
        Full dataset (all dates).
    build_strats_fn : callable
        Function(data_dict) → list[SubStrategy]. Creates strategies
        for the given data slice.
    test_window : int
        Number of trading days in each test period.
    warmup : int
        Historical days to prepend for signal warm-up.
    n_folds : int
        Maximum number of folds.
    label : str
        Name for printing.

    Returns
    -------
    list[dict] with keys: fold, test_start, test_end, sharpe, ret
    """
    # Use SPY dates as reference (equities have more regular dates)
    ref_dates = all_data["SPY"].index
    n = len(ref_dates)

    # Start test folds after enough warmup exists
    first_test_start_idx = warmup + 63  # extra 63d buffer for vol estimation
    max_folds = (n - first_test_start_idx) // test_window
    n_folds = min(n_folds, max_folds)

    results = []

    for fold in range(n_folds):
        test_start_idx = first_test_start_idx + fold * test_window
        test_end_idx = min(test_start_idx + test_window, n)

        if test_end_idx > n or (test_end_idx - test_start_idx) < 42:
            break

        # Dates
        warmup_start_idx = max(0, test_start_idx - warmup)
        warmup_start = ref_dates[warmup_start_idx]
        test_start = ref_dates[test_start_idx]
        test_end = ref_dates[test_end_idx - 1]

        # Slice data: warmup_start → test_end (expanded)
        expanded_data = {}
        for ticker, df in all_data.items():
            sliced = df.loc[warmup_start:test_end]
            if len(sliced) > 0:
                expanded_data[ticker] = sliced

        if len(expanded_data) < 2:
            continue

        try:
            # Build strategies on expanded data (signals see full history)
            strats = build_strats_fn(expanded_data)

            # Run portfolio on expanded data
            r = build_portfolio(expanded_data, strats, **PP)

            # Extract returns ONLY for test window
            test_returns = r.net_returns.loc[test_start:test_end]

            if len(test_returns) < 21:
                continue

            sr = sharpe_ratio(test_returns)
            ann_ret = (1 + test_returns).prod() ** (252 / len(test_returns)) - 1

            results.append({
                "fold": fold + 1,
                "test_start": test_start,
                "test_end": test_end,
                "sharpe": sr,
                "ann_return": ann_ret,
                "n_days": len(test_returns),
            })

        except Exception as e:
            results.append({
                "fold": fold + 1,
                "test_start": test_start,
                "test_end": test_end,
                "sharpe": float("nan"),
                "ann_return": float("nan"),
                "n_days": 0,
                "error": str(e),
            })

    return results


def print_wf_results(results, label):
    """Pretty-print walk-forward results."""
    print(f"\n  {'Fold':>5s} {'Test Period':>25s} {'Days':>5s} │ {'SR':>7s} {'Ann Ret':>9s}")
    print("  " + "─" * 58)

    for r in results:
        if "error" in r:
            print(f"  {r['fold']:>5d} {str(r['test_start'].date()):>12s}→"
                  f"{str(r['test_end'].date()):>11s} {'ERR':>5s} │ {'ERR':>7s} {'ERR':>9s}")
        else:
            print(f"  {r['fold']:>5d} {str(r['test_start'].date()):>12s}→"
                  f"{str(r['test_end'].date()):>11s} {r['n_days']:>5d} │ "
                  f"{r['sharpe']:>+6.3f} {r['ann_return']:>+8.2%}")

    sharpes = [r["sharpe"] for r in results if not np.isnan(r.get("sharpe", float("nan")))]
    if sharpes:
        pos = sum(1 for s in sharpes if s > 0)
        print(f"\n  Summary:")
        print(f"    Folds:          {len(sharpes)}")
        print(f"    Positive folds: {pos}/{len(sharpes)} ({pos/len(sharpes)*100:.0f}%)")
        print(f"    Mean Sharpe:    {np.mean(sharpes):+.3f}")
        print(f"    Median Sharpe:  {np.median(sharpes):+.3f}")
        print(f"    Min / Max:      {min(sharpes):+.3f} / {max(sharpes):+.3f}")
    return sharpes


def main():
    print("=" * 80)
    print("  M24-WF — Fixed Walk-Forward Validation")
    print("=" * 80)
    print(f"\n  Fix: Providing {WARMUP_BUFFER}-day warm-up buffer before each test window")
    print(f"  This ensures DualMomentum (lookback=252) and all other signals")
    print(f"  have sufficient history to produce valid signals.\n")

    all_data = load_all()
    print("  Data loaded:")
    for ticker, df in sorted(all_data.items()):
        print(f"    {ticker:<10s}: {df.index[0].date()} → {df.index[-1].date()} ({len(df)} bars)")

    test_window = 126  # 6 months per fold

    # ==================================================================
    # TEST 1: Equity-only Dual Momentum
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Test 1: Equity-only Dual Momentum (M22)")
    print("═" * 80)

    dm_results = walk_forward_fixed(
        all_data,
        build_strats_fn=lambda d: equity_dm_strategies(d, weight=1.0),
        test_window=test_window,
        label="DualMomentum",
    )
    dm_sharpes = print_wf_results(dm_results, "DualMomentum")

    # ==================================================================
    # TEST 2: Crypto-only (baseline)
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Test 2: Crypto-only Baseline")
    print("═" * 80)

    crypto_results = walk_forward_fixed(
        all_data,
        build_strats_fn=lambda d: crypto_strategies(weight=1.0),
        test_window=test_window,
        label="Crypto-only",
    )
    crypto_sharpes = print_wf_results(crypto_results, "Crypto-only")

    # ==================================================================
    # TEST 3: Equity-heavy 30/70 (best IS config)
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Test 3: Equity-heavy 30/70")
    print("═" * 80)

    eq_heavy_results = walk_forward_fixed(
        all_data,
        build_strats_fn=lambda d: crypto_strategies(0.30) + equity_dm_strategies(d, 0.70),
        test_window=test_window,
        label="Equity-heavy 30/70",
    )
    eq_heavy_sharpes = print_wf_results(eq_heavy_results, "Equity-heavy 30/70")

    # ==================================================================
    # TEST 4: Equal 50/50
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Test 4: Equal 50/50")
    print("═" * 80)

    equal_results = walk_forward_fixed(
        all_data,
        build_strats_fn=lambda d: crypto_strategies(1.0) + equity_dm_strategies(d, 1.0),
        test_window=test_window,
        label="Equal 50/50",
    )
    equal_sharpes = print_wf_results(equal_results, "Equal 50/50")

    # ==================================================================
    # COMPARISON TABLE
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Walk-Forward Comparison (Fixed)")
    print("═" * 80)

    configs = [
        ("Crypto-only", crypto_sharpes),
        ("Equity DM only", dm_sharpes),
        ("Equal 50/50", equal_sharpes),
        ("Equity-heavy 30/70", eq_heavy_sharpes),
    ]

    print(f"\n  {'Config':<25s} │ {'Folds':>6s} {'Pos%':>6s} │ {'Mean':>7s} {'Median':>8s} │ {'Min':>7s} {'Max':>7s}")
    print("  " + "─" * 75)

    best_config = None
    best_mean = -999

    for label, sharpes in configs:
        if not sharpes:
            print(f"  {label:<25s} │ {'N/A':>6s}")
            continue
        pos = sum(1 for s in sharpes if s > 0)
        pct = pos / len(sharpes) * 100
        mean_sr = np.mean(sharpes)
        med_sr = np.median(sharpes)
        print(f"  {label:<25s} │ {len(sharpes):>6d} {pct:>5.0f}% │ {mean_sr:>+6.3f} {med_sr:>+7.3f} │ "
              f"{min(sharpes):>+6.3f} {max(sharpes):>+6.3f}")
        if mean_sr > best_mean:
            best_mean = mean_sr
            best_config = label

    print(f"\n  Best WF config: {best_config} (mean Sharpe {best_mean:+.3f})")

    # ==================================================================
    # CHARTS
    # ==================================================================
    print("\n\nGenerating charts...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. WF Sharpe per fold — all configs
    ax = axes[0, 0]
    n_folds_max = max(len(s) for _, s in configs if s)
    x = np.arange(1, n_folds_max + 1)
    width = 0.2
    for i, (label, sharpes) in enumerate(configs):
        if not sharpes:
            continue
        xs = x[:len(sharpes)] + (i - 1.5) * width
        colors_map = {"Crypto-only": "#e67e22", "Equity DM only": "#3498db",
                      "Equal 50/50": "#9b59b6", "Equity-heavy 30/70": "#2ecc71"}
        ax.bar(xs, sharpes, width, label=label, color=colors_map.get(label, "#999"), alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Walk-Forward Sharpe per Fold", fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 2. Cumulative WF equity curves
    ax = axes[0, 1]
    for label, results_list in [("Crypto-only", crypto_results), ("Equity DM only", dm_results),
                                  ("Equal 50/50", equal_results), ("Equity-heavy 30/70", eq_heavy_results)]:
        sharpes = [r["sharpe"] for r in results_list if not np.isnan(r.get("sharpe", float("nan")))]
        if sharpes:
            # Approximate cumulative return from annualized returns
            cum = 1.0
            cum_vals = [1.0]
            for r in results_list:
                if not np.isnan(r.get("ann_return", float("nan"))):
                    fold_ret = r["ann_return"] * r["n_days"] / 252
                    cum *= (1 + fold_ret)
                    cum_vals.append(cum)
            ax.plot(range(len(cum_vals)), cum_vals, "o-", linewidth=1.5, markersize=4, label=label)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Cumulative Growth")
    ax.set_title("Cumulative WF Return", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Box plot comparison
    ax = axes[1, 0]
    box_data = []
    box_labels = []
    for label, sharpes in configs:
        if sharpes:
            box_data.append(sharpes)
            box_labels.append(label.replace(" ", "\n"))
    if box_data:
        bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
        colors_box = ["#e67e22", "#3498db", "#9b59b6", "#2ecc71"]
        for patch, color in zip(bp["boxes"], colors_box[:len(box_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Sharpe Distribution by Config", fontweight="bold")
    ax.set_ylabel("Sharpe Ratio")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", labelsize=7)

    # 4. Summary bar chart: mean WF Sharpe
    ax = axes[1, 1]
    names = [l for l, s in configs if s]
    means = [np.mean(s) for _, s in configs if s]
    colors_bar = ["#2ecc71" if m > 0 else "#e74c3c" for m in means]
    ax.barh(names, means, color=colors_bar)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Mean Walk-Forward Sharpe", fontweight="bold")
    ax.set_xlabel("Mean Sharpe")
    ax.grid(True, alpha=0.3, axis="x")

    fig.suptitle("M24-WF: Fixed Walk-Forward Validation", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "02_wf_fixed.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 02_wf_fixed.png")

    print(f"\n  Charts saved to: {CHART_DIR}/")
    print("=" * 80)
    print("  M24-WF FIXED — COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
