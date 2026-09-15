"""
Milestone 24 — All-Asset Portfolio Integration
=================================================

Combine the best Crypto strategies (from M15) with the best
Equity strategy (Dual Momentum from M22) into a diversified
All-Asset portfolio.

Portfolio Configs to Compare:
  A. Crypto-only (M15 best: BTC+VOL)
  B. Equity-only (M22: Dual Momentum on SPY/QQQ/GC/TLT)
  C. All-Asset Equal Weight
  D. All-Asset Crypto-heavy (70/30)
  E. All-Asset Equity-heavy (30/70)
  F. All-Asset Risk Parity (inverse-vol weighting)

Usage:
    python research/experiments/m24_all_asset.py
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


def load_and_clean(ticker, asset_class="equity"):
    df = load_raw(ticker, raw_dir=RAW_DIR)
    return clean(df, ticker=ticker, asset_class=asset_class)


def load_all_data(config):
    """Load all assets needed for combined portfolio."""
    tickers = {
        "BTC-USD": "crypto", "ETH-USD": "crypto",
        "SPY": "equity", "QQQ": "equity",
        "GC=F": "futures", "TLT": "equity",
    }
    all_data, is_data, val_data = {}, {}, {}
    for ticker, ac in tickers.items():
        df = load_and_clean(ticker, ac)
        all_data[ticker] = df
        splits = split_data(df, config)
        is_data[ticker] = splits["in_sample"]
        val_data[ticker] = splits["validation"]
    return all_data, is_data, val_data


# ── Strategy Builders ──────────────────────────────────────────────────

def crypto_strategies(weight=1.0):
    """BTC+ETH Trend Following + Volatility MR (M15 best)."""
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
    """Dual Momentum on SPY/QQQ/GC/TLT (M22 best: lb=252, n_top=3)."""
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


def main():
    print("=" * 80)
    print("  M24 — All-Asset Portfolio Integration")
    print("=" * 80)

    config = load_config()
    all_data, is_data, val_data = load_all_data(config)

    print("\n  Assets loaded:")
    for ticker, df in sorted(all_data.items()):
        print(f"    {ticker:<10s}: {df.index[0].date()} → {df.index[-1].date()} ({len(df)} bars)")

    # ==================================================================
    # PART A: INDIVIDUAL SLEEVE BASELINES
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Part A: Individual Sleeve Baselines")
    print("═" * 80)

    # Crypto-only
    r_crypto_is = build_portfolio(is_data, crypto_strategies(), **PP)
    r_crypto_val = build_portfolio(val_data, crypto_strategies(), **PP)
    rep_crypto_is = full_report(r_crypto_is.net_returns)
    rep_crypto_val = full_report(r_crypto_val.net_returns)

    # Equity-only (Dual Momentum)
    r_eq_is = build_portfolio(is_data, equity_dm_strategies(is_data), **PP)
    r_eq_val = build_portfolio(val_data, equity_dm_strategies(val_data), **PP)
    rep_eq_is = full_report(r_eq_is.net_returns)
    rep_eq_val = full_report(r_eq_val.net_returns)

    print(f"\n  {'Sleeve':<20s} │ {'IS SR':>7s} {'VAL SR':>8s} │ {'IS Ret':>8s} {'VAL Ret':>9s} │ {'IS DD':>7s} {'VAL DD':>8s}")
    print("  " + "─" * 75)
    print(f"  {'Crypto (TF+VOL)':<20s} │ {rep_crypto_is['sharpe_ratio']:>+6.3f} {rep_crypto_val['sharpe_ratio']:>+7.3f} │ "
          f"{rep_crypto_is['annualised_return']:>+7.2%} {rep_crypto_val['annualised_return']:>+8.2%} │ "
          f"{rep_crypto_is['max_drawdown']:>+6.3f} {rep_crypto_val['max_drawdown']:>+7.3f}")
    print(f"  {'Equity (DualMom)':<20s} │ {rep_eq_is['sharpe_ratio']:>+6.3f} {rep_eq_val['sharpe_ratio']:>+7.3f} │ "
          f"{rep_eq_is['annualised_return']:>+7.2%} {rep_eq_val['annualised_return']:>+8.2%} │ "
          f"{rep_eq_is['max_drawdown']:>+6.3f} {rep_eq_val['max_drawdown']:>+7.3f}")

    # ==================================================================
    # PART B: ALL-ASSET COMBINATIONS
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Part B: All-Asset Portfolio Combinations")
    print("═" * 80)

    configs = {
        "A. Crypto-only": (1.0, 0.0),
        "B. Equity-only (DM)": (0.0, 1.0),
        "C. Equal (50/50)": (1.0, 1.0),
        "D. Crypto-heavy (70/30)": (0.70, 0.30),
        "E. Equity-heavy (30/70)": (0.30, 0.70),
        "F. Crypto-light (80/20)": (0.80, 0.20),
    }

    results = {}
    print(f"\n  {'Config':<25s} │ {'IS SR':>7s} {'VAL SR':>8s} │ {'IS Ret':>8s} {'VAL Ret':>9s} │ {'IS DD':>7s} {'VAL DD':>8s}")
    print("  " + "─" * 80)

    for label, (cw, ew) in configs.items():
        strats_is = []
        strats_val = []
        if cw > 0:
            strats_is.extend(crypto_strategies(weight=cw))
            strats_val.extend(crypto_strategies(weight=cw))
        if ew > 0:
            strats_is.extend(equity_dm_strategies(is_data, weight=ew))
            strats_val.extend(equity_dm_strategies(val_data, weight=ew))

        try:
            r_is = build_portfolio(is_data, strats_is, **PP)
            r_val = build_portfolio(val_data, strats_val, **PP)
            rep_is = full_report(r_is.net_returns)
            rep_val = full_report(r_val.net_returns)

            results[label] = {
                "is": rep_is, "val": rep_val,
                "r_is": r_is, "r_val": r_val,
            }

            print(f"  {label:<25s} │ {rep_is['sharpe_ratio']:>+6.3f} {rep_val['sharpe_ratio']:>+7.3f} │ "
                  f"{rep_is['annualised_return']:>+7.2%} {rep_val['annualised_return']:>+8.2%} │ "
                  f"{rep_is['max_drawdown']:>+6.3f} {rep_val['max_drawdown']:>+7.3f}")
        except Exception as e:
            print(f"  {label:<25s} │ ERROR: {e}")

    # ==================================================================
    # PART C: CORRELATION BETWEEN SLEEVES
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Part C: Correlation Between Sleeves")
    print("═" * 80)

    crypto_ret_is = r_crypto_is.net_returns
    eq_ret_is = r_eq_is.net_returns
    # Align indices
    common = crypto_ret_is.index.intersection(eq_ret_is.index)
    corr = crypto_ret_is.loc[common].corr(eq_ret_is.loc[common])
    print(f"\n  Crypto ↔ Equity daily return correlation (IS): {corr:+.3f}")

    crypto_ret_val = r_crypto_val.net_returns
    eq_ret_val = r_eq_val.net_returns
    common_val = crypto_ret_val.index.intersection(eq_ret_val.index)
    if len(common_val) > 0:
        corr_val = crypto_ret_val.loc[common_val].corr(eq_ret_val.loc[common_val])
        print(f"  Crypto ↔ Equity daily return correlation (VAL): {corr_val:+.3f}")

    print(f"\n  Interpretation:")
    if abs(corr) < 0.3:
        print(f"    ✓ Low correlation ({corr:+.3f}) → Good diversification benefit!")
    elif abs(corr) < 0.6:
        print(f"    ~ Moderate correlation ({corr:+.3f}) → Some diversification benefit")
    else:
        print(f"    ✗ High correlation ({corr:+.3f}) → Limited diversification benefit")

    # ==================================================================
    # PART D: BEST CONFIG — DETAILED METRICS
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Part D: Detailed Analysis of Best Configs")
    print("═" * 80)

    # Find best VAL Sharpe
    best_label = max(results.keys(), key=lambda k: results[k]["val"]["sharpe_ratio"])
    best = results[best_label]

    print(f"\n  Best VAL config: {best_label}")
    print(f"\n  {'Metric':<30s} {'IS':>10s} {'VAL':>10s}")
    print("  " + "-" * 52)
    for key in ["sharpe_ratio", "annualised_return", "annualised_volatility",
                 "max_drawdown", "calmar_ratio", "pct_positive_months",
                 "avg_monthly_return"]:
        is_v = best["is"].get(key, 0)
        val_v = best["val"].get(key, 0)
        if "ratio" in key or "drawdown" in key:
            print(f"  {key:<30s} {is_v:>+9.3f} {val_v:>+9.3f}")
        else:
            print(f"  {key:<30s} {is_v:>+9.2%} {val_v:>+9.2%}")

    # ==================================================================
    # PART E: WALK-FORWARD
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Part E: Walk-Forward (Best Config)")
    print("═" * 80)

    # Parse best config weights
    best_cw, best_ew = configs[best_label]

    all_dates = all_data["SPY"].index
    n = len(all_dates)
    train_window = 504
    test_window = 126
    n_folds = min(13, (n - train_window) // test_window)

    wf_sharpes = []
    wf_crypto_sharpes = []
    print(f"\n  Config: {best_label}")
    print(f"  Folds: {n_folds} (train={train_window}d, test={test_window}d)")
    print(f"  {'Fold':>5s} {'Test Period':>25s} │ {'All-Asset':>10s} {'Crypto-only':>12s}")
    print("  " + "─" * 58)

    for fold in range(n_folds):
        train_start = fold * test_window
        train_end = train_start + train_window
        test_end = min(train_end + test_window, n)
        if test_end > n:
            break

        test_dates = all_dates[train_end:test_end]
        if len(test_dates) < 42:
            break

        t0, t1 = test_dates[0], test_dates[-1]
        test_data = {t: df.loc[t0:t1] for t, df in all_data.items() if len(df.loc[t0:t1]) > 0}

        try:
            # All-Asset
            strats = []
            if best_cw > 0:
                strats.extend(crypto_strategies(weight=best_cw))
            if best_ew > 0:
                strats.extend(equity_dm_strategies(test_data, weight=best_ew))
            r = build_portfolio(test_data, strats, **PP)
            sr = sharpe_ratio(r.net_returns)
            wf_sharpes.append(sr)

            # Crypto-only for comparison
            r_c = build_portfolio(test_data, crypto_strategies(), **PP)
            sr_c = sharpe_ratio(r_c.net_returns)
            wf_crypto_sharpes.append(sr_c)

            print(f"  {fold+1:>5d} {str(t0.date()):>12s}→{str(t1.date()):>11s} │ {sr:>+9.3f} {sr_c:>+11.3f}")
        except Exception as e:
            print(f"  {fold+1:>5d} ERROR: {e}")

    if wf_sharpes:
        pos_all = sum(1 for s in wf_sharpes if s > 0)
        pos_crypto = sum(1 for s in wf_crypto_sharpes if s > 0)
        print(f"\n  Walk-Forward Summary:")
        print(f"    {'':>20s} {'All-Asset':>12s} {'Crypto-only':>13s}")
        print(f"    {'Positive folds':>20s} {pos_all}/{len(wf_sharpes):>10d} {pos_crypto}/{len(wf_crypto_sharpes):>11d}")
        print(f"    {'Mean Sharpe':>20s} {np.mean(wf_sharpes):>+11.3f} {np.mean(wf_crypto_sharpes):>+12.3f}")
        print(f"    {'Median Sharpe':>20s} {np.median(wf_sharpes):>+11.3f} {np.median(wf_crypto_sharpes):>+12.3f}")

    # ==================================================================
    # PART F: COST STRESS TEST
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Part F: Cost Stress Test")
    print("═" * 80)

    cost_mults = [1.0, 1.5, 2.0, 3.0]
    print(f"\n  {'Cost ×':>7s} │ {'IS SR':>7s} {'VAL SR':>8s}")
    print("  " + "─" * 25)

    for cm in cost_mults:
        strats_is = []
        strats_val = []
        if best_cw > 0:
            strats_is.extend([
                SubStrategy("BTC_TF60", "BTC-USD", TrendFollowingSignal(60, 2.0),
                             scale_costs(CRYPTO_COSTS, cm), best_cw),
                SubStrategy("ETH_TF120", "ETH-USD", TrendFollowingSignal(120, 2.0),
                             scale_costs(CRYPTO_COSTS, cm), best_cw),
                SubStrategy("BTC_VOL", "BTC-USD", VolatilityMeanReversionSignal(10, 60),
                             scale_costs(CRYPTO_COSTS, cm), best_cw),
                SubStrategy("ETH_VOL", "ETH-USD", VolatilityMeanReversionSignal(10, 60),
                             scale_costs(CRYPTO_COSTS, cm), best_cw),
            ])
            strats_val.extend([
                SubStrategy("BTC_TF60", "BTC-USD", TrendFollowingSignal(60, 2.0),
                             scale_costs(CRYPTO_COSTS, cm), best_cw),
                SubStrategy("ETH_TF120", "ETH-USD", TrendFollowingSignal(120, 2.0),
                             scale_costs(CRYPTO_COSTS, cm), best_cw),
                SubStrategy("BTC_VOL", "BTC-USD", VolatilityMeanReversionSignal(10, 60),
                             scale_costs(CRYPTO_COSTS, cm), best_cw),
                SubStrategy("ETH_VOL", "ETH-USD", VolatilityMeanReversionSignal(10, 60),
                             scale_costs(CRYPTO_COSTS, cm), best_cw),
            ])
        if best_ew > 0:
            dm_tickers = ["SPY", "QQQ", "GC=F", "TLT"]
            dm_is = {t: is_data[t] for t in dm_tickers}
            dm_val = {t: val_data[t] for t in dm_tickers}
            cost_map = {"SPY": EQUITY_COSTS, "QQQ": EQUITY_COSTS,
                        "GC=F": FUTURES_COSTS, "TLT": scale_costs(EQUITY_COSTS, 0.5)}
            for t in dm_tickers:
                strats_is.append(SubStrategy(f"{t}_DM", t,
                    DualMomentumSignal(dm_is, t, 252, 3),
                    scale_costs(cost_map[t], cm), best_ew))
                strats_val.append(SubStrategy(f"{t}_DM", t,
                    DualMomentumSignal(dm_val, t, 252, 3),
                    scale_costs(cost_map[t], cm), best_ew))

        r_i = build_portfolio(is_data, strats_is, **PP)
        r_v = build_portfolio(val_data, strats_val, **PP)
        print(f"  {cm:>6.1f}× │ {sharpe_ratio(r_i.net_returns):>+6.3f} {sharpe_ratio(r_v.net_returns):>+7.3f}")

    # ==================================================================
    # PART G: FINAL VERDICT
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Part G: Final Verdict")
    print("═" * 80)

    print(f"\n  Best All-Asset Config: {best_label}")
    print(f"  vs Crypto-only baseline:")
    print(f"    {'':>25s} {'All-Asset':>12s} {'Crypto-only':>13s} {'Improvement':>13s}")
    
    for metric in ["sharpe_ratio", "annualised_return", "max_drawdown"]:
        all_v = best["val"].get(metric, 0)
        crypto_v = rep_crypto_val.get(metric, 0)
        diff = all_v - crypto_v
        print(f"    {metric:>25s} {all_v:>+11.3f} {crypto_v:>+12.3f} {diff:>+12.3f}")

    # Is All-Asset better than Crypto-only?
    all_better = best["val"]["sharpe_ratio"] > rep_crypto_val["sharpe_ratio"]
    if all_better:
        print(f"\n  ✓ CONCLUSION: All-Asset ({best_label}) OUTPERFORMS Crypto-only on VAL!")
        print(f"    → Use this as the production portfolio.")
    else:
        print(f"\n  ✗ CONCLUSION: Crypto-only still outperforms All-Asset on VAL.")
        print(f"    → Adding Equity did not improve the portfolio.")
        print(f"    → However, All-Asset may still provide better diversification over longer periods.")

    # ==================================================================
    # CHARTS
    # ==================================================================
    print("\n\nGenerating charts...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Equity curves comparison (VAL)
    ax = axes[0, 0]
    for label in ["A. Crypto-only", "B. Equity-only (DM)", best_label]:
        if label in results:
            eq = (1 + results[label]["r_val"].net_returns).cumprod()
            ax.plot(eq.index, eq, linewidth=1.5, label=label)
    ax.set_title("Equity Curves — VAL Period", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("Growth of $1")

    # 2. IS equity curves
    ax = axes[0, 1]
    for label in ["A. Crypto-only", "B. Equity-only (DM)", best_label]:
        if label in results:
            eq = (1 + results[label]["r_is"].net_returns).cumprod()
            ax.plot(eq.index, eq, linewidth=1.5, label=label)
    ax.set_title("Equity Curves — IS Period", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("Growth of $1")

    # 3. Walk-Forward comparison
    ax = axes[1, 0]
    if wf_sharpes and wf_crypto_sharpes:
        x = np.arange(1, len(wf_sharpes) + 1)
        w = 0.35
        ax.bar(x - w/2, wf_sharpes, w, label="All-Asset", color="#3498db")
        ax.bar(x + w/2, wf_crypto_sharpes, w, label="Crypto-only", color="#e67e22")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Fold")
        ax.set_ylabel("Sharpe")
    ax.set_title("Walk-Forward: All-Asset vs Crypto-only", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Portfolio config comparison bar chart
    ax = axes[1, 1]
    labels = list(results.keys())
    val_sharpes = [results[l]["val"]["sharpe_ratio"] for l in labels]
    colors = ["#2ecc71" if s > 0 else "#e74c3c" for s in val_sharpes]
    ax.barh(labels, val_sharpes, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("VAL Sharpe by Portfolio Config", fontweight="bold")
    ax.set_xlabel("Sharpe Ratio")
    ax.grid(True, alpha=0.3, axis="x")

    fig.suptitle("M24: All-Asset Portfolio Integration", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "01_all_asset.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 01_all_asset.png")

    print(f"\n  Charts saved to: {CHART_DIR}/")
    print("=" * 80)
    print("  M24 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
