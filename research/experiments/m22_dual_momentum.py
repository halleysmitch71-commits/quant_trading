"""
Milestone 22 — Dual Momentum (Absolute + Relative)
=====================================================

Test Dual Momentum (Antonacci 2014) across SPY, QQQ, GC=F, TLT.
Long-only: invest in top N assets with positive momentum, else cash.

Parts:
  A. Load multi-asset data
  B. Dual Momentum signal analysis
  C. Parameter sensitivity (lookback, n_top)
  D. Compare: Absolute-only, Relative-only, Dual
  E. IS / VAL / Walk-Forward validation
  F. Acceptance criteria

Usage:
    python research/experiments/m22_dual_momentum.py
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

from quant_research.backtest.costs import EQUITY_COSTS, scale_costs
from quant_research.data.loader import load_raw, clean, load_config, split_data
from quant_research.metrics.performance import full_report, sharpe_ratio
from quant_research.portfolio.allocator import SubStrategy, build_portfolio
from quant_research.strategies.dual_momentum import DualMomentumSignal

CHART_DIR = PROJECT_ROOT / "research" / "experiments" / "m22_charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

RAW_DIR = PROJECT_ROOT / "data" / "raw"
DM_TICKERS = ["SPY", "QQQ", "GC=F", "TLT"]
COST_MAP = {"SPY": EQUITY_COSTS, "QQQ": EQUITY_COSTS,
            "GC=F": scale_costs(EQUITY_COSTS, 0.5),  # futures cheaper
            "TLT": scale_costs(EQUITY_COSTS, 0.5)}


def load_and_split(tickers, config):
    """Load, clean, and split data for tickers."""
    all_data = {}
    is_data = {}
    val_data = {}
    for ticker in tickers:
        ac = "futures" if "=" in ticker else "equity"
        df = load_raw(ticker, raw_dir=RAW_DIR)
        df = clean(df, ticker=ticker, asset_class=ac)
        all_data[ticker] = df
        splits = split_data(df, config)
        is_data[ticker] = splits["in_sample"]
        val_data[ticker] = splits["validation"]
    return all_data, is_data, val_data


def run_dm_portfolio(data_dict, lookback, n_top, cost_mult=1.0):
    """Build and evaluate a dual momentum portfolio."""
    strats = []
    for ticker in DM_TICKERS:
        if ticker not in data_dict:
            continue
        sig = DualMomentumSignal(
            asset_data=data_dict, target_asset=ticker,
            lookback=lookback, n_top=n_top,
        )
        cost = scale_costs(COST_MAP.get(ticker, EQUITY_COSTS), cost_mult)
        strats.append(SubStrategy(f"{ticker}_DM", ticker, sig, cost, weight=1.0))

    r = build_portfolio(data_dict, strats,
                         vol_target=0.10, vol_lookback=63,
                         max_gross_exposure=1.0, concentration_limit=0.60)
    return r


def main():
    print("=" * 80)
    print("  M22 — Dual Momentum (Absolute + Relative)")
    print("=" * 80)

    config = load_config()

    # ==================================================================
    # PART A: LOAD DATA
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Part A: Load Multi-Asset Data")
    print("═" * 80)

    all_data, is_data, val_data = load_and_split(DM_TICKERS, config)
    for ticker, df in sorted(all_data.items()):
        print(f"  {ticker:<8s}: {df.index[0].date()} → {df.index[-1].date()} ({len(df)} bars)")

    # ==================================================================
    # PART B: SIGNAL ANALYSIS
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Part B: Dual Momentum Signal Analysis")
    print("═" * 80)

    sig = DualMomentumSignal(is_data, "SPY", lookback=252, n_top=2)
    all_signals = sig.generate_all()
    print(f"\n  Signal matrix shape: {all_signals.shape}")
    print(f"\n  Allocation frequency (IS, last 252 bars):")
    last_year = all_signals.iloc[-252:]
    for ticker in DM_TICKERS:
        if ticker in last_year.columns:
            pct = (last_year[ticker] == 1.0).mean() * 100
            print(f"    {ticker:<8s}: Long {pct:.1f}% of the time")

    cash_pct = (last_year.sum(axis=1) == 0).mean() * 100
    print(f"    {'CASH':<8s}: {cash_pct:.1f}% of the time (all assets negative momentum)")

    # ==================================================================
    # PART C: PARAMETER SENSITIVITY
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Part C: Parameter Sensitivity")
    print("═" * 80)

    lookbacks = [63, 126, 189, 252]
    n_tops = [1, 2, 3]

    print(f"\n  {'Lookback':>8s} {'N_top':>6s} │ {'IS SR':>7s} {'VAL SR':>8s} │ {'IS Ret':>8s} {'VAL Ret':>9s}")
    print("  " + "─" * 55)

    best_is_sr = -999
    best_config = None

    for lb in lookbacks:
        for nt in n_tops:
            try:
                r_is = run_dm_portfolio(is_data, lb, nt)
                r_val = run_dm_portfolio(val_data, lb, nt)
                sr_is = sharpe_ratio(r_is.net_returns)
                sr_val = sharpe_ratio(r_val.net_returns)
                rep_is = full_report(r_is.net_returns)
                rep_val = full_report(r_val.net_returns)
                print(f"  {lb:>8d} {nt:>6d} │ {sr_is:>+6.3f} {sr_val:>+7.3f} │ "
                      f"{rep_is['annualised_return']:>+7.2%} {rep_val['annualised_return']:>+8.2%}")
                if sr_is > best_is_sr:
                    best_is_sr = sr_is
                    best_config = (lb, nt)
            except Exception as e:
                print(f"  {lb:>8d} {nt:>6d} │ ERROR: {e}")

    print(f"\n  Best IS config: lookback={best_config[0]}, n_top={best_config[1]}")

    # ==================================================================
    # PART D: COMPARE ABSOLUTE vs RELATIVE vs DUAL
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Part D: Absolute vs Relative vs Dual")
    print("═" * 80)

    lb, nt = best_config

    # Dual (standard)
    r_dual_is = run_dm_portfolio(is_data, lb, nt)
    r_dual_val = run_dm_portfolio(val_data, lb, nt)
    rep_dual_is = full_report(r_dual_is.net_returns)
    rep_dual_val = full_report(r_dual_val.net_returns)

    # Buy-and-hold SPY benchmark
    spy_ret_is = is_data["SPY"]["close"].pct_change().fillna(0)
    spy_ret_val = val_data["SPY"]["close"].pct_change().fillna(0)

    print(f"\n  {'Strategy':<25s} │ {'IS SR':>7s} {'VAL SR':>8s} │ {'IS MaxDD':>9s} {'VAL MaxDD':>10s}")
    print("  " + "─" * 65)
    print(f"  {'Dual Momentum':<25s} │ {rep_dual_is['sharpe_ratio']:>+6.3f} {rep_dual_val['sharpe_ratio']:>+7.3f} │ "
          f"{rep_dual_is['max_drawdown']:>+8.3f} {rep_dual_val['max_drawdown']:>+9.3f}")
    print(f"  {'Buy-and-Hold SPY':<25s} │ {sharpe_ratio(spy_ret_is):>+6.3f} {sharpe_ratio(spy_ret_val):>+7.3f} │ "
          f"{(spy_ret_is.add(1).cumprod().div(spy_ret_is.add(1).cumprod().cummax()) - 1).min():>+8.3f} "
          f"{(spy_ret_val.add(1).cumprod().div(spy_ret_val.add(1).cumprod().cummax()) - 1).min():>+9.3f}")

    # ==================================================================
    # PART E: WALK-FORWARD
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Part E: Walk-Forward Validation")
    print("═" * 80)

    all_dates = all_data["SPY"].index
    n = len(all_dates)
    train_window = 504
    test_window = 63
    n_folds = min(19, (n - train_window) // test_window)

    wf_sharpes = []
    print(f"\n  Folds: {n_folds} (train={train_window}d, test={test_window}d)")
    print(f"  {'Fold':>5s} {'Test Period':>25s} │ {'SR':>7s}")
    print("  " + "─" * 42)

    for fold in range(n_folds):
        train_start = fold * test_window
        train_end = train_start + train_window
        test_end = min(train_end + test_window, n)
        if test_end > n:
            break

        test_dates = all_dates[train_end:test_end]
        if len(test_dates) < 21:
            break

        t0, t1 = test_dates[0], test_dates[-1]
        test_data = {t: df.loc[t0:t1] for t, df in all_data.items() if len(df.loc[t0:t1]) > 0}

        if len(test_data) < len(DM_TICKERS):
            continue

        try:
            r = run_dm_portfolio(test_data, lb, nt)
            sr = sharpe_ratio(r.net_returns)
            wf_sharpes.append(sr)
            print(f"  {fold+1:>5d} {str(t0.date()):>12s}→{str(t1.date()):>11s} │ {sr:>+6.3f}")
        except Exception as e:
            print(f"  {fold+1:>5d} ERROR: {e}")

    if wf_sharpes:
        pos_folds = sum(1 for s in wf_sharpes if s > 0)
        print(f"\n  Walk-Forward Results:")
        print(f"    Positive folds: {pos_folds}/{len(wf_sharpes)} ({pos_folds/len(wf_sharpes)*100:.0f}%)")
        print(f"    Mean Sharpe:    {np.mean(wf_sharpes):+.3f}")
        print(f"    Median Sharpe:  {np.median(wf_sharpes):+.3f}")
    else:
        pos_folds = 0

    # ==================================================================
    # PART F: ACCEPTANCE CRITERIA
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Part F: Acceptance Criteria")
    print("═" * 80)

    # 1.5× costs
    r_15 = run_dm_portfolio(is_data, lb, nt, cost_mult=1.5)
    sr_15 = sharpe_ratio(r_15.net_returns)

    criteria = [
        ("IS Sharpe > 0.3", rep_dual_is["sharpe_ratio"], 0.3),
        ("VAL Sharpe > 0", rep_dual_val["sharpe_ratio"], 0.0),
        ("WF >50% positive folds", pos_folds/len(wf_sharpes) if wf_sharpes else 0, 0.5),
        ("Survives 1.5× costs (IS SR > 0)", sr_15, 0.0),
    ]

    print(f"\n  {'Criterion':<40s} {'Actual':>8s} {'Target':>8s} {'Result':>8s}")
    print("  " + "-" * 68)
    n_pass = 0
    for desc, actual, target in criteria:
        passed = actual > target if actual is not None and not np.isnan(actual) else False
        n_pass += passed
        print(f"  {desc:<40s} {actual:>+7.3f} {target:>+7.3f} {'✓ PASS' if passed else '✗ FAIL':>8s}")

    verdict = "✓ KEEP — Integrate into portfolio" if n_pass >= 3 else "✗ ELIMINATE — Do not integrate"
    print(f"\n  Result: {n_pass}/{len(criteria)} criteria passed")
    print(f"  Verdict: {verdict}")

    # ==================================================================
    # CHARTS
    # ==================================================================
    print("\nGenerating charts...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Equity curves
    ax = axes[0, 0]
    eq_is = (1 + r_dual_is.net_returns).cumprod()
    eq_val = (1 + r_dual_val.net_returns).cumprod()
    ax.plot(eq_is.index, eq_is, color="#2ecc71", linewidth=1.5, label="IS")
    ax.plot(eq_val.index, eq_val, color="#e74c3c", linewidth=1.5, label="VAL")
    ax.set_title("Dual Momentum — Equity Curve", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Asset allocation over time
    ax = axes[0, 1]
    sigs = DualMomentumSignal(is_data, "SPY", lb, nt).generate_all()
    for i, ticker in enumerate(DM_TICKERS):
        if ticker in sigs.columns:
            ax.fill_between(sigs.index, i, i + sigs[ticker].values, alpha=0.5, label=ticker)
    ax.set_title("Asset Allocation (IS)", fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_ylabel("Allocated")

    # 3. Walk-forward
    ax = axes[1, 0]
    if wf_sharpes:
        colors = ["#2ecc71" if s > 0 else "#e74c3c" for s in wf_sharpes]
        ax.bar(range(1, len(wf_sharpes)+1), wf_sharpes, color=colors)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Walk-Forward Sharpe per Fold", fontweight="bold")
    ax.set_xlabel("Fold")
    ax.grid(True, alpha=0.3)

    # 4. Lookback sensitivity
    ax = axes[1, 1]
    srs = []
    for lb_t in lookbacks:
        try:
            r = run_dm_portfolio(is_data, lb_t, nt)
            srs.append(sharpe_ratio(r.net_returns))
        except:
            srs.append(float("nan"))
    ax.plot(lookbacks, srs, "o-", color="#3498db", linewidth=2, markersize=8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Lookback Sensitivity", fontweight="bold")
    ax.set_xlabel("Lookback (days)")
    ax.set_ylabel("IS Sharpe")
    ax.grid(True, alpha=0.3)

    fig.suptitle("M22: Dual Momentum (Antonacci)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "01_dual_momentum.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 01_dual_momentum.png")

    print(f"\n  Charts saved to: {CHART_DIR}/")
    print("=" * 80)
    print("  M22 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
