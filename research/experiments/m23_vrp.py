"""
Milestone 23 — Equity Variance Risk Premium (VRP) Signal
===========================================================

Test VRP (VIX - Realized Vol) as an equity timing signal on SPY/QQQ.

Parts:
  A. Load VIX + equity data
  B. VRP analysis and visualization
  C. Apply VRP signal to SPY and QQQ
  D. Parameter sensitivity
  E. IS / VAL / Walk-Forward
  F. Acceptance criteria

Usage:
    python research/experiments/m23_vrp.py
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
from quant_research.strategies.vrp_signal import VarianceRiskPremiumSignal

CHART_DIR = PROJECT_ROOT / "research" / "experiments" / "m23_charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def load_indicator(ticker, asset_class="equity"):
    df = load_raw(ticker, raw_dir=RAW_DIR)
    return clean(df, ticker=ticker, asset_class=asset_class)


def main():
    print("=" * 80)
    print("  M23 — Variance Risk Premium (VRP) Signal")
    print("=" * 80)

    config = load_config()

    # ==================================================================
    # PART A: LOAD DATA
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Part A: Load Data")
    print("═" * 80)

    vix = load_indicator("VIX")
    spy = load_indicator("SPY")
    qqq = load_indicator("QQQ")

    for name, df in [("VIX", vix), ("SPY", spy), ("QQQ", qqq)]:
        print(f"  {name:<5s}: {df.index[0].date()} → {df.index[-1].date()} ({len(df)} bars)")

    vix_splits = split_data(vix, config)
    spy_splits = split_data(spy, config)
    qqq_splits = split_data(qqq, config)
    vix_is, vix_val = vix_splits["in_sample"], vix_splits["validation"]
    spy_is, spy_val = spy_splits["in_sample"], spy_splits["validation"]
    qqq_is, qqq_val = qqq_splits["in_sample"], qqq_splits["validation"]

    # ==================================================================
    # PART B: VRP ANALYSIS
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Part B: VRP Analysis")
    print("═" * 80)

    # Compute raw VRP on IS
    implied = vix_is["close"] / 100
    realized = spy_is["close"].pct_change().rolling(21).std() * np.sqrt(252)
    implied_aligned = implied.reindex(spy_is.index)
    vrp = implied_aligned - realized

    print(f"\n  VRP Statistics (IS):")
    print(f"    Mean:    {vrp.mean():.4f}")
    print(f"    Std:     {vrp.std():.4f}")
    print(f"    Min:     {vrp.min():.4f}")
    print(f"    Max:     {vrp.max():.4f}")
    print(f"    % positive: {(vrp > 0).mean()*100:.1f}%")
    print(f"    (Implied > Realized {(vrp > 0).mean()*100:.1f}% of the time)")

    # ==================================================================
    # PART C: APPLY VRP TO SPY & QQQ
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Part C: VRP Applied to SPY & QQQ")
    print("═" * 80)

    for target_name, t_is, t_val in [("SPY", spy_is, spy_val), ("QQQ", qqq_is, qqq_val)]:
        sig_is = VarianceRiskPremiumSignal(vix_is)
        sig_val = VarianceRiskPremiumSignal(vix_val)

        strats_is = [SubStrategy(f"{target_name}_VRP", target_name, sig_is, EQUITY_COSTS, 1.0)]
        strats_val = [SubStrategy(f"{target_name}_VRP", target_name, sig_val, EQUITY_COSTS, 1.0)]

        r_is = build_portfolio({target_name: t_is}, strats_is,
                                vol_target=0.10, vol_lookback=63, max_gross_exposure=1.0)
        r_val = build_portfolio({target_name: t_val}, strats_val,
                                 vol_target=0.10, vol_lookback=63, max_gross_exposure=1.0)

        rep_is = full_report(r_is.net_returns)
        rep_val = full_report(r_val.net_returns)

        print(f"\n  {target_name}:")
        print(f"    IS  Sharpe: {rep_is['sharpe_ratio']:+.3f}  Return: {rep_is['annualised_return']:+.2%}  MaxDD: {rep_is['max_drawdown']:.3f}")
        print(f"    VAL Sharpe: {rep_val['sharpe_ratio']:+.3f}  Return: {rep_val['annualised_return']:+.2%}  MaxDD: {rep_val['max_drawdown']:.3f}")

    # ==================================================================
    # PART D: PARAMETER SENSITIVITY
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Part D: Parameter Sensitivity")
    print("═" * 80)

    real_windows = [10, 21, 42, 63]
    z_lookbacks = [42, 63, 126, 252]
    scales = [1.0, 1.5, 2.0, 3.0]

    print(f"\n  --- Realized Window (SPY, z_lb=63, scale=2.0) ---")
    print(f"  {'Real W':>7s} │ {'IS SR':>7s} {'VAL SR':>8s}")
    print("  " + "─" * 25)
    best_is_sr = -999
    best_params = (21, 63, 2.0)
    for rw in real_windows:
        sig_is = VarianceRiskPremiumSignal(vix_is, realized_window=rw, z_lookback=63, scale=2.0)
        sig_val = VarianceRiskPremiumSignal(vix_val, realized_window=rw, z_lookback=63, scale=2.0)
        r_i = build_portfolio({"SPY": spy_is}, [SubStrategy("S", "SPY", sig_is, EQUITY_COSTS, 1.0)],
                               vol_target=0.10, vol_lookback=63, max_gross_exposure=1.0)
        r_v = build_portfolio({"SPY": spy_val}, [SubStrategy("S", "SPY", sig_val, EQUITY_COSTS, 1.0)],
                               vol_target=0.10, vol_lookback=63, max_gross_exposure=1.0)
        sr_i = sharpe_ratio(r_i.net_returns)
        sr_v = sharpe_ratio(r_v.net_returns)
        print(f"  {rw:>7d} │ {sr_i:>+6.3f} {sr_v:>+7.3f}")
        if sr_i > best_is_sr:
            best_is_sr = sr_i
            best_params = (rw, 63, 2.0)

    print(f"\n  --- Z-Score Lookback (SPY, real_w={best_params[0]}, scale=2.0) ---")
    print(f"  {'Z LB':>7s} │ {'IS SR':>7s} {'VAL SR':>8s}")
    print("  " + "─" * 25)
    for zlb in z_lookbacks:
        sig_is = VarianceRiskPremiumSignal(vix_is, realized_window=best_params[0], z_lookback=zlb, scale=2.0)
        sig_val = VarianceRiskPremiumSignal(vix_val, realized_window=best_params[0], z_lookback=zlb, scale=2.0)
        r_i = build_portfolio({"SPY": spy_is}, [SubStrategy("S", "SPY", sig_is, EQUITY_COSTS, 1.0)],
                               vol_target=0.10, vol_lookback=63, max_gross_exposure=1.0)
        r_v = build_portfolio({"SPY": spy_val}, [SubStrategy("S", "SPY", sig_val, EQUITY_COSTS, 1.0)],
                               vol_target=0.10, vol_lookback=63, max_gross_exposure=1.0)
        sr_i = sharpe_ratio(r_i.net_returns)
        sr_v = sharpe_ratio(r_v.net_returns)
        print(f"  {zlb:>7d} │ {sr_i:>+6.3f} {sr_v:>+7.3f}")
        if sr_i > best_is_sr:
            best_is_sr = sr_i
            best_params = (best_params[0], zlb, 2.0)

    rw, zlb, sc = best_params
    print(f"\n  Best config: realized_window={rw}, z_lookback={zlb}")

    # ==================================================================
    # PART E: PORTFOLIO (SPY + QQQ) WITH BEST CONFIG
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Part E: Multi-Asset Portfolio (SPY + QQQ)")
    print("═" * 80)

    sig_spy_is = VarianceRiskPremiumSignal(vix_is, rw, zlb, sc)
    sig_qqq_is = VarianceRiskPremiumSignal(vix_is, rw, zlb, sc)
    strats_is = [
        SubStrategy("SPY_VRP", "SPY", sig_spy_is, EQUITY_COSTS, 1.0),
        SubStrategy("QQQ_VRP", "QQQ", sig_qqq_is, EQUITY_COSTS, 1.0),
    ]
    r_is = build_portfolio({"SPY": spy_is, "QQQ": qqq_is}, strats_is,
                            vol_target=0.10, vol_lookback=63, max_gross_exposure=1.0,
                            concentration_limit=0.60)
    rep_is = full_report(r_is.net_returns)

    sig_spy_val = VarianceRiskPremiumSignal(vix_val, rw, zlb, sc)
    sig_qqq_val = VarianceRiskPremiumSignal(vix_val, rw, zlb, sc)
    strats_val = [
        SubStrategy("SPY_VRP", "SPY", sig_spy_val, EQUITY_COSTS, 1.0),
        SubStrategy("QQQ_VRP", "QQQ", sig_qqq_val, EQUITY_COSTS, 1.0),
    ]
    r_val = build_portfolio({"SPY": spy_val, "QQQ": qqq_val}, strats_val,
                             vol_target=0.10, vol_lookback=63, max_gross_exposure=1.0,
                             concentration_limit=0.60)
    rep_val = full_report(r_val.net_returns)

    print(f"\n  {'Metric':<30s} {'IS':>10s} {'VAL':>10s}")
    print("  " + "-" * 52)
    for key in ["sharpe_ratio", "annualised_return", "annualised_volatility",
                 "max_drawdown", "calmar_ratio", "pct_positive_months"]:
        is_v = rep_is.get(key, 0)
        val_v = rep_val.get(key, 0)
        if "ratio" in key or "drawdown" in key:
            print(f"  {key:<30s} {is_v:>+9.3f} {val_v:>+9.3f}")
        else:
            print(f"  {key:<30s} {is_v:>+9.2%} {val_v:>+9.2%}")

    # ==================================================================
    # PART F: WALK-FORWARD
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Part F: Walk-Forward Validation")
    print("═" * 80)

    all_dates = spy.index
    n = len(all_dates)
    train_window = 504
    test_window = 126  # Longer test window since VRP is slower
    n_folds = min(15, (n - train_window) // test_window)

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
        if len(test_dates) < 42:
            break

        t0, t1 = test_dates[0], test_dates[-1]
        vix_t = vix.loc[t0:t1]
        spy_t = spy.loc[t0:t1]
        qqq_t = qqq.loc[t0:t1]

        try:
            sig_s = VarianceRiskPremiumSignal(vix_t, rw, zlb, sc)
            sig_q = VarianceRiskPremiumSignal(vix_t, rw, zlb, sc)
            strats = [
                SubStrategy("SPY_V", "SPY", sig_s, EQUITY_COSTS, 1.0),
                SubStrategy("QQQ_V", "QQQ", sig_q, EQUITY_COSTS, 1.0),
            ]
            r = build_portfolio({"SPY": spy_t, "QQQ": qqq_t}, strats,
                                 vol_target=0.10, vol_lookback=63, max_gross_exposure=1.0)
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
    # PART G: ACCEPTANCE CRITERIA
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Part G: Acceptance Criteria")
    print("═" * 80)

    # 1.5× costs
    sig_15 = VarianceRiskPremiumSignal(vix_is, rw, zlb, sc)
    r_15 = build_portfolio({"SPY": spy_is},
                            [SubStrategy("S", "SPY", sig_15, scale_costs(EQUITY_COSTS, 1.5), 1.0)],
                            vol_target=0.10, vol_lookback=63, max_gross_exposure=1.0)
    sr_15 = sharpe_ratio(r_15.net_returns)

    criteria = [
        ("IS Sharpe > 0.3", rep_is["sharpe_ratio"], 0.3),
        ("VAL Sharpe > 0", rep_val["sharpe_ratio"], 0.0),
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

    # 1. VRP over time
    ax = axes[0, 0]
    vrp_clean = vrp.dropna()
    ax.fill_between(vrp_clean.index, vrp_clean.values, 0,
                     where=vrp_clean > 0, color="#2ecc71", alpha=0.3, label="VRP > 0 (fearful)")
    ax.fill_between(vrp_clean.index, vrp_clean.values, 0,
                     where=vrp_clean < 0, color="#e74c3c", alpha=0.3, label="VRP < 0 (complacent)")
    ax.plot(vrp_clean.index, vrp_clean.values, color="#333", linewidth=0.5)
    ax.set_title("Variance Risk Premium (IS)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Equity curves
    ax = axes[0, 1]
    eq_is = (1 + r_is.net_returns).cumprod()
    eq_val = (1 + r_val.net_returns).cumprod()
    ax.plot(eq_is.index, eq_is, color="#2ecc71", linewidth=1.5, label="IS")
    ax.plot(eq_val.index, eq_val, color="#e74c3c", linewidth=1.5, label="VAL")
    ax.set_title("VRP Strategy — Equity Curve", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. VIX vs Realized Vol
    ax = axes[1, 0]
    ax.plot(implied.index, implied.values * 100, color="#e74c3c", linewidth=1, label="Implied (VIX)")
    ax.plot(realized.index, realized.values * 100, color="#3498db", linewidth=1, label="Realized")
    ax.set_title("Implied vs Realized Vol (IS)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("Annualized Vol (%)")

    # 4. Walk-forward
    ax = axes[1, 1]
    if wf_sharpes:
        colors = ["#2ecc71" if s > 0 else "#e74c3c" for s in wf_sharpes]
        ax.bar(range(1, len(wf_sharpes)+1), wf_sharpes, color=colors)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Walk-Forward Sharpe per Fold", fontweight="bold")
    ax.set_xlabel("Fold")
    ax.grid(True, alpha=0.3)

    fig.suptitle("M23: Variance Risk Premium (VRP)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "01_vrp.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 01_vrp.png")

    print(f"\n  Charts saved to: {CHART_DIR}/")
    print("=" * 80)
    print("  M23 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
