"""
Milestone 21 — Cross-Asset Macro Regime Signal
=================================================

Test a macro regime indicator (VIX + TLT + UUP) for equity timing.
Signal tells us Risk-On vs Risk-Off → Long/Short/Neutral equity.

Parts:
  A. Load macro indicator data (VIX, TLT, UUP)
  B. Regime signal visualization
  C. Apply regime signal to SPY and QQQ
  D. Parameter sensitivity
  E. IS / VAL / Walk-Forward validation
  F. Acceptance criteria → PASS / FAIL

Usage:
    python research/experiments/m21_macro_regime.py
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
from quant_research.strategies.macro_regime import MacroRegimeSignal

CHART_DIR = PROJECT_ROOT / "research" / "experiments" / "m21_charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

RAW_DIR = PROJECT_ROOT / "data" / "raw"


def load_indicator(ticker, asset_class="equity"):
    """Load and clean a single indicator."""
    df = load_raw(ticker, raw_dir=RAW_DIR)
    df = clean(df, ticker=ticker, asset_class=asset_class)
    return df


def load_vix():
    """Load VIX data (special handling: volume=0, use close only)."""
    df = load_raw("VIX", raw_dir=RAW_DIR)
    # VIX CSV has standard columns but volume=0
    # clean() works fine since we only need 'close'
    df = clean(df, ticker="VIX", asset_class="equity")
    return df


def main():
    print("=" * 80)
    print("  M21 — Cross-Asset Macro Regime Signal")
    print("=" * 80)

    config = load_config()

    # ==================================================================
    # PART A: LOAD DATA
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Part A: Load Macro Indicator Data")
    print("═" * 80)

    vix = load_vix()
    tlt = load_indicator("TLT")
    uup = load_indicator("UUP")
    spy = load_indicator("SPY")
    qqq = load_indicator("QQQ")

    for name, df in [("VIX", vix), ("TLT", tlt), ("UUP", uup), ("SPY", spy), ("QQQ", qqq)]:
        print(f"  {name:<5s}: {df.index[0].date()} → {df.index[-1].date()} ({len(df)} bars)")

    # Split
    vix_is, vix_val = split_data(vix, config)["in_sample"], split_data(vix, config)["validation"]
    tlt_is, tlt_val = split_data(tlt, config)["in_sample"], split_data(tlt, config)["validation"]
    uup_is, uup_val = split_data(uup, config)["in_sample"], split_data(uup, config)["validation"]
    spy_splits = split_data(spy, config)
    qqq_splits = split_data(qqq, config)
    spy_is, spy_val = spy_splits["in_sample"], spy_splits["validation"]
    qqq_is, qqq_val = qqq_splits["in_sample"], qqq_splits["validation"]

    print(f"\n  IS:  {spy_is.index[0].date()} → {spy_is.index[-1].date()}")
    print(f"  VAL: {spy_val.index[0].date()} → {spy_val.index[-1].date()}")

    # ==================================================================
    # PART B: REGIME SIGNAL VISUALIZATION
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Part B: Regime Signal Visualization")
    print("═" * 80)

    sig = MacroRegimeSignal(vix_is, tlt_is, uup_is,
                             vix_lookback=63, mom_lookback=60)
    regime_is = sig.generate(spy_is)
    print(f"\n  Signal distribution (IS):")
    print(f"    Mean:  {regime_is.mean():+.3f}")
    print(f"    Std:   {regime_is.std():.3f}")
    print(f"    Long (>0.1):    {(regime_is > 0.1).mean()*100:.1f}%")
    print(f"    Neutral:        {((regime_is >= -0.1) & (regime_is <= 0.1)).mean()*100:.1f}%")
    print(f"    Short (<-0.1):  {(regime_is < -0.1).mean()*100:.1f}%")

    # ==================================================================
    # PART C: APPLY TO SPY AND QQQ
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Part C: Macro Regime Applied to SPY & QQQ")
    print("═" * 80)

    targets = [
        ("SPY", spy_is, spy_val),
        ("QQQ", qqq_is, qqq_val),
    ]

    for target_name, target_is, target_val in targets:
        # IS
        sig_is = MacroRegimeSignal(vix_is, tlt_is, uup_is)
        strats_is = [SubStrategy(
            f"{target_name}_Macro", target_name, sig_is, EQUITY_COSTS, weight=1.0
        )]
        is_universe = {target_name: target_is}
        r_is = build_portfolio(is_universe, strats_is,
                                vol_target=0.10, vol_lookback=63,
                                max_gross_exposure=1.0)
        rep_is = full_report(r_is.net_returns)

        # VAL
        sig_val = MacroRegimeSignal(vix_val, tlt_val, uup_val)
        strats_val = [SubStrategy(
            f"{target_name}_Macro", target_name, sig_val, EQUITY_COSTS, weight=1.0
        )]
        val_universe = {target_name: target_val}
        r_val = build_portfolio(val_universe, strats_val,
                                 vol_target=0.10, vol_lookback=63,
                                 max_gross_exposure=1.0)
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

    vix_lookbacks = [21, 42, 63, 126]
    mom_lookbacks = [21, 42, 60, 126]
    weight_configs = [
        (0.50, 0.25, 0.25, "50/25/25"),
        (0.33, 0.33, 0.33, "equal"),
        (0.70, 0.15, 0.15, "VIX-heavy"),
        (1.00, 0.00, 0.00, "VIX-only"),
    ]

    print(f"\n  --- VIX Lookback Sensitivity (SPY, mom_lb=60, w=50/25/25) ---")
    print(f"  {'VIX LB':>7s} │ {'IS SR':>7s} {'VAL SR':>8s}")
    print("  " + "─" * 25)
    for vlb in vix_lookbacks:
        sig_is = MacroRegimeSignal(vix_is, tlt_is, uup_is, vix_lookback=vlb, mom_lookback=60)
        sig_val = MacroRegimeSignal(vix_val, tlt_val, uup_val, vix_lookback=vlb, mom_lookback=60)
        strats_is = [SubStrategy("SPY_M", "SPY", sig_is, EQUITY_COSTS, 1.0)]
        strats_val = [SubStrategy("SPY_M", "SPY", sig_val, EQUITY_COSTS, 1.0)]
        r_i = build_portfolio({"SPY": spy_is}, strats_is, vol_target=0.10, vol_lookback=63, max_gross_exposure=1.0)
        r_v = build_portfolio({"SPY": spy_val}, strats_val, vol_target=0.10, vol_lookback=63, max_gross_exposure=1.0)
        print(f"  {vlb:>7d} │ {sharpe_ratio(r_i.net_returns):>+6.3f} {sharpe_ratio(r_v.net_returns):>+7.3f}")

    print(f"\n  --- Momentum Lookback Sensitivity (SPY, vix_lb=63, w=50/25/25) ---")
    print(f"  {'Mom LB':>7s} │ {'IS SR':>7s} {'VAL SR':>8s}")
    print("  " + "─" * 25)
    for mlb in mom_lookbacks:
        sig_is = MacroRegimeSignal(vix_is, tlt_is, uup_is, vix_lookback=63, mom_lookback=mlb)
        sig_val = MacroRegimeSignal(vix_val, tlt_val, uup_val, vix_lookback=63, mom_lookback=mlb)
        strats_is = [SubStrategy("SPY_M", "SPY", sig_is, EQUITY_COSTS, 1.0)]
        strats_val = [SubStrategy("SPY_M", "SPY", sig_val, EQUITY_COSTS, 1.0)]
        r_i = build_portfolio({"SPY": spy_is}, strats_is, vol_target=0.10, vol_lookback=63, max_gross_exposure=1.0)
        r_v = build_portfolio({"SPY": spy_val}, strats_val, vol_target=0.10, vol_lookback=63, max_gross_exposure=1.0)
        print(f"  {mlb:>7d} │ {sharpe_ratio(r_i.net_returns):>+6.3f} {sharpe_ratio(r_v.net_returns):>+7.3f}")

    print(f"\n  --- Weight Config Sensitivity (SPY, vix_lb=63, mom_lb=60) ---")
    print(f"  {'Config':>12s} │ {'IS SR':>7s} {'VAL SR':>8s}")
    print("  " + "─" * 30)
    best_is_sr = -999
    best_params = None
    for wv, wb, wu, label in weight_configs:
        sig_is = MacroRegimeSignal(vix_is, tlt_is, uup_is, vix_lookback=63, mom_lookback=60,
                                    vix_weight=wv, bond_weight=wb, uup_weight=wu)
        sig_val = MacroRegimeSignal(vix_val, tlt_val, uup_val, vix_lookback=63, mom_lookback=60,
                                     vix_weight=wv, bond_weight=wb, uup_weight=wu)
        strats_is = [SubStrategy("SPY_M", "SPY", sig_is, EQUITY_COSTS, 1.0)]
        strats_val = [SubStrategy("SPY_M", "SPY", sig_val, EQUITY_COSTS, 1.0)]
        r_i = build_portfolio({"SPY": spy_is}, strats_is, vol_target=0.10, vol_lookback=63, max_gross_exposure=1.0)
        r_v = build_portfolio({"SPY": spy_val}, strats_val, vol_target=0.10, vol_lookback=63, max_gross_exposure=1.0)
        sr_i = sharpe_ratio(r_i.net_returns)
        sr_v = sharpe_ratio(r_v.net_returns)
        print(f"  {label:>12s} │ {sr_i:>+6.3f} {sr_v:>+7.3f}")
        if sr_i > best_is_sr:
            best_is_sr = sr_i
            best_params = (63, 60, wv, wb, wu, label)

    print(f"\n  Best config: vix_lb=63, mom_lb=60, weights={best_params[5]}")

    # Use best config for remaining analysis
    vlb, mlb, wv, wb, wu, _ = best_params

    # ==================================================================
    # PART E: PORTFOLIO (SPY + QQQ) WITH BEST CONFIG
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Part E: Multi-Asset Portfolio (SPY + QQQ)")
    print("═" * 80)

    # IS
    sig_spy_is = MacroRegimeSignal(vix_is, tlt_is, uup_is, vlb, mlb, wv, wb, wu)
    sig_qqq_is = MacroRegimeSignal(vix_is, tlt_is, uup_is, vlb, mlb, wv, wb, wu)
    strats_is = [
        SubStrategy("SPY_Macro", "SPY", sig_spy_is, EQUITY_COSTS, 1.0),
        SubStrategy("QQQ_Macro", "QQQ", sig_qqq_is, EQUITY_COSTS, 1.0),
    ]
    is_u = {"SPY": spy_is, "QQQ": qqq_is}
    r_is = build_portfolio(is_u, strats_is, vol_target=0.10, vol_lookback=63,
                            max_gross_exposure=1.0, concentration_limit=0.60)
    rep_is = full_report(r_is.net_returns)

    # VAL
    sig_spy_val = MacroRegimeSignal(vix_val, tlt_val, uup_val, vlb, mlb, wv, wb, wu)
    sig_qqq_val = MacroRegimeSignal(vix_val, tlt_val, uup_val, vlb, mlb, wv, wb, wu)
    strats_val = [
        SubStrategy("SPY_Macro", "SPY", sig_spy_val, EQUITY_COSTS, 1.0),
        SubStrategy("QQQ_Macro", "QQQ", sig_qqq_val, EQUITY_COSTS, 1.0),
    ]
    val_u = {"SPY": spy_val, "QQQ": qqq_val}
    r_val = build_portfolio(val_u, strats_val, vol_target=0.10, vol_lookback=63,
                             max_gross_exposure=1.0, concentration_limit=0.60)
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

        # Slice all data for test period
        t0, t1 = test_dates[0], test_dates[-1]
        vix_t = vix.loc[t0:t1]
        tlt_t = tlt.loc[t0:t1]
        uup_t = uup.loc[t0:t1]
        spy_t = spy.loc[t0:t1]
        qqq_t = qqq.loc[t0:t1]

        if len(vix_t) < 21 or len(spy_t) < 21:
            continue

        try:
            sig_spy = MacroRegimeSignal(vix_t, tlt_t, uup_t, vlb, mlb, wv, wb, wu)
            sig_qqq = MacroRegimeSignal(vix_t, tlt_t, uup_t, vlb, mlb, wv, wb, wu)
            strats = [
                SubStrategy("SPY_M", "SPY", sig_spy, EQUITY_COSTS, 1.0),
                SubStrategy("QQQ_M", "QQQ", sig_qqq, EQUITY_COSTS, 1.0),
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

    # 1.5× costs test
    sig_15 = MacroRegimeSignal(vix_is, tlt_is, uup_is, vlb, mlb, wv, wb, wu)
    strats_15 = [SubStrategy("SPY_M", "SPY", sig_15, scale_costs(EQUITY_COSTS, 1.5), 1.0)]
    r_15 = build_portfolio({"SPY": spy_is}, strats_15, vol_target=0.10, vol_lookback=63, max_gross_exposure=1.0)
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
        actual_str = f"{actual:+.3f}" if actual is not None else "N/A"
        target_str = f"{target:+.3f}" if target is not None else "N/A"
        print(f"  {desc:<40s} {actual_str:>8s} {target_str:>8s} {'✓ PASS' if passed else '✗ FAIL':>8s}")

    verdict = "✓ KEEP — Integrate into portfolio" if n_pass >= 3 else "✗ ELIMINATE — Do not integrate"
    print(f"\n  Result: {n_pass}/{len(criteria)} criteria passed")
    print(f"  Verdict: {verdict}")

    # ==================================================================
    # CHARTS
    # ==================================================================
    print("\nGenerating charts...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Regime signal over time with SPY
    ax1 = axes[0, 0]
    regime_is = sig_spy_is = MacroRegimeSignal(vix_is, tlt_is, uup_is, vlb, mlb, wv, wb, wu).generate(spy_is)
    ax1.fill_between(regime_is.index, regime_is.values, 0,
                     where=regime_is > 0, color="#2ecc71", alpha=0.3, label="Risk-On")
    ax1.fill_between(regime_is.index, regime_is.values, 0,
                     where=regime_is < 0, color="#e74c3c", alpha=0.3, label="Risk-Off")
    ax1.plot(regime_is.index, regime_is.values, color="#333", linewidth=0.5)
    ax2 = ax1.twinx()
    spy_norm = spy_is["close"] / spy_is["close"].iloc[0]
    ax2.plot(spy_norm.index, spy_norm.values, color="#3498db", linewidth=1.5, alpha=0.7, label="SPY")
    ax1.set_title("Regime Signal vs SPY (IS)", fontweight="bold")
    ax1.set_ylabel("Signal")
    ax2.set_ylabel("SPY (normalized)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # 2. Equity curves IS vs VAL
    ax = axes[0, 1]
    eq_is = (1 + r_is.net_returns).cumprod()
    eq_val = (1 + r_val.net_returns).cumprod()
    ax.plot(eq_is.index, eq_is, color="#2ecc71", linewidth=1.5, label="IS")
    ax.plot(eq_val.index, eq_val, color="#e74c3c", linewidth=1.5, label="VAL")
    ax.set_title("Equity Curve — Macro Regime Portfolio", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("Growth of $1")

    # 3. VIX vs Signal
    ax = axes[1, 0]
    ax.plot(vix_is["close"].index, vix_is["close"].values, color="#e74c3c", linewidth=1, label="VIX Level")
    ax.axhline(20, color="gray", linestyle="--", linewidth=0.8, label="VIX=20")
    ax.set_title("VIX Level (IS Period)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Walk-forward Sharpe per fold
    ax = axes[1, 1]
    if wf_sharpes:
        colors = ["#2ecc71" if s > 0 else "#e74c3c" for s in wf_sharpes]
        ax.bar(range(1, len(wf_sharpes)+1), wf_sharpes, color=colors)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Walk-Forward Sharpe per Fold", fontweight="bold")
    ax.grid(True, alpha=0.3)

    fig.suptitle("M21: Cross-Asset Macro Regime Signal", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "01_macro_regime.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 01_macro_regime.png")

    print(f"\n  Charts saved to: {CHART_DIR}/")
    print("=" * 80)
    print("  M21 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
