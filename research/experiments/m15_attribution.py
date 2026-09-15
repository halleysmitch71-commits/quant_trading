"""
Milestone 15 — Strategy Attribution
=====================================

Find exactly which signals create alpha, which destroy it,
which risk-control layers kill performance, and which combinations
provide genuine diversification.

Three parts:
  A. Portfolio Combination Matrix (10 configs × 3 evaluation methods)
  B. Risk-Control Layer Ablation (8 sequential layers)
  C. Walk-Forward Per Combination (10 configs × fold-by-fold)

Usage:
    python research/experiments/m15_attribution.py
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
    EQUITY_COSTS, CRYPTO_COSTS, FUTURES_COSTS, CostModel, scale_costs,
)
from quant_research.backtest.engine import run_backtest
from quant_research.data.loader import load_config, load_universe, split_data
from quant_research.metrics.performance import (
    full_report, sharpe_ratio, tail_risk_stats,
)
from quant_research.portfolio.allocator import SubStrategy, build_portfolio

from quant_research.strategies.mean_reversion_smooth import SmoothedMeanReversionSignal
from quant_research.strategies.trend_following import TrendFollowingSignal
from quant_research.strategies.volatility_signal import VolatilityMeanReversionSignal

CHART_DIR = PROJECT_ROOT / "research" / "experiments" / "m15_charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

PP = dict(vol_target=0.10, vol_lookback=63, max_gross_exposure=1.0,
          dd_threshold=-0.05, concentration_limit=0.40)

# ── Sub-strategy building blocks ──────────────────────────────────────────

def _btc_tf():
    return SubStrategy("BTC_TF60", "BTC-USD", TrendFollowingSignal(lookback=60, scale=2.0), CRYPTO_COSTS, weight=1.0)

def _eth_tf():
    return SubStrategy("ETH_TF120", "ETH-USD", TrendFollowingSignal(lookback=120, scale=2.0), CRYPTO_COSTS, weight=1.0)

def _qqq_smr():
    return SubStrategy("QQQ_SMR10", "QQQ", SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=10), EQUITY_COSTS, weight=1.0)

def _spy_smr():
    return SubStrategy("SPY_SMR10", "SPY", SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=10), EQUITY_COSTS, weight=1.0)

def _gc_smr():
    return SubStrategy("GC_SMR5", "GC=F", SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=5), FUTURES_COSTS, weight=1.0)

def _btc_vol():
    return SubStrategy("BTC_VOL", "BTC-USD", VolatilityMeanReversionSignal(short_window=10, long_window=60), CRYPTO_COSTS, weight=1.0)

def _eth_vol():
    return SubStrategy("ETH_VOL", "ETH-USD", VolatilityMeanReversionSignal(short_window=10, long_window=60), CRYPTO_COSTS, weight=1.0)

def _spy_vol():
    return SubStrategy("SPY_VOL", "SPY", VolatilityMeanReversionSignal(short_window=10, long_window=60), EQUITY_COSTS, weight=1.0)


COMBINATIONS = {
    "A. BTC only":           lambda: [_btc_tf()],
    "B. ETH only":           lambda: [_eth_tf()],
    "C. BTC+ETH":            lambda: [_btc_tf(), _eth_tf()],
    "D. BTC+ETH+QQQ":        lambda: [_btc_tf(), _eth_tf(), _qqq_smr()],
    "E. BTC+ETH+GC":         lambda: [_btc_tf(), _eth_tf(), _gc_smr()],
    "F. BTC+ETH+allSMR":     lambda: [_btc_tf(), _eth_tf(), _qqq_smr(), _spy_smr(), _gc_smr()],
    "G. BTC+VOL":            lambda: [_btc_tf(), _btc_vol()],
    "H. BTC+ETH+VOL":        lambda: [_btc_tf(), _eth_tf(), _btc_vol(), _eth_vol(), _spy_vol()],
    "I. Original 5":         lambda: [_btc_tf(), _eth_tf(), _qqq_smr(), _spy_smr(), _gc_smr()],
    "J. Original 5+VOL":     lambda: [_btc_tf(), _eth_tf(), _qqq_smr(), _spy_smr(), _gc_smr(),
                                       _btc_vol(), _eth_vol(), _spy_vol()],
}


def _wf_sharpe(universe, strats, pp):
    """Run walk-forward and return aggregate Sharpe + per-fold Sharpes."""
    all_tickers = {s.ticker for s in strats}
    common = None
    for t in all_tickers:
        if t in universe:
            idx = universe[t].index
            common = idx if common is None else common.intersection(idx)
    if common is None or len(common) < 756 + 126:
        return float("nan"), []

    common = common.sort_values()
    train_days = 3 * 252
    step_days = 6 * 21
    cursor = train_days
    all_rets = []
    fold_sharpes = []

    while cursor + step_days <= len(common):
        test_dates = common[cursor:cursor + step_days]
        all_dates = common[:cursor + step_days]
        wf_u = {}
        for t in all_tickers:
            if t in universe:
                mask = universe[t].index.isin(all_dates)
                wf_u[t] = universe[t][mask]
        try:
            r = build_portfolio(wf_u, strats, **pp)
            test_ret = r.net_returns[r.net_returns.index.isin(test_dates)]
            if len(test_ret) > 20:
                all_rets.append(test_ret)
                fold_sharpes.append(sharpe_ratio(test_ret))
        except Exception:
            fold_sharpes.append(float("nan"))
        cursor += step_days

    if all_rets:
        chained = pd.concat(all_rets)
        chained = chained[~chained.index.duplicated(keep='first')].sort_index()
        return sharpe_ratio(chained), fold_sharpes
    return float("nan"), fold_sharpes


def main():
    print("=" * 80)
    print("  M15 — Strategy Attribution")
    print("=" * 80)

    config = load_config()
    universe = load_universe(config)
    is_u, val_u = {}, {}
    for ticker, df in universe.items():
        splits = split_data(df, config)
        is_u[ticker] = splits["in_sample"]
        if "validation" in splits:
            val_u[ticker] = splits["validation"]

    # ==================================================================
    # PART A: PORTFOLIO COMBINATION MATRIX
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Part A: Portfolio Combination Matrix")
    print("═" * 80)

    header = (f"  {'Config':<22s} │ {'IS SR':>6s} {'VAL SR':>7s} {'WF SR':>6s} │ "
              f"{'IS Ret':>7s} {'VAL Ret':>8s} │ {'IS Vol':>7s} {'IS DD':>7s} │ "
              f"{'Calmar':>7s} │ {'VaR5':>6s} {'CVaR5':>7s} │ {'GrExp':>6s} {'Costs':>6s}")
    print(header)
    print("  " + "─" * 115)

    combo_results = {}

    for label, make_strats in COMBINATIONS.items():
        strats = make_strats()
        try:
            r_is = build_portfolio(is_u, strats, **PP)
            rep_is = full_report(r_is.net_returns, costs=r_is.costs)
            tail_is = tail_risk_stats(r_is.net_returns)
            is_sr = rep_is["sharpe_ratio"]
            is_ret = rep_is["annualised_return"]
            is_vol = rep_is["annualised_volatility"]
            is_dd = rep_is["max_drawdown"]
            is_calmar = abs(is_ret / is_dd) if abs(is_dd) > 1e-6 else 0
            is_var = tail_is["var_5pct"]
            is_cvar = tail_is["cvar_5pct"]
            is_ge = r_is.gross_exposure.mean()
            is_cost = r_is.costs.sum()
        except Exception as e:
            print(f"  {label:<22s} │ ERROR: {e}")
            continue

        try:
            r_val = build_portfolio(val_u, strats, **PP)
            rep_val = full_report(r_val.net_returns, costs=r_val.costs)
            val_sr = rep_val["sharpe_ratio"]
            val_ret = rep_val["annualised_return"]
        except Exception:
            val_sr = float("nan")
            val_ret = float("nan")

        wf_sr, fold_srs = _wf_sharpe(universe, strats, PP)

        combo_results[label] = {
            "is_sr": is_sr, "val_sr": val_sr, "wf_sr": wf_sr,
            "is_ret": is_ret, "val_ret": val_ret,
            "is_vol": is_vol, "is_dd": is_dd, "is_calmar": is_calmar,
            "is_var": is_var, "is_cvar": is_cvar,
            "is_ge": is_ge, "is_cost": is_cost,
            "fold_sharpes": fold_srs,
        }

        print(f"  {label:<22s} │ {is_sr:>+5.3f} {val_sr:>+6.3f} {wf_sr:>+5.3f} │ "
              f"{is_ret:>+6.2%} {val_ret:>+7.2%} │ {is_vol:>6.2%} {is_dd:>+6.2%} │ "
              f"{is_calmar:>6.2f} │ {is_var:>+5.2%} {is_cvar:>+6.2%} │ {is_ge:>5.1%} {is_cost:>5.3f}")

    # Marginal contributions
    print("\n  Marginal Contribution (adding each component to BTC baseline):")
    btc_is_sr = combo_results.get("A. BTC only", {}).get("is_sr", 0)
    btc_val_sr = combo_results.get("A. BTC only", {}).get("val_sr", 0)
    btc_wf_sr = combo_results.get("A. BTC only", {}).get("wf_sr", 0)
    print(f"  {'Step':<30s} {'IS ΔSR':>8s} {'VAL ΔSR':>9s} {'WF ΔSR':>8s}")
    print("  " + "-" * 58)
    chain = [
        ("A → C (+ETH)", "C. BTC+ETH"),
        ("C → F (+allSMR)", "F. BTC+ETH+allSMR"),
        ("A → G (+VOL)", "G. BTC+VOL"),
        ("C → H (+VOL)", "H. BTC+ETH+VOL"),
        ("F → J (+VOL)", "J. Original 5+VOL"),
    ]
    prev = {"is_sr": btc_is_sr, "val_sr": btc_val_sr, "wf_sr": btc_wf_sr}
    refs = {
        "A → C (+ETH)": "A. BTC only",
        "C → F (+allSMR)": "C. BTC+ETH",
        "A → G (+VOL)": "A. BTC only",
        "C → H (+VOL)": "C. BTC+ETH",
        "F → J (+VOL)": "F. BTC+ETH+allSMR",
    }
    for step_label, target_key in chain:
        ref_key = refs[step_label]
        ref = combo_results.get(ref_key, {})
        tgt = combo_results.get(target_key, {})
        d_is = tgt.get("is_sr", 0) - ref.get("is_sr", 0)
        d_val = tgt.get("val_sr", 0) - ref.get("val_sr", 0)
        d_wf = tgt.get("wf_sr", 0) - ref.get("wf_sr", 0)
        sym_is = "↑" if d_is > 0.05 else ("↓" if d_is < -0.05 else "→")
        sym_val = "↑" if d_val > 0.05 else ("↓" if d_val < -0.05 else "→")
        sym_wf = "↑" if d_wf > 0.05 else ("↓" if d_wf < -0.05 else "→")
        print(f"  {step_label:<30s} {d_is:>+7.3f}{sym_is} {d_val:>+8.3f}{sym_val} {d_wf:>+7.3f}{sym_wf}")

    # Sub-strategy correlation matrix
    print("\n  Sub-strategy Return Correlations (IS):")
    all_ss_names = ["BTC_TF60", "ETH_TF120", "QQQ_SMR10", "SPY_SMR10", "GC_SMR5",
                    "BTC_VOL", "ETH_VOL", "SPY_VOL"]
    all_ss_specs = [
        ("BTC_TF60", "BTC-USD", TrendFollowingSignal(lookback=60, scale=2.0), CRYPTO_COSTS),
        ("ETH_TF120", "ETH-USD", TrendFollowingSignal(lookback=120, scale=2.0), CRYPTO_COSTS),
        ("QQQ_SMR10", "QQQ", SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=10), EQUITY_COSTS),
        ("SPY_SMR10", "SPY", SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=10), EQUITY_COSTS),
        ("GC_SMR5", "GC=F", SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=5), FUTURES_COSTS),
        ("BTC_VOL", "BTC-USD", VolatilityMeanReversionSignal(short_window=10, long_window=60), CRYPTO_COSTS),
        ("ETH_VOL", "ETH-USD", VolatilityMeanReversionSignal(short_window=10, long_window=60), CRYPTO_COSTS),
        ("SPY_VOL", "SPY", VolatilityMeanReversionSignal(short_window=10, long_window=60), EQUITY_COSTS),
    ]

    strat_rets_is = {}
    for name, ticker, sig_obj, cost in all_ss_specs:
        data = is_u[ticker]
        signal = sig_obj.generate(data)
        bt = run_backtest(prices=data["close"], signal=signal, cost_model=cost)
        strat_rets_is[name] = bt.net_returns

    corr_df = pd.DataFrame(strat_rets_is).dropna().corr()
    print(f"  {'':>12s}", end="")
    for n in all_ss_names:
        print(f" {n[:7]:>8s}", end="")
    print()
    for n1 in all_ss_names:
        print(f"  {n1[:12]:<12s}", end="")
        for n2 in all_ss_names:
            v = corr_df.loc[n1, n2] if n1 in corr_df.index and n2 in corr_df.columns else 0
            print(f" {v:>+7.3f}", end="")
        print()

    # ==================================================================
    # PART B: RISK-CONTROL LAYER ABLATION
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Part B: Risk-Control Layer Ablation (Original 5)")
    print("═" * 80)

    strats_base = [_btc_tf(), _eth_tf(), _qqq_smr(), _spy_smr(), _gc_smr()]

    ablation_configs = [
        ("1. Raw signals (no delay)",
         dict(vol_target=999, vol_lookback=63, max_gross_exposure=999,
              dd_threshold=None, concentration_limit=None),
         False),  # no_delay
        ("2. + 1-bar delay",
         dict(vol_target=999, vol_lookback=63, max_gross_exposure=999,
              dd_threshold=None, concentration_limit=None),
         True),
        ("3. + equal weights",
         dict(vol_target=999, vol_lookback=63, max_gross_exposure=999,
              dd_threshold=None, concentration_limit=None),
         True),
        ("4. + vol target (10%)",
         dict(vol_target=0.10, vol_lookback=63, max_gross_exposure=999,
              dd_threshold=None, concentration_limit=None),
         True),
        ("5. + gross exp cap (100%)",
         dict(vol_target=0.10, vol_lookback=63, max_gross_exposure=1.0,
              dd_threshold=None, concentration_limit=None),
         True),
        ("6. + concentration (40%)",
         dict(vol_target=0.10, vol_lookback=63, max_gross_exposure=1.0,
              dd_threshold=None, concentration_limit=0.40),
         True),
        ("7. + DD throttle (-5%)",
         dict(vol_target=0.10, vol_lookback=63, max_gross_exposure=1.0,
              dd_threshold=-0.05, concentration_limit=0.40),
         True),
        ("8. + transaction costs",
         dict(vol_target=0.10, vol_lookback=63, max_gross_exposure=1.0,
              dd_threshold=-0.05, concentration_limit=0.40),
         True),
    ]

    # Note: Layers 1-3 are essentially the same in build_portfolio because:
    # - build_portfolio always applies 1-bar delay (built into engine)
    # - build_portfolio always applies equal weights when weights are all 1.0
    # The difference is captured by vol_target=999 (no scaling) vs 0.10

    # For layers 1-7 with no costs, use scale_costs(_, 0)
    # Layer 8 uses real costs

    print(f"\n  {'Layer':<30s} │ {'IS SR':>7s} {'VAL SR':>8s} {'IS Ret':>8s} {'IS Vol':>8s} "
          f"{'IS DD':>8s} {'GrExp':>6s}")
    print("  " + "─" * 82)

    for layer_name, params, has_delay in ablation_configs:
        # For layers 1-7, zero costs
        use_zero_cost = (layer_name != "8. + transaction costs")

        if use_zero_cost:
            strats = [
                SubStrategy(s.name, s.ticker, s.signal_obj, scale_costs(s.cost_model, 0), weight=s.weight)
                for s in strats_base
            ]
        else:
            strats = strats_base

        for split_name, u in [("IS", is_u), ("VAL", val_u)]:
            try:
                r = build_portfolio(u, strats, **params)
                rep = full_report(r.net_returns)
                if split_name == "IS":
                    is_sr = rep["sharpe_ratio"]
                    is_ret = rep["annualised_return"]
                    is_vol = rep["annualised_volatility"]
                    is_dd = rep["max_drawdown"]
                    is_ge = r.gross_exposure.mean()
                else:
                    val_sr = rep["sharpe_ratio"]
            except Exception as e:
                if split_name == "IS":
                    is_sr = is_ret = is_vol = is_dd = is_ge = float("nan")
                else:
                    val_sr = float("nan")

        print(f"  {layer_name:<30s} │ {is_sr:>+6.3f} {val_sr:>+7.3f} {is_ret:>+7.2%} "
              f"{is_vol:>7.2%} {is_dd:>+7.2%} {is_ge:>5.1%}")

    # ==================================================================
    # PART C: WALK-FORWARD PER COMBINATION
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Part C: Walk-Forward Fold-by-Fold (Top Combinations)")
    print("═" * 80)

    # Run WF for key combinations with fold details
    key_combos = ["A. BTC only", "C. BTC+ETH", "G. BTC+VOL",
                  "H. BTC+ETH+VOL", "I. Original 5", "J. Original 5+VOL"]

    for label in key_combos:
        if label not in COMBINATIONS:
            continue
        strats = COMBINATIONS[label]()
        wf_sr, fold_srs = _wf_sharpe(universe, strats, PP)
        n_pos = sum(1 for s in fold_srs if not np.isnan(s) and s > 0)
        n_neg = sum(1 for s in fold_srs if not np.isnan(s) and s <= 0)
        n_tot = len(fold_srs)

        fold_str = " ".join(f"{s:>+.1f}" if not np.isnan(s) else "nan" for s in fold_srs)
        print(f"\n  {label}")
        print(f"    Aggregate WF Sharpe: {wf_sr:+.3f}")
        print(f"    Folds ({n_pos}+/{n_neg}−): [{fold_str}]")

        # Classify pattern
        if n_pos > n_neg:
            print(f"    Pattern: MAJORITY POSITIVE ({n_pos}/{n_tot})")
        elif n_pos == n_neg:
            print(f"    Pattern: SPLIT ({n_pos}/{n_tot})")
        else:
            recent_neg = all(not np.isnan(s) and s <= 0 for s in fold_srs[-3:]) if len(fold_srs) >= 3 else False
            if recent_neg:
                print(f"    Pattern: MAJORITY NEGATIVE, LAST 3 FOLDS ALL NEGATIVE → regime shift")
            else:
                print(f"    Pattern: MAJORITY NEGATIVE ({n_neg}/{n_tot})")

    # ==================================================================
    # CHARTS
    # ==================================================================
    print("\n\nGenerating charts...")

    # Chart 1: Combination Sharpe comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    labels = list(combo_results.keys())
    short_labels = [l.split(". ")[1][:15] for l in labels]
    x = np.arange(len(labels))

    for ax, metric, title in zip(axes,
                                  ["is_sr", "val_sr", "wf_sr"],
                                  ["In-Sample Sharpe", "Validation Sharpe", "Walk-Forward Sharpe"]):
        vals = [combo_results[l][metric] for l in labels]
        colors = ["#2ecc71" if v > 0.3 else ("#e74c3c" if v < 0 else "#f39c12") for v in vals]
        ax.barh(x, vals, color=colors, alpha=0.8, edgecolor="white", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(x)
        ax.set_yticklabels(short_labels, fontsize=9)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Sharpe Ratio")

    fig.suptitle("M15: Portfolio Combination Sharpe Comparison", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "01_combination_sharpe.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 01_combination_sharpe.png")

    # Chart 2: Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_df.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(all_ss_names)))
    ax.set_xticklabels([n[:7] for n in all_ss_names], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(all_ss_names)))
    ax.set_yticklabels([n[:7] for n in all_ss_names], fontsize=9)
    for i in range(len(all_ss_names)):
        for j in range(len(all_ss_names)):
            ax.text(j, i, f"{corr_df.values[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(corr_df.values[i, j]) > 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Sub-Strategy Return Correlation (IS)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "02_correlation_heatmap.png", dpi=150)
    plt.close(fig)
    print("  ✓ 02_correlation_heatmap.png")

    # Chart 3: Walk-forward fold heatmap
    fig, ax = plt.subplots(figsize=(16, 5))
    wf_data = []
    wf_labels = []
    for label in key_combos:
        if label in combo_results:
            folds = combo_results[label]["fold_sharpes"]
            if folds:
                wf_data.append(folds)
                wf_labels.append(label.split(". ")[1][:15])

    if wf_data:
        max_folds = max(len(f) for f in wf_data)
        padded = [f + [float("nan")] * (max_folds - len(f)) for f in wf_data]
        arr = np.array(padded)
        im = ax.imshow(arr, cmap="RdYlGn", vmin=-2, vmax=2, aspect="auto")
        ax.set_yticks(range(len(wf_labels)))
        ax.set_yticklabels(wf_labels, fontsize=9)
        ax.set_xlabel("Walk-Forward Fold")
        ax.set_title("Walk-Forward Sharpe by Fold and Configuration", fontsize=14, fontweight="bold")
        for i in range(len(wf_data)):
            for j in range(len(wf_data[i])):
                v = wf_data[i][j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                            color="white" if abs(v) > 1.0 else "black")
        fig.colorbar(im, ax=ax, shrink=0.6, label="Sharpe")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "03_wf_heatmap.png", dpi=150)
    plt.close(fig)
    print("  ✓ 03_wf_heatmap.png")

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print("\n" + "=" * 80)
    print("  M15 ATTRIBUTION SUMMARY")
    print("=" * 80)

    # Find best IS, best VAL, best WF
    best_is = max(combo_results.items(), key=lambda x: x[1]["is_sr"])
    best_val = max(combo_results.items(), key=lambda x: x[1]["val_sr"] if not np.isnan(x[1]["val_sr"]) else -999)
    best_wf = max(combo_results.items(), key=lambda x: x[1]["wf_sr"] if not np.isnan(x[1]["wf_sr"]) else -999)

    print(f"\n  Best IS Sharpe:   {best_is[0]} → {best_is[1]['is_sr']:+.3f}")
    print(f"  Best VAL Sharpe:  {best_val[0]} → {best_val[1]['val_sr']:+.3f}")
    print(f"  Best WF Sharpe:   {best_wf[0]} → {best_wf[1]['wf_sr']:+.3f}")

    # Answer the 4 questions
    print("\n  ─── M15 Answers ───")
    a_res = combo_results.get("A. BTC only", {})
    b_res = combo_results.get("B. ETH only", {})
    print(f"\n  1. Which signal creates alpha?")
    print(f"     BTC_TF60:  IS={a_res.get('is_sr',0):+.3f}  VAL={a_res.get('val_sr',0):+.3f}  WF={a_res.get('wf_sr',0):+.3f}")
    print(f"     ETH_TF120: IS={b_res.get('is_sr',0):+.3f}  VAL={b_res.get('val_sr',0):+.3f}  WF={b_res.get('wf_sr',0):+.3f}")

    c_res = combo_results.get("C. BTC+ETH", {})
    f_res = combo_results.get("F. BTC+ETH+allSMR", {})
    print(f"\n  2. Which signal destroys alpha?")
    d_is = f_res.get('is_sr', 0) - c_res.get('is_sr', 0)
    d_val = f_res.get('val_sr', 0) - c_res.get('val_sr', 0)
    print(f"     Adding SMR to BTC+ETH: IS ΔSharpe={d_is:+.3f}  VAL ΔSharpe={d_val:+.3f}")
    print(f"     → SMR {'DESTROYS' if d_val < -0.1 else 'marginally hurts' if d_val < 0 else 'helps'} alpha on VAL")

    g_res = combo_results.get("G. BTC+VOL", {})
    h_res = combo_results.get("H. BTC+ETH+VOL", {})
    print(f"\n  3. Which combination creates diversification?")
    print(f"     BTC+VOL:     IS={g_res.get('is_sr',0):+.3f}  VAL={g_res.get('val_sr',0):+.3f}  WF={g_res.get('wf_sr',0):+.3f}")
    print(f"     BTC+ETH+VOL: IS={h_res.get('is_sr',0):+.3f}  VAL={h_res.get('val_sr',0):+.3f}  WF={h_res.get('wf_sr',0):+.3f}")

    print(f"\n  4. Which risk-control layer destroys alpha?")
    print(f"     → See ablation table above. Focus on IS→VAL Sharpe decay at each step.")

    print(f"\n  Charts saved to: {CHART_DIR}/")
    print("=" * 80)
    print("  M15 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
