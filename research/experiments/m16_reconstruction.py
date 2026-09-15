"""
Milestone 16 — Portfolio Reconstruction
=========================================

Rebuild the portfolio using sleeve-based risk-budget allocation.

Parts:
  A. Sleeve allocation vs equal weight (IS + nested WF)
  B. Nested walk-forward for risk budget selection
  C. Risk-layer ablation on sleeve portfolio
  D. Final VAL validation (one-shot, consumes holdout)

Usage:
    python research/experiments/m16_reconstruction.py
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
    EQUITY_COSTS, CRYPTO_COSTS, FUTURES_COSTS, scale_costs,
)
from quant_research.backtest.engine import run_backtest
from quant_research.data.loader import load_config, load_universe, split_data
from quant_research.metrics.performance import (
    full_report, sharpe_ratio, tail_risk_stats,
)
from quant_research.portfolio.allocator import SubStrategy, build_portfolio
from quant_research.portfolio.sleeve_allocator import Sleeve, build_sleeve_portfolio
from quant_research.strategies.mean_reversion_smooth import SmoothedMeanReversionSignal
from quant_research.strategies.trend_following import TrendFollowingSignal
from quant_research.strategies.volatility_signal import VolatilityMeanReversionSignal

CHART_DIR = PROJECT_ROOT / "research" / "experiments" / "m16_charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

PP_RISK = dict(
    vol_lookback=63, max_gross_exposure=1.0,
    dd_threshold=-0.05, concentration_limit=0.40,
)


def _make_sleeves(trend_budget, vol_budget, smr_budget):
    """Create sleeves with given risk budgets."""
    return [
        Sleeve(
            name="crypto_trend",
            sub_strategies=[
                SubStrategy("BTC_TF60", "BTC-USD", TrendFollowingSignal(lookback=60, scale=2.0), CRYPTO_COSTS, weight=1.0),
                SubStrategy("ETH_TF120", "ETH-USD", TrendFollowingSignal(lookback=120, scale=2.0), CRYPTO_COSTS, weight=1.0),
            ],
            risk_budget=trend_budget,
            internal_weighting="equal",
            vol_target=0.10,
        ),
        Sleeve(
            name="volatility",
            sub_strategies=[
                SubStrategy("BTC_VOL", "BTC-USD", VolatilityMeanReversionSignal(short_window=10, long_window=60), CRYPTO_COSTS, weight=1.0),
                SubStrategy("ETH_VOL", "ETH-USD", VolatilityMeanReversionSignal(short_window=10, long_window=60), CRYPTO_COSTS, weight=1.0),
                SubStrategy("SPY_VOL", "SPY", VolatilityMeanReversionSignal(short_window=10, long_window=60), EQUITY_COSTS, weight=1.0),
            ],
            risk_budget=vol_budget,
            internal_weighting="equal",
            vol_target=0.10,
        ),
        Sleeve(
            name="mean_reversion",
            sub_strategies=[
                SubStrategy("QQQ_SMR10", "QQQ", SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=10), EQUITY_COSTS, weight=1.0),
                SubStrategy("GC_SMR5", "GC=F", SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=5), FUTURES_COSTS, weight=1.0),
            ],
            risk_budget=smr_budget,
            internal_weighting="equal",
            vol_target=0.10,
        ),
    ]


def _wf_sharpe_sleeve(universe, sleeves, pp_risk):
    """Walk-forward for sleeve portfolio."""
    all_tickers = set()
    for sl in sleeves:
        for ss in sl.sub_strategies:
            all_tickers.add(ss.ticker)

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
            r = build_sleeve_portfolio(wf_u, sleeves, **pp_risk)
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
    print("  M16 — Portfolio Reconstruction")
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
    # PART A: SLEEVE ALLOCATION VS EQUAL WEIGHT
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Part A: Sleeve Allocation vs Equal Weight")
    print("═" * 80)

    budget_configs = [
        ("Equal (old)", None),  # Use old build_portfolio
        ("50/30/20", (0.50, 0.30, 0.20)),
        ("60/25/15", (0.60, 0.25, 0.15)),
        ("70/20/10", (0.70, 0.20, 0.10)),
        ("80/15/05", (0.80, 0.15, 0.05)),
        ("70/30/00", (0.70, 0.30, 0.00)),  # Drop SMR entirely
    ]

    print(f"\n  {'Config':<16s} │ {'IS SR':>6s} {'VAL SR':>7s} {'WF SR':>6s} │ "
          f"{'IS Ret':>7s} {'IS Vol':>7s} {'IS DD':>7s} │ {'GrExp':>6s} {'Costs':>6s}")
    print("  " + "─" * 80)

    results = {}

    for label, budgets in budget_configs:
        if budgets is None:
            # Old-style equal weight
            old_strats = [
                SubStrategy("BTC_TF60", "BTC-USD", TrendFollowingSignal(lookback=60, scale=2.0), CRYPTO_COSTS, weight=1.0),
                SubStrategy("ETH_TF120", "ETH-USD", TrendFollowingSignal(lookback=120, scale=2.0), CRYPTO_COSTS, weight=1.0),
                SubStrategy("QQQ_SMR10", "QQQ", SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=10), EQUITY_COSTS, weight=1.0),
                SubStrategy("SPY_SMR10", "SPY", SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=10), EQUITY_COSTS, weight=1.0),
                SubStrategy("GC_SMR5", "GC=F", SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=5), FUTURES_COSTS, weight=1.0),
                SubStrategy("BTC_VOL", "BTC-USD", VolatilityMeanReversionSignal(short_window=10, long_window=60), CRYPTO_COSTS, weight=1.0),
                SubStrategy("ETH_VOL", "ETH-USD", VolatilityMeanReversionSignal(short_window=10, long_window=60), CRYPTO_COSTS, weight=1.0),
                SubStrategy("SPY_VOL", "SPY", VolatilityMeanReversionSignal(short_window=10, long_window=60), EQUITY_COSTS, weight=1.0),
            ]
            pp_old = dict(vol_target=0.10, vol_lookback=63, max_gross_exposure=1.0,
                          dd_threshold=-0.05, concentration_limit=0.40)
            r_is = build_portfolio(is_u, old_strats, **pp_old)
            r_val = build_portfolio(val_u, old_strats, **pp_old)
            # WF for old-style
            from quant_research.portfolio.allocator import build_portfolio as bp
            all_tickers_old = {s.ticker for s in old_strats}
            common = None
            for t in all_tickers_old:
                if t in universe:
                    idx = universe[t].index
                    common = idx if common is None else common.intersection(idx)
            common = common.sort_values()
            train_d, step_d = 3*252, 6*21
            curs = train_d
            wf_rets = []
            while curs + step_d <= len(common):
                td = common[curs:curs+step_d]
                ad = common[:curs+step_d]
                wu = {t: universe[t][universe[t].index.isin(ad)] for t in all_tickers_old if t in universe}
                try:
                    rr = bp(wu, old_strats, **pp_old)
                    tr = rr.net_returns[rr.net_returns.index.isin(td)]
                    if len(tr) > 20:
                        wf_rets.append(tr)
                except Exception:
                    pass
                curs += step_d
            if wf_rets:
                ch = pd.concat(wf_rets)
                ch = ch[~ch.index.duplicated(keep='first')].sort_index()
                wf_sr = sharpe_ratio(ch)
            else:
                wf_sr = float("nan")
        else:
            trend_b, vol_b, smr_b = budgets
            sleeves = _make_sleeves(trend_b, vol_b, smr_b)
            # Filter out sleeves with zero budget
            sleeves = [s for s in sleeves if s.risk_budget > 0.001]
            r_is = build_sleeve_portfolio(is_u, sleeves, **PP_RISK)
            r_val = build_sleeve_portfolio(val_u, sleeves, **PP_RISK)
            wf_sr, _ = _wf_sharpe_sleeve(universe, sleeves, PP_RISK)

        rep_is = full_report(r_is.net_returns, costs=r_is.costs)
        rep_val = full_report(r_val.net_returns, costs=r_val.costs)

        results[label] = {
            "is_sr": rep_is["sharpe_ratio"],
            "val_sr": rep_val["sharpe_ratio"],
            "wf_sr": wf_sr,
            "is_ret": rep_is["annualised_return"],
            "is_vol": rep_is["annualised_volatility"],
            "is_dd": rep_is["max_drawdown"],
            "is_ge": r_is.gross_exposure.mean(),
            "is_cost": r_is.costs.sum(),
        }

        print(f"  {label:<16s} │ {rep_is['sharpe_ratio']:>+5.3f} {rep_val['sharpe_ratio']:>+6.3f} {wf_sr:>+5.3f} │ "
              f"{rep_is['annualised_return']:>+6.2%} {rep_is['annualised_volatility']:>6.2%} {rep_is['max_drawdown']:>+6.2%} │ "
              f"{r_is.gross_exposure.mean():>5.1%} {r_is.costs.sum():>5.3f}")

    # ==================================================================
    # PART B: NESTED WALK-FORWARD FOR BUDGET SELECTION
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Part B: Nested Walk-Forward Budget Selection (IS only)")
    print("═" * 80)

    # Inner WF on IS data: 2yr train, 6mo validate
    print("  Using inner walk-forward on IS period to select risk budget.")
    print("  Inner: 2yr train, 6mo test, expanding window.")

    budget_candidates = [
        (0.50, 0.30, 0.20),
        (0.60, 0.25, 0.15),
        (0.70, 0.20, 0.10),
        (0.80, 0.15, 0.05),
        (0.70, 0.30, 0.00),
    ]

    is_tickers = set()
    for b in budget_candidates:
        for sl in _make_sleeves(*b):
            for ss in sl.sub_strategies:
                is_tickers.add(ss.ticker)

    is_common = None
    for t in is_tickers:
        if t in is_u:
            idx = is_u[t].index
            is_common = idx if is_common is None else is_common.intersection(idx)
    is_common = is_common.sort_values()

    inner_train = 2 * 252
    inner_step = 6 * 21

    print(f"\n  IS common dates: {len(is_common)}  ({is_common[0].date()} → {is_common[-1].date()})")
    print(f"  Inner train: {inner_train}d, Inner step: {inner_step}d\n")

    budget_inner_sharpes = {}

    for trend_b, vol_b, smr_b in budget_candidates:
        label = f"{int(trend_b*100)}/{int(vol_b*100)}/{int(smr_b*100)}"
        sleeves = _make_sleeves(trend_b, vol_b, smr_b)
        sleeves = [s for s in sleeves if s.risk_budget > 0.001]

        cursor = inner_train
        inner_rets = []
        inner_fold_srs = []

        while cursor + inner_step <= len(is_common):
            test_dates = is_common[cursor:cursor + inner_step]
            all_dates = is_common[:cursor + inner_step]
            wf_u = {t: is_u[t][is_u[t].index.isin(all_dates)] for t in is_tickers if t in is_u}
            try:
                r = build_sleeve_portfolio(wf_u, sleeves, **PP_RISK)
                test_ret = r.net_returns[r.net_returns.index.isin(test_dates)]
                if len(test_ret) > 20:
                    inner_rets.append(test_ret)
                    inner_fold_srs.append(sharpe_ratio(test_ret))
            except Exception:
                inner_fold_srs.append(float("nan"))
            cursor += inner_step

        if inner_rets:
            ch = pd.concat(inner_rets)
            ch = ch[~ch.index.duplicated(keep='first')].sort_index()
            agg_sr = sharpe_ratio(ch)
        else:
            agg_sr = float("nan")

        budget_inner_sharpes[label] = agg_sr
        fold_str = " ".join(f"{s:>+.2f}" for s in inner_fold_srs if not np.isnan(s))
        print(f"    {label}: Inner WF Sharpe={agg_sr:+.3f}  Folds=[{fold_str}]")

    # Select best budget (based on inner WF, NOT VAL)
    best_budget_label = max(budget_inner_sharpes, key=budget_inner_sharpes.get)
    print(f"\n  ★ Selected budget: {best_budget_label} (Inner WF Sharpe={budget_inner_sharpes[best_budget_label]:+.3f})")
    print(f"    Selection is based purely on IS inner walk-forward. VAL is NOT consumed.")

    # ==================================================================
    # PART C: ABLATION ON SLEEVE PORTFOLIO
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Part C: Risk-Layer Ablation (Sleeve Portfolio)")
    print("═" * 80)

    # Parse best budget
    parts = best_budget_label.split("/")
    best_budgets = (int(parts[0])/100, int(parts[1])/100, int(parts[2])/100)
    best_sleeves = _make_sleeves(*best_budgets)
    best_sleeves = [s for s in best_sleeves if s.risk_budget > 0.001]

    ablation = [
        ("Raw (no risk ctl)",   dict(vol_lookback=63, max_gross_exposure=999, dd_threshold=None, concentration_limit=None), True),
        ("+ gross exp (100%)",  dict(vol_lookback=63, max_gross_exposure=1.0, dd_threshold=None, concentration_limit=None), True),
        ("+ concentration",     dict(vol_lookback=63, max_gross_exposure=1.0, dd_threshold=None, concentration_limit=0.40), True),
        ("+ DD throttle",       dict(vol_lookback=63, max_gross_exposure=1.0, dd_threshold=-0.05, concentration_limit=0.40), True),
        ("+ costs",             dict(vol_lookback=63, max_gross_exposure=1.0, dd_threshold=-0.05, concentration_limit=0.40), False),
    ]

    print(f"\n  {'Layer':<22s} │ {'IS SR':>7s} {'VAL SR':>8s} {'IS Ret':>8s} {'IS Vol':>8s} {'IS DD':>8s}")
    print("  " + "─" * 60)

    for layer_label, params, zero_cost in ablation:
        if zero_cost:
            abl_sleeves = []
            for sl in best_sleeves:
                new_ss = [SubStrategy(s.name, s.ticker, s.signal_obj, scale_costs(s.cost_model, 0), s.weight)
                          for s in sl.sub_strategies]
                abl_sleeves.append(Sleeve(sl.name, new_ss, sl.risk_budget, sl.internal_weighting, sl.vol_target))
        else:
            abl_sleeves = best_sleeves

        try:
            r_is = build_sleeve_portfolio(is_u, abl_sleeves, **params)
            r_val = build_sleeve_portfolio(val_u, abl_sleeves, **params)
            ri = full_report(r_is.net_returns)
            rv = full_report(r_val.net_returns)
            print(f"  {layer_label:<22s} │ {ri['sharpe_ratio']:>+6.3f} {rv['sharpe_ratio']:>+7.3f} "
                  f"{ri['annualised_return']:>+7.2%} {ri['annualised_volatility']:>7.2%} {ri['max_drawdown']:>+7.2%}")
        except Exception as e:
            print(f"  {layer_label:<22s} │ ERROR: {e}")

    # ==================================================================
    # PART D: FINAL VAL VALIDATION (ONE-SHOT)
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Part D: Final Validation (One-Shot — Consumes Holdout)")
    print("═" * 80)

    print(f"\n  Selected configuration: Sleeve {best_budget_label}")

    for split, u in [("IS", is_u), ("VAL", val_u)]:
        r = build_sleeve_portfolio(u, best_sleeves, **PP_RISK)
        rep = full_report(r.net_returns, costs=r.costs)
        tail = tail_risk_stats(r.net_returns)

        print(f"\n  {split}:")
        print(f"    Sharpe:          {rep['sharpe_ratio']:+.3f}")
        print(f"    Ann. Return:     {rep['annualised_return']:+.2%}")
        print(f"    Ann. Volatility: {rep['annualised_volatility']:.2%}")
        print(f"    Max Drawdown:    {rep['max_drawdown']:+.2%}")
        print(f"    Calmar:          {abs(rep['annualised_return']/rep['max_drawdown']) if abs(rep['max_drawdown']) > 1e-6 else 0:.2f}")
        print(f"    VaR (5%):        {tail['var_5pct']:+.2%}")
        print(f"    CVaR (5%):       {tail['cvar_5pct']:+.2%}")
        print(f"    Gross Exposure:  {r.gross_exposure.mean():.1%}")
        print(f"    Total Costs:     {r.costs.sum():.4f}")
        print(f"    Positive Months: {rep.get('pct_positive_months', 0):.0%}")

    # Outer WF with selected budget
    print(f"\n  Walk-Forward (selected budget):")
    wf_sr, fold_srs = _wf_sharpe_sleeve(universe, best_sleeves, PP_RISK)
    n_pos = sum(1 for s in fold_srs if not np.isnan(s) and s > 0)
    n_neg = sum(1 for s in fold_srs if not np.isnan(s) and s <= 0)
    fold_str = " ".join(f"{s:>+.1f}" for s in fold_srs if not np.isnan(s))
    print(f"    Aggregate WF Sharpe: {wf_sr:+.3f}")
    print(f"    Folds ({n_pos}+/{n_neg}−): [{fold_str}]")

    # ==================================================================
    # COMPARISON: Old vs New
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Comparison: Old Equal-Weight vs New Sleeve Portfolio")
    print("═" * 80)

    old_is = results.get("Equal (old)", {})
    new_is = results.get(best_budget_label, {})

    print(f"\n  {'Metric':<25s} {'Old (Equal)':>14s} {'New (Sleeve)':>14s} {'Δ':>10s}")
    print("  " + "-" * 65)
    for m, label in [("is_sr", "IS Sharpe"), ("val_sr", "VAL Sharpe"), ("wf_sr", "WF Sharpe"),
                     ("is_ret", "IS Ann. Return"), ("is_dd", "IS Max DD")]:
        o = old_is.get(m, 0)
        n = new_is.get(m, 0)
        d = n - o
        if "ret" in m or "dd" in m:
            print(f"  {label:<25s} {o:>+13.2%} {n:>+13.2%} {d:>+9.2%}")
        else:
            print(f"  {label:<25s} {o:>+13.3f} {n:>+13.3f} {d:>+9.3f}")

    # Charts
    print("\n\nGenerating charts...")

    # Chart: Budget comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    labels = list(results.keys())
    for ax, metric, title in zip(axes,
                                  ["is_sr", "val_sr", "wf_sr"],
                                  ["IS Sharpe", "VAL Sharpe", "WF Sharpe"]):
        vals = [results[l][metric] for l in labels]
        colors = ["#27ae60" if l == best_budget_label else ("#3498db" if l == "Equal (old)" else "#95a5a6") for l in labels]
        bars = ax.bar(range(len(labels)), vals, color=colors, alpha=0.85, edgecolor="white")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(title, fontsize=12, fontweight="bold")
    fig.suptitle("M16: Risk-Budget Allocation Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "01_budget_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 01_budget_comparison.png")

    print(f"\n  Charts saved to: {CHART_DIR}/")
    print("=" * 80)
    print("  M16 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
