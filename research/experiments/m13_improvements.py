"""
Milestone 13 — Senior-Level Improvements
==========================================

Addresses the critical gaps identified in the senior quant review:

  1. IC Analysis — Information Coefficient for all signals
  2. IC Decay Curves — how signal predictive power degrades with lag
  3. P&L Attribution — by asset, by direction, by year, by cost
  4. Equal-Weight vs Inverse-Vol Weight comparison
  5. Tail Risk Analysis — VaR, CVaR, stress tests
  6. Deflated Sharpe Ratio — multiple testing correction
  7. New Signal: Volatility Mean-Reversion — decorrelated alpha source

Usage:
    python research/experiments/m13_improvements.py
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

from quant_research.backtest.costs import EQUITY_COSTS, CRYPTO_COSTS, FUTURES_COSTS
from quant_research.backtest.engine import run_backtest
from quant_research.data.loader import load_config, load_universe, split_data
from quant_research.metrics.attribution import (
    asset_pnl_attribution, long_short_attribution,
    yearly_attribution, cost_attribution, format_attribution_report,
)
from quant_research.metrics.ic_analysis import (
    ic_summary, ic_decay_curve, ic_by_year, rolling_ic, format_ic_report,
)
from quant_research.metrics.performance import (
    full_report, format_report, sharpe_ratio, deflated_sharpe_ratio,
    tail_risk_stats, value_at_risk, conditional_var,
)
from quant_research.portfolio.allocator import SubStrategy, build_portfolio
from quant_research.strategies.mean_reversion_smooth import SmoothedMeanReversionSignal
from quant_research.strategies.trend_following import TrendFollowingSignal
from quant_research.strategies.volatility_signal import VolatilityMeanReversionSignal

CHART_DIR = PROJECT_ROOT / "research" / "experiments" / "m13_charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

PORTFOLIO_PARAMS = dict(
    vol_target=0.10, vol_lookback=63, max_gross_exposure=1.0,
    dd_threshold=-0.05, concentration_limit=0.40,
)


def get_sub_strategies():
    """All 5 original sub-strategies."""
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


def get_sub_strategies_with_vol():
    """Original 5 + volatility signal sub-strategies."""
    base = get_sub_strategies()
    vol_strategies = [
        SubStrategy("BTC_VOL", "BTC-USD",
                     VolatilityMeanReversionSignal(short_window=10, long_window=60),
                     CRYPTO_COSTS, weight=1.0),
        SubStrategy("ETH_VOL", "ETH-USD",
                     VolatilityMeanReversionSignal(short_window=10, long_window=60),
                     CRYPTO_COSTS, weight=1.0),
        SubStrategy("SPY_VOL", "SPY",
                     VolatilityMeanReversionSignal(short_window=10, long_window=60),
                     EQUITY_COSTS, weight=1.0),
    ]
    return base + vol_strategies


def main():
    print("=" * 70)
    print("MILESTONE 13 — Senior-Level Improvements")
    print("=" * 70)

    config = load_config()
    universe = load_universe(config)

    is_universe = {}
    val_universe = {}
    for ticker, df in universe.items():
        splits = split_data(df, config)
        is_universe[ticker] = splits["in_sample"]
        if "validation" in splits:
            val_universe[ticker] = splits["validation"]

    # ═════════════════════════════════════════════════════════════════════
    # 1. IC ANALYSIS — All Sub-Strategy Signals
    # ═════════════════════════════════════════════════════════════════════
    print("\n[1/7] Information Coefficient analysis...")

    all_signals = [
        ("BTC_TF60", "BTC-USD", TrendFollowingSignal(lookback=60, scale=2.0)),
        ("ETH_TF120", "ETH-USD", TrendFollowingSignal(lookback=120, scale=2.0)),
        ("QQQ_SMR10", "QQQ", SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=10)),
        ("SPY_SMR10", "SPY", SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=10)),
        ("GC_SMR5", "GC=F", SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=5)),
        ("BTC_VOL", "BTC-USD", VolatilityMeanReversionSignal(short_window=10, long_window=60)),
        ("ETH_VOL", "ETH-USD", VolatilityMeanReversionSignal(short_window=10, long_window=60)),
        ("SPY_VOL", "SPY", VolatilityMeanReversionSignal(short_window=10, long_window=60)),
    ]

    ic_results = []
    for name, ticker, signal_obj in all_signals:
        data = is_universe[ticker]
        signal = signal_obj.generate(data)
        fwd_ret = data["close"].pct_change().shift(-1)

        summary = ic_summary(signal, fwd_ret)
        summary["name"] = name
        summary["ticker"] = ticker
        ic_results.append(summary)

        print(f"\n  {name}:")
        print(f"    IC Mean={summary['ic_mean']:.4f}  IC_IR={summary['ic_ir']:.4f}  "
              f"Hit Rate={summary['hit_rate']:.2%}  "
              f"Spearman={summary['spearman_corr']:.4f} (p={summary['spearman_p']:.4f})  "
              f"Avg|signal|={summary['avg_signal_magnitude']:.3f}")

    # IC comparison table
    ic_df = pd.DataFrame(ic_results)
    print("\n  IC COMPARISON TABLE:")
    print("  " + "-" * 90)
    print(f"  {'Signal':<12s} {'IC Mean':>8s} {'IC_IR':>8s} {'Hit Rate':>9s} "
          f"{'Spearman':>9s} {'p-value':>8s} {'Avg|sig|':>9s} {'T-stat':>8s}")
    print("  " + "-" * 90)
    for _, row in ic_df.iterrows():
        sig = "***" if row['spearman_p'] < 0.01 else ("**" if row['spearman_p'] < 0.05 else ("*" if row['spearman_p'] < 0.10 else ""))
        print(f"  {row['name']:<12s} {row['ic_mean']:>+8.4f} {row['ic_ir']:>8.4f} "
              f"{row['hit_rate']:>8.2%} "
              f"{row['spearman_corr']:>+8.4f} {row['spearman_p']:>8.4f}{sig} "
              f"{row['avg_signal_magnitude']:>8.3f} {row['t_stat']:>+8.3f}")
    print("  " + "-" * 90)

    # ═════════════════════════════════════════════════════════════════════
    # 2. IC DECAY CURVES
    # ═════════════════════════════════════════════════════════════════════
    print("\n[2/7] IC decay curves...")

    decay_results = {}
    for name, ticker, signal_obj in all_signals:
        data = is_universe[ticker]
        signal = signal_obj.generate(data)
        returns = data["close"].pct_change()

        decay = ic_decay_curve(signal, returns, lags=[1, 2, 3, 5, 10, 20, 40])
        decay["name"] = name
        decay_results[name] = decay

        # Print condensed
        lag1_ic = decay[decay["lag"] == 1]["spearman_corr"].values[0] if len(decay) > 0 else 0
        lag5_ic = decay[decay["lag"] == 5]["spearman_corr"].values[0] if len(decay[decay["lag"] == 5]) > 0 else 0
        lag20_ic = decay[decay["lag"] == 20]["spearman_corr"].values[0] if len(decay[decay["lag"] == 20]) > 0 else 0
        print(f"  {name:<12s}  IC@1d={lag1_ic:+.4f}  IC@5d={lag5_ic:+.4f}  IC@20d={lag20_ic:+.4f}")

    # ═════════════════════════════════════════════════════════════════════
    # 3. IC BY YEAR
    # ═════════════════════════════════════════════════════════════════════
    print("\n[3/7] IC by year...")

    for name, ticker, signal_obj in all_signals[:5]:  # Original 5 only
        data = is_universe[ticker]
        signal = signal_obj.generate(data)
        fwd_ret = data["close"].pct_change().shift(-1)

        yearly = ic_by_year(signal, fwd_ret)
        if len(yearly) > 0:
            print(f"\n  {name}:")
            for _, row in yearly.iterrows():
                print(f"    {int(row['year'])}:  IC={row['ic_mean']:+.4f}  "
                      f"Hit={row['hit_rate']:.1%}  "
                      f"Spearman={row['spearman_corr']:+.4f}")

    # ═════════════════════════════════════════════════════════════════════
    # 4. P&L ATTRIBUTION
    # ═════════════════════════════════════════════════════════════════════
    print("\n[4/7] P&L attribution...")

    result = build_portfolio(is_universe, get_sub_strategies(), **PORTFOLIO_PARAMS)

    # Build asset returns df for attribution
    tickers = list(result.asset_positions.columns)
    asset_ret_df = pd.DataFrame({
        t: is_universe[t].loc[result.asset_positions.index, "close"].pct_change().fillna(0.0)
        for t in tickers
    })

    # Asset attribution
    asset_attr = asset_pnl_attribution(result.asset_positions, asset_ret_df)

    # Long/short attribution
    ls_attr = long_short_attribution(result.asset_positions, asset_ret_df)

    # Yearly attribution
    year_attr = yearly_attribution(result.net_returns, result.gross_returns, result.costs)

    # Cost attribution
    cost_rates = {}
    for ss in get_sub_strategies():
        cost_rates[ss.ticker] = ss.cost_model.total_rate
    cost_attr = cost_attribution(result.asset_positions, cost_rates)

    # Print
    print(format_attribution_report(asset_attr, ls_attr, year_attr, cost_attr))

    # ═════════════════════════════════════════════════════════════════════
    # 5. EQUAL WEIGHT vs INVERSE-VOL WEIGHT
    # ═════════════════════════════════════════════════════════════════════
    print("\n[5/7] Equal weight vs inverse-vol weight comparison...")

    eq_result = build_portfolio(
        is_universe, get_sub_strategies(), **PORTFOLIO_PARAMS, weighting_scheme="equal"
    )
    iv_result = build_portfolio(
        is_universe, get_sub_strategies(), **PORTFOLIO_PARAMS, weighting_scheme="inverse_vol"
    )

    eq_report = full_report(eq_result.net_returns, costs=eq_result.costs)
    iv_report = full_report(iv_result.net_returns, costs=iv_result.costs)

    print(f"\n  {'Metric':<25s} {'Equal Weight':>14s} {'Inverse Vol':>14s}")
    print("  " + "-" * 55)
    metrics_cmp = [
        ("Annualised Return", "annualised_return", "{:.2%}"),
        ("Sharpe Ratio", "sharpe_ratio", "{:.3f}"),
        ("Max Drawdown", "max_drawdown", "{:.2%}"),
        ("Avg Monthly Return", "avg_monthly_return", "{:.2%}"),
        ("% Positive Months", "pct_positive_months", "{:.1%}"),
        ("VaR (5%)", "var_5pct", "{:.2%}"),
        ("CVaR (5%)", "cvar_5pct", "{:.2%}"),
        ("Total Costs", "total_costs", "{:.4f}"),
    ]
    for label, key, fmt in metrics_cmp:
        eq_val = eq_report.get(key, 0)
        iv_val = iv_report.get(key, 0)
        print(f"  {label:<25s} {fmt.format(eq_val):>14s} {fmt.format(iv_val):>14s}")

    print(f"\n  Avg Gross Exposure:  EQ={eq_result.gross_exposure.mean():.1%}  "
          f"IV={iv_result.gross_exposure.mean():.1%}")

    # Also test on validation
    eq_val_result = build_portfolio(
        val_universe, get_sub_strategies(), **PORTFOLIO_PARAMS, weighting_scheme="equal"
    )
    iv_val_result = build_portfolio(
        val_universe, get_sub_strategies(), **PORTFOLIO_PARAMS, weighting_scheme="inverse_vol"
    )
    eq_val_rep = full_report(eq_val_result.net_returns)
    iv_val_rep = full_report(iv_val_result.net_returns)

    print(f"\n  VALIDATION:")
    print(f"  {'Metric':<25s} {'Equal Weight':>14s} {'Inverse Vol':>14s}")
    print("  " + "-" * 55)
    for label, key, fmt in metrics_cmp[:5]:
        eq_val = eq_val_rep.get(key, 0)
        iv_val = iv_val_rep.get(key, 0)
        print(f"  {label:<25s} {fmt.format(eq_val):>14s} {fmt.format(iv_val):>14s}")

    # ═════════════════════════════════════════════════════════════════════
    # 6. TAIL RISK & DEFLATED SHARPE
    # ═════════════════════════════════════════════════════════════════════
    print("\n[6/7] Tail risk analysis & Deflated Sharpe Ratio...")

    tail = tail_risk_stats(result.net_returns)
    print("\n  TAIL RISK (In-Sample):")
    print(f"    VaR  (1%):   {tail['var_1pct']:.2%}   (1-in-100 worst day)")
    print(f"    VaR  (5%):   {tail['var_5pct']:.2%}   (1-in-20 worst day)")
    print(f"    CVaR (1%):   {tail['cvar_1pct']:.2%}   (avg loss in worst 1%)")
    print(f"    CVaR (5%):   {tail['cvar_5pct']:.2%}   (avg loss in worst 5%)")
    print(f"    Skewness:    {tail['skewness']:.3f}")
    print(f"    Ex. Kurtosis:{tail['excess_kurtosis']:.3f}")
    print(f"    Worst day:   {tail['worst_day']:.2%}")
    print(f"    Best day:    {tail['best_day']:.2%}")
    print(f"    Worst 5 avg: {tail['worst_5_days_avg']:.2%}")

    # Stress test: worst N-day drawdowns
    print("\n  STRESS TESTS:")
    for window_name, window_days in [("1-week", 5), ("2-week", 10), ("1-month", 21)]:
        rolling_ret = result.net_returns.rolling(window_days).sum()
        worst = rolling_ret.min()
        worst_date = rolling_ret.idxmin()
        print(f"    Worst {window_name}: {worst:.2%} (ending {worst_date.strftime('%Y-%m-%d')})")

    # Deflated Sharpe Ratio
    observed_sharpe = eq_report["sharpe_ratio"]
    n_trials = 30  # From M7: 30 backtests in the grid search
    n_obs = eq_report["n_trading_days"]
    skew = tail["skewness"]
    kurt = tail["excess_kurtosis"] + 3  # DSR expects raw kurtosis

    dsr = deflated_sharpe_ratio(observed_sharpe, n_trials, n_obs, skew, kurt)

    print(f"\n  DEFLATED SHARPE RATIO (Harvey & Liu 2015):")
    print(f"    Observed Sharpe:  {observed_sharpe:.3f}")
    print(f"    N trials tested:  {n_trials}")
    print(f"    N observations:   {n_obs}")
    print(f"    DSR probability:  {dsr:.4f}")
    if dsr > 0.95:
        print(f"    ✓ Survives multiple testing at 95% level")
    elif dsr > 0.90:
        print(f"    ○ Marginal — survives at 90% but not 95%")
    else:
        print(f"    ✗ Does NOT survive multiple testing")

    # ═════════════════════════════════════════════════════════════════════
    # 7. VOLATILITY SIGNAL EVALUATION
    # ═════════════════════════════════════════════════════════════════════
    print("\n[7/7] Volatility signal standalone evaluation...")

    vol_strats = [
        ("BTC_VOL", "BTC-USD", VolatilityMeanReversionSignal(short_window=10, long_window=60), CRYPTO_COSTS),
        ("ETH_VOL", "ETH-USD", VolatilityMeanReversionSignal(short_window=10, long_window=60), CRYPTO_COSTS),
        ("SPY_VOL", "SPY", VolatilityMeanReversionSignal(short_window=10, long_window=60), EQUITY_COSTS),
        ("QQQ_VOL", "QQQ", VolatilityMeanReversionSignal(short_window=10, long_window=60), EQUITY_COSTS),
        ("GC_VOL", "GC=F", VolatilityMeanReversionSignal(short_window=10, long_window=60), FUTURES_COSTS),
    ]

    print(f"\n  {'Name':<12s} {'Sharpe':>8s} {'Ann.Ret':>9s} {'MaxDD':>8s} {'Hit%':>7s} {'AvgMo':>8s}")
    print("  " + "-" * 55)
    for name, ticker, signal_obj, cost_model in vol_strats:
        data = is_universe[ticker]
        signal = signal_obj.generate(data)
        prices = data["close"]
        bt = run_backtest(prices=prices, signal=signal, cost_model=cost_model)
        rep = full_report(bt.net_returns)
        print(f"  {name:<12s} {rep['sharpe_ratio']:>+7.3f} {rep['annualised_return']:>+8.2%} "
              f"{rep['max_drawdown']:>+7.2%} {rep['win_rate']:>6.1%} "
              f"{rep['avg_monthly_return']:>+7.2%}")

    # Compare: original 5 vs original 5 + vol signals
    print("\n  PORTFOLIO COMPARISON: 5 original vs 5+3 vol signals:")

    result_5 = build_portfolio(is_universe, get_sub_strategies(), **PORTFOLIO_PARAMS)
    result_8 = build_portfolio(is_universe, get_sub_strategies_with_vol(), **PORTFOLIO_PARAMS)

    rep_5 = full_report(result_5.net_returns, costs=result_5.costs)
    rep_8 = full_report(result_8.net_returns, costs=result_8.costs)

    print(f"\n  {'Metric':<25s} {'5 Original':>14s} {'5+3 Vol':>14s}")
    print("  " + "-" * 55)
    for label, key, fmt in metrics_cmp:
        v5 = rep_5.get(key, 0)
        v8 = rep_8.get(key, 0)
        print(f"  {label:<25s} {fmt.format(v5):>14s} {fmt.format(v8):>14s}")

    # Correlation of vol signal returns with TF/SMR returns
    print("\n  SIGNAL RETURN CORRELATIONS (vol signals vs existing):")
    strategies = get_sub_strategies_with_vol()
    strat_returns = {}
    for ss in strategies:
        data = is_universe[ss.ticker]
        signal = ss.signal_obj.generate(data)
        prices = data["close"]
        bt = run_backtest(prices=prices, signal=signal, cost_model=ss.cost_model)
        strat_returns[ss.name] = bt.net_returns

    ret_df = pd.DataFrame(strat_returns).dropna()
    corr = ret_df.corr()
    # Show vol signal correlations with TF/SMR
    vol_names = [s.name for s in strategies if "VOL" in s.name]
    other_names = [s.name for s in strategies if "VOL" not in s.name]
    if vol_names and other_names:
        print(f"\n  {'':12s}", end="")
        for vn in vol_names:
            print(f" {vn:>10s}", end="")
        print()
        for on in other_names:
            print(f"  {on:<12s}", end="")
            for vn in vol_names:
                print(f" {corr.loc[on, vn]:>+9.3f}", end="")
            print()

    # ═════════════════════════════════════════════════════════════════════
    # CHARTS
    # ═════════════════════════════════════════════════════════════════════
    print("\n\nGenerating charts...")

    # Chart 1: IC comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 5))
    ic_names = ic_df["name"].tolist()
    ic_vals = ic_df["spearman_corr"].tolist()
    colors = ["green" if v > 0 else "red" for v in ic_vals]
    bars = ax.bar(ic_names, ic_vals, color=colors, alpha=0.8, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Spearman IC by Signal (In-Sample)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Spearman Correlation")
    ax.set_xticklabels(ic_names, rotation=45, ha="right")
    for bar, v in zip(bars, ic_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{v:+.4f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "01_ic_comparison.png", dpi=150)
    plt.close(fig)
    print("  ✓ 01_ic_comparison.png")

    # Chart 2: IC decay curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, group, title in [
        (axes[0], [("BTC_TF60", "blue"), ("ETH_TF120", "orange")], "Trend-Following IC Decay"),
        (axes[1], [("QQQ_SMR10", "green"), ("SPY_SMR10", "red"), ("GC_SMR5", "purple")], "Mean-Reversion IC Decay"),
    ]:
        for name, color in group:
            if name in decay_results:
                df = decay_results[name]
                ax.plot(df["lag"], df["spearman_corr"], "-o", label=name, color=color, linewidth=1.5)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Holding Period (days)")
        ax.set_ylabel("Spearman Correlation")
        ax.legend(fontsize=9)
    fig.suptitle("IC Decay Curves", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "02_ic_decay.png", dpi=150)
    plt.close(fig)
    print("  ✓ 02_ic_decay.png")

    # Chart 3: P&L attribution pie chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # Asset attribution
    positive_assets = asset_attr[asset_attr["total_pnl"] > 0]
    negative_assets = asset_attr[asset_attr["total_pnl"] <= 0]
    all_vals = asset_attr["total_pnl"].abs().values
    all_labels = [f"{r['asset']}\n{r['total_pnl']:+.3f}" for _, r in asset_attr.iterrows()]
    colors_pie = ["#2ecc71" if v > 0 else "#e74c3c" for v in asset_attr["total_pnl"].values]
    axes[0].barh(asset_attr["asset"], asset_attr["total_pnl"], color=colors_pie, alpha=0.8)
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_title("P&L by Asset", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Total P&L (fraction of capital)")

    # Long/short attribution
    ls_vals = [ls_attr["long_pnl"], ls_attr["short_pnl"]]
    ls_labels = [f"Long\n{ls_attr['long_pnl']:+.3f}", f"Short\n{ls_attr['short_pnl']:+.3f}"]
    ls_colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in ls_vals]
    axes[1].bar(["Long", "Short"], ls_vals, color=ls_colors, alpha=0.8, width=0.5)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("P&L by Direction", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Total P&L (fraction of capital)")
    fig.suptitle("P&L Attribution (In-Sample)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "03_pnl_attribution.png", dpi=150)
    plt.close(fig)
    print("  ✓ 03_pnl_attribution.png")

    # Chart 4: Equal vs Inverse-Vol equity curves
    fig, ax = plt.subplots(figsize=(14, 7))
    eq_equity = eq_result.equity / eq_result.initial_capital
    iv_equity = iv_result.equity / iv_result.initial_capital
    ax.plot(eq_equity.index, eq_equity.values, label=f"Equal Weight (Sharpe={eq_report['sharpe_ratio']:.3f})",
            linewidth=1.5, color="blue")
    ax.plot(iv_equity.index, iv_equity.values, label=f"Inverse Vol (Sharpe={iv_report['sharpe_ratio']:.3f})",
            linewidth=1.5, color="orange")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title("Equal Weight vs Inverse-Vol Weight (In-Sample)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Equity (multiple of initial)")
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "04_equal_vs_invvol.png", dpi=150)
    plt.close(fig)
    print("  ✓ 04_equal_vs_invvol.png")

    # Chart 5: Return distribution with VaR/CVaR
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(result.net_returns.values * 100, bins=80, color="steelblue", alpha=0.7,
            edgecolor="white", density=True)
    var5 = tail["var_5pct"] * 100
    cvar5 = tail["cvar_5pct"] * 100
    ax.axvline(var5, color="orange", linewidth=2, linestyle="--",
               label=f"VaR(5%) = {var5:.2f}%")
    ax.axvline(cvar5, color="red", linewidth=2, linestyle="--",
               label=f"CVaR(5%) = {cvar5:.2f}%")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Daily Return Distribution with Tail Risk Markers", fontsize=14, fontweight="bold")
    ax.set_xlabel("Daily Return (%)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "05_tail_risk.png", dpi=150)
    plt.close(fig)
    print("  ✓ 05_tail_risk.png")

    # Chart 6: Correlation heatmap (all 8 signals)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center",
                    va="center", fontsize=7,
                    color="white" if abs(corr.values[i, j]) > 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Signal Return Correlations (8 sub-strategies)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "06_correlation_8strats.png", dpi=150)
    plt.close(fig)
    print("  ✓ 06_correlation_8strats.png")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("M13 SUMMARY")
    print("=" * 70)

    # Find best and worst IC
    best_ic = ic_df.loc[ic_df["spearman_corr"].abs().idxmax()]
    print(f"\n  Highest |IC|: {best_ic['name']} (Spearman={best_ic['spearman_corr']:+.4f})")

    # Significance
    sig_signals = ic_df[ic_df["spearman_p"] < 0.05]
    print(f"  Signals with p<0.05: {len(sig_signals)}/{len(ic_df)}")
    for _, r in sig_signals.iterrows():
        print(f"    {r['name']}: Spearman={r['spearman_corr']:+.4f} (p={r['spearman_p']:.4f})")

    # Attribution summary
    print(f"\n  P&L Attribution:")
    print(f"    Long contribution: {ls_attr['long_pct']:+.1f}%")
    print(f"    Short contribution: {ls_attr['short_pct']:+.1f}%")

    # Weighting comparison
    print(f"\n  Weighting Comparison (IS):")
    print(f"    Equal:      Sharpe={eq_report['sharpe_ratio']:.3f}")
    print(f"    Inv-Vol:    Sharpe={iv_report['sharpe_ratio']:.3f}")

    # DSR
    print(f"\n  Deflated Sharpe Ratio: {dsr:.4f}")

    print(f"\n  Charts saved to: {CHART_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
