"""
Milestone 14 — Rigorous Validation & Alpha Decomposition
==========================================================

Five parts in strict order:

  Part 1: IS → VAL decay attribution (regime / asset / year / signal)
  Part 2: Robust IC (HAC/Newey-West + block bootstrap + FDR), IS vs VAL
  Part 3: Alpha vs beta decomposition (benchmark vs buy-and-hold)
  Part 4: VOL signal validation on VAL
  Part 5: Walk-forward, parameter stability, cost stress

Usage:
    python research/experiments/m14_rigorous_validation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant_research.backtest.costs import (
    EQUITY_COSTS, CRYPTO_COSTS, FUTURES_COSTS, scale_costs,
)
from quant_research.backtest.engine import run_backtest
from quant_research.data.loader import load_config, load_universe, split_data
from quant_research.metrics.attribution import (
    asset_pnl_attribution, long_short_attribution, yearly_attribution,
)
from quant_research.metrics.ic_analysis import (
    compute_ic, ic_summary, ic_newey_west, ic_block_bootstrap,
    benjamini_hochberg,
)
from quant_research.metrics.performance import (
    full_report, sharpe_ratio, deflated_sharpe_ratio, tail_risk_stats,
)
from quant_research.portfolio.allocator import SubStrategy, build_portfolio
from quant_research.strategies.mean_reversion_smooth import SmoothedMeanReversionSignal
from quant_research.strategies.trend_following import TrendFollowingSignal
from quant_research.strategies.volatility_signal import VolatilityMeanReversionSignal

CHART_DIR = PROJECT_ROOT / "research" / "experiments" / "m14_charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

PORTFOLIO_PARAMS = dict(
    vol_target=0.10, vol_lookback=63, max_gross_exposure=1.0,
    dd_threshold=-0.05, concentration_limit=0.40,
)

ALL_SUBSTRATS = [
    ("BTC_TF60",  "BTC-USD", TrendFollowingSignal(lookback=60, scale=2.0), CRYPTO_COSTS),
    ("ETH_TF120", "ETH-USD", TrendFollowingSignal(lookback=120, scale=2.0), CRYPTO_COSTS),
    ("QQQ_SMR10", "QQQ",     SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=10), EQUITY_COSTS),
    ("SPY_SMR10", "SPY",     SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=10), EQUITY_COSTS),
    ("GC_SMR5",   "GC=F",    SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=5), FUTURES_COSTS),
]

VOL_SUBSTRATS = [
    ("BTC_VOL", "BTC-USD", VolatilityMeanReversionSignal(short_window=10, long_window=60), CRYPTO_COSTS),
    ("ETH_VOL", "ETH-USD", VolatilityMeanReversionSignal(short_window=10, long_window=60), CRYPTO_COSTS),
    ("SPY_VOL", "SPY",     VolatilityMeanReversionSignal(short_window=10, long_window=60), EQUITY_COSTS),
]


def _make_ss(specs):
    return [SubStrategy(n, t, s, c, weight=1.0) for n, t, s, c in specs]


def main():
    print("=" * 70)
    print("MILESTONE 14 — Rigorous Validation & Alpha Decomposition")
    print("=" * 70)

    config = load_config()
    universe = load_universe(config)

    is_universe, val_universe = {}, {}
    for ticker, df in universe.items():
        splits = split_data(df, config)
        is_universe[ticker] = splits["in_sample"]
        if "validation" in splits:
            val_universe[ticker] = splits["validation"]

    # ═════════════════════════════════════════════════════════════════════
    # PART 1: IS → VAL DECAY ATTRIBUTION
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  PART 1: IS → VAL Sharpe Decay Attribution")
    print("═" * 70)

    # 1a. Individual sub-strategy performance: IS vs VAL
    print("\n  [1a] Individual sub-strategy Sharpe: IS vs VAL")
    print(f"  {'Signal':<12s} {'IS Sharpe':>10s} {'VAL Sharpe':>11s} {'Decay':>8s} {'IS AvgMo':>9s} {'VAL AvgMo':>10s}")
    print("  " + "-" * 65)

    for name, ticker, signal_obj, cost_model in ALL_SUBSTRATS:
        for split_name, split_u in [("IS", is_universe), ("VAL", val_universe)]:
            data = split_u[ticker]
            signal = signal_obj.generate(data)
            bt = run_backtest(prices=data["close"], signal=signal, cost_model=cost_model)
            rep = full_report(bt.net_returns)
            if split_name == "IS":
                is_sharpe, is_avg_mo = rep["sharpe_ratio"], rep["avg_monthly_return"]
            else:
                val_sharpe, val_avg_mo = rep["sharpe_ratio"], rep["avg_monthly_return"]

        decay = (1 - val_sharpe / is_sharpe) * 100 if abs(is_sharpe) > 0.01 else float("nan")
        print(f"  {name:<12s} {is_sharpe:>+9.3f} {val_sharpe:>+10.3f} {decay:>+7.0f}% "
              f"{is_avg_mo:>+8.2%} {val_avg_mo:>+9.2%}")

    # 1b. Portfolio-level attribution on VAL
    print("\n  [1b] Portfolio attribution on VALIDATION data")
    val_result = build_portfolio(val_universe, _make_ss(ALL_SUBSTRATS), **PORTFOLIO_PARAMS)
    val_report = full_report(val_result.net_returns, costs=val_result.costs)

    tickers = list(val_result.asset_positions.columns)
    val_asset_ret = pd.DataFrame({
        t: val_universe[t].loc[val_result.asset_positions.index, "close"].pct_change().fillna(0.0)
        for t in tickers
    })

    val_asset_attr = asset_pnl_attribution(val_result.asset_positions, val_asset_ret)
    val_ls_attr = long_short_attribution(val_result.asset_positions, val_asset_ret)

    print("\n  Asset P&L (VAL):")
    for _, row in val_asset_attr.iterrows():
        print(f"    {row['asset']:<12s}  P&L={row['total_pnl']:>+8.4f}  ({row['pct_of_total']:>+6.1f}%)  "
              f"AvgPos={row['avg_position']:>+.3f}  Long={row['n_long_days']}d  Short={row['n_short_days']}d")

    print(f"\n  Direction (VAL):")
    print(f"    Long:  P&L={val_ls_attr['long_pnl']:>+.4f} ({val_ls_attr['long_pct']:>+.1f}%)  Sharpe={val_ls_attr['long_sharpe']:.3f}")
    print(f"    Short: P&L={val_ls_attr['short_pnl']:>+.4f} ({val_ls_attr['short_pct']:>+.1f}%)  Sharpe={val_ls_attr['short_sharpe']:.3f}")

    # 1c. Signal magnitude comparison
    print("\n  [1c] Signal magnitude IS vs VAL")
    print(f"  {'Signal':<12s} {'IS Avg|sig|':>12s} {'VAL Avg|sig|':>13s} {'IS %active':>11s} {'VAL %active':>12s}")
    print("  " + "-" * 60)
    for name, ticker, signal_obj, _ in ALL_SUBSTRATS:
        is_sig = signal_obj.generate(is_universe[ticker])
        val_sig = signal_obj.generate(val_universe[ticker])
        is_mag = is_sig.abs().mean()
        val_mag = val_sig.abs().mean()
        is_act = (is_sig.abs() > 0.01).mean()
        val_act = (val_sig.abs() > 0.01).mean()
        print(f"  {name:<12s} {is_mag:>11.3f} {val_mag:>12.3f} {is_act:>10.1%} {val_act:>11.1%}")

    # 1d. Gross exposure comparison
    is_result = build_portfolio(is_universe, _make_ss(ALL_SUBSTRATS), **PORTFOLIO_PARAMS)
    print(f"\n  [1d] Gross Exposure: IS mean={is_result.gross_exposure.mean():.1%}, "
          f"VAL mean={val_result.gross_exposure.mean():.1%}")

    # ═════════════════════════════════════════════════════════════════════
    # PART 2: ROBUST IC (HAC + BLOCK BOOTSTRAP + FDR)
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  PART 2: Statistically Rigorous IC Analysis")
    print("═" * 70)

    all_signals_spec = ALL_SUBSTRATS + VOL_SUBSTRATS

    # 2a. HAC/Newey-West IC: IS vs VAL
    print("\n  [2a] HAC/Newey-West IC — IS vs VAL")
    print(f"  {'Signal':<12s} │ {'IS IC':>7s} {'HAC t':>7s} {'HAC p':>7s} │ {'VAL IC':>7s} {'HAC t':>7s} {'HAC p':>7s} │ {'bw':>3s}")
    print("  " + "─" * 72)

    is_hac_results, val_hac_results = [], []
    signal_names = []

    for name, ticker, signal_obj, _ in all_signals_spec:
        signal_names.append(name)

        # IS
        is_data = is_universe[ticker]
        is_sig = signal_obj.generate(is_data)
        is_fwd = is_data["close"].pct_change().shift(-1)
        is_hac = ic_newey_west(is_sig, is_fwd)
        is_hac_results.append(is_hac)

        # VAL
        val_data = val_universe[ticker]
        val_sig = signal_obj.generate(val_data)
        val_fwd = val_data["close"].pct_change().shift(-1)
        val_hac = ic_newey_west(val_sig, val_fwd)
        val_hac_results.append(val_hac)

        print(f"  {name:<12s} │ {is_hac['ic_mean']:>+6.4f} {is_hac['hac_t_stat']:>+6.2f} {is_hac['hac_p_value']:>6.4f} │ "
              f"{val_hac['ic_mean']:>+6.4f} {val_hac['hac_t_stat']:>+6.2f} {val_hac['hac_p_value']:>6.4f} │ "
              f"{is_hac['bandwidth']:>3d}")

    # 2b. Block bootstrap
    print("\n  [2b] Block Bootstrap IC (1000 resamples, block=21d)")
    print(f"  {'Signal':<12s} │ {'IS 95% CI':>20s} {'P(>0)':>7s} │ {'VAL 95% CI':>20s} {'P(>0)':>7s}")
    print("  " + "─" * 75)

    for i, (name, ticker, signal_obj, _) in enumerate(all_signals_spec):
        is_data = is_universe[ticker]
        is_sig = signal_obj.generate(is_data)
        is_fwd = is_data["close"].pct_change().shift(-1)
        is_boot = ic_block_bootstrap(is_sig, is_fwd)

        val_data = val_universe[ticker]
        val_sig = signal_obj.generate(val_data)
        val_fwd = val_data["close"].pct_change().shift(-1)
        val_boot = ic_block_bootstrap(val_sig, val_fwd)

        print(f"  {name:<12s} │ [{is_boot['ci_lower_95']:>+.4f}, {is_boot['ci_upper_95']:>+.4f}] "
              f"{is_boot['p_gt_zero']:>6.1%} │ "
              f"[{val_boot['ci_lower_95']:>+.4f}, {val_boot['ci_upper_95']:>+.4f}] "
              f"{val_boot['p_gt_zero']:>6.1%}")

    # 2c. FDR correction
    print("\n  [2c] Benjamini-Hochberg FDR correction (α=0.05)")
    is_pvalues = [r["hac_p_value"] for r in is_hac_results]
    val_pvalues = [r["hac_p_value"] for r in val_hac_results]

    is_fdr = benjamini_hochberg(is_pvalues, alpha=0.05)
    val_fdr = benjamini_hochberg(val_pvalues, alpha=0.05)

    print(f"  {'Signal':<12s} │ {'IS raw p':>9s} {'IS adj p':>9s} {'IS rej':>7s} │ {'VAL raw p':>9s} {'VAL adj p':>9s} {'VAL rej':>7s}")
    print("  " + "─" * 72)
    for i, name in enumerate(signal_names):
        is_r = is_fdr[i]
        val_r = val_fdr[i]
        print(f"  {name:<12s} │ {is_r['original_p']:>8.4f} {is_r['adjusted_p']:>8.4f} {'✓' if is_r['reject'] else '✗':>5s}   │ "
              f"{val_r['original_p']:>8.4f} {val_r['adjusted_p']:>8.4f} {'✓' if val_r['reject'] else '✗':>5s}")

    n_is_sig = sum(1 for r in is_fdr if r["reject"])
    n_val_sig = sum(1 for r in val_fdr if r["reject"])
    print(f"\n  IS: {n_is_sig}/{len(is_fdr)} survive FDR    VAL: {n_val_sig}/{len(val_fdr)} survive FDR")

    # ═════════════════════════════════════════════════════════════════════
    # PART 3: ALPHA vs BETA DECOMPOSITION
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  PART 3: Alpha vs Beta — Is Crypto TF Just a Leveraged Long?")
    print("═" * 70)

    benchmarks = [
        ("BTC_TF60",  "BTC-USD", TrendFollowingSignal(lookback=60, scale=2.0), CRYPTO_COSTS),
        ("ETH_TF120", "ETH-USD", TrendFollowingSignal(lookback=120, scale=2.0), CRYPTO_COSTS),
    ]

    for name, ticker, signal_obj, cost_model in benchmarks:
        print(f"\n  ── {name} vs {ticker} Buy-and-Hold ──")

        for split_label, split_u in [("IS", is_universe), ("VAL", val_universe)]:
            data = split_u[ticker]
            prices = data["close"]
            asset_ret = prices.pct_change().fillna(0.0)

            # 1. Strategy returns
            signal = signal_obj.generate(data)
            bt_strat = run_backtest(prices=prices, signal=signal, cost_model=cost_model)
            strat_ret = bt_strat.net_returns

            # 2. Simple buy-and-hold (always long 100%)
            bah_signal = pd.Series(1.0, index=data.index)
            bt_bah = run_backtest(prices=prices, signal=bah_signal, cost_bps=0)
            bah_ret = bt_bah.net_returns

            # 3. Vol-targeted buy-and-hold (match portfolio vol target)
            vol = asset_ret.rolling(63, min_periods=20).std() * np.sqrt(252)
            vol_scale = (PORTFOLIO_PARAMS["vol_target"] / vol).clip(0, 2).shift(1).fillna(0)
            vol_bah_ret = vol_scale * asset_ret

            # Align all series
            common = strat_ret.index.intersection(bah_ret.index).intersection(vol_bah_ret.index)
            strat_r = strat_ret.loc[common].values
            bah_r = bah_ret.loc[common].values
            vol_bah_r = vol_bah_ret.loc[common].values

            # OLS regression: strategy = α + β × benchmark + ε
            # vs buy-and-hold
            X_bah = np.column_stack([np.ones(len(bah_r)), bah_r])
            beta_bah = np.linalg.lstsq(X_bah, strat_r, rcond=None)[0]
            resid_bah = strat_r - X_bah @ beta_bah
            alpha_ann_bah = beta_bah[0] * 252
            te_bah = resid_bah.std() * np.sqrt(252)
            ir_bah = alpha_ann_bah / te_bah if te_bah > 1e-10 else 0

            # vs vol-targeted buy-and-hold
            X_vol = np.column_stack([np.ones(len(vol_bah_r)), vol_bah_r])
            beta_vol = np.linalg.lstsq(X_vol, strat_r, rcond=None)[0]
            resid_vol = strat_r - X_vol @ beta_vol
            alpha_ann_vol = beta_vol[0] * 252
            te_vol = resid_vol.std() * np.sqrt(252)
            ir_vol = alpha_ann_vol / te_vol if te_vol > 1e-10 else 0

            # Sharpe comparisons
            sr_strat = sharpe_ratio(pd.Series(strat_r))
            sr_bah = sharpe_ratio(pd.Series(bah_r))
            sr_vol_bah = sharpe_ratio(pd.Series(vol_bah_r))

            print(f"\n    {split_label}:")
            print(f"    {'':20s} {'Sharpe':>8s} {'Ann.Ret':>9s} {'Ann.Vol':>9s}")
            ann_r = lambda r: (1 + pd.Series(r)).prod() ** (252/len(r)) - 1 if len(r) > 0 else 0
            ann_v = lambda r: pd.Series(r).std() * np.sqrt(252)
            print(f"    {'Strategy (TF)':20s} {sr_strat:>+7.3f} {ann_r(strat_r):>+8.2%} {ann_v(strat_r):>8.2%}")
            print(f"    {'Buy-and-Hold':20s} {sr_bah:>+7.3f} {ann_r(bah_r):>+8.2%} {ann_v(bah_r):>8.2%}")
            print(f"    {'Vol-Targeted B&H':20s} {sr_vol_bah:>+7.3f} {ann_r(vol_bah_r):>+8.2%} {ann_v(vol_bah_r):>8.2%}")

            print(f"\n    vs B&H:        α={alpha_ann_bah:>+.2%}  β={beta_bah[1]:>.3f}  TE={te_bah:.2%}  IR={ir_bah:>+.3f}")
            print(f"    vs Vol-Tgt B&H: α={alpha_ann_vol:>+.2%}  β={beta_vol[1]:>.3f}  TE={te_vol:.2%}  IR={ir_vol:>+.3f}")

    # ═════════════════════════════════════════════════════════════════════
    # PART 4: VOL SIGNAL VALIDATION ON VAL
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  PART 4: VOL Signal Validation (IS vs VAL)")
    print("═" * 70)

    configs = [
        ("5 Original", _make_ss(ALL_SUBSTRATS)),
        ("5+3 VOL",    _make_ss(ALL_SUBSTRATS + VOL_SUBSTRATS)),
    ]

    print(f"\n  {'Config':<14s} │ {'Split':>5s} │ {'Sharpe':>7s} {'Ann.Ret':>8s} {'MaxDD':>7s} {'AvgMo':>7s} "
          f"{'VaR5%':>7s} {'CVaR5%':>7s} {'GrossExp':>9s}")
    print("  " + "─" * 85)

    for label, strats in configs:
        for split_name, split_u in [("IS", is_universe), ("VAL", val_universe)]:
            r = build_portfolio(split_u, strats, **PORTFOLIO_PARAMS)
            rep = full_report(r.net_returns, costs=r.costs)
            tail = tail_risk_stats(r.net_returns)
            print(f"  {label:<14s} │ {split_name:>5s} │ {rep['sharpe_ratio']:>+6.3f} {rep['annualised_return']:>+7.2%} "
                  f"{rep['max_drawdown']:>+6.2%} {rep['avg_monthly_return']:>+6.2%} "
                  f"{tail['var_5pct']:>+6.2%} {tail['cvar_5pct']:>+6.2%} "
                  f"{r.gross_exposure.mean():>8.1%}")

    # Correlation on VAL
    print("\n  VOL signal correlations with TF/SMR (VALIDATION data):")
    val_strat_returns = {}
    for name, ticker, signal_obj, cost_model in ALL_SUBSTRATS + VOL_SUBSTRATS:
        data = val_universe[ticker]
        signal = signal_obj.generate(data)
        bt = run_backtest(prices=data["close"], signal=signal, cost_model=cost_model)
        val_strat_returns[name] = bt.net_returns

    val_ret_df = pd.DataFrame(val_strat_returns).dropna()
    val_corr = val_ret_df.corr()

    vol_names = [n for n, _, _, _ in VOL_SUBSTRATS]
    other_names = [n for n, _, _, _ in ALL_SUBSTRATS]
    print(f"  {'':12s}", end="")
    for vn in vol_names:
        print(f" {vn:>10s}", end="")
    print()
    for on in other_names:
        print(f"  {on:<12s}", end="")
        for vn in vol_names:
            v = val_corr.loc[on, vn] if (on in val_corr.index and vn in val_corr.columns) else float("nan")
            print(f" {v:>+9.3f}", end="")
        print()

    # ═════════════════════════════════════════════════════════════════════
    # PART 5: WALK-FORWARD, PARAM STABILITY, COST STRESS
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  PART 5: Walk-Forward, Parameter Stability, Cost Stress")
    print("═" * 70)

    # 5a. Parameter stability: TF lookback on VAL
    print("\n  [5a] TF Lookback Stability: IS vs VAL")
    lookbacks = [20, 40, 60, 80, 100, 120, 150, 180]
    print(f"  {'LB':>5s} {'IS Sharpe':>10s} {'VAL Sharpe':>11s} {'Decay':>8s} {'IS AvgMo':>9s} {'VAL AvgMo':>10s}")
    print("  " + "-" * 55)

    param_is_sharpes, param_val_sharpes = [], []

    for lb in lookbacks:
        strats = [
            SubStrategy("BTC_TF", "BTC-USD", TrendFollowingSignal(lookback=lb, scale=2.0), CRYPTO_COSTS, weight=1.0),
            SubStrategy("ETH_TF", "ETH-USD", TrendFollowingSignal(lookback=lb, scale=2.0), CRYPTO_COSTS, weight=1.0),
            SubStrategy("QQQ_SMR", "QQQ", SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=10), EQUITY_COSTS, weight=1.0),
            SubStrategy("SPY_SMR", "SPY", SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=10), EQUITY_COSTS, weight=1.0),
            SubStrategy("GC_SMR", "GC=F", SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=5), FUTURES_COSTS, weight=1.0),
        ]
        r_is = build_portfolio(is_universe, strats, **PORTFOLIO_PARAMS)
        r_val = build_portfolio(val_universe, strats, **PORTFOLIO_PARAMS)
        rep_is = full_report(r_is.net_returns)
        rep_val = full_report(r_val.net_returns)

        decay = (1 - rep_val["sharpe_ratio"] / rep_is["sharpe_ratio"]) * 100 if abs(rep_is["sharpe_ratio"]) > 0.01 else float("nan")
        param_is_sharpes.append(rep_is["sharpe_ratio"])
        param_val_sharpes.append(rep_val["sharpe_ratio"])

        print(f"  {lb:>5d} {rep_is['sharpe_ratio']:>+9.3f} {rep_val['sharpe_ratio']:>+10.3f} "
              f"{decay:>+7.0f}% {rep_is['avg_monthly_return']:>+8.2%} {rep_val['avg_monthly_return']:>+9.2%}")

    # 5b. Cost stress tests
    print("\n  [5b] Transaction Cost Stress Test")
    cost_multipliers = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    print(f"  {'Cost ×':>7s} {'IS Sharpe':>10s} {'VAL Sharpe':>11s} {'IS AvgMo':>9s} {'VAL AvgMo':>10s} {'IS TotCost':>11s}")
    print("  " + "-" * 60)

    for mult in cost_multipliers:
        strats = [
            SubStrategy("BTC_TF60", "BTC-USD", TrendFollowingSignal(lookback=60, scale=2.0),
                        scale_costs(CRYPTO_COSTS, mult), weight=1.0),
            SubStrategy("ETH_TF120", "ETH-USD", TrendFollowingSignal(lookback=120, scale=2.0),
                        scale_costs(CRYPTO_COSTS, mult), weight=1.0),
            SubStrategy("QQQ_SMR10", "QQQ", SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=10),
                        scale_costs(EQUITY_COSTS, mult), weight=1.0),
            SubStrategy("SPY_SMR10", "SPY", SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=10),
                        scale_costs(EQUITY_COSTS, mult), weight=1.0),
            SubStrategy("GC_SMR5", "GC=F", SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=5),
                        scale_costs(FUTURES_COSTS, mult), weight=1.0),
        ]

        r_is = build_portfolio(is_universe, strats, **PORTFOLIO_PARAMS)
        r_val = build_portfolio(val_universe, strats, **PORTFOLIO_PARAMS)
        rep_is = full_report(r_is.net_returns, costs=r_is.costs)
        rep_val = full_report(r_val.net_returns, costs=r_val.costs)

        print(f"  {mult:>6.1f}× {rep_is['sharpe_ratio']:>+9.3f} {rep_val['sharpe_ratio']:>+10.3f} "
              f"{rep_is['avg_monthly_return']:>+8.2%} {rep_val['avg_monthly_return']:>+9.2%} "
              f"{rep_is.get('total_costs', 0):>10.4f}")

    # 5c. Walk-forward (rolling OOS)
    print("\n  [5c] Walk-Forward: expanding window, 6-month test steps")

    all_tickers = {ss.ticker for _, ss_ticker, _, _ in ALL_SUBSTRATS for ss in [type('', (), {'ticker': ss_ticker})()]}
    all_tickers = {s[1] for s in ALL_SUBSTRATS}

    # Find common dates
    common_dates = None
    for t in all_tickers:
        if t in universe:
            idx = universe[t].index
            common_dates = idx if common_dates is None else common_dates.intersection(idx)
    common_dates = common_dates.sort_values()

    train_days = 3 * 252  # 3 years
    step_days = 6 * 21    # 6 months
    cursor = train_days

    wf_results = []
    all_wf_returns = []

    while cursor + step_days <= len(common_dates):
        test_dates = common_dates[cursor:cursor + step_days]
        all_dates = common_dates[:cursor + step_days]

        wf_u = {}
        for t in all_tickers:
            if t in universe:
                mask = universe[t].index.isin(all_dates)
                wf_u[t] = universe[t][mask]

        try:
            r = build_portfolio(wf_u, _make_ss(ALL_SUBSTRATS), **PORTFOLIO_PARAMS)
            test_mask = r.net_returns.index.isin(test_dates)
            test_ret = r.net_returns[test_mask]
            if len(test_ret) > 0:
                all_wf_returns.append(test_ret)
                tr = (1 + test_ret).prod() - 1
                sr = sharpe_ratio(test_ret)
                wf_results.append({
                    "start": test_dates[0].strftime("%Y-%m"),
                    "end": test_dates[-1].strftime("%Y-%m"),
                    "n_days": len(test_ret),
                    "return": tr,
                    "sharpe": sr,
                })
        except Exception as e:
            pass

        cursor += step_days

    if wf_results:
        print(f"\n  {'Period':>16s} {'Days':>5s} {'Return':>8s} {'Sharpe':>8s}")
        print("  " + "-" * 40)
        for w in wf_results:
            print(f"  {w['start']} → {w['end']} {w['n_days']:>5d} {w['return']:>+7.2%} {w['sharpe']:>+7.3f}")

        if all_wf_returns:
            chained = pd.concat(all_wf_returns)
            chained = chained[~chained.index.duplicated(keep='first')].sort_index()
            wf_rep = full_report(chained)
            print(f"\n  Walk-Forward Aggregate: Sharpe={wf_rep['sharpe_ratio']:.3f}, "
                  f"Ann.Ret={wf_rep['annualised_return']:.2%}, MaxDD={wf_rep['max_drawdown']:.2%}")

    # ═════════════════════════════════════════════════════════════════════
    # CHARTS
    # ═════════════════════════════════════════════════════════════════════
    print("\n\nGenerating charts...")

    # Chart 1: IS vs VAL IC comparison
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(signal_names))
    w = 0.35
    is_ics = [r["ic_mean"] for r in is_hac_results]
    val_ics = [r["ic_mean"] for r in val_hac_results]
    ax.bar(x - w/2, is_ics, w, label="In-Sample", color="steelblue", alpha=0.8)
    ax.bar(x + w/2, val_ics, w, label="Validation", color="coral", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(signal_names, rotation=45, ha="right")
    ax.set_title("IC (HAC-adjusted) — IS vs Validation", fontsize=14, fontweight="bold")
    ax.set_ylabel("Daily IC Mean")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHART_DIR / "01_ic_is_vs_val.png", dpi=150)
    plt.close(fig)
    print("  ✓ 01_ic_is_vs_val.png")

    # Chart 2: Parameter stability
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(lookbacks, param_is_sharpes, "b-o", label="In-Sample", linewidth=1.5)
    ax.plot(lookbacks, param_val_sharpes, "r-o", label="Validation", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("TF Lookback Stability: IS vs VAL", fontsize=14, fontweight="bold")
    ax.set_xlabel("Lookback (days)")
    ax.set_ylabel("Sharpe Ratio")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHART_DIR / "02_param_stability.png", dpi=150)
    plt.close(fig)
    print("  ✓ 02_param_stability.png")

    # Chart 3: Walk-forward equity curve
    if all_wf_returns:
        chained = pd.concat(all_wf_returns)
        chained = chained[~chained.index.duplicated(keep='first')].sort_index()
        fig, ax = plt.subplots(figsize=(14, 7))
        eq = (1 + chained).cumprod()
        ax.plot(eq.index, eq.values, color="purple", linewidth=1.5)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_title("Walk-Forward Equity Curve (3yr train, 6mo test)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Equity (multiple)")
        fig.tight_layout()
        fig.savefig(CHART_DIR / "03_walk_forward.png", dpi=150)
        plt.close(fig)
        print("  ✓ 03_walk_forward.png")

    # Chart 4: Cost stress
    fig, ax = plt.subplots(figsize=(10, 5))
    # Recompute for chart
    is_sharpes_cost, val_sharpes_cost = [], []
    for mult in cost_multipliers:
        strats = [
            SubStrategy("BTC_TF60", "BTC-USD", TrendFollowingSignal(lookback=60, scale=2.0),
                        scale_costs(CRYPTO_COSTS, mult), weight=1.0),
            SubStrategy("ETH_TF120", "ETH-USD", TrendFollowingSignal(lookback=120, scale=2.0),
                        scale_costs(CRYPTO_COSTS, mult), weight=1.0),
            SubStrategy("QQQ_SMR10", "QQQ", SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=10),
                        scale_costs(EQUITY_COSTS, mult), weight=1.0),
            SubStrategy("SPY_SMR10", "SPY", SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=10),
                        scale_costs(EQUITY_COSTS, mult), weight=1.0),
            SubStrategy("GC_SMR5", "GC=F", SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=5),
                        scale_costs(FUTURES_COSTS, mult), weight=1.0),
        ]
        r_is = build_portfolio(is_universe, strats, **PORTFOLIO_PARAMS)
        r_val = build_portfolio(val_universe, strats, **PORTFOLIO_PARAMS)
        is_sharpes_cost.append(full_report(r_is.net_returns)["sharpe_ratio"])
        val_sharpes_cost.append(full_report(r_val.net_returns)["sharpe_ratio"])

    ax.plot(cost_multipliers, is_sharpes_cost, "b-o", label="In-Sample", linewidth=1.5)
    ax.plot(cost_multipliers, val_sharpes_cost, "r-o", label="Validation", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Cost Stress Test: Sharpe vs Cost Multiplier", fontsize=14, fontweight="bold")
    ax.set_xlabel("Cost Multiplier (×)")
    ax.set_ylabel("Sharpe Ratio")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHART_DIR / "04_cost_stress.png", dpi=150)
    plt.close(fig)
    print("  ✓ 04_cost_stress.png")

    print(f"\n  Charts saved to: {CHART_DIR}/")
    print("=" * 70)
    print("M14 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
