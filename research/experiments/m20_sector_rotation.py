"""
Milestone 20 — Sector Rotation Momentum
==========================================

Test cross-sectional sector momentum on 9 S&P sector ETFs.
Long top 3 momentum sectors, Short bottom 3.

Parts:
  A. Download & validate sector data
  B. Standalone sector momentum backtest (all sectors)
  C. Parameter sensitivity (lookback, n_long/n_short)
  D. Portfolio integration: Sector Rotation alone
  E. Comparison: Sector Rotation vs buy-and-hold SPY
  F. IS / VAL / Walk-Forward validation
  G. Acceptance criteria → PASS / FAIL → decide whether to keep

Usage:
    python research/experiments/m20_sector_rotation.py
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
from quant_research.backtest.engine import run_backtest
from quant_research.data.loader import load_config, load_universe, split_data
from quant_research.metrics.performance import full_report, sharpe_ratio
from quant_research.portfolio.allocator import SubStrategy, build_portfolio
from quant_research.strategies.sector_rotation import SectorRotationSignal

CHART_DIR = PROJECT_ROOT / "research" / "experiments" / "m20_charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

SECTOR_TICKERS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB"]

# Borrow cost for short positions: ~1%/year for liquid sector ETFs
SECTOR_COSTS = scale_costs(EQUITY_COSTS, 1.0)  # 10 bps one-way


def load_sector_data():
    """Load sector ETF data from raw CSVs."""
    from quant_research.data.loader import load_raw, clean

    raw_dir = PROJECT_ROOT / "data" / "raw"
    sector_data = {}
    for ticker in SECTOR_TICKERS:
        try:
            df = load_raw(ticker, raw_dir=raw_dir)
            df = clean(df, ticker=ticker, asset_class="equity")
            sector_data[ticker] = df
        except Exception as e:
            print(f"  ⚠ {ticker}: {e}")
    return sector_data


def split_sector_data(sector_data, config):
    """Split sector data into IS / VAL."""
    is_data = {}
    val_data = {}
    for ticker, df in sector_data.items():
        splits = split_data(df, config)
        is_data[ticker] = splits["in_sample"]
        if "validation" in splits:
            val_data[ticker] = splits["validation"]
    return is_data, val_data


def main():
    print("=" * 80)
    print("  M20 — Sector Rotation Momentum")
    print("=" * 80)

    config = load_config()

    # ==================================================================
    # PART A: LOAD & VALIDATE SECTOR DATA
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Part A: Load & Validate Sector Data")
    print("═" * 80)

    sector_data = load_sector_data()
    print(f"\n  Loaded {len(sector_data)} sectors:")
    for ticker, df in sorted(sector_data.items()):
        print(f"    {ticker}: {df.index[0].date()} → {df.index[-1].date()} ({len(df)} bars)")

    if len(sector_data) < 9:
        print(f"\n  ✗ FAIL: Only {len(sector_data)}/9 sectors loaded. Aborting.")
        return

    is_data, val_data = split_sector_data(sector_data, config)
    print(f"\n  IS period:  {list(is_data.values())[0].index[0].date()} → {list(is_data.values())[0].index[-1].date()}")
    print(f"  VAL period: {list(val_data.values())[0].index[0].date()} → {list(val_data.values())[0].index[-1].date()}")

    # ==================================================================
    # PART B: STANDALONE SECTOR MOMENTUM
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Part B: Standalone Sector Momentum Signals")
    print("═" * 80)

    # Generate signals for all sectors
    sig = SectorRotationSignal(is_data, target_sector="XLK",
                                momentum_window=252, skip_recent=21,
                                n_long=3, n_short=3)
    all_signals_is = sig.generate_all()

    print(f"\n  Signal matrix shape: {all_signals_is.shape}")
    print(f"  Signal distribution (IS, last 252 bars):")
    last_year = all_signals_is.iloc[-252:]
    for val, label in [(1.0, "Long"), (0.0, "Neutral"), (-1.0, "Short")]:
        pct = (last_year == val).sum().mean() / len(last_year) * 100
        print(f"    {label:>8s}: {pct:.1f}%")

    # Per-sector signal summary
    print(f"\n  {'Sector':<6s} │ {'Long %':>7s} {'Short %':>8s} {'Neutral %':>10s}")
    print("  " + "─" * 35)
    for sec in SECTOR_TICKERS:
        if sec in all_signals_is.columns:
            s = all_signals_is[sec]
            n = len(s[s != 0])  # non-warmup
            total = len(s)
            long_pct = (s == 1).sum() / total * 100
            short_pct = (s == -1).sum() / total * 100
            neutral_pct = (s == 0).sum() / total * 100
            print(f"  {sec:<6s} │ {long_pct:>6.1f}% {short_pct:>7.1f}% {neutral_pct:>9.1f}%")

    # ==================================================================
    # PART C: PARAMETER SENSITIVITY
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Part C: Parameter Sensitivity")
    print("═" * 80)

    # Test different lookback windows
    lookbacks = [63, 126, 189, 252]
    skips = [0, 21]
    configs = [(3, 3), (2, 2), (4, 4)]

    print(f"\n  {'Lookback':>8s} {'Skip':>5s} {'L/S':>5s} │ {'IS SR':>7s} {'VAL SR':>8s} │ {'IS Ret':>8s} │ {'Turnover':>9s}")
    print("  " + "─" * 60)

    best_is_sr = -999
    best_config = None

    for lookback in lookbacks:
        for skip in skips:
            for n_long, n_short in configs:
                # Build strategies for each sector
                strats = []
                for sec in SECTOR_TICKERS:
                    sig_obj = SectorRotationSignal(
                        sector_data=is_data, target_sector=sec,
                        momentum_window=lookback, skip_recent=skip,
                        n_long=n_long, n_short=n_short,
                    )
                    strats.append(SubStrategy(
                        name=f"{sec}_SR{lookback}",
                        ticker=sec,
                        signal_obj=sig_obj,
                        cost_model=SECTOR_COSTS,
                        weight=1.0,
                    ))

                try:
                    r_is = build_portfolio(is_data, strats,
                                            vol_target=0.10, vol_lookback=63,
                                            max_gross_exposure=1.0,
                                            dd_threshold=-0.05, concentration_limit=0.20)
                    rep_is = full_report(r_is.net_returns, costs=r_is.costs)
                    sr_is = rep_is["sharpe_ratio"]
                    ann_ret = rep_is["annualised_return"]
                    turnover = r_is.asset_positions.diff().abs().sum().sum() / (len(r_is.net_returns) / 252)

                    # VAL
                    strats_val = []
                    for sec in SECTOR_TICKERS:
                        sig_obj_val = SectorRotationSignal(
                            sector_data=val_data, target_sector=sec,
                            momentum_window=lookback, skip_recent=skip,
                            n_long=n_long, n_short=n_short,
                        )
                        strats_val.append(SubStrategy(
                            name=f"{sec}_SR{lookback}", ticker=sec,
                            signal_obj=sig_obj_val, cost_model=SECTOR_COSTS, weight=1.0,
                        ))
                    r_val = build_portfolio(val_data, strats_val,
                                             vol_target=0.10, vol_lookback=63,
                                             max_gross_exposure=1.0,
                                             dd_threshold=-0.05, concentration_limit=0.20)
                    sr_val = sharpe_ratio(r_val.net_returns)

                    label = f"{lookback:>8d} {skip:>5d} {n_long}/{n_short:>3d}"
                    print(f"  {label} │ {sr_is:>+6.3f} {sr_val:>+7.3f} │ {ann_ret:>+7.2%} │ {turnover:>8.1f}")

                    if sr_is > best_is_sr:
                        best_is_sr = sr_is
                        best_config = (lookback, skip, n_long, n_short)

                except Exception as e:
                    print(f"  {lookback:>8d} {skip:>5d} {n_long}/{n_short:>3d} │ ERROR: {e}")

    print(f"\n  Best IS config: lookback={best_config[0]}, skip={best_config[1]}, "
          f"L/S={best_config[2]}/{best_config[3]}")

    # ==================================================================
    # PART D: BEST CONFIG — DETAILED ANALYSIS
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Part D: Best Config — Detailed Analysis")
    print("═" * 80)

    lb, sk, nl, ns = best_config

    # IS
    strats_best_is = []
    for sec in SECTOR_TICKERS:
        sig_obj = SectorRotationSignal(
            sector_data=is_data, target_sector=sec,
            momentum_window=lb, skip_recent=sk, n_long=nl, n_short=ns,
        )
        strats_best_is.append(SubStrategy(
            f"{sec}_SR", sec, sig_obj, SECTOR_COSTS, weight=1.0,
        ))

    r_is = build_portfolio(is_data, strats_best_is,
                            vol_target=0.10, vol_lookback=63,
                            max_gross_exposure=1.0,
                            dd_threshold=-0.05, concentration_limit=0.20)
    rep_is = full_report(r_is.net_returns, costs=r_is.costs)

    # VAL
    strats_best_val = []
    for sec in SECTOR_TICKERS:
        sig_obj = SectorRotationSignal(
            sector_data=val_data, target_sector=sec,
            momentum_window=lb, skip_recent=sk, n_long=nl, n_short=ns,
        )
        strats_best_val.append(SubStrategy(
            f"{sec}_SR", sec, sig_obj, SECTOR_COSTS, weight=1.0,
        ))

    r_val = build_portfolio(val_data, strats_best_val,
                             vol_target=0.10, vol_lookback=63,
                             max_gross_exposure=1.0,
                             dd_threshold=-0.05, concentration_limit=0.20)
    rep_val = full_report(r_val.net_returns, costs=r_val.costs)

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
    # PART E: VS BUY-AND-HOLD SPY
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Part E: vs Buy-and-Hold SPY")
    print("═" * 80)

    # Load SPY for benchmark
    from quant_research.data.loader import load_raw, clean
    spy_raw = load_raw("SPY", raw_dir=PROJECT_ROOT / "data" / "raw")
    spy = clean(spy_raw, ticker="SPY", asset_class="equity")
    spy_splits = split_data(spy, config)
    spy_is = spy_splits["in_sample"]
    spy_val = spy_splits["validation"] if "validation" in spy_splits else spy_splits["in_sample"]

    spy_ret_is = spy_is["close"].pct_change().fillna(0)
    spy_ret_val = spy_val["close"].pct_change().fillna(0)
    spy_sr_is = sharpe_ratio(spy_ret_is)
    spy_sr_val = sharpe_ratio(spy_ret_val)

    print(f"\n  {'Strategy':<25s} │ {'IS SR':>7s} {'VAL SR':>8s} │ {'IS Ret':>8s} {'VAL Ret':>9s}")
    print("  " + "─" * 60)
    print(f"  {'Sector Rotation':<25s} │ {rep_is['sharpe_ratio']:>+6.3f} {rep_val['sharpe_ratio']:>+7.3f} │ "
          f"{rep_is['annualised_return']:>+7.2%} {rep_val['annualised_return']:>+8.2%}")
    spy_ann_is = (1 + spy_ret_is).prod() ** (252/len(spy_ret_is)) - 1
    spy_ann_val = (1 + spy_ret_val).prod() ** (252/len(spy_ret_val)) - 1
    print(f"  {'Buy-and-Hold SPY':<25s} │ {spy_sr_is:>+6.3f} {spy_sr_val:>+7.3f} │ "
          f"{spy_ann_is:>+7.2%} {spy_ann_val:>+8.2%}")

    # ==================================================================
    # PART F: WALK-FORWARD VALIDATION
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Part F: Walk-Forward Validation")
    print("═" * 80)

    # Combine all data for walk-forward
    full_data = {}
    for sec in SECTOR_TICKERS:
        full_data[sec] = sector_data[sec]

    all_dates = list(full_data.values())[0].index
    n = len(all_dates)
    train_window = 504  # 2 years
    test_window = 63    # 3 months
    n_folds = min(19, (n - train_window) // test_window)

    wf_sharpes = []
    print(f"\n  Folds: {n_folds} (train={train_window}d, test={test_window}d)")
    print(f"  {'Fold':>5s} {'Train':>22s} {'Test':>22s} │ {'SR':>7s}")
    print("  " + "─" * 60)

    for fold in range(n_folds):
        train_start = fold * test_window
        train_end = train_start + train_window
        test_end = min(train_end + test_window, n)

        if test_end > n:
            break

        train_dates = all_dates[train_start:train_end]
        test_dates = all_dates[train_end:test_end]

        if len(test_dates) < 21:
            break

        # Slice data
        train_data = {t: df.loc[train_dates[0]:train_dates[-1]] for t, df in full_data.items()}
        test_data = {t: df.loc[test_dates[0]:test_dates[-1]] for t, df in full_data.items()}

        # Build strategies using TRAIN data for signal computation context
        strats_test = []
        for sec in SECTOR_TICKERS:
            # Use test_data for signal generation (signals look back within their own data)
            sig_obj = SectorRotationSignal(
                sector_data=test_data, target_sector=sec,
                momentum_window=lb, skip_recent=sk, n_long=nl, n_short=ns,
            )
            strats_test.append(SubStrategy(
                f"{sec}_SR", sec, sig_obj, SECTOR_COSTS, weight=1.0,
            ))

        try:
            r = build_portfolio(test_data, strats_test,
                                 vol_target=0.10, vol_lookback=63,
                                 max_gross_exposure=1.0,
                                 dd_threshold=-0.05, concentration_limit=0.20)
            sr = sharpe_ratio(r.net_returns)
            wf_sharpes.append(sr)
            print(f"  {fold+1:>5d} {str(train_dates[0].date()):>11s}→{str(train_dates[-1].date()):>10s} "
                  f"{str(test_dates[0].date()):>11s}→{str(test_dates[-1].date()):>10s} │ {sr:>+6.3f}")
        except Exception as e:
            print(f"  {fold+1:>5d} ERROR: {e}")

    if wf_sharpes:
        pos_folds = sum(1 for s in wf_sharpes if s > 0)
        print(f"\n  Walk-Forward Results:")
        print(f"    Positive folds: {pos_folds}/{len(wf_sharpes)} ({pos_folds/len(wf_sharpes)*100:.0f}%)")
        print(f"    Mean Sharpe:    {np.mean(wf_sharpes):+.3f}")
        print(f"    Median Sharpe:  {np.median(wf_sharpes):+.3f}")

    # ==================================================================
    # PART G: ACCEPTANCE CRITERIA
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Part G: Acceptance Criteria")
    print("═" * 80)

    criteria = [
        ("IS Sharpe > 0.3", rep_is["sharpe_ratio"], 0.3),
        ("VAL Sharpe > 0", rep_val["sharpe_ratio"], 0.0),
        ("WF >50% positive folds", pos_folds/len(wf_sharpes) if wf_sharpes else 0, 0.5),
        ("Survives 1.5× costs", None, None),  # computed below
    ]

    # 1.5× costs test
    strats_15 = []
    for sec in SECTOR_TICKERS:
        sig_obj = SectorRotationSignal(
            sector_data=is_data, target_sector=sec,
            momentum_window=lb, skip_recent=sk, n_long=nl, n_short=ns,
        )
        strats_15.append(SubStrategy(
            f"{sec}_SR", sec, sig_obj, scale_costs(SECTOR_COSTS, 1.5), weight=1.0,
        ))
    r_15 = build_portfolio(is_data, strats_15,
                            vol_target=0.10, vol_lookback=63,
                            max_gross_exposure=1.0,
                            dd_threshold=-0.05, concentration_limit=0.20)
    sr_15 = sharpe_ratio(r_15.net_returns)
    criteria[3] = ("Survives 1.5× costs (IS SR > 0)", sr_15, 0.0)

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
    print("\n\nGenerating charts...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Equity curve IS vs VAL
    ax = axes[0, 0]
    eq_is = (1 + r_is.net_returns).cumprod()
    eq_val = (1 + r_val.net_returns).cumprod()
    ax.plot(eq_is.index, eq_is.values, color="#2ecc71", linewidth=1.5, label="IS")
    ax.plot(eq_val.index, eq_val.values, color="#e74c3c", linewidth=1.5, label="VAL")
    ax.set_title("Sector Rotation — Equity Curve", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("Growth of $1")

    # 2. Sector signal heatmap (last 60 days IS)
    ax = axes[0, 1]
    sig_all = SectorRotationSignal(
        sector_data=is_data, target_sector="XLK",
        momentum_window=lb, skip_recent=sk, n_long=nl, n_short=ns,
    ).generate_all()
    hm_data = sig_all.iloc[-60:].T
    im = ax.imshow(hm_data.values, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_yticks(range(len(hm_data.index)))
    ax.set_yticklabels(hm_data.index, fontsize=8)
    ax.set_title("Signal Heatmap (Last 60 Days IS)", fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Signal")

    # 3. Walk-forward Sharpe per fold
    ax = axes[1, 0]
    if wf_sharpes:
        colors = ["#2ecc71" if s > 0 else "#e74c3c" for s in wf_sharpes]
        ax.bar(range(1, len(wf_sharpes)+1), wf_sharpes, color=colors)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Fold")
        ax.set_ylabel("Sharpe Ratio")
        ax.set_title("Walk-Forward Sharpe per Fold", fontweight="bold")
        ax.grid(True, alpha=0.3)

    # 4. Parameter sensitivity
    ax = axes[1, 1]
    # Plot lookback sensitivity for best skip/n_long/n_short
    param_srs = []
    for lb_test in [63, 126, 189, 252]:
        strats_t = []
        for sec in SECTOR_TICKERS:
            sig_obj = SectorRotationSignal(
                sector_data=is_data, target_sector=sec,
                momentum_window=lb_test, skip_recent=sk, n_long=nl, n_short=ns,
            )
            strats_t.append(SubStrategy(f"{sec}_SR", sec, sig_obj, SECTOR_COSTS, weight=1.0))
        try:
            r_t = build_portfolio(is_data, strats_t,
                                   vol_target=0.10, vol_lookback=63,
                                   max_gross_exposure=1.0,
                                   dd_threshold=-0.05, concentration_limit=0.20)
            param_srs.append(sharpe_ratio(r_t.net_returns))
        except:
            param_srs.append(float("nan"))
    ax.plot([63, 126, 189, 252], param_srs, "o-", color="#3498db", linewidth=2, markersize=8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Momentum Lookback (days)")
    ax.set_ylabel("IS Sharpe Ratio")
    ax.set_title("Lookback Sensitivity", fontweight="bold")
    ax.grid(True, alpha=0.3)

    fig.suptitle("M20: Sector Rotation Momentum", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "01_sector_rotation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 01_sector_rotation.png")

    print(f"\n  Charts saved to: {CHART_DIR}/")
    print("=" * 80)
    print("  M20 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
