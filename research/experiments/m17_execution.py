"""
Milestone 17 — Execution Realism
==================================

Test strategy robustness under realistic execution conditions:
  A. Realistic cost model (vol-dependent slippage, funding)
  B. Execution delay sensitivity (0–5 bars)
  C. Partial fill impact (60%–100%)
  D. Cost component decomposition

Usage:
    python research/experiments/m17_execution.py
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
from quant_research.backtest.execution import (
    RealisticCostModel, REALISTIC_CRYPTO, REALISTIC_EQUITY, REALISTIC_FUTURES,
    scale_realistic_costs, simulate_execution_delay, simulate_partial_fills,
)
from quant_research.data.loader import load_config, load_universe, split_data
from quant_research.metrics.performance import full_report, sharpe_ratio
from quant_research.portfolio.allocator import SubStrategy, build_portfolio
from quant_research.strategies.mean_reversion_smooth import SmoothedMeanReversionSignal
from quant_research.strategies.trend_following import TrendFollowingSignal
from quant_research.strategies.volatility_signal import VolatilityMeanReversionSignal

CHART_DIR = PROJECT_ROOT / "research" / "experiments" / "m17_charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

PP = dict(vol_target=0.10, vol_lookback=63, max_gross_exposure=1.0,
          dd_threshold=-0.05, concentration_limit=0.40)


def _strats():
    return [
        SubStrategy("BTC_TF60", "BTC-USD", TrendFollowingSignal(lookback=60, scale=2.0), CRYPTO_COSTS, weight=1.0),
        SubStrategy("ETH_TF120", "ETH-USD", TrendFollowingSignal(lookback=120, scale=2.0), CRYPTO_COSTS, weight=1.0),
        SubStrategy("QQQ_SMR10", "QQQ", SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=10), EQUITY_COSTS, weight=1.0),
        SubStrategy("SPY_SMR10", "SPY", SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=10), EQUITY_COSTS, weight=1.0),
        SubStrategy("GC_SMR5", "GC=F", SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=5), FUTURES_COSTS, weight=1.0),
        SubStrategy("BTC_VOL", "BTC-USD", VolatilityMeanReversionSignal(short_window=10, long_window=60), CRYPTO_COSTS, weight=1.0),
        SubStrategy("ETH_VOL", "ETH-USD", VolatilityMeanReversionSignal(short_window=10, long_window=60), CRYPTO_COSTS, weight=1.0),
        SubStrategy("SPY_VOL", "SPY", VolatilityMeanReversionSignal(short_window=10, long_window=60), EQUITY_COSTS, weight=1.0),
    ]


def _compute_realistic_costs(universe, strats, data_split, cost_map):
    """Run portfolio then re-compute costs using RealisticCostModel."""
    # First, get positions from build_portfolio (with zero costs to isolate execution)
    strats_0 = [SubStrategy(s.name, s.ticker, s.signal_obj, scale_costs(s.cost_model, 0), s.weight) for s in strats]
    r = build_portfolio(data_split, strats_0, **PP)

    # Compute realistic costs per asset
    total_realistic_cost = pd.Series(0.0, index=r.net_returns.index)
    cost_breakdown = {}

    for ticker in r.asset_positions.columns:
        pos = r.asset_positions[ticker]
        turnover = pos.diff().abs().fillna(pos.abs())

        # Get daily vol for this asset
        prices = data_split[ticker]["close"].reindex(r.net_returns.index)
        daily_ret = prices.pct_change().fillna(0.0)
        daily_vol = daily_ret.rolling(21, min_periods=5).std().fillna(daily_ret.std())

        # Select cost model
        rcm = cost_map.get(ticker, REALISTIC_EQUITY)
        costs_df = rcm.compute_costs(turnover, pos, daily_vol)
        total_realistic_cost += costs_df["total"]
        cost_breakdown[ticker] = costs_df

    # Recompute net returns
    gross_returns = r.gross_returns
    net_returns = gross_returns - total_realistic_cost
    return net_returns, total_realistic_cost, cost_breakdown, r


def main():
    print("=" * 80)
    print("  M17 — Execution Realism")
    print("=" * 80)

    config = load_config()
    universe = load_universe(config)
    is_u, val_u = {}, {}
    for ticker, df in universe.items():
        splits = split_data(df, config)
        is_u[ticker] = splits["in_sample"]
        if "validation" in splits:
            val_u[ticker] = splits["validation"]

    strats = _strats()

    cost_map = {
        "BTC-USD": REALISTIC_CRYPTO,
        "ETH-USD": REALISTIC_CRYPTO,
        "SPY": REALISTIC_EQUITY,
        "QQQ": REALISTIC_EQUITY,
        "GC=F": REALISTIC_FUTURES,
    }

    # ==================================================================
    # PART A: REALISTIC VS FLAT COST COMPARISON
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Part A: Realistic Cost Model vs Flat BPS")
    print("═" * 80)

    print("\n  Cost Model Specifications:")
    for name, model in [("CRYPTO", REALISTIC_CRYPTO), ("EQUITY", REALISTIC_EQUITY), ("FUTURES", REALISTIC_FUTURES)]:
        print(f"    {name}:  comm={model.commission_bps}bp  spread={model.spread_bps}bp  "
              f"base_slip={model.base_slippage_bps}bp  vol_coeff={model.vol_slippage_coeff}  "
              f"funding={model.daily_funding_rate:.4f}/day  borrow={model.daily_borrow_rate:.4f}/day")

    print(f"\n  {'Model':<22s} │ {'IS SR':>7s} {'VAL SR':>8s} │ {'IS Ret':>8s} {'IS Vol':>8s} │ "
          f"{'IS Cost':>8s} {'VAL Cost':>9s}")
    print("  " + "─" * 75)

    # Flat-bps baseline
    r_is_flat = build_portfolio(is_u, strats, **PP)
    r_val_flat = build_portfolio(val_u, strats, **PP)
    rep_is_flat = full_report(r_is_flat.net_returns, costs=r_is_flat.costs)
    rep_val_flat = full_report(r_val_flat.net_returns, costs=r_val_flat.costs)
    print(f"  {'Flat BPS (current)':<22s} │ {rep_is_flat['sharpe_ratio']:>+6.3f} {rep_val_flat['sharpe_ratio']:>+7.3f} │ "
          f"{rep_is_flat['annualised_return']:>+7.2%} {rep_is_flat['annualised_volatility']:>7.2%} │ "
          f"{r_is_flat.costs.sum():>7.4f} {r_val_flat.costs.sum():>8.4f}")

    # Realistic costs
    for mult_label, mult in [("Realistic 1×", 1.0), ("Realistic 1.5×", 1.5), ("Realistic 2×", 2.0), ("Realistic 3×", 3.0)]:
        scaled_map = {k: scale_realistic_costs(v, mult) for k, v in cost_map.items()}
        nr_is, tc_is, _, _ = _compute_realistic_costs(is_u, strats, is_u, scaled_map)
        nr_val, tc_val, _, _ = _compute_realistic_costs(val_u, strats, val_u, scaled_map)
        sr_is = sharpe_ratio(nr_is)
        sr_val = sharpe_ratio(nr_val)
        ann_ret = (1 + nr_is).prod() ** (252 / len(nr_is)) - 1
        ann_vol = nr_is.std() * np.sqrt(252)
        print(f"  {mult_label:<22s} │ {sr_is:>+6.3f} {sr_val:>+7.3f} │ "
              f"{ann_ret:>+7.2%} {ann_vol:>7.2%} │ {tc_is.sum():>7.4f} {tc_val.sum():>8.4f}")

    # ==================================================================
    # PART B: COST COMPONENT DECOMPOSITION
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Part B: Cost Component Decomposition (IS, Realistic 1×)")
    print("═" * 80)

    nr_is, tc_is, breakdown_is, r_base = _compute_realistic_costs(is_u, strats, is_u, cost_map)

    agg_breakdown = {}
    for ticker, cb in breakdown_is.items():
        for col in cb.columns:
            if col not in agg_breakdown:
                agg_breakdown[col] = 0.0
            agg_breakdown[col] += cb[col].sum()

    total = agg_breakdown.get("total", 1)
    print(f"\n  {'Component':<18s} {'Total':>10s} {'% of Total':>12s} {'Ann. Drag':>12s}")
    print("  " + "-" * 55)
    n_years_is = len(nr_is) / 252
    for comp in ["commission", "spread", "slippage_fixed", "slippage_vol", "slippage_impact", "funding", "borrow"]:
        v = agg_breakdown.get(comp, 0)
        pct = v / total * 100 if total > 0 else 0
        ann = v / n_years_is if n_years_is > 0 else 0
        print(f"  {comp:<18s} {v:>9.4f} {pct:>11.1f}% {ann:>11.2%}")
    print(f"  {'TOTAL':<18s} {total:>9.4f} {'100.0':>11s}% {total/n_years_is:>11.2%}")

    # Per-asset breakdown
    print(f"\n  Per-Asset Costs (IS, ann.):")
    print(f"  {'Ticker':<10s} {'Comm':>7s} {'Spread':>8s} {'Slip(F)':>8s} {'Slip(V)':>8s} {'Fund':>7s} {'Total':>8s}")
    print("  " + "-" * 55)
    for ticker in sorted(breakdown_is.keys()):
        cb = breakdown_is[ticker]
        for col in cb.columns:
            if col != "total":
                ann = cb[col].sum() / n_years_is
        c = cb.sum() / n_years_is
        print(f"  {ticker:<10s} {c.get('commission',0):>6.2%} {c.get('spread',0):>7.2%} "
              f"{c.get('slippage_fixed',0):>7.2%} {c.get('slippage_vol',0):>7.2%} "
              f"{c.get('funding',0):>6.2%} {c.get('total',0):>7.2%}")

    # ==================================================================
    # PART C: EXECUTION DELAY SENSITIVITY
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Part C: Execution Delay Sensitivity")
    print("═" * 80)
    print("  Note: Engine already applies 1-bar delay. Values below show ADDITIONAL delay.")

    delay_results = []
    print(f"\n  {'Extra Delay':>12s} {'Total Delay':>12s} │ {'IS SR':>7s} {'VAL SR':>8s} │ {'IS Ret':>8s} │ {'Verdict':>10s}")
    print("  " + "─" * 65)

    sig_specs = [
        ("BTC_TF60", "BTC-USD", TrendFollowingSignal(lookback=60, scale=2.0), CRYPTO_COSTS),
        ("ETH_TF120", "ETH-USD", TrendFollowingSignal(lookback=120, scale=2.0), CRYPTO_COSTS),
        ("QQQ_SMR10", "QQQ", SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=10), EQUITY_COSTS),
        ("SPY_SMR10", "SPY", SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=10), EQUITY_COSTS),
        ("GC_SMR5", "GC=F", SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=5), FUTURES_COSTS),
        ("BTC_VOL", "BTC-USD", VolatilityMeanReversionSignal(short_window=10, long_window=60), CRYPTO_COSTS),
        ("ETH_VOL", "ETH-USD", VolatilityMeanReversionSignal(short_window=10, long_window=60), CRYPTO_COSTS),
        ("SPY_VOL", "SPY", VolatilityMeanReversionSignal(short_window=10, long_window=60), EQUITY_COSTS),
    ]

    for extra_delay in [0, 1, 2, 5]:
        total_delay = 1 + extra_delay
        # Create strategies with extra-delayed signals
        delayed_strats = []
        for name, ticker, sig_obj, cost in sig_specs:
            class DelayedSignal:
                def __init__(self, base, delay):
                    self._base = base
                    self._delay = delay
                def generate(self, data):
                    raw = self._base.generate(data)
                    return simulate_execution_delay(raw, self._delay)

            delayed_strats.append(
                SubStrategy(name, ticker, DelayedSignal(sig_obj, extra_delay), cost, weight=1.0)
            )

        try:
            r_is = build_portfolio(is_u, delayed_strats, **PP)
            r_val = build_portfolio(val_u, delayed_strats, **PP)
            sr_is = sharpe_ratio(r_is.net_returns)
            sr_val = sharpe_ratio(r_val.net_returns)
            ann_ret = (1 + r_is.net_returns).prod() ** (252/len(r_is.net_returns)) - 1
            verdict = "✓ PASS" if sr_val > 0 else "✗ FAIL"
        except Exception as e:
            sr_is = sr_val = ann_ret = float("nan")
            verdict = f"ERROR"

        delay_results.append({"extra": extra_delay, "total": total_delay, "is_sr": sr_is, "val_sr": sr_val})
        print(f"  {f'+{extra_delay} bar':>12s} {f'{total_delay} bar':>12s} │ {sr_is:>+6.3f} {sr_val:>+7.3f} │ "
              f"{ann_ret:>+7.2%} │ {verdict:>10s}")

    # ==================================================================
    # PART D: PARTIAL FILL IMPACT
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Part D: Partial Fill Impact")
    print("═" * 80)

    print(f"\n  {'Fill Rate':>10s} │ {'IS SR':>7s} {'VAL SR':>8s} │ {'IS Ret':>8s} │ {'Turnover':>9s}")
    print("  " + "─" * 48)

    for fill_rate in [1.0, 0.9, 0.8, 0.7, 0.6]:
        pf_strats = []
        for name, ticker, sig_obj, cost in sig_specs:
            class PartialFillSignal:
                def __init__(self, base, rate):
                    self._base = base
                    self._rate = rate
                def generate(self, data):
                    raw = self._base.generate(data)
                    return simulate_partial_fills(raw, self._rate)

            pf_strats.append(
                SubStrategy(name, ticker, PartialFillSignal(sig_obj, fill_rate), cost, weight=1.0)
            )

        try:
            r_is = build_portfolio(is_u, pf_strats, **PP)
            r_val = build_portfolio(val_u, pf_strats, **PP)
            sr_is = sharpe_ratio(r_is.net_returns)
            sr_val = sharpe_ratio(r_val.net_returns)
            ann_ret = (1 + r_is.net_returns).prod() ** (252/len(r_is.net_returns)) - 1
            turnover = r_is.asset_positions.diff().abs().sum().sum()
        except Exception as e:
            sr_is = sr_val = ann_ret = turnover = float("nan")

        print(f"  {fill_rate:>9.0%} │ {sr_is:>+6.3f} {sr_val:>+7.3f} │ {ann_ret:>+7.2%} │ {turnover:>8.1f}")

    # ==================================================================
    # PART E: ACCEPTANCE CRITERIA CHECK
    # ==================================================================
    print("\n" + "═" * 80)
    print("  Part E: Acceptance Criteria")
    print("═" * 80)

    criteria = [
        ("Base costs: Sharpe > 0.7", rep_is_flat["sharpe_ratio"], 0.7),
        ("1.5× costs: Sharpe > 0.5", None, 0.5),
        ("2× costs: Sharpe > 0", None, 0.0),
        ("+1 delay: Sharpe > 0", None, 0.0),
    ]

    # Compute missing
    r15 = build_portfolio(is_u, [SubStrategy(s.name, s.ticker, s.signal_obj, scale_costs(s.cost_model, 1.5), s.weight) for s in strats], **PP)
    criteria[1] = ("1.5× costs: IS Sharpe > 0.5", sharpe_ratio(r15.net_returns), 0.5)

    r20 = build_portfolio(is_u, [SubStrategy(s.name, s.ticker, s.signal_obj, scale_costs(s.cost_model, 2.0), s.weight) for s in strats], **PP)
    criteria[2] = ("2× costs: IS Sharpe > 0", sharpe_ratio(r20.net_returns), 0.0)

    delay1_sr = delay_results[1]["is_sr"] if len(delay_results) > 1 else float("nan")
    criteria[3] = ("+1 extra delay: IS Sharpe > 0", delay1_sr, 0.0)

    print(f"\n  {'Criterion':<35s} {'Actual':>8s} {'Target':>8s} {'Result':>8s}")
    print("  " + "-" * 63)

    n_pass = 0
    for desc, actual, target in criteria:
        passed = actual > target if not np.isnan(actual) else False
        n_pass += passed
        print(f"  {desc:<35s} {actual:>+7.3f} {target:>+7.3f} {'✓ PASS' if passed else '✗ FAIL':>8s}")

    print(f"\n  Result: {n_pass}/{len(criteria)} criteria passed")

    # Charts
    print("\n\nGenerating charts...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Delay sensitivity
    ax = axes[0]
    delays = [d["extra"] for d in delay_results]
    is_srs = [d["is_sr"] for d in delay_results]
    val_srs = [d["val_sr"] for d in delay_results]
    ax.plot(delays, is_srs, "o-", color="#2ecc71", label="IS", linewidth=2, markersize=8)
    ax.plot(delays, val_srs, "s--", color="#e74c3c", label="VAL", linewidth=2, markersize=8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Additional Execution Delay (bars)")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Execution Delay Sensitivity", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cost stress
    ax = axes[1]
    mults = [1.0, 1.5, 2.0, 3.0]
    is_cost_sr = []
    for m in mults:
        r_m = build_portfolio(is_u, [SubStrategy(s.name, s.ticker, s.signal_obj, scale_costs(s.cost_model, m), s.weight) for s in strats], **PP)
        is_cost_sr.append(sharpe_ratio(r_m.net_returns))
    ax.plot(mults, is_cost_sr, "o-", color="#3498db", linewidth=2, markersize=8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(0.5, color="orange", linewidth=0.8, linestyle=":", label="Target (0.5)")
    ax.set_xlabel("Cost Multiplier")
    ax.set_ylabel("IS Sharpe Ratio")
    ax.set_title("Cost Stress Sensitivity", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("M17: Execution Realism", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "01_execution_realism.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 01_execution_realism.png")

    print(f"\n  Charts saved to: {CHART_DIR}/")
    print("=" * 80)
    print("  M17 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
