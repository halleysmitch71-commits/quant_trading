"""
run_experiments.py — Run exposure-optimization experiments.

Implements the 4 experiments from the diagnosis to address the 0.62%
monthly return caused by 81% cash drag:

  Exp 0 (Baseline):  Current config (5 sub-strategies, vol=10%, DD=-5%)
  Exp 1 (Drop SMR):  Remove 3 negative-Sharpe SMR strategies
  Exp 2 (Raise Vol): Raise vol target from 10% to 20%
  Exp 3 (Relax DD):  Relax DD throttle from -5% to -10%
  Exp 4 (Combined):  All three changes together

Usage:
    python scripts/run_experiments.py
    python scripts/run_experiments.py --split validation
    python scripts/run_experiments.py --experiments 0 1 4
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant_research.backtest.costs import EQUITY_COSTS, CRYPTO_COSTS, FUTURES_COSTS
from quant_research.data.loader import load_config, load_universe, split_data
from quant_research.metrics.performance import (
    full_report,
    format_report,
    monthly_returns,
    annualised_return,
    annualised_volatility,
    sharpe_ratio,
    max_drawdown,
)
from quant_research.portfolio.allocator import SubStrategy, build_portfolio


# ── Sub-Strategy Definitions ────────────────────────────────────────────────

def get_all_sub_strategies() -> list[SubStrategy]:
    """All 5 sub-strategies (current baseline)."""
    from quant_research.strategies.mean_reversion_smooth import SmoothedMeanReversionSignal
    from quant_research.strategies.trend_following import TrendFollowingSignal

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


def get_crypto_only_sub_strategies() -> list[SubStrategy]:
    """Only the 2 positive-Sharpe crypto TF strategies."""
    from quant_research.strategies.trend_following import TrendFollowingSignal

    return [
        SubStrategy("BTC_TF60", "BTC-USD",
                     TrendFollowingSignal(lookback=60, scale=2.0),
                     CRYPTO_COSTS, weight=1.0),
        SubStrategy("ETH_TF120", "ETH-USD",
                     TrendFollowingSignal(lookback=120, scale=2.0),
                     CRYPTO_COSTS, weight=1.0),
    ]


# ── Experiment Configuration ────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    description: str
    sub_strategies_fn: str   # "all" or "crypto_only"
    vol_target: float
    dd_threshold: float
    dd_min_scale: float
    concentration_limit: float

    def get_sub_strategies(self) -> list[SubStrategy]:
        if self.sub_strategies_fn == "crypto_only":
            return get_crypto_only_sub_strategies()
        return get_all_sub_strategies()


EXPERIMENTS = {
    0: ExperimentConfig(
        name="Baseline",
        description="Current config: 5 sub-strategies, vol=10%, DD=-5%, conc=40%",
        sub_strategies_fn="all",
        vol_target=0.10,
        dd_threshold=-0.05,
        dd_min_scale=0.25,
        concentration_limit=0.40,
    ),
    1: ExperimentConfig(
        name="Drop SMR",
        description="Remove 3 negative-Sharpe SMR strategies → 2 crypto TF only",
        sub_strategies_fn="crypto_only",
        vol_target=0.10,
        dd_threshold=-0.05,
        dd_min_scale=0.25,
        concentration_limit=0.60,
    ),
    2: ExperimentConfig(
        name="Raise Vol",
        description="Raise vol target from 10% to 20% (keep all 5 strategies)",
        sub_strategies_fn="all",
        vol_target=0.20,
        dd_threshold=-0.05,
        dd_min_scale=0.25,
        concentration_limit=0.40,
    ),
    3: ExperimentConfig(
        name="Relax DD",
        description="Relax DD throttle from -5% to -10% (keep all 5 strategies)",
        sub_strategies_fn="all",
        vol_target=0.10,
        dd_threshold=-0.10,
        dd_min_scale=0.25,
        concentration_limit=0.40,
    ),
    4: ExperimentConfig(
        name="Combined",
        description="Drop SMR + vol=20% + DD=-10% + conc=60%",
        sub_strategies_fn="crypto_only",
        vol_target=0.20,
        dd_threshold=-0.10,
        dd_min_scale=0.25,
        concentration_limit=0.60,
    ),
}


# ── Experiment Runner ────────────────────────────────────────────────────────

@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    exp_id: int
    config: ExperimentConfig
    report: dict
    avg_gross_exposure: float
    max_gross_exposure: float
    median_gross_exposure: float
    n_strategies: int
    elapsed_seconds: float


def run_single_experiment(
    exp_id: int,
    config: ExperimentConfig,
    split_universe: dict,
) -> ExperimentResult:
    """Run a single experiment and return results."""
    t0 = time.time()

    sub_strategies = config.get_sub_strategies()

    result = build_portfolio(
        split_universe,
        sub_strategies,
        vol_target=config.vol_target,
        dd_threshold=config.dd_threshold,
        dd_min_scale=config.dd_min_scale,
        concentration_limit=config.concentration_limit,
    )

    report = full_report(result.net_returns, costs=result.costs)
    elapsed = time.time() - t0

    return ExperimentResult(
        exp_id=exp_id,
        config=config,
        report=report,
        avg_gross_exposure=result.gross_exposure.mean(),
        max_gross_exposure=result.gross_exposure.max(),
        median_gross_exposure=result.gross_exposure.median(),
        n_strategies=len(sub_strategies),
        elapsed_seconds=elapsed,
    )


# ── Comparison Table ─────────────────────────────────────────────────────────

def print_comparison_table(results: list[ExperimentResult], split_name: str) -> None:
    """Print a side-by-side comparison of all experiment results."""

    print(f"\n{'='*100}")
    print(f"  EXPERIMENT COMPARISON — {split_name.upper()}")
    print(f"{'='*100}")

    # Header
    header = f"{'Metric':<30s}"
    for r in results:
        header += f" {'Exp'+str(r.exp_id)+': '+r.config.name:>14s}"
    print(header)
    print("-" * 100)

    # Config rows
    print(f"{'# Sub-Strategies':<30s}", end="")
    for r in results:
        print(f" {r.n_strategies:>14d}", end="")
    print()

    print(f"{'Vol Target':<30s}", end="")
    for r in results:
        print(f" {r.config.vol_target:>13.0%}", end="")
    print()

    print(f"{'DD Threshold':<30s}", end="")
    for r in results:
        print(f" {r.config.dd_threshold:>13.0%}", end="")
    print()

    print(f"{'Concentration Limit':<30s}", end="")
    for r in results:
        print(f" {r.config.concentration_limit:>13.0%}", end="")
    print()

    print("-" * 100)

    # Performance rows
    metrics = [
        ("Total Return",          "total_return",          "{:>13.2%}"),
        ("Annualised Return",     "annualised_return",     "{:>13.2%}"),
        ("Annualised Volatility", "annualised_volatility", "{:>13.2%}"),
        ("Sharpe Ratio",          "sharpe_ratio",          "{:>13.3f}"),
        ("Sortino Ratio",         "sortino_ratio",         "{:>13.3f}"),
        ("Max Drawdown",          "max_drawdown",          "{:>13.2%}"),
        ("Calmar Ratio",          "calmar_ratio",          "{:>13.3f}"),
        ("Win Rate",              "win_rate",              "{:>13.2%}"),
        ("Avg Monthly Return",    "avg_monthly_return",    "{:>13.2%}"),
        ("% Positive Months",     "pct_positive_months",   "{:>13.1%}"),
        ("Best Month",            "best_month",            "{:>13.2%}"),
        ("Worst Month",           "worst_month",           "{:>13.2%}"),
        ("Total Costs",           "total_costs",           "{:>13.4f}"),
    ]

    for label, key, fmt in metrics:
        print(f"{label:<30s}", end="")
        for r in results:
            val = r.report.get(key, 0.0)
            print(f" {fmt.format(val)}", end="")
        print()

    print("-" * 100)

    # Exposure rows
    print(f"{'Avg Gross Exposure':<30s}", end="")
    for r in results:
        print(f" {r.avg_gross_exposure:>13.1%}", end="")
    print()

    print(f"{'Median Gross Exposure':<30s}", end="")
    for r in results:
        print(f" {r.median_gross_exposure:>13.1%}", end="")
    print()

    print(f"{'Max Gross Exposure':<30s}", end="")
    for r in results:
        print(f" {r.max_gross_exposure:>13.1%}", end="")
    print()

    print("-" * 100)

    # Delta vs baseline
    if len(results) > 1:
        baseline = results[0]
        print(f"\n{'Δ vs Baseline':<30s}")
        print(f"{'  Avg Monthly Return':<30s}", end="")
        base_monthly = baseline.report.get("avg_monthly_return", 0)
        for r in results:
            delta = r.report.get("avg_monthly_return", 0) - base_monthly
            sign = "+" if delta >= 0 else ""
            print(f" {sign}{delta:>12.2%}", end="")
        print()

        print(f"{'  Avg Gross Exposure':<30s}", end="")
        for r in results:
            delta = r.avg_gross_exposure - baseline.avg_gross_exposure
            sign = "+" if delta >= 0 else ""
            print(f" {sign}{delta:>12.1%}", end="")
        print()

        print(f"{'  Sharpe Ratio':<30s}", end="")
        base_sharpe = baseline.report.get("sharpe_ratio", 0)
        for r in results:
            delta = r.report.get("sharpe_ratio", 0) - base_sharpe
            sign = "+" if delta >= 0 else ""
            print(f" {sign}{delta:>12.3f}", end="")
        print()

        print(f"{'  Max Drawdown':<30s}", end="")
        base_dd = baseline.report.get("max_drawdown", 0)
        for r in results:
            delta = r.report.get("max_drawdown", 0) - base_dd
            sign = "+" if delta >= 0 else ""
            print(f" {sign}{delta:>12.2%}", end="")
        print()

    print(f"\n{'='*100}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run exposure-optimization experiments"
    )
    parser.add_argument(
        "--split", type=str, default="in_sample",
        choices=["in_sample", "validation", "out_of_sample", "all"],
        help="Data split to run on (default: in_sample)",
    )
    parser.add_argument(
        "--experiments", type=int, nargs="+", default=None,
        help="Which experiments to run (default: all). E.g.: --experiments 0 1 4",
    )
    args = parser.parse_args()

    exp_ids = args.experiments if args.experiments else sorted(EXPERIMENTS.keys())

    # Validate experiment IDs
    for eid in exp_ids:
        if eid not in EXPERIMENTS:
            print(f"ERROR: Unknown experiment ID {eid}. Valid: {sorted(EXPERIMENTS.keys())}")
            sys.exit(1)

    # Load data
    print("Loading data...")
    config = load_config()
    universe = load_universe(config)

    splits_to_run = (
        ["in_sample", "validation", "out_of_sample"]
        if args.split == "all"
        else [args.split]
    )

    for split_name in splits_to_run:
        print(f"\n{'#'*100}")
        print(f"  DATA SPLIT: {split_name.upper()}")
        print(f"{'#'*100}")

        # Prepare split universe
        split_universe = {}
        for ticker, df in universe.items():
            splits = split_data(df, config)
            if split_name in splits and len(splits[split_name]) > 0:
                split_universe[ticker] = splits[split_name]

        if not split_universe:
            print(f"  WARNING: No data for split '{split_name}'. Skipping.")
            continue

        # Run experiments
        results = []
        for eid in exp_ids:
            exp_config = EXPERIMENTS[eid]
            print(f"\n  Running Exp {eid}: {exp_config.name} — {exp_config.description}")

            try:
                result = run_single_experiment(eid, exp_config, split_universe)
                results.append(result)
                print(f"    ✓ Done in {result.elapsed_seconds:.1f}s | "
                      f"Avg GE={result.avg_gross_exposure:.1%} | "
                      f"Monthly={result.report['avg_monthly_return']:.2%} | "
                      f"Sharpe={result.report['sharpe_ratio']:.3f}")
            except Exception as e:
                print(f"    ✗ FAILED: {e}")
                import traceback
                traceback.print_exc()

        # Print comparison
        if results:
            print_comparison_table(results, split_name)

            # Print individual full reports
            for r in results:
                print(f"\n{'─'*60}")
                print(f"  FULL REPORT — Exp {r.exp_id}: {r.config.name}")
                print(f"{'─'*60}")
                print(format_report(r.report))
                print(f"  Avg Gross Exposure:    {r.avg_gross_exposure:.1%}")
                print(f"  Median Gross Exposure: {r.median_gross_exposure:.1%}")
                print(f"  Max Gross Exposure:    {r.max_gross_exposure:.1%}")


if __name__ == "__main__":
    main()
