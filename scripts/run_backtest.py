"""
run_backtest.py — Run the full multi-strategy portfolio backtest.

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --split validation
    python scripts/run_backtest.py --split all
    python scripts/run_backtest.py --mode crypto_only
    python scripts/run_backtest.py --mode equity_dm
    python scripts/run_backtest.py --mode combined
    python scripts/run_backtest.py --vol-target 0.15 --dd-threshold -0.10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant_research.backtest.costs import (
    EQUITY_COSTS, CRYPTO_COSTS, FUTURES_COSTS, scale_costs,
)
from quant_research.data.loader import load_config, load_raw, clean, split_data
from quant_research.metrics.performance import full_report, format_report
from quant_research.portfolio.allocator import SubStrategy, build_portfolio
from quant_research.strategies.dual_momentum import DualMomentumSignal
from quant_research.strategies.trend_following import TrendFollowingSignal
from quant_research.strategies.volatility_signal import VolatilityMeanReversionSignal

RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Assets used in the Dual Momentum sleeve
DM_TICKERS = ["SPY", "QQQ", "GC=F", "TLT"]
DM_COST_MAP = {
    "SPY": EQUITY_COSTS,
    "QQQ": EQUITY_COSTS,
    "GC=F": FUTURES_COSTS,
    "TLT": scale_costs(EQUITY_COSTS, 0.5),
}


def load_and_clean(ticker: str, asset_class: str = "equity"):
    """Load and clean a single ticker."""
    df = load_raw(ticker, raw_dir=RAW_DIR)
    return clean(df, ticker=ticker, asset_class=asset_class)


def make_dm_strategies(data: dict, weight: float = 1.0) -> list[SubStrategy]:
    """Create Dual Momentum sub-strategies for available assets."""
    dm_data = {t: data[t] for t in DM_TICKERS if t in data}
    strats = []
    for t in DM_TICKERS:
        if t in dm_data:
            strats.append(SubStrategy(
                f"{t}_DM", t,
                DualMomentumSignal(dm_data, t, lookback=252, n_top=3),
                DM_COST_MAP.get(t, EQUITY_COSTS),
                weight,
            ))
    return strats


def make_crypto_strategies(weight: float = 1.0) -> list[SubStrategy]:
    """Create crypto trend-following + vol mean-reversion strategies."""
    return [
        SubStrategy("BTC_TF60", "BTC-USD",
                     TrendFollowingSignal(lookback=60, scale=2.0),
                     CRYPTO_COSTS, weight=weight),
        SubStrategy("ETH_TF120", "ETH-USD",
                     TrendFollowingSignal(lookback=120, scale=2.0),
                     CRYPTO_COSTS, weight=weight),
        SubStrategy("BTC_VOL", "BTC-USD",
                     VolatilityMeanReversionSignal(short_window=10,
                                                   long_window=60),
                     CRYPTO_COSTS, weight=weight),
        SubStrategy("ETH_VOL", "ETH-USD",
                     VolatilityMeanReversionSignal(short_window=10,
                                                   long_window=60),
                     CRYPTO_COSTS, weight=weight),
    ]


def get_strategies(mode: str, data: dict) -> list[SubStrategy]:
    """Build strategy list based on the requested mode.

    Modes
    -----
    crypto_only : 4 crypto strategies (TF + Vol MR)
    equity_dm   : 4 Dual Momentum strategies (SPY/QQQ/GC/TLT)
    combined    : 30% crypto + 70% equity DM (default, best portfolio)
    """
    if mode == "crypto_only":
        return make_crypto_strategies(weight=1.0)
    elif mode == "equity_dm":
        return make_dm_strategies(data, weight=1.0)
    elif mode == "combined":
        return (make_crypto_strategies(weight=0.30)
                + make_dm_strategies(data, weight=0.70))
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multi-strategy portfolio backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_backtest.py                     # Combined 30/70, IS
  python scripts/run_backtest.py --split all          # All 3 splits
  python scripts/run_backtest.py --mode equity_dm     # Equity DM only
  python scripts/run_backtest.py --mode crypto_only   # Crypto only
  python scripts/run_backtest.py --vol-target 0.15    # Higher vol target
        """,
    )
    parser.add_argument(
        "--split", type=str, default="in_sample",
        choices=["in_sample", "validation", "out_of_sample", "all"],
        help="Data split to run on (default: in_sample)",
    )
    parser.add_argument(
        "--mode", type=str, default="combined",
        choices=["combined", "crypto_only", "equity_dm"],
        help="Portfolio mode (default: combined = 30%% crypto + 70%% equity DM)",
    )
    parser.add_argument("--vol-target", type=float, default=0.10,
                        help="Annualised vol target (default: 0.10)")
    parser.add_argument("--dd-threshold", type=float, default=-0.05,
                        help="Drawdown throttle threshold (default: -0.05)")
    parser.add_argument("--concentration-limit", type=float, default=0.40,
                        help="Per-asset concentration limit (default: 0.40)")
    args = parser.parse_args()

    config = load_config()

    # Load all assets needed
    tickers = {
        "BTC-USD": "crypto", "ETH-USD": "crypto",
        "SPY": "equity", "QQQ": "equity",
        "GC=F": "futures", "TLT": "equity",
    }

    splits_to_run = (
        ["in_sample", "validation", "out_of_sample"]
        if args.split == "all"
        else [args.split]
    )

    for split_name in splits_to_run:
        print(f"\n{'='*64}")
        print(f"  BACKTEST — {split_name.upper()}")
        print(f"  Mode: {args.mode}")
        print(f"  Vol target: {args.vol_target:.0%}  |  "
              f"DD threshold: {args.dd_threshold:.0%}  |  "
              f"Conc limit: {args.concentration_limit:.0%}")
        print(f"{'='*64}")

        # Load and split data
        split_data_dict = {}
        for ticker, ac in tickers.items():
            try:
                df = load_and_clean(ticker, ac)
                splits = split_data(df, config)
                if split_name in splits and len(splits[split_name]) > 0:
                    split_data_dict[ticker] = splits[split_name]
            except FileNotFoundError:
                print(f"  ⚠ {ticker}: data file not found, skipping.")

        sub_strategies = get_strategies(args.mode, split_data_dict)

        print(f"  Strategies: {len(sub_strategies)}")
        for s in sub_strategies:
            print(f"    • {s.name} ({s.ticker})")

        result = build_portfolio(
            split_data_dict,
            sub_strategies,
            vol_target=args.vol_target,
            dd_threshold=args.dd_threshold,
            concentration_limit=args.concentration_limit,
        )

        report = full_report(result.net_returns, costs=result.costs)
        print(format_report(report))

        print(f"\n  Avg Gross Exposure: {result.gross_exposure.mean():.1%}")
        print(f"  Max Gross Exposure: {result.gross_exposure.max():.1%}")
        print(f"  Trading Days:       {len(result.net_returns)}")


if __name__ == "__main__":
    main()
