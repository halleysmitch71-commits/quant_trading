"""
Tests for portfolio construction.

Tests verify:
  1. Portfolio combines sub-strategies correctly.
  2. One-bar delay is preserved at portfolio level.
  3. Gross exposure never exceeds the cap.
  4. Vol-scaling adjusts positions appropriately.
  5. Costs are computed per-asset with correct cost models.
  6. Edge cases (single strategy, same-asset strategies).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research.backtest.costs import CostModel, EQUITY_COSTS, CRYPTO_COSTS
from quant_research.portfolio.allocator import (
    SubStrategy,
    PortfolioResult,
    build_portfolio,
)
from quant_research.strategies.mean_reversion_smooth import SmoothedMeanReversionSignal
from quant_research.strategies.trend_following import TrendFollowingSignal


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_universe(n: int = 200, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Create a synthetic universe with 3 assets."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n)

    universe = {}
    for ticker, drift in [("SPY", 0.0003), ("BTC", 0.001), ("GLD", 0.0001)]:
        close = 100 * np.cumprod(1 + rng.normal(drift, 0.015, n))
        universe[ticker] = pd.DataFrame(
            {
                "open": close * (1 + rng.normal(0, 0.002, n)),
                "high": close * (1 + np.abs(rng.normal(0, 0.005, n))),
                "low": close * (1 - np.abs(rng.normal(0, 0.005, n))),
                "close": close,
                "volume": rng.integers(1e6, 1e7, n).astype(float),
            },
            index=dates,
        )
    return universe


def _simple_strategies() -> list[SubStrategy]:
    """Two sub-strategies on different assets."""
    return [
        SubStrategy(
            name="SPY_MR",
            ticker="SPY",
            signal_obj=SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=5),
            cost_model=EQUITY_COSTS,
            weight=1.0,
        ),
        SubStrategy(
            name="BTC_TF",
            ticker="BTC",
            signal_obj=TrendFollowingSignal(lookback=60, scale=2.0),
            cost_model=CRYPTO_COSTS,
            weight=1.0,
        ),
    ]


# ── Basic functionality ─────────────────────────────────────────────────────


class TestPortfolioBasic:
    def test_returns_portfolio_result(self):
        universe = _make_universe()
        result = build_portfolio(universe, _simple_strategies())
        assert isinstance(result, PortfolioResult)

    def test_all_series_same_length(self):
        universe = _make_universe()
        result = build_portfolio(universe, _simple_strategies())
        n = len(result.net_returns)
        assert len(result.gross_returns) == n
        assert len(result.gross_exposure) == n
        assert len(result.costs) == n
        assert len(result.equity) == n

    def test_equity_starts_at_initial_capital(self):
        universe = _make_universe()
        result = build_portfolio(universe, _simple_strategies(), initial_capital=50_000)
        assert result.equity.iloc[0] == pytest.approx(50_000, rel=0.01)

    def test_deterministic(self):
        universe = _make_universe()
        r1 = build_portfolio(universe, _simple_strategies())
        r2 = build_portfolio(universe, _simple_strategies())
        pd.testing.assert_series_equal(r1.equity, r2.equity)

    def test_needs_at_least_one_strategy(self):
        universe = _make_universe()
        with pytest.raises(ValueError, match="at least one"):
            build_portfolio(universe, [])

    def test_missing_ticker_raises(self):
        universe = _make_universe()
        bad_ss = [SubStrategy("bad", "MISSING", SmoothedMeanReversionSignal(),
                              EQUITY_COSTS)]
        with pytest.raises(ValueError, match="not found"):
            build_portfolio(universe, bad_ss)


# ── Gross exposure constraint ────────────────────────────────────────────────


class TestGrossExposure:
    def test_exposure_never_exceeds_cap(self):
        universe = _make_universe(n=300)
        result = build_portfolio(
            universe, _simple_strategies(), max_gross_exposure=1.0,
        )
        assert (result.gross_exposure <= 1.0 + 1e-10).all(), (
            f"Max exposure = {result.gross_exposure.max():.6f} > 1.0"
        )

    def test_lower_cap_reduces_exposure(self):
        universe = _make_universe(n=300)
        r_100 = build_portfolio(universe, _simple_strategies(), max_gross_exposure=1.0)
        r_50 = build_portfolio(universe, _simple_strategies(), max_gross_exposure=0.5)
        assert r_50.gross_exposure.mean() <= r_100.gross_exposure.mean()

    def test_exposure_cap_at_half(self):
        universe = _make_universe(n=300)
        result = build_portfolio(
            universe, _simple_strategies(), max_gross_exposure=0.5,
        )
        assert (result.gross_exposure <= 0.5 + 1e-10).all()


# ── Volatility scaling ──────────────────────────────────────────────────────


class TestVolScaling:
    def test_higher_vol_target_larger_positions(self):
        universe = _make_universe(n=300)
        r_low = build_portfolio(universe, _simple_strategies(), vol_target=0.05)
        r_high = build_portfolio(universe, _simple_strategies(), vol_target=0.20)
        assert r_high.gross_exposure.mean() > r_low.gross_exposure.mean()

    def test_vol_scaling_preserves_delay(self):
        universe = _make_universe(n=200)
        result = build_portfolio(universe, _simple_strategies())
        assert result.asset_positions.iloc[0].abs().sum() == pytest.approx(0.0)


# ── Same-asset aggregation ───────────────────────────────────────────────────


class TestSameAssetAggregation:
    def test_two_strategies_on_same_asset(self):
        universe = _make_universe()
        strategies = [
            SubStrategy("SPY_MR", "SPY", SmoothedMeanReversionSignal(),
                        EQUITY_COSTS, weight=1.0),
            SubStrategy("SPY_TF", "SPY", TrendFollowingSignal(lookback=60),
                        EQUITY_COSTS, weight=1.0),
        ]
        result = build_portfolio(universe, strategies)
        assert isinstance(result, PortfolioResult)
        assert "SPY" in result.asset_positions.columns


# ── Single strategy ──────────────────────────────────────────────────────────


class TestSingleStrategy:
    def test_single_strategy_works(self):
        universe = _make_universe()
        strategies = [
            SubStrategy("SPY_MR", "SPY", SmoothedMeanReversionSignal(),
                        EQUITY_COSTS, weight=1.0),
        ]
        result = build_portfolio(universe, strategies)
        assert isinstance(result, PortfolioResult)
        assert len(result.net_returns) > 0


# ── Costs ────────────────────────────────────────────────────────────────────


class TestPortfolioCosts:
    def test_costs_are_non_negative(self):
        universe = _make_universe()
        result = build_portfolio(universe, _simple_strategies())
        assert (result.costs >= 0).all()

    def test_net_less_than_gross(self):
        universe = _make_universe()
        result = build_portfolio(universe, _simple_strategies())
        assert result.net_returns.sum() <= result.gross_returns.sum() + 1e-10
