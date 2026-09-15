"""
Tests for transaction cost and slippage models.

Tests verify:
  1. CostModel arithmetic (bps → rate conversion, total).
  2. Cost computation on turnover series.
  3. Preset models have correct values.
  4. Asset-class factory function.
  5. Cost breakdown reporting.
  6. Scale-costs utility for robustness analysis.
  7. Integration with the backtest engine.
  8. Cost sensitivity: strategy is worse with higher costs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research.backtest.costs import (
    CRYPTO_COSTS,
    EQUITY_COSTS,
    FUTURES_COSTS,
    CostModel,
    cost_model_for_asset_class,
    cost_model_from_bps,
    scale_costs,
)
from quant_research.backtest.engine import run_backtest


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_prices(values: list[float], start: str = "2020-01-02") -> pd.Series:
    dates = pd.bdate_range(start=start, periods=len(values))
    return pd.Series(values, index=dates, name="close")


def _make_signal(values: list[float], start: str = "2020-01-02") -> pd.Series:
    dates = pd.bdate_range(start=start, periods=len(values))
    return pd.Series(values, index=dates, name="signal")


# ── CostModel unit tests ────────────────────────────────────────────────────


class TestCostModel:
    def test_total_bps(self):
        model = CostModel(commission_bps=3, spread_bps=4, slippage_bps=5)
        assert model.total_bps == 12.0

    def test_total_rate(self):
        model = CostModel(commission_bps=10, spread_bps=0, slippage_bps=0)
        assert model.total_rate == pytest.approx(10 / 10_000)

    def test_zero_cost_model(self):
        model = CostModel()
        assert model.total_bps == 0.0
        assert model.total_rate == 0.0

    def test_immutable(self):
        """CostModel should be frozen (immutable)."""
        model = CostModel(commission_bps=5)
        with pytest.raises(AttributeError):
            model.commission_bps = 10  # type: ignore

    def test_repr(self):
        model = CostModel(commission_bps=3, spread_bps=3, slippage_bps=4)
        r = repr(model)
        assert "10" in r  # total
        assert "commission" in r.lower()

    def test_compute_costs_simple(self):
        model = CostModel(commission_bps=100)  # 1% one-way
        turnover = pd.Series([0.0, 1.0, 0.5, 2.0])
        costs = model.compute_costs(turnover)

        assert costs.iloc[0] == pytest.approx(0.0)
        assert costs.iloc[1] == pytest.approx(0.01)     # 1% × 1.0
        assert costs.iloc[2] == pytest.approx(0.005)    # 1% × 0.5
        assert costs.iloc[3] == pytest.approx(0.02)     # 1% × 2.0

    def test_compute_costs_zero_turnover(self):
        model = CostModel(commission_bps=50, spread_bps=50)
        turnover = pd.Series([0.0, 0.0, 0.0])
        costs = model.compute_costs(turnover)
        assert (costs == 0.0).all()


# ── Cost breakdown ───────────────────────────────────────────────────────────


class TestCostBreakdown:
    def test_breakdown_columns(self):
        model = CostModel(commission_bps=3, spread_bps=3, slippage_bps=4)
        turnover = pd.Series([1.0, 0.5])
        breakdown = model.cost_breakdown(turnover)

        assert set(breakdown.columns) == {"commission", "spread", "slippage", "total"}

    def test_breakdown_sums_to_total(self):
        model = CostModel(commission_bps=3, spread_bps=3, slippage_bps=4)
        turnover = pd.Series([1.0, 0.5, 2.0])
        breakdown = model.cost_breakdown(turnover)

        components = breakdown["commission"] + breakdown["spread"] + breakdown["slippage"]
        pd.testing.assert_series_equal(components, breakdown["total"], check_names=False)

    def test_breakdown_values(self):
        model = CostModel(commission_bps=100, spread_bps=200, slippage_bps=0)
        turnover = pd.Series([1.0])
        breakdown = model.cost_breakdown(turnover)

        assert breakdown["commission"].iloc[0] == pytest.approx(0.01)
        assert breakdown["spread"].iloc[0] == pytest.approx(0.02)
        assert breakdown["slippage"].iloc[0] == pytest.approx(0.0)
        assert breakdown["total"].iloc[0] == pytest.approx(0.03)


# ── Preset models ────────────────────────────────────────────────────────────


class TestPresetModels:
    def test_equity_costs_total(self):
        assert EQUITY_COSTS.total_bps == pytest.approx(10.0)

    def test_crypto_costs_total(self):
        assert CRYPTO_COSTS.total_bps == pytest.approx(15.0)

    def test_futures_costs_total(self):
        assert FUTURES_COSTS.total_bps == pytest.approx(5.0)

    def test_cost_ordering(self):
        """Futures should be cheapest, crypto most expensive."""
        assert FUTURES_COSTS.total_bps < EQUITY_COSTS.total_bps < CRYPTO_COSTS.total_bps

    def test_all_components_non_negative(self):
        for model in [EQUITY_COSTS, CRYPTO_COSTS, FUTURES_COSTS]:
            assert model.commission_bps >= 0
            assert model.spread_bps >= 0
            assert model.slippage_bps >= 0


# ── Factory functions ────────────────────────────────────────────────────────


class TestFactories:
    def test_cost_model_for_equity(self):
        model = cost_model_for_asset_class("equity")
        assert model == EQUITY_COSTS

    def test_cost_model_for_crypto(self):
        model = cost_model_for_asset_class("crypto")
        assert model == CRYPTO_COSTS

    def test_cost_model_for_futures(self):
        model = cost_model_for_asset_class("futures")
        assert model == FUTURES_COSTS

    def test_unknown_asset_class_raises(self):
        with pytest.raises(ValueError, match="Unknown asset class"):
            cost_model_for_asset_class("forex")

    def test_cost_model_from_bps(self):
        model = cost_model_from_bps(25.0)
        assert model.total_bps == pytest.approx(25.0)


# ── Scale costs ──────────────────────────────────────────────────────────────


class TestScaleCosts:
    def test_double_costs(self):
        doubled = scale_costs(EQUITY_COSTS, 2.0)
        assert doubled.total_bps == pytest.approx(EQUITY_COSTS.total_bps * 2)

    def test_half_costs(self):
        halved = scale_costs(EQUITY_COSTS, 0.5)
        assert halved.total_bps == pytest.approx(EQUITY_COSTS.total_bps * 0.5)

    def test_zero_scale(self):
        zeroed = scale_costs(EQUITY_COSTS, 0.0)
        assert zeroed.total_bps == pytest.approx(0.0)

    def test_components_scale_independently(self):
        model = CostModel(commission_bps=10, spread_bps=20, slippage_bps=30)
        scaled = scale_costs(model, 3.0)
        assert scaled.commission_bps == pytest.approx(30)
        assert scaled.spread_bps == pytest.approx(60)
        assert scaled.slippage_bps == pytest.approx(90)


# ── Engine integration ───────────────────────────────────────────────────────


class TestEngineIntegration:
    def test_cost_model_overrides_cost_bps(self):
        """When cost_model is provided, cost_bps should be ignored."""
        prices = _make_prices([100.0, 105.0, 110.0])
        signal = _make_signal([1.0, 1.0, 1.0])

        # cost_bps=0 but model has 100 bps
        model = CostModel(commission_bps=100)
        result = run_backtest(prices, signal, cost_bps=0, cost_model=model)

        assert result.cost_bps == pytest.approx(100.0)
        assert result.costs.sum() > 0

    def test_backward_compatible_cost_bps(self):
        """Using cost_bps without cost_model should still work."""
        prices = _make_prices([100.0, 105.0, 110.0])
        signal = _make_signal([1.0, 1.0, 1.0])

        result = run_backtest(prices, signal, cost_bps=10)
        assert result.cost_bps == pytest.approx(10.0)

    def test_cost_model_stored_in_result(self):
        """The cost model should be accessible from the result."""
        prices = _make_prices([100.0, 105.0, 110.0])
        signal = _make_signal([1.0, 1.0, 1.0])

        result = run_backtest(prices, signal, cost_model=EQUITY_COSTS)
        assert result.cost_model == EQUITY_COSTS

    def test_equity_costs_vs_crypto_costs(self):
        """Crypto costs should produce lower net returns than equity costs."""
        prices = _make_prices([100.0, 105.0, 100.0, 105.0, 100.0])
        signal = _make_signal([1.0, 0.0, 1.0, 0.0, 1.0])  # High turnover

        res_equity = run_backtest(prices, signal, cost_model=EQUITY_COSTS)
        res_crypto = run_backtest(prices, signal, cost_model=CRYPTO_COSTS)

        # Crypto has higher costs → lower equity
        assert res_crypto.equity.iloc[-1] < res_equity.equity.iloc[-1]

    def test_doubled_costs_reduce_returns(self):
        """Doubling costs should reduce net returns."""
        prices = _make_prices([100.0, 105.0, 100.0, 110.0])
        signal = _make_signal([1.0, 0.0, 1.0, 0.0])

        base = run_backtest(prices, signal, cost_model=EQUITY_COSTS)
        doubled = run_backtest(
            prices, signal, cost_model=scale_costs(EQUITY_COSTS, 2.0)
        )

        assert doubled.equity.iloc[-1] < base.equity.iloc[-1]
        # Gross returns should be identical
        pd.testing.assert_series_equal(
            base.gross_returns, doubled.gross_returns, check_names=False
        )
