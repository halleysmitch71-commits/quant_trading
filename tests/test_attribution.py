"""
Tests for P&L attribution module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research.metrics.attribution import (
    asset_pnl_attribution, long_short_attribution,
    yearly_attribution, cost_attribution,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_df(data_dict, start="2020-01-02", periods=None):
    if periods is None:
        periods = len(next(iter(data_dict.values())))
    dates = pd.bdate_range(start=start, periods=periods)
    return pd.DataFrame(data_dict, index=dates)


class TestAssetPnlAttribution:
    """Tests for asset-level P&L decomposition."""

    def test_total_pnl_sums_correctly(self):
        """Sum of per-asset P&L should equal portfolio P&L."""
        positions = _make_df({"A": [0.5, 0.5, -0.3], "B": [0.3, -0.2, 0.4]})
        returns = _make_df({"A": [0.01, -0.02, 0.03], "B": [-0.01, 0.02, -0.01]})

        attr = asset_pnl_attribution(positions, returns)
        total_from_attr = attr["total_pnl"].sum()
        expected = (positions * returns).sum().sum()
        assert total_from_attr == pytest.approx(expected)

    def test_pct_of_total_sums_to_100(self):
        """Percentages should sum to approximately 100%."""
        positions = _make_df({"A": [0.5, 0.3], "B": [0.3, 0.2]})
        returns = _make_df({"A": [0.01, 0.02], "B": [0.01, 0.01]})

        attr = asset_pnl_attribution(positions, returns)
        assert attr["pct_of_total"].sum() == pytest.approx(100.0, abs=0.1)

    def test_long_short_day_counts(self):
        """Long and short day counts should be correct."""
        positions = _make_df({"A": [0.5, -0.3, 0.0, 0.2]})
        returns = _make_df({"A": [0.01, 0.01, 0.01, 0.01]})

        attr = asset_pnl_attribution(positions, returns)
        assert attr.iloc[0]["n_long_days"] == 2
        assert attr.iloc[0]["n_short_days"] == 1
        assert attr.iloc[0]["n_flat_days"] == 1


class TestLongShortAttribution:
    """Tests for directional P&L decomposition."""

    def test_long_short_sum_to_total(self):
        """Long P&L + Short P&L should equal total P&L."""
        positions = _make_df({"A": [0.5, -0.3, 0.2], "B": [-0.1, 0.4, 0.3]})
        returns = _make_df({"A": [0.01, -0.02, 0.015], "B": [0.005, 0.01, -0.005]})

        ls = long_short_attribution(positions, returns)
        assert ls["long_pnl"] + ls["short_pnl"] == pytest.approx(ls["total_pnl"], abs=1e-10)

    def test_long_pct_plus_short_pct_is_100(self):
        """Directional percentages should sum to 100%."""
        positions = _make_df({"A": [0.5, -0.3], "B": [0.2, 0.4]})
        returns = _make_df({"A": [0.02, 0.01], "B": [0.01, -0.01]})

        ls = long_short_attribution(positions, returns)
        assert ls["long_pct"] + ls["short_pct"] == pytest.approx(100.0, abs=0.1)


class TestYearlyAttribution:
    """Tests for yearly P&L breakdown."""

    def test_returns_one_row_per_year(self):
        """Should have one row per calendar year."""
        dates = pd.bdate_range("2020-01-02", "2022-12-30")
        returns = pd.Series(np.random.default_rng(42).normal(0, 0.01, len(dates)),
                            index=dates)
        yearly = yearly_attribution(returns)
        assert len(yearly) == 3
        assert list(yearly["year"]) == [2020, 2021, 2022]

    def test_total_return_is_compounded(self):
        """Total return for each year should be compounded, not summed."""
        dates = pd.bdate_range("2020-01-02", periods=5)
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.01], index=dates)
        yearly = yearly_attribution(returns)
        expected = (1 + returns).prod() - 1
        assert yearly.iloc[0]["total_return"] == pytest.approx(expected, abs=1e-6)


class TestCostAttribution:
    """Tests for cost decomposition."""

    def test_cost_proportional_to_turnover_and_rate(self):
        """Cost = turnover × rate."""
        positions = _make_df({"A": [0.0, 0.5, 0.5, -0.3]})
        cost_rates = {"A": 0.001}  # 10 bps

        attr = cost_attribution(positions, cost_rates)
        # Turnover: |0.5| + |0| + |-0.8| = 1.3
        # Cost = 1.3 × 0.001 = 0.0013
        expected_turnover = abs(0.5) + abs(0.0) + abs(-0.8)
        assert attr.iloc[0]["total_turnover"] == pytest.approx(expected_turnover)
        assert attr.iloc[0]["total_cost"] == pytest.approx(expected_turnover * 0.001)
