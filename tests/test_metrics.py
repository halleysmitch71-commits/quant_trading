"""
Tests for performance metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research.metrics.performance import (
    annualised_return,
    annualised_volatility,
    calmar_ratio,
    drawdown_series,
    format_report,
    full_report,
    max_drawdown,
    monthly_returns,
    monthly_stats,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)


def _constant_returns(value: float, n: int = 252) -> pd.Series:
    """Create a Series of constant daily returns."""
    dates = pd.bdate_range("2020-01-02", periods=n)
    return pd.Series([value] * n, index=dates)


def _random_returns(n: int = 504, seed: int = 42) -> pd.Series:
    """Create a Series of random returns for testing."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n)
    return pd.Series(rng.normal(0.0004, 0.01, n), index=dates)


class TestAnnualisedReturn:
    def test_zero_returns(self):
        ret = _constant_returns(0.0)
        assert annualised_return(ret) == pytest.approx(0.0)

    def test_positive_returns(self):
        # 0.04% per day for 252 days
        ret = _constant_returns(0.0004)
        ann = annualised_return(ret)
        # Should be approximately (1.0004)^252 - 1 ≈ 10.6%
        expected = (1 + 0.0004) ** 252 - 1
        assert ann == pytest.approx(expected, rel=0.01)

    def test_negative_returns(self):
        ret = _constant_returns(-0.001)
        assert annualised_return(ret) < 0


class TestAnnualisedVolatility:
    def test_zero_vol(self):
        ret = _constant_returns(0.001)
        assert annualised_volatility(ret) == pytest.approx(0.0)

    def test_positive_vol(self):
        ret = _random_returns()
        vol = annualised_volatility(ret)
        assert vol > 0
        # Daily std ≈ 0.01, annualised ≈ 0.01 * sqrt(252) ≈ 15.9%
        assert 0.10 < vol < 0.25


class TestSharpeRatio:
    def test_zero_vol_returns_zero(self):
        ret = _constant_returns(0.001)
        assert sharpe_ratio(ret) == pytest.approx(0.0)

    def test_positive_sharpe(self):
        ret = _random_returns()
        sr = sharpe_ratio(ret)
        assert isinstance(sr, float)

    def test_negative_returns_negative_sharpe(self):
        ret = _constant_returns(-0.001) + pd.Series(
            np.random.default_rng(42).normal(0, 0.01, 252),
            index=pd.bdate_range("2020-01-02", periods=252),
        )
        assert sharpe_ratio(ret) < 0


class TestSortinoRatio:
    def test_all_positive_returns(self):
        ret = _constant_returns(0.001)
        # No downside → inf
        assert sortino_ratio(ret) == float("inf")

    def test_sortino_computed(self):
        ret = _random_returns()
        sr = sortino_ratio(ret)
        assert isinstance(sr, float)


class TestMaxDrawdown:
    def test_no_drawdown(self):
        ret = _constant_returns(0.001)
        assert max_drawdown(ret) == pytest.approx(0.0)

    def test_known_drawdown(self):
        # Up 10%, then down 20% from peak
        returns = pd.Series(
            [0.10, -0.10, -0.10],
            index=pd.bdate_range("2020-01-02", periods=3),
        )
        mdd = max_drawdown(returns)
        # Peak at 1.1, then 1.1*0.9=0.99, then 0.99*0.9=0.891
        # DD = 0.891/1.1 - 1 = -0.19
        assert mdd < 0
        assert mdd == pytest.approx(0.891 / 1.1 - 1, abs=0.001)


class TestCalmarRatio:
    def test_calmar(self):
        ret = _random_returns()
        cr = calmar_ratio(ret)
        assert isinstance(cr, float)


class TestWinRate:
    def test_all_positive(self):
        ret = _constant_returns(0.001)
        assert win_rate(ret) == pytest.approx(1.0)

    def test_all_negative(self):
        ret = _constant_returns(-0.001)
        assert win_rate(ret) == pytest.approx(0.0)

    def test_mixed(self):
        ret = pd.Series([0.01, -0.01, 0.02, -0.005],
                        index=pd.bdate_range("2020-01-02", periods=4))
        assert win_rate(ret) == pytest.approx(0.5)


class TestMonthlyReturns:
    def test_monthly_aggregation(self):
        ret = _random_returns(n=504)  # ~2 years
        monthly = monthly_returns(ret)
        assert len(monthly) > 20

    def test_monthly_stats_keys(self):
        ret = _random_returns(n=504)
        stats = monthly_stats(ret)
        expected_keys = {
            "avg_monthly_return", "median_monthly_return",
            "pct_positive_months", "best_month", "worst_month", "n_months",
        }
        assert expected_keys == set(stats.keys())


class TestFullReport:
    def test_report_keys(self):
        ret = _random_returns()
        report = full_report(ret)
        assert "total_return" in report
        assert "sharpe_ratio" in report
        assert "max_drawdown" in report
        assert "pct_positive_months" in report

    def test_report_with_turnover_and_costs(self):
        ret = _random_returns()
        turnover = pd.Series(np.abs(np.random.default_rng(42).normal(0.1, 0.05, len(ret))),
                             index=ret.index)
        costs = turnover * 0.001
        report = full_report(ret, turnover=turnover, costs=costs)
        assert "total_turnover" in report
        assert "total_costs" in report

    def test_format_report(self):
        ret = _random_returns()
        report = full_report(ret)
        formatted = format_report(report)
        assert "PERFORMANCE REPORT" in formatted
        assert "Sharpe" in formatted
