"""
Tests for IC analysis module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research.metrics.ic_analysis import (
    compute_ic, rolling_ic, ic_summary, ic_decay_curve, ic_by_year,
    format_ic_report,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_series(values, start="2020-01-02"):
    dates = pd.bdate_range(start=start, periods=len(values))
    return pd.Series(values, index=dates)


class TestComputeIC:
    """Tests for the basic IC computation."""

    def test_perfect_positive_signal(self):
        """A signal that always predicts direction correctly has IC = +1."""
        signal = _make_series([1.0, 1.0, -1.0, 1.0, -1.0])
        fwd_ret = _make_series([0.01, 0.02, -0.01, 0.03, -0.02])
        ic = compute_ic(signal, fwd_ret)
        assert (ic == 1.0).all()

    def test_perfect_negative_signal(self):
        """A signal that always predicts wrong has IC = -1."""
        signal = _make_series([-1.0, -1.0, 1.0, -1.0, 1.0])
        fwd_ret = _make_series([0.01, 0.02, -0.01, 0.03, -0.02])
        ic = compute_ic(signal, fwd_ret)
        assert (ic == -1.0).all()

    def test_random_signal_near_zero(self):
        """A random signal should have IC near zero."""
        rng = np.random.default_rng(42)
        n = 1000
        signal = _make_series(rng.standard_normal(n))
        fwd_ret = _make_series(rng.standard_normal(n))
        ic = compute_ic(signal, fwd_ret)
        assert abs(ic.mean()) < 0.1  # Should be close to 0

    def test_handles_nan(self):
        """IC should handle NaN values gracefully."""
        signal = _make_series([1.0, np.nan, -1.0, 1.0, -1.0])
        fwd_ret = _make_series([0.01, 0.02, -0.01, np.nan, -0.02])
        ic = compute_ic(signal, fwd_ret)
        assert len(ic) == 3  # Only non-NaN pairs


class TestICSummary:
    """Tests for the comprehensive IC summary."""

    def test_hit_rate_perfect(self):
        """Perfect signal should have hit rate of 1.0."""
        signal = _make_series([1.0, 1.0, -1.0, 1.0, -1.0] * 20)
        fwd_ret = _make_series([0.01, 0.02, -0.01, 0.03, -0.02] * 20)
        summary = ic_summary(signal, fwd_ret)
        assert summary["hit_rate"] == pytest.approx(1.0)

    def test_summary_keys(self):
        """Summary should contain all expected keys."""
        rng = np.random.default_rng(42)
        n = 200
        signal = _make_series(rng.standard_normal(n))
        fwd_ret = _make_series(rng.standard_normal(n))
        summary = ic_summary(signal, fwd_ret)

        expected_keys = [
            "ic_mean", "ic_std", "ic_ir", "hit_rate",
            "spearman_corr", "spearman_p", "pearson_corr", "pearson_p",
            "t_stat", "p_value", "weighted_ic", "n_obs", "n_active",
            "avg_signal_magnitude",
        ]
        for key in expected_keys:
            assert key in summary, f"Missing key: {key}"

    def test_ic_ir_computation(self):
        """IC_IR should be ic_mean / ic_std."""
        rng = np.random.default_rng(42)
        n = 200
        signal = _make_series(rng.standard_normal(n))
        fwd_ret = _make_series(rng.standard_normal(n))
        summary = ic_summary(signal, fwd_ret)
        if summary["ic_std"] > 1e-10:
            expected_ir = summary["ic_mean"] / summary["ic_std"]
            assert summary["ic_ir"] == pytest.approx(expected_ir)

    def test_small_sample_returns_defaults(self):
        """With fewer than 10 observations, should return safe defaults."""
        signal = _make_series([1.0, -1.0, 1.0])
        fwd_ret = _make_series([0.01, -0.01, 0.02])
        summary = ic_summary(signal, fwd_ret)
        assert summary["n_obs"] == 3


class TestICDecay:
    """Tests for IC decay curve computation."""

    def test_returns_all_lags(self):
        """Should return a row for each requested lag."""
        rng = np.random.default_rng(42)
        n = 300
        signal = _make_series(rng.standard_normal(n))
        returns = _make_series(rng.standard_normal(n) * 0.01)
        lags = [1, 5, 10, 20]
        decay = ic_decay_curve(signal, returns, lags=lags)
        assert len(decay) == len(lags)
        assert list(decay["lag"]) == lags

    def test_decay_has_spearman_column(self):
        """Each row should have spearman_corr."""
        rng = np.random.default_rng(42)
        n = 200
        signal = _make_series(rng.standard_normal(n))
        returns = _make_series(rng.standard_normal(n) * 0.01)
        decay = ic_decay_curve(signal, returns, lags=[1, 5])
        assert "spearman_corr" in decay.columns


class TestICByYear:
    """Tests for yearly IC breakdown."""

    def test_returns_one_row_per_year(self):
        """Should return a row for each year with enough data."""
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2020-01-02", "2022-12-30")
        n = len(dates)
        signal = pd.Series(rng.standard_normal(n), index=dates)
        fwd_ret = pd.Series(rng.standard_normal(n) * 0.01, index=dates)
        yearly = ic_by_year(signal, fwd_ret)
        assert len(yearly) == 3  # 2020, 2021, 2022
        assert list(yearly["year"]) == [2020, 2021, 2022]


class TestFormatICReport:
    """Tests for IC report formatting."""

    def test_format_returns_string(self):
        """Format should return a non-empty string."""
        summary = {
            "ic_mean": 0.05, "ic_std": 0.3, "ic_ir": 0.167,
            "hit_rate": 0.52, "spearman_corr": 0.03, "spearman_p": 0.15,
            "pearson_corr": 0.02, "pearson_p": 0.20,
            "t_stat": 1.5, "p_value": 0.13, "weighted_ic": 0.001,
            "n_obs": 500, "n_active": 480, "avg_signal_magnitude": 0.3,
        }
        result = format_ic_report(summary)
        assert isinstance(result, str)
        assert "IC Mean" in result
        assert "Hit Rate" in result
