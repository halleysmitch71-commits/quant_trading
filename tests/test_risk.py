"""
Tests for risk overlay functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research.portfolio.risk import (
    drawdown_throttle,
    max_concentration,
    apply_risk_controls,
)


def _make_positions(n: int = 100) -> pd.DataFrame:
    """Simple 2-asset position DataFrame."""
    dates = pd.bdate_range("2020-01-02", periods=n)
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {"A": rng.uniform(-0.3, 0.3, n), "B": rng.uniform(-0.2, 0.2, n)},
        index=dates,
    )


def _make_returns(n: int = 100) -> pd.Series:
    """Simple return series."""
    dates = pd.bdate_range("2020-01-02", periods=n)
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(0.0003, 0.01, n), index=dates)


def _make_drawdown_returns(n: int = 100) -> pd.Series:
    """Returns that produce a meaningful drawdown."""
    dates = pd.bdate_range("2020-01-02", periods=n)
    # First 50 bars: positive, then 50 bars: negative
    returns = np.concatenate([
        np.full(50, 0.005),   # Up 0.5% per day
        np.full(50, -0.008),  # Down 0.8% per day
    ])
    return pd.Series(returns, index=dates)


class TestDrawdownThrottle:
    def test_no_drawdown_no_change(self):
        """When no drawdown, positions unchanged."""
        positions = _make_positions()
        returns = pd.Series(0.001, index=positions.index)  # Always positive
        adjusted = drawdown_throttle(positions, returns, dd_threshold=-0.05)
        # First bar is lagged → may differ, compare from bar 1
        pd.testing.assert_frame_equal(
            positions.iloc[1:], adjusted.iloc[1:], check_names=False,
        )

    def test_drawdown_reduces_positions(self):
        """During drawdown, positions should be smaller."""
        positions = _make_positions()
        returns = _make_drawdown_returns(len(positions))
        adjusted = drawdown_throttle(positions, returns, dd_threshold=-0.05)

        # In the drawdown phase (last 30 bars), positions should be reduced
        original_mag = positions.iloc[-30:].abs().mean().mean()
        adjusted_mag = adjusted.iloc[-30:].abs().mean().mean()
        assert adjusted_mag < original_mag

    def test_scale_never_below_min(self):
        """Scale factor should never drop below min_scale."""
        positions = _make_positions()
        returns = _make_drawdown_returns(len(positions))
        adjusted = drawdown_throttle(
            positions, returns, dd_threshold=-0.01, min_scale=0.5,
        )
        # The ratio should be >= min_scale for non-zero positions
        ratio = adjusted.abs() / positions.abs().replace(0, np.nan)
        assert ratio.min().min() >= 0.5 - 1e-10

    def test_uses_lagged_drawdown(self):
        """Throttle should use PREVIOUS day's drawdown (no lookahead)."""
        positions = _make_positions(n=10)
        # Crash on day 5: all positions should still be unchanged on day 5
        # because the lagged scale uses day 4's drawdown
        returns = pd.Series(
            [0.01, 0.01, 0.01, 0.01, -0.20, -0.20, -0.05, -0.05, 0.01, 0.01],
            index=positions.index,
        )
        adjusted = drawdown_throttle(positions, returns, dd_threshold=-0.05)
        # Day 5 (index 4): crash happens, but lagged scale uses day 4 (no DD yet)
        pd.testing.assert_series_equal(
            positions.iloc[4], adjusted.iloc[4], check_names=False,
        )


class TestMaxConcentration:
    def test_no_concentration_breach(self):
        """When no asset exceeds limit, positions unchanged."""
        dates = pd.bdate_range("2020-01-02", periods=5)
        positions = pd.DataFrame(
            {"A": [0.2, 0.2, 0.2, 0.2, 0.2], "B": [0.2, 0.2, 0.2, 0.2, 0.2]},
            index=dates,
        )
        adjusted = max_concentration(positions, max_pct=0.6)
        pd.testing.assert_frame_equal(positions, adjusted)

    def test_clamps_dominant_asset(self):
        """An asset exceeding the limit should be clamped."""
        dates = pd.bdate_range("2020-01-02", periods=1)
        positions = pd.DataFrame(
            {"A": [0.8], "B": [0.2]},
            index=dates,
        )
        # Gross = 1.0, max 40% → A should be clamped to 0.4
        adjusted = max_concentration(positions, max_pct=0.40)
        assert adjusted["A"].iloc[0] <= 0.40 + 1e-10

    def test_preserves_sign(self):
        """Concentration cap should preserve position sign."""
        dates = pd.bdate_range("2020-01-02", periods=1)
        positions = pd.DataFrame(
            {"A": [-0.8], "B": [0.2]},
            index=dates,
        )
        adjusted = max_concentration(positions, max_pct=0.40)
        assert adjusted["A"].iloc[0] < 0  # Still negative


class TestApplyRiskControls:
    def test_returns_dataframe(self):
        positions = _make_positions()
        returns = _make_returns()
        adjusted = apply_risk_controls(positions, returns)
        assert isinstance(adjusted, pd.DataFrame)

    def test_same_shape(self):
        positions = _make_positions()
        returns = _make_returns()
        adjusted = apply_risk_controls(positions, returns)
        assert adjusted.shape == positions.shape

    def test_backward_compatible_with_no_controls(self):
        """With very lenient settings, should approximately match original."""
        positions = _make_positions()
        returns = _make_returns()
        adjusted = apply_risk_controls(
            positions, returns,
            dd_threshold=-1.0,  # Will never trigger
            concentration_limit=1.0,  # No constraint
        )
        # Should be very close to original (except first row from lag)
        pd.testing.assert_frame_equal(
            positions.iloc[1:], adjusted.iloc[1:], check_names=False,
        )
