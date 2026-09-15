"""
Tests for the backtesting engine.

These tests verify:
  1. The one-bar delay is correctly applied.
  2. Returns are computed correctly.
  3. Transaction costs are computed correctly.
  4. Edge cases are handled.
  5. The engine produces deterministic results.

All tests use synthetic data with known, hand-verifiable outcomes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research.backtest.engine import BacktestResult, run_backtest


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_prices(
    values: list[float],
    start: str = "2020-01-02",
) -> pd.Series:
    """Create a price Series from a list of values."""
    dates = pd.bdate_range(start=start, periods=len(values))
    return pd.Series(values, index=dates, name="close")


def _make_signal(
    values: list[float],
    start: str = "2020-01-02",
) -> pd.Series:
    """Create a signal Series from a list of values."""
    dates = pd.bdate_range(start=start, periods=len(values))
    return pd.Series(values, index=dates, name="signal")


# ── Input validation ─────────────────────────────────────────────────────────


class TestInputValidation:
    def test_prices_must_be_series(self):
        with pytest.raises(ValueError, match="pd.Series"):
            run_backtest(
                prices=[100, 101],  # type: ignore
                signal=_make_signal([0.5, 0.5]),
            )

    def test_signal_must_be_series(self):
        with pytest.raises(ValueError, match="pd.Series"):
            run_backtest(
                prices=_make_prices([100, 101]),
                signal=[0.5, 0.5],  # type: ignore
            )

    def test_prices_need_datetime_index(self):
        prices = pd.Series([100, 101], index=[0, 1])
        signal = pd.Series([0.5, 0.5], index=[0, 1])
        with pytest.raises(ValueError, match="DatetimeIndex"):
            run_backtest(prices=prices, signal=signal)

    def test_minimum_two_bars(self):
        with pytest.raises(ValueError, match="at least 2"):
            run_backtest(
                prices=_make_prices([100]),
                signal=_make_signal([0.5]),
            )

    def test_mismatched_index(self):
        prices = _make_prices([100, 101, 102])
        signal = _make_signal([0.5, 0.5], start="2020-01-06")
        with pytest.raises(ValueError, match="same index"):
            run_backtest(prices=prices, signal=signal)


# ── One-bar delay ───────────────────────────────────────────────────────────


class TestOneBarDelay:
    """The most critical tests: verify the timing contract."""

    def test_position_is_signal_shifted_by_one(self):
        """position[t] = signal[t-1], position[0] = 0."""
        prices = _make_prices([100, 102, 104, 103, 105])
        signal = _make_signal([0.0, 0.5, 1.0, 0.3, 0.0])

        result = run_backtest(prices, signal, cost_bps=0)

        expected_positions = pd.Series(
            [0.0, 0.0, 0.5, 1.0, 0.3],
            index=prices.index,
        )
        pd.testing.assert_series_equal(
            result.positions, expected_positions, check_names=False
        )

    def test_first_bar_position_is_zero(self):
        """On the first bar, we have no prior signal → flat."""
        prices = _make_prices([100, 110, 120])
        signal = _make_signal([1.0, 1.0, 1.0])

        result = run_backtest(prices, signal, cost_bps=0)

        assert result.positions.iloc[0] == 0.0

    def test_signal_change_affects_next_bar_only(self):
        """Changing signal[t] should only affect position[t+1], not position[t]."""
        prices = _make_prices([100, 105, 110, 115])

        # Signal A: go long on bar 1
        sig_a = _make_signal([0.0, 1.0, 1.0, 1.0])
        res_a = run_backtest(prices, sig_a, cost_bps=0)

        # Signal B: go long on bar 2 (one bar later)
        sig_b = _make_signal([0.0, 0.0, 1.0, 1.0])
        res_b = run_backtest(prices, sig_b, cost_bps=0)

        # Positions at bar 2 should differ
        assert res_a.positions.iloc[2] == 1.0  # signal[1]=1.0 → pos[2]=1.0
        assert res_b.positions.iloc[2] == 0.0  # signal[1]=0.0 → pos[2]=0.0

        # Positions at bar 1 should be the same (both flat)
        assert res_a.positions.iloc[1] == 0.0
        assert res_b.positions.iloc[1] == 0.0


# ── Return calculations ─────────────────────────────────────────────────────


class TestReturnCalculations:
    def test_zero_signal_gives_zero_return(self):
        """Flat position should produce zero returns regardless of price movement."""
        prices = _make_prices([100, 120, 80, 150])
        signal = _make_signal([0.0, 0.0, 0.0, 0.0])

        result = run_backtest(prices, signal, cost_bps=0)

        assert (result.gross_returns == 0.0).all()
        assert result.equity.iloc[-1] == pytest.approx(100_000.0)

    def test_full_long_captures_full_return(self):
        """100% long position should capture the full asset return (after delay)."""
        # Price goes: 100 → 110 → 121  (10% then 10%)
        prices = _make_prices([100.0, 110.0, 121.0])
        signal = _make_signal([1.0, 1.0, 1.0])

        result = run_backtest(prices, signal, cost_bps=0)

        # position = [0, 1, 1]  (first bar is flat due to delay)
        # asset_return = [0, 0.10, 0.10]
        # gross_return = [0, 0.10, 0.10]
        assert result.gross_returns.iloc[0] == pytest.approx(0.0)
        assert result.gross_returns.iloc[1] == pytest.approx(0.10)
        assert result.gross_returns.iloc[2] == pytest.approx(0.10)

    def test_half_position_captures_half_return(self):
        """50% position should earn 50% of the asset return."""
        prices = _make_prices([100.0, 110.0, 121.0])
        signal = _make_signal([0.5, 0.5, 0.5])

        result = run_backtest(prices, signal, cost_bps=0)

        # position = [0, 0.5, 0.5]
        # gross_return = [0, 0.05, 0.05]
        assert result.gross_returns.iloc[1] == pytest.approx(0.05)
        assert result.gross_returns.iloc[2] == pytest.approx(0.05)

    def test_short_position_inverts_return(self):
        """Short position should earn negative of asset return."""
        prices = _make_prices([100.0, 110.0, 100.0])
        signal = _make_signal([-1.0, -1.0, -1.0])

        result = run_backtest(prices, signal, cost_bps=0)

        # position = [0, -1, -1]
        # asset_return[1] = 10/100 = 0.10
        # gross_return[1] = -1 * 0.10 = -0.10
        assert result.gross_returns.iloc[1] == pytest.approx(-0.10)

        # asset_return[2] = -10/110 ≈ -0.0909
        # gross_return[2] = -1 * -0.0909 ≈ +0.0909
        assert result.gross_returns.iloc[2] == pytest.approx(10.0 / 110.0)

    def test_equity_curve_is_compounded(self):
        """Equity should compound daily returns multiplicatively."""
        prices = _make_prices([100.0, 110.0, 121.0])
        signal = _make_signal([1.0, 1.0, 1.0])

        result = run_backtest(prices, signal, initial_capital=10_000, cost_bps=0)

        # Bar 0: equity = 10000 * (1+0) = 10000
        # Bar 1: equity = 10000 * (1+0.10) = 11000
        # Bar 2: equity = 11000 * (1+0.10) = 12100
        assert result.equity.iloc[0] == pytest.approx(10_000.0)
        assert result.equity.iloc[1] == pytest.approx(11_000.0)
        assert result.equity.iloc[2] == pytest.approx(12_100.0)

    def test_known_sequence(self):
        """Verify a fully hand-computed example end-to-end."""
        # Prices: 100 → 105 → 100 → 110
        prices = _make_prices([100.0, 105.0, 100.0, 110.0])
        # Signal: flat, long, flat, long
        signal = _make_signal([0.0, 1.0, 0.0, 1.0])

        result = run_backtest(prices, signal, initial_capital=100, cost_bps=0)

        # Positions (shifted): [0, 0, 1, 0]
        assert list(result.positions) == [0.0, 0.0, 1.0, 0.0]

        # Asset returns: [0, 5%, -4.76%, 10%]
        assert result.asset_returns.iloc[1] == pytest.approx(0.05)
        assert result.asset_returns.iloc[2] == pytest.approx(-5.0 / 105.0)

        # Gross returns: [0, 0, -4.76%, 0]
        assert result.gross_returns.iloc[1] == pytest.approx(0.0)
        assert result.gross_returns.iloc[2] == pytest.approx(-5.0 / 105.0)
        assert result.gross_returns.iloc[3] == pytest.approx(0.0)

        # Final equity = 100 * (1+0) * (1+0) * (1 + (-5/105)) * (1+0)
        expected_equity = 100 * (1 - 5.0 / 105.0)
        assert result.equity.iloc[-1] == pytest.approx(expected_equity)


# ── Transaction costs ────────────────────────────────────────────────────────


class TestTransactionCosts:
    def test_zero_cost_no_drag(self):
        """Zero bps should produce identical gross and net returns."""
        prices = _make_prices([100.0, 110.0, 121.0])
        signal = _make_signal([1.0, 1.0, 1.0])

        result = run_backtest(prices, signal, cost_bps=0)

        pd.testing.assert_series_equal(
            result.gross_returns, result.net_returns, check_names=False
        )

    def test_cost_on_entry(self):
        """Entering a position should incur cost = |Δposition| × rate."""
        prices = _make_prices([100.0, 110.0, 121.0])
        signal = _make_signal([1.0, 1.0, 1.0])

        result = run_backtest(prices, signal, cost_bps=100)  # 1% cost

        # position = [0, 1, 1]
        # turnover = [0, 1, 0]  (enter fully on bar 1, hold on bar 2)
        assert result.turnover.iloc[0] == pytest.approx(0.0)
        assert result.turnover.iloc[1] == pytest.approx(1.0)
        assert result.turnover.iloc[2] == pytest.approx(0.0)

        # costs = [0, 0.01, 0]
        assert result.costs.iloc[1] == pytest.approx(0.01)
        assert result.costs.iloc[2] == pytest.approx(0.0)

    def test_cost_on_reversal(self):
        """Going from +1 to −1 should incur cost on |Δ| = 2."""
        prices = _make_prices([100.0, 105.0, 110.0, 105.0])
        signal = _make_signal([1.0, -1.0, -1.0, -1.0])

        result = run_backtest(prices, signal, cost_bps=100)  # 1%

        # position = [0, 1, -1, -1]
        # turnover = [0, 1, 2, 0]
        assert result.turnover.iloc[1] == pytest.approx(1.0)
        assert result.turnover.iloc[2] == pytest.approx(2.0)
        assert result.turnover.iloc[3] == pytest.approx(0.0)

    def test_costs_reduce_equity(self):
        """Net equity should be less than gross equity when costs > 0."""
        prices = _make_prices([100.0, 100.0, 100.0, 100.0, 100.0])
        signal = _make_signal([1.0, 0.0, 1.0, 0.0, 1.0])  # Lots of trading

        result_free = run_backtest(prices, signal, cost_bps=0)
        result_costly = run_backtest(prices, signal, cost_bps=50)

        # Prices don't move, so gross return = 0.
        # Costly version should be lower due to churning.
        assert result_costly.equity.iloc[-1] < result_free.equity.iloc[-1]

    def test_holding_incurs_no_ongoing_cost(self):
        """Holding a static position should not incur any cost after entry."""
        prices = _make_prices([100.0, 105.0, 110.0, 115.0, 120.0])
        signal = _make_signal([1.0, 1.0, 1.0, 1.0, 1.0])

        result = run_backtest(prices, signal, cost_bps=100)

        # Only one cost event: entering on bar 1
        assert result.costs.iloc[1] > 0
        assert result.costs.iloc[2] == pytest.approx(0.0)
        assert result.costs.iloc[3] == pytest.approx(0.0)
        assert result.costs.iloc[4] == pytest.approx(0.0)


# ── Result structure ─────────────────────────────────────────────────────────


class TestBacktestResult:
    def test_result_type(self):
        prices = _make_prices([100.0, 110.0, 121.0])
        signal = _make_signal([0.5, 0.5, 0.5])
        result = run_backtest(prices, signal)

        assert isinstance(result, BacktestResult)

    def test_all_series_same_length(self):
        prices = _make_prices([100.0, 110.0, 121.0, 130.0])
        signal = _make_signal([0.5, 0.5, 0.5, 0.5])
        result = run_backtest(prices, signal)

        n = len(prices)
        assert len(result.signal) == n
        assert len(result.positions) == n
        assert len(result.asset_returns) == n
        assert len(result.gross_returns) == n
        assert len(result.turnover) == n
        assert len(result.costs) == n
        assert len(result.net_returns) == n
        assert len(result.equity) == n

    def test_deterministic(self):
        """Running the same inputs twice must produce identical results."""
        prices = _make_prices([100.0, 105.0, 102.0, 108.0, 106.0])
        signal = _make_signal([0.5, 1.0, -0.5, 0.3, 0.0])

        r1 = run_backtest(prices, signal)
        r2 = run_backtest(prices, signal)

        pd.testing.assert_series_equal(r1.equity, r2.equity)
        pd.testing.assert_series_equal(r1.net_returns, r2.net_returns)

    def test_repr(self):
        prices = _make_prices([100.0, 110.0])
        signal = _make_signal([1.0, 1.0])
        result = run_backtest(prices, signal)
        r = repr(result)
        assert "BacktestResult" in r
        assert "bars=" in r
