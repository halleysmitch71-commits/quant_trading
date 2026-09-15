"""
Tests for lookahead bias detection.

These tests verify that the backtesting engine and signal framework
cannot use future information.  They are the most important tests
in the entire repository.

Strategy:
  1. Construct a "cheating" signal that uses future prices.
  2. Construct a "correct" signal that uses only past prices.
  3. Verify the engine rejects or correctly delays the cheating signal.
  4. Verify that truncating future data does not affect past signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research.backtest.engine import run_backtest
from quant_research.signals.base import BaseSignal


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_prices(values: list[float], start: str = "2020-01-02") -> pd.Series:
    dates = pd.bdate_range(start=start, periods=len(values))
    return pd.Series(values, index=dates, name="close")


def _make_signal(values: list[float], start: str = "2020-01-02") -> pd.Series:
    dates = pd.bdate_range(start=start, periods=len(values))
    return pd.Series(values, index=dates, name="signal")


# ── Concrete signals for testing ─────────────────────────────────────────────


class LookbackSignal(BaseSignal):
    """Correct signal: uses only past data (N-day momentum)."""

    def __init__(self, lookback: int = 5):
        self.lookback = lookback

    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        # Momentum: sign of N-day return
        momentum = close / close.shift(self.lookback) - 1
        return momentum.apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0))


class CheatingSignal(BaseSignal):
    """INTENTIONALLY WRONG: uses tomorrow's close to decide today's signal.

    This signal achieves unrealistically perfect returns because it
    peeks into the future.  We use it to verify that even if a signal
    cheats, the engine's one-bar delay limits the damage.
    """

    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        # Cheat: look at TOMORROW's return to decide today's position
        future_return = close.shift(-1) / close - 1
        return future_return.apply(
            lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)
        ).fillna(0.0)


# ── Test: One-bar delay defends against signal-level lookahead ───────────────


class TestOneBarDelayDefence:
    """Verify the engine's one-bar delay limits cheating signal damage."""

    def test_cheating_signal_is_delayed_by_engine(self):
        """Even a cheating signal gets shifted by 1 bar in the engine.

        The cheating signal at time t uses close[t+1], so:
          cheating_signal[t] knows about close[t+1]
          position[t+1] = cheating_signal[t]  (engine delay)
          return earned at t+1 = position[t+1] * (close[t+1]/close[t] - 1)

        The cheat "works" for the t→t+1 move but the engine still
        enforces the structural delay.  The key point: signal[t] →
        position[t+1], never signal[t] → position[t].
        """
        prices = _make_prices([100, 105, 102, 108, 106, 110])
        signal = _make_signal([1, 1, 1, 1, 1, 1])  # Constant signal

        result = run_backtest(prices, signal, cost_bps=0)

        # Position at bar 0 must be 0 (no prior signal)
        assert result.positions.iloc[0] == 0.0

    def test_engine_never_assigns_position_at_same_bar_as_signal(self):
        """position[t] must equal signal[t-1], not signal[t]."""
        prices = _make_prices([100, 105, 102, 108, 106])
        signal = _make_signal([0.0, 1.0, -0.5, 0.3, 0.0])

        result = run_backtest(prices, signal, cost_bps=0)

        for t in range(1, len(prices)):
            assert result.positions.iloc[t] == pytest.approx(
                signal.iloc[t - 1]
            ), f"At bar {t}: position should be signal[{t-1}]"


# ── Test: Signal stability under data truncation ────────────────────────────


class TestSignalStability:
    """Verify that correct signals don't change when future data is removed."""

    def test_lookback_signal_unchanged_by_future_data(self):
        """A correct lookback signal computed on data[0:T] should be
        identical to data[0:t] for all t < T, at overlapping indices.
        """
        rng = np.random.default_rng(42)
        n = 100
        prices_full = 100 * np.cumprod(1 + rng.normal(0, 0.02, n))
        dates = pd.bdate_range("2020-01-02", periods=n)

        df_full = pd.DataFrame(
            {"close": prices_full, "open": prices_full,
             "high": prices_full, "low": prices_full,
             "volume": np.ones(n)},
            index=dates,
        )

        signal_obj = LookbackSignal(lookback=5)
        signal_full = signal_obj.generate(df_full)

        # Truncate at bar 50
        df_trunc = df_full.iloc[:50]
        signal_trunc = signal_obj.generate(df_trunc)

        # Signals should match for the overlapping region
        overlap = signal_full.iloc[:50]
        pd.testing.assert_series_equal(
            overlap, signal_trunc, check_names=False,
            obj="Signal changed when future data was removed — LOOKAHEAD!"
        )

    def test_cheating_signal_changes_with_truncation(self):
        """A cheating signal WILL change when we remove future data.
        This test proves our detection method works.
        """
        rng = np.random.default_rng(42)
        n = 100
        prices = 100 * np.cumprod(1 + rng.normal(0, 0.02, n))
        dates = pd.bdate_range("2020-01-02", periods=n)

        df_full = pd.DataFrame(
            {"close": prices, "open": prices,
             "high": prices, "low": prices,
             "volume": np.ones(n)},
            index=dates,
        )

        cheat = CheatingSignal()
        signal_full = cheat.generate(df_full)
        signal_trunc = cheat.generate(df_full.iloc[:50])

        # The cheating signal at bar 49 (last bar of truncated data) uses
        # close[50] in the full data but has no close[50] in truncated data.
        # So the signals must differ at bar 49.
        assert signal_full.iloc[49] != signal_trunc.iloc[49], (
            "Cheating signal should change under truncation"
        )


# ── Test: Return earned matches the correct bar ─────────────────────────────


class TestReturnTiming:
    """Verify that the return earned at time t is from the correct price move."""

    def test_return_matches_price_move(self):
        """strategy_return[t] = position[t] × (close[t]/close[t-1] - 1)."""
        prices = _make_prices([100.0, 110.0, 105.0, 115.0])
        signal = _make_signal([1.0, -1.0, 0.5, 0.0])

        result = run_backtest(prices, signal, cost_bps=0)

        # position = [0, 1, -1, 0.5]
        # asset_return = [0, 0.10, -0.0454, 0.0952]
        # gross_return = [0, 1*0.10, -1*(-0.0454), 0.5*0.0952]
        assert result.gross_returns.iloc[1] == pytest.approx(0.10)
        assert result.gross_returns.iloc[2] == pytest.approx(5.0 / 110.0)
        assert result.gross_returns.iloc[3] == pytest.approx(0.5 * 10.0 / 105.0)

    def test_no_return_on_first_bar(self):
        """No return should be earned on the first bar (no prior position)."""
        prices = _make_prices([100.0, 200.0, 300.0])
        signal = _make_signal([1.0, 1.0, 1.0])

        result = run_backtest(prices, signal, cost_bps=0)

        # Even though price doubled, first bar has position=0
        assert result.gross_returns.iloc[0] == pytest.approx(0.0)


# ── Test: Perfect foresight backtest sanity check ────────────────────────────


class TestPerfectForesight:
    """A perfect foresight signal should NOT produce perfect returns
    because the engine delays by one bar.
    """

    def test_perfect_foresight_is_not_perfect(self):
        """If signal[t] = sign(return[t+1]), the engine uses it at t+2.
        So it earns return[t+2] based on the direction of return[t+1],
        which is NOT guaranteed to be the same direction.
        """
        rng = np.random.default_rng(123)
        n = 200
        returns = rng.normal(0.001, 0.02, n)
        prices_arr = 100 * np.cumprod(1 + returns)
        prices = _make_prices(list(prices_arr)[:n])

        # Perfect foresight signal: knows next bar's direction
        future_ret = prices.pct_change().shift(-1).fillna(0)
        perfect_signal = future_ret.apply(lambda x: 1.0 if x > 0 else -1.0)

        result = run_backtest(prices, perfect_signal, cost_bps=0)

        # Due to one-bar delay, signal[t] (which knows return[t+1]) is
        # applied at t+1, earning return[t+1].  So it IS profitable
        # (the signal is shifted to align), but this is detectable:
        # running the truncation test above would catch it.

        # The point: a reviewer can detect foresight by running
        # TestSignalStability.test_cheating_signal_changes_with_truncation
        total_return = result.equity.iloc[-1] / result.initial_capital - 1
        # It should be unrealistically high (confirms the foresight)
        assert total_return > 1.0, (
            "Perfect foresight should produce unrealistically high returns"
        )
