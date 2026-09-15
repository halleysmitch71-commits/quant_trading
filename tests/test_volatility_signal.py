"""
Tests for the Volatility Mean-Reversion Signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research.strategies.volatility_signal import VolatilityMeanReversionSignal


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_ohlcv(n=300, seed=42):
    """Generate synthetic OHLCV data."""
    rng = np.random.default_rng(seed)
    prices = 100 * np.cumprod(1 + rng.normal(0, 0.02, n))
    dates = pd.bdate_range("2020-01-02", periods=n)
    return pd.DataFrame({
        "open": prices,
        "high": prices * (1 + rng.uniform(0, 0.02, n)),
        "low": prices * (1 - rng.uniform(0, 0.02, n)),
        "close": prices,
        "volume": rng.uniform(1e6, 1e7, n),
    }, index=dates)


class TestVolSignalProperties:
    """Test basic signal properties."""

    def test_signal_bounded(self):
        """Signal must be in [-1, 1]."""
        data = _make_ohlcv(500)
        signal_obj = VolatilityMeanReversionSignal()
        signal = signal_obj.generate(data)
        assert signal.min() >= -1.0
        assert signal.max() <= 1.0

    def test_signal_length_matches_data(self):
        """Signal should have same length as input data."""
        data = _make_ohlcv(200)
        signal_obj = VolatilityMeanReversionSignal()
        signal = signal_obj.generate(data)
        assert len(signal) == len(data)

    def test_no_nan_in_output(self):
        """Signal should have no NaN values (filled with 0)."""
        data = _make_ohlcv(300)
        signal_obj = VolatilityMeanReversionSignal()
        signal = signal_obj.generate(data)
        assert not signal.isna().any()

    def test_early_values_are_zero(self):
        """Not enough history → signal should be 0."""
        data = _make_ohlcv(300)
        signal_obj = VolatilityMeanReversionSignal(
            short_window=10, long_window=60, z_lookback=120
        )
        signal = signal_obj.generate(data)
        # The first `long_window` bars don't have enough data for long_vol
        # so vol_ratio is NaN → signal is 0
        assert (signal.iloc[:60] == 0.0).all()


class TestVolSignalNoLookahead:
    """Anti-lookahead tests for the vol signal."""

    def test_signal_stable_under_truncation(self):
        """Truncating future data should not change past signal values."""
        data_full = _make_ohlcv(500)
        signal_obj = VolatilityMeanReversionSignal()

        signal_full = signal_obj.generate(data_full)
        signal_trunc = signal_obj.generate(data_full.iloc[:300])

        # Overlapping region should be identical
        pd.testing.assert_series_equal(
            signal_full.iloc[:300], signal_trunc,
            check_names=False,
            obj="Vol signal changed when future data was removed — LOOKAHEAD!"
        )

    def test_different_parameters_produce_different_signals(self):
        """Different parameter configs should produce different signals."""
        data = _make_ohlcv(400)
        s1 = VolatilityMeanReversionSignal(short_window=5, long_window=30)
        s2 = VolatilityMeanReversionSignal(short_window=20, long_window=120)

        sig1 = s1.generate(data)
        sig2 = s2.generate(data)

        # They shouldn't be identical (ignoring warmup zeros)
        non_zero_1 = sig1[sig1 != 0]
        non_zero_2 = sig2[sig2 != 0]
        if len(non_zero_1) > 0 and len(non_zero_2) > 0:
            # At least some values should differ
            common = non_zero_1.index.intersection(non_zero_2.index)
            if len(common) > 10:
                assert not (sig1.loc[common] == sig2.loc[common]).all()


class TestVolSignalBehaviour:
    """Test the economic intuition of the signal."""

    def test_vol_spike_produces_nonzero_signal(self):
        """When vol regime changes dramatically, signal should be non-zero."""
        rng = np.random.default_rng(42)
        n = 400

        # Calm period followed by a vol spike
        returns_calm = rng.normal(0, 0.005, 300)  # Low vol
        returns_spike = rng.normal(0, 0.05, 100)   # 10× vol spike
        returns = np.concatenate([returns_calm, returns_spike])
        prices = 100 * np.cumprod(1 + returns)

        dates = pd.bdate_range("2020-01-02", periods=n)
        data = pd.DataFrame({
            "open": prices, "high": prices, "low": prices,
            "close": prices, "volume": np.ones(n),
        }, index=dates)

        signal_obj = VolatilityMeanReversionSignal(
            short_window=10, long_window=60, z_lookback=120
        )
        signal = signal_obj.generate(data)

        # During the vol regime change, the signal should have non-trivial
        # magnitude (positive or negative depending on normalisation)
        spike_signal = signal.iloc[-50:]
        non_zero_spike = spike_signal[spike_signal.abs() > 0.01]
        assert len(non_zero_spike) > 0, (
            "Signal should react to a major vol regime change"
        )


class TestVolSignalRepr:
    """Test string representation."""

    def test_repr(self):
        s = VolatilityMeanReversionSignal(short_window=10, long_window=60)
        r = repr(s)
        assert "VolatilityMeanReversionSignal" in r
        assert "short=10" in r
        assert "long=60" in r
