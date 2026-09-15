"""
Tests for strategies — mean-reversion baseline and trend-following.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research.strategies.mean_reversion import MeanReversionSignal


def _make_ohlcv(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV data."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.015, n))
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.002, n)),
            "high": close * (1 + np.abs(rng.normal(0, 0.005, n))),
            "low": close * (1 - np.abs(rng.normal(0, 0.005, n))),
            "close": close,
            "volume": rng.integers(1e6, 1e7, n).astype(float),
        },
        index=dates,
    )


class TestMeanReversionSignal:
    def test_output_is_series(self):
        data = _make_ohlcv()
        signal = MeanReversionSignal().generate(data)
        assert isinstance(signal, pd.Series)

    def test_output_same_length(self):
        data = _make_ohlcv()
        signal = MeanReversionSignal().generate(data)
        assert len(signal) == len(data)

    def test_no_nans(self):
        data = _make_ohlcv()
        signal = MeanReversionSignal().generate(data)
        assert signal.isna().sum() == 0

    def test_clipped_to_range(self):
        data = _make_ohlcv()
        signal = MeanReversionSignal().generate(data)
        assert signal.min() >= -1.0
        assert signal.max() <= 1.0

    def test_warmup_period_is_flat(self):
        """First vol_lookback bars should be 0 (no vol estimate yet)."""
        data = _make_ohlcv(n=100)
        sig_obj = MeanReversionSignal(vol_lookback=20)
        signal = sig_obj.generate(data)
        # First 20 bars need vol_lookback data → signal should be 0
        assert (signal.iloc[:20] == 0.0).all()

    def test_inverts_direction(self):
        """After a large up day, signal should be negative (short bias)."""
        n = 50
        dates = pd.bdate_range("2020-01-02", periods=n)
        # Flat prices, then a big up day
        close = np.ones(n) * 100.0
        close[-1] = 110.0  # +10% on last day
        data = pd.DataFrame(
            {"close": close, "open": close, "high": close, "low": close,
             "volume": np.ones(n) * 1e6},
            index=dates,
        )
        signal = MeanReversionSignal(vol_lookback=20).generate(data)
        # After a +10% day, the mean-reversion signal should be negative
        assert signal.iloc[-1] < 0

    def test_stable_under_truncation(self):
        """Signal should not change when future data is removed (no lookahead)."""
        data_full = _make_ohlcv(n=200)
        data_trunc = data_full.iloc[:100]

        sig_obj = MeanReversionSignal(vol_lookback=20)
        signal_full = sig_obj.generate(data_full)
        signal_trunc = sig_obj.generate(data_trunc)

        # Signals should match for the overlapping period
        pd.testing.assert_series_equal(
            signal_full.iloc[:100],
            signal_trunc,
            check_names=False,
        )

    def test_repr(self):
        sig = MeanReversionSignal(vol_lookback=30)
        assert "30" in repr(sig)

    def test_different_lookback(self):
        """Different lookbacks should produce different signals."""
        data = _make_ohlcv(n=100)
        sig_20 = MeanReversionSignal(vol_lookback=20).generate(data)
        sig_40 = MeanReversionSignal(vol_lookback=40).generate(data)
        # They should differ (at least somewhere after warmup)
        assert not sig_20.equals(sig_40)


# ── Trend-Following tests ───────────────────────────────────────────────────

from quant_research.strategies.trend_following import TrendFollowingSignal


class TestTrendFollowingSignal:
    def test_output_is_series(self):
        data = _make_ohlcv(n=200)
        signal = TrendFollowingSignal().generate(data)
        assert isinstance(signal, pd.Series)

    def test_output_same_length(self):
        data = _make_ohlcv(n=200)
        signal = TrendFollowingSignal().generate(data)
        assert len(signal) == len(data)

    def test_no_nans(self):
        data = _make_ohlcv(n=200)
        signal = TrendFollowingSignal().generate(data)
        assert signal.isna().sum() == 0

    def test_clipped_to_range(self):
        data = _make_ohlcv(n=200)
        signal = TrendFollowingSignal().generate(data)
        assert signal.min() >= -1.0
        assert signal.max() <= 1.0

    def test_warmup_period_is_flat(self):
        """First `lookback` bars should be 0 (no trend estimate yet)."""
        data = _make_ohlcv(n=200)
        sig_obj = TrendFollowingSignal(lookback=60)
        signal = sig_obj.generate(data)
        # First 60 bars need lookback data → signal should be 0
        assert (signal.iloc[:60] == 0.0).all()

    def test_uptrend_gives_positive_signal(self):
        """A clear uptrend should produce a positive (long) signal."""
        n = 150
        dates = pd.bdate_range("2020-01-02", periods=n)
        # Steady uptrend: 1% per day
        close = 100.0 * (1.01 ** np.arange(n))
        data = pd.DataFrame(
            {"close": close, "open": close, "high": close, "low": close,
             "volume": np.ones(n) * 1e6},
            index=dates,
        )
        signal = TrendFollowingSignal(lookback=60).generate(data)
        # After warmup, signal should be positive
        assert signal.iloc[-1] > 0

    def test_downtrend_gives_negative_signal(self):
        """A clear downtrend should produce a negative (short) signal."""
        n = 150
        dates = pd.bdate_range("2020-01-02", periods=n)
        # Steady downtrend: -0.5% per day
        close = 100.0 * (0.995 ** np.arange(n))
        data = pd.DataFrame(
            {"close": close, "open": close, "high": close, "low": close,
             "volume": np.ones(n) * 1e6},
            index=dates,
        )
        signal = TrendFollowingSignal(lookback=60).generate(data)
        assert signal.iloc[-1] < 0

    def test_stable_under_truncation(self):
        """Signal should not change when future data is removed."""
        data_full = _make_ohlcv(n=300)
        data_trunc = data_full.iloc[:150]

        sig_obj = TrendFollowingSignal(lookback=60)
        signal_full = sig_obj.generate(data_full)
        signal_trunc = sig_obj.generate(data_trunc)

        pd.testing.assert_series_equal(
            signal_full.iloc[:150],
            signal_trunc,
            check_names=False,
        )

    def test_lower_turnover_than_mean_reversion(self):
        """Trend-following should have lower turnover than daily mean-reversion."""
        data = _make_ohlcv(n=300, seed=42)

        mr_signal = MeanReversionSignal(vol_lookback=20).generate(data)
        tf_signal = TrendFollowingSignal(lookback=60).generate(data)

        mr_turnover = mr_signal.diff().abs().sum()
        tf_turnover = tf_signal.diff().abs().sum()

        assert tf_turnover < mr_turnover, (
            f"Trend turnover ({tf_turnover:.1f}) should be less than "
            f"MR turnover ({mr_turnover:.1f})"
        )

    def test_different_lookback(self):
        data = _make_ohlcv(n=300)
        sig_60 = TrendFollowingSignal(lookback=60).generate(data)
        sig_120 = TrendFollowingSignal(lookback=120).generate(data)
        assert not sig_60.equals(sig_120)

    def test_repr(self):
        sig = TrendFollowingSignal(lookback=90, scale=3.0)
        r = repr(sig)
        assert "90" in r
        assert "3.0" in r


# ── Smoothed Mean-Reversion tests ────────────────────────────────────────────

from quant_research.strategies.mean_reversion_smooth import SmoothedMeanReversionSignal


class TestSmoothedMeanReversionSignal:
    def test_output_is_series(self):
        data = _make_ohlcv(n=100)
        signal = SmoothedMeanReversionSignal().generate(data)
        assert isinstance(signal, pd.Series)

    def test_no_nans(self):
        data = _make_ohlcv(n=100)
        signal = SmoothedMeanReversionSignal().generate(data)
        assert signal.isna().sum() == 0

    def test_clipped_to_range(self):
        data = _make_ohlcv(n=100)
        signal = SmoothedMeanReversionSignal().generate(data)
        assert signal.min() >= -1.0
        assert signal.max() <= 1.0

    def test_lower_turnover_than_raw_mr(self):
        """Smoothed MR should have lower turnover than raw MR."""
        data = _make_ohlcv(n=300, seed=42)

        raw_signal = MeanReversionSignal(vol_lookback=20).generate(data)
        smooth_signal = SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=5).generate(data)

        raw_turnover = raw_signal.diff().abs().sum()
        smooth_turnover = smooth_signal.diff().abs().sum()

        assert smooth_turnover < raw_turnover, (
            f"Smooth turnover ({smooth_turnover:.1f}) should be less than "
            f"raw turnover ({raw_turnover:.1f})"
        )

    def test_stable_under_truncation(self):
        """No lookahead — truncation test."""
        data_full = _make_ohlcv(n=200)
        data_trunc = data_full.iloc[:100]

        sig_obj = SmoothedMeanReversionSignal()
        signal_full = sig_obj.generate(data_full)
        signal_trunc = sig_obj.generate(data_trunc)

        pd.testing.assert_series_equal(
            signal_full.iloc[:100], signal_trunc, check_names=False,
        )

    def test_higher_halflife_lower_turnover(self):
        """Increasing EMA halflife should reduce turnover further."""
        data = _make_ohlcv(n=300, seed=42)

        sig_5 = SmoothedMeanReversionSignal(ema_halflife=5).generate(data)
        sig_20 = SmoothedMeanReversionSignal(ema_halflife=20).generate(data)

        to_5 = sig_5.diff().abs().sum()
        to_20 = sig_20.diff().abs().sum()

        assert to_20 < to_5

    def test_repr(self):
        sig = SmoothedMeanReversionSignal(vol_lookback=30, ema_halflife=10)
        r = repr(sig)
        assert "30" in r
        assert "10" in r


# ── Ensemble Signal tests ────────────────────────────────────────────────────

from quant_research.strategies.ensemble import EnsembleSignal


class TestEnsembleSignal:
    def test_requires_at_least_2_signals(self):
        with pytest.raises(ValueError, match="at least 2"):
            EnsembleSignal([MeanReversionSignal()])

    def test_output_is_series(self):
        data = _make_ohlcv(n=200)
        ensemble = EnsembleSignal([
            SmoothedMeanReversionSignal(),
            TrendFollowingSignal(lookback=60),
        ])
        signal = ensemble.generate(data)
        assert isinstance(signal, pd.Series)

    def test_output_same_length(self):
        data = _make_ohlcv(n=200)
        ensemble = EnsembleSignal([
            SmoothedMeanReversionSignal(),
            TrendFollowingSignal(lookback=60),
        ])
        signal = ensemble.generate(data)
        assert len(signal) == len(data)

    def test_no_nans(self):
        data = _make_ohlcv(n=200)
        ensemble = EnsembleSignal([
            SmoothedMeanReversionSignal(),
            TrendFollowingSignal(lookback=60),
        ])
        signal = ensemble.generate(data)
        assert signal.isna().sum() == 0

    def test_clipped_to_range(self):
        data = _make_ohlcv(n=200)
        ensemble = EnsembleSignal([
            SmoothedMeanReversionSignal(),
            TrendFollowingSignal(lookback=60),
        ])
        signal = ensemble.generate(data)
        assert signal.min() >= -1.0
        assert signal.max() <= 1.0

    def test_equal_weight_by_default(self):
        ensemble = EnsembleSignal([
            MeanReversionSignal(),
            TrendFollowingSignal(),
        ])
        assert ensemble.weights == pytest.approx([0.5, 0.5])

    def test_custom_weights_normalised(self):
        ensemble = EnsembleSignal(
            [MeanReversionSignal(), TrendFollowingSignal()],
            weights=[3.0, 1.0],
        )
        assert ensemble.weights == pytest.approx([0.75, 0.25])

    def test_weight_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            EnsembleSignal(
                [MeanReversionSignal(), TrendFollowingSignal()],
                weights=[1.0],
            )

    def test_stable_under_truncation(self):
        data_full = _make_ohlcv(n=300)
        data_trunc = data_full.iloc[:150]

        ensemble = EnsembleSignal([
            SmoothedMeanReversionSignal(),
            TrendFollowingSignal(lookback=60),
        ])
        signal_full = ensemble.generate(data_full)
        signal_trunc = ensemble.generate(data_trunc)

        pd.testing.assert_series_equal(
            signal_full.iloc[:150], signal_trunc, check_names=False,
        )

    def test_repr(self):
        ensemble = EnsembleSignal([
            MeanReversionSignal(),
            TrendFollowingSignal(),
        ])
        r = repr(ensemble)
        assert "Ensemble" in r
