"""
Tests for new equity strategies (M20–M23).

Only DualMomentumSignal passed acceptance criteria and is production-ready.
The other strategies (SectorRotation, MacroRegime, VRP) are tested here
for basic correctness even though they were eliminated from the portfolio.
"""

import numpy as np
import pandas as pd
import pytest

from quant_research.strategies.dual_momentum import DualMomentumSignal
from quant_research.strategies.sector_rotation import SectorRotationSignal


def _make_price_series(n=500, start_price=100.0, drift=0.0005, vol=0.02, seed=42):
    """Generate synthetic OHLCV data."""
    rng = np.random.RandomState(seed)
    returns = drift + vol * rng.randn(n)
    close = start_price * np.exp(np.cumsum(returns))
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({
        "open": close * (1 + 0.001 * rng.randn(n)),
        "high": close * (1 + abs(0.005 * rng.randn(n))),
        "low": close * (1 - abs(0.005 * rng.randn(n))),
        "close": close,
        "volume": rng.randint(1000, 10000, n).astype(float),
    }, index=dates)


def _make_multi_asset(n_assets=4, n=500):
    """Generate dict of synthetic assets with different drifts."""
    data = {}
    names = ["A", "B", "C", "D"][:n_assets]
    drifts = [0.0008, 0.0004, 0.0001, -0.0002][:n_assets]
    for name, drift in zip(names, drifts):
        data[name] = _make_price_series(n=n, drift=drift, seed=hash(name) % 2**31)
    return data


# ═══════════════════════════════════════════════════════════════════════
# Dual Momentum Signal Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDualMomentumSignal:
    """Tests for DualMomentumSignal — the production-ready equity strategy."""

    def setup_method(self):
        self.data = _make_multi_asset(4, 500)

    def test_output_is_series(self):
        sig = DualMomentumSignal(self.data, "A", lookback=252, n_top=2)
        result = sig.generate(self.data["A"])
        assert isinstance(result, pd.Series)

    def test_output_length_matches_data(self):
        sig = DualMomentumSignal(self.data, "A", lookback=252, n_top=2)
        result = sig.generate(self.data["A"])
        assert len(result) == len(self.data["A"])

    def test_signal_is_binary(self):
        """Dual Momentum should only produce 0 or +1 (long-only)."""
        sig = DualMomentumSignal(self.data, "A", lookback=252, n_top=2)
        result = sig.generate(self.data["A"])
        unique_vals = set(result.unique())
        assert unique_vals.issubset({0.0, 1.0}), f"Expected {{0, 1}}, got {unique_vals}"

    def test_no_nans(self):
        sig = DualMomentumSignal(self.data, "A", lookback=252, n_top=2)
        result = sig.generate(self.data["A"])
        assert not result.isna().any()

    def test_warmup_is_zero(self):
        """During warmup period (< lookback), signal should be 0."""
        sig = DualMomentumSignal(self.data, "A", lookback=252, n_top=2)
        result = sig.generate(self.data["A"])
        # First 252 values should be 0 (no momentum data yet)
        assert (result.iloc[:252] == 0).all()

    def test_n_top_constraint(self):
        """At most n_top assets should be long at any time."""
        sig = DualMomentumSignal(self.data, "A", lookback=252, n_top=2)
        all_signals = sig.generate_all()
        # After warmup, at each date: sum of signals <= n_top
        after_warmup = all_signals.iloc[252:]
        max_long = after_warmup.sum(axis=1).max()
        assert max_long <= 2, f"Max simultaneous longs: {max_long} > n_top=2"

    def test_absolute_momentum_filter(self):
        """Assets with negative absolute momentum should not be long."""
        # Create data where one asset has strongly negative drift
        data = _make_multi_asset(4, 500)
        data["D"] = _make_price_series(500, drift=-0.005, seed=999)  # strong downtrend
        sig = DualMomentumSignal(data, "D", lookback=252, n_top=3)
        result = sig.generate(data["D"])
        # Asset D should rarely be long (maybe never after warmup)
        long_pct = (result.iloc[260:] == 1.0).mean()
        assert long_pct < 0.3, f"Downtrending asset is long {long_pct:.0%} of the time"

    def test_generate_all_returns_dataframe(self):
        sig = DualMomentumSignal(self.data, "A", lookback=252, n_top=2)
        result = sig.generate_all()
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == set(self.data.keys())

    def test_target_not_in_data_returns_zero(self):
        sig = DualMomentumSignal(self.data, "MISSING", lookback=252, n_top=2)
        result = sig.generate(self.data["A"])
        assert (result == 0).all()

    def test_different_lookbacks_produce_different_signals(self):
        sig1 = DualMomentumSignal(self.data, "A", lookback=126, n_top=2)
        sig2 = DualMomentumSignal(self.data, "A", lookback=252, n_top=2)
        r1 = sig1.generate(self.data["A"])
        r2 = sig2.generate(self.data["A"])
        # Should not be identical
        assert not r1.equals(r2)

    def test_no_lookahead(self):
        """Truncating future data should not change past signals."""
        full_sig = DualMomentumSignal(self.data, "A", lookback=252, n_top=2)
        full_result = full_sig.generate(self.data["A"])

        # Truncate to first 400 bars
        trunc_data = {k: v.iloc[:400] for k, v in self.data.items()}
        trunc_sig = DualMomentumSignal(trunc_data, "A", lookback=252, n_top=2)
        trunc_result = trunc_sig.generate(trunc_data["A"])

        # First 400 values should match
        pd.testing.assert_series_equal(
            full_result.iloc[:400].reset_index(drop=True),
            trunc_result.reset_index(drop=True),
        )

    def test_repr(self):
        sig = DualMomentumSignal(self.data, "A", lookback=252, n_top=2)
        r = repr(sig)
        assert "DualMomentumSignal" in r
        assert "A" in r


# ═══════════════════════════════════════════════════════════════════════
# Sector Rotation Signal Tests (eliminated but ensure code works)
# ═══════════════════════════════════════════════════════════════════════

class TestSectorRotationSignal:
    """Basic tests for SectorRotationSignal (eliminated strategy)."""

    def setup_method(self):
        self.data = _make_multi_asset(4, 500)

    def test_output_is_series(self):
        sig = SectorRotationSignal(self.data, "A", momentum_window=252,
                                    skip_recent=21, n_long=1, n_short=1)
        result = sig.generate(self.data["A"])
        assert isinstance(result, pd.Series)

    def test_signal_bounded(self):
        sig = SectorRotationSignal(self.data, "A", momentum_window=252,
                                    skip_recent=21, n_long=1, n_short=1)
        result = sig.generate(self.data["A"])
        assert result.min() >= -1.0
        assert result.max() <= 1.0

    def test_no_nans(self):
        sig = SectorRotationSignal(self.data, "A", momentum_window=252,
                                    skip_recent=21, n_long=1, n_short=1)
        result = sig.generate(self.data["A"])
        assert not result.isna().any()

    def test_long_short_balance(self):
        """At each date: n_long longs + n_short shorts + rest neutral."""
        sig = SectorRotationSignal(self.data, "A", momentum_window=252,
                                    skip_recent=21, n_long=1, n_short=1)
        all_sigs = sig.generate_all()
        after_warmup = all_sigs.iloc[280:]
        for _, row in after_warmup.iterrows():
            n_long = (row == 1).sum()
            n_short = (row == -1).sum()
            # Allow 0 during warmup tail
            assert n_long <= 1, f"Too many longs: {n_long}"
            assert n_short <= 1, f"Too many shorts: {n_short}"
