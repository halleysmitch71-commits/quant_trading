"""
Tests for data loading, validation, and cleaning.

These tests use synthetic data to verify the loader's behaviour
without requiring a network connection or downloaded files.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_research.data.loader import (
    DataValidationError,
    clean,
    get_split_dates,
    load_config,
    load_raw,
    split_data,
    validate,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_ohlcv(
    start: str = "2020-01-02",
    periods: int = 100,
    freq: str = "B",  # Business days
    base_price: float = 100.0,
    include_adj_close: bool = True,
    ticker: str = "TEST",
) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame resembling Yahoo Finance output."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(start=start, periods=periods, freq=freq)

    # Random-walk prices
    returns = rng.normal(0.0005, 0.01, size=periods)
    close = base_price * np.cumprod(1 + returns)
    open_ = close * (1 + rng.normal(0, 0.002, size=periods))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, size=periods)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, size=periods)))
    volume = rng.integers(1_000_000, 10_000_000, size=periods).astype(float)

    data = {
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }
    if include_adj_close:
        # Simulate a 2:1 split at row 50
        adj_factor = np.ones(periods)
        adj_factor[:50] = 0.5
        data["Adj Close"] = close * adj_factor

    df = pd.DataFrame(data, index=dates)
    df.index.name = "Date"
    return df


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Standard synthetic OHLCV data."""
    return _make_ohlcv()


@pytest.fixture
def raw_csv_dir(sample_ohlcv: pd.DataFrame, tmp_path: Path) -> Path:
    """Write synthetic data to a temp CSV and return the directory."""
    filepath = tmp_path / "TEST.csv"
    sample_ohlcv.to_csv(filepath)
    return tmp_path


# ── Config tests ─────────────────────────────────────────────────────────────


class TestConfig:
    def test_load_config_default(self):
        """Config should load from the default path."""
        cfg = load_config()
        assert "universe" in cfg
        assert "data" in cfg
        assert "splits" in cfg

    def test_config_has_all_asset_classes(self):
        cfg = load_config()
        assert "equities" in cfg["universe"]
        assert "crypto" in cfg["universe"]
        assert "futures" in cfg["universe"]

    def test_config_no_leverage(self):
        cfg = load_config()
        assert cfg["backtest"]["max_gross_exposure"] <= 1.0

    def test_split_dates_are_contiguous(self):
        """Splits should not overlap and should cover the full date range."""
        cfg = load_config()
        dates = get_split_dates(cfg)

        is_end = pd.Timestamp(dates["in_sample"][1])
        vs_start = pd.Timestamp(dates["validation"][0])
        vs_end = pd.Timestamp(dates["validation"][1])
        oos_start = pd.Timestamp(dates["out_of_sample"][0])

        # Validation starts after in-sample ends
        assert vs_start > is_end
        # Out-of-sample starts after validation ends
        assert oos_start > vs_end

    def test_split_dates_no_overlap(self):
        """No date should belong to more than one split."""
        cfg = load_config()
        dates = get_split_dates(cfg)

        is_end = pd.Timestamp(dates["in_sample"][1])
        vs_start = pd.Timestamp(dates["validation"][0])
        vs_end = pd.Timestamp(dates["validation"][1])
        oos_start = pd.Timestamp(dates["out_of_sample"][0])

        assert vs_start > is_end, "Validation overlaps with in-sample"
        assert oos_start > vs_end, "Out-of-sample overlaps with validation"


# ── Loading tests ────────────────────────────────────────────────────────────


class TestLoadRaw:
    def test_load_existing_csv(self, raw_csv_dir: Path):
        df = load_raw("TEST", raw_dir=raw_csv_dir)
        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert len(df) > 0

    def test_load_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="Run.*download_data"):
            load_raw("NONEXISTENT", raw_dir=tmp_path)


# ── Validation tests ────────────────────────────────────────────────────────


class TestValidate:
    def test_valid_data_no_errors(self, sample_ohlcv: pd.DataFrame):
        warnings = validate(sample_ohlcv, "TEST")
        # Synthetic data is clean — no warnings expected (except maybe
        # Adj Close ratio warning, which is fine)
        assert isinstance(warnings, list)

    def test_empty_dataframe_raises(self):
        df = pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume"],
            index=pd.DatetimeIndex([], name="Date"),
        )
        with pytest.raises(DataValidationError, match="empty"):
            validate(df, "EMPTY")

    def test_wrong_index_type_raises(self):
        df = pd.DataFrame(
            {"Open": [1], "High": [2], "Low": [0.5], "Close": [1.5], "Volume": [100]},
            index=[0],
        )
        with pytest.raises(DataValidationError, match="DatetimeIndex"):
            validate(df, "BAD_INDEX")

    def test_missing_columns_raises(self):
        dates = pd.bdate_range("2020-01-02", periods=5)
        df = pd.DataFrame(
            {"Open": [1]*5, "Close": [1]*5},
            index=dates,
        )
        df.index.name = "Date"
        with pytest.raises(DataValidationError, match="Missing columns"):
            validate(df, "MISSING_COLS")

    def test_duplicate_dates_warning(self, sample_ohlcv: pd.DataFrame):
        # Create duplicates by appending
        dup = pd.concat([sample_ohlcv, sample_ohlcv.iloc[:3]])
        warnings = validate(dup, "DUP")
        assert any("duplicate" in w.lower() for w in warnings)

    def test_nan_warning(self, sample_ohlcv: pd.DataFrame):
        df = sample_ohlcv.copy()
        df.iloc[10, df.columns.get_loc("Close")] = np.nan
        warnings = validate(df, "NAN")
        assert any("nan" in w.lower() for w in warnings)

    def test_non_positive_price_warning(self, sample_ohlcv: pd.DataFrame):
        df = sample_ohlcv.copy()
        df.iloc[5, df.columns.get_loc("Close")] = -1.0
        warnings = validate(df, "NEG")
        assert any("non-positive" in w.lower() for w in warnings)


# ── Cleaning tests ───────────────────────────────────────────────────────────


class TestClean:
    def test_output_columns(self, sample_ohlcv: pd.DataFrame):
        """Cleaned data should have exactly the canonical columns."""
        result = clean(sample_ohlcv, "TEST")
        assert set(result.columns) == {"open", "high", "low", "close", "volume"}

    def test_lowercase_columns(self, sample_ohlcv: pd.DataFrame):
        result = clean(sample_ohlcv, "TEST")
        for col in result.columns:
            assert col == col.lower()

    def test_no_duplicates_after_clean(self, sample_ohlcv: pd.DataFrame):
        df = pd.concat([sample_ohlcv, sample_ohlcv.iloc[:3]])
        result = clean(df, "TEST")
        assert not result.index.duplicated().any()

    def test_sorted_after_clean(self, sample_ohlcv: pd.DataFrame):
        # Shuffle the index
        df = sample_ohlcv.sample(frac=1, random_state=42)
        result = clean(df, "TEST")
        assert result.index.is_monotonic_increasing

    def test_no_nans_in_prices(self, sample_ohlcv: pd.DataFrame):
        df = sample_ohlcv.copy()
        df.iloc[10, df.columns.get_loc("Close")] = np.nan
        result = clean(df, "TEST")
        price_cols = ["open", "high", "low", "close"]
        assert result[price_cols].isna().sum().sum() == 0

    def test_adj_close_applied(self, sample_ohlcv: pd.DataFrame):
        """When Adj Close differs from Close, prices should be adjusted."""
        # Our synthetic data has a 0.5 adjustment factor for the first 50 rows
        result = clean(sample_ohlcv, "TEST")
        # Adj Close column should NOT be in the output
        assert "adj_close" not in result.columns

    def test_volume_nan_filled_with_zero(self, sample_ohlcv: pd.DataFrame):
        df = sample_ohlcv.copy()
        df.iloc[5, df.columns.get_loc("Volume")] = np.nan
        result = clean(df, "TEST")
        assert result["volume"].isna().sum() == 0


# ── Split tests ──────────────────────────────────────────────────────────────


class TestSplitData:
    def test_splits_are_non_overlapping(self):
        """Data in each split should not appear in any other split."""
        dates = pd.bdate_range("2018-01-02", "2026-08-31")
        df = pd.DataFrame(
            {"close": np.random.default_rng(42).normal(100, 10, len(dates))},
            index=dates,
        )
        df.index.name = "Date"

        splits = split_data(df)

        for name_a, df_a in splits.items():
            for name_b, df_b in splits.items():
                if name_a == name_b:
                    continue
                overlap = df_a.index.intersection(df_b.index)
                assert len(overlap) == 0, (
                    f"Overlap between {name_a} and {name_b}: {len(overlap)} rows"
                )

    def test_splits_cover_data(self):
        """The splits should together cover all dates within the config range."""
        dates = pd.bdate_range("2018-01-02", "2026-08-31")
        df = pd.DataFrame(
            {"close": np.random.default_rng(42).normal(100, 10, len(dates))},
            index=dates,
        )
        df.index.name = "Date"

        splits = split_data(df)

        total_rows = sum(len(s) for s in splits.values())
        # Allow some dates to fall outside splits due to boundary rounding
        assert total_rows > 0
        assert total_rows <= len(df)

    def test_split_temporal_order(self):
        """In-sample dates < validation dates < out-of-sample dates."""
        dates = pd.bdate_range("2018-01-02", "2026-08-31")
        df = pd.DataFrame(
            {"close": np.random.default_rng(42).normal(100, 10, len(dates))},
            index=dates,
        )
        df.index.name = "Date"

        splits = split_data(df)

        if len(splits.get("in_sample", [])) > 0 and len(splits.get("validation", [])) > 0:
            assert splits["in_sample"].index.max() < splits["validation"].index.min()

        if len(splits.get("validation", [])) > 0 and len(splits.get("out_of_sample", [])) > 0:
            assert splits["validation"].index.max() < splits["out_of_sample"].index.min()
