"""
quant_research.data.loader
~~~~~~~~~~~~~~~~~~~~~~~~~~
Data loading, validation, and cleaning utilities.

Design principles:
  • Raw data is never modified in place.
  • All validation checks are explicit and logged.
  • The loader returns clean DataFrames with a DatetimeIndex.
  • Column names are normalised to lowercase.
  • Adjusted close is used as the canonical "close" to account for
    splits and dividends (equities/ETFs only).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ── Canonical column names after cleaning ────────────────────────────────────
REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}


def load_config(config_path: Optional[str | Path] = None) -> dict:
    """Load the YAML configuration file.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to config.yaml.  Defaults to ``config/config.yaml`` relative
        to the project root (two levels up from this file).
    """
    if config_path is None:
        config_path = (
            Path(__file__).resolve().parents[3] / "config" / "config.yaml"
        )
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_all_tickers(config: dict) -> list[dict]:
    """Extract a flat list of ticker info dicts from the universe config.

    Returns
    -------
    list[dict]
        Each dict has keys: ticker, name, asset_class, cost_bps.
    """
    tickers = []
    seen = set()
    universe = config.get("universe", {})
    for group_name, items in universe.items():
        if not isinstance(items, list):
            continue
        for item in items:
            t = item.get("ticker")
            if t and t not in seen:
                tickers.append(item)
                seen.add(t)
    return tickers


# ── Loading ──────────────────────────────────────────────────────────────────


def load_raw(
    ticker: str,
    raw_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """Load a single raw CSV file as downloaded by ``download_data.py``.

    Parameters
    ----------
    ticker : str
        Ticker symbol (e.g. "SPY", "BTC-USD").
    raw_dir : str or Path
        Directory containing raw CSV files.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with a DatetimeIndex named ``"Date"``.
    """
    raw_dir = Path(raw_dir)
    # Sanitise ticker for filename (e.g. "BTC-USD" → "BTC-USD.csv")
    filepath = raw_dir / f"{ticker}.csv"
    if not filepath.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {filepath}.  "
            f"Run 'python scripts/download_data.py' first."
        )

    df = pd.read_csv(filepath, parse_dates=["Date"], index_col="Date")
    df.index.name = "Date"
    logger.info(
        "Loaded %s: %d rows, %s to %s",
        ticker,
        len(df),
        df.index.min().date(),
        df.index.max().date(),
    )
    return df


# ── Validation ───────────────────────────────────────────────────────────────


class DataValidationError(Exception):
    """Raised when data fails a critical validation check."""


def validate(
    df: pd.DataFrame,
    ticker: str,
    asset_class: str = "equity",
) -> list[str]:
    """Run validation checks on a raw OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data with DatetimeIndex.
    ticker : str
        Ticker symbol (for logging).
    asset_class : str
        One of "equity", "crypto", "futures".

    Returns
    -------
    list[str]
        A list of warning messages (non-fatal issues).

    Raises
    ------
    DataValidationError
        If a critical check fails (e.g., no data at all).
    """
    warnings_list: list[str] = []

    # 1. Non-empty
    if len(df) == 0:
        raise DataValidationError(f"{ticker}: DataFrame is empty.")

    # 2. Index is DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataValidationError(
            f"{ticker}: Index is {type(df.index).__name__}, expected DatetimeIndex."
        )

    # 3. Check for duplicate dates
    dup_count = df.index.duplicated().sum()
    if dup_count > 0:
        warnings_list.append(
            f"{ticker}: {dup_count} duplicate date(s) found — will keep last."
        )

    # 4. Check for required columns (case-insensitive)
    cols_lower = {c.lower() for c in df.columns}
    # Yahoo Finance provides "Adj Close" which we'll map to "close"
    effective_cols = cols_lower | ({"close"} if "adj close" in cols_lower else set())
    missing = REQUIRED_COLUMNS - effective_cols
    if missing:
        raise DataValidationError(
            f"{ticker}: Missing columns {missing}. Available: {list(df.columns)}"
        )

    # 5. Check for NaN in critical price columns
    price_cols = [c for c in df.columns if c.lower() in {"open", "high", "low", "close", "adj close"}]
    nan_counts = df[price_cols].isna().sum()
    total_nans = nan_counts.sum()
    if total_nans > 0:
        pct = total_nans / (len(df) * len(price_cols)) * 100
        warnings_list.append(
            f"{ticker}: {total_nans} NaN(s) in price columns ({pct:.2f}%)."
        )

    # 6. Check for non-positive prices
    for col in price_cols:
        non_positive = (df[col] <= 0).sum()
        if non_positive > 0:
            warnings_list.append(
                f"{ticker}: {non_positive} non-positive values in '{col}'."
            )

    # 7. Check for suspicious gaps (equity: >5 trading days; crypto: >2 days)
    date_diffs = df.index.to_series().diff().dropna()
    if asset_class == "crypto":
        gap_threshold = pd.Timedelta(days=3)
    else:
        gap_threshold = pd.Timedelta(days=7)  # Allow weekends + holidays

    large_gaps = date_diffs[date_diffs > gap_threshold]
    if len(large_gaps) > 0:
        for gap_date, gap_size in large_gaps.items():
            warnings_list.append(
                f"{ticker}: Gap of {gap_size.days} day(s) ending {gap_date.date()}."
            )

    # 8. Check chronological order
    if not df.index.is_monotonic_increasing:
        warnings_list.append(f"{ticker}: Index is not monotonically increasing.")

    for w in warnings_list:
        logger.warning(w)

    return warnings_list


# ── Cleaning ─────────────────────────────────────────────────────────────────


def clean(
    df: pd.DataFrame,
    ticker: str,
    asset_class: str = "equity",
) -> pd.DataFrame:
    """Clean and normalise a raw OHLCV DataFrame.

    Steps:
      1. Remove duplicate dates (keep last).
      2. Sort by date.
      3. Normalise column names to lowercase.
      4. Use Adjusted Close as canonical "close" (for equities/ETFs).
      5. Drop rows where all price columns are NaN.
      6. Forward-fill small gaps (≤ 3 days for equity, ≤ 1 day for crypto).
      7. Select only the canonical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame (output of ``load_raw``).
    ticker : str
        For logging.
    asset_class : str
        "equity", "crypto", or "futures".

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns: open, high, low, close, volume.
    """
    df = df.copy()

    # 1. Remove duplicate dates
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="last")]
        logger.info("%s: Removed duplicate dates.", ticker)

    # 2. Sort chronologically
    df = df.sort_index()

    # 3. Normalise column names
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # 4. Use adjusted close as canonical close (accounts for splits/dividends)
    if "adj_close" in df.columns:
        # Calculate adjustment ratio
        if "close" in df.columns:
            adj_ratio = df["adj_close"] / df["close"]
            # Adjust OHLC by the same ratio for consistency
            for col in ["open", "high", "low"]:
                if col in df.columns:
                    df[col] = df[col] * adj_ratio
            df["close"] = df["adj_close"]
        df = df.drop(columns=["adj_close"], errors="ignore")
        logger.info("%s: Applied split/dividend adjustment via Adj Close.", ticker)

    # 5. Drop rows where all prices are NaN
    price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    df = df.dropna(subset=price_cols, how="all")

    # 6. Forward-fill small gaps
    max_fill = 3 if asset_class in ("equity", "futures") else 1
    df[price_cols] = df[price_cols].ffill(limit=max_fill)
    # Volume: fill NaN with 0 (no trading = zero volume)
    if "volume" in df.columns:
        df["volume"] = df["volume"].fillna(0)

    # 7. Select canonical columns
    canonical = ["open", "high", "low", "close", "volume"]
    available = [c for c in canonical if c in df.columns]
    df = df[available]

    # 8. Drop any remaining rows with NaN in price columns
    df = df.dropna(subset=[c for c in ["open", "high", "low", "close"] if c in df.columns])

    logger.info(
        "%s: Cleaned → %d rows, %s to %s",
        ticker,
        len(df),
        df.index.min().date(),
        df.index.max().date(),
    )
    return df


# ── Convenience: full pipeline ───────────────────────────────────────────────


def load_and_clean(
    ticker: str,
    asset_class: str = "equity",
    raw_dir: str | Path = "data/raw",
) -> tuple[pd.DataFrame, list[str]]:
    """Load, validate, and clean a single ticker.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        (cleaned DataFrame, list of validation warnings)
    """
    raw_df = load_raw(ticker, raw_dir=raw_dir)
    warnings_list = validate(raw_df, ticker, asset_class=asset_class)
    clean_df = clean(raw_df, ticker, asset_class=asset_class)
    return clean_df, warnings_list


def load_universe(
    config: Optional[dict] = None,
    config_path: Optional[str | Path] = None,
    raw_dir: Optional[str | Path] = None,
) -> dict[str, pd.DataFrame]:
    """Load and clean all tickers defined in the config universe.

    Parameters
    ----------
    config : dict, optional
        Pre-loaded config dict.
    config_path : str or Path, optional
        Path to config.yaml (used if config is None).
    raw_dir : str or Path, optional
        Override for raw data directory.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of ticker → cleaned DataFrame.
    """
    if config is None:
        config = load_config(config_path)

    if raw_dir is None:
        raw_dir = config["data"]["raw_dir"]

    all_tickers = get_all_tickers(config)
    universe: dict[str, pd.DataFrame] = {}

    for info in all_tickers:
        ticker = info["ticker"]
        asset_class = info["asset_class"]
        try:
            df, warnings = load_and_clean(
                ticker, asset_class=asset_class, raw_dir=raw_dir
            )
            universe[ticker] = df
            if warnings:
                logger.warning(
                    "%s: %d validation warning(s).", ticker, len(warnings)
                )
        except (FileNotFoundError, DataValidationError) as e:
            logger.error("Skipping %s: %s", ticker, e)

    logger.info("Universe loaded: %d / %d tickers.", len(universe), len(all_tickers))
    return universe


def get_split_dates(config: Optional[dict] = None) -> dict[str, tuple[str, str]]:
    """Return the data split boundaries from config.

    Returns
    -------
    dict
        Keys: "in_sample", "validation", "out_of_sample".
        Values: (start_date, end_date) as strings.
    """
    if config is None:
        config = load_config()
    splits = config["splits"]
    return {
        name: (split["start"], split["end"])
        for name, split in splits.items()
    }


def split_data(
    df: pd.DataFrame,
    config: Optional[dict] = None,
) -> dict[str, pd.DataFrame]:
    """Split a DataFrame into in-sample, validation, and out-of-sample.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex.
    config : dict, optional
        Config dict.  If None, loads from default path.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: "in_sample", "validation", "out_of_sample".
    """
    dates = get_split_dates(config)
    result = {}
    for name, (start, end) in dates.items():
        mask = (df.index >= start) & (df.index <= end)
        result[name] = df.loc[mask].copy()
    return result
