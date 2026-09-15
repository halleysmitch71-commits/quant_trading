"""
download_data.py — Download raw OHLCV data for the research universe.

Usage:
    python scripts/download_data.py

This script downloads daily OHLCV data from Yahoo Finance for every
ticker defined in config/config.yaml and saves each as a CSV file
in data/raw/.

Notes:
  • Yahoo Finance data is free but has known limitations:
    - Survivorship bias for individual stocks (we mitigate by using ETFs).
    - Adjusted close may be recalculated retroactively.
    - Futures continuous-contract quality is variable.
  • Each run overwrites existing files to ensure freshness.
  • The script is idempotent: safe to re-run at any time.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure the project root is on the path so we can import quant_research
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import yfinance as yf  # noqa: E402

from quant_research.data.loader import get_all_tickers, load_config  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def download_ticker(
    ticker: str,
    start: str,
    end: str,
    raw_dir: Path,
) -> bool:
    """Download a single ticker and save to CSV.

    Returns True on success, False on failure.
    """
    logger.info("Downloading %s  (%s → %s) ...", ticker, start, end)
    try:
        data = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,   # Keep both Close and Adj Close
            progress=False,
        )
        if data is None or len(data) == 0:
            logger.error("  ✗  %s: No data returned.", ticker)
            return False

        # yfinance sometimes returns MultiIndex columns for single ticker
        if isinstance(data.columns, __import__('pandas').MultiIndex):
            data.columns = data.columns.get_level_values(0)

        filepath = raw_dir / f"{ticker}.csv"
        data.to_csv(filepath)
        logger.info(
            "  ✓  %s: %d rows saved to %s",
            ticker,
            len(data),
            filepath,
        )
        return True

    except Exception as e:
        logger.error("  ✗  %s: Download failed — %s", ticker, e)
        return False


def main() -> None:
    config = load_config()

    raw_dir = Path(config["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    start = config["data"]["start_date"]
    end = config["data"]["end_date"]
    tickers = get_all_tickers(config)

    logger.info("=" * 60)
    logger.info("Downloading %d tickers: %s → %s", len(tickers), start, end)
    logger.info("Output directory: %s", raw_dir.resolve())
    logger.info("=" * 60)

    success = 0
    failed = 0

    for info in tickers:
        ticker = info["ticker"]
        if download_ticker(ticker, start, end, raw_dir):
            success += 1
        else:
            failed += 1

    logger.info("=" * 60)
    logger.info("Done.  %d succeeded, %d failed.", success, failed)
    if failed > 0:
        logger.warning("Some downloads failed — check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
