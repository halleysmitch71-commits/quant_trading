"""
quant_research.strategies.sector_rotation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Strategy: Cross-Sectional Sector Momentum (Sector Rotation).

Hypothesis
----------
Sector-level momentum explains >50% of individual stock momentum
(Moskowitz & Grinblatt 1999). Sectors that have outperformed over
the past 12 months (excluding the most recent month) tend to
continue outperforming, while lagging sectors tend to continue
underperforming.

This is fundamentally DIFFERENT from the existing Mean Reversion:
  • MR looks at a SINGLE asset and bets on price reversal
  • Sector Rotation looks ACROSS 9 sectors and bets on relative continuation

Signal construction
-------------------
For each sector i at time t:

  1. mom[i,t] = close[i,t-skip] / close[i,t-lookback] - 1
     (12-month return excluding most recent month)

  2. rank[i,t] = cross-sectional rank of mom[i,t] among all sectors

  3. signal[i,t] = +1 if rank in top n_long
                   -1 if rank in bottom n_short
                    0 otherwise

The skip-recent-month is critical: Jegadeesh & Titman (1993) showed
that the most recent month exhibits SHORT-TERM REVERSAL, not momentum.
Including it would contaminate the momentum signal.

Parameters
----------
  momentum_window : int = 252  (12 months of trading days)
  skip_recent : int = 21       (skip most recent month)
  n_long : int = 3             (long top 3 sectors)
  n_short : int = 3            (short bottom 3 sectors)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from quant_research.signals.base import BaseSignal


class SectorRotationSignal(BaseSignal):
    """Cross-sectional sector momentum signal.

    Unlike other signals that operate on a single asset's DataFrame,
    this signal requires a dictionary of DataFrames (one per sector)
    and produces signals for all sectors simultaneously.

    For compatibility with the existing BaseSignal interface, this class
    can be instantiated for a SPECIFIC sector. The full sector universe
    data is injected via the constructor, and generate() returns the
    signal for the target sector only.

    Parameters
    ----------
    sector_data : dict[str, pd.DataFrame]
        Ticker → OHLCV DataFrame for ALL sectors in the universe.
    target_sector : str
        Which sector this signal instance produces output for.
    momentum_window : int
        Total lookback for momentum calculation. Default: 252 (~12 months).
    skip_recent : int
        Skip the most recent N days to avoid short-term reversal.
        Default: 21 (~1 month).
    n_long : int
        Number of top-ranked sectors to go long. Default: 3.
    n_short : int
        Number of bottom-ranked sectors to go short. Default: 3.
    """

    def __init__(
        self,
        sector_data: dict[str, pd.DataFrame],
        target_sector: str,
        momentum_window: int = 252,
        skip_recent: int = 21,
        n_long: int = 3,
        n_short: int = 3,
    ):
        self.sector_data = sector_data
        self.target_sector = target_sector
        self.momentum_window = momentum_window
        self.skip_recent = skip_recent
        self.n_long = n_long
        self.n_short = n_short

    def _compute_all_signals(self) -> pd.DataFrame:
        """Compute momentum signals for all sectors.

        Returns DataFrame with sectors as columns, dates as index,
        values ∈ {-1, 0, +1}.
        """
        # Build close price matrix: dates × sectors
        closes = {}
        for ticker, df in self.sector_data.items():
            closes[ticker] = df["close"]
        close_matrix = pd.DataFrame(closes)
        close_matrix = close_matrix.dropna(how="all")

        # Momentum: return from (t - momentum_window) to (t - skip_recent)
        # This avoids short-term reversal contamination
        lagged_close = close_matrix.shift(self.skip_recent)
        past_close = close_matrix.shift(self.momentum_window)
        momentum = lagged_close / past_close - 1

        # Cross-sectional rank at each date (ascending: 1 = worst, N = best)
        ranks = momentum.rank(axis=1, method="average", ascending=True)
        n_sectors = ranks.shape[1]

        # Signal: +1 for top n_long, -1 for bottom n_short, 0 otherwise
        signals = pd.DataFrame(0.0, index=ranks.index, columns=ranks.columns)

        top_threshold = n_sectors - self.n_long + 0.5  # rank > this → long
        bottom_threshold = self.n_short + 0.5          # rank < this → short

        signals[ranks >= top_threshold] = 1.0
        signals[ranks <= bottom_threshold] = -1.0

        # NaN during warmup
        warmup_mask = momentum.isna()
        signals[warmup_mask] = 0.0

        return signals

    def generate(self, data: pd.DataFrame) -> pd.Series:
        """Compute sector rotation signal for the target sector.

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV data for the target sector (used for index alignment).

        Returns
        -------
        pd.Series
            Signal values ∈ {-1, 0, +1} for the target sector.
        """
        all_signals = self._compute_all_signals()

        if self.target_sector not in all_signals.columns:
            return pd.Series(0.0, index=data.index)

        signal = all_signals[self.target_sector]

        # Align to the input data's index
        signal = signal.reindex(data.index).fillna(0.0)

        return signal

    def generate_all(self) -> pd.DataFrame:
        """Compute signals for ALL sectors at once.

        Returns
        -------
        pd.DataFrame
            Sectors as columns, signal values ∈ {-1, 0, +1}.
        """
        return self._compute_all_signals()

    def __repr__(self) -> str:
        return (
            f"SectorRotationSignal(target={self.target_sector}, "
            f"window={self.momentum_window}, skip={self.skip_recent}, "
            f"long={self.n_long}, short={self.n_short})"
        )
