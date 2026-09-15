"""
quant_research.strategies.dual_momentum
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Strategy: Dual Momentum (Absolute + Relative).

Based on Gary Antonacci (2014) "Dual Momentum Investing".

Hypothesis
----------
Combining two types of momentum filtering produces a more robust signal:

1. Absolute Momentum (Time-Series):
   Only go long if asset return > 0 over lookback period.
   This acts as a trend filter — avoids buying in bear markets.

2. Relative Momentum (Cross-Sectional):
   Among assets with positive absolute momentum, pick the strongest.
   This concentrates capital in the best performer.

The combination is powerful because:
  • Absolute momentum avoids drawdowns (crash protection)
  • Relative momentum captures the strongest trend
  • Together they produce better risk-adjusted returns than either alone

Signal construction
-------------------
For each asset i at time t:

  abs_mom[i,t] = return(asset_i, lookback)
  rel_rank[i,t] = rank of abs_mom across all assets

  signal[i,t] = +1 if abs_mom > 0 AND rank in top n_top
                 0 otherwise (cash)

This is LONG-ONLY: no shorting. When all assets have negative
momentum, the signal is 0 for everything (100% cash).

Parameters
----------
  lookback : int = 252   (12 months)
  n_top : int = 2        (invest in top 2 assets)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from quant_research.signals.base import BaseSignal


class DualMomentumSignal(BaseSignal):
    """Dual Momentum signal (Absolute + Relative).

    Like SectorRotationSignal, this requires multi-asset data
    injected via constructor. generate() returns the signal
    for one specific target asset.

    Parameters
    ----------
    asset_data : dict[str, pd.DataFrame]
        Ticker → OHLCV DataFrame for all assets in the universe.
    target_asset : str
        Which asset this signal produces output for.
    lookback : int
        Momentum lookback in trading days. Default: 252 (~12 months).
    n_top : int
        Number of top-ranked assets to go long. Default: 2.
    """

    def __init__(
        self,
        asset_data: dict[str, pd.DataFrame],
        target_asset: str,
        lookback: int = 252,
        n_top: int = 2,
    ):
        self.asset_data = asset_data
        self.target_asset = target_asset
        self.lookback = lookback
        self.n_top = n_top

    def _compute_all_signals(self) -> pd.DataFrame:
        """Compute dual momentum signals for all assets."""
        # Build close price matrix
        closes = {}
        for ticker, df in self.asset_data.items():
            closes[ticker] = df["close"]
        close_matrix = pd.DataFrame(closes).dropna(how="all")

        # Absolute momentum: N-day return
        abs_mom = close_matrix / close_matrix.shift(self.lookback) - 1

        # Relative momentum: cross-sectional rank (ascending)
        ranks = abs_mom.rank(axis=1, method="average", ascending=True)
        n_assets = ranks.shape[1]

        # Signal: +1 only if abs_mom > 0 AND rank in top n_top
        signals = pd.DataFrame(0.0, index=close_matrix.index, columns=close_matrix.columns)

        top_threshold = n_assets - self.n_top + 0.5
        is_top = ranks >= top_threshold
        is_positive = abs_mom > 0

        signals[is_top & is_positive] = 1.0

        # NaN during warmup → 0
        signals[abs_mom.isna()] = 0.0

        return signals

    def generate(self, data: pd.DataFrame) -> pd.Series:
        """Compute dual momentum signal for the target asset."""
        all_signals = self._compute_all_signals()

        if self.target_asset not in all_signals.columns:
            return pd.Series(0.0, index=data.index)

        signal = all_signals[self.target_asset]
        signal = signal.reindex(data.index).fillna(0.0)
        return signal

    def generate_all(self) -> pd.DataFrame:
        """Compute signals for all assets."""
        return self._compute_all_signals()

    def __repr__(self) -> str:
        return (
            f"DualMomentumSignal(target={self.target_asset}, "
            f"lookback={self.lookback}, n_top={self.n_top})"
        )
