"""
quant_research.strategies.vrp_signal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Strategy: Variance Risk Premium (VRP) Signal.

Hypothesis
----------
Implied volatility (VIX) is almost always higher than realized
volatility. This gap is called the Variance Risk Premium (VRP).

When VRP is abnormally HIGH:
  → Market is "too fearful" relative to actual risk
  → Expected equity returns are high → Long

When VRP is abnormally LOW or NEGATIVE:
  → Market is complacent or actual risk exceeds expectations
  → Expected equity returns are poor → Reduce/Exit

Signal construction
-------------------
  implied_vol[t] = VIX_close[t] / 100
  realized_vol[t] = rolling_std(SPY_returns, realized_window) × √252
  vrp[t] = implied_vol[t] - realized_vol[t]
  vrp_z[t] = z_score(vrp, z_lookback)
  signal[t] = clip(vrp_z / scale, -1, +1)

Parameters
----------
  realized_window : int = 21   (1 month for realized vol)
  z_lookback : int = 63        (3 months for z-score normalisation)
  scale : float = 2.0
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research.signals.base import BaseSignal


class VarianceRiskPremiumSignal(BaseSignal):
    """Variance Risk Premium signal.

    Exploits the gap between implied (VIX) and realized volatility
    to time equity market exposure.

    VIX data is injected via constructor for BaseSignal compatibility.

    Parameters
    ----------
    vix_data : pd.DataFrame
        VIX OHLCV data (only 'close' used). VIX is in index points
        (e.g., 15 = 15% annualised implied vol).
    realized_window : int
        Window for realized vol estimation. Default: 21 (~1 month).
    z_lookback : int
        Lookback for z-score normalisation. Default: 63 (~3 months).
    scale : float
        Z-score scaling. Default: 2.0.
    """

    def __init__(
        self,
        vix_data: pd.DataFrame,
        realized_window: int = 21,
        z_lookback: int = 63,
        scale: float = 2.0,
    ):
        self.vix_data = vix_data
        self.realized_window = realized_window
        self.z_lookback = z_lookback
        self.scale = scale

    def generate(self, data: pd.DataFrame) -> pd.Series:
        """Compute VRP signal.

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV data for the TARGET equity (SPY/QQQ).

        Returns
        -------
        pd.Series
            Signal ∈ [-1, +1]. Positive = VRP high → long equity.
        """
        # Implied vol from VIX (convert from index points to decimal)
        vix_close = self.vix_data["close"]
        implied_vol = vix_close / 100.0

        # Realized vol from target equity
        equity_ret = data["close"].pct_change()
        realized_vol = equity_ret.rolling(
            window=self.realized_window, min_periods=self.realized_window
        ).std() * np.sqrt(252)

        # VRP = implied - realized
        # Align indices
        implied_aligned = implied_vol.reindex(data.index)
        vrp = implied_aligned - realized_vol

        # Z-score (rolling, no lookahead)
        vrp_mean = vrp.rolling(self.z_lookback, min_periods=self.z_lookback).mean()
        vrp_std = vrp.rolling(self.z_lookback, min_periods=self.z_lookback).std()
        vrp_z = (vrp - vrp_mean) / vrp_std.replace(0, np.nan)

        # Signal: high VRP → long (positive signal)
        raw_signal = vrp_z / self.scale
        signal = raw_signal.clip(-1, 1).fillna(0.0)

        return signal

    def __repr__(self) -> str:
        return (
            f"VarianceRiskPremiumSignal(real_w={self.realized_window}, "
            f"z_lb={self.z_lookback}, scale={self.scale})"
        )
