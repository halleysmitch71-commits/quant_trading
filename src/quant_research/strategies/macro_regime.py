"""
quant_research.strategies.macro_regime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Strategy: Cross-Asset Macro Regime Signal (Risk-On / Risk-Off).

Hypothesis
----------
Equity returns are strongly conditional on the macro regime:
  • Risk-On: VIX low, bonds weak, USD weak → equities trend up
  • Risk-Off: VIX high, bonds strong (flight to safety), USD strong → equities suffer

By combining signals from VIX, Treasury bonds (TLT), and USD (UUP),
we can build a regime indicator that tells us WHEN to be long equity
and when to reduce/exit.

This is fundamentally different from looking at equity price alone:
  • MR/TF on SPY only → single-asset, single-dimension
  • Macro Regime → multi-asset, multi-dimension (vol + rates + FX)

Signal construction
-------------------
  vix_z[t]  = -z_score(VIX_close, lookback)
              (negative because high VIX = bad for equity)
  tlt_mom[t] = -TLT_return(lookback)
              (negative because strong bonds = risk-off)
  uup_mom[t] = -UUP_return(lookback)
              (negative because strong USD = risk-off)

  regime_score[t] = w_vix × vix_z + w_tlt × tlt_mom_z + w_uup × uup_mom_z
  signal[t] = clip(regime_score / scale, -1, +1)

All z-scores are computed with EXPANDING or ROLLING windows — no lookahead.

Parameters
----------
  vix_lookback : int = 63   (3 months for VIX z-score)
  mom_lookback : int = 60   (3 months for bond/USD momentum)
  vix_weight : float = 0.50
  bond_weight : float = 0.25
  usd_weight : float = 0.25
  scale : float = 2.0
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research.signals.base import BaseSignal


class MacroRegimeSignal(BaseSignal):
    """Cross-asset macro regime signal for equity timing.

    Combines VIX level, bond momentum, and USD momentum into
    a single Risk-On / Risk-Off indicator.

    The indicator data (VIX, TLT, UUP) is injected via the constructor,
    keeping the generate() interface compatible with BaseSignal.

    Parameters
    ----------
    vix_data : pd.DataFrame
        VIX OHLCV data (only 'close' used).
    tlt_data : pd.DataFrame
        TLT (Treasury bond ETF) OHLCV data.
    uup_data : pd.DataFrame
        UUP (US Dollar ETF) OHLCV data.
    vix_lookback : int
        Rolling window for VIX z-score. Default: 63.
    mom_lookback : int
        Lookback for bond/USD momentum. Default: 60.
    vix_weight : float
        Weight for VIX component. Default: 0.50.
    bond_weight : float
        Weight for bond component. Default: 0.25.
    uup_weight : float
        Weight for USD component. Default: 0.25.
    scale : float
        Z-score scaling for final signal. Default: 2.0.
    """

    def __init__(
        self,
        vix_data: pd.DataFrame,
        tlt_data: pd.DataFrame,
        uup_data: pd.DataFrame,
        vix_lookback: int = 63,
        mom_lookback: int = 60,
        vix_weight: float = 0.50,
        bond_weight: float = 0.25,
        uup_weight: float = 0.25,
        scale: float = 2.0,
    ):
        self.vix_data = vix_data
        self.tlt_data = tlt_data
        self.uup_data = uup_data
        self.vix_lookback = vix_lookback
        self.mom_lookback = mom_lookback
        self.vix_weight = vix_weight
        self.bond_weight = bond_weight
        self.uup_weight = uup_weight
        self.scale = scale

    def _z_score_rolling(self, series: pd.Series, window: int) -> pd.Series:
        """Compute rolling z-score (no lookahead)."""
        mean = series.rolling(window, min_periods=window).mean()
        std = series.rolling(window, min_periods=window).std()
        z = (series - mean) / std.replace(0, np.nan)
        return z.fillna(0.0)

    def generate(self, data: pd.DataFrame) -> pd.Series:
        """Compute macro regime signal.

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV data for the TARGET equity asset (SPY/QQQ).
            Used for index alignment.

        Returns
        -------
        pd.Series
            Signal ∈ [-1, +1]. Positive = Risk-On (long equity).
        """
        # Extract close prices
        vix_close = self.vix_data["close"]
        tlt_close = self.tlt_data["close"]
        uup_close = self.uup_data["close"]

        # Component 1: VIX z-score (inverted: high VIX = negative signal)
        vix_z = -self._z_score_rolling(vix_close, self.vix_lookback)

        # Component 2: Bond momentum (inverted: strong bonds = risk-off)
        tlt_ret = tlt_close.pct_change(self.mom_lookback)
        tlt_z = -self._z_score_rolling(tlt_ret, self.vix_lookback)

        # Component 3: USD momentum (inverted: strong USD = risk-off)
        uup_ret = uup_close.pct_change(self.mom_lookback)
        uup_z = -self._z_score_rolling(uup_ret, self.vix_lookback)

        # Combine with weights
        regime_score = (
            self.vix_weight * vix_z +
            self.bond_weight * tlt_z +
            self.uup_weight * uup_z
        )

        # Scale and clip
        raw_signal = regime_score / self.scale
        signal = raw_signal.clip(-1, 1).fillna(0.0)

        # Align to the target equity's index
        signal = signal.reindex(data.index).fillna(0.0)

        return signal

    def __repr__(self) -> str:
        return (
            f"MacroRegimeSignal(vix_lb={self.vix_lookback}, mom_lb={self.mom_lookback}, "
            f"w=[{self.vix_weight},{self.bond_weight},{self.uup_weight}], "
            f"scale={self.scale})"
        )
