"""
quant_research.strategies.volatility_signal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Volatility Mean-Reversion Signal.

Hypothesis
----------
Realized volatility is mean-reverting.  When short-term vol is abnormally
high relative to longer-term vol, it tends to revert down.  Conversely,
when short-term vol is unusually low, a vol expansion may be imminent.

This signal exploits the well-documented volatility clustering and
mean-reversion properties of financial assets (Bollerslev 1986, Engle 1982).

Signal construction
-------------------
  short_vol[t] = rolling_std(returns, short_window)
  long_vol[t]  = rolling_std(returns, long_window)
  vol_ratio[t] = short_vol[t] / long_vol[t]
  z[t] = (vol_ratio[t] - mean(vol_ratio)) / std(vol_ratio)
  signal[t] = clip(-z[t] / scale, -1, 1)

Interpretation:
  • vol_ratio > 1 → short-term vol elevated → expect mean-reversion → reduce risk
  • vol_ratio < 1 → short-term vol suppressed → expect vol expansion → take risk

This signal is fundamentally DIFFERENT from momentum and mean-reversion:
  • Momentum uses price level → directional
  • Mean-reversion uses 1-day returns → directional
  • Vol signal uses return MAGNITUDE → non-directional, risk-based

Decorrelation with existing signals is expected because vol signals
measure a different dimension of market behaviour.

Parameters
----------
  short_window : int = 10  (2 weeks)
  long_window : int = 60   (3 months)
  z_lookback : int = 120   (6 months for z-score normalisation)
  scale : float = 2.0      (2σ → full position)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research.signals.base import BaseSignal


class VolatilityMeanReversionSignal(BaseSignal):
    """Volatility mean-reversion signal.

    Goes long when short-term vol is low relative to long-term vol
    (expecting calm conditions to persist), and reduces exposure when
    short-term vol spikes (expecting continued turbulence).

    Parameters
    ----------
    short_window : int
        Window for short-term vol estimation.  Default: 10.
    long_window : int
        Window for long-term vol estimation.  Default: 60.
    z_lookback : int
        Lookback for z-score normalisation of the vol ratio.  Default: 120.
    scale : float
        Number of z-score units for a full ±1 position.  Default: 2.0.
    """

    def __init__(
        self,
        short_window: int = 10,
        long_window: int = 60,
        z_lookback: int = 120,
        scale: float = 2.0,
    ):
        self.short_window = short_window
        self.long_window = long_window
        self.z_lookback = z_lookback
        self.scale = scale

    def generate(self, data: pd.DataFrame) -> pd.Series:
        """Compute the volatility mean-reversion signal.

        signal[t] uses only data up to and including time t (no lookahead).
        """
        close = data["close"]
        daily_ret = close.pct_change()

        # Short-term and long-term realised vol
        short_vol = daily_ret.rolling(
            window=self.short_window, min_periods=self.short_window
        ).std()
        long_vol = daily_ret.rolling(
            window=self.long_window, min_periods=self.long_window
        ).std()

        # Vol ratio: short / long
        vol_ratio = short_vol / long_vol
        vol_ratio = vol_ratio.replace([np.inf, -np.inf], np.nan)

        # Z-score of vol ratio (rolling normalisation — no lookahead)
        vol_ratio_mean = vol_ratio.rolling(
            window=self.z_lookback, min_periods=self.z_lookback
        ).mean()
        vol_ratio_std = vol_ratio.rolling(
            window=self.z_lookback, min_periods=self.z_lookback
        ).std()

        z_score = (vol_ratio - vol_ratio_mean) / vol_ratio_std
        z_score = z_score.replace([np.inf, -np.inf], 0.0)

        # Invert: high vol ratio (vol spike) → negative signal (reduce risk)
        #          low vol ratio (calm)      → positive signal (take risk)
        raw_signal = -z_score / self.scale

        signal = raw_signal.clip(-1, 1)

        return signal.fillna(0.0)

    def __repr__(self) -> str:
        return (
            f"VolatilityMeanReversionSignal("
            f"short={self.short_window}, long={self.long_window}, "
            f"z_lookback={self.z_lookback}, scale={self.scale})"
        )
