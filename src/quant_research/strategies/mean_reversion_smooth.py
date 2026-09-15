"""
quant_research.strategies.mean_reversion_smooth
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Strategy 1b: Smoothed Mean-Reversion (turnover-controlled).

Motivation
----------
Milestone 5 showed that the raw daily mean-reversion signal has a
real edge (Sharpe 0.48 at zero costs) but generates 1,063 units of
turnover over 5 years, consuming 106% of capital in costs.

Fix: Smooth the z-score with an exponential moving average (EMA).
This keeps positions longer and dramatically reduces turnover.

Signal
------
  z[t] = return_1d[t] / rolling_vol[t, N]
  smoothed[t] = EMA(z, halflife=H)[t]
  signal[t] = clip(−smoothed[t], −1, 1)

The EMA acts as a low-pass filter: rapid daily z-score changes are
dampened, and the signal only shifts meaningfully when the z-score
is persistently elevated.

Parameters
----------
  vol_lookback : int = 20   (same as raw MR)
  ema_halflife : int = 5    (5-day EMA — positions last ~5 days on average)

Expected turnover reduction: 60–80% vs raw mean-reversion.
"""

from __future__ import annotations

import pandas as pd

from quant_research.signals.base import BaseSignal


class SmoothedMeanReversionSignal(BaseSignal):
    """Smoothed daily mean-reversion signal.

    Parameters
    ----------
    vol_lookback : int
        Rolling window for volatility estimation.  Default: 20.
    ema_halflife : int
        Half-life for exponential smoothing of the z-score.
        Higher values → smoother signal → lower turnover.
        Default: 5 (positions last ~5 trading days on average).
    """

    def __init__(self, vol_lookback: int = 20, ema_halflife: int = 5):
        self.vol_lookback = vol_lookback
        self.ema_halflife = ema_halflife

    def generate(self, data: pd.DataFrame) -> pd.Series:
        """Compute the smoothed mean-reversion signal.

        Uses only data up to and including time t (no lookahead).
        """
        close = data["close"]
        return_1d = close.pct_change()

        # Rolling volatility (backward-looking)
        rolling_vol = return_1d.rolling(
            window=self.vol_lookback,
            min_periods=self.vol_lookback,
        ).std()

        # Z-score
        z_score = return_1d / rolling_vol

        # Smooth with EMA (backward-looking — no lookahead)
        smoothed = z_score.ewm(halflife=self.ema_halflife, adjust=False).mean()

        # Invert and clip
        signal = (-smoothed).clip(-1, 1)

        return signal.fillna(0.0)

    def __repr__(self) -> str:
        return (
            f"SmoothedMeanReversionSignal("
            f"vol_lookback={self.vol_lookback}, "
            f"ema_halflife={self.ema_halflife})"
        )
