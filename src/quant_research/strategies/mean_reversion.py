"""
quant_research.strategies.mean_reversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Strategy 1: Daily Return Mean-Reversion.

Hypothesis
----------
Daily equity returns exhibit short-term mean-reversion.  After a
large up-day (relative to recent volatility), the next day tends
to be down, and vice versa.

Evidence
--------
Milestone 2 EDA found statistically significant negative lag-1
autocorrelation in SPY (−0.134) and QQQ (−0.122).  This pattern
is consistent with:
  • Behavioral overreaction to daily news
  • Market microstructure (bid-ask bounce)
  • Liquidity provision by market makers

Signal
------
  z[t] = return_1d[t] / rolling_vol[t, N]
  signal[t] = clip(−z[t], −1, 1)

The signal inverts the normalised daily return:
  • If today was a large up day → go short tomorrow
  • If today was a large down day → go long tomorrow

The z-score normalisation ensures the signal is volatility-adjusted:
during high-vol regimes, it takes a larger move to generate the
same signal magnitude.

Parameters
----------
  vol_lookback : int = 20  (≈ 1 month of trading days)

This is the ONLY free parameter.  We deliberately choose a standard
value (20 days) without optimisation.

Failure Criteria
----------------
This strategy should be rejected if:
  1. In-sample Sharpe ratio < 0.3 (no meaningful edge)
  2. > 50% of months are negative (no consistency)
  3. Strategy does not survive 2× transaction costs
  4. Performance is highly sensitive to vol_lookback parameter
"""

from __future__ import annotations

import pandas as pd

from quant_research.signals.base import BaseSignal


class MeanReversionSignal(BaseSignal):
    """Daily return reversal signal.

    Parameters
    ----------
    vol_lookback : int
        Number of days for rolling volatility estimation.
        Default: 20 (≈ 1 trading month).
    """

    def __init__(self, vol_lookback: int = 20):
        self.vol_lookback = vol_lookback

    def generate(self, data: pd.DataFrame) -> pd.Series:
        """Compute the mean-reversion signal.

        signal[t] = −z_score[t], clipped to [−1, 1].

        Only uses data up to and including time t (no lookahead).
        """
        close = data["close"]

        # 1-day simple return
        return_1d = close.pct_change()

        # Rolling volatility (backward-looking only)
        rolling_vol = return_1d.rolling(
            window=self.vol_lookback,
            min_periods=self.vol_lookback,  # Require full window
        ).std()

        # Z-score: how extreme is today's return relative to recent vol?
        z_score = return_1d / rolling_vol

        # Invert: positive z → short (expect reversion down)
        #          negative z → long  (expect reversion up)
        raw_signal = -z_score

        # Clip to [−1, 1] (no leverage)
        signal = raw_signal.clip(-1, 1)

        # Fill NaN (from warmup period) with 0 (flat)
        return signal.fillna(0.0)

    def __repr__(self) -> str:
        return f"MeanReversionSignal(vol_lookback={self.vol_lookback})"
