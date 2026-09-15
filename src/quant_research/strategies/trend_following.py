"""
quant_research.strategies.trend_following
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Strategy 2: Time-Series Momentum (Trend Following).

Hypothesis
----------
Asset prices exhibit medium-term momentum.  An asset that has been
trending up (down) over the past 2–3 months is more likely to
continue in the same direction than to reverse.

This is one of the most well-documented anomalies in finance:
  • Moskowitz, Ooi & Pedersen (2012) "Time Series Momentum"
  • Jegadeesh & Titman (1993) "Returns to Buying Winners and Selling Losers"
  • AQR Capital Management — extensive practitioner research

Economic rationale
------------------
  • Gradual information diffusion (investors react slowly to news)
  • Behavioral herding (trend-following begets trend-following)
  • Institutional momentum (fund flows chase performance)
  • Central bank policy regimes (rates trend for extended periods)

Signal
------
  lookback_return[t] = close[t] / close[t - N] − 1
  vol[t] = rolling_std(daily_returns, N)
  z[t] = lookback_return[t] / (vol[t] × √N)
  signal[t] = clip(z[t] / scale, −1, 1)

The z-score normalisation ensures the signal is comparable across
assets with different volatility levels.  The `scale` parameter
controls how many σ moves are needed for a full position.

Parameters
----------
  lookback : int = 60  (≈ 3 months of trading days)
  scale : float = 2.0  (2σ move → full position)

Why low turnover?
-----------------
  • Trends persist for weeks to months → positions change slowly.
  • The 60-day lookback smooths out daily noise.
  • Expected daily turnover: ~5–10% (vs. ~80% for daily mean-reversion).

Failure Criteria
----------------
This strategy should be rejected if:
  1. In-sample Sharpe ratio < 0.3 (no meaningful edge)
  2. > 50% of months are negative
  3. Does not survive 2× transaction costs
  4. Turnover is not meaningfully lower than the mean-reversion baseline
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research.signals.base import BaseSignal


class TrendFollowingSignal(BaseSignal):
    """Time-series momentum signal.

    Parameters
    ----------
    lookback : int
        Number of days for the momentum lookback.  Default: 60.
    scale : float
        Number of standard deviations for a full ±1 position.
        Default: 2.0 (a 2σ trend move → 100% position).
    """

    def __init__(self, lookback: int = 60, scale: float = 2.0):
        self.lookback = lookback
        self.scale = scale

    def generate(self, data: pd.DataFrame) -> pd.Series:
        """Compute the trend-following signal.

        signal[t] uses only data up to and including time t.
        """
        close = data["close"]

        # N-day return
        lookback_return = close / close.shift(self.lookback) - 1

        # Daily returns for volatility estimation
        daily_ret = close.pct_change()

        # Rolling volatility over the lookback period
        rolling_vol = daily_ret.rolling(
            window=self.lookback,
            min_periods=self.lookback,
        ).std()

        # Annualise the lookback return's expected std:
        # std(N-day return) ≈ daily_vol × √N
        expected_std = rolling_vol * np.sqrt(self.lookback)

        # Z-score: how many σ has the asset trended?
        z_score = lookback_return / expected_std

        # Scale and clip: ±scale σ → ±1 position
        raw_signal = z_score / self.scale

        signal = raw_signal.clip(-1, 1)

        return signal.fillna(0.0)

    def __repr__(self) -> str:
        return f"TrendFollowingSignal(lookback={self.lookback}, scale={self.scale})"
