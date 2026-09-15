"""
quant_research.signals.base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Abstract base class for all trading signals.

Design notes
------------
Every signal must implement a ``generate()`` method that takes a DataFrame of
*historically available* data and returns a Series of signal values aligned to
the same index.

Critical contract:
  • ``signal[t]`` may only use data from rows with index ``<= t``.
  • The backtester will act on ``signal[t]`` at time ``t + 1`` (next bar open)
    to guarantee no lookahead.

Implementation deferred to Milestone 3–5.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseSignal(ABC):
    """Abstract base for all signal generators."""

    @abstractmethod
    def generate(self, data: pd.DataFrame) -> pd.Series:
        """Compute signal values.

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV (or similar) DataFrame with a DatetimeIndex.
            Must contain only *historically available* information.

        Returns
        -------
        pd.Series
            Signal values aligned to ``data.index``.  Positive values
            indicate a long bias; negative values indicate a short bias;
            zero indicates no position.
        """
        ...
