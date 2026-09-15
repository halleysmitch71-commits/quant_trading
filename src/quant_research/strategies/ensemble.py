"""
quant_research.strategies.ensemble
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ensemble signal combiner.

Combines multiple signals for the same asset into a single
blended signal.  This is a simple equal-weight (or custom-weight)
average, clipped to [−1, 1].

Rationale
---------
Mean-reversion and trend-following are negatively correlated
strategies.  Blending them reduces drawdowns and smooths the
equity curve, even if neither signal is individually strong.

The ensemble does NOT optimise weights — it uses equal weights
by default to avoid overfitting.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from quant_research.signals.base import BaseSignal


class EnsembleSignal(BaseSignal):
    """Equal-weight (or custom-weight) ensemble of multiple signals.

    Parameters
    ----------
    signals : sequence of BaseSignal
        The component signals to combine.
    weights : sequence of float, optional
        Weight for each signal.  If None, equal-weight is used.
        Weights are normalised to sum to 1.
    """

    def __init__(
        self,
        signals: Sequence[BaseSignal],
        weights: Sequence[float] | None = None,
    ):
        if len(signals) < 2:
            raise ValueError("Ensemble requires at least 2 signals.")

        self.signals = list(signals)

        if weights is None:
            n = len(signals)
            self.weights = [1.0 / n] * n
        else:
            if len(weights) != len(signals):
                raise ValueError("weights must have same length as signals.")
            total = sum(weights)
            self.weights = [w / total for w in weights]

    def generate(self, data: pd.DataFrame) -> pd.Series:
        """Combine signals using weighted average, clipped to [−1, 1]."""
        combined = pd.Series(0.0, index=data.index)

        for signal, weight in zip(self.signals, self.weights):
            s = signal.generate(data)
            combined = combined + weight * s

        return combined.clip(-1, 1).fillna(0.0)

    def __repr__(self) -> str:
        components = ", ".join(
            f"{w:.2f}×{s!r}" for w, s in zip(self.weights, self.signals)
        )
        return f"EnsembleSignal([{components}])"
