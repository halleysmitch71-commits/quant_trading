"""
quant_research.backtest.costs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Transaction cost and slippage models.

Cost decomposition:
  total_cost = commission + spread + slippage

  • Commission:  Broker fee per trade (one-way).
  • Spread:      Half the bid-ask spread (one-way).  You cross the spread
                 on entry and again on exit.
  • Slippage:    Market impact — the price moves against you when you trade.

All costs are expressed in basis points (1 bp = 0.01%).  Each component
is a one-way cost.  A round-trip trade (buy + sell) incurs 2× the
one-way total.

Example:
  Equity: 3 bps commission + 3 bps spread + 4 bps slippage = 10 bps one-way
  A full reversal (+1 → −1) has turnover = 2.0, so cost = 2 × 10 bps = 20 bps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModel:
    """Immutable transaction cost model.

    Parameters
    ----------
    commission_bps : float
        One-way broker commission in basis points.
    spread_bps : float
        One-way half-spread cost in basis points.
    slippage_bps : float
        One-way market impact / slippage in basis points.

    Examples
    --------
    >>> model = CostModel(commission_bps=3, spread_bps=3, slippage_bps=4)
    >>> model.total_bps
    10.0
    """

    commission_bps: float = 0.0
    spread_bps: float = 0.0
    slippage_bps: float = 0.0

    @property
    def total_bps(self) -> float:
        """Total one-way cost in basis points."""
        return self.commission_bps + self.spread_bps + self.slippage_bps

    @property
    def total_rate(self) -> float:
        """Total one-way cost as a decimal fraction."""
        return self.total_bps / 10_000.0

    def compute_costs(self, turnover: pd.Series) -> pd.Series:
        """Compute dollar-proportional costs from turnover.

        Parameters
        ----------
        turnover : pd.Series
            Absolute change in position weight (|Δw|).  A value of 1.0
            means the full portfolio weight changed.

        Returns
        -------
        pd.Series
            Cost as a fraction of equity at each bar.
        """
        return turnover * self.total_rate

    def cost_breakdown(self, turnover: pd.Series) -> pd.DataFrame:
        """Return a DataFrame showing each cost component.

        Useful for analysis and reporting.
        """
        return pd.DataFrame(
            {
                "commission": turnover * self.commission_bps / 10_000,
                "spread": turnover * self.spread_bps / 10_000,
                "slippage": turnover * self.slippage_bps / 10_000,
                "total": turnover * self.total_rate,
            },
            index=turnover.index,
        )

    def __repr__(self) -> str:
        return (
            f"CostModel(commission={self.commission_bps}bp, "
            f"spread={self.spread_bps}bp, "
            f"slippage={self.slippage_bps}bp, "
            f"total={self.total_bps}bp)"
        )


# ── Preset cost models ──────────────────────────────────────────────────────
# These are conservative estimates for liquid instruments.
# Justification documented inline for interview defensibility.

# Liquid US ETFs (SPY, QQQ): tight spreads, low commission
# Commission ~2–5 bps, half-spread ~1–3 bps, slippage ~2–4 bps
EQUITY_COSTS = CostModel(commission_bps=3.0, spread_bps=3.0, slippage_bps=4.0)

# Major crypto (BTC, ETH) on large exchanges (e.g., Binance, Coinbase Pro)
# Commission ~5–10 bps (maker/taker), spread ~3–5 bps, slippage ~3–5 bps
CRYPTO_COSTS = CostModel(commission_bps=5.0, spread_bps=5.0, slippage_bps=5.0)

# Liquid futures (Gold, Oil — front month)
# Commission ~1–2 bps, spread ~1–2 bps, slippage ~1–2 bps
FUTURES_COSTS = CostModel(commission_bps=2.0, spread_bps=2.0, slippage_bps=1.0)


def cost_model_for_asset_class(asset_class: str) -> CostModel:
    """Return the preset CostModel for a given asset class.

    Parameters
    ----------
    asset_class : str
        One of "equity", "crypto", "futures".

    Returns
    -------
    CostModel

    Raises
    ------
    ValueError
        If asset_class is not recognised.
    """
    models = {
        "equity": EQUITY_COSTS,
        "crypto": CRYPTO_COSTS,
        "futures": FUTURES_COSTS,
    }
    if asset_class not in models:
        raise ValueError(
            f"Unknown asset class '{asset_class}'. "
            f"Expected one of: {list(models.keys())}"
        )
    return models[asset_class]


def cost_model_from_bps(total_bps: float) -> CostModel:
    """Create a simple CostModel from a single total-bps value.

    This is a convenience for backward compatibility with the engine's
    ``cost_bps`` parameter.  All cost is assigned to the "commission"
    bucket for simplicity.

    Parameters
    ----------
    total_bps : float
        Total one-way cost in basis points.
    """
    return CostModel(commission_bps=total_bps, spread_bps=0.0, slippage_bps=0.0)


# ── Cost sensitivity analysis helper ────────────────────────────────────────


def scale_costs(model: CostModel, factor: float) -> CostModel:
    """Scale all cost components by a multiplicative factor.

    Useful for robustness analysis:
      - ``factor=2.0`` → "What if costs double?"
      - ``factor=0.5`` → "What if costs halve?"

    Parameters
    ----------
    model : CostModel
        Base cost model.
    factor : float
        Multiplicative factor.

    Returns
    -------
    CostModel
        New model with scaled costs.
    """
    return CostModel(
        commission_bps=model.commission_bps * factor,
        spread_bps=model.spread_bps * factor,
        slippage_bps=model.slippage_bps * factor,
    )
