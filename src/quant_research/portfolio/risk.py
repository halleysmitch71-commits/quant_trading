"""
quant_research.portfolio.risk
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Risk overlay functions applied after initial portfolio construction.

Architecture:
  Raw positions → Drawdown throttle → Concentration cap → Final positions

These are applied sequentially as position-level adjustments.
They do NOT introduce lookahead because they use only realised
(past) portfolio performance to adjust current positions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def drawdown_throttle(
    positions: pd.DataFrame,
    portfolio_returns: pd.Series,
    dd_threshold: float = -0.05,
    min_scale: float = 0.25,
) -> pd.DataFrame:
    """Reduce positions when the portfolio is in drawdown.

    When the running drawdown exceeds `dd_threshold`, positions are
    scaled down linearly.  At the maximum drawdown threshold, positions
    are reduced to `min_scale` of their original size.

    This is a dynamic risk control — NOT a stop-loss.  Positions are
    reduced proportionally, not eliminated.

    Parameters
    ----------
    positions : pd.DataFrame
        Per-asset position time series (from allocator).
    portfolio_returns : pd.Series
        Portfolio gross returns (used to compute running drawdown).
        Must be aligned with positions index.
    dd_threshold : float
        Drawdown level at which throttling begins.
        Default: -0.05 (5% drawdown → start reducing).
    min_scale : float
        Minimum position scale factor at extreme drawdown.
        Default: 0.25 (positions reduced to 25% at worst).

    Returns
    -------
    pd.DataFrame
        Adjusted positions.
    """
    equity = (1 + portfolio_returns).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1  # Negative values

    # Compute scale: 1.0 when dd=0, min_scale when dd <= dd_threshold
    # Linear interpolation between 0 and dd_threshold
    scale = pd.Series(1.0, index=positions.index)

    in_dd = drawdown < dd_threshold
    if in_dd.any():
        # How deep are we relative to threshold?
        # dd_ratio: 0 at threshold, 1 at 2× threshold
        dd_ratio = (drawdown[in_dd] - dd_threshold) / dd_threshold
        dd_ratio = dd_ratio.clip(0, 1)  # Cap at 1

        # Scale: 1.0 at threshold → min_scale at 2× threshold
        scale[in_dd] = 1.0 - (1.0 - min_scale) * dd_ratio

    # Use PREVIOUS day's scale (no lookahead)
    scale_lagged = scale.shift(1).fillna(1.0)

    return positions.multiply(scale_lagged, axis=0)


def max_concentration(
    positions: pd.DataFrame,
    max_pct: float = 0.40,
) -> pd.DataFrame:
    """Cap per-asset exposure at max_pct of total gross exposure.

    If any single asset's |position| exceeds max_pct × gross_exposure,
    it is clipped down.

    Parameters
    ----------
    positions : pd.DataFrame
        Per-asset position time series.
    max_pct : float
        Maximum allowed fraction per asset.  Default: 0.40 (40%).

    Returns
    -------
    pd.DataFrame
        Adjusted positions.
    """
    result = positions.copy()
    gross = positions.abs().sum(axis=1)

    for col in result.columns:
        # Max allowed absolute position for this asset
        max_abs = gross * max_pct
        # Clip positions: preserve sign, limit magnitude
        sign = np.sign(result[col])
        magnitude = result[col].abs()
        clipped_magnitude = pd.DataFrame(
            {"mag": magnitude, "max": max_abs}
        ).min(axis=1)
        result[col] = sign * clipped_magnitude

    return result


def apply_risk_controls(
    positions: pd.DataFrame,
    portfolio_returns: pd.Series,
    dd_threshold: float = -0.05,
    dd_min_scale: float = 0.25,
    concentration_limit: float = 0.40,
) -> pd.DataFrame:
    """Apply all risk controls in sequence.

    Order:
      1. Drawdown throttle (reduce positions during drawdowns)
      2. Concentration cap (prevent single-asset dominance)

    Parameters
    ----------
    positions : pd.DataFrame
        Raw per-asset positions.
    portfolio_returns : pd.Series
        Portfolio gross returns for drawdown computation.
    dd_threshold : float
        Drawdown threshold for throttling.
    dd_min_scale : float
        Minimum scale factor during extreme drawdown.
    concentration_limit : float
        Maximum per-asset concentration.

    Returns
    -------
    pd.DataFrame
        Risk-adjusted positions.
    """
    # Step 1: Drawdown throttle
    adjusted = drawdown_throttle(
        positions, portfolio_returns,
        dd_threshold=dd_threshold,
        min_scale=dd_min_scale,
    )

    # Step 2: Concentration cap
    adjusted = max_concentration(adjusted, max_pct=concentration_limit)

    return adjusted
