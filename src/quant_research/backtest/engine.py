"""
quant_research.backtest.engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Minimal event-safe backtesting engine.

Timing contract (the most important invariant):
  ┌──────────┐     ┌──────────────┐     ┌──────────────┐
  │ close[t] │ ──▶ │ signal[t]    │ ──▶ │ position[t+1]│
  │ observed │     │ computed     │     │ held         │
  └──────────┘     └──────────────┘     └──────────────┘

  • signal[t] uses only data up to and including time t.
  • position[t+1] = signal[t]  (one-bar delay).
  • strategy_return[t+1] = position[t+1] × asset_return[t+1].
  • asset_return[t+1] = close[t+1] / close[t] − 1.
  • Trades are filled at close[t] (equivalently, start of bar t+1).
  • Transaction costs are deducted from the return at the bar the
    trade occurs (t+1).

This one-bar delay is the primary defence against lookahead bias.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from quant_research.backtest.costs import CostModel, cost_model_from_bps

logger = logging.getLogger(__name__)


# ── Result container ─────────────────────────────────────────────────────────


@dataclass
class BacktestResult:
    """Container for all backtest outputs.

    All Series share the same DatetimeIndex as the input price data.
    """

    # --- Core series ---
    # Target signal before delay (signal[t] is the raw output)
    signal: pd.Series = field(repr=False)
    # Actual held position after one-bar delay (position[t] = signal[t-1])
    positions: pd.Series = field(repr=False)
    # Daily asset returns (close-to-close)
    asset_returns: pd.Series = field(repr=False)
    # Daily gross strategy returns (before costs)
    gross_returns: pd.Series = field(repr=False)
    # Daily turnover (absolute change in position weight)
    turnover: pd.Series = field(repr=False)
    # Daily transaction costs
    costs: pd.Series = field(repr=False)
    # Daily net strategy returns (after costs)
    net_returns: pd.Series = field(repr=False)
    # Cumulative equity curve (net of costs), starting at initial_capital
    equity: pd.Series = field(repr=False)

    # --- Scalars ---
    initial_capital: float = 0.0
    cost_bps: float = 0.0
    # The cost model used (for reporting and sensitivity analysis)
    cost_model: Optional[CostModel] = field(default=None, repr=False)

    def __repr__(self) -> str:
        n = len(self.net_returns)
        total_ret = self.equity.iloc[-1] / self.initial_capital - 1
        return (
            f"BacktestResult(bars={n}, "
            f"total_return={total_ret:.4f}, "
            f"cost_bps={self.cost_bps})"
        )

    @property
    def total_return(self) -> float:
        """Total cumulative return (net of costs)."""
        return self.equity.iloc[-1] / self.initial_capital - 1

    @property
    def total_cost_drag(self) -> float:
        """Total transaction costs as a fraction of initial capital."""
        return self.costs.sum()

    @property
    def total_turnover(self) -> float:
        """Total absolute turnover over the backtest."""
        return self.turnover.sum()


# ── Core engine ──────────────────────────────────────────────────────────────


def run_backtest(
    prices: pd.Series,
    signal: pd.Series,
    initial_capital: float = 100_000.0,
    cost_bps: float = 10.0,
    cost_model: Optional[CostModel] = None,
) -> BacktestResult:
    """Run a vectorised single-asset backtest with one-bar delay.

    Parameters
    ----------
    prices : pd.Series
        Close prices with a DatetimeIndex.  Must be positive (except
        for known edge cases like CL=F in April 2020).
    signal : pd.Series
        Target position weight at each bar.  ``signal[t]`` is the
        desired weight computed using data up to and including time t.
        Values are typically in [−1, 1] where 1 = 100% long, −1 = 100%
        short, 0 = flat.  Must have the same index as ``prices``.
    initial_capital : float
        Starting equity.
    cost_bps : float
        One-way transaction cost in basis points.  Used only when
        ``cost_model`` is not provided.  10 bps = 0.10% one-way.
    cost_model : CostModel, optional
        A structured cost model with commission/spread/slippage breakdown.
        If provided, overrides ``cost_bps``.

    Returns
    -------
    BacktestResult
        All time series and scalar results.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    # ── Input validation ────────────────────────────────────────────────
    if not isinstance(prices, pd.Series):
        raise ValueError("prices must be a pd.Series.")
    if not isinstance(signal, pd.Series):
        raise ValueError("signal must be a pd.Series.")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("prices must have a DatetimeIndex.")
    if len(prices) < 2:
        raise ValueError("Need at least 2 price bars.")
    if not prices.index.equals(signal.index):
        raise ValueError(
            "prices and signal must have the same index.  "
            f"prices has {len(prices)} bars, signal has {len(signal)} bars."
        )

    # ── Resolve cost model ──────────────────────────────────────────────
    if cost_model is None:
        cost_model = cost_model_from_bps(cost_bps)
    effective_bps = cost_model.total_bps

    # ── Step 1: Asset returns (close-to-close) ──────────────────────────
    asset_returns = prices.pct_change().fillna(0.0)

    # ── Step 2: Apply one-bar delay ─────────────────────────────────────
    # position[t] = signal[t-1]
    # The first bar has no prior signal → position = 0 (flat).
    positions = signal.shift(1).fillna(0.0)

    # ── Step 3: Gross strategy returns ──────────────────────────────────
    gross_returns = positions * asset_returns

    # ── Step 4: Turnover and costs ──────────────────────────────────────
    # Turnover = |Δposition|.  First bar: going from 0 to positions[0].
    turnover = positions.diff().abs().fillna(positions.abs())
    costs = cost_model.compute_costs(turnover)

    # ── Step 5: Net returns ─────────────────────────────────────────────
    net_returns = gross_returns - costs

    # ── Step 6: Equity curve ────────────────────────────────────────────
    equity = initial_capital * (1 + net_returns).cumprod()

    logger.info(
        "Backtest complete: %d bars, total return = %.4f, "
        "total costs = %.4f of initial capital",
        len(net_returns),
        equity.iloc[-1] / initial_capital - 1,
        costs.sum(),
    )

    return BacktestResult(
        signal=signal,
        positions=positions,
        asset_returns=asset_returns,
        gross_returns=gross_returns,
        turnover=turnover,
        costs=costs,
        net_returns=net_returns,
        equity=equity,
        initial_capital=initial_capital,
        cost_bps=effective_bps,
        cost_model=cost_model,
    )
