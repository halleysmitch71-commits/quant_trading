"""
quant_research.portfolio.allocator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Portfolio construction: combine sub-strategies into a single portfolio.

Architecture:
  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
  │ SubStrategy 1 │   │ SubStrategy 2 │   │ SubStrategy N │
  │ BTC × TF_60   │   │ QQQ × SMR_h10 │   │ GC=F × SMR    │
  └──────┬────────┘   └──────┬────────┘   └──────┬────────┘
         │                   │                   │
         └─────────┬─────────┴─────────┬─────────┘
                   ▼                   │
          ┌───────────────────┐        │
          │ Allocate weights  │◄───────┘
          │ (equal or custom) │
          └───────┬───────────┘
                  ▼
          ┌───────────────────┐
          │ Aggregate by asset│
          │ (sum positions)   │
          └───────┬───────────┘
                  ▼
          ┌───────────────────┐
          │ Vol-scale to      │
          │ target volatility │
          └───────┬───────────┘
                  ▼
          ┌───────────────────┐
          │ Cap gross exposure│
          │ at ≤ 100%         │
          └───────┬───────────┘
                  ▼
          ┌───────────────────┐
          │ Portfolio return   │
          │ and equity curve   │
          └───────────────────┘

Key invariant: gross exposure (sum of |position_per_asset|) never
exceeds max_gross_exposure (default: 1.0 = 100% of equity).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from quant_research.backtest.costs import CostModel, cost_model_from_bps
from quant_research.signals.base import BaseSignal

logger = logging.getLogger(__name__)


@dataclass
class SubStrategy:
    """Definition of a single sub-strategy."""

    name: str
    ticker: str
    signal_obj: BaseSignal
    cost_model: CostModel
    weight: float = 1.0  # Relative weight (normalised internally)


@dataclass
class PortfolioResult:
    """Container for portfolio-level backtest results."""

    # Daily portfolio net returns
    net_returns: pd.Series = field(repr=False)
    # Daily portfolio gross returns
    gross_returns: pd.Series = field(repr=False)
    # Daily gross exposure (sum of |asset positions|)
    gross_exposure: pd.Series = field(repr=False)
    # Per-asset net position time series
    asset_positions: pd.DataFrame = field(repr=False)
    # Per-sub-strategy raw signal time series
    sub_strategy_signals: pd.DataFrame = field(repr=False)
    # Daily portfolio costs
    costs: pd.Series = field(repr=False)
    # Equity curve
    equity: pd.Series = field(repr=False)

    initial_capital: float = 100_000.0
    vol_target: float = 0.10
    max_gross_exposure: float = 1.0

    def __repr__(self) -> str:
        total_ret = self.equity.iloc[-1] / self.initial_capital - 1
        return (
            f"PortfolioResult(bars={len(self.net_returns)}, "
            f"total_return={total_ret:.4f}, "
            f"vol_target={self.vol_target:.2f})"
        )


def build_portfolio(
    universe: dict[str, pd.DataFrame],
    sub_strategies: list[SubStrategy],
    vol_target: float = 0.10,
    vol_lookback: int = 63,
    max_gross_exposure: float = 1.0,
    initial_capital: float = 100_000.0,
    dd_threshold: Optional[float] = None,
    dd_min_scale: float = 0.25,
    concentration_limit: Optional[float] = None,
    weighting_scheme: str = "equal",
) -> PortfolioResult:
    """Construct a multi-asset portfolio from sub-strategies.

    Parameters
    ----------
    universe : dict[str, pd.DataFrame]
        Ticker → OHLCV DataFrame (must share overlapping dates).
    sub_strategies : list[SubStrategy]
        The sub-strategies to combine.
    vol_target : float
        Target annualised portfolio volatility for vol-scaling.
        Default: 0.10 (10%).
    vol_lookback : int
        Rolling window for realised vol estimation.  Default: 63.
    max_gross_exposure : float
        Maximum sum of |asset positions|.  Default: 1.0 (100%).
    initial_capital : float
        Starting equity.
    dd_threshold : float, optional
        Drawdown level at which positions start to be reduced.
        E.g., -0.05 means throttling begins at 5% drawdown.
        None (default) = no drawdown throttling.
    dd_min_scale : float
        Minimum position scale factor during extreme drawdown.
        Default: 0.25 (positions reduced to 25%).
    concentration_limit : float, optional
        Maximum per-asset exposure as fraction of gross exposure.
        E.g., 0.40 means no single asset exceeds 40%.
        None (default) = no concentration limit.
    weighting_scheme : str
        How to weight sub-strategies:
        - "equal" (default): each sub-strategy gets weight = 1/N.
        - "inverse_vol": each sub-strategy is weighted by 1/σ,
          where σ is the rolling realised vol of that strategy's returns.
          This implements a simple risk-parity approach.

    Returns
    -------
    PortfolioResult
    """
    if not sub_strategies:
        raise ValueError("Need at least one sub-strategy.")

    # ── Step 0: Identify common date range ──────────────────────────────
    tickers_needed = {ss.ticker for ss in sub_strategies}
    for t in tickers_needed:
        if t not in universe:
            raise ValueError(f"Ticker '{t}' not found in universe.")

    # Find overlapping date range
    all_indices = [universe[t].index for t in tickers_needed]
    common_dates = all_indices[0]
    for idx in all_indices[1:]:
        common_dates = common_dates.intersection(idx)
    common_dates = common_dates.sort_values()

    if len(common_dates) < 2:
        raise ValueError("Not enough overlapping dates.")

    # ── Step 1: Generate signals and compute asset returns ──────────────
    # Normalise weights
    total_weight = sum(ss.weight for ss in sub_strategies)
    norm_weights = {ss.name: ss.weight / total_weight for ss in sub_strategies}

    # Generate signals (on common dates)
    raw_signals = {}
    for ss in sub_strategies:
        data = universe[ss.ticker].loc[common_dates]
        signal = ss.signal_obj.generate(data)
        raw_signals[ss.name] = signal

    raw_signals_df = pd.DataFrame(raw_signals, index=common_dates)

    # Asset returns (close-to-close)
    asset_returns = {}
    for t in tickers_needed:
        asset_returns[t] = universe[t].loc[common_dates, "close"].pct_change().fillna(0.0)
    asset_returns_df = pd.DataFrame(asset_returns, index=common_dates)

    # ── Step 2: Apply one-bar delay and weights ─────────────────────────
    # position[t] = signal[t-1] for each sub-strategy
    delayed_signals = raw_signals_df.shift(1).fillna(0.0)

    # Apply sub-strategy weights
    weighted_positions = delayed_signals.copy()

    if weighting_scheme == "inverse_vol":
        # Inverse-vol weighting: weight each sub-strategy by 1/σ
        # Compute rolling vol for each sub-strategy's returns
        strategy_returns = {}
        for ss in sub_strategies:
            sr = delayed_signals[ss.name] * asset_returns_df[ss.ticker]
            strategy_returns[ss.name] = sr
        strat_ret_df = pd.DataFrame(strategy_returns, index=common_dates)

        rolling_vols = strat_ret_df.rolling(
            window=vol_lookback, min_periods=vol_lookback
        ).std() * np.sqrt(252)
        rolling_vols = rolling_vols.replace(0, np.nan)

        # Inverse vol, normalised
        inv_vol = 1.0 / rolling_vols
        inv_vol_sum = inv_vol.sum(axis=1)
        dynamic_weights = inv_vol.div(inv_vol_sum, axis=0).fillna(
            1.0 / len(sub_strategies)
        )

        for ss in sub_strategies:
            weighted_positions[ss.name] = delayed_signals[ss.name] * dynamic_weights[ss.name]
    else:
        # Equal weight (default)
        for ss in sub_strategies:
            weighted_positions[ss.name] = delayed_signals[ss.name] * norm_weights[ss.name]

    # ── Step 3: Aggregate positions by asset ────────────────────────────
    # Multiple sub-strategies may trade the same asset
    asset_positions = pd.DataFrame(0.0, index=common_dates, columns=list(tickers_needed))
    for ss in sub_strategies:
        asset_positions[ss.ticker] = (
            asset_positions[ss.ticker] + weighted_positions[ss.name]
        )

    # ── Step 4: Volatility scaling ──────────────────────────────────────
    # Compute portfolio return for vol estimation (unscaled)
    unscaled_returns = (asset_positions * asset_returns_df[list(tickers_needed)]).sum(axis=1)

    # Rolling realised vol
    realised_vol = unscaled_returns.rolling(
        window=vol_lookback, min_periods=vol_lookback
    ).std() * np.sqrt(252)

    # Vol scalar: target_vol / realised_vol
    vol_scalar = (vol_target / realised_vol).replace([np.inf, -np.inf], 0.0).fillna(1.0)

    # Cap the scalar to avoid extreme leverage in low-vol periods
    vol_scalar = vol_scalar.clip(upper=3.0)

    # Apply vol scaling to all asset positions
    scaled_positions = asset_positions.multiply(vol_scalar, axis=0)

    # ── Step 5: Enforce gross exposure constraint ───────────────────────
    gross_exposure = scaled_positions.abs().sum(axis=1)

    # Where exposure exceeds max, proportionally reduce
    exposure_ratio = (max_gross_exposure / gross_exposure).clip(upper=1.0)
    capped_positions = scaled_positions.multiply(exposure_ratio, axis=0)

    # ── Step 5b: Risk controls (optional) ────────────────────────────────
    if dd_threshold is not None or concentration_limit is not None:
        from quant_research.portfolio.risk import apply_risk_controls

        # Compute gross returns for drawdown estimation
        pre_risk_gross = (
            capped_positions * asset_returns_df[list(tickers_needed)]
        ).sum(axis=1)

        capped_positions = apply_risk_controls(
            capped_positions,
            portfolio_returns=pre_risk_gross,
            dd_threshold=dd_threshold if dd_threshold is not None else -1.0,
            dd_min_scale=dd_min_scale,
            concentration_limit=concentration_limit if concentration_limit is not None else 1.0,
        )

    # Recompute gross exposure after all adjustments
    gross_exposure_final = capped_positions.abs().sum(axis=1)

    # ── Step 6: Compute portfolio returns and costs ─────────────────────
    # Gross returns
    gross_returns = (capped_positions * asset_returns_df[list(tickers_needed)]).sum(axis=1)

    # Costs: compute per-asset turnover and apply asset-specific costs
    # Build a mapping: ticker → blended cost rate (weighted by sub-strategy contribution)
    asset_cost_rates = {}
    for t in tickers_needed:
        # Use the cost model from any sub-strategy for this ticker
        # (they should all use the same cost model for the same asset)
        for ss in sub_strategies:
            if ss.ticker == t:
                asset_cost_rates[t] = ss.cost_model.total_rate
                break

    total_costs = pd.Series(0.0, index=common_dates)
    for t in tickers_needed:
        turnover = capped_positions[t].diff().abs().fillna(capped_positions[t].abs())
        total_costs = total_costs + turnover * asset_cost_rates[t]

    # Net returns
    net_returns = gross_returns - total_costs

    # Equity curve
    equity = initial_capital * (1 + net_returns).cumprod()

    logger.info(
        "Portfolio built: %d bars, %d sub-strategies, %d assets, "
        "total return = %.4f",
        len(net_returns),
        len(sub_strategies),
        len(tickers_needed),
        equity.iloc[-1] / initial_capital - 1,
    )

    return PortfolioResult(
        net_returns=net_returns,
        gross_returns=gross_returns,
        gross_exposure=gross_exposure_final,
        asset_positions=capped_positions,
        sub_strategy_signals=raw_signals_df,
        costs=total_costs,
        equity=equity,
        initial_capital=initial_capital,
        vol_target=vol_target,
        max_gross_exposure=max_gross_exposure,
    )
