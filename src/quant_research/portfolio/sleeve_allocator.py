"""
quant_research.portfolio.sleeve_allocator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Sleeve-based portfolio construction with risk-budget allocation.

Architecture:
  ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
  │  Crypto Trend     │   │   Volatility      │   │  Mean Reversion   │
  │  Sleeve           │   │   Sleeve           │   │  Sleeve           │
  │  risk_budget=0.50 │   │  risk_budget=0.30 │   │  risk_budget=0.20 │
  │  ┌─────┐ ┌─────┐ │   │  ┌─────┐ ┌─────┐ │   │  ┌─────┐ ┌─────┐ │
  │  │BTC  │ │ETH  │ │   │  │BTC  │ │ETH  │ │   │  │QQQ  │ │GC   │ │
  │  │TF60 │ │TF120│ │   │  │VOL  │ │VOL  │ │   │  │SMR  │ │SMR  │ │
  │  └─────┘ └─────┘ │   │  └─────┘ └─────┘ │   │  └─────┘ └─────┘ │
  └───────┬───────────┘   └───────┬───────────┘   └───────┬───────────┘
          │                       │                       │
          └─────────┬─────────────┴─────────┬─────────────┘
                    ▼                       │
          ┌─────────────────────┐           │
          │ Risk-Budget Scaling │◄──────────┘
          │ Each sleeve vol ──▶ │
          │  its risk budget    │
          └─────────┬───────────┘
                    ▼
          ┌─────────────────────┐
          │ Portfolio-level     │
          │ risk overlay        │
          └─────────────────────┘

Key difference from allocator.py:
  - allocator.py: equal capital weight → vol-scale total portfolio
  - sleeve_allocator.py: risk-budget per sleeve → vol-scale per sleeve → combine
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from quant_research.portfolio.allocator import SubStrategy, PortfolioResult
from quant_research.portfolio.risk import apply_risk_controls

logger = logging.getLogger(__name__)


@dataclass
class Sleeve:
    """A group of related sub-strategies sharing a risk budget.

    Parameters
    ----------
    name : str
        Sleeve identifier (e.g., "crypto_trend").
    sub_strategies : list[SubStrategy]
        The sub-strategies in this sleeve.
    risk_budget : float
        Fraction of total portfolio risk allocated to this sleeve.
        All sleeve risk_budgets should sum to 1.0.
    internal_weighting : str
        How to weight sub-strategies within the sleeve:
        - "equal": each gets 1/N weight.
        - "inverse_vol": weight by 1/σ (risk parity within sleeve).
    vol_target : float
        Annualised volatility target for this sleeve (before risk-budget scaling).
        Default: 0.10 (10%).
    """
    name: str
    sub_strategies: list[SubStrategy]
    risk_budget: float = 0.20
    internal_weighting: str = "equal"
    vol_target: float = 0.10


def build_sleeve_portfolio(
    universe: dict[str, pd.DataFrame],
    sleeves: list[Sleeve],
    vol_lookback: int = 63,
    max_gross_exposure: float = 1.0,
    dd_threshold: Optional[float] = None,
    dd_min_scale: float = 0.25,
    concentration_limit: Optional[float] = None,
    initial_capital: float = 100_000.0,
) -> PortfolioResult:
    """Construct a portfolio using sleeve-based risk-budget allocation.

    Each sleeve is independently vol-scaled to its target, then scaled
    by its risk budget to achieve the desired risk contribution.
    Portfolio-level risk controls are applied after combination.

    Parameters
    ----------
    universe : dict[str, pd.DataFrame]
        Ticker → OHLCV DataFrame.
    sleeves : list[Sleeve]
        The sleeves to combine, each with a risk_budget.
    vol_lookback : int
        Rolling window for realised vol estimation.
    max_gross_exposure : float
        Maximum gross exposure constraint.
    dd_threshold : float, optional
        Drawdown threshold for position throttling.
    dd_min_scale : float
        Minimum scale at extreme drawdown.
    concentration_limit : float, optional
        Maximum per-asset concentration.
    initial_capital : float
        Starting equity.

    Returns
    -------
    PortfolioResult
    """
    if not sleeves:
        raise ValueError("Need at least one sleeve.")

    # Normalise risk budgets to sum to 1.0
    total_budget = sum(s.risk_budget for s in sleeves)
    norm_budgets = {s.name: s.risk_budget / total_budget for s in sleeves}

    # Identify all tickers
    tickers_needed = set()
    for sleeve in sleeves:
        for ss in sleeve.sub_strategies:
            tickers_needed.add(ss.ticker)

    for t in tickers_needed:
        if t not in universe:
            raise ValueError(f"Ticker '{t}' not found in universe.")

    # Find common dates
    all_indices = [universe[t].index for t in tickers_needed]
    common_dates = all_indices[0]
    for idx in all_indices[1:]:
        common_dates = common_dates.intersection(idx)
    common_dates = common_dates.sort_values()

    if len(common_dates) < vol_lookback + 2:
        raise ValueError("Not enough overlapping dates.")

    # Compute asset returns
    asset_returns_df = pd.DataFrame({
        t: universe[t].loc[common_dates, "close"].pct_change().fillna(0.0)
        for t in tickers_needed
    })

    # Build per-sleeve positions
    sleeve_positions = {}  # sleeve_name → DataFrame (asset positions)

    for sleeve in sleeves:
        n_ss = len(sleeve.sub_strategies)
        if n_ss == 0:
            continue

        # Generate signals and apply 1-bar delay
        ss_signals = {}
        for ss in sleeve.sub_strategies:
            data = universe[ss.ticker].loc[common_dates]
            raw_sig = ss.signal_obj.generate(data)
            delayed = raw_sig.reindex(common_dates).shift(1).fillna(0.0)
            ss_signals[ss.name] = delayed

        # Internal weighting
        if sleeve.internal_weighting == "inverse_vol":
            # Compute rolling vol of each sub-strategy
            inv_vol = pd.DataFrame()
            for ss in sleeve.sub_strategies:
                ss_ret = ss_signals[ss.name] * asset_returns_df[ss.ticker]
                rv = ss_ret.rolling(vol_lookback, min_periods=vol_lookback // 2).std()
                rv = rv.replace(0.0, np.nan)
                inv_vol[ss.name] = 1.0 / rv
            inv_vol = inv_vol.fillna(0.0)
            inv_vol_sum = inv_vol.sum(axis=1).replace(0.0, np.nan)
            dyn_weights = inv_vol.div(inv_vol_sum, axis=0).fillna(1.0 / n_ss)
        else:
            # Equal weight
            dyn_weights = pd.DataFrame(
                {ss.name: 1.0 / n_ss for ss in sleeve.sub_strategies},
                index=common_dates,
            )

        # Compute weighted positions per asset
        sleeve_asset_pos = pd.DataFrame(0.0, index=common_dates, columns=list(tickers_needed))
        for ss in sleeve.sub_strategies:
            sleeve_asset_pos[ss.ticker] += ss_signals[ss.name] * dyn_weights[ss.name]

        # Compute sleeve-level return for vol estimation
        sleeve_ret = (sleeve_asset_pos * asset_returns_df[list(tickers_needed)]).sum(axis=1)

        # Vol-scale the sleeve to its target
        sleeve_vol = sleeve_ret.rolling(
            window=vol_lookback, min_periods=vol_lookback
        ).std() * np.sqrt(252)

        vol_scalar = (sleeve.vol_target / sleeve_vol).replace(
            [np.inf, -np.inf], 0.0
        ).fillna(1.0).clip(upper=3.0)

        scaled_pos = sleeve_asset_pos.multiply(vol_scalar, axis=0)

        # Apply risk budget: scale positions by √(risk_budget)
        # Why √? Because risk (variance) scales with weight², so position
        # needs to scale with √(budget) to achieve the right risk contribution.
        budget_scale = np.sqrt(norm_budgets[sleeve.name])
        sleeve_positions[sleeve.name] = scaled_pos * budget_scale

    # Combine all sleeves
    combined_positions = pd.DataFrame(0.0, index=common_dates, columns=list(tickers_needed))
    for sleeve_name, pos_df in sleeve_positions.items():
        combined_positions += pos_df

    # Portfolio-level gross exposure cap
    gross_exposure = combined_positions.abs().sum(axis=1)
    exposure_ratio = (max_gross_exposure / gross_exposure).clip(upper=1.0)
    capped_positions = combined_positions.multiply(exposure_ratio, axis=0)

    # Portfolio-level risk controls
    if dd_threshold is not None or concentration_limit is not None:
        pre_risk_ret = (capped_positions * asset_returns_df[list(tickers_needed)]).sum(axis=1)
        capped_positions = apply_risk_controls(
            capped_positions,
            portfolio_returns=pre_risk_ret,
            dd_threshold=dd_threshold if dd_threshold is not None else -1.0,
            dd_min_scale=dd_min_scale,
            concentration_limit=concentration_limit if concentration_limit is not None else 1.0,
        )

    # Compute final returns and costs
    final_gross = capped_positions.abs().sum(axis=1)

    # Compute per-asset costs using each sub-strategy's cost model
    # Map asset → cost rate (use the most expensive cost model for that asset)
    asset_cost_rates = {}
    for sleeve in sleeves:
        for ss in sleeve.sub_strategies:
            rate = ss.cost_model.total_rate
            if ss.ticker not in asset_cost_rates or rate > asset_cost_rates[ss.ticker]:
                asset_cost_rates[ss.ticker] = rate

    total_costs = pd.Series(0.0, index=common_dates)
    for t in tickers_needed:
        turnover = capped_positions[t].diff().abs().fillna(capped_positions[t].abs())
        total_costs += turnover * asset_cost_rates.get(t, 0.0)

    # Portfolio returns
    gross_returns = (capped_positions * asset_returns_df[list(tickers_needed)]).sum(axis=1)
    net_returns = gross_returns - total_costs

    # Equity curve
    equity = initial_capital * (1 + net_returns).cumprod()

    # Collect all sub-strategy signals for reporting
    all_delayed_signals = {}
    for sleeve in sleeves:
        for ss in sleeve.sub_strategies:
            data = universe[ss.ticker].loc[common_dates]
            raw_sig = ss.signal_obj.generate(data)
            delayed = raw_sig.reindex(common_dates).shift(1).fillna(0.0)
            all_delayed_signals[ss.name] = delayed
    sub_strategy_signals_df = pd.DataFrame(all_delayed_signals)

    return PortfolioResult(
        net_returns=net_returns,
        gross_returns=gross_returns,
        costs=total_costs,
        asset_positions=capped_positions,
        sub_strategy_signals=sub_strategy_signals_df,
        gross_exposure=final_gross,
        equity=equity,
    )
