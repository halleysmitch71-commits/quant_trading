"""
quant_research.metrics.attribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Portfolio P&L attribution analysis.

Decomposes portfolio performance into contributions from:
  • Individual assets (BTC, ETH, QQQ, etc.)
  • Direction (long vs short positions)
  • Time periods (yearly, monthly)
  • Cost components (per-asset cost breakdown)

A senior quant cannot defend a strategy without knowing WHERE
the alpha comes from.  These tools provide that visibility.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def asset_pnl_attribution(
    positions: pd.DataFrame,
    asset_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Decompose portfolio P&L by asset.

    Parameters
    ----------
    positions : pd.DataFrame
        Per-asset position weights (columns = tickers).
    asset_returns : pd.DataFrame
        Per-asset daily returns (columns = tickers).

    Returns
    -------
    pd.DataFrame
        Summary with columns: asset, total_pnl, ann_pnl, pct_of_total,
        avg_position, avg_abs_position, n_long_days, n_short_days.
    """
    # Per-asset daily P&L
    pnl = positions * asset_returns
    total_portfolio_pnl = pnl.sum(axis=1).sum()

    rows = []
    for col in pnl.columns:
        asset_total = pnl[col].sum()
        n_days = len(pnl)
        ann_factor = 252 / n_days if n_days > 0 else 0

        rows.append({
            "asset": col,
            "total_pnl": asset_total,
            "ann_pnl": asset_total * ann_factor,
            "pct_of_total": asset_total / total_portfolio_pnl * 100 if total_portfolio_pnl != 0 else 0,
            "avg_position": positions[col].mean(),
            "avg_abs_position": positions[col].abs().mean(),
            "n_long_days": (positions[col] > 1e-8).sum(),
            "n_short_days": (positions[col] < -1e-8).sum(),
            "n_flat_days": (positions[col].abs() <= 1e-8).sum(),
        })

    return pd.DataFrame(rows)


def long_short_attribution(
    positions: pd.DataFrame,
    asset_returns: pd.DataFrame,
) -> dict:
    """Decompose portfolio P&L into long and short contributions.

    Parameters
    ----------
    positions : pd.DataFrame
        Per-asset position weights.
    asset_returns : pd.DataFrame
        Per-asset daily returns.

    Returns
    -------
    dict
        Attribution with keys: long_pnl, short_pnl, total_pnl,
        long_pct, short_pct, avg_long_exposure, avg_short_exposure.
    """
    pnl = positions * asset_returns

    # Long P&L: return earned when position > 0
    long_mask = positions > 1e-8
    long_pnl_daily = (pnl * long_mask).sum(axis=1)
    long_pnl_total = long_pnl_daily.sum()

    # Short P&L: return earned when position < 0
    short_mask = positions < -1e-8
    short_pnl_daily = (pnl * short_mask).sum(axis=1)
    short_pnl_total = short_pnl_daily.sum()

    total = long_pnl_total + short_pnl_total

    # Exposure
    long_exposure = (positions * long_mask).sum(axis=1).mean()
    short_exposure = (positions.abs() * short_mask).sum(axis=1).mean()

    return {
        "long_pnl": long_pnl_total,
        "short_pnl": short_pnl_total,
        "total_pnl": total,
        "long_pct": long_pnl_total / total * 100 if total != 0 else 0,
        "short_pct": short_pnl_total / total * 100 if total != 0 else 0,
        "avg_long_exposure": long_exposure,
        "avg_short_exposure": short_exposure,
        "long_sharpe": _series_sharpe(long_pnl_daily),
        "short_sharpe": _series_sharpe(short_pnl_daily),
    }


def yearly_attribution(
    net_returns: pd.Series,
    gross_returns: Optional[pd.Series] = None,
    costs: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Compute P&L attribution by calendar year.

    Parameters
    ----------
    net_returns : pd.Series
        Daily net returns.
    gross_returns : pd.Series, optional
        Daily gross returns (before costs).
    costs : pd.Series, optional
        Daily costs.

    Returns
    -------
    pd.DataFrame
        Yearly breakdown of return, vol, Sharpe, max DD.
    """
    rows = []
    for year, group in net_returns.groupby(net_returns.index.year):
        n_days = len(group)
        total_ret = (1 + group).prod() - 1
        ann_vol = group.std() * np.sqrt(252) if n_days > 1 else 0
        ann_ret = (1 + total_ret) ** (252 / n_days) - 1 if n_days > 0 else 0

        # Max drawdown
        equity = (1 + group).cumprod()
        max_dd = (equity / equity.cummax() - 1).min() if len(equity) > 0 else 0

        # Monthly stats
        monthly = (1 + group).resample("ME").prod() - 1
        pct_pos = (monthly > 0).mean() if len(monthly) > 0 else 0

        row = {
            "year": year,
            "total_return": total_ret,
            "ann_return": ann_ret,
            "ann_vol": ann_vol,
            "sharpe": ann_ret / ann_vol if ann_vol > 1e-10 else 0,
            "max_dd": max_dd,
            "pct_pos_months": pct_pos,
            "n_days": n_days,
            "avg_monthly": monthly.mean() if len(monthly) > 0 else 0,
        }

        if gross_returns is not None:
            gross_year = gross_returns.loc[group.index]
            row["gross_return"] = (1 + gross_year).prod() - 1

        if costs is not None:
            costs_year = costs.loc[group.index]
            row["total_costs"] = costs_year.sum()

        rows.append(row)

    return pd.DataFrame(rows)


def cost_attribution(
    positions: pd.DataFrame,
    cost_rates: dict[str, float],
) -> pd.DataFrame:
    """Compute cost breakdown per asset.

    Parameters
    ----------
    positions : pd.DataFrame
        Per-asset position weights.
    cost_rates : dict
        Ticker → one-way cost rate (decimal, not bps).

    Returns
    -------
    pd.DataFrame
        Per-asset cost breakdown.
    """
    rows = []
    total_cost = 0.0

    for col in positions.columns:
        turnover = positions[col].diff().abs().fillna(positions[col].abs())
        rate = cost_rates.get(col, 0.0)
        asset_cost = (turnover * rate).sum()
        total_cost += asset_cost

        rows.append({
            "asset": col,
            "total_turnover": turnover.sum(),
            "avg_daily_turnover": turnover.mean(),
            "cost_rate_bps": rate * 10_000,
            "total_cost": asset_cost,
            "cost_as_pct_capital": asset_cost * 100,
        })

    # Add percentage of total
    for row in rows:
        row["pct_of_total_cost"] = row["total_cost"] / total_cost * 100 if total_cost > 0 else 0

    return pd.DataFrame(rows)


def format_attribution_report(
    asset_attr: pd.DataFrame,
    ls_attr: dict,
    yearly_attr: pd.DataFrame,
    cost_attr: Optional[pd.DataFrame] = None,
) -> str:
    """Format all attribution results as a readable string."""
    lines = []

    # Asset attribution
    lines.append("=" * 70)
    lines.append("P&L ATTRIBUTION BY ASSET")
    lines.append("=" * 70)
    for _, row in asset_attr.iterrows():
        lines.append(
            f"  {row['asset']:<12s}  P&L={row['total_pnl']:>+8.4f}  "
            f"({row['pct_of_total']:>+6.1f}%)  "
            f"AvgPos={row['avg_position']:>+6.3f}  "
            f"Long={row['n_long_days']:>4d}d  Short={row['n_short_days']:>4d}d"
        )

    # Long/short attribution
    lines.append("")
    lines.append("=" * 70)
    lines.append("P&L ATTRIBUTION BY DIRECTION")
    lines.append("=" * 70)
    lines.append(f"  Long P&L:   {ls_attr['long_pnl']:>+.4f}  ({ls_attr['long_pct']:>+.1f}%)  "
                 f"Avg Exposure: {ls_attr['avg_long_exposure']:.3f}  "
                 f"Sharpe: {ls_attr['long_sharpe']:.3f}")
    lines.append(f"  Short P&L:  {ls_attr['short_pnl']:>+.4f}  ({ls_attr['short_pct']:>+.1f}%)  "
                 f"Avg Exposure: {ls_attr['avg_short_exposure']:.3f}  "
                 f"Sharpe: {ls_attr['short_sharpe']:.3f}")
    lines.append(f"  Total P&L:  {ls_attr['total_pnl']:>+.4f}")

    # Yearly attribution
    lines.append("")
    lines.append("=" * 70)
    lines.append("P&L ATTRIBUTION BY YEAR")
    lines.append("=" * 70)
    for _, row in yearly_attr.iterrows():
        cost_str = f"  Costs={row['total_costs']:.4f}" if "total_costs" in row else ""
        lines.append(
            f"  {int(row['year'])}  "
            f"Ret={row['total_return']:>+7.2%}  "
            f"Sharpe={row['sharpe']:>+6.3f}  "
            f"MaxDD={row['max_dd']:>+7.2%}  "
            f"PosMo={row['pct_pos_months']:>5.0%}  "
            f"AvgMo={row['avg_monthly']:>+6.2%}"
            f"{cost_str}"
        )

    # Cost attribution
    if cost_attr is not None:
        lines.append("")
        lines.append("=" * 70)
        lines.append("COST ATTRIBUTION BY ASSET")
        lines.append("=" * 70)
        for _, row in cost_attr.iterrows():
            lines.append(
                f"  {row['asset']:<12s}  "
                f"Turnover={row['total_turnover']:>7.2f}  "
                f"Rate={row['cost_rate_bps']:>5.1f}bp  "
                f"Cost={row['total_cost']:>.4f}  "
                f"({row['pct_of_total_cost']:>5.1f}% of total)"
            )

    return "\n".join(lines)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _series_sharpe(returns: pd.Series) -> float:
    """Quick annualised Sharpe for a return series."""
    if len(returns) < 2:
        return 0.0
    ann_ret = (1 + returns).prod() ** (252 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    if ann_vol < 1e-10:
        return 0.0
    return ann_ret / ann_vol
