"""
quant_research.metrics.performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Performance measurement functions.

All metrics are computed from a daily return series.  Results are
reported net of transaction costs unless explicitly stated otherwise.

Convention:
  • 252 trading days per year for annualisation (equities/futures).
  • All returns are simple (not log) returns.
  • Sharpe ratio uses 0% risk-free rate by default.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


TRADING_DAYS_PER_YEAR = 252


def annualised_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Compound annualised return."""
    total = (1 + returns).prod()
    n_years = len(returns) / periods_per_year
    if n_years <= 0:
        return 0.0
    return total ** (1 / n_years) - 1


def annualised_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualised standard deviation of returns."""
    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualised Sharpe ratio.

    Sharpe = (ann_return - risk_free) / ann_vol
    """
    ann_ret = annualised_return(returns, periods_per_year)
    ann_vol = annualised_volatility(returns, periods_per_year)
    if ann_vol < 1e-10:
        return 0.0
    return (ann_ret - risk_free) / ann_vol


def sortino_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualised Sortino ratio (downside deviation only)."""
    ann_ret = annualised_return(returns, periods_per_year)
    downside = returns[returns < 0]
    if len(downside) == 0:
        return float("inf")
    downside_vol = downside.std() * np.sqrt(periods_per_year)
    if downside_vol == 0:
        return 0.0
    return (ann_ret - risk_free) / downside_vol


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown (as a negative fraction, e.g., −0.15 = −15%)."""
    equity = (1 + returns).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return drawdown.min()


def calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Calmar ratio = annualised return / |max drawdown|."""
    ann_ret = annualised_return(returns, periods_per_year)
    mdd = max_drawdown(returns)
    if mdd == 0:
        return 0.0
    return ann_ret / abs(mdd)


def value_at_risk(
    returns: pd.Series,
    level: float = 0.05,
) -> float:
    """Historical Value at Risk.

    Returns the loss threshold at the given confidence level.
    E.g., VaR(5%) = -2.1% means there is a 5% chance of losing
    more than 2.1% in a single day.

    Parameters
    ----------
    returns : pd.Series
        Daily returns.
    level : float
        Significance level.  Default: 0.05 (95% VaR).

    Returns
    -------
    float
        VaR as a negative number (e.g., -0.021 = -2.1%).
    """
    if len(returns) < 10:
        return 0.0
    return float(np.percentile(returns.dropna(), level * 100))


def conditional_var(
    returns: pd.Series,
    level: float = 0.05,
) -> float:
    """Conditional Value at Risk (Expected Shortfall / CVaR).

    Average loss in the worst `level` fraction of days.
    More informative than VaR because it captures tail severity.

    Parameters
    ----------
    returns : pd.Series
        Daily returns.
    level : float
        Significance level.  Default: 0.05 (worst 5% of days).

    Returns
    -------
    float
        CVaR as a negative number.
    """
    if len(returns) < 10:
        return 0.0
    var = value_at_risk(returns, level)
    tail = returns[returns <= var]
    if len(tail) == 0:
        return var
    return float(tail.mean())


def tail_risk_stats(returns: pd.Series) -> dict:
    """Compute tail risk statistics.

    Returns
    -------
    dict
        VaR(1%), VaR(5%), CVaR(1%), CVaR(5%), skewness, kurtosis,
        worst day, worst week, worst 5 days.
    """
    weekly = (1 + returns).resample("W").prod() - 1 if hasattr(returns.index, 'freq') or isinstance(returns.index, pd.DatetimeIndex) else returns

    return {
        "var_1pct": value_at_risk(returns, 0.01),
        "var_5pct": value_at_risk(returns, 0.05),
        "cvar_1pct": conditional_var(returns, 0.01),
        "cvar_5pct": conditional_var(returns, 0.05),
        "skewness": float(returns.skew()),
        "excess_kurtosis": float(returns.kurtosis()),
        "worst_day": float(returns.min()),
        "best_day": float(returns.max()),
        "worst_5_days_avg": float(returns.nsmallest(5).mean()) if len(returns) >= 5 else 0.0,
    }


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_obs: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio (Harvey & Liu 2015).

    Adjusts the Sharpe ratio for multiple testing.  Given that we
    tested `n_trials` strategies, what is the probability that the
    best one would have a Sharpe this high by chance alone?

    Parameters
    ----------
    observed_sharpe : float
        The observed (annualised) Sharpe ratio.
    n_trials : int
        Number of strategy variants tested (including parameter sweeps).
    n_obs : int
        Number of return observations (trading days).
    skewness : float
        Return skewness.  Default: 0 (normal).
    kurtosis : float
        Return kurtosis.  Default: 3 (normal).

    Returns
    -------
    float
        Probability that the observed Sharpe is genuine (0 to 1).
        Values > 0.95 suggest the Sharpe survives multiple testing.

    References
    ----------
    Harvey, C.R. and Liu, Y. (2015). "Backtesting", Journal of Portfolio
    Management.
    """
    if n_trials <= 0 or n_obs <= 1:
        return 0.0

    # Expected maximum Sharpe under the null (all strategies have Sharpe=0)
    # E[max(Z_1, ..., Z_N)] ≈ sqrt(2 * log(N)) for N iid standard normals
    euler_mascheroni = 0.5772156649
    expected_max = np.sqrt(2 * np.log(n_trials)) - (
        euler_mascheroni / (2 * np.sqrt(2 * np.log(n_trials)))
    ) if n_trials > 1 else 0.0

    # Standard error of the Sharpe ratio
    # SE(SR) = sqrt((1 + 0.5*SR^2 - skew*SR + (kurt-3)/4 * SR^2) / (n-1))
    sr = observed_sharpe / np.sqrt(252)  # Convert to per-period
    se = np.sqrt(
        (1 + 0.5 * sr**2 - skewness * sr + (kurtosis - 3) / 4 * sr**2)
        / (n_obs - 1)
    )

    if se < 1e-10:
        return 0.0

    # Test statistic: (observed - expected_max) / SE
    test_stat = (sr - expected_max * se) / se

    # One-sided p-value
    return float(scipy_stats.norm.cdf(test_stat))


def win_rate(returns: pd.Series) -> float:
    """Fraction of days with positive returns."""
    non_zero = returns[returns != 0]
    if len(non_zero) == 0:
        return 0.0
    return (non_zero > 0).mean()


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Full drawdown time series."""
    equity = (1 + returns).cumprod()
    running_max = equity.cummax()
    return equity / running_max - 1


def monthly_returns(returns: pd.Series) -> pd.Series:
    """Aggregate daily returns into monthly returns."""
    return (1 + returns).resample("ME").prod() - 1


def monthly_stats(returns: pd.Series) -> dict:
    """Compute monthly return statistics."""
    monthly = monthly_returns(returns)
    return {
        "avg_monthly_return": monthly.mean(),
        "median_monthly_return": monthly.median(),
        "pct_positive_months": (monthly > 0).mean(),
        "best_month": monthly.max(),
        "worst_month": monthly.min(),
        "n_months": len(monthly),
    }


def full_report(
    returns: pd.Series,
    turnover: Optional[pd.Series] = None,
    costs: Optional[pd.Series] = None,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict:
    """Compute the full performance report.

    Parameters
    ----------
    returns : pd.Series
        Daily net returns.
    turnover : pd.Series, optional
        Daily turnover for reporting.
    costs : pd.Series, optional
        Daily costs for reporting.
    periods_per_year : int
        Trading days per year.

    Returns
    -------
    dict
        All performance metrics.
    """
    total_ret = (1 + returns).prod() - 1
    ann_ret = annualised_return(returns, periods_per_year)
    ann_vol = annualised_volatility(returns, periods_per_year)
    mdd = max_drawdown(returns)
    m_stats = monthly_stats(returns)
    tail = tail_risk_stats(returns)

    report = {
        # Return metrics
        "total_return": total_ret,
        "annualised_return": ann_ret,
        "annualised_volatility": ann_vol,
        # Risk-adjusted
        "sharpe_ratio": sharpe_ratio(returns, periods_per_year=periods_per_year),
        "sortino_ratio": sortino_ratio(returns, periods_per_year=periods_per_year),
        "max_drawdown": mdd,
        "calmar_ratio": calmar_ratio(returns, periods_per_year=periods_per_year),
        # Tail risk
        "var_5pct": tail["var_5pct"],
        "cvar_5pct": tail["cvar_5pct"],
        "skewness": tail["skewness"],
        "excess_kurtosis": tail["excess_kurtosis"],
        # Win/loss
        "win_rate": win_rate(returns),
        "n_trading_days": len(returns),
        # Monthly
        **m_stats,
    }

    if turnover is not None:
        report["total_turnover"] = turnover.sum()
        report["avg_daily_turnover"] = turnover.mean()

    if costs is not None:
        report["total_costs"] = costs.sum()

    return report


def format_report(report: dict) -> str:
    """Format a performance report as a readable string."""
    lines = []
    lines.append("=" * 50)
    lines.append("PERFORMANCE REPORT")
    lines.append("=" * 50)

    fmt = {
        "total_return": ("Total Return", "{:.2%}"),
        "annualised_return": ("Annualised Return", "{:.2%}"),
        "annualised_volatility": ("Annualised Volatility", "{:.2%}"),
        "sharpe_ratio": ("Sharpe Ratio", "{:.3f}"),
        "sortino_ratio": ("Sortino Ratio", "{:.3f}"),
        "max_drawdown": ("Max Drawdown", "{:.2%}"),
        "calmar_ratio": ("Calmar Ratio", "{:.3f}"),
        "win_rate": ("Win Rate", "{:.2%}"),
        "n_trading_days": ("Trading Days", "{:.0f}"),
        "avg_monthly_return": ("Avg Monthly Return", "{:.2%}"),
        "median_monthly_return": ("Median Monthly Return", "{:.2%}"),
        "pct_positive_months": ("% Positive Months", "{:.1%}"),
        "best_month": ("Best Month", "{:.2%}"),
        "worst_month": ("Worst Month", "{:.2%}"),
        "n_months": ("Number of Months", "{:.0f}"),
        "var_5pct": ("VaR (5%)", "{:.2%}"),
        "cvar_5pct": ("CVaR (5%)", "{:.2%}"),
        "skewness": ("Skewness", "{:.3f}"),
        "excess_kurtosis": ("Excess Kurtosis", "{:.3f}"),
        "total_turnover": ("Total Turnover", "{:.2f}"),
        "avg_daily_turnover": ("Avg Daily Turnover", "{:.4f}"),
        "total_costs": ("Total Costs (frac)", "{:.4f}"),
    }

    for key, (label, f) in fmt.items():
        if key in report:
            lines.append(f"  {label:.<30s} {f.format(report[key])}")

    lines.append("=" * 50)
    return "\n".join(lines)
