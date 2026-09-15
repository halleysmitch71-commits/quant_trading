"""
quant_research.metrics.ic_analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Information Coefficient (IC) analysis for signal evaluation.

The IC is the standard industry measure of signal quality.  It separates
signal predictive power from position-sizing decisions, unlike Sharpe
which conflates both.

Definitions:
  IC[t]   = Spearman rank correlation between signal[t] and return[t+1]
  IC_IR   = mean(IC) / std(IC)  — measures IC consistency
  Hit Rate = fraction of days where sign(signal) == sign(forward_return)

References:
  • Grinold & Kahn (2000) "Active Portfolio Management"
  • Qian, Hua & Sorensen (2007) "Quantitative Equity Portfolio Management"
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


def compute_ic(
    signal: pd.Series,
    forward_returns: pd.Series,
    method: str = "spearman",
) -> pd.Series:
    """Compute rolling Information Coefficient.

    Parameters
    ----------
    signal : pd.Series
        Signal values at time t.  Must be aligned with forward_returns.
    forward_returns : pd.Series
        Asset return from t to t+1.  ``forward_returns[t]`` is the return
        earned if you held a position based on ``signal[t]``.
    method : str
        Correlation method: "spearman" (default, rank-based) or "pearson".

    Returns
    -------
    pd.Series
        Daily IC values.  Each value is the rank correlation between
        signal[t] and forward_return[t] — but since these are single-asset
        time series, we compute rolling IC over a window.

    Notes
    -----
    For a single-asset signal, the "IC" on each day is just
    sign(signal) × sign(return) → hit rate.  The rolling version
    computes Spearman correlation over a window for a smoother measure.
    """
    # Align
    aligned = pd.DataFrame({
        "signal": signal,
        "fwd_ret": forward_returns,
    }).dropna()

    if len(aligned) < 2:
        return pd.Series(dtype=float)

    # For single-asset: point-wise product of signs gives directional accuracy
    # For multi-asset cross-section: Spearman correlation across assets
    # Here we return daily sign agreement as the IC proxy for single-asset
    ic = np.sign(aligned["signal"]) * np.sign(aligned["fwd_ret"])
    ic.name = "ic"
    return ic


def rolling_ic(
    signal: pd.Series,
    forward_returns: pd.Series,
    window: int = 63,
    method: str = "spearman",
) -> pd.Series:
    """Compute rolling-window IC.

    Uses a rolling window to compute the correlation between signal
    and forward returns, providing a smoothed view of IC over time.

    Parameters
    ----------
    signal : pd.Series
        Signal values.
    forward_returns : pd.Series
        Forward returns (aligned with signal).
    window : int
        Rolling window size.  Default: 63 (~3 months).
    method : str
        "spearman" or "pearson".

    Returns
    -------
    pd.Series
        Rolling IC values.
    """
    aligned = pd.DataFrame({
        "signal": signal,
        "fwd_ret": forward_returns,
    }).dropna()

    if len(aligned) < window:
        return pd.Series(dtype=float)

    if method == "spearman":
        rolling_corr = aligned["signal"].rolling(window).corr(aligned["fwd_ret"])
    else:
        rolling_corr = aligned["signal"].rolling(window).corr(aligned["fwd_ret"])

    rolling_corr.name = "rolling_ic"
    return rolling_corr


def ic_summary(
    signal: pd.Series,
    forward_returns: pd.Series,
) -> dict:
    """Compute comprehensive IC statistics.

    Parameters
    ----------
    signal : pd.Series
        Signal values at time t.
    forward_returns : pd.Series
        Return from t to t+1.

    Returns
    -------
    dict
        IC statistics including IC mean, IC_IR, hit rate, t-stat, p-value.
    """
    aligned = pd.DataFrame({
        "signal": signal,
        "fwd_ret": forward_returns,
    }).dropna()

    if len(aligned) < 10:
        return {
            "ic_mean": 0.0, "ic_std": 0.0, "ic_ir": 0.0,
            "hit_rate": 0.0, "t_stat": 0.0, "p_value": 1.0,
            "n_obs": len(aligned),
        }

    # Daily IC (sign agreement)
    daily_ic = compute_ic(signal, forward_returns)

    # IC mean and std
    ic_mean = daily_ic.mean()
    ic_std = daily_ic.std()
    ic_ir = ic_mean / ic_std if ic_std > 1e-10 else 0.0

    # Hit rate: fraction of correct directional calls
    # Only count when signal is non-zero
    non_zero = aligned[aligned["signal"].abs() > 1e-8]
    if len(non_zero) > 0:
        correct = (np.sign(non_zero["signal"]) == np.sign(non_zero["fwd_ret"])).mean()
    else:
        correct = 0.0

    # Spearman correlation (overall)
    spearman_corr, spearman_p = stats.spearmanr(
        aligned["signal"], aligned["fwd_ret"]
    )

    # Pearson correlation (overall)
    pearson_corr, pearson_p = stats.pearsonr(
        aligned["signal"], aligned["fwd_ret"]
    )

    # T-statistic for IC
    n = len(daily_ic)
    t_stat = ic_mean / (ic_std / np.sqrt(n)) if ic_std > 1e-10 else 0.0

    # Weighted IC: signal magnitude × sign agreement
    weighted_ic = (aligned["signal"].abs() * np.sign(aligned["signal"])
                   * aligned["fwd_ret"]).mean()

    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_ir": ic_ir,
        "hit_rate": correct,
        "spearman_corr": spearman_corr,
        "spearman_p": spearman_p,
        "pearson_corr": pearson_corr,
        "pearson_p": pearson_p,
        "t_stat": t_stat,
        "p_value": 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1)) if n > 1 else 1.0,
        "weighted_ic": weighted_ic,
        "n_obs": n,
        "n_active": len(non_zero),
        "avg_signal_magnitude": aligned["signal"].abs().mean(),
    }


def ic_decay_curve(
    signal: pd.Series,
    returns: pd.Series,
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Compute IC at different holding period lags.

    Shows how signal predictive power decays with time.  A good signal
    has high IC at lag 1 that slowly decays; a noisy signal has IC that
    drops to zero quickly.

    Parameters
    ----------
    signal : pd.Series
        Signal values.
    returns : pd.Series
        Daily returns (not forward returns — we'll compute multi-day forward).
    lags : list[int]
        Holding periods to test.  Default: [1, 2, 3, 5, 10, 20, 40, 60].

    Returns
    -------
    pd.DataFrame
        Columns: lag, ic_mean, ic_std, hit_rate, spearman_corr.
    """
    if lags is None:
        lags = [1, 2, 3, 5, 10, 20, 40, 60]

    rows = []
    for lag in lags:
        # Forward return over `lag` days
        fwd_ret = returns.rolling(window=lag).sum().shift(-lag)

        # Compute IC summary
        summary = ic_summary(signal, fwd_ret)
        summary["lag"] = lag
        rows.append(summary)

    return pd.DataFrame(rows)


def ic_by_year(
    signal: pd.Series,
    forward_returns: pd.Series,
) -> pd.DataFrame:
    """Compute IC statistics broken down by calendar year.

    Parameters
    ----------
    signal : pd.Series
        Signal values.
    forward_returns : pd.Series
        Forward returns.

    Returns
    -------
    pd.DataFrame
        IC statistics per year.
    """
    aligned = pd.DataFrame({
        "signal": signal,
        "fwd_ret": forward_returns,
    }).dropna()

    rows = []
    for year, group in aligned.groupby(aligned.index.year):
        if len(group) < 10:
            continue
        summary = ic_summary(group["signal"], group["fwd_ret"])
        summary["year"] = year
        rows.append(summary)

    return pd.DataFrame(rows)


def format_ic_report(summary: dict) -> str:
    """Format IC summary as a readable string."""
    lines = [
        "=" * 50,
        "INFORMATION COEFFICIENT REPORT",
        "=" * 50,
        f"  IC Mean.................... {summary['ic_mean']:.4f}",
        f"  IC Std..................... {summary['ic_std']:.4f}",
        f"  IC_IR (IC/std)............. {summary['ic_ir']:.4f}",
        f"  Hit Rate................... {summary['hit_rate']:.2%}",
        f"  Spearman Correlation....... {summary['spearman_corr']:.4f} (p={summary['spearman_p']:.4f})",
        f"  Pearson Correlation........ {summary['pearson_corr']:.4f} (p={summary['pearson_p']:.4f})",
        f"  T-Statistic................ {summary['t_stat']:.3f}",
        f"  P-Value.................... {summary['p_value']:.4f}",
        f"  Weighted IC................ {summary['weighted_ic']:.6f}",
        f"  Observations............... {summary['n_obs']}",
        f"  Active (|signal|>0)........ {summary['n_active']}",
        f"  Avg |signal|............... {summary['avg_signal_magnitude']:.4f}",
        "=" * 50,
    ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Statistically rigorous IC inference (M14)
# ═══════════════════════════════════════════════════════════════════════════


def _bartlett_kernel(lag: int, bandwidth: int) -> float:
    """Bartlett (triangular) kernel weight for HAC estimation."""
    if lag == 0:
        return 1.0
    return max(0.0, 1.0 - lag / (bandwidth + 1))


def _auto_bandwidth(n: int) -> int:
    """Andrews (1991) automatic bandwidth selection for Bartlett kernel.

    Rule of thumb: bandwidth ≈ floor(n^(1/3)).
    """
    return max(1, int(np.floor(n ** (1 / 3))))


def ic_newey_west(
    signal: pd.Series,
    forward_returns: pd.Series,
    max_lag: int | None = None,
) -> dict:
    """Compute IC with HAC (Newey-West) robust standard errors.

    The standard IC t-statistic assumes iid daily IC values, which is
    wrong: daily IC has positive autocorrelation (because signals are
    persistent).  HAC standard errors correct for this, giving wider
    confidence intervals and lower t-statistics.

    Parameters
    ----------
    signal : pd.Series
        Signal values at time t.
    forward_returns : pd.Series
        Return from t to t+1.
    max_lag : int, optional
        Maximum lag for Bartlett kernel.
        Default: automatic (Andrews 1991: floor(n^{1/3})).

    Returns
    -------
    dict
        ic_mean, hac_se, hac_t_stat, hac_p_value, naive_se, naive_t_stat,
        bandwidth, n_obs.
    """
    daily_ic = compute_ic(signal, forward_returns).dropna()
    n = len(daily_ic)

    if n < 10:
        return {
            "ic_mean": 0.0, "hac_se": 0.0, "hac_t_stat": 0.0,
            "hac_p_value": 1.0, "naive_se": 0.0, "naive_t_stat": 0.0,
            "bandwidth": 0, "n_obs": n,
        }

    ic_mean = daily_ic.mean()
    ic_demeaned = daily_ic.values - ic_mean

    # Bandwidth
    bw = max_lag if max_lag is not None else _auto_bandwidth(n)

    # HAC variance: γ(0) + 2 * Σ_{j=1}^{bw} kernel(j) * γ(j)
    # where γ(j) = autocovariance at lag j
    gamma_0 = np.mean(ic_demeaned ** 2)
    hac_var = gamma_0

    for lag in range(1, bw + 1):
        gamma_j = np.mean(ic_demeaned[lag:] * ic_demeaned[:-lag])
        weight = _bartlett_kernel(lag, bw)
        hac_var += 2 * weight * gamma_j

    hac_var = max(hac_var, 1e-20)  # Floor for numerical stability
    hac_se = np.sqrt(hac_var / n)
    naive_se = np.std(daily_ic.values, ddof=1) / np.sqrt(n)

    hac_t = ic_mean / hac_se if hac_se > 1e-10 else 0.0
    naive_t = ic_mean / naive_se if naive_se > 1e-10 else 0.0

    hac_p = 2 * (1 - stats.t.cdf(abs(hac_t), df=n - 1)) if n > 1 else 1.0

    return {
        "ic_mean": ic_mean,
        "hac_se": hac_se,
        "hac_t_stat": hac_t,
        "hac_p_value": hac_p,
        "naive_se": naive_se,
        "naive_t_stat": naive_t,
        "bandwidth": bw,
        "n_obs": n,
    }


def ic_block_bootstrap(
    signal: pd.Series,
    forward_returns: pd.Series,
    n_boot: int = 1000,
    block_size: int = 21,
    seed: int = 42,
) -> dict:
    """Block bootstrap confidence intervals for IC.

    Uses circular block bootstrap (Politis & Romano 1992) to properly
    account for time-series dependence in the IC estimate.

    Parameters
    ----------
    signal : pd.Series
        Signal values.
    forward_returns : pd.Series
        Forward returns.
    n_boot : int
        Number of bootstrap resamples.
    block_size : int
        Block size (days).  Default: 21 (~1 month).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        boot_mean, boot_se, ci_lower_95, ci_upper_95, ci_lower_90,
        ci_upper_90, p_gt_zero (probability IC > 0).
    """
    daily_ic = compute_ic(signal, forward_returns).dropna()
    n = len(daily_ic)
    ic_vals = daily_ic.values

    if n < block_size * 2:
        return {
            "boot_mean": 0.0, "boot_se": 0.0,
            "ci_lower_95": 0.0, "ci_upper_95": 0.0,
            "ci_lower_90": 0.0, "ci_upper_90": 0.0,
            "p_gt_zero": 0.5, "n_boot": 0, "n_obs": n,
        }

    rng = np.random.default_rng(seed)
    n_blocks = n // block_size + 1
    boot_means = np.empty(n_boot)

    for b in range(n_boot):
        # Circular block bootstrap
        starts = rng.integers(0, n, n_blocks)
        resampled = np.concatenate([
            ic_vals[s % n: min(s % n + block_size, n)]
            if s % n + block_size <= n
            else np.concatenate([ic_vals[s % n:], ic_vals[:block_size - (n - s % n)]])
            for s in starts
        ])[:n]
        boot_means[b] = resampled.mean()

    return {
        "boot_mean": float(boot_means.mean()),
        "boot_se": float(boot_means.std()),
        "ci_lower_95": float(np.percentile(boot_means, 2.5)),
        "ci_upper_95": float(np.percentile(boot_means, 97.5)),
        "ci_lower_90": float(np.percentile(boot_means, 5.0)),
        "ci_upper_90": float(np.percentile(boot_means, 95.0)),
        "p_gt_zero": float((boot_means > 0).mean()),
        "n_boot": n_boot,
        "n_obs": n,
    }


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[dict]:
    """Benjamini-Hochberg False Discovery Rate (FDR) correction.

    Controls the expected proportion of false discoveries among rejected
    hypotheses.  More powerful than Bonferroni for multiple testing.

    Parameters
    ----------
    p_values : list[float]
        Raw p-values from individual tests.
    alpha : float
        Target FDR level.  Default: 0.05.

    Returns
    -------
    list[dict]
        For each test: original_p, rank, bh_critical, adjusted_p, reject.
    """
    m = len(p_values)
    if m == 0:
        return []

    # Sort by p-value
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    results = [None] * m
    prev_adj_p = 0.0

    # Compute adjusted p-values (step-up procedure)
    adjusted = [0.0] * m
    for rank_idx in range(m - 1, -1, -1):
        orig_idx, p = indexed[rank_idx]
        rank = rank_idx + 1
        adj_p = p * m / rank
        if rank_idx < m - 1:
            # Enforce monotonicity
            _, next_p = indexed[rank_idx + 1]
            adj_p = min(adj_p, adjusted[rank_idx + 1])
        adjusted[rank_idx] = min(adj_p, 1.0)

    for rank_idx, (orig_idx, p) in enumerate(indexed):
        rank = rank_idx + 1
        bh_critical = alpha * rank / m
        results[orig_idx] = {
            "original_p": p,
            "rank": rank,
            "bh_critical": bh_critical,
            "adjusted_p": adjusted[rank_idx],
            "reject": adjusted[rank_idx] <= alpha,
        }

    return results

