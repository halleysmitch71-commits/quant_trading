"""
Milestone 2 — Exploratory Market Research
==========================================

This script performs exploratory data analysis (EDA) on the full universe.
It generates statistics and visualisations to understand the data *before*
any strategy design.

IMPORTANT:
  • This is OBSERVATION only — no models are fitted, no parameters tuned.
  • We look at full history for structural understanding, but any signal
    ideas will only be developed on in-sample data in later milestones.
  • All charts are saved to research/experiments/m2_charts/.

Usage:
    python research/experiments/m2_exploratory_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for script execution

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# ── Project setup ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant_research.data.loader import load_config, load_universe, split_data

CHART_DIR = PROJECT_ROOT / "research" / "experiments" / "m2_charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-darkgrid")
COLORS = {
    "SPY": "#1f77b4",
    "QQQ": "#ff7f0e",
    "BTC-USD": "#2ca02c",
    "ETH-USD": "#d62728",
    "GC=F": "#9467bd",
    "CL=F": "#8c564b",
}
TRADING_DAYS_PER_YEAR = 252
CRYPTO_DAYS_PER_YEAR = 365


def annualisation_factor(ticker: str) -> int:
    """Return the number of trading days per year for annualisation."""
    return CRYPTO_DAYS_PER_YEAR if "USD" in ticker else TRADING_DAYS_PER_YEAR


# =============================================================================
# 1. LOAD DATA
# =============================================================================
def load_data():
    """Load and prepare the universe data."""
    config = load_config()
    universe = load_universe(config)

    # Compute daily returns
    returns = {}
    for ticker, df in universe.items():
        returns[ticker] = df["close"].pct_change().dropna()

    returns_df = pd.DataFrame(returns)
    return config, universe, returns, returns_df


# =============================================================================
# 2. BASIC STATISTICS
# =============================================================================
def compute_basic_stats(universe, returns) -> pd.DataFrame:
    """Compute basic descriptive statistics for each asset."""
    stats = []
    for ticker, ret in returns.items():
        n = annualisation_factor(ticker)
        ann_ret = (1 + ret.mean()) ** n - 1
        ann_vol = ret.std() * np.sqrt(n)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

        # Maximum drawdown
        prices = universe[ticker]["close"]
        cummax = prices.cummax()
        drawdown = (prices - cummax) / cummax
        max_dd = drawdown.min()

        stats.append({
            "ticker": ticker,
            "start": universe[ticker].index.min().date(),
            "end": universe[ticker].index.max().date(),
            "n_days": len(ret),
            "ann_return": ann_ret,
            "ann_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "skewness": ret.skew(),
            "kurtosis": ret.kurt(),
            "max_drawdown": max_dd,
            "best_day": ret.max(),
            "worst_day": ret.min(),
            "pct_positive_days": (ret > 0).mean(),
        })

    return pd.DataFrame(stats).set_index("ticker")


# =============================================================================
# 3. CORRELATION ANALYSIS
# =============================================================================
def compute_correlations(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise correlation of daily returns."""
    return returns_df.corr()


# =============================================================================
# 4. VISUALISATIONS
# =============================================================================
def plot_price_history(universe: dict):
    """Plot normalised price history (base 100) for all assets."""
    fig, ax = plt.subplots(figsize=(14, 7))

    for ticker, df in universe.items():
        normalised = df["close"] / df["close"].iloc[0] * 100
        ax.plot(normalised.index, normalised.values,
                label=ticker, color=COLORS.get(ticker), linewidth=1.2)

    ax.set_title("Normalised Price History (Base = 100)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Normalised Price")
    ax.set_xlabel("")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.axvline(pd.Timestamp("2020-03-23"), color="gray", linestyle="--", alpha=0.5, label="COVID low")
    ax.axvline(pd.Timestamp("2022-01-03"), color="gray", linestyle=":", alpha=0.5, label="Rate hike start")

    fig.tight_layout()
    fig.savefig(CHART_DIR / "01_price_history.png", dpi=150)
    plt.close(fig)
    print("  ✓ 01_price_history.png")


def plot_return_distributions(returns: dict):
    """Histogram of daily returns for each asset."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, (ticker, ret) in enumerate(returns.items()):
        ax = axes[i]
        ax.hist(ret.values, bins=100, color=COLORS.get(ticker), alpha=0.75,
                edgecolor="white", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"{ticker}  (skew={ret.skew():.2f}, kurt={ret.kurt():.1f})",
                     fontsize=11)
        ax.set_xlabel("Daily Return")
        ax.set_ylabel("Frequency")

    fig.suptitle("Daily Return Distributions", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "02_return_distributions.png", dpi=150)
    plt.close(fig)
    print("  ✓ 02_return_distributions.png")


def plot_rolling_volatility(returns: dict):
    """21-day and 63-day rolling annualised volatility."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, (ticker, ret) in enumerate(returns.items()):
        ax = axes[i]
        n = annualisation_factor(ticker)
        vol_21 = ret.rolling(21).std() * np.sqrt(n) * 100
        vol_63 = ret.rolling(63).std() * np.sqrt(n) * 100

        ax.plot(vol_21.index, vol_21.values, label="21-day", alpha=0.8,
                color=COLORS.get(ticker))
        ax.plot(vol_63.index, vol_63.values, label="63-day", alpha=0.8,
                color="gray", linewidth=1.5)
        ax.set_title(ticker, fontsize=12)
        ax.set_ylabel("Ann. Vol (%)")
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle("Rolling Annualised Volatility", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "03_rolling_volatility.png", dpi=150)
    plt.close(fig)
    print("  ✓ 03_rolling_volatility.png")


def plot_drawdowns(universe: dict):
    """Drawdown curves for each asset."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, (ticker, df) in enumerate(universe.items()):
        ax = axes[i]
        prices = df["close"]
        cummax = prices.cummax()
        drawdown = (prices - cummax) / cummax * 100

        ax.fill_between(drawdown.index, drawdown.values, 0,
                        color=COLORS.get(ticker), alpha=0.4)
        ax.plot(drawdown.index, drawdown.values,
                color=COLORS.get(ticker), linewidth=0.8)
        ax.set_title(f"{ticker}  (max DD = {drawdown.min():.1f}%)", fontsize=12)
        ax.set_ylabel("Drawdown (%)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle("Drawdown Curves", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "04_drawdowns.png", dpi=150)
    plt.close(fig)
    print("  ✓ 04_drawdowns.png")


def plot_correlation_matrix(corr: pd.DataFrame):
    """Heatmap of return correlations."""
    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    tickers = list(corr.index)
    ax.set_xticks(range(len(tickers)))
    ax.set_yticks(range(len(tickers)))
    ax.set_xticklabels(tickers, rotation=45, ha="right")
    ax.set_yticklabels(tickers)

    # Annotate cells
    for i in range(len(tickers)):
        for j in range(len(tickers)):
            val = corr.values[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=10, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Daily Return Correlation Matrix", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "05_correlation_matrix.png", dpi=150)
    plt.close(fig)
    print("  ✓ 05_correlation_matrix.png")


def plot_rolling_correlation(returns_df: pd.DataFrame):
    """63-day rolling correlation of each asset vs SPY."""
    fig, ax = plt.subplots(figsize=(14, 7))

    spy_ret = returns_df["SPY"]
    for ticker in returns_df.columns:
        if ticker == "SPY":
            continue
        rolling_corr = returns_df[ticker].rolling(63).corr(spy_ret)
        ax.plot(rolling_corr.index, rolling_corr.values,
                label=ticker, color=COLORS.get(ticker), linewidth=1.2, alpha=0.8)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_title("63-Day Rolling Correlation vs SPY", fontsize=14, fontweight="bold")
    ax.set_ylabel("Correlation")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylim(-1, 1)

    fig.tight_layout()
    fig.savefig(CHART_DIR / "06_rolling_corr_vs_spy.png", dpi=150)
    plt.close(fig)
    print("  ✓ 06_rolling_corr_vs_spy.png")


def plot_monthly_returns_heatmap(universe: dict):
    """Monthly return heatmap for each asset."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    axes = axes.flatten()

    for i, (ticker, df) in enumerate(universe.items()):
        ax = axes[i]
        monthly = df["close"].resample("ME").last().pct_change().dropna()
        # Pivot to year x month
        monthly_df = pd.DataFrame({
            "year": monthly.index.year,
            "month": monthly.index.month,
            "return": monthly.values * 100,
        })
        pivot = monthly_df.pivot_table(index="year", columns="month", values="return")

        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                       vmin=-20, vmax=20)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xticks(range(12))
        ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                           fontsize=8)
        ax.set_title(ticker, fontsize=12)

        # Annotate
        for yi in range(pivot.shape[0]):
            for xi in range(pivot.shape[1]):
                val = pivot.values[yi, xi]
                if not np.isnan(val):
                    color = "white" if abs(val) > 12 else "black"
                    ax.text(xi, yi, f"{val:.0f}", ha="center", va="center",
                            fontsize=7, color=color)

    fig.suptitle("Monthly Returns (%) Heatmap", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "07_monthly_returns_heatmap.png", dpi=150)
    plt.close(fig)
    print("  ✓ 07_monthly_returns_heatmap.png")


def plot_regime_analysis(universe: dict, returns: dict):
    """Overlay rolling Sharpe ratio to identify regime changes."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, (ticker, ret) in enumerate(returns.items()):
        ax = axes[i]
        n = annualisation_factor(ticker)
        # 63-day (~3 month) rolling Sharpe
        rolling_mean = ret.rolling(63).mean() * n
        rolling_std = ret.rolling(63).std() * np.sqrt(n)
        rolling_sharpe = rolling_mean / rolling_std

        ax.plot(rolling_sharpe.index, rolling_sharpe.values,
                color=COLORS.get(ticker), linewidth=1.0, alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(2, color="green", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.axhline(-2, color="red", linewidth=0.8, linestyle="--", alpha=0.5)

        # Shade major events
        ax.axvspan("2020-02-20", "2020-04-01", alpha=0.15, color="red", label="COVID")
        ax.axvspan("2022-01-01", "2022-12-31", alpha=0.10, color="orange", label="Rate hikes")

        ax.set_title(f"{ticker} — 63-Day Rolling Sharpe", fontsize=12)
        ax.set_ylabel("Sharpe Ratio")
        ax.set_ylim(-6, 6)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle("Regime Analysis — Rolling Sharpe Ratio", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "08_regime_rolling_sharpe.png", dpi=150)
    plt.close(fig)
    print("  ✓ 08_regime_rolling_sharpe.png")


def plot_autocorrelation(returns: dict):
    """Autocorrelation of daily returns (lags 1–20) to look for momentum/mean-reversion."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()
    max_lag = 20

    for i, (ticker, ret) in enumerate(returns.items()):
        ax = axes[i]
        autocorrs = [ret.autocorr(lag=lag) for lag in range(1, max_lag + 1)]
        ax.bar(range(1, max_lag + 1), autocorrs, color=COLORS.get(ticker), alpha=0.7)

        # Significance bounds (approximate 95% CI)
        n = len(ret)
        ci = 1.96 / np.sqrt(n)
        ax.axhline(ci, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axhline(-ci, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axhline(0, color="black", linewidth=0.8)

        ax.set_title(f"{ticker}  (lag-1 AC = {autocorrs[0]:.4f})", fontsize=12)
        ax.set_xlabel("Lag (days)")
        ax.set_ylabel("Autocorrelation")

    fig.suptitle("Return Autocorrelation (Lags 1–20)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "09_autocorrelation.png", dpi=150)
    plt.close(fig)
    print("  ✓ 09_autocorrelation.png")


# =============================================================================
# 5. MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("MILESTONE 2 — Exploratory Market Research")
    print("=" * 70)

    # Load
    print("\n[1/5] Loading data...")
    config, universe, returns, returns_df = load_data()
    print(f"  Loaded {len(universe)} tickers.")

    # Stats
    print("\n[2/5] Computing basic statistics...")
    stats = compute_basic_stats(universe, returns)
    print("\n" + stats.to_string(float_format=lambda x: f"{x:.4f}"))

    # Correlations
    print("\n[3/5] Computing correlations...")
    corr = compute_correlations(returns_df)
    print("\n" + corr.to_string(float_format=lambda x: f"{x:.3f}"))

    # Charts
    print("\n[4/5] Generating charts...")
    plot_price_history(universe)
    plot_return_distributions(returns)
    plot_rolling_volatility(returns)
    plot_drawdowns(universe)
    plot_correlation_matrix(corr)
    plot_rolling_correlation(returns_df)
    plot_monthly_returns_heatmap(universe)
    plot_regime_analysis(universe, returns)
    plot_autocorrelation(returns)

    # Summary
    print("\n[5/5] Key observations:")
    print("-" * 50)

    # Identify diversification potential
    off_diag = corr.values[np.triu_indices_from(corr.values, k=1)]
    print(f"  Mean pairwise correlation:  {off_diag.mean():.3f}")
    print(f"  Min pairwise correlation:   {off_diag.min():.3f}")
    print(f"  Max pairwise correlation:   {off_diag.max():.3f}")

    # Volatility regimes
    for ticker, ret in returns.items():
        n = annualisation_factor(ticker)
        recent_vol = ret.iloc[-63:].std() * np.sqrt(n) * 100
        full_vol = ret.std() * np.sqrt(n) * 100
        print(f"  {ticker:10s}: full vol = {full_vol:5.1f}%,  recent 63d vol = {recent_vol:5.1f}%")

    # Autocorrelation at lag-1
    print("\n  Lag-1 autocorrelation (negative → mean-reversion, positive → momentum):")
    for ticker, ret in returns.items():
        ac1 = ret.autocorr(lag=1)
        interpretation = "mean-reversion" if ac1 < 0 else "momentum"
        sig = "**" if abs(ac1) > 1.96 / np.sqrt(len(ret)) else ""
        print(f"    {ticker:10s}: {ac1:+.4f} {sig}  ({interpretation})")

    print(f"\n  Charts saved to: {CHART_DIR}/")
    print("=" * 70)
    print("DONE — No models fitted, no parameters optimised.")
    print("=" * 70)


if __name__ == "__main__":
    main()
