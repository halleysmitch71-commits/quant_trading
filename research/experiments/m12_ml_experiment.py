"""
Milestone 12 — Optional ML Experiment
=======================================

Question: Can a machine learning model extract more signal than
the hand-crafted MR/TF signals we built?

Approach:
  • Use the SAME features that our signals use (z-scores, momentum)
  • Train a Ridge regression to predict next-day returns
  • Walk-forward: train on expanding window, predict next quarter
  • Compare ML signal vs hand-crafted signals

Hypothesis: ML is unlikely to help here because:
  1. Our features are already well-designed signal transformations
  2. The feature space is small (5 features) — not enough for ML to shine
  3. Daily returns are extremely noisy (signal-to-noise << 1)
  4. Linear models will approximate our hand-crafted signals

This experiment is designed to DEMONSTRATE the reasoning, not to
produce a magical alpha.

Usage:
    python research/experiments/m12_ml_experiment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant_research.backtest.costs import CRYPTO_COSTS
from quant_research.backtest.engine import run_backtest
from quant_research.data.loader import load_config, load_universe, split_data
from quant_research.metrics.performance import full_report, sharpe_ratio
from quant_research.strategies.mean_reversion_smooth import SmoothedMeanReversionSignal
from quant_research.strategies.trend_following import TrendFollowingSignal

CHART_DIR = PROJECT_ROOT / "research" / "experiments" / "m12_charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix from OHLCV data.

    Features (all backward-looking, no lookahead):
      1. z_score:  1-day return / 20-day vol
      2. z_smooth: EMA(z_score, halflife=10)
      3. mom_60:   60-day return / (60-day vol × √60)
      4. mom_120:  120-day return / (120-day vol × √120)
      5. vol_ratio: 5-day vol / 60-day vol  (vol regime)
      6. return_5d: 5-day cumulative return
    """
    close = data["close"]
    ret_1d = close.pct_change()

    vol_20 = ret_1d.rolling(20, min_periods=20).std()
    vol_5 = ret_1d.rolling(5, min_periods=5).std()
    vol_60 = ret_1d.rolling(60, min_periods=60).std()

    z_score = ret_1d / vol_20
    z_smooth = z_score.ewm(halflife=10, adjust=False).mean()

    mom_60 = close.pct_change(60) / (vol_60 * np.sqrt(60))
    mom_120 = close.pct_change(120) / (
        ret_1d.rolling(120, min_periods=120).std() * np.sqrt(120)
    )

    vol_ratio = vol_5 / vol_60
    return_5d = close.pct_change(5)

    features = pd.DataFrame({
        "z_score": z_score,
        "z_smooth": z_smooth,
        "mom_60": mom_60,
        "mom_120": mom_120,
        "vol_ratio": vol_ratio,
        "return_5d": return_5d,
    }, index=data.index)

    return features


def walk_forward_ml(
    data: pd.DataFrame,
    cost_model,
    min_train_days: int = 504,   # ~2 years
    step_days: int = 63,         # ~3 months
    alpha: float = 1.0,
) -> dict:
    """Walk-forward ML signal generation.

    For each step:
      1. Train Ridge on [0 : train_end]
      2. Predict signal for [train_end : train_end + step_days]
      3. Advance by step_days
      4. Chain all predictions together

    Returns dict with signal, returns, and diagnostics.
    """
    features = build_features(data)
    close = data["close"]
    target = close.pct_change().shift(-1)  # Next-day return (target)

    # Drop NaN rows
    valid = features.join(target.rename("target")).dropna()
    X_all = valid.drop("target", axis=1)
    y_all = valid["target"]

    n = len(valid)
    all_signals = []
    all_dates = []
    coef_history = []

    cursor = min_train_days
    while cursor < n:
        test_end = min(cursor + step_days, n)

        # Train
        X_train = X_all.iloc[:cursor]
        y_train = y_all.iloc[:cursor]

        # Standardise features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = Ridge(alpha=alpha)
        model.fit(X_train_scaled, y_train)

        # Predict (test period)
        X_test = X_all.iloc[cursor:test_end]
        X_test_scaled = scaler.transform(X_test)
        predictions = model.predict(X_test_scaled)

        # Convert predictions to signal: clip to [-1, 1]
        # Scale predictions by their std to normalise
        pred_std = np.std(predictions)
        if pred_std > 0:
            normalised = predictions / (2 * pred_std)
        else:
            normalised = predictions
        signal_values = np.clip(normalised, -1, 1)

        all_signals.extend(signal_values)
        all_dates.extend(X_test.index)

        # Record coefficients
        coef_history.append({
            "train_end": X_train.index[-1],
            "n_train": len(X_train),
            **dict(zip(X_all.columns, model.coef_)),
            "intercept": model.intercept_,
        })

        cursor = test_end

    signal = pd.Series(all_signals, index=all_dates)

    # Run backtest
    prices = close.loc[signal.index]
    result = run_backtest(prices=prices, signal=signal, cost_model=cost_model)

    return {
        "signal": signal,
        "result": result,
        "coef_history": pd.DataFrame(coef_history),
        "features": features,
    }


def main():
    print("=" * 70)
    print("MILESTONE 12 — Optional ML Experiment")
    print("=" * 70)
    print("\nQuestion: Can Ridge regression extract more signal than")
    print("hand-crafted MR/TF signals?")

    config = load_config()
    universe = load_universe(config)

    # Focus on BTC-USD (our best-performing asset)
    ticker = "BTC-USD"
    splits = split_data(universe[ticker], config)
    is_data = splits["in_sample"]
    val_data = splits["validation"]

    # ── 1. ML on in-sample ───────────────────────────────────────────────
    print(f"\n[1/5] Walk-forward ML on {ticker} (IN-SAMPLE)...")
    ml_is = walk_forward_ml(is_data, CRYPTO_COSTS, min_train_days=504, step_days=63)
    ml_is_report = full_report(ml_is["result"].net_returns,
                                turnover=ml_is["result"].turnover)
    print(f"  ML Signal — IS:")
    print(f"    Sharpe:   {ml_is_report['sharpe_ratio']:.3f}")
    print(f"    Ann.Ret:  {ml_is_report['annualised_return']:.2%}")
    print(f"    Turnover: {ml_is_report.get('total_turnover', 0):.0f}")

    # ── 2. Hand-crafted baseline comparison ──────────────────────────────
    print(f"\n[2/5] Hand-crafted baselines on {ticker} (IN-SAMPLE)...")
    baselines = {
        "TF_60": TrendFollowingSignal(lookback=60, scale=2.0),
        "TF_120": TrendFollowingSignal(lookback=120, scale=2.0),
        "SmoothMR_h10": SmoothedMeanReversionSignal(vol_lookback=20, ema_halflife=10),
    }

    baseline_results = {}
    for name, sig_obj in baselines.items():
        signal = sig_obj.generate(is_data)
        result = run_backtest(prices=is_data["close"], signal=signal,
                              cost_model=CRYPTO_COSTS)
        report = full_report(result.net_returns, turnover=result.turnover)
        baseline_results[name] = {"result": result, "report": report}
        print(f"    {name:15s}: Sharpe={report['sharpe_ratio']:.3f}, "
              f"Turnover={report.get('total_turnover', 0):.0f}")

    # ── 3. ML on validation ──────────────────────────────────────────────
    print(f"\n[3/5] Walk-forward ML on {ticker} (VALIDATION)...")
    # For validation, we train on IS data and predict on VAL
    # Combine IS+VAL for walk-forward
    full_data = pd.concat([is_data, val_data])
    full_data = full_data[~full_data.index.duplicated(keep='first')]
    full_data = full_data.sort_index()

    ml_full = walk_forward_ml(full_data, CRYPTO_COSTS,
                               min_train_days=len(is_data) - 63,  # Start predicting near IS end
                               step_days=63)

    # Extract only validation period
    val_mask = ml_full["signal"].index >= val_data.index[0]
    if val_mask.any():
        val_signal = ml_full["signal"][val_mask]
        val_result = run_backtest(
            prices=val_data["close"].loc[val_signal.index],
            signal=val_signal,
            cost_model=CRYPTO_COSTS,
        )
        ml_val_report = full_report(val_result.net_returns)
        print(f"  ML Signal — VAL:")
        print(f"    Sharpe:   {ml_val_report['sharpe_ratio']:.3f}")
        print(f"    Ann.Ret:  {ml_val_report['annualised_return']:.2%}")
    else:
        print("  ⚠ No validation predictions available")
        ml_val_report = {"sharpe_ratio": 0, "annualised_return": 0}

    # Baselines on validation
    print(f"\n  Baselines on {ticker} (VALIDATION):")
    for name, sig_obj in baselines.items():
        signal = sig_obj.generate(val_data)
        result = run_backtest(prices=val_data["close"], signal=signal,
                              cost_model=CRYPTO_COSTS)
        report = full_report(result.net_returns)
        print(f"    {name:15s}: Sharpe={report['sharpe_ratio']:.3f}")

    # ── 4. Coefficient stability ─────────────────────────────────────────
    print(f"\n[4/5] Coefficient stability over time...")
    coef_df = ml_is["coef_history"]
    feature_cols = [c for c in coef_df.columns
                    if c not in ("train_end", "n_train", "intercept")]
    print(f"\n  Feature coefficients (first → last window):")
    for col in feature_cols:
        vals = coef_df[col]
        print(f"    {col:12s}: {vals.iloc[0]:+.4f} → {vals.iloc[-1]:+.4f}  "
              f"(mean={vals.mean():+.4f}, std={vals.std():.4f})")

    # ── 5. Ridge alpha sensitivity ───────────────────────────────────────
    print(f"\n[5/5] Ridge alpha sensitivity...")
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    alpha_rows = []
    for alpha in alphas:
        ml_a = walk_forward_ml(is_data, CRYPTO_COSTS, alpha=alpha)
        rep = full_report(ml_a["result"].net_returns)
        alpha_rows.append({
            "alpha": alpha,
            "sharpe": rep["sharpe_ratio"],
            "ann_return": rep["annualised_return"],
        })
    alpha_df = pd.DataFrame(alpha_rows)
    print(alpha_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ═══════════════════════════════════════════════════════════════════
    # CHARTS
    # ═══════════════════════════════════════════════════════════════════
    print("\nGenerating charts...")

    # Chart 1: ML vs baselines equity curves (IS)
    fig, ax = plt.subplots(figsize=(14, 7))
    ml_eq = ml_is["result"].equity / 100_000
    ax.plot(ml_eq.index, ml_eq.values, label="ML (Ridge)", linewidth=1.5,
            color="purple")
    for name, data in baseline_results.items():
        eq = data["result"].equity / 100_000
        ax.plot(eq.index, eq.values, label=name, linewidth=1.2, alpha=0.8)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title(f"{ticker} — ML vs Hand-Crafted Signals (In-Sample)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Equity (multiple)")
    ax.legend(framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(CHART_DIR / "01_ml_vs_baseline.png", dpi=150)
    plt.close(fig)
    print("  ✓ 01_ml_vs_baseline.png")

    # Chart 2: Coefficient evolution
    fig, ax = plt.subplots(figsize=(14, 6))
    for col in feature_cols:
        ax.plot(coef_df["train_end"], coef_df[col], label=col, linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Ridge Coefficients Over Time (Walk-Forward)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Coefficient Value")
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "02_coef_evolution.png", dpi=150)
    plt.close(fig)
    print("  ✓ 02_coef_evolution.png")

    # Chart 3: Alpha sensitivity
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogx(alpha_df["alpha"], alpha_df["sharpe"], "b-o", linewidth=1.5)
    ax.set_title("Ridge Alpha vs Sharpe Ratio", fontsize=14, fontweight="bold")
    ax.set_xlabel("Ridge Alpha (regularisation)")
    ax.set_ylabel("Sharpe Ratio")
    ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "03_alpha_sensitivity.png", dpi=150)
    plt.close(fig)
    print("  ✓ 03_alpha_sensitivity.png")

    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("ML EXPERIMENT SUMMARY")
    print("=" * 70)

    print(f"\n  BTC-USD comparison:")
    print(f"  {'Signal':20s} {'IS Sharpe':>12s} {'VAL Sharpe':>12s}")
    print(f"  {'-'*20} {'-'*12} {'-'*12}")
    print(f"  {'ML (Ridge)':20s} {ml_is_report['sharpe_ratio']:>12.3f} "
          f"{ml_val_report['sharpe_ratio']:>12.3f}")
    for name, data in baseline_results.items():
        print(f"  {name:20s} {data['report']['sharpe_ratio']:>12.3f}")

    print(f"\n  Conclusion:")
    if ml_is_report["sharpe_ratio"] > max(
        d["report"]["sharpe_ratio"] for d in baseline_results.values()
    ):
        print("  ML signal outperforms baselines in-sample.")
        print("  BUT: check if this survives out-of-sample (likely not).")
    else:
        print("  ML signal does NOT outperform hand-crafted signals.")
        print("  This confirms: the features are well-designed, and ML")
        print("  cannot extract more from the same information set.")

    print(f"\n  Key insight: ML adds value when the feature space is rich")
    print(f"  and the signal-to-noise ratio is manageable. For simple")
    print(f"  momentum/MR signals on daily returns, the noise dominates.")

    print(f"\nCharts saved to: {CHART_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
