"""
quant_research.backtest.execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Execution simulation: realistic cost models, latency, partial fills.

Provides:
  - RealisticCostModel: vol-dependent slippage, funding/borrow costs
  - simulate_execution_delay: additional signal-to-trade delay
  - simulate_partial_fills: partial fill rate modeling
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RealisticCostModel:
    """Execution cost model with volatility-dependent and size-dependent components.

    Cost decomposition:
      total_cost = commission + spread + slippage + funding

    Slippage model (simplified Almgren-Chriss):
      slippage = base_slippage_bps
                 + vol_slippage_coeff × σ_daily (in bps)
                 + participation_coeff × √(|Δw|)

    Funding:
      daily_funding_rate × |position| (for crypto perpetuals)

    Borrow:
      daily_borrow_rate × |short_position| (for short equity)

    Parameters
    ----------
    commission_bps : float
        One-way broker commission in basis points.
    spread_bps : float
        One-way half-spread cost in basis points.
    base_slippage_bps : float
        Fixed slippage component in basis points.
    vol_slippage_coeff : float
        Coefficient for volatility-dependent slippage.
        slippage_vol = vol_slippage_coeff × daily_vol × 10000 (bps).
    participation_coeff : float
        Coefficient for size-dependent market impact.
        slippage_size = participation_coeff × √(|Δw|) × 10000 (bps).
    daily_funding_rate : float
        Daily carry cost as a decimal (e.g., 0.0001 = 1bp/day).
        Applied to absolute position size.
    daily_borrow_rate : float
        Daily borrow cost for short positions.
        Applied to absolute short position size.
    """
    commission_bps: float = 0.0
    spread_bps: float = 0.0
    base_slippage_bps: float = 0.0
    vol_slippage_coeff: float = 0.0
    participation_coeff: float = 0.0
    daily_funding_rate: float = 0.0
    daily_borrow_rate: float = 0.0

    @property
    def fixed_bps(self) -> float:
        """Fixed cost component (commission + spread + base slippage)."""
        return self.commission_bps + self.spread_bps + self.base_slippage_bps

    def compute_costs(
        self,
        turnover: pd.Series,
        positions: pd.Series,
        daily_vol: pd.Series,
    ) -> pd.DataFrame:
        """Compute all cost components.

        Parameters
        ----------
        turnover : pd.Series
            Absolute change in position weight |Δw|.
        positions : pd.Series
            Current position weights (signed).
        daily_vol : pd.Series
            Rolling daily volatility of the asset.

        Returns
        -------
        pd.DataFrame
            Columns: commission, spread, slippage_fixed, slippage_vol,
                     slippage_impact, funding, borrow, total
        """
        n = len(turnover)

        commission = turnover * (self.commission_bps / 10_000)
        spread = turnover * (self.spread_bps / 10_000)
        slip_fixed = turnover * (self.base_slippage_bps / 10_000)

        # Vol-dependent slippage
        dv = daily_vol.reindex(turnover.index).fillna(0.0)
        slip_vol = turnover * self.vol_slippage_coeff * dv

        # Size-dependent market impact (√turnover)
        slip_impact = self.participation_coeff * np.sqrt(turnover.clip(lower=0)) / 10_000

        # Funding cost (on absolute position)
        funding = positions.abs() * self.daily_funding_rate

        # Borrow cost (on short positions only)
        borrow = positions.clip(upper=0).abs() * self.daily_borrow_rate

        total = commission + spread + slip_fixed + slip_vol + slip_impact + funding + borrow

        return pd.DataFrame({
            "commission": commission,
            "spread": spread,
            "slippage_fixed": slip_fixed,
            "slippage_vol": slip_vol,
            "slippage_impact": slip_impact,
            "funding": funding,
            "borrow": borrow,
            "total": total,
        }, index=turnover.index)


# ── Preset realistic cost models ─────────────────────────────────────────────

REALISTIC_CRYPTO = RealisticCostModel(
    commission_bps=5.0,
    spread_bps=5.0,
    base_slippage_bps=2.0,
    vol_slippage_coeff=0.5,     # 50% of daily vol added as slippage
    participation_coeff=0.0,    # No participation impact for now
    daily_funding_rate=0.0003,  # ~0.03%/day ≈ ~11% annualised (crypto perps)
    daily_borrow_rate=0.0,      # No shorting in spot crypto
)

REALISTIC_EQUITY = RealisticCostModel(
    commission_bps=3.0,
    spread_bps=3.0,
    base_slippage_bps=2.0,
    vol_slippage_coeff=0.3,     # 30% of daily vol
    participation_coeff=0.0,
    daily_funding_rate=0.0,     # No funding for spot equity
    daily_borrow_rate=0.0001,   # ~0.01%/day ≈ ~2.5% annualised for shorts
)

REALISTIC_FUTURES = RealisticCostModel(
    commission_bps=2.0,
    spread_bps=2.0,
    base_slippage_bps=1.0,
    vol_slippage_coeff=0.2,
    participation_coeff=0.0,
    daily_funding_rate=0.0,
    daily_borrow_rate=0.0,
)

def scale_realistic_costs(model: RealisticCostModel, factor: float) -> RealisticCostModel:
    """Scale all cost components uniformly."""
    return RealisticCostModel(
        commission_bps=model.commission_bps * factor,
        spread_bps=model.spread_bps * factor,
        base_slippage_bps=model.base_slippage_bps * factor,
        vol_slippage_coeff=model.vol_slippage_coeff * factor,
        participation_coeff=model.participation_coeff * factor,
        daily_funding_rate=model.daily_funding_rate * factor,
        daily_borrow_rate=model.daily_borrow_rate * factor,
    )


# ── Execution delay simulator ────────────────────────────────────────────────

def simulate_execution_delay(
    signal: pd.Series,
    delay_bars: int = 1,
) -> pd.Series:
    """Apply additional execution delay to a signal.

    The backtest engine already applies 1-bar delay (signal[t] → position[t+1]).
    This function adds ADDITIONAL delay on top of that.

    Parameters
    ----------
    signal : pd.Series
        Raw signal output.
    delay_bars : int
        Additional bars of delay (0 = no extra delay = standard 1-bar).

    Returns
    -------
    pd.Series
        Delayed signal.
    """
    if delay_bars <= 0:
        return signal
    return signal.shift(delay_bars).fillna(0.0)


def simulate_partial_fills(
    signal: pd.Series,
    fill_rate: float = 1.0,
    seed: int = 42,
) -> pd.Series:
    """Simulate partial fills by randomly under-filling position changes.

    Parameters
    ----------
    signal : pd.Series
        Target position signal.
    fill_rate : float
        Expected fill rate (0.0 to 1.0). 1.0 = always fully filled.
    seed : int
        Random seed.

    Returns
    -------
    pd.Series
        Actual achieved signal after partial fills.
    """
    if fill_rate >= 1.0:
        return signal

    rng = np.random.RandomState(seed)

    # Compute target changes
    target_change = signal.diff().fillna(signal)

    # Apply random fill rate to each change
    fill_mask = rng.uniform(0, 1, len(target_change))
    actual_fill_pct = np.where(fill_mask < fill_rate, 1.0, fill_rate)

    actual_change = target_change * actual_fill_pct
    actual_position = actual_change.cumsum()

    return actual_position
