"""
quant_research.risk.engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Production risk engine with check/enforce/breach/halt capabilities.

Sits between portfolio output and execution:

  Strategy → Target Position → [Risk Engine] → Approved Position → Execution

All checks return RiskBreach objects. The engine can:
  1. Check all limits against current portfolio state
  2. Enforce limits by adjusting target positions
  3. Halt trading via kill switch on critical breaches
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Severity(Enum):
    """Risk breach severity levels."""
    INFO = "INFO"             # Logged, no action
    WARNING = "WARNING"       # Logged, positions reduced
    HARD_LIMIT = "HARD_LIMIT" # Positions capped, alert sent
    KILL = "KILL"             # All trading halted


@dataclass(frozen=True)
class RiskBreach:
    """Record of a risk limit violation."""
    check_name: str
    severity: Severity
    current_value: float
    limit_value: float
    message: str
    timestamp: datetime = field(default_factory=datetime.now)

    def __repr__(self) -> str:
        return (f"RiskBreach({self.severity.value}: {self.check_name} "
                f"current={self.current_value:.4f} limit={self.limit_value:.4f})")


@dataclass
class PortfolioState:
    """Snapshot of current portfolio state for risk checks.

    All fields computed from the portfolio's perspective at time t.
    """
    positions: dict[str, float]         # ticker → position weight
    equity: float = 100_000.0           # Current equity
    daily_pnl: float = 0.0             # Today's P&L as fraction of equity
    drawdown: float = 0.0             # Current drawdown (negative)
    portfolio_vol: float = 0.0        # Annualised portfolio volatility
    daily_turnover: float = 0.0       # Today's absolute turnover
    last_data_time: Optional[datetime] = None  # Last market data timestamp

    @property
    def gross_exposure(self) -> float:
        return sum(abs(v) for v in self.positions.values())

    @property
    def net_exposure(self) -> float:
        return sum(v for v in self.positions.values())

    @property
    def max_concentration(self) -> float:
        """Largest single-asset |position| / gross_exposure."""
        ge = self.gross_exposure
        if ge < 1e-10:
            return 0.0
        return max(abs(v) for v in self.positions.values()) / ge


@dataclass
class RiskLimits:
    """Configurable risk limits.

    All limits are "soft" by default — the engine adjusts positions
    to comply. Some limits trigger KILL severity for emergency halt.
    """
    # Position limits
    max_position_per_asset: float = 1.0     # Max |weight| per asset
    max_gross_exposure: float = 1.0         # Max Σ|weights|
    max_net_exposure: float = 0.5           # Max |Σ weights|
    max_concentration: float = 0.40         # Max single-asset % of gross

    # P&L limits (KILL triggers)
    max_daily_loss: float = -0.03           # -3% daily loss → KILL
    max_drawdown: float = -0.15             # -15% DD → KILL
    max_portfolio_vol: float = 0.20         # 20% annualised vol → WARNING

    # Activity limits
    max_daily_turnover: float = 2.0         # Max daily turnover

    # Data quality limits
    max_data_staleness_hours: float = 2.0   # Hours before data considered stale


class RiskEngine:
    """Production risk engine.

    Usage:
        engine = RiskEngine(RiskLimits())
        breaches = engine.check_all(state)
        approved = engine.enforce(target_positions, state)
        if engine.is_halted:
            broker.cancel_all_orders()
    """

    def __init__(self, limits: RiskLimits):
        self.limits = limits
        self.breach_log: list[RiskBreach] = []
        self.is_halted: bool = False
        self._halt_reason: Optional[str] = None

    def check_all(self, state: PortfolioState) -> list[RiskBreach]:
        """Run all risk checks and return breaches.

        Also logs breaches and triggers halt if severity is KILL.
        """
        checks = [
            self.check_position_limit,
            self.check_gross_exposure,
            self.check_net_exposure,
            self.check_concentration,
            self.check_drawdown,
            self.check_daily_loss,
            self.check_volatility,
            self.check_turnover,
            self.check_stale_data,
        ]

        breaches = []
        for check in checks:
            breach = check(state)
            if breach is not None:
                breaches.append(breach)
                self.breach_log.append(breach)
                logger.warning(f"Risk breach: {breach}")

                if breach.severity == Severity.KILL:
                    self.halt(breach.message)

        return breaches

    def check_position_limit(self, state: PortfolioState) -> Optional[RiskBreach]:
        """Check if any single position exceeds the per-asset limit."""
        for ticker, weight in state.positions.items():
            if abs(weight) > self.limits.max_position_per_asset:
                return RiskBreach(
                    check_name="position_limit",
                    severity=Severity.HARD_LIMIT,
                    current_value=abs(weight),
                    limit_value=self.limits.max_position_per_asset,
                    message=f"{ticker} position {weight:.4f} exceeds limit {self.limits.max_position_per_asset:.4f}",
                )
        return None

    def check_gross_exposure(self, state: PortfolioState) -> Optional[RiskBreach]:
        """Check if gross exposure exceeds limit."""
        ge = state.gross_exposure
        if ge > self.limits.max_gross_exposure:
            return RiskBreach(
                check_name="gross_exposure",
                severity=Severity.HARD_LIMIT,
                current_value=ge,
                limit_value=self.limits.max_gross_exposure,
                message=f"Gross exposure {ge:.4f} exceeds {self.limits.max_gross_exposure:.4f}",
            )
        return None

    def check_net_exposure(self, state: PortfolioState) -> Optional[RiskBreach]:
        """Check if net exposure exceeds limit."""
        ne = abs(state.net_exposure)
        if ne > self.limits.max_net_exposure:
            return RiskBreach(
                check_name="net_exposure",
                severity=Severity.WARNING,
                current_value=ne,
                limit_value=self.limits.max_net_exposure,
                message=f"Net exposure {state.net_exposure:+.4f} exceeds ±{self.limits.max_net_exposure:.4f}",
            )
        return None

    def check_concentration(self, state: PortfolioState) -> Optional[RiskBreach]:
        """Check if any single asset exceeds concentration limit."""
        mc = state.max_concentration
        if mc > self.limits.max_concentration:
            return RiskBreach(
                check_name="concentration",
                severity=Severity.WARNING,
                current_value=mc,
                limit_value=self.limits.max_concentration,
                message=f"Max concentration {mc:.1%} exceeds {self.limits.max_concentration:.1%}",
            )
        return None

    def check_drawdown(self, state: PortfolioState) -> Optional[RiskBreach]:
        """Check if drawdown exceeds kill threshold."""
        if state.drawdown < self.limits.max_drawdown:
            return RiskBreach(
                check_name="drawdown",
                severity=Severity.KILL,
                current_value=state.drawdown,
                limit_value=self.limits.max_drawdown,
                message=f"CRITICAL: Drawdown {state.drawdown:.2%} exceeds {self.limits.max_drawdown:.2%}",
            )
        return None

    def check_daily_loss(self, state: PortfolioState) -> Optional[RiskBreach]:
        """Check if daily loss exceeds kill threshold."""
        if state.daily_pnl < self.limits.max_daily_loss:
            return RiskBreach(
                check_name="daily_loss",
                severity=Severity.KILL,
                current_value=state.daily_pnl,
                limit_value=self.limits.max_daily_loss,
                message=f"CRITICAL: Daily loss {state.daily_pnl:.2%} exceeds {self.limits.max_daily_loss:.2%}",
            )
        return None

    def check_volatility(self, state: PortfolioState) -> Optional[RiskBreach]:
        """Check if portfolio volatility exceeds warning threshold."""
        if state.portfolio_vol > self.limits.max_portfolio_vol:
            return RiskBreach(
                check_name="volatility",
                severity=Severity.WARNING,
                current_value=state.portfolio_vol,
                limit_value=self.limits.max_portfolio_vol,
                message=f"Portfolio vol {state.portfolio_vol:.2%} exceeds {self.limits.max_portfolio_vol:.2%}",
            )
        return None

    def check_turnover(self, state: PortfolioState) -> Optional[RiskBreach]:
        """Check if daily turnover exceeds limit."""
        if state.daily_turnover > self.limits.max_daily_turnover:
            return RiskBreach(
                check_name="turnover",
                severity=Severity.WARNING,
                current_value=state.daily_turnover,
                limit_value=self.limits.max_daily_turnover,
                message=f"Daily turnover {state.daily_turnover:.2f} exceeds {self.limits.max_daily_turnover:.2f}",
            )
        return None

    def check_stale_data(self, state: PortfolioState) -> Optional[RiskBreach]:
        """Check if market data is stale."""
        if state.last_data_time is None:
            return RiskBreach(
                check_name="stale_data",
                severity=Severity.KILL,
                current_value=float("inf"),
                limit_value=self.limits.max_data_staleness_hours,
                message="CRITICAL: No market data timestamp available",
            )

        hours_since = (datetime.now() - state.last_data_time).total_seconds() / 3600
        if hours_since > self.limits.max_data_staleness_hours:
            return RiskBreach(
                check_name="stale_data",
                severity=Severity.KILL,
                current_value=hours_since,
                limit_value=self.limits.max_data_staleness_hours,
                message=f"CRITICAL: Data stale for {hours_since:.1f}h (limit: {self.limits.max_data_staleness_hours}h)",
            )
        return None

    def enforce(
        self,
        target_positions: dict[str, float],
        state: PortfolioState,
    ) -> dict[str, float]:
        """Apply risk limits to target positions.

        Returns adjusted positions that comply with all limits.
        If the engine is halted, returns zero positions.

        Parameters
        ----------
        target_positions : dict[str, float]
            Desired position weights from portfolio allocator.
        state : PortfolioState
            Current portfolio state.

        Returns
        -------
        dict[str, float]
            Approved position weights.
        """
        if self.is_halted:
            logger.critical(f"Risk engine HALTED ({self._halt_reason}). Returning zero positions.")
            return {t: 0.0 for t in target_positions}

        approved = dict(target_positions)

        # 1. Per-asset position limit
        for ticker in approved:
            limit = self.limits.max_position_per_asset
            approved[ticker] = np.clip(approved[ticker], -limit, limit)

        # 2. Gross exposure cap
        ge = sum(abs(v) for v in approved.values())
        if ge > self.limits.max_gross_exposure:
            scale = self.limits.max_gross_exposure / ge
            approved = {t: v * scale for t, v in approved.items()}

        # 3. Net exposure cap
        ne = sum(v for v in approved.values())
        if abs(ne) > self.limits.max_net_exposure:
            # Reduce all positions proportionally
            excess = abs(ne) - self.limits.max_net_exposure
            ge = sum(abs(v) for v in approved.values())
            if ge > 1e-10:
                scale = max(0, 1.0 - excess / ge)
                approved = {t: v * scale for t, v in approved.items()}

        # 4. Concentration limit
        ge = sum(abs(v) for v in approved.values())
        if ge > 1e-10:
            for ticker in approved:
                max_abs = ge * self.limits.max_concentration
                if abs(approved[ticker]) > max_abs:
                    approved[ticker] = np.sign(approved[ticker]) * max_abs

        # 5. Turnover limit
        old_pos = state.positions
        turnover = sum(abs(approved.get(t, 0) - old_pos.get(t, 0)) for t in set(list(approved.keys()) + list(old_pos.keys())))
        if turnover > self.limits.max_daily_turnover:
            # Scale down the CHANGE, not the position
            scale = self.limits.max_daily_turnover / turnover
            for t in approved:
                change = approved[t] - old_pos.get(t, 0.0)
                approved[t] = old_pos.get(t, 0.0) + change * scale

        return approved

    def halt(self, reason: str) -> None:
        """Halt all trading. Called automatically on KILL-severity breaches."""
        self.is_halted = True
        self._halt_reason = reason
        logger.critical(f"RISK ENGINE HALTED: {reason}")

    def resume(self) -> None:
        """Resume trading after manual review."""
        self.is_halted = False
        self._halt_reason = None
        logger.info("Risk engine resumed after manual review.")

    @property
    def halt_reason(self) -> Optional[str]:
        return self._halt_reason

    def summary(self) -> str:
        """Return a summary of the engine state."""
        lines = [
            f"RiskEngine Status: {'HALTED (' + self._halt_reason + ')' if self.is_halted else 'ACTIVE'}",
            f"  Total breaches logged: {len(self.breach_log)}",
        ]
        if self.breach_log:
            by_severity = {}
            for b in self.breach_log:
                by_severity.setdefault(b.severity.value, []).append(b)
            for sev in ["KILL", "HARD_LIMIT", "WARNING", "INFO"]:
                if sev in by_severity:
                    lines.append(f"    {sev}: {len(by_severity[sev])} breaches")
        return "\n".join(lines)
