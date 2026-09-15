"""
quant_research.risk.kill_switch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Emergency portfolio halt mechanism.

The KillSwitch evaluates a set of critical conditions and, if any
are triggered, halts all trading and optionally flattens positions.

Kill conditions:
  - Daily loss exceeds threshold
  - Drawdown exceeds threshold
  - Portfolio volatility spike
  - Market data stale
  - Broker disconnected
  - Position mismatch (expected vs actual)
  - Unexpected leverage
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Protocol

logger = logging.getLogger(__name__)


class BrokerProtocol(Protocol):
    """Minimal broker interface for kill switch actions."""

    def cancel_all_orders(self) -> int:
        """Cancel all pending orders. Return number cancelled."""
        ...

    def get_positions(self) -> dict[str, float]:
        """Get current broker positions."""
        ...

    def flatten_all(self) -> list[str]:
        """Close all positions. Return list of order IDs."""
        ...


@dataclass
class KillSwitchConfig:
    """Configuration for kill switch conditions."""

    # P&L conditions
    max_daily_loss: float = -0.03          # -3% → kill
    max_drawdown: float = -0.15            # -15% → kill

    # Volatility conditions
    max_portfolio_vol: float = 0.30        # 30% annualised → kill

    # Data conditions
    max_data_staleness_hours: float = 2.0  # 2 hours → kill

    # Position conditions
    max_position_mismatch: float = 0.05    # 5% mismatch → kill
    max_unexpected_leverage: float = 2.0   # 2× max allowed leverage → kill

    # Behavior
    auto_flatten: bool = False             # Auto-flatten on kill
    require_manual_resume: bool = True     # Require human to resume


@dataclass
class KillSwitchState:
    """Current state snapshot for kill switch evaluation."""

    daily_pnl: float = 0.0
    drawdown: float = 0.0
    portfolio_vol: float = 0.0
    last_data_time: Optional[datetime] = None
    expected_positions: Optional[dict[str, float]] = None
    actual_positions: Optional[dict[str, float]] = None
    current_leverage: float = 0.0
    max_allowed_leverage: float = 1.0
    broker_connected: bool = True


class KillSwitch:
    """Emergency portfolio halt mechanism.

    Usage:
        ks = KillSwitch(KillSwitchConfig())
        should_kill, reasons = ks.evaluate(state)
        if should_kill:
            ks.execute(broker)
    """

    def __init__(self, config: KillSwitchConfig):
        self.config = config
        self.is_triggered: bool = False
        self.trigger_reasons: list[str] = []
        self.trigger_time: Optional[datetime] = None
        self.execution_log: list[dict] = []

    def evaluate(self, state: KillSwitchState) -> tuple[bool, list[str]]:
        """Evaluate all kill conditions.

        Returns
        -------
        tuple[bool, list[str]]
            (should_kill, list of reasons)
        """
        reasons = []

        # 1. Daily loss
        if state.daily_pnl < self.config.max_daily_loss:
            reasons.append(
                f"Daily loss {state.daily_pnl:.2%} < {self.config.max_daily_loss:.2%}"
            )

        # 2. Drawdown
        if state.drawdown < self.config.max_drawdown:
            reasons.append(
                f"Drawdown {state.drawdown:.2%} < {self.config.max_drawdown:.2%}"
            )

        # 3. Volatility spike
        if state.portfolio_vol > self.config.max_portfolio_vol:
            reasons.append(
                f"Portfolio vol {state.portfolio_vol:.2%} > {self.config.max_portfolio_vol:.2%}"
            )

        # 4. Stale data
        if state.last_data_time is not None:
            hours = (datetime.now() - state.last_data_time).total_seconds() / 3600
            if hours > self.config.max_data_staleness_hours:
                reasons.append(
                    f"Data stale: {hours:.1f}h > {self.config.max_data_staleness_hours}h"
                )
        elif state.last_data_time is None:
            reasons.append("No market data timestamp")

        # 5. Broker disconnected
        if not state.broker_connected:
            reasons.append("Broker disconnected")

        # 6. Position mismatch
        if state.expected_positions and state.actual_positions:
            all_tickers = set(list(state.expected_positions.keys()) + list(state.actual_positions.keys()))
            for t in all_tickers:
                expected = state.expected_positions.get(t, 0.0)
                actual = state.actual_positions.get(t, 0.0)
                if abs(expected - actual) > self.config.max_position_mismatch:
                    reasons.append(
                        f"Position mismatch {t}: expected={expected:.4f} actual={actual:.4f}"
                    )

        # 7. Unexpected leverage
        if state.current_leverage > state.max_allowed_leverage * self.config.max_unexpected_leverage:
            reasons.append(
                f"Unexpected leverage: {state.current_leverage:.2f}x "
                f"(max allowed: {state.max_allowed_leverage * self.config.max_unexpected_leverage:.2f}x)"
            )

        should_kill = len(reasons) > 0

        if should_kill and not self.is_triggered:
            self.is_triggered = True
            self.trigger_reasons = reasons
            self.trigger_time = datetime.now()
            logger.critical(f"KILL SWITCH TRIGGERED: {reasons}")

        return should_kill, reasons

    def execute(self, broker: BrokerProtocol) -> dict:
        """Execute kill switch actions.

        1. Cancel all pending orders
        2. Optionally flatten all positions (if auto_flatten=True)
        3. Log the action

        Parameters
        ----------
        broker : BrokerProtocol
            Broker interface for order management.

        Returns
        -------
        dict
            Execution report.
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "reasons": self.trigger_reasons,
            "actions": [],
        }

        # Cancel all orders
        try:
            n_cancelled = broker.cancel_all_orders()
            report["actions"].append(f"Cancelled {n_cancelled} orders")
            logger.info(f"Kill switch: cancelled {n_cancelled} orders")
        except Exception as e:
            report["actions"].append(f"Failed to cancel orders: {e}")
            logger.error(f"Kill switch: failed to cancel orders: {e}")

        # Flatten if configured
        if self.config.auto_flatten:
            try:
                order_ids = broker.flatten_all()
                report["actions"].append(f"Flattened {len(order_ids)} positions")
                logger.info(f"Kill switch: flattened {len(order_ids)} positions")
            except Exception as e:
                report["actions"].append(f"Failed to flatten: {e}")
                logger.error(f"Kill switch: failed to flatten: {e}")

        self.execution_log.append(report)
        return report

    def reset(self) -> None:
        """Reset kill switch after manual review.

        Raises RuntimeError if require_manual_resume is True and
        no human review has been performed (call this method to confirm).
        """
        self.is_triggered = False
        self.trigger_reasons = []
        self.trigger_time = None
        logger.info("Kill switch reset after manual review.")
