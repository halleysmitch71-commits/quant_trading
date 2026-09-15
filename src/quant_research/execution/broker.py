"""
quant_research.execution.broker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Abstract broker interface and backtest implementation.

Architecture:
  BacktestEngine / LiveEngine
     │
     ├── uses same → SignalGenerator
     ├── uses same → PortfolioAllocator
     ├── uses same → RiskEngine
     │
     └── differs only in → Broker
          ├── BacktestBroker  (historical fills)
          ├── PaperBroker     (simulated live fills)
          └── LiveBroker      (real exchange API — future)
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from typing import Optional

import numpy as np

from quant_research.core.events import (
    FillEvent, OrderEvent, OrderSide, OrderType,
)

logger = logging.getLogger(__name__)


class Broker(ABC):
    """Abstract broker interface.

    All concrete brokers must implement these methods.
    The trading engine uses this interface exclusively,
    enabling the same logic for backtest, paper, and live trading.
    """

    @abstractmethod
    def submit_order(self, order: OrderEvent) -> str:
        """Submit an order. Returns order_id."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order. Returns True if cancelled."""

    @abstractmethod
    def cancel_all_orders(self) -> int:
        """Cancel all pending orders. Returns count cancelled."""

    @abstractmethod
    def get_position(self, ticker: str) -> float:
        """Get current position for a ticker (in units)."""

    @abstractmethod
    def get_positions(self) -> dict[str, float]:
        """Get all current positions."""

    @abstractmethod
    def get_fills(self, since: Optional[datetime] = None) -> list[FillEvent]:
        """Get recent fills, optionally since a timestamp."""

    @abstractmethod
    def get_equity(self) -> float:
        """Get current account equity."""

    def flatten_all(self) -> list[str]:
        """Close all positions. Returns list of order IDs."""
        order_ids = []
        for ticker, qty in self.get_positions().items():
            if abs(qty) > 1e-10:
                side = OrderSide.SELL if qty > 0 else OrderSide.BUY
                order = OrderEvent(
                    timestamp=datetime.now(),
                    ticker=ticker,
                    side=side,
                    quantity=abs(qty),
                    order_type=OrderType.MARKET,
                )
                oid = self.submit_order(order)
                order_ids.append(oid)
        return order_ids


class BacktestBroker(Broker):
    """Simulated broker for backtesting against historical prices.

    Fills orders instantly at the current bar's close price,
    matching the behavior of the existing `run_backtest()` engine.

    Parameters
    ----------
    initial_capital : float
        Starting account equity.
    commission_bps : float
        One-way commission in basis points.
    slippage_bps : float
        One-way slippage in basis points.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission_bps: float = 0.0,
        slippage_bps: float = 0.0,
    ):
        self._equity = initial_capital
        self._initial_capital = initial_capital
        self._commission_bps = commission_bps
        self._slippage_bps = slippage_bps

        self._positions: dict[str, float] = defaultdict(float)  # ticker → units
        self._fills: list[FillEvent] = []
        self._pending_orders: dict[str, OrderEvent] = {}

        # Current market prices (set externally each bar)
        self._current_prices: dict[str, float] = {}

    def set_prices(self, prices: dict[str, float]) -> None:
        """Update current market prices (called each bar)."""
        self._current_prices = dict(prices)

    def submit_order(self, order: OrderEvent) -> str:
        """Submit and immediately fill at current price (backtest)."""
        order_id = str(uuid.uuid4())[:8]

        price = self._current_prices.get(order.ticker)
        if price is None or price <= 0:
            logger.warning(f"No price for {order.ticker}, order rejected")
            return order_id

        # Apply slippage
        slip = price * self._slippage_bps / 10_000
        if order.side == OrderSide.BUY:
            fill_price = price + slip
        else:
            fill_price = price - slip

        # Signed quantity
        fill_qty = order.quantity if order.side == OrderSide.BUY else -order.quantity

        # Commission
        commission = abs(fill_qty) * fill_price * self._commission_bps / 10_000

        # Update position
        self._positions[order.ticker] += fill_qty

        # Update equity
        self._equity -= commission

        fill = FillEvent(
            timestamp=order.timestamp,
            ticker=order.ticker,
            order_id=order_id,
            fill_price=fill_price,
            fill_quantity=fill_qty,
            commission=commission,
            slippage=abs(slip * fill_qty),
        )
        self._fills.append(fill)

        return order_id

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order (no pending orders in backtest)."""
        if order_id in self._pending_orders:
            del self._pending_orders[order_id]
            return True
        return False

    def cancel_all_orders(self) -> int:
        """Cancel all pending orders."""
        n = len(self._pending_orders)
        self._pending_orders.clear()
        return n

    def get_position(self, ticker: str) -> float:
        return self._positions.get(ticker, 0.0)

    def get_positions(self) -> dict[str, float]:
        return dict(self._positions)

    def get_fills(self, since: Optional[datetime] = None) -> list[FillEvent]:
        if since is None:
            return list(self._fills)
        return [f for f in self._fills if f.timestamp >= since]

    def get_equity(self) -> float:
        # Mark positions to market
        pnl = sum(
            qty * self._current_prices.get(ticker, 0.0)
            for ticker, qty in self._positions.items()
        )
        return self._equity + pnl

    def reset(self) -> None:
        """Reset broker to initial state."""
        self._equity = self._initial_capital
        self._positions.clear()
        self._fills.clear()
        self._pending_orders.clear()


class PaperBroker(Broker):
    """Simulated broker for paper trading against live/delayed prices.

    Same as BacktestBroker but designed to receive real-time price updates
    and track P&L in real-time.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission_bps: float = 10.0,
        slippage_bps: float = 5.0,
    ):
        self._inner = BacktestBroker(
            initial_capital=initial_capital,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
        )
        self._trade_log: list[dict] = []

    def update_prices(self, prices: dict[str, float]) -> None:
        """Update prices from live feed."""
        self._inner.set_prices(prices)

    def submit_order(self, order: OrderEvent) -> str:
        oid = self._inner.submit_order(order)
        self._trade_log.append({
            "timestamp": order.timestamp.isoformat(),
            "ticker": order.ticker,
            "side": order.side.value,
            "quantity": order.quantity,
            "order_id": oid,
        })
        return oid

    def cancel_order(self, order_id: str) -> bool:
        return self._inner.cancel_order(order_id)

    def cancel_all_orders(self) -> int:
        return self._inner.cancel_all_orders()

    def get_position(self, ticker: str) -> float:
        return self._inner.get_position(ticker)

    def get_positions(self) -> dict[str, float]:
        return self._inner.get_positions()

    def get_fills(self, since: Optional[datetime] = None) -> list[FillEvent]:
        return self._inner.get_fills(since)

    def get_equity(self) -> float:
        return self._inner.get_equity()

    @property
    def trade_log(self) -> list[dict]:
        return list(self._trade_log)
