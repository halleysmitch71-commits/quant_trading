"""
quant_research.live.engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Event-driven trading engine for live/paper trading.

This engine reuses the same signal generation, portfolio allocation,
and risk management logic as the backtest — only the Broker differs.

Architecture:
  ┌──────────────┐
  │  DataHandler  │  ← market data feed (live or replay)
  └──────┬───────┘
         │ MarketDataEvent
  ┌──────▼───────┐
  │ SignalEngine  │  ← same signal logic as backtest
  └──────┬───────┘
         │ SignalEvent
  ┌──────▼────────┐
  │PortfolioEngine│  ← same allocation logic
  └──────┬────────┘
         │ target positions
  ┌──────▼───────┐
  │  RiskEngine   │  ← same risk checks (M18)
  └──────┬───────┘
         │ approved positions
  ┌──────▼───────┐
  │   Broker      │  ← ONLY DIFFERENCE: Backtest vs Paper vs Live
  └──────────────┘
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd

from quant_research.core.bus import EventBus
from quant_research.core.events import (
    FillEvent, MarketDataEvent, OrderEvent, OrderSide, OrderType, SignalEvent,
)
from quant_research.execution.broker import Broker
from quant_research.risk.engine import PortfolioState, RiskEngine, RiskLimits
from quant_research.signals.base import BaseSignal

logger = logging.getLogger(__name__)


@dataclass
class TradingEngineConfig:
    """Configuration for the trading engine."""
    initial_capital: float = 100_000.0
    risk_limits: RiskLimits = field(default_factory=RiskLimits)
    rebalance_frequency: str = "daily"  # "daily" | "hourly" | "on_signal"


class TradingEngine:
    """Event-driven trading engine.

    Can run in three modes:
      1. Backtest: replay historical data through BacktestBroker
      2. Paper: process live data through PaperBroker
      3. Live: process live data through LiveBroker (future)

    All three modes share the same signal, portfolio, and risk logic.
    """

    def __init__(
        self,
        broker: Broker,
        signals: dict[str, tuple[str, BaseSignal]],  # name → (ticker, signal_obj)
        config: TradingEngineConfig = None,
    ):
        self.broker = broker
        self.signals = signals  # signal_name → (ticker, signal_obj)
        self.config = config or TradingEngineConfig()

        self.bus = EventBus()
        self.risk_engine = RiskEngine(self.config.risk_limits)

        # State
        self._equity_curve: list[float] = [self.config.initial_capital]
        self._current_signals: dict[str, float] = {}
        self._market_data_buffer: dict[str, list[dict]] = {}
        self._bar_count = 0

        # Wire up event handlers
        self.bus.subscribe(MarketDataEvent, self._on_market_data)
        self.bus.subscribe(SignalEvent, self._on_signal)
        self.bus.subscribe(FillEvent, self._on_fill)

    def _on_market_data(self, event: MarketDataEvent) -> None:
        """Process incoming market data."""
        # Buffer data for signal generation
        if event.ticker not in self._market_data_buffer:
            self._market_data_buffer[event.ticker] = []
        self._market_data_buffer[event.ticker].append({
            "open": event.open,
            "high": event.high,
            "low": event.low,
            "close": event.close,
            "volume": event.volume,
        })

    def _on_signal(self, event: SignalEvent) -> None:
        """Process signal event."""
        self._current_signals[event.signal_name] = event.signal_value
        logger.debug(f"Signal received: {event}")

    def _on_fill(self, event: FillEvent) -> None:
        """Process fill event."""
        self._equity_curve.append(self.broker.get_equity())
        logger.debug(f"Fill received: {event}")

    def generate_signals(self, timestamp: datetime) -> dict[str, float]:
        """Generate signals from buffered market data."""
        result = {}
        for sig_name, (ticker, sig_obj) in self.signals.items():
            if ticker in self._market_data_buffer:
                # Convert buffer to DataFrame for signal generation
                df = pd.DataFrame(self._market_data_buffer[ticker])
                if len(df) > 0:
                    signal_series = sig_obj.generate(df)
                    if len(signal_series) > 0:
                        value = signal_series.iloc[-1]
                        result[sig_name] = value

                        # Publish signal event
                        self.bus.publish(SignalEvent(
                            timestamp=timestamp,
                            ticker=ticker,
                            signal_value=value,
                            signal_name=sig_name,
                        ))
        return result

    def compute_target_positions(
        self,
        signals: dict[str, float],
    ) -> dict[str, float]:
        """Convert signals to target position weights.

        Simple equal-weight implementation. Override for sleeve-based.
        """
        n = len(signals)
        if n == 0:
            return {}

        weight = 1.0 / n
        target = {}
        for sig_name, sig_value in signals.items():
            ticker = self.signals[sig_name][0]
            if ticker not in target:
                target[ticker] = 0.0
            target[ticker] += sig_value * weight

        return target

    def rebalance(self, timestamp: datetime) -> dict[str, float]:
        """Full rebalance cycle: signals → positions → risk → orders."""
        if self.risk_engine.is_halted:
            logger.critical("Risk engine halted. Skipping rebalance.")
            return {}

        # 1. Generate signals
        signals = self.generate_signals(timestamp)

        # 2. Compute target positions
        target_positions = self.compute_target_positions(signals)

        # 3. Risk checks on current state
        current_pos = self.broker.get_positions()
        state = PortfolioState(
            positions=current_pos,
            equity=self.broker.get_equity(),
            last_data_time=timestamp,
        )
        self.risk_engine.check_all(state)

        # 4. Enforce risk limits on target
        approved = self.risk_engine.enforce(target_positions, state)

        # 5. Generate orders for position changes
        for ticker, target_weight in approved.items():
            current_weight = current_pos.get(ticker, 0.0)
            delta = target_weight - current_weight

            if abs(delta) < 1e-6:
                continue

            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            order = OrderEvent(
                timestamp=timestamp,
                ticker=ticker,
                side=side,
                quantity=abs(delta),
                order_type=OrderType.MARKET,
            )
            order_id = self.broker.submit_order(order)
            logger.info(f"Order submitted: {order} → {order_id}")

        self._bar_count += 1
        return approved

    def run_backtest(
        self,
        universe: dict[str, pd.DataFrame],
    ) -> dict:
        """Run a full backtest by replaying historical data.

        Parameters
        ----------
        universe : dict[str, pd.DataFrame]
            Ticker → OHLCV DataFrame.

        Returns
        -------
        dict
            Backtest results including equity curve and metrics.
        """
        # Find common dates
        tickers_needed = set()
        for _, (ticker, _) in self.signals.items():
            tickers_needed.add(ticker)

        all_indices = [universe[t].index for t in tickers_needed if t in universe]
        if not all_indices:
            raise ValueError("No data for any signal ticker")

        common_dates = all_indices[0]
        for idx in all_indices[1:]:
            common_dates = common_dates.intersection(idx)
        common_dates = common_dates.sort_values()

        # Replay each bar
        results = []
        for i, date in enumerate(common_dates):
            # Update market data
            prices = {}
            for t in tickers_needed:
                if t in universe:
                    row = universe[t].loc[date]
                    prices[t] = row["close"]

                    event = MarketDataEvent(
                        timestamp=date,
                        ticker=t,
                        open=row.get("open", row["close"]),
                        high=row.get("high", row["close"]),
                        low=row.get("low", row["close"]),
                        close=row["close"],
                        volume=row.get("volume", 0),
                    )
                    self.bus.publish(event)

            # Set broker prices
            if hasattr(self.broker, "set_prices"):
                self.broker.set_prices(prices)

            # Rebalance (skip first few bars for warmup)
            if i >= 60:
                self.rebalance(date)

            results.append({
                "date": date,
                "equity": self.broker.get_equity(),
                "positions": dict(self.broker.get_positions()),
            })

        return {
            "equity_curve": pd.Series(
                {r["date"]: r["equity"] for r in results}
            ),
            "positions": results,
            "fills": self.broker.get_fills(),
            "risk_breaches": self.risk_engine.breach_log,
        }
