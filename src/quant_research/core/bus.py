"""
quant_research.core.bus
~~~~~~~~~~~~~~~~~~~~~~~~
Simple event bus for component communication.

The bus decouples producers from consumers:
  - Components register handlers for specific event types
  - Events are dispatched synchronously (backtest) or asynchronously (live)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable, Type

logger = logging.getLogger(__name__)


class EventBus:
    """Synchronous event bus for component communication.

    Usage:
        bus = EventBus()
        bus.subscribe(MarketDataEvent, my_handler)
        bus.publish(MarketDataEvent(...))  # calls my_handler
    """

    def __init__(self):
        self._handlers: dict[Type, list[Callable]] = defaultdict(list)
        self._event_log: list[Any] = []
        self._log_events: bool = False

    def subscribe(self, event_type: Type, handler: Callable) -> None:
        """Register a handler for an event type.

        Parameters
        ----------
        event_type : Type
            The event class to subscribe to.
        handler : Callable
            Function that takes the event as its single argument.
        """
        self._handlers[event_type].append(handler)
        logger.debug(f"Subscribed {handler.__name__} to {event_type.__name__}")

    def unsubscribe(self, event_type: Type, handler: Callable) -> None:
        """Remove a handler for an event type."""
        if handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)

    def publish(self, event: Any) -> None:
        """Publish an event to all subscribed handlers.

        Handlers are called synchronously in registration order.

        Parameters
        ----------
        event : Any
            The event to dispatch.
        """
        event_type = type(event)

        if self._log_events:
            self._event_log.append(event)

        handlers = self._handlers.get(event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Handler {handler.__name__} failed on {event_type.__name__}: {e}")
                raise

    def clear(self) -> None:
        """Remove all handlers."""
        self._handlers.clear()

    def enable_logging(self) -> None:
        """Start logging all published events."""
        self._log_events = True

    def get_event_log(self) -> list[Any]:
        """Return all logged events."""
        return list(self._event_log)

    @property
    def handler_count(self) -> int:
        """Total number of registered handlers."""
        return sum(len(h) for h in self._handlers.values())
