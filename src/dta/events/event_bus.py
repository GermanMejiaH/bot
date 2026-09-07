"""Thread-safe event bus implementation supporting type-based event subscription and publishing."""

import threading
from collections import defaultdict
from collections.abc import Callable
from functools import lru_cache
from typing import Any, TypeVar

from dta.core.logger import logger

T = TypeVar("T")


class EventBus:
    """Central thread-safe publish-subscribe EventBus."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._subscribers: dict[type[Any], list[Callable[[Any], None]]] = defaultdict(list)

    def subscribe(self, event_type: type[T], handler: Callable[[T], None]) -> None:
        """Subscribe a callback handler to a specific event type."""
        with self._lock:
            handlers = self._subscribers[event_type]
            if handler not in handlers:
                handlers.append(handler)
                logger.debug(f"Subscribed {getattr(handler, '__name__', str(handler))} to event {event_type.__name__}")

    def unsubscribe(self, event_type: type[T], handler: Callable[[T], None]) -> None:
        """Unsubscribe a callback handler from an event type."""
        with self._lock:
            handlers = self._subscribers[event_type]
            if handler in handlers:
                handlers.remove(handler)
                logger.debug(f"Unsubscribed {getattr(handler, '__name__', str(handler))} from event {event_type.__name__}")

    def publish(self, event: Any) -> None:
        """Publish an event instance to all registered handlers for its type."""
        event_type = type(event)
        with self._lock:
            # Create a defensive copy of handlers under lock to prevent mutation issues during invocation
            handlers = list(self._subscribers.get(event_type, []))

        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                handler_name = getattr(handler, '__name__', str(handler))
                logger.error(f"Error executing handler {handler_name} for event {event_type.__name__}: {e}")


    def clear(self) -> None:
        """Clear all subscribers (useful for unit test isolation)."""
        with self._lock:
            self._subscribers.clear()


@lru_cache(maxsize=1)
def get_event_bus() -> EventBus:
    """Return singleton EventBus instance."""
    return EventBus()
