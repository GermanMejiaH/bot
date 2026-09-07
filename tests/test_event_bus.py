"""Unit tests for Stage 2 EventBus and Event models."""

import concurrent.futures
import threading
import time

from dta.events.event_bus import EventBus
from dta.events.events import BaseEvent, FrameCaptured, GameStateUpdated
from dta.models.game_state import GameState


class SampleTestEvent(BaseEvent):
    message: str


def test_event_bus_single_subscription() -> None:
    bus = EventBus()
    received: list[SampleTestEvent] = []

    def handler(event: SampleTestEvent) -> None:
        received.append(event)

    bus.subscribe(SampleTestEvent, handler)
    evt = SampleTestEvent(message="hello")
    bus.publish(evt)

    assert len(received) == 1
    assert received[0].message == "hello"


def test_event_bus_multiple_subscribers() -> None:
    bus = EventBus()
    results_a: list[str] = []
    results_b: list[str] = []

    def handler_a(event: SampleTestEvent) -> None:
        results_a.append(event.message)

    def handler_b(event: SampleTestEvent) -> None:
        results_b.append(event.message)

    bus.subscribe(SampleTestEvent, handler_a)
    bus.subscribe(SampleTestEvent, handler_b)

    bus.publish(SampleTestEvent(message="broadcast"))

    assert results_a == ["broadcast"]
    assert results_b == ["broadcast"]


def test_event_bus_unsubscription() -> None:
    bus = EventBus()
    received: list[str] = []

    def handler(event: SampleTestEvent) -> None:
        received.append(event.message)

    bus.subscribe(SampleTestEvent, handler)
    bus.publish(SampleTestEvent(message="msg1"))
    assert len(received) == 1

    bus.unsubscribe(SampleTestEvent, handler)
    bus.publish(SampleTestEvent(message="msg2"))
    assert len(received) == 1  # Unsubscribed, so count remains 1


def test_specific_event_models() -> None:
    bus = EventBus()
    frame_events: list[FrameCaptured] = []
    state_events: list[GameStateUpdated] = []

    bus.subscribe(FrameCaptured, lambda e: frame_events.append(e))
    bus.subscribe(GameStateUpdated, lambda e: state_events.append(e))

    raw_frame_mock = [0, 1, 2]
    evt_frame = FrameCaptured(frame_id="f_01", frame=raw_frame_mock, width=1920, height=1080)
    evt_state = GameStateUpdated(state=GameState(frame_id="f_01", timestamp=time.time()))

    bus.publish(evt_frame)
    bus.publish(evt_state)

    assert len(frame_events) == 1
    assert frame_events[0].width == 1920
    assert len(state_events) == 1
    assert state_events[0].state.frame_id == "f_01"


def test_event_bus_thread_safety() -> None:
    bus = EventBus()
    counter = 0
    lock = threading.Lock()

    def handler(event: SampleTestEvent) -> None:
        nonlocal counter
        with lock:
            counter += 1

    bus.subscribe(SampleTestEvent, handler)

    def worker(idx: int) -> None:
        for _ in range(50):
            bus.publish(SampleTestEvent(message=f"worker_{idx}"))

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, i) for i in range(8)]
        concurrent.futures.wait(futures)

    assert counter == 400
