"""Unit tests for Stage 8.5 DebugOverlay component."""

import os
from pathlib import Path

import numpy as np

from dta.events.event_bus import EventBus
from dta.events.events import GameStateUpdated
from dta.models.detections import BoundingBox, ResourceDetection
from dta.models.game_state import GameState
from dta.models.perception_state import PerceptionState
from dta.tracking.entity_tracker import TrackedEntity
from dta.visualization.debug_overlay import DebugOverlay


def test_debug_overlay_subscription_lifecycle() -> None:
    event_bus = EventBus()
    overlay = DebugOverlay(event_bus=event_bus)

    assert overlay.is_subscribed is False
    overlay.start_listening()
    assert overlay.is_subscribed is True

    overlay.stop_listening()
    assert overlay.is_subscribed is False


def test_debug_overlay_draw_overlay() -> None:
    dummy_image = np.zeros((400, 600, 3), dtype=np.uint8)

    bbox = BoundingBox(x=50, y=50, w=40, h=40)
    entity = TrackedEntity(
        entity_id="entity_0001",
        centroid=(70, 70),
        bbox=bbox,
        tracking_confidence=0.95,
    )
    perception = PerceptionState(
        frame_id="f1",
        resources=ResourceDetection(pa=12, pm=6, hp=1050),
        tracked_entities=[entity],
    )
    game_state = GameState(
        frame_id="f1",
        perception_state=perception,
        frame_image=dummy_image,
    )

    overlay = DebugOverlay()
    annotated = overlay.draw_overlay(game_state)

    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == dummy_image.shape
    # Canvas top HUD rectangle was rendered (non-zero pixels in top banner)
    assert np.any(annotated[:45, :, :] > 0)


def test_debug_overlay_on_game_state_updated_event() -> None:
    event_bus = EventBus()
    overlay = DebugOverlay(event_bus=event_bus)
    overlay.start_listening()

    dummy_image = np.zeros((300, 300, 3), dtype=np.uint8)
    game_state = GameState(
        frame_id="f_evt_1",
        perception_state=PerceptionState(frame_id="f_evt_1"),
        frame_image=dummy_image,
    )

    event_bus.publish(GameStateUpdated(state=game_state))

    assert overlay.latest_game_state is game_state
    assert overlay.latest_annotated_frame is not None
    assert overlay.latest_annotated_frame.shape == dummy_image.shape

    overlay.close()


def test_debug_overlay_save_screenshot(tmp_path: Path) -> None:

    test_dir = str(tmp_path / "debug_screenshots")
    overlay = DebugOverlay(output_dir=test_dir)

    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    saved_file = overlay.save_debug_screenshot(dummy_image, "test_f1")

    assert os.path.exists(saved_file)
    assert os.path.isfile(saved_file)
    assert "test_f1" in os.path.basename(saved_file)

