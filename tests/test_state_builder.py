"""Unit tests for Stage 7 StateBuilder and PerceptionState."""

import time

from dta.models.detections import (
    BoundingBox,
    DetectionBatch,
    RawCharacterDetection,
    ResourceDetection,
)
from dta.models.game_state import GameState
from dta.models.perception_state import PerceptionState
from dta.state.state_builder import StateBuilder
from dta.tracking.entity_tracker import TrackedEntity


def test_state_builder_empty_perception_state() -> None:
    builder = StateBuilder()
    perception_state = builder.build_perception_state(frame_id="frame_0001")

    assert isinstance(perception_state, PerceptionState)
    assert perception_state.frame_id == "frame_0001"
    assert perception_state.resources.pa is None
    assert perception_state.resources.pm is None
    assert len(perception_state.tracked_entities) == 0


def test_state_builder_populated_perception_state() -> None:
    builder = StateBuilder()

    resources = ResourceDetection(pa=12, pm=6, hp=1250)

    bbox = BoundingBox(x=100, y=200, w=30, h=40)
    entity = TrackedEntity(
        entity_id="entity_0001",
        centroid=bbox.centroid,
        bbox=bbox,
        confidence=0.9,
        frames_seen=5,
        frames_missing=1,
    )
    entity.update_confidence()
    assert entity.tracking_confidence == 5.0 / 6.0

    raw_det = RawCharacterDetection(bbox=bbox, centroid=bbox.centroid, area=1200.0)
    batch = DetectionBatch(frame_id="frame_0001", raw_characters=[raw_det], resources=resources)

    perception_state = builder.build_perception_state(
        frame_id="frame_0001",
        resources=resources,
        tracked_entities=[entity],
        raw_detections=batch,
    )

    assert perception_state.resources.pa == 12
    assert perception_state.resources.pm == 6
    assert len(perception_state.tracked_entities) == 1
    assert perception_state.tracked_entities[0].entity_id == "entity_0001"
    assert perception_state.tracked_entities[0].tracking_confidence == 5.0 / 6.0


def test_state_builder_game_state_wrapping() -> None:
    builder = StateBuilder()
    now = time.time()

    resources = ResourceDetection(pa=10, pm=5, hp=800)
    game_state = builder.build_game_state(
        frame_id="frame_0002",
        resources=resources,
        raw_frame_path="screenshots/raw/frame_0002.png",
        debug_frame_path="screenshots/debug/frame_0002.png",
        timestamp=now,
    )

    assert isinstance(game_state, GameState)
    assert game_state.frame_id == "frame_0002"
    assert game_state.timestamp == now
    assert game_state.perception_state is not None
    assert game_state.perception_state.resources.pa == 10
    assert game_state.combat_state is None  # Pure perception builder does not fabricate combat roles
    assert game_state.raw_frame_path == "screenshots/raw/frame_0002.png"
