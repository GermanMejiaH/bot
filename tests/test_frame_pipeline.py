"""Unit tests for Stage 8 FramePipeline using mocks."""

import time
from unittest.mock import MagicMock

import numpy as np

from dta.detectors.character_detector import CharacterDetector
from dta.detectors.resource_detector import ResourceDetector
from dta.events.event_bus import EventBus
from dta.events.events import FrameCaptured, GameStateUpdated
from dta.models.detections import BoundingBox, RawCharacterDetection, ResourceDetection
from dta.models.frame import CapturedFrame, FrameMetadata
from dta.models.game_state import GameState
from dta.models.perception_state import PerceptionState
from dta.services.frame_pipeline import FramePipeline, PipelineMetrics
from dta.state.state_builder import StateBuilder
from dta.tracking.entity_tracker import EntityTracker, TrackedEntity


def test_frame_pipeline_process_frame_and_metrics() -> None:
    mock_resource_detector = MagicMock(spec=ResourceDetector)
    mock_resource_detector.detect_resources.return_value = ResourceDetection(pa=12, pm=6, hp=1000)

    mock_char_detector = MagicMock(spec=CharacterDetector)
    bbox = BoundingBox(x=10, y=10, w=20, h=20)
    raw_det = RawCharacterDetection(bbox=bbox, centroid=bbox.centroid, area=400.0)
    mock_char_detector.detect_characters.return_value = [raw_det]

    mock_tracker = MagicMock(spec=EntityTracker)
    tracked = TrackedEntity(entity_id="entity_0001", centroid=(20, 20), bbox=bbox)
    mock_tracker.update.return_value = [tracked]

    mock_state_builder = MagicMock(spec=StateBuilder)
    dummy_game_state = GameState(
        frame_id="frame_0001",
        perception_state=PerceptionState(frame_id="frame_0001", tracked_entities=[tracked]),
    )
    mock_state_builder.build_game_state.return_value = dummy_game_state

    event_bus = EventBus()
    published_events: list[GameStateUpdated] = []
    event_bus.subscribe(GameStateUpdated, lambda e: published_events.append(e))

    pipeline = FramePipeline(
        resource_detector=mock_resource_detector,
        character_detector=mock_char_detector,
        entity_tracker=mock_tracker,
        state_builder=mock_state_builder,
        event_bus=event_bus,
    )

    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    captured_frame = CapturedFrame(
        metadata=FrameMetadata(frame_id="frame_0001", width=100, height=100),
        image=dummy_image,
    )

    state = pipeline.process_frame(captured_frame)

    assert state is dummy_game_state
    assert len(published_events) == 1
    assert published_events[0].state is dummy_game_state

    metrics = pipeline.get_latest_metrics()
    assert isinstance(metrics, PipelineMetrics)
    assert metrics.total_pipeline_time_ms > 0.0


def test_frame_pipeline_ocr_throttling_cache() -> None:
    mock_resource_detector = MagicMock(spec=ResourceDetector)
    mock_resource_detector.detect_resources.return_value = ResourceDetection(pa=12, pm=6)

    mock_state_builder = MagicMock(spec=StateBuilder)
    dummy_game_state = GameState(
        frame_id="f1",
        perception_state=PerceptionState(frame_id="f1"),
    )
    mock_state_builder.build_game_state.return_value = dummy_game_state

    pipeline = FramePipeline(
        resource_detector=mock_resource_detector,
        character_detector=MagicMock(spec=CharacterDetector, detect_characters=MagicMock(return_value=[])),
        entity_tracker=MagicMock(spec=EntityTracker, update=MagicMock(return_value=[])),
        state_builder=mock_state_builder,
        event_bus=EventBus(),
    )
    pipeline.ocr_update_interval = 0.25

    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    frame1 = CapturedFrame(metadata=FrameMetadata(frame_id="f1", width=100, height=100), image=dummy_image)
    frame2 = CapturedFrame(metadata=FrameMetadata(frame_id="f2", width=100, height=100), image=dummy_image)

    # Frame 1: Triggers initial OCR call
    pipeline.process_frame(frame1)
    assert mock_resource_detector.detect_resources.call_count == 1

    # Frame 2: Processed immediately (< 0.25s), reuses cached OCR without calling detect_resources
    pipeline.process_frame(frame2)
    assert mock_resource_detector.detect_resources.call_count == 1  # Still 1 call!

    # Simulate time passing > 0.25s
    pipeline._last_ocr_time = time.perf_counter() - 0.30
    frame3 = CapturedFrame(metadata=FrameMetadata(frame_id="f3", width=100, height=100), image=dummy_image)
    pipeline.process_frame(frame3)
    assert mock_resource_detector.detect_resources.call_count == 2  # Called second time!



def test_frame_pipeline_event_bus_listener() -> None:
    event_bus = EventBus()

    mock_resource_detector = MagicMock(spec=ResourceDetector)
    mock_resource_detector.detect_resources.return_value = ResourceDetection(pa=12, pm=6)

    mock_state_builder = MagicMock(spec=StateBuilder)
    dummy_game_state = GameState(
        frame_id="f_evt_1",
        perception_state=PerceptionState(frame_id="f_evt_1"),
    )
    mock_state_builder.build_game_state.return_value = dummy_game_state

    pipeline = FramePipeline(
        resource_detector=mock_resource_detector,
        character_detector=MagicMock(spec=CharacterDetector, detect_characters=MagicMock(return_value=[])),
        entity_tracker=MagicMock(spec=EntityTracker, update=MagicMock(return_value=[])),
        state_builder=mock_state_builder,
        event_bus=event_bus,
    )

    published_game_states: list[GameStateUpdated] = []
    event_bus.subscribe(GameStateUpdated, lambda e: published_game_states.append(e))

    pipeline.start_listening()
    assert pipeline.is_listening is True

    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    frame_evt = FrameCaptured(
        frame_id="f_evt_1",
        frame=dummy_image,
        width=100,
        height=100,
        captured_frame=CapturedFrame(
            metadata=FrameMetadata(frame_id="f_evt_1", width=100, height=100),
            image=dummy_image,
        ),
    )

    event_bus.publish(frame_evt)
    assert len(published_game_states) == 1

    pipeline.stop_listening()
    assert pipeline.is_listening is False


def test_frame_pipeline_ocr_exception_fallback() -> None:
    mock_resource_detector = MagicMock(spec=ResourceDetector)
    # First call returns valid detection, second call raises exception
    mock_resource_detector.detect_resources.side_effect = [
        ResourceDetection(pa=12, pm=6),
        RuntimeError("OCR Provider error"),
    ]

    mock_state_builder = MagicMock(spec=StateBuilder)
    dummy_game_state = GameState(frame_id="f1", perception_state=PerceptionState(frame_id="f1"))
    mock_state_builder.build_game_state.return_value = dummy_game_state

    pipeline = FramePipeline(
        resource_detector=mock_resource_detector,
        character_detector=MagicMock(spec=CharacterDetector, detect_characters=MagicMock(return_value=[])),
        entity_tracker=MagicMock(spec=EntityTracker, update=MagicMock(return_value=[])),
        state_builder=mock_state_builder,
        event_bus=EventBus(),
    )
    pipeline.ocr_update_interval = 0.1

    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    frame1 = CapturedFrame(metadata=FrameMetadata(frame_id="f1", width=100, height=100), image=dummy_image)
    frame2 = CapturedFrame(metadata=FrameMetadata(frame_id="f2", width=100, height=100), image=dummy_image)

    # Frame 1: Valid OCR
    state1 = pipeline.process_frame(frame1)
    assert state1 is dummy_game_state
    assert pipeline._cached_resources.pa == 12

    # Frame 2: Trigger exception in OCR (force interval expiry)
    pipeline._last_ocr_time = time.perf_counter() - 0.2
    # Frame 2 should catch exception, fallback to cached resources (pa=12), and not crash
    state2 = pipeline.process_frame(frame2)
    assert state2 is dummy_game_state
    assert pipeline._cached_resources.pa == 12


