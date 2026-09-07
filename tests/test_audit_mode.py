"""Unit tests for Phase 1 TRACE & AUDIT MODE telemetry, candidate lifecycle tracking, and stage export."""

import json
import os
import shutil
import tempfile

import numpy as np

from dta.config.settings import Settings
from dta.detectors.character_detector import CharacterDetector
from dta.models.detections import (
    BoundingBox,
    DetectionBatch,
    RawCharacterDetection,
    ResourceDetection,
)
from dta.models.game_state import GameState
from dta.models.perception_state import PerceptionState
from dta.services.dataset_capture import DatasetCaptureService


def test_character_detector_trace_telemetry():
    """Verify detect_characters_with_trace returns complete candidate lifecycle traces and stage masks."""
    detector = CharacterDetector()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Add synthetic rectangular sprite contours
    frame[200:300, 400:450] = (255, 255, 255)
    frame[350:450, 600:650] = (100, 200, 250)

    trace = detector.detect_characters_with_trace(frame)

    assert "raw_characters" in trace
    assert "rejected_characters" in trace
    assert "all_candidates" in trace
    assert "stage_masks" in trace
    assert "statistics" in trace

    stats = trace["statistics"]
    assert "raw_contours" in stats
    assert "raw_hsv" in stats
    assert "raw_combat_bases" in stats
    assert "after_filtering" in stats
    assert "after_nms" in stats
    assert "final_entities" in stats

    candidates = trace["all_candidates"]
    assert len(candidates) > 0

    for c in candidates:
        assert isinstance(c.candidate_id, int)
        assert hasattr(c, "accepted")
        assert hasattr(c, "reason")
        assert len(c.lifecycle) >= 1
        assert "stage" in c.lifecycle[0]


def test_dataset_capture_combat_status_with_reason():
    """Verify get_combat_status_with_reason returns boolean, reason string, and details dict."""
    from dta.models.combat import CombatState

    settings = Settings()
    service = DatasetCaptureService(settings=settings)

    # Test out-of-combat empty perception state
    empty_perc = PerceptionState(frame_id="f1")
    empty_state = GameState(frame_id="f1", perception_state=empty_perc)

    is_combat, reason, details = service.get_combat_status_with_reason(empty_state)
    assert is_combat is False
    assert "none" in reason
    assert details["pa_detected"] == 0
    assert details["pm_detected"] == 0
    assert details["combat_base_count"] == 0

    # Test PA > 0 resource + turn indicator combat trigger
    pa_perc = PerceptionState(frame_id="f2", resources=ResourceDetection(pa=6, pm=3))
    pa_state = GameState(frame_id="f2", perception_state=pa_perc, combat_state=CombatState())

    is_combat_pa, reason_pa, details_pa = service.get_combat_status_with_reason(pa_state)
    assert is_combat_pa is True
    assert "turn_indicator" in reason_pa
    assert details_pa["pa_detected"] == 6

    # Test single combat_base detection (ignored under score rules)
    base_det = RawCharacterDetection(
        bbox=BoundingBox(x=100, y=100, w=20, h=40),
        centroid=(110, 120),
        area=800.0,
        method="combat_base",
    )
    batch = DetectionBatch(frame_id="f3", raw_characters=[base_det])
    base_perc = PerceptionState(frame_id="f3", raw_detections=batch)
    base_state = GameState(frame_id="f3", perception_state=base_perc)

    is_combat_base, reason_base, details_base = service.get_combat_status_with_reason(base_state)
    assert is_combat_base is False
    assert "single_combat_base_ignored" in reason_base
    assert details_base["combat_base_count"] == 1


def test_dataset_capture_process_audit_frame():
    """Verify process_audit_frame creates original, overlay, stages, crops, excluded_regions, and detections.json."""
    temp_dir = tempfile.mkdtemp()
    try:
        settings = Settings()
        settings.audit.enabled = True
        settings.audit.max_frames = 5
        settings.audit.output_dir = temp_dir

        service = DatasetCaptureService(settings=settings)

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[150:250, 200:250] = (200, 200, 200)

        detector = CharacterDetector()
        trace_res = detector.detect_characters_with_trace(frame)

        batch = DetectionBatch(
            frame_id="audit_f1",
            raw_characters=trace_res["raw_characters"],
            rejected_characters=trace_res["rejected_characters"],
            all_candidates=trace_res["all_candidates"],
            debug_masks=trace_res["stage_masks"],
            statistics=trace_res["statistics"],
        )
        perc = PerceptionState(frame_id="audit_f1", raw_detections=batch)
        state = GameState(frame_id="audit_f1", perception_state=perc, frame_image=frame)

        service.process_audit_frame(state)

        # Check directory existence
        assert os.path.exists(os.path.join(temp_dir, "original", "frame_0001.png"))
        assert os.path.exists(os.path.join(temp_dir, "overlay", "frame_0001.png"))
        assert os.path.exists(os.path.join(temp_dir, "stages", "frame_0001_contours.png"))
        assert os.path.exists(os.path.join(temp_dir, "stages", "frame_0001_hsv.png"))
        assert os.path.exists(os.path.join(temp_dir, "stages", "frame_0001_combat_bases.png"))
        assert os.path.exists(os.path.join(temp_dir, "stages", "frame_0001_final.png"))
        assert os.path.exists(os.path.join(temp_dir, "excluded_regions", "frame_0001_roi_mask.png"))

        json_path = os.path.join(temp_dir, "detections.json")
        assert os.path.exists(json_path)

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        assert data["total_frames_audited"] == 1
        assert len(data["frames"]) == 1
        f1_record = data["frames"][0]

        assert f1_record["frame_id"] == "audit_f1"
        assert "game_state" in f1_record
        assert "combat_reason" in f1_record["game_state"]
        assert "statistics" in f1_record
        assert "candidates" in f1_record
    finally:
        shutil.rmtree(temp_dir)
