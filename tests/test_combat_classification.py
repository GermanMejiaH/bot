"""Unit tests for Phase 2 Evidence-Driven CombatClassifier scoring and classification logic."""

from dta.models.combat import CombatState
from dta.models.detections import (
    BoundingBox,
    DetectionBatch,
    RawCharacterDetection,
    ResourceDetection,
)
from dta.models.game_state import GameState
from dta.models.perception_state import PerceptionState
from dta.services.combat_classifier import CombatClassifier


def test_combat_classifier_empty_exploration_map():
    """Verify empty exploration map scores 0 and classifies as exploration (is_combat=False)."""
    classifier = CombatClassifier()

    perc = PerceptionState(frame_id="f1")
    state = GameState(frame_id="f1", perception_state=perc)

    evidence = classifier.evaluate(state)

    assert evidence.score == 0
    assert evidence.is_combat is False
    assert evidence.combat_threshold == 4
    assert evidence.combat_base_count == 0
    assert evidence.pa_detected == 0
    assert evidence.pm_detected == 0
    assert evidence.turn_indicator_detected is False


def test_combat_classifier_single_combat_base_ignored():
    """Verify exploration map with a single false combat_base scores 0 and classifies as exploration."""
    classifier = CombatClassifier()

    det = RawCharacterDetection(
        bbox=BoundingBox(x=100, y=100, w=20, h=40),
        centroid=(110, 120),
        area=800.0,
        method="combat_base",
    )
    batch = DetectionBatch(frame_id="f2", raw_characters=[det])
    perc = PerceptionState(frame_id="f2", raw_detections=batch)
    state = GameState(frame_id="f2", perception_state=perc)

    evidence = classifier.evaluate(state)

    assert evidence.combat_base_count == 1
    assert evidence.score == 0
    assert evidence.is_combat is False
    assert "single_combat_base_ignored" in evidence.reason


def test_combat_classifier_noisy_ocr_single_resource():
    """Verify noisy OCR with single PA=6 and 0 bases scores 1 (< 4) and classifies as exploration."""
    classifier = CombatClassifier()

    perc = PerceptionState(frame_id="f3", resources=ResourceDetection(pa=6, pm=0))
    state = GameState(frame_id="f3", perception_state=perc)

    evidence = classifier.evaluate(state)

    assert evidence.pa_detected == 6
    assert evidence.pm_detected == 0
    assert evidence.score == 1
    assert evidence.is_combat is False


def test_combat_classifier_user_explicit_pa11_pm6_single_base():
    """User Explicit Requirement: PA=11, PM=6, combat_base_count=1, turn_indicator=False -> score=2, is_combat=False."""
    classifier = CombatClassifier()

    det = RawCharacterDetection(
        bbox=BoundingBox(x=100, y=100, w=20, h=40),
        centroid=(110, 120),
        area=800.0,
        method="combat_base",
    )
    batch = DetectionBatch(frame_id="f4", raw_characters=[det])
    perc = PerceptionState(frame_id="f4", resources=ResourceDetection(pa=11, pm=6), raw_detections=batch)
    state = GameState(frame_id="f4", perception_state=perc)

    evidence = classifier.evaluate(state)

    assert evidence.pa_detected == 11
    assert evidence.pm_detected == 6
    assert evidence.combat_base_count == 1
    assert evidence.turn_indicator_detected is False
    assert evidence.score == 2
    assert evidence.combat_threshold == 4
    assert evidence.is_combat is False
    assert "single_combat_base_ignored" in evidence.reason


def test_combat_classifier_active_turn_indicator():
    """Verify active turn indicator (combat_state model) scores 4 and classifies as combat."""
    classifier = CombatClassifier()

    c_state = CombatState()
    state = GameState(frame_id="f5", combat_state=c_state)

    evidence = classifier.evaluate(state)

    assert evidence.turn_indicator_detected is True
    assert evidence.score == 4
    assert evidence.is_combat is True


def test_combat_classifier_real_combat_pa_pm_multi_bases():
    """Verify real combat with PA=11, PM=6, and 2 combat bases scores 4 (1 PA + 1 PM + 2 bases) and classifies as combat."""
    classifier = CombatClassifier()

    det1 = RawCharacterDetection(bbox=BoundingBox(x=100, y=100, w=20, h=40), centroid=(110, 120), area=800.0, method="combat_base")
    det2 = RawCharacterDetection(bbox=BoundingBox(x=200, y=200, w=20, h=40), centroid=(210, 220), area=800.0, method="combat_base")

    batch = DetectionBatch(frame_id="f6", raw_characters=[det1, det2])
    perc = PerceptionState(frame_id="f6", resources=ResourceDetection(pa=11, pm=6), raw_detections=batch)
    state = GameState(frame_id="f6", perception_state=perc)

    evidence = classifier.evaluate(state)

    assert evidence.pa_detected == 11
    assert evidence.pm_detected == 6
    assert evidence.combat_base_count == 2
    assert evidence.score == 4
    assert evidence.is_combat is True
