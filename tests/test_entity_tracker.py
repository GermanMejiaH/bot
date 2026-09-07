"""Unit tests for Stage 6 EntityTracker using synthetic detections."""

from dta.models.detections import BoundingBox, RawCharacterDetection
from dta.tracking.entity_tracker import EntityTracker, TrackedEntity


def test_entity_tracker_registration() -> None:
    tracker = EntityTracker(max_distance_threshold=50.0, max_disappeared_frames=3)
    bbox1 = BoundingBox(x=100, y=100, w=20, h=20)
    det1 = RawCharacterDetection(bbox=bbox1, centroid=bbox1.centroid, area=400.0)

    entities = tracker.update([det1])

    assert len(entities) == 1
    assert isinstance(entities[0], TrackedEntity)
    assert entities[0].entity_id == "entity_0001"
    assert entities[0].centroid == (110, 110)
    assert entities[0].frames_seen == 1
    assert entities[0].frames_missing == 0


def test_entity_tracker_id_persistence() -> None:
    tracker = EntityTracker(max_distance_threshold=50.0)

    # Frame 1: Detection at (100, 100)
    bbox1 = BoundingBox(x=100, y=100, w=20, h=20)
    det1 = RawCharacterDetection(bbox=bbox1, centroid=bbox1.centroid, area=400.0)
    tracker.update([det1])

    # Frame 2: Same entity moved slightly to (105, 105)
    bbox2 = BoundingBox(x=105, y=105, w=20, h=20)
    det2 = RawCharacterDetection(bbox=bbox2, centroid=bbox2.centroid, area=400.0)
    entities = tracker.update([det2])

    assert len(entities) == 1
    assert entities[0].entity_id == "entity_0001"  # ID persisted!
    assert entities[0].centroid == (115, 115)
    assert entities[0].frames_seen == 2
    assert entities[0].frames_missing == 0


def test_entity_tracker_disappearance_and_removal() -> None:
    tracker = EntityTracker(max_distance_threshold=50.0, max_disappeared_frames=2)

    bbox = BoundingBox(x=50, y=50, w=10, h=10)
    det = RawCharacterDetection(bbox=bbox, centroid=bbox.centroid, area=100.0)
    tracker.update([det])

    # Frame 2: Entity disappears
    entities = tracker.update([])
    assert len(entities) == 1
    assert entities[0].frames_missing == 1

    # Frame 3: Entity still missing
    entities = tracker.update([])
    assert len(entities) == 1
    assert entities[0].frames_missing == 2

    # Frame 4: Entity missing exceeds limit (2), removed!
    entities = tracker.update([])
    assert len(entities) == 0


def test_entity_tracker_multiple_entities_and_crossing() -> None:
    tracker = EntityTracker(max_distance_threshold=40.0)

    # Frame 1: Entity A at (50, 50), Entity B at (200, 200)
    bbox_a1 = BoundingBox(x=50, y=50, w=10, h=10)
    bbox_b1 = BoundingBox(x=200, y=200, w=10, h=10)
    det_a1 = RawCharacterDetection(bbox=bbox_a1, centroid=bbox_a1.centroid, area=100.0)
    det_b1 = RawCharacterDetection(bbox=bbox_b1, centroid=bbox_b1.centroid, area=100.0)

    tracker.update([det_a1, det_b1])
    active = tracker.get_active_entities()
    assert len(active) == 2
    id_a = active[0].entity_id
    id_b = active[1].entity_id

    # Frame 2: Entity A moves to (60, 60), Entity B moves to (190, 190)
    bbox_a2 = BoundingBox(x=60, y=60, w=10, h=10)
    bbox_b2 = BoundingBox(x=190, y=190, w=10, h=10)
    det_a2 = RawCharacterDetection(bbox=bbox_a2, centroid=bbox_a2.centroid, area=100.0)
    det_b2 = RawCharacterDetection(bbox=bbox_b2, centroid=bbox_b2.centroid, area=100.0)

    updated = tracker.update([det_a2, det_b2])
    assert len(updated) == 2

    updated_ids = {e.entity_id for e in updated}
    assert updated_ids == {id_a, id_b}



def test_entity_tracker_distance_threshold_exceeded() -> None:
    tracker = EntityTracker(max_distance_threshold=30.0)

    bbox1 = BoundingBox(x=10, y=10, w=10, h=10)
    det1 = RawCharacterDetection(bbox=bbox1, centroid=bbox1.centroid, area=100.0)
    tracker.update([det1])

    # Frame 2: Teleport/far detection at (300, 300) > threshold
    bbox2 = BoundingBox(x=300, y=300, w=10, h=10)
    det2 = RawCharacterDetection(bbox=bbox2, centroid=bbox2.centroid, area=100.0)
    entities = tracker.update([det2])

    # Should register det2 as a new entity (entity_0002) and mark entity_0001 as missing
    assert len(entities) == 2
    ids = [e.entity_id for e in entities]
    assert "entity_0001" in ids
    assert "entity_0002" in ids


def test_entity_tracker_defensive_copying() -> None:
    tracker = EntityTracker(max_distance_threshold=50.0)

    bbox1 = BoundingBox(x=10, y=10, w=10, h=10)
    det1 = RawCharacterDetection(bbox=bbox1, centroid=bbox1.centroid, area=100.0)
    frame1_snapshot = tracker.update([det1])

    # Frame 2: Entity moves to (20, 20)
    bbox2 = BoundingBox(x=20, y=20, w=10, h=10)
    det2 = RawCharacterDetection(bbox=bbox2, centroid=bbox2.centroid, area=100.0)
    frame2_snapshot = tracker.update([det2])

    # Verify frame1_snapshot was NOT mutated by frame 2 processing
    assert frame1_snapshot[0].centroid == (15, 15)
    assert frame2_snapshot[0].centroid == (25, 25)

