"""Centroid-based EntityTracker for maintaining spatial identity persistence across frames."""

import math
import time
from typing import Any

from pydantic import BaseModel, Field

from dta.config.settings import Settings, get_settings
from dta.core.logger import logger
from dta.models.detections import BoundingBox, RawCharacterDetection


class TrackedEntity(BaseModel):
    """Tracked entity maintaining spatial identity across consecutive frames."""

    entity_id: str
    centroid: tuple[int, int]
    bbox: BoundingBox
    confidence: float = 1.0
    last_seen_timestamp: float = Field(default_factory=time.time)
    frames_seen: int = 1
    frames_missing: int = 0
    tracking_confidence: float = 1.0
    extra_data: dict[str, Any] = Field(default_factory=dict)

    def update_confidence(self) -> None:
        """Update tracking confidence metric based on seen vs missing frames ratio."""
        total = self.frames_seen + self.frames_missing
        if total > 0:
            self.tracking_confidence = float(self.frames_seen / total)
        else:
            self.tracking_confidence = 1.0



class EntityTracker:
    """Identity persistence tracker mapping RawCharacterDetections to TrackedEntities using Euclidean centroid distance."""

    def __init__(
        self,
        max_distance_threshold: float | None = None,
        max_disappeared_frames: int | None = None,
        settings: Settings | None = None,
    ) -> None:
        cfg = settings or get_settings()
        self.max_distance_threshold = (
            max_distance_threshold if max_distance_threshold is not None else cfg.tracking.max_distance_threshold
        )
        self.max_disappeared_frames = (
            max_disappeared_frames if max_disappeared_frames is not None else cfg.tracking.max_disappeared_frames
        )

        self._tracked_entities: dict[str, TrackedEntity] = {}
        self._next_id_counter: int = 1

    @property
    def tracked_entities(self) -> dict[str, TrackedEntity]:
        """Return dict copy of currently tracked entities keyed by entity_id."""
        return dict(self._tracked_entities)

    def _register(self, detection: RawCharacterDetection) -> TrackedEntity:
        """Register a new candidate detection as a TrackedEntity."""
        entity_id = f"entity_{self._next_id_counter:04d}"
        self._next_id_counter += 1

        entity = TrackedEntity(
            entity_id=entity_id,
            centroid=detection.centroid,
            bbox=detection.bbox,
            confidence=detection.confidence,
            last_seen_timestamp=time.time(),
            frames_seen=1,
            frames_missing=0,
        )
        self._tracked_entities[entity_id] = entity
        logger.debug(f"Registered new TrackedEntity: {entity_id} at {detection.centroid}")
        return entity

    def _deregister(self, entity_id: str) -> None:
        """Deregister an entity when its disappearance timeout is exceeded."""
        if entity_id in self._tracked_entities:
            del self._tracked_entities[entity_id]
            logger.debug(f"Deregistered stale TrackedEntity: {entity_id}")

    @staticmethod
    def _euclidean_distance(pt1: tuple[int, int], pt2: tuple[int, int]) -> float:
        """Compute 2D Euclidean distance between two centroid coordinates."""
        return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

    def update(self, detections: list[RawCharacterDetection]) -> list[TrackedEntity]:
        """Update tracked entities with new frame detections.

        Algorithm:
        1. If no existing entities, register all detections.
        2. If no detections, increment frames_missing for all entities and purge stale ones.
        3. Pairwise Euclidean centroid matching: greedily match pairs under max_distance_threshold.
        4. Update matched entities, register unmatched detections, increment frames_missing for unmatched entities.
        5. Purge entities exceeding max_disappeared_frames.
        """
        existing_entities = list(self._tracked_entities.values())

        # Case 1: No active tracked entities
        if not existing_entities:
            for det in detections:
                self._register(det)
            return [entity.model_copy(deep=True) for entity in self._tracked_entities.values()]

        # Case 2: No incoming detections
        if not detections:
            stale_ids: list[str] = []
            for entity_id, entity in list(self._tracked_entities.items()):
                entity.frames_missing += 1
                entity.update_confidence()
                if entity.frames_missing > self.max_disappeared_frames:
                    stale_ids.append(entity_id)

            for eid in stale_ids:
                self._deregister(eid)
            return [entity.model_copy(deep=True) for entity in self._tracked_entities.values()]

        # Case 3: Pairwise centroid distance matrix calculation
        # Building list of candidate pairs: (distance, existing_index, detection_index)
        candidate_pairs: list[tuple[float, int, int]] = []
        for i, entity in enumerate(existing_entities):
            for j, det in enumerate(detections):
                dist = self._euclidean_distance(entity.centroid, det.centroid)
                if dist <= self.max_distance_threshold:
                    candidate_pairs.append((dist, i, j))

        # Sort candidate pairs by ascending Euclidean distance
        candidate_pairs.sort(key=lambda x: x[0])

        used_existing_indices: set[int] = set()
        used_detection_indices: set[int] = set()

        # Greedy matching under distance threshold
        for _dist, e_idx, d_idx in candidate_pairs:
            if e_idx in used_existing_indices or d_idx in used_detection_indices:
                continue

            # Match found
            entity = existing_entities[e_idx]
            det = detections[d_idx]

            entity.centroid = det.centroid
            entity.bbox = det.bbox
            entity.confidence = det.confidence
            entity.last_seen_timestamp = time.time()
            entity.frames_seen += 1
            entity.frames_missing = 0
            entity.update_confidence()

            used_existing_indices.add(e_idx)
            used_detection_indices.add(d_idx)

        # Unmatched existing entities: increment missing counter
        for i, entity in enumerate(existing_entities):
            if i not in used_existing_indices:
                entity.frames_missing += 1
                entity.update_confidence()

        # Unmatched detections: register as new entities
        for j, det in enumerate(detections):
            if j not in used_detection_indices:
                self._register(det)

        # Purge stale entities
        stale_ids = [
            eid
            for eid, entity in list(self._tracked_entities.items())
            if entity.frames_missing > self.max_disappeared_frames
        ]
        for eid in stale_ids:
            self._deregister(eid)

        return [entity.model_copy(deep=True) for entity in self._tracked_entities.values()]

    def get_active_entities(self) -> list[TrackedEntity]:
        """Return list of currently tracked active entities."""
        return [entity.model_copy(deep=True) for entity in self._tracked_entities.values()]



    def reset(self) -> None:
        """Reset internal tracker state and counters."""
        self._tracked_entities.clear()
        self._next_id_counter = 1
