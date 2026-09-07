"""StateBuilder assembling structured PerceptionState and wrapping it inside GameState."""

import time
from typing import Any

from dta.core.logger import logger
from dta.models.detections import DetectionBatch, ResourceDetection
from dta.models.game_state import GameState
from dta.models.perception_state import PerceptionState
from dta.tracking.entity_tracker import TrackedEntity


class StateBuilder:
    """Builder converting perception detections and tracked entities into structured PerceptionState and GameState containers."""

    def build_perception_state(
        self,
        frame_id: str,
        resources: ResourceDetection | None = None,
        tracked_entities: list[TrackedEntity] | None = None,
        raw_detections: DetectionBatch | None = None,
        timestamp: float | None = None,
    ) -> PerceptionState:
        """Construct a PerceptionState containing resources and spatial tracked entities without domain classification."""
        ts = timestamp if timestamp is not None else time.time()
        res = resources if resources is not None else ResourceDetection()
        entities = tracked_entities if tracked_entities is not None else []

        perception_state = PerceptionState(
            frame_id=frame_id,
            timestamp=ts,
            resources=res,
            tracked_entities=entities,
            raw_detections=raw_detections,
        )

        logger.debug(
            f"Built PerceptionState for {frame_id} with {len(entities)} tracked entities (PA={res.pa}, PM={res.pm}, HP={res.hp})."
        )
        return perception_state

    def build_game_state(
        self,
        frame_id: str,
        resources: ResourceDetection | None = None,
        tracked_entities: list[TrackedEntity] | None = None,
        raw_detections: DetectionBatch | None = None,
        raw_frame_path: str | None = None,
        debug_frame_path: str | None = None,
        timestamp: float | None = None,
        frame_image: Any | None = None,
    ) -> GameState:
        """Construct a global GameState snapshot embedding PerceptionState and file metadata."""
        ts = timestamp if timestamp is not None else time.time()

        perception_state = self.build_perception_state(
            frame_id=frame_id,
            resources=resources,
            tracked_entities=tracked_entities,
            raw_detections=raw_detections,
            timestamp=ts,
        )

        game_state = GameState(
            frame_id=frame_id,
            timestamp=ts,
            perception_state=perception_state,
            combat_state=None,  # Reserved for future classification phase
            raw_frame_path=raw_frame_path,
            debug_frame_path=debug_frame_path,
            frame_image=frame_image,
        )

        logger.debug(f"Built GameState snapshot for {frame_id}.")
        return game_state

