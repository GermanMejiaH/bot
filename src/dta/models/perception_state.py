"""PerceptionState container storing aggregated vision outputs and tracked entities."""

import time

from pydantic import BaseModel, Field

from dta.models.detections import DetectionBatch, ResourceDetection
from dta.tracking.entity_tracker import TrackedEntity


class PerceptionState(BaseModel):
    """Perception-level state snapshot storing resources and tracked entities without combat semantics."""

    frame_id: str
    timestamp: float = Field(default_factory=time.time)
    resources: ResourceDetection = Field(default_factory=ResourceDetection)
    tracked_entities: list[TrackedEntity] = Field(default_factory=list)
    raw_detections: DetectionBatch | None = None

    model_config = {"arbitrary_types_allowed": True}
