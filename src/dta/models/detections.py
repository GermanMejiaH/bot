"""Detection layer models separating vision outputs from domain logic."""

import time
from typing import Any

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box rectangular bounds in pixel coordinates."""

    x: int
    y: int
    w: int
    h: int

    @property
    def centroid(self) -> tuple[int, int]:
        """Compute bounding box centroid (cx, cy)."""
        return (self.x + self.w // 2, self.y + self.h // 2)

    @property
    def area(self) -> int:
        """Compute bounding box area in square pixels."""
        return self.w * self.h


class RawCharacterDetection(BaseModel):
    """Raw perception-only character detection bounding box without domain classification."""

    bbox: BoundingBox
    centroid: tuple[int, int]
    area: float
    confidence: float = 1.0
    method: str = "contour"


class CharacterDetection(BaseModel):
    """Classified character detection with domain label."""

    label: str = "character"
    bbox: BoundingBox
    confidence: float = 1.0
    centroid: tuple[int, int] = (0, 0)


class ResourceDetection(BaseModel):
    """Extracted numeric HUD resource values."""

    pa: int | None = None
    pm: int | None = None
    hp: int | None = None
    confidence: float = 1.0


class DetectionBatch(BaseModel):
    """Aggregated computer vision detections for a single frame."""

    frame_id: str
    timestamp: float = Field(default_factory=time.time)
    raw_characters: list[RawCharacterDetection] = Field(default_factory=list)
    characters: list[CharacterDetection] = Field(default_factory=list)
    resources: ResourceDetection = Field(default_factory=ResourceDetection)
    extra_detections: dict[str, Any] = Field(default_factory=dict)
    debug_masks: dict[str, Any] = Field(default_factory=dict)
    debug_overlay: Any | None = None

    model_config = {"arbitrary_types_allowed": True}
