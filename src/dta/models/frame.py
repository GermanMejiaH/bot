"""Frame metadata and CapturedFrame container models."""

import time
import uuid
from typing import Any

from pydantic import BaseModel, Field


class FrameMetadata(BaseModel):
    """Metadata describing a captured frame."""

    frame_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=time.time)
    width: int
    height: int
    source_window: str = "Dofus"
    capture_fps: int = 15


class CapturedFrame(BaseModel):
    """Container holding a raw image buffer (numpy array) and its associated metadata.

    Performance Note: The image field holds a direct reference to the numpy array.
    Pipeline components must treat this array as read-only to avoid unnecessary buffer copies.
    """

    metadata: FrameMetadata
    image: Any = Field(description="Raw BGR/RGB numpy ndarray image buffer")

    model_config = {"arbitrary_types_allowed": True}

    @property
    def frame_id(self) -> str:
        return self.metadata.frame_id

    @property
    def timestamp(self) -> float:
        return self.metadata.timestamp

    @property
    def shape(self) -> tuple[int, ...]:
        if hasattr(self.image, "shape"):
            return self.image.shape  # type: ignore[no-any-return]
        return (self.metadata.height, self.metadata.width, 3)
