"""Event data models for DTA event-driven perception pipeline."""

import time
import uuid
from typing import Any

from pydantic import BaseModel, Field

from dta.models.frame import CapturedFrame
from dta.models.game_state import GameState


class BaseEvent(BaseModel):
    """Base event contract for all system notifications."""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=time.time)

    model_config = {"arbitrary_types_allowed": True}


class FrameCaptured(BaseEvent):
    """Event emitted when a raw frame is captured from screen."""

    frame_id: str
    frame: Any = Field(description="Raw frame array (e.g. numpy.ndarray)")
    width: int
    height: int
    captured_frame: CapturedFrame | None = None



class GameStateUpdated(BaseEvent):
    """Event emitted when a new GameState snapshot is assembled."""

    state: GameState


class CombatStarted(BaseEvent):
    """Event emitted when a new combat session starts.

    Note: Reserved for future CombatSessionManager integration.
    """

    combat_id: str


class CombatEnded(BaseEvent):
    """Event emitted when a combat session terminates.

    Note: Reserved for future CombatSessionManager integration.
    """

    combat_id: str
    duration: float = 0.0

