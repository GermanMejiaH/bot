"""Global GameState model encapsulating frame snapshot and combat state."""

import time
from typing import Any

from pydantic import BaseModel, Field

from dta.models.combat import CombatState
from dta.models.perception_state import PerceptionState


class GameState(BaseModel):
    """Global system snapshot linking raw frame metadata to perception state and combat state."""

    frame_id: str
    timestamp: float = Field(default_factory=time.time)
    perception_state: PerceptionState | None = None
    combat_state: CombatState | None = None
    raw_frame_path: str | None = None
    debug_frame_path: str | None = None
    frame_image: Any | None = Field(default=None, description="Raw numpy array reference for zero-copy visual overlay rendering")

    model_config = {"arbitrary_types_allowed": True}

