"""Domain and detection models package initialization."""

from dta.models.combat import Ally, CombatEntity, CombatState, Enemy, Player, Position
from dta.models.detections import (
    BoundingBox,
    CharacterDetection,
    DetectionBatch,
    RawCharacterDetection,
    ResourceDetection,
)
from dta.models.frame import CapturedFrame, FrameMetadata
from dta.models.game_state import GameState
from dta.models.perception_state import PerceptionState

__all__ = [
    "Position",
    "CombatEntity",
    "Player",
    "Enemy",
    "Ally",
    "CombatState",
    "GameState",
    "PerceptionState",
    "CapturedFrame",
    "FrameMetadata",
    "BoundingBox",
    "RawCharacterDetection",
    "CharacterDetection",
    "ResourceDetection",
    "DetectionBatch",
]
