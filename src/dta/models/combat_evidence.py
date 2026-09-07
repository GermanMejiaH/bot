"""Combat evidence model storing evidence scores and classification rationale."""

from dataclasses import dataclass
from typing import Any


@dataclass
class CombatEvidence:
    """Dataclass encapsulating individual combat indicators, total evidence score, and decision reason."""

    combat_hud_detected: bool = False
    combat_base_count: int = 0
    pa_detected: int = 0
    pm_detected: int = 0
    turn_indicator_detected: bool = False
    score: int = 0
    combat_threshold: int = 4
    reason: str = "none"

    @property
    def is_combat(self) -> bool:
        """Return True if total evidence score meets or exceeds combat threshold (>= 4)."""
        return self.score >= self.combat_threshold

    def to_dict(self) -> dict[str, Any]:
        """Export evidence properties as a plain dictionary for audit logging and JSON serialization."""
        return {
            "combat_hud_detected": self.combat_hud_detected,
            "combat_base_count": self.combat_base_count,
            "pa_detected": self.pa_detected,
            "pm_detected": self.pm_detected,
            "turn_indicator_detected": self.turn_indicator_detected,
            "score": self.score,
            "combat_threshold": self.combat_threshold,
            "reason": self.reason,
            "is_combat": self.is_combat,
        }
