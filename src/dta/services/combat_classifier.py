"""CombatClassifier service evaluating GameState and PerceptionState indicators for evidence-driven combat classification."""

from dta.core.logger import logger
from dta.models.combat_evidence import CombatEvidence
from dta.models.game_state import GameState


class CombatClassifier:
    """Evidence-driven combat classifier evaluating PA/PM resources, turn indicators, and combat bases."""

    COMBAT_THRESHOLD: int = 4

    def evaluate(self, game_state: GameState) -> CombatEvidence:
        """Evaluate game state indicators and return a structured CombatEvidence model with total score and rationale."""
        perc = game_state.perception_state
        res = perc.resources if perc else None

        pa_val = int(res.pa) if (res and res.pa is not None and res.pa > 0) else 0
        pm_val = int(res.pm) if (res and res.pm is not None and res.pm > 0) else 0

        turn_indicator = game_state.combat_state is not None
        hud_detected = False  # Reserved for future computer vision HUD detector

        combat_base_count = 0
        if perc and perc.raw_detections:
            raw_chars = perc.raw_detections.raw_characters
            combat_base_count = sum(1 for c in raw_chars if c.method == "combat_base")

        score = 0
        reasons: list[str] = []

        # 1. Turn Banner / Active Combat Model (4 points)
        if turn_indicator:
            score += 4
            reasons.append("turn_indicator")

        # 2. Combat HUD Detector (3 points)
        if hud_detected:
            score += 3
            reasons.append("combat_hud")

        # 3. Action Points (PA) (1 point)
        if pa_val > 0:
            score += 1
            reasons.append(f"pa({pa_val})")

        # 4. Movement Points (PM) (1 point)
        if pm_val > 0:
            score += 1
            reasons.append(f"pm({pm_val})")

        # 5. Combat Base Ring Count (>=2 bases -> 2 points, 1 base -> 0 points)
        if combat_base_count >= 2:
            score += 2
            reasons.append(f"multi_combat_bases({combat_base_count})")
        elif combat_base_count == 1:
            reasons.append("single_combat_base_ignored")

        reason_str = " + ".join(reasons) if reasons else "none (no combat evidence)"

        evidence = CombatEvidence(
            combat_hud_detected=hud_detected,
            combat_base_count=combat_base_count,
            pa_detected=pa_val,
            pm_detected=pm_val,
            turn_indicator_detected=turn_indicator,
            score=score,
            combat_threshold=self.COMBAT_THRESHOLD,
            reason=reason_str,
        )

        logger.debug(
            f"CombatClassifier evaluated frame {game_state.frame_id}: "
            f"score={score}/{self.COMBAT_THRESHOLD} (is_combat={evidence.is_combat}) -> reason='{reason_str}'"
        )
        return evidence
