"""Dataset capture service automatically collecting and categorizing Dofus 3 game frames for YOLOv8 training."""

import json
import os
import threading
import time
from datetime import UTC, datetime
from typing import Any

import cv2
import numpy as np

from dta.config.settings import Settings, get_settings
from dta.core.logger import logger
from dta.events.event_bus import EventBus, get_event_bus
from dta.events.events import GameStateUpdated
from dta.models.game_state import GameState


class DatasetCaptureService:
    """Service capturing, deduplicating, classifying, and saving game frames for dataset generation."""

    def __init__(
        self,
        settings: Settings | None = None,
        event_bus: EventBus | None = None,
        dataset_dir: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.event_bus = event_bus or get_event_bus()

        ds_config = self.settings.dataset
        self.enabled = ds_config.enabled
        self.save_interval_seconds = ds_config.save_interval_seconds
        self.max_images_per_session = ds_config.max_images_per_session
        self.min_frame_difference = ds_config.min_frame_difference
        self.save_raw = ds_config.save_raw

        self.dataset_dir = dataset_dir or ds_config.dataset_dir
        self.raw_dir = os.path.join(self.dataset_dir, "raw")
        self.combat_dir = os.path.join(self.dataset_dir, "combat")
        self.exploration_dir = os.path.join(self.dataset_dir, "exploration")
        self.validation_dir = os.path.join(self.dataset_dir, "validation")

        self._ensure_directories()

        self._last_capture_time: float = 0.0
        self._last_captured_frame: np.ndarray | None = None
        self._captured_count: int = 0
        self._is_listening: bool = False
        self._lock = threading.RLock()

    def _ensure_directories(self) -> None:
        """Create dataset target directory structure if missing."""
        for d in [self.dataset_dir, self.raw_dir, self.combat_dir, self.exploration_dir, self.validation_dir]:
            if not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
                logger.debug(f"Created dataset directory: {d}")

    @property
    def captured_count(self) -> int:
        """Return total number of frames captured in current session."""
        with self._lock:
            return self._captured_count

    @property
    def is_listening(self) -> bool:
        """Return True if currently subscribed to GameStateUpdated events."""
        with self._lock:
            return self._is_listening

    def start_listening(self) -> None:
        """Subscribe service to GameStateUpdated events on EventBus."""
        with self._lock:
            if not self._is_listening:
                self.event_bus.subscribe(GameStateUpdated, self.on_game_state_updated)
                self._is_listening = True
                logger.info("DatasetCaptureService subscribed to GameStateUpdated events.")

    def stop_listening(self) -> None:
        """Unsubscribe service from GameStateUpdated events on EventBus."""
        with self._lock:
            if self._is_listening:
                self.event_bus.unsubscribe(GameStateUpdated, self.on_game_state_updated)
                self._is_listening = False
                logger.info("DatasetCaptureService unsubscribed from GameStateUpdated events.")

    def compute_phash(self, image: np.ndarray) -> str:
        """Compute a 64-bit difference perceptual hash (dHash) string for an image."""
        if image is None or image.size == 0:
            return "0" * 16

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)

        # Difference between adjacent pixels
        diff = resized[:, 1:] > resized[:, :-1]
        # Convert boolean array to hex string
        decimal_val = 0
        for bit in diff.flatten():
            decimal_val = (decimal_val << 1) | int(bit)

        return f"{decimal_val:016x}"

    def compute_frame_difference(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute normalized mean absolute pixel difference (MAE) between two frames in range [0.0, 1.0]."""
        if img1 is None or img2 is None or img1.size == 0 or img2.size == 0:
            return 1.0

        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2

        r1 = cv2.resize(g1, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
        r2 = cv2.resize(g2, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)

        diff = float(np.mean(np.abs(r1 - r2))) / 255.0
        return diff

    def is_combat_active(self, game_state: GameState) -> bool:
        """Determine whether the GameState snapshot represents a combat session."""
        if game_state.combat_state is not None:
            return True

        perc = game_state.perception_state
        if perc is None:
            return False

        # Resource check (PA/PM > 0 in combat HUD)
        res = perc.resources
        if (res.pa is not None and res.pa > 0) or (res.pm is not None and res.pm > 0):
            return True

        # Check raw detections for combat base detection method
        if perc.raw_detections:
            for char_det in perc.raw_detections.raw_characters:
                if char_det.method == "combat_base":
                    return True

        return False

    def capture_frame(self, game_state: GameState) -> dict[str, str] | None:
        """Process and conditionally save a GameState frame and metadata to the dataset directory structure."""
        with self._lock:
            if not self.enabled:
                return None

            if self._captured_count >= self.max_images_per_session:
                logger.debug("Dataset capture limit reached for current session.")
                return None

            now = time.time()
            if self._last_capture_time > 0 and (now - self._last_capture_time) < self.save_interval_seconds:
                return None

            frame_img = game_state.frame_image
            if frame_img is None or frame_img.size == 0:
                return None

            # Deduplication check
            diff = 1.0
            if self._last_captured_frame is not None:
                diff = self.compute_frame_difference(self._last_captured_frame, frame_img)
                if diff < self.min_frame_difference:
                    logger.debug(f"Skipped duplicate frame (diff={diff:.4f} < min={self.min_frame_difference}).")
                    return None

            # Classification & Save
            is_combat = self.is_combat_active(game_state)
            target_dir = self.combat_dir if is_combat else self.exploration_dir

            timestamp_str = datetime.fromtimestamp(game_state.timestamp, tz=UTC).strftime("%Y%m%d_%H%M%S_%f")[:23]
            base_name = f"frame_{timestamp_str}_{game_state.frame_id}"

            img_filename = f"{base_name}.png"
            json_filename = f"{base_name}.json"

            img_path = os.path.join(target_dir, img_filename)
            json_path = os.path.join(target_dir, json_filename)

            # Write Image
            cv2.imwrite(img_path, frame_img)

            raw_path: str | None = None
            if self.save_raw:
                raw_path = os.path.join(self.raw_dir, img_filename)
                cv2.imwrite(raw_path, frame_img)

            # Prepare Metadata JSON
            h, w = frame_img.shape[:2]
            perc = game_state.perception_state
            res = perc.resources if perc else None
            entities_count = len(perc.tracked_entities) if perc else 0

            phash = self.compute_phash(frame_img)
            iso_time = datetime.fromtimestamp(game_state.timestamp, tz=UTC).isoformat()

            metadata: dict[str, Any] = {
                "timestamp": iso_time,
                "frame_id": game_state.frame_id,
                "combat": is_combat,
                "resolution": [w, h],
                "pa": res.pa if res else None,
                "pm": res.pm if res else None,
                "hp": res.hp if res else None,
                "detected_entities": entities_count,
                "phash": phash,
                "frame_difference": round(diff, 4),
            }

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            self._last_capture_time = now
            self._last_captured_frame = frame_img.copy()
            self._captured_count += 1

            logger.info(
                f"Saved dataset sample [{self._captured_count}/{self.max_images_per_session}] -> "
                f"cat={'combat' if is_combat else 'exploration'}, file={img_filename}"
            )

            result = {"image": img_path, "json": json_path}
            if raw_path:
                result["raw_image"] = raw_path
            return result

    def on_game_state_updated(self, event: GameStateUpdated) -> None:
        """Event handler callback triggered when a GameStateUpdated event is published."""
        if event.state is not None:
            self.capture_frame(event.state)
