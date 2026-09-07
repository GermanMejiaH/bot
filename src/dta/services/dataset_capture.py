"""Dataset capture service automatically collecting and categorizing Dofus 3 game frames for YOLOv8 training."""

import json
import os
import threading
import time
from datetime import UTC, datetime
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel

from dta.config.settings import Settings, get_settings
from dta.core.logger import logger
from dta.events.event_bus import EventBus, get_event_bus
from dta.events.events import GameStateUpdated
from dta.models.game_state import GameState


class DatasetMetrics(BaseModel):
    """Real-time capture and deduplication metrics model."""

    total_frames_seen: int = 0
    total_frames_saved: int = 0
    duplicate_frames_skipped: int = 0
    combat_frames_saved: int = 0
    exploration_frames_saved: int = 0
    average_frame_difference: float = 0.0


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

        # Audit Mode Configuration
        audit_config = self.settings.audit
        self.audit_enabled = audit_config.enabled
        self.audit_max_frames = audit_config.max_frames
        self.audit_dir = audit_config.output_dir
        self.audit_original_dir = os.path.join(self.audit_dir, "original")
        self.audit_overlay_dir = os.path.join(self.audit_dir, "overlay")
        self.audit_stages_dir = os.path.join(self.audit_dir, "stages")
        self.audit_crops_dir = os.path.join(self.audit_dir, "crops")
        self.audit_excluded_dir = os.path.join(self.audit_dir, "excluded_regions")
        self.audit_json_path = os.path.join(self.audit_dir, "detections.json")

        self._audit_frame_count: int = 0
        self._audit_records: list[dict[str, Any]] = []

        self._ensure_directories()

        self._session_start_time: float = time.time()
        self._last_capture_time: float = 0.0
        self._last_captured_frame: np.ndarray | None = None

        self._total_frames_seen: int = 0
        self._captured_count: int = 0
        self._duplicate_frames_skipped: int = 0
        self._combat_frames_saved: int = 0
        self._exploration_frames_saved: int = 0
        self._diff_sum: float = 0.0
        self._diff_count: int = 0

        self._is_listening: bool = False
        self._lock = threading.RLock()

    def _ensure_directories(self) -> None:
        """Create dataset target directory structure if missing."""
        dirs = [self.dataset_dir, self.raw_dir, self.combat_dir, self.exploration_dir, self.validation_dir]
        if self.audit_enabled:
            dirs.extend([
                self.audit_dir,
                self.audit_original_dir,
                self.audit_overlay_dir,
                self.audit_stages_dir,
                self.audit_crops_dir,
                self.audit_excluded_dir,
            ])
        for d in dirs:
            if not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
                logger.debug(f"Created target directory: {d}")

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

    def get_metrics(self) -> DatasetMetrics:
        """Return a snapshot of current dataset capture metrics."""
        with self._lock:
            avg_diff = (self._diff_sum / self._diff_count) if self._diff_count > 0 else 0.0
            return DatasetMetrics(
                total_frames_seen=self._total_frames_seen,
                total_frames_saved=self._captured_count,
                duplicate_frames_skipped=self._duplicate_frames_skipped,
                combat_frames_saved=self._combat_frames_saved,
                exploration_frames_saved=self._exploration_frames_saved,
                average_frame_difference=round(avg_diff, 4),
            )

    def generate_session_report(self, report_filename: str = "report.json") -> str:
        """Generate and save session execution metrics report JSON to dataset directory."""
        with self._lock:
            metrics = self.get_metrics()
            now = time.time()
            start_iso = datetime.fromtimestamp(self._session_start_time, tz=UTC).isoformat()
            end_iso = datetime.fromtimestamp(now, tz=UTC).isoformat()
            duration = round(now - self._session_start_time, 2)

            report_data = {
                "session_start_time": start_iso,
                "session_end_time": end_iso,
                "duration_seconds": duration,
                "metrics": metrics.model_dump(),
            }

            report_path = os.path.join(self.dataset_dir, report_filename)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2)

            logger.info(f"Generated dataset session report: {report_path}")
            return report_path

    def start_listening(self) -> None:
        """Subscribe service to GameStateUpdated events on EventBus."""
        with self._lock:
            if not self._is_listening:
                self._session_start_time = time.time()
                self.event_bus.subscribe(GameStateUpdated, self.on_game_state_updated)
                self._is_listening = True
                logger.info("DatasetCaptureService subscribed to GameStateUpdated events.")

    def stop_listening(self) -> None:
        """Unsubscribe service from GameStateUpdated events on EventBus and generate session report."""
        with self._lock:
            if self._is_listening:
                self.event_bus.unsubscribe(GameStateUpdated, self.on_game_state_updated)
                self._is_listening = False
                logger.info("DatasetCaptureService unsubscribed from GameStateUpdated events.")
                self.generate_session_report()

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

    def get_combat_status_with_reason(self, game_state: GameState) -> tuple[bool, str, dict[str, Any]]:
        """Determine combat state classification returning boolean, exact reason string, and details dict."""
        from dta.services.combat_classifier import CombatClassifier

        classifier = CombatClassifier()
        evidence = classifier.evaluate(game_state)
        return evidence.is_combat, evidence.reason, evidence.to_dict()

    def is_combat_active(self, game_state: GameState) -> bool:
        """Determine whether the GameState snapshot represents a combat session."""
        is_combat, _, _ = self.get_combat_status_with_reason(game_state)
        return is_combat

    def process_audit_frame(self, game_state: GameState) -> None:
        """Process and save AUDIT mode trace data, crops, stage images, and structured JSON record."""
        if not self.audit_enabled or self._audit_frame_count >= self.audit_max_frames:
            return

        frame_img = game_state.frame_image
        if frame_img is None or frame_img.size == 0:
            return

        self._audit_frame_count += 1
        fid = f"{self._audit_frame_count:04d}"

        # 1. Original frame
        orig_path = os.path.join(self.audit_original_dir, f"frame_{fid}.png")
        cv2.imwrite(orig_path, frame_img)

        perc = game_state.perception_state
        batch = perc.raw_detections if perc else None
        res = perc.resources if perc else None

        from dta.services.combat_classifier import CombatClassifier
        classifier = CombatClassifier()
        evidence = classifier.evaluate(game_state)
        is_combat = evidence.is_combat
        combat_reason = evidence.reason
        combat_details = evidence.to_dict()

        all_candidates = batch.all_candidates if batch else []
        if not all_candidates and batch and batch.raw_characters:
            all_candidates = batch.raw_characters

        # 2. Candidate Crops (both accepted and rejected)
        candidate_records = []
        for det in all_candidates:
            cid = det.candidate_id if det.candidate_id is not None else 0
            status_tag = "accepted" if det.accepted else "rejected"
            crop_filename = f"entity_{cid:04d}_{status_tag}.png"
            crop_path = os.path.join(self.audit_crops_dir, crop_filename)

            bx, by, bw, bh = det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h
            h, w = frame_img.shape[:2]
            y1, y2 = max(0, by), min(h, by + bh)
            x1, x2 = max(0, bx), min(w, bx + bw)

            if (y2 - y1) > 0 and (x2 - x1) > 0:
                crop_img = frame_img[y1:y2, x1:x2]
                cv2.imwrite(crop_path, crop_img)
            else:
                crop_filename = "invalid_crop"

            det.crop_file = crop_filename

            candidate_records.append({
                "candidate_id": cid,
                "method": det.method,
                "bbox": [det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h],
                "centroid": list(det.centroid),
                "area": det.area,
                "confidence": det.confidence,
                "accepted": det.accepted,
                "reason": det.reason,
                "crop_file": crop_filename,
                "lifecycle": det.lifecycle,
            })

        # 3. Trace Overlay
        from dta.debug.detection_visualizer import DetectionVisualizer
        visualizer = DetectionVisualizer(settings=self.settings)
        overlay_img = visualizer.draw_trace_overlay(frame_img, all_candidates)
        overlay_path = os.path.join(self.audit_overlay_dir, f"frame_{fid}.png")
        cv2.imwrite(overlay_path, overlay_img)

        # 4. Stage Images
        debug_masks = batch.debug_masks if batch else {}
        contours_mask = debug_masks.get("closed_contours")
        hsv_mask = debug_masks.get("hsv_mask")
        ring_mask = debug_masks.get("ring_mask")

        contours_stage_path = os.path.join(self.audit_stages_dir, f"frame_{fid}_contours.png")
        hsv_stage_path = os.path.join(self.audit_stages_dir, f"frame_{fid}_hsv.png")
        combat_bases_stage_path = os.path.join(self.audit_stages_dir, f"frame_{fid}_combat_bases.png")
        final_stage_path = os.path.join(self.audit_stages_dir, f"frame_{fid}_final.png")

        cv2.imwrite(contours_stage_path, contours_mask if contours_mask is not None else np.zeros_like(frame_img[:, :, 0]))
        cv2.imwrite(hsv_stage_path, hsv_mask if hsv_mask is not None else np.zeros_like(frame_img[:, :, 0]))
        cv2.imwrite(combat_bases_stage_path, ring_mask if ring_mask is not None else np.zeros_like(frame_img[:, :, 0]))
        cv2.imwrite(final_stage_path, overlay_img)

        # 5. Excluded ROI mask
        from dta.vision.map_roi_extractor import MapROIExtractor
        roi_extractor = MapROIExtractor(settings=self.settings)
        roi_mask = roi_extractor.get_roi_mask(frame_img.shape)
        roi_mask_path = os.path.join(self.audit_excluded_dir, f"frame_{fid}_roi_mask.png")
        cv2.imwrite(roi_mask_path, roi_mask)

        # 6. Structured Trace Record
        stats = batch.statistics if batch else {}
        entities_by_method = {
            "contour": sum(1 for c in all_candidates if c.accepted and c.method == "contour"),
            "hsv": sum(1 for c in all_candidates if c.accepted and c.method == "hsv"),
            "combat_base": sum(1 for c in all_candidates if c.accepted and c.method == "combat_base"),
        }

        audit_record = {
            "frame_id": game_state.frame_id,
            "audit_sequence": self._audit_frame_count,
            "timestamp": datetime.fromtimestamp(game_state.timestamp, tz=UTC).isoformat(),
            "game_state": {
                "combat": is_combat,
                "combat_score": evidence.score,
                "combat_threshold": evidence.combat_threshold,
                "combat_reason": combat_reason,
                "combat_evidence": combat_details,
                "player_hp": res.hp if res else None,
                "player_pa": res.pa if res else None,
                "player_pm": res.pm if res else None,
                "entities_count": len([c for c in all_candidates if c.accepted]),
                "entities_by_method": entities_by_method,
            },
            "statistics": stats,
            "candidates": candidate_records,
        }

        self._audit_records.append(audit_record)

        with open(self.audit_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "total_frames_audited": len(self._audit_records),
                "audit_timestamp": datetime.now(UTC).isoformat(),
                "frames": self._audit_records,
            }, f, indent=2)

        logger.info(f"Saved AUDIT frame trace [{self._audit_frame_count}/{self.audit_max_frames}] -> {orig_path}")

    def capture_frame(self, game_state: GameState) -> dict[str, str] | None:
        """Process and conditionally save a GameState frame and metadata to the dataset directory structure."""
        with self._lock:
            # Process audit trace first if audit mode enabled
            if self.audit_enabled:
                self.process_audit_frame(game_state)

            if not self.enabled:
                return None

            self._total_frames_seen += 1

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
                self._diff_sum += diff
                self._diff_count += 1

                if diff < self.min_frame_difference:
                    self._duplicate_frames_skipped += 1
                    logger.debug(f"Skipped duplicate frame (diff={diff:.4f} < min={self.min_frame_difference}).")
                    return None
            else:
                self._diff_sum += 1.0
                self._diff_count += 1

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

            if is_combat:
                self._combat_frames_saved += 1
            else:
                self._exploration_frames_saved += 1

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
