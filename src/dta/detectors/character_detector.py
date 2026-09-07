"""Dofus-specific CharacterDetector using vertical sprite morphology and aspect ratio filtering."""

from typing import Any

import cv2
import numpy as np

from dta.core.logger import logger
from dta.models.detections import BoundingBox, RawCharacterDetection


class CharacterDetector:
    """Perception-only detector identifying candidate Dofus character entity bounding boxes."""

    def __init__(
        self,
        min_width: int = 12,
        max_width: int = 95,
        min_height: int = 16,
        max_height: int = 160,
        min_area: float = 200.0,
        max_area: float = 12000.0,
        min_aspect_ratio: float = 0.50,
        max_aspect_ratio: float = 3.5,
        iou_threshold: float = 0.35,
    ) -> None:
        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.max_height = max_height
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.iou_threshold = iou_threshold

    def _filter_box(self, w: int, h: int, bbox_area: float, cnt_area: float) -> bool:
        """Validate candidate bounding box dimensions and fill ratio."""
        aspect_ratio = h / float(w) if w > 0 else 0.0
        if not (self.min_width <= w <= self.max_width):
            return False
        if not (self.min_height <= h <= self.max_height):
            return False
        if not (self.min_area <= bbox_area <= self.max_area):
            return False
        if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
            return False
        if bbox_area > 0 and (cnt_area / bbox_area) < 0.05:
            return False
        return True

    def detect_contours(self, frame: np.ndarray) -> list[RawCharacterDetection]:
        """Detect candidate character bounding boxes using image gradients and vertical morphology."""
        if frame is None or frame.size == 0:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 30, 90)

        # Morphological vertical closing to connect vertical character body features
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[RawCharacterDetection] = []

        for cnt in contours:
            cnt_area = float(cv2.contourArea(cnt))
            x, y, w, h = cv2.boundingRect(cnt)
            bbox_area = float(w * h)

            if self._filter_box(w, h, bbox_area, cnt_area):
                bbox = BoundingBox(x=x, y=y, w=w, h=h)
                detection = RawCharacterDetection(
                    bbox=bbox,
                    centroid=bbox.centroid,
                    area=bbox_area,
                    confidence=0.85,
                    method="contour",
                )
                detections.append(detection)

        logger.debug(f"Contour character detection found {len(detections)} candidates.")
        return detections

    def detect_blobs(self, frame: np.ndarray) -> list[RawCharacterDetection]:
        """Legacy blob detector wrapper maintaining API compatibility with strictly filtered candidates."""
        return []

    def detect_hsv_regions(
        self,
        frame: np.ndarray,
        lower_hsv: tuple[int, int, int] = (0, 15, 20),
        upper_hsv: tuple[int, int, int] = (180, 255, 255),
    ) -> list[RawCharacterDetection]:
        """Detect candidate character regions using HSV color segmentation with aspect ratio constraints."""
        if frame is None or frame.size == 0:
            return []

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) if len(frame.shape) == 3 else frame
        mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))

        # Explicitly mask out solid green PM movement range tiles on combat grid
        green_tile_mask = cv2.inRange(hsv, np.array([35, 60, 60]), np.array([85, 255, 255]))
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(green_tile_mask))

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[RawCharacterDetection] = []

        for cnt in contours:
            cnt_area = float(cv2.contourArea(cnt))
            x, y, w, h = cv2.boundingRect(cnt)
            bbox_area = float(w * h)

            if self._filter_box(w, h, bbox_area, cnt_area):
                bbox = BoundingBox(x=x, y=y, w=w, h=h)
                detections.append(
                    RawCharacterDetection(
                        bbox=bbox,
                        centroid=bbox.centroid,
                        area=bbox_area,
                        confidence=0.88,
                        method="hsv",
                    )
                )

        logger.debug(f"HSV segmentation found {len(detections)} candidates.")
        return detections

    def detect_combat_bases(self, frame: np.ndarray) -> list[RawCharacterDetection]:
        """Detect red (enemy) and blue (ally) isometric selection rings under entity feet during combat."""
        if frame is None or frame.size == 0:
            return []

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) if len(frame.shape) == 3 else frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        edges = cv2.Canny(gray, 40, 120)

        # Red & Blue ring masks (S>=50, V>=70)
        r1 = cv2.inRange(hsv, np.array([0, 85, 85]), np.array([12, 255, 255]))
        r2 = cv2.inRange(hsv, np.array([165, 85, 85]), np.array([180, 255, 255]))
        red_ring = cv2.bitwise_or(r1, r2)
        blue_ring = cv2.inRange(hsv, np.array([85, 40, 70]), np.array([135, 255, 255]))

        # Include white corner brackets under feet in normal mode (S <= 60, V >= 180)
        white_corners = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 60, 255]))
        dilated_blue = cv2.dilate(blue_ring, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        valid_white_corners = cv2.bitwise_and(white_corners, dilated_blue)

        blue_with_corners = cv2.bitwise_or(blue_ring, valid_white_corners)

        # Morphological closing to connect base ring segments broken by pets or pets standing beside feet
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
        closed_blue = cv2.morphologyEx(blue_with_corners, cv2.MORPH_CLOSE, kernel)
        closed_red = cv2.morphologyEx(red_ring, cv2.MORPH_CLOSE, kernel)

        ring_mask = cv2.bitwise_or(closed_red, closed_blue)

        contours, _ = cv2.findContours(ring_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: list[dict[str, Any]] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = h / float(w) if w > 0 else 0.0
            if 14 <= w <= 85 and 8 <= h <= 50 and 0.22 <= aspect <= 0.95:
                crop_mask = ring_mask[y : y + h, x : x + w]
                fill_ratio = np.count_nonzero(crop_mask) / float(w * h) if (w * h) > 0 else 1.0

                # Hollow base rings under characters have thin borders (fill_ratio < 0.42)
                # Solid grid selection tiles have high fill ratio (fill_ratio >= 0.42)
                if fill_ratio >= 0.42:
                    body_hsv = hsv[max(0, y - int(h * 1.8)) : y, x : x + w]
                    sat_std = float(np.std(body_hsv[:, :, 1])) if body_hsv.size > 0 else 0.0
                    val_std = float(np.std(body_hsv[:, :, 2])) if body_hsv.size > 0 else 0.0
                    if sat_std < 32.0 and val_std < 30.0:
                        continue

                # Check edge density directly above base for character body presence
                body_y1 = max(0, y - int(h * 1.8))
                body_y2 = y
                sprite_region = edges[body_y1:body_y2, x : x + w]
                edge_density = (
                    np.count_nonzero(sprite_region) / float(w * (body_y2 - body_y1))
                    if (body_y2 - body_y1) > 0
                    else 0.0
                )

                if edge_density < 0.035:
                    continue

                blue_crop = closed_blue[y : y + h, x : x + w]
                has_blue = bool(np.count_nonzero(blue_crop) > 10)

                candidates.append({
                    "bbox": [x, y, w, h],
                    "cx": x + w / 2.0,
                    "cy": y + h / 2.0,
                    "fill_ratio": fill_ratio,
                    "has_blue": has_blue,
                })

        # Suppress solid red pet/mount blobs sitting directly beside a player blue base ring
        filtered_candidates: list[dict[str, Any]] = []
        for c in candidates:
            if c["fill_ratio"] >= 0.42 and not c["has_blue"]:
                near_blue = any(
                    b["has_blue"] and np.hypot(c["cx"] - b["cx"], c["cy"] - b["cy"]) < 45.0
                    for b in candidates
                )
                if near_blue:
                    continue
            filtered_candidates.append(c)

        base_boxes: list[list[int]] = [c["bbox"] for c in filtered_candidates]
        scores: list[float] = [0.92] * len(base_boxes)

        # Perform NMS on ground-level base rings before height expansion
        # This prevents adjacent characters from suppressing each other
        indices = cv2.dnn.NMSBoxes(
            bboxes=base_boxes,
            scores=scores,
            score_threshold=0.3,
            nms_threshold=0.30,
        )

        detections: list[RawCharacterDetection] = []
        if len(indices) > 0:
            flat_indices = indices.flatten() if hasattr(indices, "flatten") else list(indices)
            for idx in flat_indices:
                x, y, w, h = base_boxes[int(idx)]
                bottom_y = y + h
                sprite_h = max(int(h * 4.8), int(w * 2.4))
                char_y = max(0, bottom_y - sprite_h)
                bbox_area = float(w * sprite_h)
                bbox = BoundingBox(x=x, y=char_y, w=w, h=sprite_h)
                detections.append(
                    RawCharacterDetection(
                        bbox=bbox,
                        centroid=bbox.centroid,
                        area=bbox_area,
                        confidence=0.92,
                        method="combat_base",
                    )
                )

        logger.debug(f"Combat base detection found {len(detections)} candidates.")
        return detections

    def merge_candidates(
        self,
        detections: list[RawCharacterDetection],
        iou_threshold: float | None = None,
    ) -> list[RawCharacterDetection]:
        """Merge overlapping candidate bounding boxes using Non-Maximum Suppression (NMS)."""
        if not detections:
            return []

        thresh = iou_threshold if iou_threshold is not None else self.iou_threshold
        scores = np.array([d.confidence for d in detections])

        indices = cv2.dnn.NMSBoxes(
            bboxes=[[d.bbox.x, d.bbox.y, d.bbox.w, d.bbox.h] for d in detections],
            scores=scores.tolist(),
            score_threshold=0.3,
            nms_threshold=thresh,
        )

        merged: list[RawCharacterDetection] = []
        if len(indices) > 0:
            flat_indices = indices.flatten() if hasattr(indices, "flatten") else list(indices)
            for idx in flat_indices:
                merged.append(detections[int(idx)])

        return merged

    def detect_characters(self, frame: np.ndarray) -> list[RawCharacterDetection]:
        """Run Dofus-specific character detection pipeline combining contour, HSV, and combat base features."""
        candidates = self.detect_contours(frame) + self.detect_hsv_regions(frame) + self.detect_combat_bases(frame)
        return self.merge_candidates(candidates)

    def generate_debug_overlay(
        self,
        frame: np.ndarray,
        detections: list[RawCharacterDetection],
    ) -> np.ndarray:
        """Render candidate bounding boxes and centroids onto a copy of the frame for debug visual overlay."""
        if frame is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)

        overlay = frame.copy()
        for det in detections:
            bbox = det.bbox
            cv2.rectangle(
                overlay,
                (bbox.x, bbox.y),
                (bbox.x + bbox.w, bbox.y + bbox.h),
                (0, 255, 0),
                2,
            )
            cv2.circle(overlay, det.centroid, 4, (0, 0, 255), -1)
            cv2.putText(
                overlay,
                f"{det.method}:{det.confidence:.2f}",
                (bbox.x, max(15, bbox.y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )
        return overlay
