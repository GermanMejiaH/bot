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

    def _filter_box_with_reason(self, w: int, h: int, bbox_area: float, cnt_area: float) -> tuple[bool, str]:
        """Validate candidate bounding box dimensions and fill ratio returning pass status and reason string."""
        aspect_ratio = h / float(w) if w > 0 else 0.0
        if not (self.min_width <= w <= self.max_width):
            return False, f"failed_width ({w} not in [{self.min_width}, {self.max_width}])"
        if not (self.min_height <= h <= self.max_height):
            return False, f"failed_height ({h} not in [{self.min_height}, {self.max_height}])"
        if not (self.min_area <= bbox_area <= self.max_area):
            return False, f"failed_area ({bbox_area:.0f} not in [{self.min_area}, {self.max_area}])"
        if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
            return False, f"failed_aspect_ratio ({aspect_ratio:.2f} not in [{self.min_aspect_ratio}, {self.max_aspect_ratio}])"
        if bbox_area > 0 and (cnt_area / bbox_area) < 0.05:
            return False, f"failed_contour_fill_ratio ({cnt_area/bbox_area:.3f} < 0.05)"
        return True, "passed_filtering"

    def _filter_box(self, w: int, h: int, bbox_area: float, cnt_area: float) -> bool:
        """Validate candidate bounding box dimensions and fill ratio."""
        passed, _ = self._filter_box_with_reason(w, h, bbox_area, cnt_area)
        return passed

    def _compute_candidate_diagnostics(
        self,
        frame: np.ndarray,
        hsv: np.ndarray | None,
        edges: np.ndarray | None,
        bbox: tuple[int, int, int, int],
        mask_crop: np.ndarray | None = None,
        cnt_area: float = 0.0,
        base_diag: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compute comprehensive diagnostic features across geometry, edges, color, texture, and mask."""
        x, y, w, h = bbox
        bbox_area = float(w * h)
        aspect_ratio = float(h / float(w)) if w > 0 else 0.0

        diag: dict[str, Any] = base_diag.copy() if base_diag else {}

        # Geometry
        diag["bbox_width"] = int(w)
        diag["bbox_height"] = int(h)
        diag["bbox_area"] = float(bbox_area)
        diag["aspect_ratio"] = round(aspect_ratio, 4)

        # Safe crop slicing
        img_h, img_w = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img_w, x + w), min(img_h, y + h)

        crop_bgr = frame[y1:y2, x1:x2]
        if crop_bgr.size == 0 or bbox_area == 0:
            default_keys = [
                "extent",
                "edge_density",
                "mean_h",
                "mean_s",
                "mean_v",
                "std_h",
                "std_s",
                "std_v",
                "dominant_hue",
                "gray_variance",
                "laplacian_variance",
                "gradient_magnitude_mean",
                "mask_ratio",
                "largest_connected_component_ratio",
                "connected_component_count",
            ]
            for k in default_keys:
                diag.setdefault(k, 0.0 if "count" not in k and "dominant" not in k else 0)
            return diag

        # Grayscale & HSV crops
        crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if len(crop_bgr.shape) == 3 else crop_bgr
        if hsv is not None and hsv.shape[:2] == frame.shape[:2]:
            crop_hsv = hsv[y1:y2, x1:x2]
        else:
            crop_hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV) if len(crop_bgr.shape) == 3 else crop_bgr

        # Edges
        if edges is not None and edges.shape[:2] == frame.shape[:2]:
            crop_edges = edges[y1:y2, x1:x2]
        else:
            blurred = cv2.GaussianBlur(crop_gray, (3, 3), 0)
            crop_edges = cv2.Canny(blurred, 30, 90)

        edge_pixels = float(np.count_nonzero(crop_edges))
        edge_density = edge_pixels / bbox_area
        diag["edge_density"] = round(edge_density, 4)

        # Colors (HSV)
        h_chan = crop_hsv[:, :, 0]
        s_chan = crop_hsv[:, :, 1]
        v_chan = crop_hsv[:, :, 2]

        diag["mean_h"] = round(float(np.mean(h_chan)), 2)
        diag["mean_s"] = round(float(np.mean(s_chan)), 2)
        diag["mean_v"] = round(float(np.mean(v_chan)), 2)

        diag["std_h"] = round(float(np.std(h_chan)), 2)
        diag["std_s"] = round(float(np.std(s_chan)), 2)
        diag["std_v"] = round(float(np.std(v_chan)), 2)

        hist = cv2.calcHist([h_chan], [0], None, [180], [0, 180])
        dominant_hue = int(np.argmax(hist))
        diag["dominant_hue"] = dominant_hue

        # Texture
        diag["gray_variance"] = round(float(np.var(crop_gray)), 2)
        diag["laplacian_variance"] = round(float(cv2.Laplacian(crop_gray, cv2.CV_64F).var()), 2)

        sobelx = cv2.Sobel(crop_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(crop_gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.hypot(sobelx, sobely)
        diag["gradient_magnitude_mean"] = round(float(np.mean(grad_mag)), 2)

        # Mask Features
        if mask_crop is not None and mask_crop.size > 0:
            m_crop = mask_crop[: y2 - y1, : x2 - x1]
            mask_pixels = float(np.count_nonzero(m_crop))
            mask_ratio = mask_pixels / bbox_area
            extent = cnt_area / bbox_area if cnt_area > 0 else mask_ratio

            binary_mask = (m_crop > 0).astype(np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask)

            if num_labels > 1:
                cc_areas = stats[1:, cv2.CC_STAT_AREA]
                cc_count = len(cc_areas)
                largest_cc_area = float(np.max(cc_areas))
                largest_cc_ratio = largest_cc_area / bbox_area
            else:
                cc_count = 0
                largest_cc_ratio = 0.0
        else:
            mask_ratio = cnt_area / bbox_area if cnt_area > 0 else 0.0
            extent = mask_ratio
            cc_count = 1 if cnt_area > 0 else 0
            largest_cc_ratio = mask_ratio

        diag["extent"] = round(extent, 4)
        diag["mask_ratio"] = round(mask_ratio, 4)
        diag["connected_component_count"] = int(cc_count)
        diag["largest_connected_component_ratio"] = round(largest_cc_ratio, 4)

        return diag


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

    def detect_characters_with_trace(
        self,
        frame: np.ndarray,
    ) -> dict[str, Any]:
        """Run perception pipeline with end-to-end trace telemetry, candidate lifecycle tracking, and stage masks."""
        if frame is None or frame.size == 0:
            return {
                "raw_characters": [],
                "rejected_characters": [],
                "all_candidates": [],
                "stage_masks": {},
                "statistics": {
                    "raw_contours": 0,
                    "raw_hsv": 0,
                    "raw_combat_bases": 0,
                    "after_filtering": 0,
                    "after_nms": 0,
                    "final_entities": 0,
                },
            }

        candidate_counter = 1
        all_candidates: list[RawCharacterDetection] = []

        # --- STAGE 1: CONTOUR DETECTION ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 30, 90)
        kernel_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
        closed_contours = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_vert)
        contours, _ = cv2.findContours(closed_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour_accepted: list[RawCharacterDetection] = []
        raw_contours_count = len(contours)

        for cnt in contours:
            cnt_area = float(cv2.contourArea(cnt))
            x, y, w, h = cv2.boundingRect(cnt)
            bbox_area = float(w * h)
            bbox = BoundingBox(x=x, y=y, w=w, h=h)

            cid = candidate_counter
            candidate_counter += 1

            passed, filter_reason = self._filter_box_with_reason(w, h, bbox_area, cnt_area)
            fill_ratio = cnt_area / bbox_area if bbox_area > 0 else 0.0
            edge_region = edges[y : y + h, x : x + w]
            edge_density = float(np.count_nonzero(edge_region) / bbox_area) if bbox_area > 0 else 0.0

            diagnostics = self._compute_candidate_diagnostics(
                frame=frame,
                hsv=None,
                edges=edges,
                bbox=(x, y, w, h),
                mask_crop=closed_contours[y : y + h, x : x + w],
                cnt_area=cnt_area,
                base_diag={
                    "contour_area": cnt_area,
                    "bounding_rect_area": bbox_area,
                    "fill_ratio": round(fill_ratio, 4),
                },
            )

            triggered_rules = ["canny_edge_extract"]
            if passed:
                triggered_rules.extend([
                    "min_max_width_pass",
                    "min_max_height_pass",
                    "min_max_area_pass",
                    "aspect_ratio_pass",
                    "contour_fill_ratio_pass",
                ])

            lifecycle = [
                {"stage": "contour_detection", "accepted": True, "reason": "contour_extracted"},
                {"stage": "filtering", "accepted": passed, "reason": filter_reason},
            ]

            det = RawCharacterDetection(
                bbox=bbox,
                centroid=bbox.centroid,
                area=bbox_area,
                confidence=0.85,
                method="contour",
                candidate_id=cid,
                accepted=passed,
                reason=filter_reason,
                lifecycle=lifecycle,
                diagnostics=diagnostics,
                triggered_rules=triggered_rules,
            )
            all_candidates.append(det)
            if passed:
                contour_accepted.append(det)

        # --- STAGE 2: HSV SEGMENTATION ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) if len(frame.shape) == 3 else frame
        hsv_mask = cv2.inRange(hsv, np.array([0, 15, 20]), np.array([180, 255, 255]))
        green_tile_mask = cv2.inRange(hsv, np.array([35, 60, 60]), np.array([85, 255, 255]))
        hsv_mask = cv2.bitwise_and(hsv_mask, cv2.bitwise_not(green_tile_mask))
        closed_hsv = cv2.morphologyEx(hsv_mask, cv2.MORPH_CLOSE, kernel_vert)
        hsv_contours, _ = cv2.findContours(closed_hsv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        hsv_accepted: list[RawCharacterDetection] = []
        raw_hsv_count = len(hsv_contours)

        for cnt in hsv_contours:
            cnt_area = float(cv2.contourArea(cnt))
            x, y, w, h = cv2.boundingRect(cnt)
            bbox_area = float(w * h)
            bbox = BoundingBox(x=x, y=y, w=w, h=h)

            cid = candidate_counter
            candidate_counter += 1

            passed, filter_reason = self._filter_box_with_reason(w, h, bbox_area, cnt_area)
            mask_crop = closed_hsv[y : y + h, x : x + w]
            fill_ratio = round(cnt_area / bbox_area, 4) if bbox_area > 0 else 0.0

            diagnostics = self._compute_candidate_diagnostics(
                frame=frame,
                hsv=hsv,
                edges=edges,
                bbox=(x, y, w, h),
                mask_crop=mask_crop,
                cnt_area=cnt_area,
                base_diag={
                    "contour_area": cnt_area,
                    "fill_ratio": fill_ratio,
                },
            )

            triggered_rules = ["hsv_segmentation_pass", "green_tile_exclusion_pass"]
            if passed:
                triggered_rules.extend([
                    "min_max_width_pass",
                    "min_max_height_pass",
                    "min_max_area_pass",
                    "aspect_ratio_pass",
                    "contour_fill_ratio_pass",
                ])

            lifecycle = [
                {"stage": "hsv_detection", "accepted": True, "reason": "hsv_segmented"},
                {"stage": "filtering", "accepted": passed, "reason": filter_reason},
            ]

            det = RawCharacterDetection(
                bbox=bbox,
                centroid=bbox.centroid,
                area=bbox_area,
                confidence=0.88,
                method="hsv",
                candidate_id=cid,
                accepted=passed,
                reason=filter_reason,
                lifecycle=lifecycle,
                diagnostics=diagnostics,
                triggered_rules=triggered_rules,
            )
            all_candidates.append(det)
            if passed:
                hsv_accepted.append(det)

        # --- STAGE 3: COMBAT BASE RINGS ---
        r1 = cv2.inRange(hsv, np.array([0, 85, 85]), np.array([12, 255, 255]))
        r2 = cv2.inRange(hsv, np.array([165, 85, 85]), np.array([180, 255, 255]))
        red_ring = cv2.bitwise_or(r1, r2)
        blue_ring = cv2.inRange(hsv, np.array([85, 40, 70]), np.array([135, 255, 255]))
        white_corners = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 60, 255]))
        dilated_blue = cv2.dilate(blue_ring, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        valid_white_corners = cv2.bitwise_and(white_corners, dilated_blue)
        blue_with_corners = cv2.bitwise_or(blue_ring, valid_white_corners)

        kernel_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
        closed_blue = cv2.morphologyEx(blue_with_corners, cv2.MORPH_CLOSE, kernel_horiz)
        closed_red = cv2.morphologyEx(red_ring, cv2.MORPH_CLOSE, kernel_horiz)
        ring_mask = cv2.bitwise_or(closed_red, closed_blue)

        base_contours, _ = cv2.findContours(ring_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_combat_bases_count = len(base_contours)

        base_candidates: list[dict[str, Any]] = []
        for cnt in base_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = h / float(w) if w > 0 else 0.0
            if 14 <= w <= 85 and 8 <= h <= 50 and 0.22 <= aspect <= 0.95:
                crop_mask = ring_mask[y : y + h, x : x + w]
                fill_ratio = np.count_nonzero(crop_mask) / float(w * h) if (w * h) > 0 else 1.0

                sat_std, val_std = 0.0, 0.0
                if fill_ratio >= 0.42:
                    body_hsv = hsv[max(0, y - int(h * 1.8)) : y, x : x + w]
                    sat_std = float(np.std(body_hsv[:, :, 1])) if body_hsv.size > 0 else 0.0
                    val_std = float(np.std(body_hsv[:, :, 2])) if body_hsv.size > 0 else 0.0
                    if sat_std < 32.0 and val_std < 30.0:
                        continue

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
                red_crop = closed_red[y : y + h, x : x + w]
                red_pixels = int(np.count_nonzero(red_crop))
                blue_pixels = int(np.count_nonzero(blue_crop))
                ring_area = float(w * h)
                red_ratio = round(red_pixels / ring_area, 4) if ring_area > 0 else 0.0
                blue_ratio = round(blue_pixels / ring_area, 4) if ring_area > 0 else 0.0
                mask_area = int(np.count_nonzero(crop_mask))
                has_blue = bool(blue_pixels > 10)

                base_candidates.append({
                    "bbox": [x, y, w, h],
                    "cx": x + w / 2.0,
                    "cy": y + h / 2.0,
                    "fill_ratio": fill_ratio,
                    "has_blue": has_blue,
                    "red_pixels": red_pixels,
                    "blue_pixels": blue_pixels,
                    "red_ratio": red_ratio,
                    "blue_ratio": blue_ratio,
                    "mask_area": mask_area,
                    "sat_std": sat_std,
                    "val_std": val_std,
                    "edge_density": edge_density,
                })

        filtered_base_candidates: list[dict[str, Any]] = []
        for c in base_candidates:
            if c["fill_ratio"] >= 0.42 and not c["has_blue"]:
                near_blue = any(
                    b["has_blue"] and np.hypot(c["cx"] - b["cx"], c["cy"] - b["cy"]) < 45.0
                    for b in base_candidates
                )
                if near_blue:
                    continue
            filtered_base_candidates.append(c)

        base_boxes: list[list[int]] = [c["bbox"] for c in filtered_base_candidates]
        scores: list[float] = [0.92] * len(base_boxes)

        indices = cv2.dnn.NMSBoxes(
            bboxes=base_boxes,
            scores=scores,
            score_threshold=0.3,
            nms_threshold=0.30,
        )

        combat_base_accepted: list[RawCharacterDetection] = []
        if len(indices) > 0:
            flat_indices = indices.flatten() if hasattr(indices, "flatten") else list(indices)
            for idx in flat_indices:
                x, y, w, h = base_boxes[int(idx)]
                c_info = filtered_base_candidates[int(idx)]
                bottom_y = y + h
                sprite_h = max(int(h * 4.8), int(w * 2.4))
                char_y = max(0, bottom_y - sprite_h)
                bbox_area = float(w * sprite_h)
                bbox = BoundingBox(x=x, y=char_y, w=w, h=sprite_h)

                cid = candidate_counter
                candidate_counter += 1

                mask_crop = ring_mask[y : y + h, x : x + w]

                diagnostics = self._compute_candidate_diagnostics(
                    frame=frame,
                    hsv=hsv,
                    edges=edges,
                    bbox=(x, char_y, w, sprite_h),
                    mask_crop=mask_crop,
                    cnt_area=float(c_info["mask_area"]),
                    base_diag={
                        "red_pixels": c_info["red_pixels"],
                        "blue_pixels": c_info["blue_pixels"],
                        "red_ratio": c_info["red_ratio"],
                        "blue_ratio": c_info["blue_ratio"],
                        "mask_area": c_info["mask_area"],
                        "fill_ratio": round(c_info["fill_ratio"], 4),
                        "sat_std": round(c_info["sat_std"], 2),
                        "val_std": round(c_info["val_std"], 2),
                        "has_blue": c_info["has_blue"],
                        "ring_w": w,
                        "ring_h": h,
                    },
                )

                triggered_rules = [
                    "hsv_color_ring_pass",
                    "dim_aspect_ratio_pass",
                    "fill_sat_val_pass",
                    "sprite_edge_density_pass",
                ]

                lifecycle = [
                    {"stage": "combat_base_detection", "accepted": True, "reason": "ring_base_extracted"},
                    {"stage": "filtering", "accepted": True, "reason": "passed_filtering"},
                ]

                det = RawCharacterDetection(
                    bbox=bbox,
                    centroid=bbox.centroid,
                    area=bbox_area,
                    confidence=0.92,
                    method="combat_base",
                    candidate_id=cid,
                    accepted=True,
                    reason="passed_filtering",
                    lifecycle=lifecycle,
                    diagnostics=diagnostics,
                    triggered_rules=triggered_rules,
                )
                all_candidates.append(det)
                combat_base_accepted.append(det)

        candidates_passed_filtering = contour_accepted + hsv_accepted + combat_base_accepted
        after_filtering_count = len(candidates_passed_filtering)

        # --- STAGE 4: NON-MAXIMUM SUPPRESSION (NMS) ---
        merged_accepted: list[RawCharacterDetection] = []
        if candidates_passed_filtering:
            nms_scores = np.array([d.confidence for d in candidates_passed_filtering])
            nms_indices = cv2.dnn.NMSBoxes(
                bboxes=[[d.bbox.x, d.bbox.y, d.bbox.w, d.bbox.h] for d in candidates_passed_filtering],
                scores=nms_scores.tolist(),
                score_threshold=0.3,
                nms_threshold=self.iou_threshold,
            )

            kept_set: set[int] = set()
            if len(nms_indices) > 0:
                flat_idx = nms_indices.flatten() if hasattr(nms_indices, "flatten") else list(nms_indices)
                kept_set = {int(i) for i in flat_idx}

            for idx, candidate in enumerate(candidates_passed_filtering):
                if idx in kept_set:
                    candidate.lifecycle.append({"stage": "nms", "accepted": True, "reason": "retained_by_nms"})
                    candidate.lifecycle.append({"stage": "final_selection", "accepted": True, "reason": "final_selected"})
                    candidate.accepted = True
                    candidate.reason = "final_selected"
                    if "nms_survived" not in candidate.triggered_rules:
                        candidate.triggered_rules.append("nms_survived")
                    merged_accepted.append(candidate)
                else:
                    candidate.lifecycle.append({"stage": "nms", "accepted": False, "reason": "suppressed_by_nms"})
                    candidate.lifecycle.append({"stage": "final_selection", "accepted": False, "reason": "suppressed_by_nms"})
                    candidate.accepted = False
                    candidate.reason = "suppressed_by_nms"

        after_nms_count = len(merged_accepted)
        rejected_candidates = [c for c in all_candidates if not c.accepted]

        stage_masks = {
            "closed_contours": closed_contours,
            "hsv_mask": closed_hsv,
            "ring_mask": ring_mask,
        }

        statistics = {
            "raw_contours": raw_contours_count,
            "raw_hsv": raw_hsv_count,
            "raw_combat_bases": raw_combat_bases_count,
            "after_filtering": after_filtering_count,
            "after_nms": after_nms_count,
            "final_entities": len(merged_accepted),
        }

        return {
            "raw_characters": merged_accepted,
            "rejected_characters": rejected_candidates,
            "all_candidates": all_candidates,
            "stage_masks": stage_masks,
            "statistics": statistics,
        }

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
