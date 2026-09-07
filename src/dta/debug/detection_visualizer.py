"""Visual Detection Debugger for CharacterDetector bounding boxes and method-based color overlays."""

import os
import time
from datetime import UTC, datetime

import cv2
import numpy as np

from dta.config.settings import Settings, get_settings
from dta.core.logger import logger
from dta.models.detections import RawCharacterDetection

# BGR Color Mapping
METHOD_COLORS: dict[str, tuple[int, int, int]] = {
    "contour": (0, 255, 255),      # Yellow
    "hsv": (0, 255, 0),            # Green
    "combat_base": (0, 0, 255),    # Red
}
DEFAULT_COLOR: tuple[int, int, int] = (255, 255, 0)  # Cyan fallback


class DetectionVisualizer:
    """Visual debugger rendering method-colored bounding boxes and metadata labels onto game frames."""

    def __init__(
        self,
        settings: Settings | None = None,
        output_dir: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.output_dir = output_dir or self.settings.debug.output_dir

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    def get_method_color(self, method: str) -> tuple[int, int, int]:
        """Return BGR color tuple corresponding to detection method."""
        return METHOD_COLORS.get(method.lower(), DEFAULT_COLOR)

    def draw_trace_overlay(
        self,
        frame: np.ndarray,
        candidates: list[RawCharacterDetection],
    ) -> np.ndarray:
        """Render candidate bounding boxes with candidate ID, method, confidence, status, and rejection reason."""
        if frame is None or frame.size == 0:
            return np.zeros((100, 100, 3), dtype=np.uint8)

        overlay = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.38
        thickness = 1

        for det in candidates:
            bbox = det.bbox
            cid_str = f"ID:{det.candidate_id}" if det.candidate_id is not None else "ID:?"
            status_str = "ACCEPTED" if det.accepted else f"REJECTED: {det.reason}"

            if det.accepted:
                color = self.get_method_color(det.method)
                line_thickness = 2
            else:
                color = (128, 128, 128)  # Gray for rejected
                line_thickness = 1

            # Bounding box
            cv2.rectangle(
                overlay,
                (bbox.x, bbox.y),
                (bbox.x + bbox.w, bbox.y + bbox.h),
                color,
                line_thickness,
            )

            # Centroid
            cv2.circle(overlay, det.centroid, 3, (255, 255, 255), -1)
            cv2.circle(overlay, det.centroid, 1, color, -1)

            # Multi-line label lines
            l1 = f"{cid_str} | {det.method}"
            l2 = f"conf:{det.confidence:.2f} | {status_str}"
            lines = [l1, l2]

            # Determine background box dimensions
            max_w = 0
            total_h = 0
            line_heights: list[int] = []

            for line in lines:
                (tw, th), baseline = cv2.getTextSize(line, font, font_scale, thickness)
                max_w = max(max_w, tw)
                line_heights.append(th + baseline + 2)
                total_h += th + baseline + 2

            text_x = bbox.x
            text_y = max(total_h + 4, bbox.y - 4)

            # Background rectangle for crisp text readability
            cv2.rectangle(
                overlay,
                (text_x, text_y - total_h - 2),
                (text_x + max_w + 6, text_y + 2),
                (0, 0, 0),
                -1,
            )

            curr_y = text_y - total_h + line_heights[0] - 2
            for idx, line in enumerate(lines):
                line_color = color if idx == 0 else ((0, 255, 0) if det.accepted else (0, 0, 255))
                cv2.putText(
                    overlay,
                    line,
                    (text_x + 3, curr_y),
                    font,
                    font_scale,
                    line_color,
                    thickness,
                    cv2.LINE_AA,
                )
                if idx < len(lines) - 1:
                    curr_y += line_heights[idx + 1]

        return overlay

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: list[RawCharacterDetection],
    ) -> np.ndarray:
        """Render colored bounding boxes, centroids, and method/confidence/area labels onto frame copy."""
        return self.draw_trace_overlay(frame, detections)

    def save_detection_overlay(
        self,
        frame: np.ndarray,
        detections: list[RawCharacterDetection],
        output_path: str | None = None,
    ) -> str:
        """Render detection overlay and save result image to disk."""
        overlay = self.draw_trace_overlay(frame, detections)

        if output_path is None:
            ts_str = datetime.fromtimestamp(time.time(), tz=UTC).strftime("%Y%m%d_%H%M%S_%f")[:23]
            filename = f"detection_{ts_str}.png"
            output_path = os.path.join(self.output_dir, filename)

        target_dir = os.path.dirname(output_path)
        if target_dir and not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)

        cv2.imwrite(output_path, overlay)
        logger.debug(f"Saved visual detection overlay to: {output_path}")
        return output_path


def save_detection_overlay(
    frame: np.ndarray,
    detections: list[RawCharacterDetection],
    output_path: str | None = None,
) -> str:
    """Standalone convenience function drawing detection overlay and saving to disk."""
    visualizer = DetectionVisualizer()
    return visualizer.save_detection_overlay(frame, detections, output_path=output_path)
