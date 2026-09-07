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

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: list[RawCharacterDetection],
    ) -> np.ndarray:
        """Render colored bounding boxes, centroids, and method/confidence/area labels onto frame copy."""
        if frame is None or frame.size == 0:
            return np.zeros((100, 100, 3), dtype=np.uint8)

        overlay = frame.copy()

        for det in detections:
            bbox = det.bbox
            method = det.method
            color = self.get_method_color(method)

            # Draw bounding box rectangle
            cv2.rectangle(
                overlay,
                (bbox.x, bbox.y),
                (bbox.x + bbox.w, bbox.y + bbox.h),
                color,
                2,
            )

            # Draw centroid circle
            cv2.circle(overlay, det.centroid, 4, (255, 255, 255), -1)
            cv2.circle(overlay, det.centroid, 2, color, -1)

            # Text Label: method | conf | area
            label = f"{method} | conf:{det.confidence:.2f} | area:{det.area:.0f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1

            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Position text above bbox, or inside top of bbox if near top edge
            text_x = bbox.x
            text_y = max(text_h + 4, bbox.y - 6)

            # Background rectangle for crisp text readability
            cv2.rectangle(
                overlay,
                (text_x, text_y - text_h - 2),
                (text_x + text_w + 4, text_y + baseline),
                (0, 0, 0),
                -1,
            )

            cv2.putText(
                overlay,
                label,
                (text_x + 2, text_y),
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )

        return overlay

    def save_detection_overlay(
        self,
        frame: np.ndarray,
        detections: list[RawCharacterDetection],
        output_path: str | None = None,
    ) -> str:
        """Render detection overlay and save result image to disk."""
        overlay = self.draw_detections(frame, detections)

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
