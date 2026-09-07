"""DetectorEvaluator component for quantitative detection precision metrics and 2D spatial heatmap generation."""

import os
from statistics import median

import cv2
import numpy as np
from pydantic import BaseModel, Field

from dta.core.logger import logger
from dta.models.detections import RawCharacterDetection


class FrameMetrics(BaseModel):
    """Quantitative detection metrics for a single frame."""

    frame_id: str
    total_detections: int = 0
    average_bbox_width: float = 0.0
    average_bbox_height: float = 0.0
    average_bbox_area: float = 0.0
    aspect_ratios: list[float] = Field(default_factory=list)


class BatchEvaluationSummary(BaseModel):
    """Aggregated batch metrics across multiple evaluated frames."""

    total_frames_evaluated: int = 0
    min_detections: int = 0
    max_detections: int = 0
    mean_detections: float = 0.0
    median_detections: float = 0.0
    overall_avg_width: float = 0.0
    overall_avg_height: float = 0.0
    overall_avg_area: float = 0.0


class DetectorEvaluator:
    """Evaluates character detection precision and exports spatial false positive heatmaps."""

    def __init__(self, output_dir: str = "debug_captures") -> None:
        self.output_dir = output_dir
        self.frame_history: list[FrameMetrics] = []
        self._heatmap_accumulator: np.ndarray | None = None
        self._canvas_shape: tuple[int, int] | None = None

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    def evaluate_frame(
        self,
        frame_id: str,
        detections: list[RawCharacterDetection],
        frame_shape: tuple[int, int, int] | tuple[int, int] | None = None,
    ) -> FrameMetrics:
        """Evaluate detections for a single frame, update history, and accumulate spatial heatmap."""
        total = len(detections)
        if total == 0:
            metrics = FrameMetrics(frame_id=frame_id, total_detections=0)
            self.frame_history.append(metrics)
            return metrics

        widths = [float(d.bbox.w) for d in detections]
        heights = [float(d.bbox.h) for d in detections]
        areas = [float(d.area) for d in detections]
        ratios = [float(d.bbox.h / d.bbox.w) if d.bbox.w > 0 else 0.0 for d in detections]

        metrics = FrameMetrics(
            frame_id=frame_id,
            total_detections=total,
            average_bbox_width=float(np.mean(widths)),
            average_bbox_height=float(np.mean(heights)),
            average_bbox_area=float(np.mean(areas)),
            aspect_ratios=ratios,
        )
        self.frame_history.append(metrics)

        # Accumulate heatmap density if frame_shape provided
        if frame_shape is not None:
            h, w = frame_shape[:2]
            if self._heatmap_accumulator is None or self._canvas_shape != (h, w):
                self._heatmap_accumulator = np.zeros((h, w), dtype=np.float32)
                self._canvas_shape = (h, w)

            for det in detections:
                cx, cy = det.centroid
                bx, by, bw, bh = det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h

                # Draw bounding box footprint in accumulator
                x1, y1 = max(0, bx), max(0, by)
                x2, y2 = min(w, bx + bw), min(h, by + bh)
                if x2 > x1 and y2 > y1:
                    self._heatmap_accumulator[y1:y2, x1:x2] += 1.0

                # Extra weight for centroid
                if 0 <= cy < h and 0 <= cx < w:
                    self._heatmap_accumulator[cy, cx] += 5.0

        return metrics

    def compute_summary(self) -> BatchEvaluationSummary:
        """Compute aggregated summary metrics across all evaluated frames."""
        if not self.frame_history:
            return BatchEvaluationSummary()

        counts = [m.total_detections for m in self.frame_history]
        widths = [m.average_bbox_width for m in self.frame_history if m.total_detections > 0]
        heights = [m.average_bbox_height for m in self.frame_history if m.total_detections > 0]
        areas = [m.average_bbox_area for m in self.frame_history if m.total_detections > 0]


        summary = BatchEvaluationSummary(
            total_frames_evaluated=len(self.frame_history),
            min_detections=int(np.min(counts)),
            max_detections=int(np.max(counts)),
            mean_detections=float(np.mean(counts)),
            median_detections=float(median(counts)),
            overall_avg_width=float(np.mean(widths)) if widths else 0.0,
            overall_avg_height=float(np.mean(heights)) if heights else 0.0,
            overall_avg_area=float(np.mean(areas)) if areas else 0.0,
        )
        return summary

    def generate_heatmap_image(self, output_filename: str = "detection_heatmap.png") -> str | None:
        """Render and save 2D colorized detection heatmap image to disk."""
        if self._heatmap_accumulator is None or np.max(self._heatmap_accumulator) == 0:
            logger.warning("No heatmap data available to render.")
            return None

        # Normalize accumulator to 0-255 uint8 matrix
        acc_max = float(np.max(self._heatmap_accumulator))
        if acc_max > 0:
            acc_norm: np.ndarray = (self._heatmap_accumulator / acc_max * 255.0).astype(np.uint8)
        else:
            acc_norm = np.zeros_like(self._heatmap_accumulator, dtype=np.uint8)

        # Apply JET color map for spatial intensity representation
        heatmap_color: np.ndarray = cv2.applyColorMap(acc_norm, cv2.COLORMAP_JET)



        output_path = os.path.join(self.output_dir, output_filename)
        cv2.imwrite(output_path, heatmap_color)
        logger.info(f"Generated detection heatmap: {output_path}")
        return output_path

    def reset(self) -> None:
        """Reset evaluation history and heatmap accumulator."""
        self.frame_history.clear()
        self._heatmap_accumulator = None
        self._canvas_shape = None
