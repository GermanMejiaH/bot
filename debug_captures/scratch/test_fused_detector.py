"""Test script for fused saliency + gradient CharacterDetector algorithm."""

import os
from typing import Any

import cv2
import numpy as np

from dta.vision.map_roi_extractor import MapROIExtractor

roi_extractor = MapROIExtractor()
raw_dir = "debug_captures/raw"


def detect_fused(frame: np.ndarray) -> list[dict[str, Any]]:
    masked = roi_extractor.apply_roi_mask(frame)
    if masked is None or masked.size == 0:
        return []

    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]

    # 1. Edge gradient
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    sobel_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(sobel_x, sobel_y)
    grad_u8 = np.clip(grad, 0, 255).astype(np.uint8)

    # 2. Saliency map: combine gradient and color saturation
    sat_norm = sat.astype(np.float32) / 255.0
    grad_norm = grad_u8.astype(np.float32) / 255.0

    saliency = (sat_norm * 0.5 + grad_norm * 0.5) * 255.0
    saliency_u8 = np.clip(saliency, 0, 255).astype(np.uint8)

    # 3. Otsu Thresholding
    _, thresh = cv2.threshold(saliency_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Vertical Morphology to join character head/torso/legs
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[dict[str, Any]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cnt_area = float(cv2.contourArea(cnt))
        bbox_area = float(w * h)
        aspect = h / float(w) if w > 0 else 0
        fill_ratio = cnt_area / bbox_area if bbox_area > 0 else 0

        if 15 <= w <= 90 and 25 <= h <= 160 and 400 <= bbox_area <= 12000 and 0.70 <= aspect <= 3.5 and fill_ratio >= 0.15:
            box_sal = saliency_u8[y : y + h, x : x + w]
            conf = float(np.mean(box_sal) / 255.0) if box_sal.size > 0 else 0.8
            conf = min(0.95, max(0.65, conf * 1.2))
            candidates.append(
                {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "bbox_area": bbox_area,
                    "cnt_area": cnt_area,
                    "aspect": aspect,
                    "conf": conf,
                }
            )

    if not candidates:
        return []
    bboxes = [[c["x"], c["y"], c["w"], c["h"]] for c in candidates]
    scores = [c["conf"] for c in candidates]
    indices = cv2.dnn.NMSBoxes(bboxes, scores, score_threshold=0.5, nms_threshold=0.3)

    merged: list[dict[str, Any]] = []
    if len(indices) > 0:
        flat = indices.flatten() if hasattr(indices, "flatten") else list(indices)
        for idx in flat:
            merged.append(candidates[int(idx)])
    return merged


def main() -> None:
    if not os.path.exists(raw_dir):
        print(f"Directory {raw_dir} does not exist.")
        return
    for fname in sorted(os.listdir(raw_dir)):
        if not fname.endswith(".png"):
            continue
        img = cv2.imread(os.path.join(raw_dir, fname))
        if img is None:
            continue
        dets = detect_fused(img)
        print(f"=== {fname} === (fused detections: {len(dets)})")
        for d in dets:
            print(
                f"  bbox=(x={d['x']}, y={d['y']}, w={d['w']}, h={d['h']}), area={d['bbox_area']:.1f}, conf={d['conf']:.2f}, aspect={d['aspect']:.2f}"
            )


if __name__ == "__main__":
    main()
