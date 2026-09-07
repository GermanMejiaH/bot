"""Test dilate-restricted white corner mask to eliminate map grid line noise."""

import os

import cv2
import numpy as np

from dta.detectors.character_detector import CharacterDetector
from dta.models.detections import BoundingBox, RawCharacterDetection
from dta.vision.map_roi_extractor import MapROIExtractor

USER_IMAGES_DIR = r"C:\Users\Andres\.gemini\antigravity-ide\brain\aaee6007-432e-4086-9822-f16f6f0a3661\.user_uploaded"

def run_test():
    path = os.path.join(USER_IMAGES_DIR, "media_1788759305759.jpg")
    raw_frame = cv2.imread(path)
    if raw_frame is None:
        return

    h, w = raw_frame.shape[:2]

    # 1. MapROIExtractor (masking top 16% height roofs/scenery)
    roi_extractor = MapROIExtractor()
    masked_frame = roi_extractor.apply_roi_mask(raw_frame)
    masked_frame[0:int(h * 0.16), :] = 0

    gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    hsv = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)

    # Red & Blue ring masks (S>=50, V>=70)
    r1 = cv2.inRange(hsv, np.array([0, 85, 85]), np.array([12, 255, 255]))
    r2 = cv2.inRange(hsv, np.array([165, 85, 85]), np.array([180, 255, 255]))
    red_ring = cv2.bitwise_or(r1, r2)
    blue_ring = cv2.inRange(hsv, np.array([85, 40, 70]), np.array([135, 255, 255]))

    # White corner brackets: only white pixels near blue ring
    white_corners = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 60, 255]))
    dilated_blue = cv2.dilate(blue_ring, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    valid_white_corners = cv2.bitwise_and(white_corners, dilated_blue)

    blue_with_corners = cv2.bitwise_or(blue_ring, valid_white_corners)

    # Morphological closing to connect base ring segments broken by pets
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
    closed_blue = cv2.morphologyEx(blue_with_corners, cv2.MORPH_CLOSE, kernel)

    ring_mask = cv2.bitwise_or(red_ring, closed_blue)

    cnts, _ = cv2.findContours(ring_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    base_boxes = []
    scores = []

    for c in cnts:
        bx, by, bw, bh = cv2.boundingRect(c)
        aspect = bh / float(bw) if bw > 0 else 0
        if 16 <= bw <= 85 and 8 <= bh <= 50 and 0.25 <= aspect <= 0.95:
            crop_mask = ring_mask[by:by+bh, bx:bx+bw]
            fill_ratio = np.count_nonzero(crop_mask) / float(bw * bh) if (bw * bh) > 0 else 1.0

            # Filter out solid empty placement phase tiles
            if fill_ratio >= 0.40:
                body_hsv = hsv[max(0, by - int(bh * 1.8)):by, bx:bx+bw]
                sat_std = float(np.std(body_hsv[:, :, 1])) if body_hsv.size > 0 else 0.0
                val_std = float(np.std(body_hsv[:, :, 2])) if body_hsv.size > 0 else 0.0
                if sat_std < 32.0 and val_std < 30.0:
                    continue

            body_y1 = max(0, by - int(bh * 1.8))
            body_y2 = by
            sprite_region = edges[body_y1:body_y2, bx:bx+bw]
            edge_density = np.count_nonzero(sprite_region) / float(bw * (body_y2 - body_y1)) if (body_y2 - body_y1) > 0 else 0

            if edge_density < 0.04:
                continue

            base_boxes.append([bx, by, bw, bh])
            scores.append(0.92)

    # NMS on base rings
    indices = cv2.dnn.NMSBoxes(
        bboxes=base_boxes,
        scores=scores,
        score_threshold=0.3,
        nms_threshold=0.30,
    )

    detections = []
    if len(indices) > 0:
        flat_indices = indices.flatten() if hasattr(indices, "flatten") else list(indices)
        for idx in flat_indices:
            bx, by, bw, bh = base_boxes[int(idx)]
            char_y = max(0, by - int(bh * 1.8))
            char_h = int(bh * 2.8)
            bbox = BoundingBox(x=bx, y=char_y, w=bw, h=char_h)
            detections.append(
                RawCharacterDetection(
                    bbox=bbox,
                    centroid=bbox.centroid,
                    area=float(bw * char_h),
                    confidence=0.92,
                    method="combat_base",
                )
            )

    detector = CharacterDetector()
    print(f"\nFinal Detected Character Entities: {len(detections)}")
    for d in detections:
        print(f"   -> Entity Box: ({d.bbox.x}, {d.bbox.y}, {d.bbox.w}, {d.bbox.h}), conf={d.confidence:.2f}")

    overlay = detector.generate_debug_overlay(masked_frame, detections)
    cv2.imwrite("debug_captures/scratch/perfect_normal_mode_overlay.png", overlay)

if __name__ == "__main__":
    run_test()
