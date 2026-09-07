"""Test NMS on base rings before height expansion to keep adjacent entities separate."""

import os

import cv2
import numpy as np

from dta.detectors.character_detector import CharacterDetector
from dta.models.detections import BoundingBox, RawCharacterDetection

USER_IMAGES_DIR = r"C:\Users\Andres\.gemini\antigravity-ide\brain\aaee6007-432e-4086-9822-f16f6f0a3661\.user_uploaded"

def test_base_nms():
    img_files = [
        ("Image 1 (Placement Phase)", os.path.join(USER_IMAGES_DIR, "media_1788758171041.jpg")),
        ("Image 2 (Adjacent Combat Entities)", os.path.join(USER_IMAGES_DIR, "media_1788758300811.jpg")),
        ("Image 3 (4 Entities Close-up)", os.path.join(USER_IMAGES_DIR, "media_1788758352113.png")),
    ]

    for label, path in img_files:
        frame = cv2.imread(path)
        h, w = frame.shape[:2]
        print(f"\n==================== {label} ({w}x{h}) ====================")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Ring mask
        r1 = cv2.inRange(hsv, np.array([0, 130, 120]), np.array([12, 255, 255]))
        r2 = cv2.inRange(hsv, np.array([165, 130, 120]), np.array([180, 255, 255]))
        red_ring = cv2.bitwise_or(r1, r2)
        blue_ring = cv2.inRange(hsv, np.array([100, 130, 120]), np.array([130, 255, 255]))
        ring_mask = cv2.bitwise_or(red_ring, blue_ring)

        cnts, _ = cv2.findContours(ring_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        base_boxes = []
        scores = []
        valid_contours = []

        for c in cnts:
            bx, by, bw, bh = cv2.boundingRect(c)
            aspect = bh / float(bw) if bw > 0 else 0
            if 16 <= bw <= 80 and 8 <= bh <= 50 and 0.25 <= aspect <= 0.95:
                crop_mask = ring_mask[by:by+bh, bx:bx+bw]
                fill_ratio = np.count_nonzero(crop_mask) / float(bw * bh) if (bw * bh) > 0 else 1.0

                # Reject solid empty placement tiles (fill_ratio >= 0.40)
                if fill_ratio >= 0.40:
                    continue

                body_y1 = max(0, by - int(bh * 1.8))
                body_y2 = by
                sprite_region = edges[body_y1:body_y2, bx:bx+bw]
                edge_density = np.count_nonzero(sprite_region) / float(bw * (body_y2 - body_y1)) if (body_y2 - body_y1) > 0 else 0

                if edge_density < 0.03:
                    continue

                base_boxes.append([bx, by, bw, bh])
                scores.append(0.92)
                valid_contours.append((bx, by, bw, bh))

        # Perform NMS on actual BASE RINGS (ground level) with nms_threshold = 0.30
        indices = cv2.dnn.NMSBoxes(
            bboxes=base_boxes,
            scores=scores,
            score_threshold=0.3,
            nms_threshold=0.30,
        )

        final_detections = []
        if len(indices) > 0:
            flat_indices = indices.flatten() if hasattr(indices, "flatten") else list(indices)
            for idx in flat_indices:
                bx, by, bw, bh = base_boxes[int(idx)]
                char_y = max(0, by - int(bh * 1.6))
                char_h = int(bh * 2.6)
                bbox = BoundingBox(x=bx, y=char_y, w=bw, h=char_h)
                final_detections.append(
                    RawCharacterDetection(
                        bbox=bbox,
                        centroid=bbox.centroid,
                        area=float(bw * char_h),
                        confidence=0.92,
                        method="combat_base",
                    )
                )

        print(f"Detected Character Entities: {len(final_detections)}")
        for d in final_detections:
            print(f"   -> Entity Box: ({d.bbox.x}, {d.bbox.y}, {d.bbox.w}, {d.bbox.h}), conf={d.confidence:.2f}")

        detector = CharacterDetector()
        overlay = detector.generate_debug_overlay(frame, final_detections)
        cv2.imwrite(f"debug_captures/scratch/base_nms_{label.split()[1]}.png", overlay)

if __name__ == "__main__":
    test_base_nms()
