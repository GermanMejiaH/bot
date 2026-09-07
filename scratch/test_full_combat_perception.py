"""Full combat perception test handling placement phase vs combat phase entity base detection."""

import os

import cv2
import numpy as np

from dta.detectors.character_detector import CharacterDetector
from dta.models.detections import BoundingBox, RawCharacterDetection

USER_IMAGES_DIR = r"C:\Users\Andres\.gemini\antigravity-ide\brain\aaee6007-432e-4086-9822-f16f6f0a3661\.user_uploaded"

def run_perception_test():
    img_files = [
        ("Image 1 (Placement Phase)", os.path.join(USER_IMAGES_DIR, "media_1788758171041.jpg")),
        ("Image 2 (Adjacent Combat Entities)", os.path.join(USER_IMAGES_DIR, "media_1788758300811.jpg")),
        ("Image 3 (4 Entities Close-up)", os.path.join(USER_IMAGES_DIR, "media_1788758352113.png")),
    ]

    detector = CharacterDetector()

    for label, path in img_files:
        if not os.path.exists(path):
            continue

        frame = cv2.imread(path)
        h, w = frame.shape[:2]
        print(f"\n==================== {label} ({w}x{h}) ====================")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red & Blue ring masks
        r1 = cv2.inRange(hsv, np.array([0, 90, 90]), np.array([12, 255, 255]))
        r2 = cv2.inRange(hsv, np.array([165, 90, 90]), np.array([180, 255, 255]))
        red_ring = cv2.bitwise_or(r1, r2)
        blue_ring = cv2.inRange(hsv, np.array([95, 90, 90]), np.array([135, 255, 255]))
        ring_mask = cv2.bitwise_or(red_ring, blue_ring)

        cnts, _ = cv2.findContours(ring_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        base_boxes = []
        scores = []

        for c in cnts:
            bx, by, bw, bh = cv2.boundingRect(c)
            aspect = bh / float(bw) if bw > 0 else 0
            if 16 <= bw <= 80 and 8 <= bh <= 50 and 0.25 <= aspect <= 0.95:
                crop_mask = ring_mask[by:by+bh, bx:bx+bw]
                fill_ratio = np.count_nonzero(crop_mask) / float(bw * bh) if (bw * bh) > 0 else 1.0

                body_y1 = max(0, by - int(bh * 1.8))
                body_y2 = by
                sprite_region = edges[body_y1:body_y2, bx:bx+bw]
                edge_density = np.count_nonzero(sprite_region) / float(bw * (body_y2 - body_y1)) if (body_y2 - body_y1) > 0 else 0

                body_hsv = hsv[body_y1:body_y2, bx:bx+bw]
                sat_std = float(np.std(body_hsv[:, :, 1])) if body_hsv.size > 0 else 0.0
                val_std = float(np.std(body_hsv[:, :, 2])) if body_hsv.size > 0 else 0.0

                # Empty placement phase tiles have flat ground texture above (low sat_std / val_std) and high fill ratio
                if fill_ratio >= 0.50 and (sat_std < 35.0 and val_std < 35.0):
                    continue

                # Require minimal edge density above base to confirm character sprite
                if edge_density < 0.05:
                    continue

                base_boxes.append([bx, by, bw, bh])
                scores.append(0.92)

        # Apply NMS directly on BASE RINGS (ground feet level)
        indices = cv2.dnn.NMSBoxes(
            bboxes=base_boxes,
            scores=scores,
            score_threshold=0.3,
            nms_threshold=0.35,
        )

        detections = []
        if len(indices) > 0:
            flat_indices = indices.flatten() if hasattr(indices, "flatten") else list(indices)
            for idx in flat_indices:
                bx, by, bw, bh = base_boxes[int(idx)]
                char_y = max(0, by - int(bh * 1.6))
                char_h = int(bh * 2.6)
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

        print(f"Detected Character Entities: {len(detections)}")
        for d in detections:
            print(f"   -> Entity Box: ({d.bbox.x}, {d.bbox.y}, {d.bbox.w}, {d.bbox.h}), conf={d.confidence:.2f}")

        overlay = detector.generate_debug_overlay(frame, detections)
        cv2.imwrite(f"debug_captures/scratch/final_{label.split()[1]}.png", overlay)

if __name__ == "__main__":
    run_perception_test()
