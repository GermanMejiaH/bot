"""Full validation test on user's 3 images applying placement phase filtering and anchored HUD OCR."""

import os

import cv2
import numpy as np

from dta.detectors.resource_detector import ResourceDetector
from dta.services.ocr.easyocr_provider import EasyOCRProvider

USER_IMAGES_DIR = r"C:\Users\Andres\.gemini\antigravity-ide\brain\aaee6007-432e-4086-9822-f16f6f0a3661\.user_uploaded"

def run_fixed_perception():
    img_files = [
        ("Image 1 (Placement Phase)", os.path.join(USER_IMAGES_DIR, "media_1788756198739.jpg")),
        ("Image 2 (Combat Turn)", os.path.join(USER_IMAGES_DIR, "media_1788756252003.jpg")),
        ("Image 3 (Double Detection Test)", os.path.join(USER_IMAGES_DIR, "media_1788756427309.jpg")),
    ]

    ocr = EasyOCRProvider()
    ResourceDetector(ocr_provider=ocr)

    for label, path in img_files:
        frame = cv2.imread(path)
        h, w = frame.shape[:2]
        print(f"\n==================== {label} ({w}x{h}) ====================")

        # 1. Anchored HUD OCR test
        rx1, ry1 = int(w * 0.75), int(h * 0.65)
        crop_br = frame[ry1:h, rx1:w]
        hsv_br = cv2.cvtColor(crop_br, cv2.COLOR_BGR2HSV)

        r1 = cv2.inRange(hsv_br, np.array([0, 90, 90]), np.array([12, 255, 255]))
        r2 = cv2.inRange(hsv_br, np.array([165, 90, 90]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(r1, r2)
        cnts, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        heart_box = None
        for c in cnts:
            bx, by, bw, bh = cv2.boundingRect(c)
            aspect = bh / float(bw) if bw > 0 else 0
            if 35 <= bw <= 100 and 35 <= bh <= 100 and 0.8 <= aspect <= 1.25:
                heart_box = (bx, by, bw, bh)
                break

        if heart_box:
            hx, hy, hw, hh = heart_box
            ghx, ghy = rx1 + hx, ry1 + hy

            # HP
            hp_crop = frame[ghy + int(hh*0.08) : ghy + int(hh*0.58), ghx : ghx + hw]
            hsv_hp = cv2.cvtColor(hp_crop, cv2.COLOR_BGR2HSV)
            wmask_hp = cv2.inRange(hsv_hp, np.array([0, 0, 160]), np.array([180, 90, 255]))
            up_hp = cv2.resize(wmask_hp, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
            hp_val = ocr.parse_first_int(ocr.extract_text(up_hp))

            # PA (Blue Star)
            pa_crop = frame[ghy + 40 : ghy + 64, ghx - 7 : ghx + 18]
            hsv_pa = cv2.cvtColor(pa_crop, cv2.COLOR_BGR2HSV)
            wmask_pa = cv2.inRange(hsv_pa, np.array([0, 0, 140]), np.array([180, 130, 255]))
            up_pa = cv2.resize(wmask_pa, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
            pa_val = ocr.parse_first_int(ocr.extract_text(up_pa))

            # PM (Green Diamond)
            pm_crop = frame[ghy + 40 : ghy + 64, ghx + 27 : ghx + 52]
            hsv_pm = cv2.cvtColor(pm_crop, cv2.COLOR_BGR2HSV)
            wmask_pm = cv2.inRange(hsv_pm, np.array([0, 0, 140]), np.array([180, 130, 255]))
            up_pm = cv2.resize(wmask_pm, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
            raw_pm_text = ocr.extract_text(up_pm)
            pm_val = ocr.parse_first_int(raw_pm_text)
            if pm_val is not None and pm_val > 20:
                pm_val = pm_val % 10  # Take single digit if prefixed by noise

            print(f"HUD Resources -> PA: {pa_val}, PM: {pm_val}, HP: {hp_val}")
        else:
            print("Red Heart NOT found in HUD region!")

        # 2. Combat Base & Character Detection Test
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Base masks
        r1 = cv2.inRange(hsv, np.array([0, 140, 140]), np.array([10, 255, 255]))
        r2 = cv2.inRange(hsv, np.array([170, 140, 140]), np.array([180, 255, 255]))
        red_ring = cv2.bitwise_or(r1, r2)
        blue_ring = cv2.inRange(hsv, np.array([100, 140, 140]), np.array([125, 255, 255]))
        ring_mask = cv2.bitwise_or(red_ring, blue_ring)

        cnts, _ = cv2.findContours(ring_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in cnts:
            bx, by, bw, bh = cv2.boundingRect(c)
            aspect = bh / float(bw) if bw > 0 else 0
            if 18 <= bw <= 75 and 10 <= bh <= 50 and 0.30 <= aspect <= 0.90:
                crop_mask = ring_mask[by:by+bh, bx:bx+bw]
                fill_ratio = np.count_nonzero(crop_mask) / float(bw * bh) if (bw * bh) > 0 else 1.0

                # Body region above tile
                body_y1 = max(0, by - int(bh * 1.6))
                body_y2 = by
                sprite_region = edges[body_y1:body_y2, bx:bx+bw]
                edge_density = np.count_nonzero(sprite_region) / float(bw * (body_y2 - body_y1)) if (body_y2 - body_y1) > 0 else 0

                body_hsv = hsv[body_y1:body_y2, bx:bx+bw]
                sat_std = float(np.std(body_hsv[:, :, 1])) if body_hsv.size > 0 else 0
                val_std = float(np.std(body_hsv[:, :, 2])) if body_hsv.size > 0 else 0

                # Reject solid empty placement tiles
                if fill_ratio >= 0.50 and (sat_std < 32.0 or val_std < 30.0):
                    continue

                if edge_density < 0.04:
                    continue

                char_y = max(0, by - int(bh * 1.6))
                char_h = int(bh * 2.6)
                candidates.append((bx, char_y, bw, char_h))

        print(f"Filtered Valid Character Entities: {len(candidates)}")
        for cb in candidates:
            print(f"   -> Entity Box ({cb[0]}, {cb[1]}, {cb[2]}, {cb[3]})")

if __name__ == "__main__":
    run_fixed_perception()
