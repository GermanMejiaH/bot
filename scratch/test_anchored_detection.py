"""Test geometric HUD resource extraction relative to Red Heart."""

import os

import cv2
import numpy as np

from dta.services.ocr.easyocr_provider import EasyOCRProvider

USER_IMAGES_DIR = r"C:\Users\Andres\.gemini\antigravity-ide\brain\aaee6007-432e-4086-9822-f16f6f0a3661\.user_uploaded"

def test_anchored_detection():
    img_files = [
        ("Image 1", os.path.join(USER_IMAGES_DIR, "media_1788756198739.jpg")),
        ("Image 2", os.path.join(USER_IMAGES_DIR, "media_1788756252003.jpg")),
        ("Image 3", os.path.join(USER_IMAGES_DIR, "media_1788756427309.jpg")),
    ]

    ocr = EasyOCRProvider()

    for label, path in img_files:
        frame = cv2.imread(path)
        h, w = frame.shape[:2]

        # Search in bottom right: x from 75% to 100%, y from 65% to 100%
        search_x1 = int(w * 0.75)
        search_y1 = int(h * 0.65)
        hud_crop = frame[search_y1:h, search_x1:w]
        hsv_hud = cv2.cvtColor(hud_crop, cv2.COLOR_BGR2HSV)

        # 1. Locate Red Heart
        r1 = cv2.inRange(hsv_hud, np.array([0, 90, 90]), np.array([12, 255, 255]))
        r2 = cv2.inRange(hsv_hud, np.array([165, 90, 90]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(r1, r2)

        cnts, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        heart_box = None
        for c in cnts:
            bx, by, bw, bh = cv2.boundingRect(c)
            aspect = bh / float(bw) if bw > 0 else 0
            if 35 <= bw <= 100 and 35 <= bh <= 100 and 0.8 <= aspect <= 1.25:
                heart_box = (bx, by, bw, bh)
                break

        print(f"\n==================== {label} ====================")
        if not heart_box:
            print("Red Heart NOT found in HUD region!")
            continue

        hx, hy, hw, hh = heart_box
        # Global coords of heart
        ghx = search_x1 + hx
        ghy = search_y1 + hy

        # HP ROI: upper 55% of heart
        hp_roi = (ghx, ghy, ghx + hw, ghy + int(hh * 0.58))

        # PA ROI: Blue Star icon directly below-left of Heart
        pa_roi = (
            max(0, ghx - int(hw * 0.45)),
            ghy + int(hh * 0.75),
            ghx + int(hw * 0.55),
            ghy + int(hh * 1.50),
        )

        # PM ROI: Green Diamond icon directly below-right of Heart
        pm_roi = (
            ghx + int(hw * 0.40),
            ghy + int(hh * 0.75),
            min(w, ghx + int(hw * 1.40)),
            ghy + int(hh * 1.50),
        )

        # Preprocess white text for HP
        hp_crop = frame[hp_roi[1]:hp_roi[3], hp_roi[0]:hp_roi[2]]
        hsv_hp = cv2.cvtColor(hp_crop, cv2.COLOR_BGR2HSV)
        wmask_hp = cv2.inRange(hsv_hp, np.array([0, 0, 160]), np.array([180, 90, 255]))
        up_hp = cv2.resize(wmask_hp, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        hp_val = ocr.parse_first_int(ocr.extract_text(up_hp))

        # PA extraction
        pa_crop = frame[pa_roi[1]:pa_roi[3], pa_roi[0]:pa_roi[2]]
        hsv_pa = cv2.cvtColor(pa_crop, cv2.COLOR_BGR2HSV)
        wmask_pa = cv2.inRange(hsv_pa, np.array([0, 0, 160]), np.array([180, 90, 255]))
        up_pa = cv2.resize(wmask_pa, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        pa_val = ocr.parse_first_int(ocr.extract_text(up_pa))

        # PM extraction
        pm_crop = frame[pm_roi[1]:pm_roi[3], pm_roi[0]:pm_roi[2]]
        hsv_pm = cv2.cvtColor(pm_crop, cv2.COLOR_BGR2HSV)
        wmask_pm = cv2.inRange(hsv_pm, np.array([0, 0, 160]), np.array([180, 90, 255]))
        up_pm = cv2.resize(wmask_pm, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        pm_val = ocr.parse_first_int(ocr.extract_text(up_pm))

        print(f"Detected Values: PA={pa_val}, PM={pm_val}, HP={hp_val}")

if __name__ == "__main__":
    test_anchored_detection()
