"""Test OCR on exact relative bounding boxes for PA, PM, and HP."""

import os

import cv2
import numpy as np

from dta.services.ocr.easyocr_provider import EasyOCRProvider

USER_IMAGES_DIR = r"C:\Users\Andres\.gemini\antigravity-ide\brain\aaee6007-432e-4086-9822-f16f6f0a3661\.user_uploaded"

def test_exact_offsets():
    img_files = [
        ("Image 1", os.path.join(USER_IMAGES_DIR, "media_1788756198739.jpg")),
        ("Image 2", os.path.join(USER_IMAGES_DIR, "media_1788756252003.jpg")),
        ("Image 3", os.path.join(USER_IMAGES_DIR, "media_1788756427309.jpg")),
    ]

    ocr = EasyOCRProvider()

    for label, path in img_files:
        frame = cv2.imread(path)
        h, w = frame.shape[:2]

        rx1, ry1 = int(w * 0.75), int(h * 0.65)
        crop_br = frame[ry1:h, rx1:w]
        hsv_br = cv2.cvtColor(crop_br, cv2.COLOR_BGR2HSV)

        # Red Heart
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

        print(f"\n==================== {label} ====================")
        if not heart_box:
            print("Red Heart NOT found in HUD region!")
            continue

        hx, hy, hw, hh = heart_box
        ghx = rx1 + hx
        ghy = ry1 + hy

        # 1. HP: top half of heart
        hp_crop = frame[ghy + int(hh*0.08) : ghy + int(hh*0.58), ghx : ghx + hw]
        hsv_hp = cv2.cvtColor(hp_crop, cv2.COLOR_BGR2HSV)
        wmask_hp = cv2.inRange(hsv_hp, np.array([0, 0, 160]), np.array([180, 90, 255]))
        up_hp = cv2.resize(wmask_hp, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        hp_val = ocr.parse_first_int(ocr.extract_text(up_hp))

        # 2. PA: Blue Star at (ghx - 6, ghy + 41, 24, 22)
        pa_crop = frame[ghy + 40 : ghy + 64, ghx - 7 : ghx + 18]
        hsv_pa = cv2.cvtColor(pa_crop, cv2.COLOR_BGR2HSV)
        wmask_pa = cv2.inRange(hsv_pa, np.array([0, 0, 160]), np.array([180, 90, 255]))
        up_pa = cv2.resize(wmask_pa, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        pa_txt = ocr.extract_text(up_pa)
        pa_val = ocr.parse_first_int(pa_txt)

        # 3. PM: Green Diamond at (ghx + 28, ghy + 41, 23, 22)
        pm_crop = frame[ghy + 40 : ghy + 64, ghx + 27 : ghx + 52]
        hsv_pm = cv2.cvtColor(pm_crop, cv2.COLOR_BGR2HSV)
        wmask_pm = cv2.inRange(hsv_pm, np.array([0, 0, 160]), np.array([180, 90, 255]))
        up_pm = cv2.resize(wmask_pm, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        pm_txt = ocr.extract_text(up_pm)
        pm_val = ocr.parse_first_int(pm_txt)

        print(f"Extracted -> HP: {hp_val}, PA: {pa_val} (text='{pa_txt}'), PM: {pm_val} (text='{pm_txt}')")

if __name__ == "__main__":
    test_exact_offsets()
