"""Test R,G,B >= 175 for Image 3 PM 4 digits."""

import os

import cv2
import numpy as np

from dta.services.ocr.easyocr_provider import EasyOCRProvider

USER_IMAGES_DIR = r"C:\Users\Andres\.gemini\antigravity-ide\brain\aaee6007-432e-4086-9822-f16f6f0a3661\.user_uploaded"

def test_pm_four():
    ocr = EasyOCRProvider()
    path = os.path.join(USER_IMAGES_DIR, "media_1788756427309.jpg")
    frame = cv2.imread(path)
    h, w = frame.shape[:2]

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
        ghx = rx1 + hx
        ghy = ry1 + hy

        pm_crop = frame[ghy + 38 : ghy + 64, ghx + 26 : ghx + 54]
        pm_white = cv2.inRange(pm_crop, np.array([170, 170, 170]), np.array([255, 255, 255]))

        up_pm = cv2.resize(pm_white, (0, 0), fx=3.5, fy=3.5, interpolation=cv2.INTER_CUBIC)
        up_pm = cv2.copyMakeBorder(up_pm, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=0)

        txt = ocr.extract_text(up_pm)
        val = ocr.parse_first_int(txt)
        print(f"Image 3 PM with RGB>=170: raw='{txt}' -> {val}")

if __name__ == "__main__":
    test_pm_four()
