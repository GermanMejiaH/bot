"""Test BGR neutral white text thresholding for PA and PM OCR."""

import os

import cv2
import numpy as np

from dta.services.ocr.easyocr_provider import EasyOCRProvider

USER_IMAGES_DIR = r"C:\Users\Andres\.gemini\antigravity-ide\brain\aaee6007-432e-4086-9822-f16f6f0a3661\.user_uploaded"

def test_bgr_white_text():
    ocr = EasyOCRProvider()

    for label, name in [("Image 1", "media_1788756198739.jpg"), ("Image 2", "media_1788756252003.jpg"), ("Image 3", "media_1788756427309.jpg")]:
        path = os.path.join(USER_IMAGES_DIR, name)
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

        if not heart_box:
            continue

        hx, hy, hw, hh = heart_box
        ghx = rx1 + hx
        ghy = ry1 + hy

        # PA crop
        pa_crop = frame[ghy + 39 : ghy + 64, ghx - 8 : ghx + 18]
        # PM crop
        pm_crop = frame[ghy + 39 : ghy + 64, ghx + 27 : ghx + 53]

        # Neutral white threshold: R >= 200, G >= 200, B >= 200
        pa_white = cv2.inRange(pa_crop, np.array([200, 200, 200]), np.array([255, 255, 255]))
        pm_white = cv2.inRange(pm_crop, np.array([200, 200, 200]), np.array([255, 255, 255]))

        # Upscale 3x and pad
        up_pa = cv2.resize(pa_white, (0, 0), fx=3.5, fy=3.5, interpolation=cv2.INTER_CUBIC)
        up_pa = cv2.copyMakeBorder(up_pa, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)

        up_pm = cv2.resize(pm_white, (0, 0), fx=3.5, fy=3.5, interpolation=cv2.INTER_CUBIC)
        up_pm = cv2.copyMakeBorder(up_pm, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)

        txt_pa = ocr.extract_text(up_pa)
        val_pa = ocr.parse_first_int(txt_pa)

        txt_pm = ocr.extract_text(up_pm)
        val_pm = ocr.parse_first_int(txt_pm)

        print(f"--- {label} ---")
        print(f"  PA (Blue Star): raw='{txt_pa}' -> {val_pa}")
        print(f"  PM (Green Diamond): raw='{txt_pm}' -> {val_pm}")

        cv2.imwrite(f"debug_captures/scratch/bgr_white_pa_{label.replace(' ', '_')}.png", up_pa)
        cv2.imwrite(f"debug_captures/scratch/bgr_white_pm_{label.replace(' ', '_')}.png", up_pm)

if __name__ == "__main__":
    test_bgr_white_text()
