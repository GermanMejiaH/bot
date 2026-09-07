"""Inspect HSV ranges and contours of HUD icons in user's 3 images."""

import os

import cv2
import numpy as np

from dta.services.ocr.easyocr_provider import EasyOCRProvider

USER_IMAGES_DIR = r"C:\Users\Andres\.gemini\antigravity-ide\brain\aaee6007-432e-4086-9822-f16f6f0a3661\.user_uploaded"

def inspect_user_hud_details():
    img_files = [
        ("Image 1", os.path.join(USER_IMAGES_DIR, "media_1788756198739.jpg")),
        ("Image 2", os.path.join(USER_IMAGES_DIR, "media_1788756252003.jpg")),
        ("Image 3", os.path.join(USER_IMAGES_DIR, "media_1788756427309.jpg")),
    ]

    ocr = EasyOCRProvider()

    for label, path in img_files:
        frame = cv2.imread(path)
        h, w = frame.shape[:2]
        print(f"\n--- {label} ({w}x{h}) ---")

        # Bottom right crop: x from 80% to 98%, y from 70% to 98%
        x1, y1 = int(w * 0.78), int(h * 0.70)
        x2, y2 = int(w * 0.98), int(h * 0.98)
        crop = frame[y1:y2, x1:x2]
        cv2.imwrite(f"debug_captures/scratch/hud_{label.replace(' ', '_')}.png", crop)

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # 1. Red Heart (HP) - Red has two ranges in HSV (0..10 and 165..180)
        r1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([12, 255, 255]))
        r2 = cv2.inRange(hsv, np.array([165, 80, 80]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(r1, r2)
        cnts_r, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"Red heart candidate contours found: {len(cnts_r)}")
        for c in cnts_r:
            bx, by, bw, bh = cv2.boundingRect(c)
            print(f"   Red box: local=({bx}, {by}, {bw}, {bh})")
            if 30 <= bw <= 120 and 30 <= bh <= 120:
                icon_crop = crop[by:by+bh, bx:bx+bw]
                top_half = icon_crop[int(bh*0.08):int(bh*0.58), :]
                hsv_top = cv2.cvtColor(top_half, cv2.COLOR_BGR2HSV)
                wmask = cv2.inRange(hsv_top, np.array([0, 0, 160]), np.array([180, 90, 255]))
                up = cv2.resize(wmask, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
                txt = ocr.extract_text(up)
                print(f"   HP Heart OCR Text: '{txt}' -> parsed: {ocr.parse_first_int(txt)}")
                cv2.imwrite(f"debug_captures/scratch/hp_crop_{label.replace(' ', '_')}.png", up)

        # 2. Blue Star (PA)
        blue_mask = cv2.inRange(hsv, np.array([85, 50, 80]), np.array([130, 255, 255]))
        cnts_b, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"Blue star candidate contours found: {len(cnts_b)}")
        for c in cnts_b:
            bx, by, bw, bh = cv2.boundingRect(c)
            print(f"   Blue box: local=({bx}, {by}, {bw}, {bh})")
            if 15 <= bw <= 60 and 15 <= bh <= 60:
                icon_crop = crop[by:by+bh, bx:bx+bw]
                hsv_icon = cv2.cvtColor(icon_crop, cv2.COLOR_BGR2HSV)
                wmask = cv2.inRange(hsv_icon, np.array([0, 0, 160]), np.array([180, 90, 255]))
                up = cv2.resize(wmask, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
                txt = ocr.extract_text(up)
                print(f"   PA Star OCR Text: '{txt}' -> parsed: {ocr.parse_first_int(txt)}")

        # 3. Green Diamond (PM)
        green_mask = cv2.inRange(hsv, np.array([35, 40, 80]), np.array([85, 255, 255]))
        cnts_g, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"Green diamond candidate contours found: {len(cnts_g)}")
        for c in cnts_g:
            bx, by, bw, bh = cv2.boundingRect(c)
            print(f"   Green box: local=({bx}, {by}, {bw}, {bh})")
            if 15 <= bw <= 60 and 15 <= bh <= 60:
                icon_crop = crop[by:by+bh, bx:bx+bw]
                hsv_icon = cv2.cvtColor(icon_crop, cv2.COLOR_BGR2HSV)
                wmask = cv2.inRange(hsv_icon, np.array([0, 0, 160]), np.array([180, 90, 255]))
                up = cv2.resize(wmask, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
                txt = ocr.extract_text(up)
                print(f"   PM Diamond OCR Text: '{txt}' -> parsed: {ocr.parse_first_int(txt)}")
                cv2.imwrite(f"debug_captures/scratch/pm_crop_{label.replace(' ', '_')}.png", up)

if __name__ == "__main__":
    inspect_user_hud_details()
