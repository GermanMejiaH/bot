"""Detailed script to locate and extract PA (Blue Star), PM (Green Diamond), and HP (Heart) from raw captures."""

import glob

import cv2
import numpy as np

from dta.services.ocr.easyocr_provider import EasyOCRProvider


def test_hud_extraction():
    raw_files = sorted(glob.glob("debug_captures/raw/*.png"))
    ocr = EasyOCRProvider()

    for idx, path in enumerate(raw_files[:3]):
        frame = cv2.imread(path)
        if frame is None:
            continue
        h, w = frame.shape[:2]

        # HUD sits at bottom right: x from 70% to 100%, y from 60% to 100%
        br_y1, br_y2 = int(h * 0.60), h
        br_x1, br_x2 = int(w * 0.70), w
        hud_crop = frame[br_y1:br_y2, br_x1:br_x2]
        hsv_hud = cv2.cvtColor(hud_crop, cv2.COLOR_BGR2HSV)

        # 1. Blue Star (PA)
        blue_mask = cv2.inRange(hsv_hud, np.array([90, 80, 100]), np.array([130, 255, 255]))
        cnts, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pa_val = None
        for c in cnts:
            x, y, bw, bh = cv2.boundingRect(c)
            if 20 <= bw <= 60 and 20 <= bh <= 60:
                star_crop = hud_crop[y:y+bh, x:x+bw]
                # White mask inside star
                hsv_star = cv2.cvtColor(star_crop, cv2.COLOR_BGR2HSV)
                wmask = cv2.inRange(hsv_star, np.array([0, 0, 180]), np.array([180, 80, 255]))
                up = cv2.resize(wmask, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
                txt = ocr.extract_text(up)
                pa_val = ocr.parse_first_int(txt)
                cv2.imwrite(f"debug_captures/scratch/pa_{idx}.png", up)
                break

        # 2. Green Diamond (PM)
        green_mask = cv2.inRange(hsv_hud, np.array([35, 80, 100]), np.array([85, 255, 255]))
        cnts_g, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pm_val = None
        for c in cnts_g:
            x, y, bw, bh = cv2.boundingRect(c)
            if 20 <= bw <= 60 and 20 <= bh <= 60:
                diamond_crop = hud_crop[y:y+bh, x:x+bw]
                hsv_dia = cv2.cvtColor(diamond_crop, cv2.COLOR_BGR2HSV)
                wmask = cv2.inRange(hsv_dia, np.array([0, 0, 180]), np.array([180, 80, 255]))
                up = cv2.resize(wmask, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
                txt = ocr.extract_text(up)
                pm_val = ocr.parse_first_int(txt)
                cv2.imwrite(f"debug_captures/scratch/pm_{idx}.png", up)
                break

        # 3. Red Heart (HP)
        r1 = cv2.inRange(hsv_hud, np.array([0, 100, 100]), np.array([12, 255, 255]))
        r2 = cv2.inRange(hsv_hud, np.array([168, 100, 100]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(r1, r2)
        cnts_r, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hp_val = None
        for c in cnts_r:
            x, y, bw, bh = cv2.boundingRect(c)
            if 40 <= bw <= 120 and 40 <= bh <= 120:
                heart_crop = hud_crop[y:y+bh, x:x+bw]
                top_half = heart_crop[int(bh*0.12):int(bh*0.58), int(bw*0.05):int(bw*0.95)]
                hsv_top = cv2.cvtColor(top_half, cv2.COLOR_BGR2HSV)
                wmask = cv2.inRange(hsv_top, np.array([0, 0, 180]), np.array([180, 80, 255]))
                up = cv2.resize(wmask, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
                txt = ocr.extract_text(up)
                hp_val = ocr.parse_first_int(txt)
                cv2.imwrite(f"debug_captures/scratch/hp_{idx}.png", up)
                break

        print(f"Frame {idx}: PA={pa_val}, PM={pm_val}, HP={hp_val}")

if __name__ == "__main__":
    test_hud_extraction()
