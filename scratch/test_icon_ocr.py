"""Test inverted mask and Tesseract/EasyOCR on PA and PM crops."""

import glob

import cv2
import numpy as np

from dta.services.ocr.easyocr_provider import EasyOCRProvider


def test_inverted_crops():
    ocr = EasyOCRProvider()
    crops = sorted(glob.glob("debug_captures/scratch/crop_*.png"))

    for path in crops:
        img = cv2.imread(path)
        if img is None:
            continue

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # White digit mask
        white_mask = cv2.inRange(hsv, np.array([0, 0, 160]), np.array([180, 80, 255]))

        # Invert mask -> black text on white background
        inverted = cv2.bitwise_not(white_mask)

        # Upscale 3x
        up = cv2.resize(inverted, (0, 0), fx=3.5, fy=3.5, interpolation=cv2.INTER_CUBIC)
        padded = cv2.copyMakeBorder(up, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)

        txt = ocr.extract_text(padded)
        val = ocr.parse_first_int(txt)

        print(f"Inverted {os.path.basename(path):25s} -> Raw Text: '{txt}' -> Parsed Int: {val}")

if __name__ == "__main__":
    import os
    test_inverted_crops()
