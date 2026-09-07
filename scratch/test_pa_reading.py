"""Test PA reading with grayscale, Otsu, thresholding, and EasyOCR options."""

import cv2

from dta.services.ocr.easyocr_provider import EasyOCRProvider


def test_pa():
    ocr = EasyOCRProvider()

    for tag in ["Image_1", "Image_2", "Image_3"]:
        raw = cv2.imread(f"debug_captures/scratch/pa_raw_{tag}.png")
        mask = cv2.imread(f"debug_captures/scratch/pa_mask_{tag}.png", cv2.IMREAD_GRAYSCALE)

        if raw is None:
            continue

        # 1. Raw upscaled x3
        raw_up = cv2.resize(raw, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        txt1 = ocr.extract_text(raw_up)

        # 2. Grayscale upscaled x3
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        gray_up = cv2.resize(gray, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        txt2 = ocr.extract_text(gray_up)

        # 3. Mask upscaled x3 padded
        padded = cv2.copyMakeBorder(mask, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=0)
        txt3 = ocr.extract_text(padded)

        # 4. Inverted mask padded
        inv = cv2.bitwise_not(padded)
        txt4 = ocr.extract_text(inv)

        print(f"--- {tag} ---")
        print(f"  Raw Upscaled: '{txt1}' -> {ocr.parse_first_int(txt1)}")
        print(f"  Gray Upscaled: '{txt2}' -> {ocr.parse_first_int(txt2)}")
        print(f"  Mask Padded: '{txt3}' -> {ocr.parse_first_int(txt3)}")
        print(f"  Inverted Mask: '{txt4}' -> {ocr.parse_first_int(txt4)}")

if __name__ == "__main__":
    test_pa()
