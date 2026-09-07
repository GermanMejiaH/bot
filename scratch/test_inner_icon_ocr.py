"""Test inner icon cropping for PA and PM OCR."""

import cv2

from dta.services.ocr.easyocr_provider import EasyOCRProvider


def test_inner_crops():
    ocr = EasyOCRProvider()

    for _idx, img_name in enumerate(["Image_1", "Image_2", "Image_3"]):
        pa_path = f"debug_captures/scratch/crop_pa_{img_name}.png"
        pm_path = f"debug_captures/scratch/crop_pm_{img_name}.png"

        pa_img = cv2.imread(pa_path)
        pm_img = cv2.imread(pm_path)

        if pa_img is not None:
            # Crop center 75% of PA star
            h, w = pa_img.shape[:2]
            pa_center = pa_img[int(h*0.12):int(h*0.88), int(w*0.12):int(w*0.88)]
            # Upscale without thresholding (EasyOCR handles color/gray naturally)
            pa_up = cv2.resize(pa_center, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
            pa_txt = ocr.extract_text(pa_up)
            print(f"{img_name} PA -> Raw: '{pa_txt}' -> Int: {ocr.parse_first_int(pa_txt)}")

        if pm_img is not None:
            h, w = pm_img.shape[:2]
            pm_center = pm_img[int(h*0.12):int(h*0.88), int(w*0.12):int(w*0.88)]
            pm_up = cv2.resize(pm_center, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
            pm_txt = ocr.extract_text(pm_up)
            print(f"{img_name} PM -> Raw: '{pm_txt}' -> Int: {ocr.parse_first_int(pm_txt)}")

if __name__ == "__main__":
    test_inner_crops()
