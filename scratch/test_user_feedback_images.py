"""Script to inspect the user's latest 3 screenshots and test fixes for:
1. Initial placement phase empty red/blue tiles false positive entity detection.
2. ResourceDetector PA, PM, HP OCR extraction.
3. Overlapping double entity detection on character.
"""

import os

import cv2

from dta.detectors.character_detector import CharacterDetector
from dta.detectors.resource_detector import ResourceDetector
from dta.services.ocr.easyocr_provider import EasyOCRProvider

USER_IMAGES_DIR = r"C:\Users\Andres\.gemini\antigravity-ide\brain\aaee6007-432e-4086-9822-f16f6f0a3661\.user_uploaded"

def test_user_images():
    img_files = [
        ("Image 1 (Placement Phase)", os.path.join(USER_IMAGES_DIR, "media_1788756198739.jpg")),
        ("Image 2 (Combat Turn)", os.path.join(USER_IMAGES_DIR, "media_1788756252003.jpg")),
        ("Image 3 (Double Detection)", os.path.join(USER_IMAGES_DIR, "media_1788756427309.jpg")),
    ]

    ocr = EasyOCRProvider()
    res_detector = ResourceDetector(ocr_provider=ocr)
    char_detector = CharacterDetector()

    for label, path in img_files:
        if not os.path.exists(path):
            print(f"File missing: {path}")
            continue

        frame = cv2.imread(path)
        h, w = frame.shape[:2]
        print(f"\n==================== {label} ({w}x{h}) ====================")

        # 1. Test Resource Detection (PA, PM, HP)
        pa = res_detector.detect_pa(frame)
        pm = res_detector.detect_pm(frame)
        hp = res_detector.detect_hp(frame)
        print(f"Resource Detection Result: PA={pa}, PM={pm}, HP={hp}")

        # Save HUD crop for visual debugging
        search_x1 = int(w * 0.70)
        search_y1 = int(h * 0.60)
        hud_crop = frame[search_y1:h, search_x1:w]
        cv2.imwrite(f"debug_captures/scratch/user_hud_{label.split()[1]}.png", hud_crop)

        # 2. Test Character Detection
        raw_dets = char_detector.detect_characters(frame)
        print(f"Character Detection Count: {len(raw_dets)}")
        for d in raw_dets:
            print(f"   -> Box ({d.bbox.x}, {d.bbox.y}, {d.bbox.w}, {d.bbox.h}), method={d.method}, conf={d.confidence:.2f}")

if __name__ == "__main__":
    test_user_images()
