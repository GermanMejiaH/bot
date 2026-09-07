"""Script to test CharacterDetector fixes against raw captures."""

import glob

import cv2

from dta.detectors.character_detector import CharacterDetector
from dta.vision.map_roi_extractor import MapROIExtractor


def test_character_detection():
    raw_files = sorted(glob.glob("debug_captures/raw/*.png"))
    roi_extractor = MapROIExtractor()
    detector = CharacterDetector()

    for idx, path in enumerate(raw_files[:5]):
        raw_frame = cv2.imread(path)
        if raw_frame is None:
            continue

        h, w = raw_frame.shape[:2]

        # 1. Apply updated ROI mask
        # Add mask for bottom-right HUD area (x: 76%-100%, y: 65%-100%)
        masked_frame = roi_extractor.apply_roi_mask(raw_frame)
        masked_frame[int(h * 0.65):h, int(w * 0.76):w] = 0
        masked_frame[0:h, int(w * 0.96):w] = 0

        # Run detector
        detections = detector.detect_characters(masked_frame)
        print(f"Frame {idx} ({path}): {len(detections)} character detections.")
        for d in detections:
            print(f"   -> Box x={d.bbox.x}, y={d.bbox.y}, w={d.bbox.w}, h={d.bbox.h}, method={d.method}, conf={d.confidence:.2f}")

        # Render debug overlay
        overlay = detector.generate_debug_overlay(masked_frame, detections)
        cv2.imwrite(f"debug_captures/scratch/det_overlay_{idx}.png", overlay)

if __name__ == "__main__":
    test_character_detection()
