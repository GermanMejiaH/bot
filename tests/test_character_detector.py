"""Unit tests for Stage 5 CharacterDetector using synthetic numpy images."""

import cv2
import numpy as np

from dta.detectors.character_detector import CharacterDetector
from dta.models.detections import RawCharacterDetection


def test_character_detector_contours() -> None:
    # Synthetic black image with vertical rectangle candidate (width 40, height 100)
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (90, 150), (255, 255, 255), -1)

    detector = CharacterDetector(min_width=10, min_height=20, min_area=100.0)
    detections = detector.detect_contours(img)

    assert len(detections) >= 1
    det = detections[0]
    assert isinstance(det, RawCharacterDetection)
    assert det.method == "contour"
    assert 30 <= det.bbox.w <= 50
    assert 90 <= det.bbox.h <= 125


def test_character_detector_hsv_regions() -> None:
    # Synthetic image with a bright red vertical rectangle (width 30, height 70)
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (80, 120), (0, 0, 255), -1)

    detector = CharacterDetector(min_width=10, min_height=20, min_area=50.0, min_y=0, max_y=1080)
    lower_red = (0, 100, 100)
    upper_red = (10, 255, 255)

    detections = detector.detect_hsv_regions(img, lower_red, upper_red)

    assert len(detections) == 1
    assert detections[0].method == "hsv"
    assert 28 <= detections[0].bbox.w <= 32
    assert 68 <= detections[0].bbox.h <= 72



def test_character_detector_merge_candidates() -> None:
    detector = CharacterDetector(min_width=10, min_height=20, iou_threshold=0.3)
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (90, 150), (255, 255, 255), -1)

    # Detect contours & blobs, then merge
    contours = detector.detect_contours(img)
    blobs = detector.detect_blobs(img)
    all_candidates = contours + blobs

    merged = detector.merge_candidates(all_candidates)
    assert len(merged) <= len(all_candidates)


def test_character_detector_debug_overlay() -> None:
    detector = CharacterDetector(min_width=10, min_height=20, min_y=0, max_y=1080)
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (80, 120), (255, 255, 255), -1)

    detections = detector.detect_contours(img)
    overlay = detector.generate_debug_overlay(img, detections)

    assert overlay.shape == img.shape
    assert not np.array_equal(overlay, img)  # Bounding box green/red drawn onto overlay

