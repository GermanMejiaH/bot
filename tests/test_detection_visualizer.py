"""Unit tests for DetectionVisualizer visual detection debugger."""

import os
import shutil
import tempfile

import cv2
import numpy as np

from dta.config.settings import Settings
from dta.debug.detection_visualizer import DetectionVisualizer, save_detection_overlay
from dta.models.detections import BoundingBox, RawCharacterDetection


def test_detection_visualizer_colors() -> None:
    visualizer = DetectionVisualizer()

    assert visualizer.get_method_color("contour") == (0, 255, 255)      # Yellow
    assert visualizer.get_method_color("hsv") == (0, 255, 0)            # Green
    assert visualizer.get_method_color("combat_base") == (0, 0, 255)    # Red
    assert visualizer.get_method_color("unknown") == (255, 255, 0)      # Default Cyan


def test_detection_visualizer_draw_detections() -> None:
    visualizer = DetectionVisualizer()
    frame = np.zeros((300, 300, 3), dtype=np.uint8)

    detections = [
        RawCharacterDetection(
            bbox=BoundingBox(x=10, y=10, w=30, h=40),
            centroid=(25, 30),
            area=1200.0,
            confidence=0.85,
            method="contour",
        ),
        RawCharacterDetection(
            bbox=BoundingBox(x=60, y=60, w=40, h=50),
            centroid=(80, 85),
            area=2000.0,
            confidence=0.88,
            method="hsv",
        ),
        RawCharacterDetection(
            bbox=BoundingBox(x=120, y=120, w=50, h=80),
            centroid=(145, 160),
            area=4000.0,
            confidence=0.92,
            method="combat_base",
        ),
    ]

    overlay = visualizer.draw_detections(frame, detections)

    assert overlay.shape == frame.shape
    assert not np.array_equal(overlay, frame)


def test_save_detection_overlay_function_and_method() -> None:
    temp_dir = tempfile.mkdtemp()
    try:
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        detections = [
            RawCharacterDetection(
                bbox=BoundingBox(x=20, y=20, w=40, h=60),
                centroid=(40, 50),
                area=2400.0,
                confidence=0.90,
                method="combat_base",
            )
        ]

        custom_path = os.path.join(temp_dir, "sub", "test_overlay.png")
        saved_path = save_detection_overlay(frame, detections, output_path=custom_path)

        assert saved_path == custom_path
        assert os.path.exists(saved_path)

        saved_img = cv2.imread(saved_path)
        assert saved_img is not None
        assert saved_img.shape == frame.shape
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_detection_visualizer_settings_integration() -> None:
    temp_dir = tempfile.mkdtemp()
    try:
        settings = Settings()
        settings.debug.output_dir = temp_dir
        settings.debug.save_detection_overlays = True

        visualizer = DetectionVisualizer(settings=settings)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = []

        saved_path = visualizer.save_detection_overlay(frame, detections)

        assert os.path.exists(saved_path)
        assert temp_dir in saved_path
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
