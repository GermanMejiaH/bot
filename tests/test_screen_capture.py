"""Unit tests for Stage 4 Screen Capture and Window Finder using mocks."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from dta.events.event_bus import EventBus
from dta.events.events import FrameCaptured
from dta.models.frame import CapturedFrame, FrameMetadata
from dta.services.screen_capture import ScreenCaptureService
from dta.vision.window_finder import WindowFinder


def test_frame_metadata_and_captured_frame() -> None:
    meta = FrameMetadata(width=1920, height=1080, source_window="Dofus", capture_fps=15)
    dummy_img = np.zeros((1080, 1920, 3), dtype=np.uint8)

    captured = CapturedFrame(metadata=meta, image=dummy_img)

    assert captured.metadata.width == 1920
    assert captured.metadata.height == 1080
    assert captured.shape == (1080, 1920, 3)
    assert captured.image is dummy_img  # Zero-copy reference equality


def test_window_finder_mocked() -> None:
    mock_window = MagicMock()
    mock_window.left = 100
    mock_window.top = 200
    mock_window.width = 1920
    mock_window.height = 1080
    mock_window.visible = True
    mock_window.isMinimized = False

    mock_gw = MagicMock()
    mock_gw.getWindowsWithTitle.return_value = [mock_window]

    with patch.dict(sys.modules, {"pygetwindow": mock_gw}):
        assert WindowFinder.is_window_available("Dofus") is True
        assert WindowFinder.get_window_bounds("Dofus") == (100, 200, 1920, 1080)
        assert WindowFinder.get_window_resolution("Dofus") == (1920, 1080)


def test_window_finder_missing_window() -> None:
    mock_gw = MagicMock()
    mock_gw.getWindowsWithTitle.return_value = []

    with patch.dict(sys.modules, {"pygetwindow": mock_gw}):
        assert WindowFinder.is_window_available("NonExistentWindow") is False
        assert WindowFinder.get_window_bounds("NonExistentWindow") is None
        assert WindowFinder.get_window_resolution("NonExistentWindow") is None


def test_screen_capture_service_lifecycle_and_events() -> None:
    event_bus = EventBus()
    received_events: list[FrameCaptured] = []

    event_bus.subscribe(FrameCaptured, lambda e: received_events.append(e))

    dummy_bgr = np.ones((100, 200, 4), dtype=np.uint8) * 128

    mock_mss_instance = MagicMock()
    mock_mss_instance.monitors = [{"left": 0, "top": 0, "width": 800, "height": 600}]
    mock_mss_instance.grab.return_value = dummy_bgr

    mock_mss_module = MagicMock()
    mock_mss_module.mss.return_value.__enter__.return_value = mock_mss_instance

    with patch.dict(sys.modules, {"mss": mock_mss_module}):
        service = ScreenCaptureService(event_bus=event_bus)
        assert service.is_running is False

        service.start()
        assert service.is_running is True

        # Allow capture loop to run for a few iterations
        import time
        time.sleep(0.2)

        latest = service.get_latest_frame()
        service.stop()

        assert service.is_running is False
        assert latest is not None
        assert isinstance(latest, CapturedFrame)
        assert len(received_events) > 0
        assert received_events[0].captured_frame is not None


def test_screen_capture_saving(tmp_path) -> None:
    service = ScreenCaptureService()
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    meta = FrameMetadata(width=100, height=100)
    service._latest_frame = CapturedFrame(metadata=meta, image=dummy_img)

    raw_path = tmp_path / "raw.png"
    debug_path = tmp_path / "debug.png"

    saved_raw = service.save_raw_frame(raw_path)
    saved_debug = service.save_annotated_frame(dummy_img, debug_path)

    assert Path(saved_raw).exists()
    assert Path(saved_debug).exists()
