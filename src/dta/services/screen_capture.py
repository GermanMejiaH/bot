"""Thread-safe ScreenCaptureService using MSS for high-performance window and region capture."""

import threading
import time
from pathlib import Path

import cv2
import numpy as np

from dta.config.settings import Settings, get_settings
from dta.core.logger import logger
from dta.events.event_bus import EventBus, get_event_bus
from dta.events.events import FrameCaptured
from dta.models.frame import CapturedFrame, FrameMetadata
from dta.vision.window_finder import WindowFinder


class ScreenCaptureService:
    """Service responsible for continuously capturing target window frames using MSS."""

    def __init__(
        self,
        settings: Settings | None = None,
        event_bus: EventBus | None = None,
        save_raw_captures: bool = True,
        raw_output_dir: str = "debug_captures/raw",
    ) -> None:
        self.settings = settings or get_settings()
        self.event_bus = event_bus or get_event_bus()
        self.save_raw_captures = save_raw_captures
        self.raw_output_dir = raw_output_dir

        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._latest_frame: CapturedFrame | None = None
        self._frame_count = 0

        if self.save_raw_captures and not Path(self.raw_output_dir).exists():
            Path(self.raw_output_dir).mkdir(parents=True, exist_ok=True)

    @property
    def is_running(self) -> bool:
        """Return True if background capture thread is active."""
        return self._running

    def start(self) -> None:
        """Start the background screen capture thread."""
        with self._lock:
            if self._running:
                logger.warning("ScreenCaptureService is already running.")
                return

            self._running = True
            self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="ScreenCaptureThread")
            self._thread.start()

            bounds = WindowFinder.get_window_bounds(self.settings.window_name)
            hwnd = WindowFinder.get_hwnd(self.settings.window_name)
            hwnd_str = hex(hwnd) if hwnd else "None"
            logger.info(
                f"[DTA Window Finder]\n"
                f"  Title: {self.settings.window_name}\n"
                f"  HWND: {hwnd_str}\n"
                f"  Resolution: {bounds[2] if bounds else 'Unknown'}x{bounds[3] if bounds else 'Unknown'}\n"
                f"  Client Area: {'OK' if bounds else 'FAILED'}"
            )

    def stop(self) -> None:
        """Stop the background screen capture thread."""
        with self._lock:
            if not self._running:
                return
            self._running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            logger.info("ScreenCaptureService stopped.")

    def get_latest_frame(self) -> CapturedFrame | None:
        """Retrieve the latest captured frame (thread-safe, zero-copy reference)."""
        with self._lock:
            return self._latest_frame

    def _capture_loop(self) -> None:
        """Background thread loop capturing frames at configured FPS."""
        interval = 1.0 / max(1, self.settings.capture_fps)

        try:
            import mss  # Dynamic import for runtime mocking
        except ImportError:
            logger.error("mss library is not installed. Screen capture loop cannot execute.")
            self._running = False
            return

        try:
            with mss.mss() as sct:
                while self._running:
                    start_time = time.time()

                    # Find target window bounding box or fallback to primary monitor
                    bounds = WindowFinder.get_window_bounds(self.settings.window_name)
                    if bounds:
                        left, top, width, height = bounds
                        monitor = {"left": left, "top": top, "width": width, "height": height}
                    else:
                        # Fallback to primary monitor bounds
                        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]

                    sct_img = sct.grab(monitor)
                    # Convert BGRA MSS raw bytes to BGR numpy array (single view conversion)
                    frame_bgr = np.array(sct_img, dtype=np.uint8)[:, :, :3]

                    self._frame_count += 1
                    frame_id_str = f"frame_{self._frame_count:06d}"
                    metadata = FrameMetadata(
                        frame_id=frame_id_str,
                        timestamp=time.time(),
                        width=frame_bgr.shape[1],
                        height=frame_bgr.shape[0],
                        source_window=self.settings.window_name,
                        capture_fps=self.settings.capture_fps,
                    )

                    captured_frame = CapturedFrame(metadata=metadata, image=frame_bgr)

                    with self._lock:
                        self._latest_frame = captured_frame

                    # Step 2: Auto-save raw unannotated frame BEFORE any processing/overlay
                    if self.save_raw_captures and self._frame_count % 30 == 1:
                        raw_filename = f"raw_{frame_id_str}.png"
                        raw_filepath = Path(self.raw_output_dir) / raw_filename
                        cv2.imwrite(str(raw_filepath), frame_bgr)
                        logger.debug(f"Saved raw frame to {raw_filepath}")

                    # Publish FrameCaptured event
                    event = FrameCaptured(
                        frame_id=captured_frame.frame_id,
                        frame=captured_frame.image,
                        width=captured_frame.metadata.width,
                        height=captured_frame.metadata.height,
                        captured_frame=captured_frame,
                    )
                    self.event_bus.publish(event)

                    # FPS throttling delay calculation
                    elapsed = time.time() - start_time
                    sleep_time = max(0.0, interval - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

        except Exception as e:
            logger.error(f"Unexpected error in capture loop: {e}")
            self._running = False


    def save_raw_frame(self, target_path: str | Path = "screenshots/raw/capture.png") -> str:
        """Save latest raw frame to disk."""
        frame_obj = self.get_latest_frame()
        if frame_obj is None or frame_obj.image is None:
            raise ValueError("No frame available to save.")

        path = Path(target_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), frame_obj.image)
        logger.info(f"Saved raw frame to {path}")
        return str(path)

    def save_annotated_frame(self, frame: np.ndarray, target_path: str | Path = "screenshots/debug/debug.png") -> str:
        """Save an annotated debug frame to disk (makes an explicit copy before writing)."""
        path = Path(target_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Explicit copy to preserve original frame buffer
        output_frame = frame.copy()
        cv2.imwrite(str(path), output_frame)
        logger.info(f"Saved annotated debug frame to {path}")
        return str(path)
