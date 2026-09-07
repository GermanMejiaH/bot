"""ResourceDetector extracting PA, PM, and HP HUD values via abstract OCRProvider."""

import cv2
import numpy as np

from dta.config.settings import Settings, get_settings
from dta.core.logger import logger
from dta.models.detections import ResourceDetection
from dta.services.ocr.easyocr_provider import EasyOCRProvider
from dta.services.ocr.provider import OCRProvider


class ResourceDetector:
    """Detector for HUD resources (PA, PM, HP) relying on an abstract OCRProvider."""

    def __init__(
        self,
        ocr_provider: OCRProvider | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.ocr_provider = ocr_provider or EasyOCRProvider()
        self.settings = settings or get_settings()

    def _find_red_heart_box(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        """Locate the main Red Heart HUD icon in the bottom-right region of the frame."""
        if frame is None or frame.size == 0:
            return None

        h, w = frame.shape[:2]
        search_x1 = int(w * 0.70)
        search_y1 = int(h * 0.60)
        crop_hud = frame[search_y1:h, search_x1:w]
        hsv = cv2.cvtColor(crop_hud, cv2.COLOR_BGR2HSV)

        # Red Heart covers both lower (0..12) and upper (165..180) Hue ranges in HSV
        r1 = cv2.inRange(hsv, np.array([0, 85, 85]), np.array([12, 255, 255]))
        r2 = cv2.inRange(hsv, np.array([165, 85, 85]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(r1, r2)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            bx, by, bw, bh = cv2.boundingRect(c)
            aspect = bh / float(bw) if bw > 0 else 0
            if 30 <= bw <= 110 and 30 <= bh <= 110 and 0.75 <= aspect <= 1.30:
                # Return global (x, y, w, h) of red heart
                return (search_x1 + bx, search_y1 + by, bw, bh)

        return None

    def detect_hp(self, frame: np.ndarray) -> int | None:
        """Extract HP (Health Points) integer value from dynamic Red Heart HUD icon top half."""
        heart_box = self._find_red_heart_box(frame)
        val: int | None = None

        if heart_box:
            hx, hy, hw, hh = heart_box
            hp_crop = frame[hy + int(hh * 0.08) : hy + int(hh * 0.58), hx : hx + hw]
            if hp_crop.size > 0:
                # White text threshold
                wmask = cv2.inRange(hp_crop, np.array([180, 180, 180]), np.array([255, 255, 255]))
                up = cv2.resize(wmask, (0, 0), fx=3.5, fy=3.5, interpolation=cv2.INTER_CUBIC)
                up = cv2.copyMakeBorder(up, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
                text = self.ocr_provider.extract_text(up)
                val = self.ocr_provider.parse_first_int(text)

        if val is None:
            roi_config = self.settings.rois.hp
            roi = (roi_config.x_min, roi_config.y_min, roi_config.x_max, roi_config.y_max)
            val = self.ocr_provider.extract_number(frame, roi=roi)

        logger.debug(f"Detected HP: {val}")
        return val

    def detect_pa(self, frame: np.ndarray) -> int | None:
        """Extract PA (Action Points) integer value from dynamic Blue Star HUD icon."""
        heart_box = self._find_red_heart_box(frame)
        val: int | None = None

        if heart_box:
            hx, hy, hw, hh = heart_box
            # PA star is anchored directly below-left of Red Heart
            pa_crop = frame[hy + int(hh * 0.95) : hy + int(hh * 1.65), max(0, hx - int(hw * 0.20)) : hx + int(hw * 0.45)]
            if pa_crop.size > 0:
                wmask = cv2.inRange(pa_crop, np.array([180, 180, 180]), np.array([255, 255, 255]))
                up = cv2.resize(wmask, (0, 0), fx=3.5, fy=3.5, interpolation=cv2.INTER_CUBIC)
                up = cv2.copyMakeBorder(up, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=0)
                text = self.ocr_provider.extract_text(up)
                val = self.ocr_provider.parse_first_int(text)

        if val is None:
            roi_config = self.settings.rois.pa
            roi = (roi_config.x_min, roi_config.y_min, roi_config.x_max, roi_config.y_max)
            val = self.ocr_provider.extract_number(frame, roi=roi)

        logger.debug(f"Detected PA: {val}")
        return val

    def detect_pm(self, frame: np.ndarray) -> int | None:
        """Extract PM (Movement Points) integer value from dynamic Green Diamond HUD icon."""
        heart_box = self._find_red_heart_box(frame)
        val: int | None = None

        if heart_box:
            hx, hy, hw, hh = heart_box
            # PM diamond is anchored directly below-right of Red Heart
            pm_crop = frame[hy + int(hh * 0.95) : hy + int(hh * 1.65), hx + int(hw * 0.55) : min(frame.shape[1], hx + int(hw * 1.25))]
            if pm_crop.size > 0:
                wmask = cv2.inRange(pm_crop, np.array([170, 170, 170]), np.array([255, 255, 255]))
                up = cv2.resize(wmask, (0, 0), fx=3.5, fy=3.5, interpolation=cv2.INTER_CUBIC)
                up = cv2.copyMakeBorder(up, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=0)
                text = self.ocr_provider.extract_text(up)
                parsed = self.ocr_provider.parse_first_int(text)
                if parsed is not None:
                    # Filter valid PM range (0..20)
                    val = parsed if parsed <= 20 else parsed % 10

        if val is None:
            roi_config = self.settings.rois.pm
            roi = (roi_config.x_min, roi_config.y_min, roi_config.x_max, roi_config.y_max)
            val = self.ocr_provider.extract_number(frame, roi=roi)

        logger.debug(f"Detected PM: {val}")
        return val

    def detect_resources(self, frame: np.ndarray) -> ResourceDetection:
        """Extract all HUD resource levels (PA, PM, HP) into a structured ResourceDetection object."""
        if frame is None or frame.size == 0:
            return ResourceDetection(confidence=0.0)

        pa = self.detect_pa(frame)
        pm = self.detect_pm(frame)
        hp = self.detect_hp(frame)

        confidence = 1.0 if (pa is not None or pm is not None or hp is not None) else 0.0
        return ResourceDetection(pa=pa, pm=pm, hp=hp, confidence=confidence)

