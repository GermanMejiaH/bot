"""Abstract OCRProvider interface and image preprocessing utilities for DTA."""

import re
from abc import ABC, abstractmethod
from typing import Any

import cv2
import numpy as np


class OCRProvider(ABC):
    """Abstract interface defining contract for all OCR providers."""

    @abstractmethod
    def extract_text(self, image: Any, roi: tuple[int, int, int, int] | None = None) -> str:
        """Extract plain text string from image or ROI (x_min, y_min, x_max, y_max)."""
        pass

    @abstractmethod
    def extract_number(self, image: Any, roi: tuple[int, int, int, int] | None = None) -> int | None:
        """Extract clean integer numeric value from image or ROI."""
        pass

    @staticmethod
    def crop_roi(image: np.ndarray, roi: tuple[int, int, int, int] | None) -> np.ndarray:
        """Crop region of interest (x_min, y_min, x_max, y_max) from numpy image array."""
        if roi is None:
            return image
        x_min, y_min, x_max, y_max = roi
        h, w = image.shape[:2]
        x_min = max(0, min(x_min, w))
        x_max = max(x_min, min(x_max, w))
        y_min = max(0, min(y_min, h))
        y_max = max(y_min, min(y_max, h))
        return image[y_min:y_max, x_min:x_max]

    @staticmethod
    def preprocess_image(image: np.ndarray, scale_factor: float = 2.0) -> np.ndarray:
        """Preprocess HUD image for enhanced OCR accuracy (grayscale, resize, contrast/thresholding)."""
        if image is None or image.size == 0:
            return image

        # Convert to grayscale if 3-channel BGR
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            gray = image.copy()

        # Upscale ROI for higher character resolution
        if scale_factor > 1.0:
            h, w = gray.shape[:2]
            gray = cv2.resize(gray, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_CUBIC)

        # Contrast enhancement via Otsu thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    @staticmethod
    def parse_first_int(text: str) -> int | None:
        """Parse the first valid integer sequence from an extracted text string."""
        digits = re.findall(r"\d+", text)
        if digits:
            return int(digits[0])
        return None
