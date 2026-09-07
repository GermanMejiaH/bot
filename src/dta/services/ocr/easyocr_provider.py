"""Concrete EasyOCR implementation of OCRProvider with lazy loading and reader caching."""

from typing import Any

import numpy as np

from dta.core.logger import logger
from dta.services.ocr.provider import OCRProvider


class EasyOCRProvider(OCRProvider):
    """EasyOCR implementation of OCRProvider interface."""

    def __init__(self, languages: list[str] | None = None, gpu: bool = False) -> None:
        self.languages = languages or ["fr", "en"]
        self.gpu = gpu
        self._reader: Any = None

    def _get_reader(self) -> Any:
        """Lazily initialize and cache the EasyOCR Reader instance."""
        if self._reader is None:
            logger.info(f"Initializing EasyOCR Reader (languages={self.languages}, gpu={self.gpu})...")
            import easyocr  # Lazy import to keep module loading lightweight

            self._reader = easyocr.Reader(self.languages, gpu=self.gpu)
        return self._reader

    def extract_text(self, image: Any, roi: tuple[int, int, int, int] | None = None) -> str:
        """Extract plain text using EasyOCR on preprocessed image/ROI."""
        if image is None:
            return ""

        img_np = np.asarray(image)
        cropped = self.crop_roi(img_np, roi)
        if cropped.size == 0:
            return ""

        preprocessed = self.preprocess_image(cropped)
        reader = self._get_reader()

        # easyocr.readtext returns list of tuples: (bbox, text, prob)
        results = reader.readtext(preprocessed, detail=0)
        return " ".join(results).strip()

    def extract_number(self, image: Any, roi: tuple[int, int, int, int] | None = None) -> int | None:
        """Extract numeric integer value using EasyOCR."""
        raw_text = self.extract_text(image, roi=roi)
        return self.parse_first_int(raw_text)
