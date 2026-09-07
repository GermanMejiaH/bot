"""OCR Services package initialization."""

from dta.services.ocr.easyocr_provider import EasyOCRProvider
from dta.services.ocr.provider import OCRProvider

__all__ = ["OCRProvider", "EasyOCRProvider"]
