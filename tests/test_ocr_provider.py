"""Unit tests for Stage 3 OCRProvider and EasyOCRProvider using mocks."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np

from dta.services.ocr.easyocr_provider import EasyOCRProvider
from dta.services.ocr.provider import OCRProvider


def test_ocr_provider_crop_roi() -> None:
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    roi = (10, 20, 50, 60)
    cropped = OCRProvider.crop_roi(img, roi)

    assert cropped.shape == (40, 40, 3)


def test_ocr_provider_preprocess_image() -> None:
    img_bgr = np.zeros((20, 40, 3), dtype=np.uint8)
    processed = OCRProvider.preprocess_image(img_bgr, scale_factor=2.0)

    # Upscaled from 20x40 to 40x80 grayscale threshold image
    assert processed.shape == (40, 80)


def test_ocr_provider_parse_first_int() -> None:
    assert OCRProvider.parse_first_int("PA: 12") == 12
    assert OCRProvider.parse_first_int("PM 6 pts") == 6
    assert OCRProvider.parse_first_int("No digits here") is None
    assert OCRProvider.parse_first_int("HP: 1250 / 1250") == 1250


def test_easyocr_provider_lazy_loading_and_mocked_extract() -> None:
    provider = EasyOCRProvider(languages=["fr"])
    assert provider._reader is None  # Reader is lazily initialized

    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)

    mock_easyocr = MagicMock()
    mock_reader_instance = MagicMock()
    mock_reader_instance.readtext.return_value = ["PA", "12"]
    mock_easyocr.Reader.return_value = mock_reader_instance

    with patch.dict(sys.modules, {"easyocr": mock_easyocr}):
        text = provider.extract_text(dummy_image)
        assert text == "PA 12"
        assert provider._reader is not None  # Reader is now initialized and cached

        num = provider.extract_number(dummy_image)
        assert num == 12

