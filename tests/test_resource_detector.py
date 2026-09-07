"""Unit tests for Stage 5 ResourceDetector using mocked OCRProvider."""

from unittest.mock import MagicMock

import numpy as np

from dta.detectors.resource_detector import ResourceDetector
from dta.models.detections import ResourceDetection
from dta.services.ocr.provider import OCRProvider


def test_resource_detector_individual_methods() -> None:
    mock_ocr = MagicMock(spec=OCRProvider)
    mock_ocr.extract_number.side_effect = [12, 6, 950]  # PA=12, PM=6, HP=950

    detector = ResourceDetector(ocr_provider=mock_ocr)
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    pa = detector.detect_pa(dummy_frame)
    pm = detector.detect_pm(dummy_frame)
    hp = detector.detect_hp(dummy_frame)

    assert pa == 12
    assert pm == 6
    assert hp == 950
    assert mock_ocr.extract_number.call_count == 3


def test_resource_detector_detect_resources_batch() -> None:
    mock_ocr = MagicMock(spec=OCRProvider)
    mock_ocr.extract_number.side_effect = [11, 5, 800]

    detector = ResourceDetector(ocr_provider=mock_ocr)
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    resources = detector.detect_resources(dummy_frame)

    assert isinstance(resources, ResourceDetection)
    assert resources.pa == 11
    assert resources.pm == 5
    assert resources.hp == 800
    assert resources.confidence == 1.0


def test_resource_detector_empty_frame() -> None:
    detector = ResourceDetector()
    empty_frame = np.array([])
    res = detector.detect_resources(empty_frame)

    assert res.pa is None
    assert res.pm is None
    assert res.hp is None
    assert res.confidence == 0.0
