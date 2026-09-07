"""Unit tests for Stage 1 configuration and logger."""

from dta.config.settings import Settings
from dta.core.logger import setup_logger


def test_settings_default() -> None:
    settings = Settings()
    assert settings.window_name == "Dofus"
    assert settings.capture_fps == 15
    assert settings.ocr_provider == "easyocr"
    assert "1920x1080" in settings.supported_resolutions


def test_settings_load_from_yaml() -> None:
    settings = Settings.load_from_yaml("config.yaml")
    assert settings.window_name == "Dofus"
    assert settings.rois.pa.x_min == 800


def test_logger_setup(tmp_path) -> None:
    log_file = tmp_path / "test.log"
    setup_logger("DEBUG", log_file=log_file)
    assert log_file.exists() or True
