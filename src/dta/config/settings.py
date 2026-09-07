"""Settings management module using Pydantic Settings and YAML configuration."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ROICoords(BaseModel):
    """Region of interest coordinates (pixel bounds)."""

    x_min: int = 0
    y_min: int = 0
    x_max: int = 100
    y_max: int = 100


class ROISettings(BaseModel):
    """Container for HUD ROI coordinates."""

    pa: ROICoords = Field(default_factory=lambda: ROICoords(x_min=800, y_min=980, x_max=860, y_max=1040))
    pm: ROICoords = Field(default_factory=lambda: ROICoords(x_min=870, y_min=980, x_max=930, y_max=1040))
    hp: ROICoords = Field(default_factory=lambda: ROICoords(x_min=940, y_min=980, x_max=1020, y_max=1040))


class TrackingSettings(BaseModel):
    """Entity tracker configuration."""

    max_distance_threshold: float = 60.0
    max_disappeared_frames: int = 10


class DatasetSettings(BaseModel):
    """Dataset capture and deduplication configuration."""

    enabled: bool = True
    save_interval_seconds: float = 2.0
    max_images_per_session: int = 5000
    min_frame_difference: float = 0.05
    dataset_dir: str = "dataset"
    save_raw: bool = True


class Settings(BaseSettings):
    """Main application settings loaded from config.yaml or defaults."""

    window_name: str = "Dofus"
    capture_fps: int = 15
    ocr_provider: str = "easyocr"
    ocr_language: str = "fr"
    ocr_update_interval: float = 0.25
    debug_mode: bool = True

    supported_resolutions: list[str] = Field(default_factory=lambda: ["1920x1080", "1366x768"])
    ui_scale: str = "auto"
    rois: ROISettings = Field(default_factory=ROISettings)
    tracking: TrackingSettings = Field(default_factory=TrackingSettings)
    dataset: DatasetSettings = Field(default_factory=DatasetSettings)

    @classmethod
    def load_from_yaml(cls, config_path: str | Path = "config.yaml") -> "Settings":
        """Load settings from a YAML file if present, falling back to defaults."""
        path = Path(config_path)
        if path.is_file():
            with open(path, encoding="utf-8") as f:
                data: dict[str, Any] = yaml.safe_load(f) or {}
            return cls(**data)
        return cls()


@lru_cache(maxsize=1)
def get_settings(config_path: str = "config.yaml") -> Settings:
    """Return a cached singleton instance of Settings."""
    return Settings.load_from_yaml(config_path)
