"""MapROIExtractor preprocessing component masking out non-playable UI regions from game frames."""

import os

import numpy as np

from dta.config.settings import Settings, get_settings
from dta.core.logger import logger


class MapROIExtractor:
    """Preprocesses captured game frames by masking static UI regions (chat, minimap, quest log, spell bar)."""

    def __init__(
        self,
        settings: Settings | None = None,
        output_dir: str = "debug_captures/roi",
    ) -> None:
        self.settings = settings or get_settings()
        self.output_dir = output_dir

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    def get_roi_mask(self, shape: tuple[int, ...]) -> np.ndarray:
        """Return a binary 8-bit mask (255 for playable map ROI, 0 for excluded UI regions)."""
        h, w = shape[:2]
        mask = np.full((h, w), 255, dtype=np.uint8)

        # 1. Top HUD & Scenery (top 16% height)
        top_h = int(h * 0.16)
        mask[0:top_h, :] = 0

        # 2. Left Quest Tracker (left 15% width, top 65% height)
        left_w = int(w * 0.15)
        left_h = int(h * 0.65)
        mask[0:left_h, 0:left_w] = 0

        # 3. Bottom-left Chat Panel (left 32% width, bottom 30% height)
        chat_w = int(w * 0.32)
        chat_top = int(h * 0.70)
        mask[chat_top:h, 0:chat_w] = 0

        # 4. Bottom Spell & Action Bar (bottom 15% height)
        bottom_top = int(h * 0.85)
        mask[bottom_top:h, :] = 0

        # 5. Bottom-Right HUD area
        hud_left = int(w * 0.75)
        hud_top = int(h * 0.65)
        mask[hud_top:h, hud_left:w] = 0

        # 6. Extreme right border UI widgets
        right_edge = int(w * 0.96)
        mask[:, right_edge:w] = 0

        return mask

    def apply_roi_mask(self, frame: np.ndarray) -> np.ndarray:
        """Return a copy of frame with static UI regions zeroed out (blacked out).

        Masked Regions:
        - Top banner & Minimap (top 8% height)
        - Left Quest Tracker (left 15% width, top 65% height)
        - Bottom-left Chat Panel (left 32% width, bottom 30% height)
        - Bottom Spell & Action Bar (bottom 15% height across full width)
        """
        if frame is None or frame.size == 0:
            return frame

        masked_frame = frame.copy()
        mask = self.get_roi_mask(frame.shape)
        masked_frame[mask == 0] = 0

        return masked_frame

    def save_roi_frame(self, roi_frame: np.ndarray, frame_id: str) -> str:
        """Save masked ROI frame to disk for visual verification."""
        import cv2

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        filename = f"roi_{frame_id}.png"
        filepath = os.path.join(self.output_dir, filename)
        cv2.imwrite(filepath, roi_frame)
        logger.debug(f"Saved ROI frame to {filepath}")
        return filepath
