"""Real-time OpenCV Debug Overlay subscriber consuming GameStateUpdated events."""

import os
import time

import cv2
import numpy as np

from dta.core.logger import logger
from dta.events.event_bus import EventBus, get_event_bus
from dta.events.events import GameStateUpdated
from dta.models.game_state import GameState


class DebugOverlay:
    """Subscriber component rendering real-time OpenCV debug overlay from GameStateUpdated events."""

    def __init__(
        self,
        event_bus: EventBus | None = None,
        window_name: str = "DTA - Debug Perception Overlay",
        save_screenshots: bool = False,
        output_dir: str = "debug_captures",
    ) -> None:
        self.event_bus = event_bus or get_event_bus()
        self.window_name = window_name
        self.save_screenshots = save_screenshots
        self.output_dir = output_dir

        self.should_exit: bool = False
        self.latest_annotated_frame: np.ndarray | None = None
        self.latest_game_state: GameState | None = None

        self._is_subscribed: bool = False
        self._last_frame_time: float = time.perf_counter()
        self._current_fps: float = 0.0
        self._frame_count: int = 0

        if self.save_screenshots and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    @property
    def is_subscribed(self) -> bool:
        """Return True if overlay is actively subscribed to GameStateUpdated events."""
        return self._is_subscribed

    def start_listening(self) -> None:
        """Subscribe debug overlay to GameStateUpdated events on EventBus."""
        if not self._is_subscribed:
            self.event_bus.subscribe(GameStateUpdated, self.on_game_state_updated)
            self._is_subscribed = True
            logger.info("DebugOverlay subscribed to GameStateUpdated events.")

    def stop_listening(self) -> None:
        """Unsubscribe debug overlay from GameStateUpdated events on EventBus."""
        if self._is_subscribed:
            self.event_bus.unsubscribe(GameStateUpdated, self.on_game_state_updated)
            self._is_subscribed = False
            logger.info("DebugOverlay unsubscribed from GameStateUpdated events.")

    def on_game_state_updated(self, event: GameStateUpdated) -> None:
        """Event callback triggered when a new GameState snapshot is published."""
        state = event.state
        self.latest_game_state = state

        # Calculate FPS rate
        now = time.perf_counter()
        elapsed = now - self._last_frame_time
        if elapsed > 0:
            self._current_fps = 1.0 / elapsed
        self._last_frame_time = now
        self._frame_count += 1

        # Draw visual overlay on frame image if available
        if state.frame_image is not None:
            annotated = self.draw_overlay(state)
            self.latest_annotated_frame = annotated

            if self.save_screenshots and self._frame_count % 30 == 0:
                self.save_debug_screenshot(annotated, state.frame_id)

    def draw_overlay(self, state: GameState) -> np.ndarray:
        """Render debug annotations (boxes, IDs, PA/PM/HP HUD) on a copy of the frame image."""
        if state.frame_image is None:
            # Fallback canvas if no frame image was attached
            canvas = np.zeros((600, 800, 3), dtype=np.uint8)
        else:
            canvas = state.frame_image.copy()

        h, w = canvas.shape[:2]
        perception = state.perception_state

        # 1. Render Top HUD Bar
        hud_height = 45
        cv2.rectangle(canvas, (0, 0), (w, hud_height), (30, 30, 30), -1)
        cv2.line(canvas, (0, hud_height), (w, hud_height), (0, 255, 255), 1)

        # Resources info
        pa_val = perception.resources.pa if perception else 0
        pm_val = perception.resources.pm if perception else 0
        hp_val = perception.resources.hp if perception else 0

        # Construct HUD text items
        hud_left = f"FRAME: {state.frame_id} | FPS: {self._current_fps:.1f}"
        entity_cnt = len(perception.tracked_entities) if perception else 0
        hud_right = f"ENTITIES: {entity_cnt}"


        # Draw left HUD text (White)
        cv2.putText(canvas, hud_left, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw center resources text (Cyan PA, Green PM, Red HP)
        cv2.putText(canvas, f"PA: {pa_val}", (w // 2 - 140, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"PM: {pm_val}", (w // 2 - 40, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"HP: {hp_val}", (w // 2 + 50, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        # Draw right HUD text (Yellow)
        cv2.putText(canvas, hud_right, (w - 150, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

        # 2. Render Tracked Entities
        if perception and perception.tracked_entities:
            for entity in perception.tracked_entities:
                bbox = entity.bbox
                x, y, bw, bh = bbox.x, bbox.y, bbox.w, bbox.h

                # Bounding Box
                color = (0, 255, 0)  # Bright green bounding box
                cv2.rectangle(canvas, (x, y), (x + bw, y + bh), color, 2)

                # Centroid marker
                cx, cy = entity.centroid
                cv2.circle(canvas, (cx, cy), 4, (0, 0, 255), -1)

                # Label text: ID + tracking confidence
                label = f"{entity.entity_id} ({entity.tracking_confidence * 100:.0f}%)"
                (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

                # Solid background box for label
                label_y = max(y - 6, label_h + 10)
                cv2.rectangle(
                    canvas,
                    (x, label_y - label_h - 4),
                    (x + label_w + 6, label_y + baseline),
                    (20, 20, 20),
                    -1,
                )
                # Label text
                cv2.putText(
                    canvas,
                    label,
                    (x + 3, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # 3. Render bottom status bar instructions
        footer_text = "Press 'Q' to Exit | 'S' to Save Screenshot"
        cv2.putText(canvas, footer_text, (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        return canvas

    def show_window_frame(self, frame: np.ndarray) -> bool:
        """Display an annotated frame in an OpenCV window and handle keyboard input.

        Returns True if exit was requested ('q' key or window closed).
        """
        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q")):
            logger.info("Exit key 'Q' pressed in Debug Overlay window.")
            self.should_exit = True
            return True

        if key in (ord("s"), ord("S")):
            frame_id = self.latest_game_state.frame_id if self.latest_game_state else "manual"
            self.save_debug_screenshot(frame, frame_id)

        # Check if window was closed via GUI 'X' button
        try:
            visible = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)
            if visible < 1:
                logger.info("Debug Overlay window closed by user.")
                self.should_exit = True
                return True
        except cv2.error:
            # Handle window destroy error gracefully
            pass

        return False

    def save_debug_screenshot(self, frame: np.ndarray, frame_id: str) -> str:
        """Save an annotated debug screenshot to disk."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        filename = f"debug_{frame_id}_{int(time.time() * 1000)}.png"
        filepath = os.path.join(self.output_dir, filename)
        cv2.imwrite(filepath, frame)
        logger.info(f"Saved debug overlay screenshot: {filepath}")
        return filepath

    def close(self) -> None:
        """Clean up OpenCV windows and unsubscribe from EventBus."""
        self.stop_listening()
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass
        logger.info("DebugOverlay closed cleanly.")
