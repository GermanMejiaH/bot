"""Central FramePipeline orchestrating perception processing from CapturedFrame to GameState."""

import threading
import time

from pydantic import BaseModel

from dta.config.settings import Settings, get_settings
from dta.core.logger import logger
from dta.detectors.character_detector import CharacterDetector
from dta.detectors.resource_detector import ResourceDetector
from dta.events.event_bus import EventBus, get_event_bus
from dta.events.events import FrameCaptured, GameStateUpdated
from dta.models.detections import DetectionBatch, ResourceDetection
from dta.models.frame import CapturedFrame
from dta.models.game_state import GameState
from dta.state.state_builder import StateBuilder
from dta.tracking.entity_tracker import EntityTracker
from dta.vision.map_roi_extractor import MapROIExtractor


class PipelineMetrics(BaseModel):
    """Performance execution timing metrics for a single pipeline frame iteration."""

    capture_time_ms: float = 0.0
    ocr_time_ms: float = 0.0
    detection_time_ms: float = 0.0
    tracking_time_ms: float = 0.0
    state_build_time_ms: float = 0.0
    total_pipeline_time_ms: float = 0.0


class FramePipeline:
    """Central pipeline orchestrator executing perception steps and publishing GameStateUpdated events."""

    def __init__(
        self,
        resource_detector: ResourceDetector | None = None,
        character_detector: CharacterDetector | None = None,
        entity_tracker: EntityTracker | None = None,
        state_builder: StateBuilder | None = None,
        map_roi_extractor: MapROIExtractor | None = None,
        event_bus: EventBus | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.event_bus = event_bus or get_event_bus()

        self.resource_detector = resource_detector or ResourceDetector(settings=self.settings)
        self.character_detector = character_detector or CharacterDetector()
        self.entity_tracker = entity_tracker or EntityTracker(settings=self.settings)
        self.state_builder = state_builder or StateBuilder()
        self.map_roi_extractor = map_roi_extractor or MapROIExtractor(settings=self.settings)

        self.ocr_update_interval = self.settings.ocr_update_interval
        self._last_ocr_time: float = 0.0
        self._cached_resources: ResourceDetection = ResourceDetection()

        self._latest_metrics: PipelineMetrics | None = None
        self._is_listening: bool = False
        self._lock = threading.RLock()
        self._frame_count: int = 0


    @property
    def is_listening(self) -> bool:
        """Return True if currently subscribed to FrameCaptured events."""
        with self._lock:
            return self._is_listening

    def start_listening(self) -> None:
        """Subscribe pipeline to FrameCaptured events on EventBus."""
        with self._lock:
            if not self._is_listening:
                self.event_bus.subscribe(FrameCaptured, self.on_frame_captured)
                self._is_listening = True
                logger.info("FramePipeline subscribed to FrameCaptured events.")

    def stop_listening(self) -> None:
        """Unsubscribe pipeline from FrameCaptured events on EventBus."""
        with self._lock:
            if self._is_listening:
                self.event_bus.unsubscribe(FrameCaptured, self.on_frame_captured)
                self._is_listening = False
                logger.info("FramePipeline unsubscribed from FrameCaptured events.")

    def get_latest_metrics(self) -> PipelineMetrics | None:
        """Return performance metrics from the most recent processed frame."""
        with self._lock:
            return self._latest_metrics

    def on_frame_captured(self, event: FrameCaptured) -> None:
        """Event handler callback invoked when a new FrameCaptured event is published."""
        captured_frame = event.captured_frame
        if captured_frame is None and event.frame is not None:
            # Fallback wrapper if raw numpy array event was published
            from dta.models.frame import FrameMetadata

            meta = FrameMetadata(
                frame_id=event.frame_id,
                timestamp=event.timestamp,
                width=event.width,
                height=event.height,
            )
            captured_frame = CapturedFrame(metadata=meta, image=event.frame)

        if captured_frame is not None:
            self.process_frame(captured_frame)

    def process_frame(self, captured_frame: CapturedFrame) -> GameState:
        """Execute perception pipeline for a single CapturedFrame instance.

        Pipeline Sequence:
        CapturedFrame -> OCR Throttling -> ResourceDetector -> CharacterDetector -> EntityTracker -> StateBuilder -> GameState
        """
        with self._lock:
            t_total_start = time.perf_counter()
            frame_img = captured_frame.image
            frame_id = captured_frame.frame_id
            now_perf = time.perf_counter()
            now_wall = time.time()

            # Step 1: OCR Throttling check (using monotonic perf_counter)
            t_ocr_start = time.perf_counter()
            if self._last_ocr_time == 0.0 or (now_perf - self._last_ocr_time) >= self.ocr_update_interval:
                try:
                    resources = self.resource_detector.detect_resources(frame_img)
                    self._cached_resources = resources
                    self._last_ocr_time = now_perf
                except Exception as exc:
                    logger.error(f"Resource detection failed during frame processing: {exc}")
                    resources = self._cached_resources
            else:
                resources = self._cached_resources
            t_ocr_end = time.perf_counter()

            # Step 2: Map ROI Extraction & Character Detection
            t_det_start = time.perf_counter()
            self._frame_count += 1
            roi_frame = self.map_roi_extractor.apply_roi_mask(frame_img)
            if self._frame_count % 30 == 1:
                self.map_roi_extractor.save_roi_frame(roi_frame, frame_id)

            if hasattr(self.character_detector, "detect_characters_with_trace"):
                trace_res = self.character_detector.detect_characters_with_trace(roi_frame)
                if isinstance(trace_res, dict):
                    raw_characters = trace_res.get("raw_characters", [])
                    rejected_characters = trace_res.get("rejected_characters", [])
                    all_candidates = trace_res.get("all_candidates", [])
                    stage_masks = trace_res.get("stage_masks", {})
                    statistics = trace_res.get("statistics", {})
                else:
                    raw_characters = self.character_detector.detect_characters(roi_frame)
                    rejected_characters = []
                    all_candidates = raw_characters
                    stage_masks = {}
                    statistics = {}
            else:
                raw_characters = self.character_detector.detect_characters(roi_frame)
                rejected_characters = []
                all_candidates = raw_characters
                stage_masks = {}
                statistics = {}
            t_det_end = time.perf_counter()

            # Step 3: Entity Tracking
            t_track_start = time.perf_counter()
            tracked_entities = self.entity_tracker.update(raw_characters)
            t_track_end = time.perf_counter()

            # Step 4: Detection Batch & State Building
            t_state_start = time.perf_counter()
            batch = DetectionBatch(
                frame_id=frame_id,
                timestamp=now_wall,
                raw_characters=raw_characters,
                rejected_characters=rejected_characters,
                all_candidates=all_candidates,
                resources=resources,
                debug_masks=stage_masks,
                statistics=statistics,
            )

            # Auto-save validation overlay if configured
            if self.settings.debug.save_detection_overlays:
                try:
                    from dta.debug.detection_visualizer import save_detection_overlay

                    val_path = f"{self.settings.debug.output_dir}/overlay_{frame_id}.png"
                    save_detection_overlay(frame_img, raw_characters, output_path=val_path)
                except Exception as val_exc:
                    logger.debug(f"Failed saving validation overlay: {val_exc}")

            game_state = self.state_builder.build_game_state(
                frame_id=frame_id,
                resources=resources,
                tracked_entities=tracked_entities,
                raw_detections=batch,
                timestamp=now_wall,
                frame_image=frame_img,
            )

            t_state_end = time.perf_counter()

            t_total_end = time.perf_counter()

            # Metrics calculation
            metrics = PipelineMetrics(
                ocr_time_ms=(t_ocr_end - t_ocr_start) * 1000.0,
                detection_time_ms=(t_det_end - t_det_start) * 1000.0,
                tracking_time_ms=(t_track_end - t_track_start) * 1000.0,
                state_build_time_ms=(t_state_end - t_state_start) * 1000.0,
                total_pipeline_time_ms=(t_total_end - t_total_start) * 1000.0,
            )
            self._latest_metrics = metrics

            logger.debug(
                f"Processed {frame_id} in {metrics.total_pipeline_time_ms:.2f}ms "
                f"(OCR={metrics.ocr_time_ms:.2f}ms, Det={metrics.detection_time_ms:.2f}ms, Track={metrics.tracking_time_ms:.2f}ms)."
            )

        # Step 5: Publish GameStateUpdated event outside lock to avoid potential deadlocks in subscribers
        self.event_bus.publish(GameStateUpdated(state=game_state))
        return game_state

