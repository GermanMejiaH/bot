"""CLI audit script executing 50-frame TRACE & AUDIT MODE session for CharacterDetector and perception pipeline."""

import argparse
import os
import sys
import time

import cv2
import numpy as np

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from dta.config.settings import get_settings
from dta.events.event_bus import get_event_bus
from dta.models.frame import CapturedFrame, FrameMetadata
from dta.services.dataset_capture import DatasetCaptureService
from dta.services.frame_pipeline import FramePipeline
from dta.vision.window_finder import WindowFinder


def run_audit(
    frames_count: int = 50,
    window_title: str | None = None,
    output_dir: str = "audit",
) -> dict:
    """Execute TRACE & AUDIT MODE across frames_count iterations and return audit summary dictionary."""
    settings = get_settings()
    if window_title:
        settings.window_name = window_title

    settings.audit.enabled = True
    settings.audit.max_frames = frames_count
    settings.audit.output_dir = output_dir

    event_bus = get_event_bus()

    pipeline = FramePipeline(settings=settings, event_bus=event_bus)
    dataset_service = DatasetCaptureService(settings=settings, event_bus=event_bus, dataset_dir="dataset")

    pipeline.start_listening()
    dataset_service.start_listening()

    is_live = WindowFinder.is_window_available(settings.window_name)

    print("\n=======================================================")
    print("  DTA CharacterDetector AUDIT MODE (TRACE PHASE 1)")
    print("=======================================================")
    print(f"Target Window:     {settings.window_name} (Live: {is_live})")
    print(f"Audit Max Frames:  {frames_count}")
    print(f"Audit Output Dir:  {output_dir}")
    print("-------------------------------------------------------\n")

    if is_live:
        bounds = WindowFinder.get_window_bounds(settings.window_name)
        if bounds is None:
            print("Error: Target window bounds could not be determined.")
            return {}

        import mss

        with mss.mss() as sct:
            left, top, width, height = bounds
            monitor = {"left": left, "top": top, "width": width, "height": height}

            for i in range(frames_count):
                sct_img = sct.grab(monitor)
                frame_bgr = np.array(sct_img, dtype=np.uint8)[:, :, :3]
                meta = FrameMetadata(
                    frame_id=f"audit_{i+1:04d}",
                    timestamp=time.time(),
                    width=width,
                    height=height,
                )
                captured = CapturedFrame(metadata=meta, image=frame_bgr)
                pipeline.process_frame(captured)
                time.sleep(0.03)
    else:
        print("Note: Target Dofus window not active. Executing audit pipeline using synthetic frame batch...")
        for i in range(frames_count):
            # Create synthetic test frame simulating exploration screen with map edges & HUD
            synth_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            # Add subtle scenery textures
            cv2.rectangle(synth_frame, (100, 150), (250, 450), (45, 80, 120), -1)
            cv2.rectangle(synth_frame, (300, 200), (340, 400), (80, 120, 40), -1)
            cv2.circle(synth_frame, (600, 350), 30, (200, 200, 200), -1)

            meta = FrameMetadata(
                frame_id=f"audit_synth_{i+1:04d}",
                timestamp=time.time(),
                width=1280,
                height=720,
            )
            captured = CapturedFrame(metadata=meta, image=synth_frame)
            pipeline.process_frame(captured)
            time.sleep(0.01)

    dataset_service.stop_listening()
    pipeline.stop_listening()

    json_report_path = os.path.join(output_dir, "detections.json")
    print("\nAUDIT Execution Complete!")
    print(f"  - Original Frames:   {os.path.join(output_dir, 'original')}")
    print(f"  - Overlay Images:    {os.path.join(output_dir, 'overlay')}")
    print(f"  - Stage Images:      {os.path.join(output_dir, 'stages')}")
    print(f"  - Candidate Crops:   {os.path.join(output_dir, 'crops')}")
    print(f"  - Excluded ROI:      {os.path.join(output_dir, 'excluded_regions')}")
    print(f"  - Audit JSON Trace:  {json_report_path}\n")

    return {"json_path": json_report_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="DTA CharacterDetector AUDIT Mode Runner (Phase 1 Trace)")
    parser.add_argument("--frames", "-n", type=int, default=50, help="Number of frames to audit (default: 50)")
    parser.add_argument("--window-title", "-w", type=str, default=None, help="Target Dofus window title override")
    parser.add_argument("--output-dir", "-o", type=str, default="audit", help="Audit output directory (default: audit)")

    args = parser.parse_args()
    run_audit(frames_count=args.frames, window_title=args.window_title, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
