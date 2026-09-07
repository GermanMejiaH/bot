"""Command Line Interface (CLI) application for Dofus Tactical Assistant (DTA)."""

import time

import cv2
import numpy as np
import typer

from dta.config.settings import get_settings
from dta.core.logger import logger
from dta.events.event_bus import get_event_bus
from dta.services.frame_pipeline import FramePipeline
from dta.services.screen_capture import ScreenCaptureService
from dta.vision.window_finder import WindowFinder
from dta.visualization.debug_overlay import DebugOverlay

app = typer.Typer(
    name="dta",
    help="Dofus Tactical Assistant (DTA) — Passive real-time computer vision & perception CLI",
    add_completion=False,
)


@app.command()
def debug(
    fps: int = typer.Option(15, "--fps", "-f", help="Target capture frames per second"),
    window_title: str | None = typer.Option(None, "--window-title", "-w", help="Dofus target window title override"),
    ocr_interval: float = typer.Option(0.25, "--ocr-interval", "-o", help="OCR update interval in seconds"),
    save_screenshots: bool = typer.Option(False, "--save-screenshots", "-s", help="Enable saving debug screenshots"),
) -> None:
    """Run full perception pipeline on live Dofus window with interactive OpenCV debug overlay."""
    settings = get_settings()
    if window_title:
        settings.window_name = window_title
    settings.capture_fps = fps
    settings.ocr_update_interval = ocr_interval

    if not WindowFinder.is_window_available(settings.window_name):
        typer.secho(
            f"Error: Window matching '{settings.window_name}' not found.",
            fg=typer.colors.RED,
            err=True,
        )
        typer.echo("Make sure Dofus is running, or specify a custom title with --window-title.")
        raise typer.Exit(code=1)

    bounds = WindowFinder.get_window_bounds(settings.window_name)
    typer.secho(
        f"Target Dofus window bound: {bounds} at target {fps} FPS.",
        fg=typer.colors.GREEN,
    )

    event_bus = get_event_bus()

    # Initialize perception services
    pipeline = FramePipeline(settings=settings, event_bus=event_bus)
    capture_service = ScreenCaptureService(settings=settings, event_bus=event_bus)
    overlay = DebugOverlay(event_bus=event_bus, save_screenshots=save_screenshots)

    typer.echo("Starting perception pipeline... Press 'Q' in the overlay window to stop.")

    pipeline.start_listening()
    overlay.start_listening()
    capture_service.start()

    try:
        while capture_service.is_running and not overlay.should_exit:
            if overlay.latest_annotated_frame is not None:
                should_stop = overlay.show_window_frame(overlay.latest_annotated_frame)
                if should_stop:
                    break
            time.sleep(0.01)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt detected. Shutting down debug session...")
    finally:
        overlay.close()
        pipeline.stop_listening()
        capture_service.stop()
        typer.secho("Debug perception session stopped cleanly.", fg=typer.colors.GREEN)


@app.command()
def screenshot(
    output_path: str = typer.Option("captured_frame.png", "--output", "-o", help="File path to save screenshot"),
    window_title: str | None = typer.Option(None, "--window-title", "-w", help="Dofus window title override"),
) -> None:
    """Capture a single frame from the Dofus window and save it to disk."""
    settings = get_settings()
    if window_title:
        settings.window_name = window_title

    if not WindowFinder.is_window_available(settings.window_name):
        typer.secho(f"Error: Window matching '{settings.window_name}' not found.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # Grab frame snapshot directly using window bounds
    bounds = WindowFinder.get_window_bounds(settings.window_name)
    if bounds is None:
        typer.secho("Failed to locate window bounds.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        import mss

        with mss.mss() as sct:
            left, top, width, height = bounds
            monitor = {"left": left, "top": top, "width": width, "height": height}
            sct_img = sct.grab(monitor)
            frame_bgr = np.array(sct_img, dtype=np.uint8)[:, :, :3]
            cv2.imwrite(output_path, frame_bgr)
            typer.secho(f"Screenshot successfully saved to: {output_path}", fg=typer.colors.GREEN)
    except Exception as exc:
        typer.secho(f"Failed to capture screenshot: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc




@app.command()
def list_windows(
    query: str = typer.Option("Dofus", "--query", "-q", help="Title search substring (leave empty to list all active windows)"),
) -> None:
    """Search and list open desktop windows matching query substring."""
    from typing import Any

    import pygetwindow as gw

    get_all_fn: Any = getattr(gw, "getAllWindows", lambda: [])

    if query:
        matching = gw.getWindowsWithTitle(query)
        if not matching:
            # Fallback to searching all active windows
            all_wins_fallback: list[Any] = get_all_fn()
            matching = [
                w
                for w in all_wins_fallback
                if getattr(w, "visible", True) and query.lower() in str(getattr(w, "title", "")).lower()
            ]
    else:
        all_wins: list[Any] = get_all_fn()
        matching = [w for w in all_wins if getattr(w, "visible", True) and str(getattr(w, "title", "")).strip()]

    if not matching:
        typer.echo(f"No active windows found matching query '{query}'.")
        return

    typer.secho(f"Found {len(matching)} open window(s) matching '{query}':", fg=typer.colors.CYAN)
    for win in matching:
        typer.echo(f" - Title: '{win.title}' | Bounds: (x={win.left}, y={win.top}, w={win.width}, h={win.height})")


@app.command()
def evaluate(
    frames_count: int = typer.Option(30, "--frames", "-n", help="Number of frames to capture/evaluate"),
    window_title: str | None = typer.Option(None, "--window-title", "-w", help="Dofus window title override"),
    output_dir: str = typer.Option("debug_captures", "--output-dir", "-o", help="Output directory for metrics and heatmap"),
) -> None:
    """Run perception evaluation batch, print metrics summary, and generate detection heatmap."""
    from dta.detectors.character_detector import CharacterDetector
    from dta.detectors.detector_evaluator import DetectorEvaluator
    from dta.vision.map_roi_extractor import MapROIExtractor

    settings = get_settings()
    if window_title:
        settings.window_name = window_title

    evaluator = DetectorEvaluator(output_dir=output_dir)
    roi_extractor = MapROIExtractor(settings=settings, output_dir=f"{output_dir}/roi")
    detector = CharacterDetector()

    if WindowFinder.is_window_available(settings.window_name):
        bounds = WindowFinder.get_window_bounds(settings.window_name)
        typer.secho(f"Running evaluation against live window '{settings.window_name}' ({bounds})...", fg=typer.colors.GREEN)
        if bounds is None:
            typer.secho("Failed to locate window bounds.", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)

        import mss
        with mss.mss() as sct:
            left, top, width, height = bounds
            monitor = {"left": left, "top": top, "width": width, "height": height}
            for i in range(frames_count):
                sct_img = sct.grab(monitor)
                frame_bgr = np.array(sct_img, dtype=np.uint8)[:, :, :3]
                roi_frame = roi_extractor.apply_roi_mask(frame_bgr)
                detections = detector.detect_characters(roi_frame)
                evaluator.evaluate_frame(f"eval_frame_{i+1:04d}", detections, frame_shape=frame_bgr.shape)
                time.sleep(0.05)
    else:
        typer.secho(f"Window '{settings.window_name}' not active. Generating synthetic evaluation batch for testing...", fg=typer.colors.YELLOW)
        for i in range(frames_count):
            dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            roi_frame = roi_extractor.apply_roi_mask(dummy_frame)
            detections = detector.detect_characters(roi_frame)
            evaluator.evaluate_frame(f"synth_frame_{i+1:04d}", detections, frame_shape=dummy_frame.shape)

    summary = evaluator.compute_summary()
    heatmap_path = evaluator.generate_heatmap_image("detection_heatmap.png")

    typer.secho("\n=== DTA Perception Evaluation Summary ===", fg=typer.colors.CYAN, bold=True)
    typer.echo(f"Evaluated Frames:    {summary.total_frames_evaluated}")
    typer.echo(f"Min Detections:      {summary.min_detections}")
    typer.echo(f"Max Detections:      {summary.max_detections}")
    typer.echo(f"Mean Detections:     {summary.mean_detections:.2f}")
    typer.echo(f"Median Detections:   {summary.median_detections:.2f}")
    typer.echo(f"Overall Avg BBox:    W={summary.overall_avg_width:.1f}px, H={summary.overall_avg_height:.1f}px, Area={summary.overall_avg_area:.1f}px²")
    if heatmap_path:
        typer.secho(f"Detection Heatmap:   {heatmap_path}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()

