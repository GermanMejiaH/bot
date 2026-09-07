"""Script to generate annotated visual validation screenshots and report data for DTA Stage 8.5."""

from pathlib import Path

import cv2

from dta.detectors.character_detector import CharacterDetector
from dta.vision.map_roi_extractor import MapROIExtractor


def generate_validation_report() -> None:
    roi_extractor = MapROIExtractor()
    detector = CharacterDetector()

    raw_dir = Path("debug_captures/raw")
    val_dir = Path("debug_captures/validation")
    val_dir.mkdir(parents=True, exist_ok=True)

    # Scenarios mapping to frame IDs
    scenarios = {
        "Empty Map": "raw_frame_000091.png",
        "Crowded Astrub Map": "raw_frame_000151.png",
        "Combat Scene": "raw_frame_000271.png",
    }

    report_data = {}

    for scenario_name, fname in scenarios.items():
        img_path = raw_dir / fname
        if not img_path.exists():
            print(f"Warning: {img_path} does not exist.")
            continue

        img = cv2.imread(str(img_path))
        masked = roi_extractor.apply_roi_mask(img)
        detections = detector.detect_characters(masked)

        # Generate annotated frame
        annotated = img.copy()
        for det in detections:
            bbox = det.bbox
            # Draw green bounding box around detected character entity
            cv2.rectangle(
                annotated,
                (bbox.x, bbox.y),
                (bbox.x + bbox.w, bbox.y + bbox.h),
                (0, 255, 0),
                2,
            )
            # Draw red centroid marker
            cv2.circle(annotated, det.centroid, 4, (0, 0, 255), -1)
            # Label method, dimensions, confidence
            label = f"{det.method} {bbox.w}x{bbox.h} c:{det.confidence:.2f}"
            cv2.putText(
                annotated,
                label,
                (bbox.x, max(15, bbox.y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
            )

        output_filename = f"validation_{scenario_name.lower().replace(' ', '_')}.png"
        output_path = val_dir / output_filename
        cv2.imwrite(str(output_path), annotated)

        report_data[scenario_name] = {
            "frame_id": fname.replace(".png", ""),
            "output_path": str(output_path),
            "detections_count": len(detections),
            "detections": detections,
        }

        print(f"Scenario '{scenario_name}' ({fname}): {len(detections)} detections -> saved {output_path}")
        for idx, d in enumerate(detections, 1):
            print(
                f"   Detection #{idx}: bbox=(x={d.bbox.x}, y={d.bbox.y}, w={d.bbox.w}, h={d.bbox.h}), "
                f"dim={d.bbox.w}x{d.bbox.h}, area={d.area:.1f}, conf={d.confidence:.2f}, method={d.method}"
            )

    return report_data


if __name__ == "__main__":
    generate_validation_report()
