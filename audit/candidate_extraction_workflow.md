# Candidate Extraction Workflow & Dataset Generation Report

## Executive Summary

This report documents the exact technical workflow used to generate candidate crops in the DTA perception pipeline, identifies the scripts responsible for `audit/crops/`, `audit/review/accepted/`, and `audit/manual_labels.xlsx`, and specifies the exact command procedure required to extract new candidate crops from any screenshot folder (redirected to `audit/stress_candidates/`).

---

## 1. Original Dataset Generation Pipeline

The original 152 manually labeled candidate crops in Phase 4C were generated via a **two-step workflow**:

```
[ Captured / Synthetic Screenshots ]
                │
                ▼
[ scripts/audit_character_detector.py ]  ──(FramePipeline / CharacterDetector)
                │
                ├──► audit/crops/                   (Raw accepted & rejected crops)
                ├──► audit/review/accepted/          (Accepted candidate crops & sidecars)
                └──► audit/detections.json           (Full trace telemetry)
                │
                ▼
[ scripts/generate_labeling_sheet.py ]
                │
                └──► audit/manual_labels.xlsx        (Excel workbook with dropdowns)
```

### Script Breakdown

1. **`scripts/audit_character_detector.py`**:
   - **Role**: Executes perception pipeline in **TRACE & AUDIT MODE**.
   - **Invocation**: `python scripts/audit_character_detector.py --frames 50 --output-dir audit`
   - **Internal Execution**: Instantiates `FramePipeline` and `CharacterDetector`, passing captured game window frames or synthetic frames through `detect_characters_with_trace(roi_frame)`.
   - **Outputs Produced**:
     - `audit/crops/`: Contains `entity_XXXX_accepted.png` and `entity_XXXX_rejected.png`.
     - `audit/review/accepted/`: Contains candidate crop images (`candidate_XXXX_method.png`) and sidecar JSONs (`candidate_XXXX_method.json`).
     - `audit/detections.json`: Full detection trace telemetry.

2. **`scripts/generate_labeling_sheet.py`**:
   - **Role**: Builds the manual annotation sheet.
   - **Invocation**: `python scripts/generate_labeling_sheet.py`
   - **Internal Execution**: Scans `audit/review/accepted/*.png`, extracts candidate metadata, creates `audit/manual_labels.xlsx` formatted table with validation dropdowns (`player`, `monster`, `npc`, `flower`, `roof`, etc.).

---

## 2. Perception Detector Entry Point & Architecture

The candidate extraction entry point is **`CharacterDetector`** (`src/dta/detectors/character_detector.py`).

### Component Breakdown

1. **Map ROI Masking**: `MapROIExtractor.apply_roi_mask(frame)` (masks HUD UI elements).
2. **Candidate Generators**:
   - `detect_contours(frame)`: Morphological vertical closing + Canny edge detection.
   - `detect_hsv_regions(frame)`: HSV color segmentation + green tile mask exclusion.
   - `detect_combat_bases(frame)`: Combat floor ring anchor detection (red/blue ring HSV range).
3. **Diagnostic Telemetry**: `_compute_candidate_diagnostics()` extracts 19 geometry, color, texture, and edge features.
4. **NMS Candidate Merging**: `merge_candidates()` / `cv2.dnn.NMSBoxes()` with `iou_threshold = 0.35`.
5. **Trace Pipeline**: `detect_characters_with_trace(roi_frame)` outputs accepted candidates, rejected candidates, lifecycle trace steps, and diagnostic sidecars.

---

## 3. Extraction Command & Script Strategy for `audit/stress_candidates/`

To extract new candidate crops from screenshot folders (e.g. `dataset/exploration/` or `dataset/combat/`) without modifying perception code, run a standalone extraction script that passes images through `CharacterDetector.detect_characters_with_trace()` and saves output to `audit/stress_candidates/`.

### Python Extraction Command Procedure

```python
import glob
import json
import os
import cv2
from dta.config.settings import get_settings
from dta.detectors.character_detector import CharacterDetector
from dta.vision.map_roi_extractor import MapROIExtractor

settings = get_settings()
detector = CharacterDetector()
roi_extractor = MapROIExtractor(settings=settings)

input_dir = "dataset/exploration"  # or dataset/combat
output_dir = os.path.join("audit", "stress_candidates")
os.makedirs(output_dir, exist_ok=True)

frame_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))
candidate_counter = 1

for fpath in frame_files:
    frame_name = os.path.splitext(os.path.basename(fpath))[0]
    frame = cv2.imread(fpath)
    if frame is None:
        continue

    roi_frame = roi_extractor.apply_roi_mask(frame)
    trace = detector.detect_characters_with_trace(roi_frame)
    accepted_dets = trace.get("raw_characters", [])

    for det in accepted_dets:
        bbox = det.bbox
        crop = frame[max(0, bbox.y) : bbox.y + bbox.h, max(0, bbox.x) : bbox.x + bbox.w]
        if crop.size == 0:
            continue

        cid_str = f"{candidate_counter:04d}"
        candidate_counter += 1

        img_name = f"candidate_{cid_str}_{det.method}.png"
        json_name = f"candidate_{cid_str}_{det.method}.json"

        cv2.imwrite(os.path.join(output_dir, img_name), crop)

        sidecar = {
            "candidate_id": candidate_counter - 1,
            "filename": img_name,
            "method": det.method,
            "bbox": [bbox.x, bbox.y, bbox.w, bbox.h],
            "area": det.area,
            "confidence": det.confidence,
            "frame_id": frame_name,
            "diagnostics": det.diagnostics,
        }
        with open(os.path.join(output_dir, json_name), "w", encoding="utf-8") as f:
            json.dump(sidecar, f, indent=2)

print(f"Extracted {candidate_counter - 1} candidate crops to '{output_dir}'.")
```

---

## 4. Estimated Candidate Crop Yield

Based on the average candidate density of Dofus maps ($3-6$ accepted candidates per screenshot):

| Source Folder | Input Screenshots | Estimated Candidate Crops | Primary Scene Type |
| :--- | :---: | :---: | :--- |
| `dataset/exploration/*.png` | 86 | **$\approx 250 – 350$ crops** | Astrub surface, fields, forests, map borders |
| `dataset/combat/*.png` | 99 | **$\approx 300 – 450$ crops** | Combat floor grids, monsters, player characters, ring bases |
| **Combined Pool** | **185** | **$\approx 550 – 800$ crops** | Unseen out-of-sample stress dataset |

- **Yield**: Extracting candidates from `dataset/exploration` and `dataset/combat` will yield **$\approx 550 – 800$ new candidate crops**, providing more than enough data to construct a robust $\ge 150$-sample stress validation set for Phase 4C.5.
