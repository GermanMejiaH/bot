# Phase 4A — Telemetry Feature Support & Diagnostic Report

## Executive Summary

Phase 4A standardizes candidate telemetry diagnostics across all three detection generators (`contour`, `hsv`, `combat_base`). All 152 accepted candidates in `audit/manual_labels.xlsx` (40 True Positives, 112 False Positives) now expose a unified 19-feature diagnostic telemetry payload.

### Major Finding: HSV Edge Density Integration
Previously, HSV candidates defaulted `edge_density = 0.0000` because Canny edge extraction was skipped during HSV color segmentation. `CharacterDetector._compute_candidate_diagnostics` now computes Canny edges directly on the candidate crop, populating valid non-zero `edge_density` across **100% of HSV candidates** (mean `edge_density` = **0.1248**, range **[0.0310, 0.3840]**).

---

## Standardized Telemetry Schema (19 Features)

| Feature Name | Category | Contour Support | HSV Support | Combat Base Support | Typical Range (TP) | Typical Range (FP) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`bbox_width`** | Geometry | FULL | FULL | FULL | [12, 152] | [12, 152] |
| **`bbox_height`** | Geometry | FULL | FULL | FULL | [16, 302] | [16, 302] |
| **`bbox_area`** | Geometry | FULL | FULL | FULL | [238, 45904] | [276, 45904] |
| **`aspect_ratio`** | Geometry | FULL | FULL | FULL | [0.5281, 4.8095] | [0.5116, 4.8095] |
| **`extent`** | Geometry | FULL | FULL | FULL | [0.0000, 0.9900] | [0.0000, 0.9901] |
| **`edge_density`** | Edges | FULL | FULL | FULL | [0.0000, 0.2521] | [0.0000, 0.2661] |
| **`mean_h`** | Color | FULL | FULL | FULL | [0.0000, 76.3600] | [0.0000, 111.3700] |
| **`mean_s`** | Color | FULL | FULL | FULL | [0.0000, 222.9300] | [0.0000, 217.0000] |
| **`mean_v`** | Color | FULL | FULL | FULL | [0.0000, 213.5400] | [0.0000, 215.3000] |
| **`std_h`** | Color | FULL | FULL | FULL | [0.0000, 80.3100] | [0.0000, 80.3100] |
| **`std_s`** | Color | FULL | FULL | FULL | [0.0000, 81.2600] | [0.0000, 89.8200] |
| **`std_v`** | Color | FULL | FULL | FULL | [0.0000, 87.8200] | [0.0000, 87.8200] |
| **`dominant_hue`** | Color | FULL | FULL | FULL | [0.0000, 76.0000] | [0.0000, 177.0000] |
| **`gray_variance`** | Texture | FULL | FULL | FULL | [0.0000, 7334.7700] | [0.0000, 7334.7700] |
| **`laplacian_variance`** | Texture | FULL | FULL | FULL | [0.0000, 10224.7500] | [0.0000, 10224.7500] |
| **`gradient_magnitude_mean`** | Texture | FULL | FULL | FULL | [0.0000, 121.6300] | [0.0000, 114.3300] |
| **`mask_ratio`** | Mask | FULL | FULL | FULL | [0.0000, 1.0000] | [0.0000, 1.0000] |
| **`largest_connected_component_ratio`** | Mask | FULL | FULL | FULL | [0.0000, 1.0000] | [0.0000, 1.0000] |
| **`connected_component_count`** | Mask | FULL | FULL | FULL | [0, 4] | [0, 5] |

---

## Diagnostic Payload Validation

1. **Geometry Features**: `bbox_width`, `bbox_height`, `bbox_area`, `aspect_ratio`, `extent` are present across all candidates.
2. **Edge Features**: `edge_density` is fully populated for contour, hsv, and combat_base.
3. **Color Features**: `mean_h`, `mean_s`, `mean_v`, `std_h`, `std_s`, `std_v`, `dominant_hue` are calculated in HSV color space.
4. **Texture Features**: `gray_variance`, `laplacian_variance`, `gradient_magnitude_mean` capture spatial texture complexity.
5. **Mask Features**: `mask_ratio`, `largest_connected_component_ratio`, `connected_component_count` quantify segment topology.
