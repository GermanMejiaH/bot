# Phase 3C — Human-Labeled Performance Analysis Report

**Ground Truth Source**: `audit/manual_labels.xlsx`

## Executive Summary

- **Total Accepted Samples Evaluated**: `152`
- **True Positives (Valid Entities)**: `40` (26.32%)
- **False Positives (Non-Entities)**: `112` (73.68%)
- **Global Precision**: **`26.32%`**

---

## 1. Precision by Method

| Method | Total Candidates | True Positives (Entities) | False Positives | Precision (%) | False Positive Rate (%) |
| --- | --- | --- | --- | --- | --- |
| `combat_base` | 67 | 16 | 51 | **23.88%** | 76.12% |
| `contour` | 34 | 12 | 22 | **35.29%** | 64.71% |
| `hsv` | 51 | 12 | 39 | **23.53%** | 76.47% |
| **GLOBAL** | **152** | **40** | **112** | **26.32%** | **73.68%** |

---

## 2. Class Distribution (Ground Truth)

| Class Label | Count | Percentage of Total | Entity Type |
| --- | --- | --- | --- |
| `decoration` | 39 | 25.66% | No (False Positive) |
| `unknown` | 38 | 25.0% | No (False Positive) |
| `monster` | 34 | 22.37% | Yes (True Positive) |
| `flower` | 16 | 10.53% | No (False Positive) |
| `movement_cell` | 14 | 9.21% | No (False Positive) |
| `player` | 6 | 3.95% | Yes (True Positive) |
| `box` | 3 | 1.97% | No (False Positive) |
| `bag` | 2 | 1.32% | No (False Positive) |

---

## 3. False Positive Distribution

| False Positive Class | FP Count | % of All False Positives | % of Total Detections |
| --- | --- | --- | --- |
| `decoration` | 39 | 34.82% | 25.66% |
| `unknown` | 38 | 33.93% | 25.0% |
| `flower` | 16 | 14.29% | 10.53% |
| `movement_cell` | 14 | 12.5% | 9.21% |
| `box` | 3 | 2.68% | 1.97% |
| `bag` | 2 | 1.79% | 1.32% |

---

## 4. Unknown Category Analysis

- **Total Unknown Candidates**: `38` (25.0% of total dataset)
- **Share of All False Positives**: `33.93%`
- **Breakdown by Detector Method**:
  - `contour`: `8` unknown instances
  - `hsv`: `15` unknown instances
  - `combat_base`: `15` unknown instances
- **Analysis & Characteristics**:
  - Represent unclassified background terrain fragments, non-standard building features, or UI overlays.
  - Second largest category of false positives after static map decorations.

---

## 5. Method × Class Matrix

| Method | `bag` | `box` | `decoration` | `flower` | `monster` | `movement_cell` | `player` | `unknown` | Total | Precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `combat_base` | 1 | 1 | 15 | 11 | 12 | 8 | 4 | 15 | 67 | **23.88%** |
| `contour` | 1 | 2 | 9 | 0 | 11 | 2 | 1 | 8 | 34 | **35.29%** |
| `hsv` | 0 | 0 | 15 | 5 | 11 | 4 | 1 | 15 | 51 | **23.53%** |
| **Total** | 2 | 3 | 39 | 16 | 34 | 14 | 6 | 38 | **152** | **26.32%** |

---

## 6. Phase 4 Recommendations (Based Exclusively on Human Labels)

1. **Decoration & Unknown Noise Suppression**:
   - **Problem**: `decoration` (39) + `unknown` (38) account for **68.75% of all false positives** (77 / 112).
   - **Action**: Implement texture variance and color uniformity filters to discard static background tiles and decorative elements.

2. **Vibrant Plant / Flower Sprite Filtering**:
   - **Problem**: `flower` candidates (16) are frequently triggered in `combat_base` (11) and `hsv` (5).
   - **Action**: Refine red/yellow HSV saturation and brightness ring constraints to prevent small floral pixels from triggering candidate bounding boxes.

3. **Tactical Movement Grid Cell Exclusion**:
   - **Problem**: `movement_cell` highlights (14) are being accepted as candidate entities.
   - **Action**: Add grid mask detection or bounding box aspect-ratio/edge-regularity rules to suppress map cell highlights.

4. **Contour Pipeline Tuning**:
   - **Observation**: `contour` achieves the highest precision (**35.29%**), but still accepts 22 non-entity crops.
   - **Action**: Tighten minimum area and vertical aspect ratio parameters.

5. **Multi-Channel Candidate Consensus**:
   - **Observation**: `hsv` (23.53%) and `combat_base` (23.88%) have near-identical 76% false positive rates.
   - **Action**: Require candidate consensus (e.g., HSV ring match + contour edge verification) before accepting candidates in candidate merger.
