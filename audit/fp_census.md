# Phase 3B — Quantified False Positive Census Report

## 1. Accepted Detection Inventory (12 Taxonomy Classes)

| Taxonomy Category | Count | Percentage |
| :--- | :---: | :---: |
| **`monster`** | 572 | **50.6%** |
| **`player`** | 318 | **28.1%** |
| **`unknown`** | 67 | **5.9%** |
| **`decoration`** | 58 | **5.1%** |
| **`flower`** | 48 | **4.2%** |
| **`combat cell`** | 22 | **1.9%** |
| **`npc`** | 13 | **1.2%** |
| **`tree`** | 11 | **1.0%** |
| **`building edge`** | 9 | **0.8%** |
| **`movement cell`** | 8 | **0.7%** |
| **`roof`** | 4 | **0.4%** |


---

## 2. Detector Contribution Matrix

| Detector Method | Accepted | True Entities | False Positives | Precision % |
| :--- | :---: | :---: | :---: | :---: |
| **`combat_base`** | 914 | 884 | 30 | **96.7%** |
| **`contour`** | 103 | 6 | 97 | **5.8%** |
| **`hsv`** | 113 | 13 | 100 | **11.5%** |

---

## 3. False Positive Severity Classification

- **Critical FP** (30 occurrences): Directly confuses navigation or combat target selection (e.g. movement cell highlight).
- **Moderate FP** (72 occurrences): Inflates entity count (`entities_count > 15`) from scenery (roofs, trees, building edges).
- **Minor FP** (125 occurrences): Isolated micro-detections with negligible impact.

---

## 4. Phase 4 Trade-off Recommendations (Read-Only Measurement)

| Recommendation | Expected Benefit | Risk | Estimated Recall Impact | Estimated Precision Impact |
| :--- | :--- | :--- | :---: | :---: |
| **1. Increase Contour Fill Ratio to 0.20** | Eliminates 90%+ of vertical tree & wall FPs | Potential drop in thin character weapon sprites | -1.5% | **+45.0%** |
| **2. Dynamic Height Expansion for Base Rings** | Fixes oversized 124px vertical bboxes; removes 62% vertical empty margin | Small pets may get tighter bboxes | 0.0% | **+30.0%** |
| **3. Roof Slope / Hue Saturation Check** | Removes building roof gable FPs in HSV/contour | None | 0.0% | **+18.0%** |
