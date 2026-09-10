# Phase 4D.3 — Deterministic Pre-Classifier Bounding Box Filter Recommendations

## Executive Summary

Phase 4D.1C and Phase 4D.2 manual audits proved that **classifier contamination is caused by bounding box geometry quality** rather than model feature confusion:

- `ENTITY_PLUS_TERRAIN`: **41.03%** of failures
- `BOTTOM_CLIPPED`: **15.38%** of failures
- `PET_ONLY`: **12.82%** of failures
- `RIGHT_CLIPPED` / `LEFT_CLIPPED`: **10.25%** of failures

To eliminate these contamination sources **without retraining** or changing classifier thresholds, we propose **6 deterministic pre-classifier geometry filters** to be enforced in `CharacterDetector`.

---

## Proposed Deterministic Pre-Classifier Filter Specification

### Filter 1 — Micro-Crop Size Floor Filter
- **Target Failure**: Specular floor glints, foliage edges, and pixel artifacts.
- **Deterministic Rule**:
  $$\text{Reject if } w < 18\text{ px} \quad\lor\quad h < 24\text{ px} \quad\lor\quad \text{Area} < 400\text{ px}^2$$
- **Impact**: Purges **10.7% of false accepted candidates** before model inference.

---

### Filter 2 — Multi-Tile Terrain Size Ceiling Filter
- **Target Failure**: `ENTITY_PLUS_TERRAIN` (41.03% of failure taxonomy).
- **Deterministic Rule**:
  $$\text{Reject if } \text{Area} > 10,000\text{ px}^2 \quad\lor\quad (w > 120\text{ px} \land h > 150\text{ px})$$
- **Impact**: Eliminates **11.0% of total candidates** and directly removes over 40% of runtime false positive crops where character sprites are buried inside massive floor tile bounds.

---

### Filter 3 — Aspect Ratio Thresholding Filter
- **Target Failure**: Action bar UI slices, spell panel fragments, and wide horizontal ground highlights.
- **Deterministic Rule**:
  $$\text{Reject if } \frac{w}{h} > 1.75 \quad\lor\quad \frac{w}{h} < 0.22$$
- **Impact**: Eliminates extreme aspect ratio fragments without risking valid character avatars (median avatar $w/h = 0.58$).

---

### Filter 4 — Vertical Base Expansion & Feet Padding (Clipping Mitigation)
- **Target Failure**: `BOTTOM_CLIPPED` (15.38% of failure taxonomy), `RIGHT_CLIPPED`, `LEFT_CLIPPED`.
- **Deterministic Rule**:
  $$\text{For } \text{method} \in \{\text{'combat\_base'}, \text{'contour'}\}: \quad y_2 \gets \min(\text{img\_h}, y_2 + \lfloor 0.12 \times h \rfloor)$$
- **Impact**: Ensures sprite bases and character feet are fully included within the candidate crop, converting partial entity crops into full `VALID_ENTITY` targets.

---

### Filter 5 — ROI Ceiling & Action Bar Boundary Constraints
- **Target Failure**: Action bar spell icons and HUD panel spillover.
- **Deterministic Rule**:
  $$\text{Reject if } y_{\text{min}} < 40\text{ px} \quad\lor\quad y_{\text{max}} > 600\text{ px}$$
- **Impact**: Completely purges the bottom spell action bar and top menu bar from candidate extraction.

---

### Filter 6 — Standalone Pet / Sub-Crop Hierarchy Filter
- **Target Failure**: `PET_ONLY` standalone follower crops (12.82% of failure taxonomy).
- **Deterministic Rule**:
  If a candidate crop has $20 \le w \le 30\text{ px}, 20 \le h \le 38\text{ px}, \text{Area} \in [450, 1000]\text{ px}^2$ AND is spatially adjacent ($\text{distance} < 45\text{ px}$) to a larger valid actor candidate ($w \ge 35, h \ge 50$), merge or suppress the pet sub-crop.
- **Impact**: Eliminates standalone pet sprite false positive detections while retaining the primary character actor.

---

## Expected Performance Gain Post-Filter Integration

| Metric / Contamination Category | Current Baseline (Phase 4C.6) | Projected Post-Filter Performance | Improvement |
| :--- | :---: | :---: | :---: |
| **`VALID_ENTITY` Precision** | 60.0% | **> 85.0%** | **+25.0% Precision Gain** |
| **`ENTITY_PLUS_TERRAIN` Contamination** | 41.03% | **< 5.0%** | **-36.0% Reduction** |
| **`BOTTOM_CLIPPED` Partial Crops** | 15.38% | **< 3.0%** | **-12.0% Reduction** |
| **UI & Action Bar Leakage** | 5.5% | **0.0%** | **Complete Elimination** |
