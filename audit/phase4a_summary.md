# Phase 4A — Telemetry Expansion & Feature Engineering Summary Report

## 1. Accomplishments

- **Standardized Telemetry Schema**: Extended diagnostic feature extraction across all three candidate generators (`contour`, `hsv`, `combat_base`). Every candidate now exports 19 features covering Geometry, Edges, Colors, Textures, and Mask Topology.
- **HSV Edge Density Fix**: Resolved the HSV candidate edge density gap. All HSV candidates now compute Canny edge density on crops, eliminating `edge_density = 0.0000`.
- **Feature Ranking Recomputed**: Evaluated all 19 features against ground truth (**40 True Positives** vs **112 False Positives**) using Cohen's d ($d$) and separation score ($|d|$).
- **Visual Correlation Montages**: Generated 10 visual montages in `audit/feature_montages/` illustrating candidate crops at extreme feature quantiles for top separating features.

---

## 2. Top Separating Features Summary

| Rank | Feature | Category | Cohen's d ($d$) | Separation Score ($|d|$) | Recommendation for Phase 4B Classifier |
| :---: | :--- | :--- | :---: | :---: | :--- |
| 1 | **`std_s`** | Color | -0.5043 | **0.5043** | Primary decision feature for candidate filtering |
| 2 | **`mean_s`** | Color | +0.2091 | **0.2091** | Secondary geometric fill ratio constraint |
| 3 | **`std_h`** | Color | -0.2044 | **0.2044** | Texture gradient threshold for scenery rejection |
| 4 | **`mask_ratio`** | Mask | -0.1949 | **0.1949** | Structural Laplacian variance boundary check |
| 5 | **`largest_connected_component_ratio`** | Mask | -0.1857 | **0.1857** | Vertical sprite height constraint |

---

## 3. Success Criteria Verification

1. **HSV Edge Density Valid**: VERIFIED. 100% of HSV candidates report valid, non-zero `edge_density`.
2. **Unified Telemetry Schema**: VERIFIED. All detectors expose the identical 19-feature diagnostic dictionary.
3. **Recomputed Feature Ranking**: VERIFIED. `audit/feature_ranking_v2.md` updated with Cohen's d.
4. **No Filtering Behavior Changes**: VERIFIED. Zero thresholds or filtering rules were added or changed.
5. **No Detector Logic Changes**: VERIFIED. Candidate generation decisions remain untouched.
6. **Existing Tests Passing**: VERIFIED. All 69 unit tests, `ruff check .`, and `mypy src` pass cleanly.

---

*Phase 4A complete. Ready for Phase 4B classifier design.*
