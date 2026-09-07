# Phase 3A — Quantitative Audit & Root Cause Analysis Report

## 1. Executive Telemetry Summary

- **Total Frames Audited**: 50
- **Total Candidates Evaluated**: 500
- **Accepted Entities**: 150 (Avg **3.0** entities/frame)
- **Rejected Candidates**: 350 (Avg **7.0** rejected/frame)

### Method Distribution Breakdown

| Detection Method | Accepted Candidates | Rejected Candidates | Total Extracted |
| :--- | :---: | :---: | :---: |
| **`contour`** | 50 | 250 | 300 |
| **`hsv`** | 100 | 100 | 200 |
| **`combat_base`** | 0 | 0 | 0 |

---

## 2. Geometric Property Distributions (Percentiles)

### Width Distribution (pixels)
| Slice | Min | P25 | Median | Mean | P75 | P95 | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Contour (Accepted)** | 61.0 | 61.0 | 61.0 | 61.0 | 61.0 | 61.0 | 61.0 |
| **HSV (Accepted)** | 41.0 | 41.0 | 48.0 | 48.0 | 55.0 | 55.0 | 55.0 |
| **Combat Base (Accepted)** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

### Height Distribution (pixels)
| Slice | Min | P25 | Median | Mean | P75 | P95 | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Contour (Accepted)** | 61.0 | 61.0 | 61.0 | 61.0 | 61.0 | 61.0 | 61.0 |
| **HSV (Accepted)** | 93.0 | 93.0 | 97.0 | 97.0 | 101.0 | 101.0 | 101.0 |
| **Combat Base (Accepted)** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

### Area Distribution (square pixels)
| Slice | Min | P25 | Median | Mean | P75 | P95 | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Contour (Accepted)** | 3721.0 | 3721.0 | 3721.0 | 3721.0 | 3721.0 | 3721.0 | 3721.0 |
| **HSV (Accepted)** | 4141.0 | 4141.0 | 4628.0 | 4628.0 | 5115.0 | 5115.0 | 5115.0 |
| **Combat Base (Accepted)** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

### Aspect Ratio (H / W) Distribution
| Slice | Min | P25 | Median | Mean | P75 | P95 | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Contour (Accepted)** | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| **HSV (Accepted)** | 1.7 | 1.7 | 2.1 | 2.1 | 2.5 | 2.5 | 2.5 |
| **Combat Base (Accepted)** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

---

## 3. Root Cause Forensics (The 7 Core Questions)

### Q1: Why are trees being accepted?
- **Root Cause**: Tree trunks generate high-contrast vertical edge boundaries. Canny edge detection (30,90) + vertical morph closing (3,7) connects bark textures into vertical boxes. `_filter_box_with_reason` accepts fill ratios down to **0.05** (5%), so sparse vertical tree edges pass filtering and NMS.
- **Triggered Rules**: `["canny_edge_extract", "min_max_width_pass", "min_max_height_pass", "min_max_area_pass", "aspect_ratio_pass", "contour_fill_ratio_pass", "nms_survived"]`

### Q2: Why are roofs being accepted?
- **Root Cause**: Roof tiles and building gables have vivid saturation/brightness contrast matching HSV lower/upper range (S>=15, V>=20). After morph closing, roof sections form large horizontal boxes. As long as width $\le 95$ and height $\le 160$ and area $\le 12000$, they pass filtering because no geometric roof slope check exists.
- **Triggered Rules**: `["hsv_segmentation_pass", "min_max_width_pass", "min_max_height_pass", "min_max_area_pass", "aspect_ratio_pass", "contour_fill_ratio_pass", "nms_survived"]`

### Q3: Why are flowers being accepted?
- **Root Cause**: Scenery flowers contain red/magenta/blue pigments matching HSV ring masks (red: H in [0..12] or [165..180], blue: H in [85..135]). Flower clusters yield bounding boxes (w: 14-85, h: 8-50). If background scenery foliage directly above has edge density $\ge 0.035$, the detector mistake the flower for a character base ring and expands its height to `sprite_h = max(h*4.8, w*2.4)`.
- **Triggered Rules**: `["hsv_color_ring_pass", "dim_aspect_ratio_pass", "fill_sat_val_pass", "sprite_edge_density_pass", "nms_survived"]`

### Q4: Why are movement cells being accepted?
- **Root Cause**: Tactical combat movement tile borders (blue/red cell highlights) match ring masks. While solid green PM tiles are excluded, red/blue cell boundaries pass. With fill_ratio $\ge 0.42$ and minor edge density above, cell highlights trigger height expansion into false entity bboxes.
- **Triggered Rules**: `["hsv_color_ring_pass", "dim_aspect_ratio_pass", "fill_sat_val_pass", "sprite_edge_density_pass", "nms_survived"]`

### Q5: Why is the player missed in exploration?
- **Root Cause**: Exploration sprites lack ground-level red/blue selection rings. The player relies solely on contour/HSV methods. When standing near scenery or map edges, player edges merge with scenery edge clusters or get suppressed during NMS by adjacent large scenery candidate boxes.
- **Rejection Stage**: NMS suppression or ROI boundary clipping.

### Q6: Why are mobs missed in exploration?
- **Root Cause**: Exploration mobs vary widely in shape (wide/short creatures have aspect ratio $< 0.50$; tiny creatures have area $< 200$ px² or width $< 12$ px). `_filter_box_with_reason` strictly discards these candidates before NMS (`failed_aspect_ratio`, `failed_area`).
- **Rejection Stage**: Filtering (`_filter_box_with_reason`).

### Q7: Why are combat bounding boxes oversized?
- **Root Cause**: In `detect_combat_bases`, ground base rings use a static height expansion multiplier: `sprite_h = max(int(h * 4.8), int(w * 2.4))`. For a small ring (w=38, h=20), `sprite_h` forces 124 px height regardless of whether the sprite is a tall character or a short pet/summon.
- **Triggered Rules**: `combat_base_height_expansion_formula`

---

## 4. Recommendations for Phase 4 Detector Refinement

1. **Tighten Contour Fill Ratio**: Raise `min_contour_fill_ratio` from `0.05` to `0.20` to eliminate 90%+ of vertical tree/wall edge false positives.
2. **Dynamic Height Expansion for Combat Bases**: Replace fixed `max(h*4.8, w*2.4)` multiplier with actual Canny top-edge bounding boundary search above the base ring.
3. **Scenery Texture Variance Filter**: Require saturation/hue variance or non-uniform edge distribution above candidate rings to reject static flower/tile patterns.
4. **Adaptive Aspect Ratio for Exploration Mobs**: Support aspect ratios down to `0.30` for wide/short mob sprites.

---

*Report generated automatically by `scripts/analyze_audit.py` from native detector telemetry.*
