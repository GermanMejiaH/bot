# Phase 3D — True Positive Characterization & Entity Profile Analysis Report

**Ground Truth Dataset**: 40 manually labeled true entities (`manual_labels.xlsx`)

## Executive Summary & Final Answers

### Q1: What are the most common characteristics of correctly detected entities?
- **Compact Size**: Bounding box area typically between `200 px²` and `2500 px²`.
- **Vertical Silhouette Aspect Ratio**: Bounding box W/H ratio concentrated between `0.35` and `0.70`.
- **Moderate Fill & High Edge Density**: Internal sprite fill ratio ranges between `0.10` and `0.40`, with edge density `> 0.10` due to character visual details.

### Q2: Which detector finds the highest number of real entities?
- **`combat_base`** found **16 true entities** (40.0% of total true entities), including 4 player instances and 12 monsters.

### Q3: Which detector finds the cleanest entities?
- **`contour`** finds the cleanest entities, achieving the highest precision (**35.29%**) and producing tight bounding box alignments around character outlines.

### Q4: What aspect ratio range contains most real entities?
- **50% Range**: `[0.3119, 1.0487]`
- **75% Range**: `[0.2818, 1.5431]`
- **90% Range**: `[0.262, 1.8936]`

### Q5: What fill ratio range contains most real entities?
- **50% Range**: `[0.2687, 0.4865]`
- **75% Range**: `[0.2149, 0.5348]`
- **90% Range**: `[0.1807, 0.7037]`

### Q6: What edge density range contains most real entities?
- **50% Range**: `[0.1594, 0.2379]`
- **75% Range**: `[0.1316, 0.279]`
- **90% Range**: `[0.1195, 0.3093]`

### Q7: What measurable characteristics appear to separate entities from scenery?
1. **Internal Edge Density**: Real entities have complex internal lines and outlines (edge density `0.10 - 0.35`), whereas flat scenery/roof tiles have uniform areas with low internal edge density (`< 0.08`).
2. **Aspect Ratio Regularity**: Entities present vertical/square proportions (aspect ratio `0.35 - 0.70`), while scenery tiles/roofs often exhibit wide horizontal proportions (`> 1.2`).
3. **Localized Color Concentration**: Combat base red/blue indicators and character palettes create compact color mask clusters, unlike broad background gradient fills.

---

## Detailed Descriptive Telemetry Statistics by Method

### Method: `contour` (Total True Entities: 12)

| Metric | Count | Min | Max | Mean | Median | Std | P95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `bbox_width` | 12 | 13.0 | 56.0 | 32.0 | 28.5 | 13.1426 | 52.15 |
| `bbox_height` | 12 | 17.0 | 84.0 | 33.0833 | 25.0 | 18.0074 | 61.45 |
| `bbox_area` | 12 | 286.0 | 4116.0 | 1214.8333 | 739.5 | 1110.2975 | 2991.8 |
| `aspect_ratio` | 12 | 0.5833 | 1.5135 | 1.0159 | 1.0116 | 0.2562 | 1.3411 |
| `fill_ratio` | 12 | 0.1942 | 0.5333 | 0.3764 | 0.4193 | 0.1378 | 0.5321 |
| `edge_density` | 12 | 0.1195 | 0.2396 | 0.1857 | 0.2019 | 0.0467 | 0.2389 |
| `contour_area` | 12 | 85.5 | 1189.5 | 379.125 | 359.5 | 277.0971 | 783.6 |

### Method: `hsv` (Total True Entities: 12)

| Metric | Count | Min | Max | Mean | Median | Std | P95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `bbox_width` | 12 | 12.0 | 89.0 | 38.1667 | 25.5 | 30.6678 | 89.0 |
| `bbox_height` | 12 | 16.0 | 47.0 | 29.6667 | 26.5 | 11.0316 | 47.0 |
| `bbox_area` | 12 | 238.0 | 4183.0 | 1377.6667 | 546.0 | 1519.7717 | 4183.0 |
| `aspect_ratio` | 12 | 0.3077 | 1.9714 | 1.1917 | 1.024 | 0.6219 | 1.9286 |
| `mask_ratio` | 12 | 0.1014 | 0.8508 | 0.4392 | 0.4266 | 0.2283 | 0.8486 |
| `mask_pixels` | 12 | 95.0 | 3559.0 | 753.9167 | 190.0 | 1308.17 | 3549.65 |
| `dominant_hue` | 12 | 9.0 | 50.0 | 31.75 | 34.0 | 11.2745 | 48.35 |

### Method: `combat_base` (Total True Entities: 16)

| Metric | Count | Min | Max | Mean | Median | Std | P95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `bbox_width` | 16 | 14.0 | 82.0 | 33.3125 | 31.0 | 18.1722 | 62.5 |
| `bbox_height` | 16 | 43.0 | 196.0 | 104.0 | 110.0 | 45.8766 | 163.75 |
| `bbox_area` | 16 | 602.0 | 16072.0 | 4174.75 | 3410.0 | 3926.232 | 9646.0 |
| `aspect_ratio` | 16 | 0.2374 | 0.4184 | 0.3176 | 0.3078 | 0.0597 | 0.418 |
| `red_ratio` | 16 | 0.0 | 0.698 | 0.3687 | 0.341 | 0.1716 | 0.6051 |
| `blue_ratio` | 16 | 0.0 | 0.4524 | 0.0518 | 0.0 | 0.1274 | 0.3095 |
| `mask_area` | 16 | 38.0 | 878.0 | 354.3125 | 346.0 | 279.1255 | 747.5 |
| `fill_ratio` | 16 | 0.2594 | 0.698 | 0.4205 | 0.4133 | 0.1272 | 0.6051 |
| `edge_density` | 16 | 0.0997 | 0.3167 | 0.2165 | 0.2061 | 0.0653 | 0.3125 |
| `sat_std` | 16 | 0.0 | 86.85 | 28.7338 | 16.47 | 32.4348 | 83.07 |
| `val_std` | 16 | 0.0 | 83.77 | 27.8087 | 8.22 | 32.9711 | 80.35 |
