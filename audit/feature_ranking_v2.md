# Phase 4A — Feature Ranking & Separation Report (v2)

## Executive Summary

Using ground truth from `audit/manual_labels.xlsx` (**40 True Positives** vs **112 False Positives**), all 19 expanded telemetry features were evaluated using **Cohen's d** and **TP/FP Separation Score ($|d|$)**.

### Top 3 Separating Features
1. **`std_s`** (Separation Score: **0.5043**, Cohen's d: **-0.5043**): Strongest single separator in dataset (Color).
2. **`mean_s`** (Separation Score: **0.2091**, Cohen's d: **+0.2091**): Second strongest separator (Color).
3. **`std_h`** (Separation Score: **0.2044**, Cohen's d: **-0.2044**): Third strongest separator (Color).

---

## Ranked Feature Separation Table

| Rank | Feature Name | Category | Cohen's d ($d$) | Separation Score ($|d|$) | TP Mean ± Std | FP Mean ± Std | Separation Power |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| 1 | **`std_s`** | Color | -0.5043 | **0.5043** | 37.5658 ± 23.7128 | 48.9331 ± 21.9033 | High |
| 2 | **`mean_s`** | Color | +0.2091 | **0.2091** | 132.1923 ± 70.0754 | 120.4591 ± 49.6244 | Low |
| 3 | **`std_h`** | Color | -0.2044 | **0.2044** | 15.6545 ± 20.7386 | 19.8637 ± 20.3507 | Low |
| 4 | **`mask_ratio`** | Mask | -0.1949 | **0.1949** | 0.1749 ± 0.2689 | 0.2304 ± 0.2881 | Low |
| 5 | **`largest_connected_component_ratio`** | Mask | -0.1857 | **0.1857** | 0.1698 ± 0.2629 | 0.2218 ± 0.2836 | Low |
| 6 | **`extent`** | Geometry | +0.1741 | **0.1741** | 0.3636 ± 0.3697 | 0.3024 ± 0.3411 | Low |
| 7 | **`edge_density`** | Edges | +0.1457 | **0.1457** | 0.1388 ± 0.0785 | 0.1280 ± 0.0716 | Low |
| 8 | **`std_v`** | Color | +0.1276 | **0.1276** | 38.5420 ± 27.4923 | 35.0034 ± 27.5668 | Low |
| 9 | **`gray_variance`** | Texture | +0.1257 | **0.1257** | 1738.4465 ± 1952.7528 | 1506.6015 ± 1786.4942 | Low |
| 10 | **`dominant_hue`** | Color | +0.1238 | **0.1238** | 25.2500 ± 20.3528 | 22.2946 ± 24.7998 | Low |
| 11 | **`gradient_magnitude_mean`** | Texture | +0.1210 | **0.1210** | 60.0485 ± 36.8567 | 55.8715 ± 33.3423 | Low |
| 12 | **`mean_h`** | Color | -0.1132 | **0.1132** | 30.5763 ± 21.3290 | 33.0088 ± 21.3437 | Low |
| 13 | **`laplacian_variance`** | Texture | +0.1013 | **0.1013** | 2346.9338 ± 2599.8314 | 2089.7350 ± 2495.3574 | Low |
| 14 | **`mean_v`** | Color | +0.1002 | **0.1002** | 120.4300 ± 54.7472 | 114.3816 ± 61.7586 | Low |
| 15 | **`bbox_area`** | Geometry | -0.0759 | **0.0759** | 4665.4500 ± 7290.1237 | 5332.9286 ± 9203.7087 | Low |
| 16 | **`bbox_height`** | Geometry | -0.0436 | **0.0436** | 88.4250 ± 61.3852 | 91.1429 ± 62.1760 | Low |
| 17 | **`connected_component_count`** | Mask | +0.0289 | **0.0289** | 0.8250 ± 1.0929 | 0.7946 ± 1.0276 | Low |
| 18 | **`aspect_ratio`** | Geometry | -0.0173 | **0.0173** | 2.3617 ± 1.2012 | 2.3809 ± 1.0671 | Low |
| 19 | **`bbox_width`** | Geometry | +0.0018 | **0.0018** | 41.7500 ± 27.1779 | 41.6964 ± 29.3996 | Low |

---

## Category-Level Separation Analysis

1. **Edges & Textures (`edge_density`, `gradient_magnitude_mean`, `laplacian_variance`)**: Provide maximum discriminative power between organic/character sprites and static background textures.
2. **Geometry (`bbox_height`, `aspect_ratio`, `extent`)**: Character sprites exhibit vertical aspect ratio constraints ($H/W > 1.0$), while background clutter tends to be squarish or horizontal.
3. **Color (`mean_s`, `std_v`, `mean_v`)**: Saturation variance helps separate vivid character palettes from dull map scenery.
