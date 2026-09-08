# Phase 3E — Feature Separation Analysis Report

**Dataset**: `152` total samples (`40` True Positives, `112` False Positives)
**Baseline Precision**: `26.32%`

## Feature Separation Ranking (Strongest to Weakest)

| Rank | Feature | TP Mean (Median) | FP Mean (Median) | Absolute Diff | Separation Score (Cohen's d) |
| --- | --- | --- | --- | --- | --- |
| 1 | `edge_density` | 0.1423 (0.1684) | 0.1034 (0.1136) | 0.0389 | **0.395** |
| 2 | `aspect_ratio` | 0.7893 (0.6204) | 0.6414 (0.4244) | 0.1479 | **0.3095** |
| 3 | `bbox_width` | 34.375 (28.0) | 31.7768 (25.5) | 2.5982 | **0.1356** |
| 4 | `fill_ratio` | 0.3865 (0.3566) | 0.3636 (0.3255) | 0.0229 | **0.1318** |
| 5 | `contour_area` | 319.9625 (109.5) | 237.1741 (52.5) | 82.7884 | **0.1295** |
| 6 | `bbox_height` | 60.425 (43.0) | 65.3125 (57.0) | 4.8875 | **0.1078** |
| 7 | `mask_ratio` | 0.2792 (0.3097) | 0.2603 (0.2747) | 0.0189 | **0.076** |
| 8 | `bbox_area` | 2447.65 (1135.0) | 2516.9464 (1268.0) | 69.2964 | **0.0233** |

## Detailed Feature Quantiles Breakdown

### Feature: `edge_density`
- **TP Quantiles**: P5=`0.0`, P25=`0.0`, P50=`0.1684`, P75=`0.2162`, P95=`0.3062`
- **FP Quantiles**: P5=`0.0`, P25=`0.0`, P50=`0.1136`, P75=`0.1781`, P95=`0.246`

### Feature: `aspect_ratio`
- **TP Quantiles**: P5=`0.262`, P25=`0.3119`, P50=`0.6204`, P75=`1.0487`, P95=`1.8936`
- **FP Quantiles**: P5=`0.2402`, P25=`0.318`, P50=`0.4244`, P75=`0.8636`, P95=`1.4201`

### Feature: `bbox_width`
- **TP Quantiles**: P5=`12.95`, P25=`19.0`, P50=`28.0`, P75=`44.0`, P95=`82.35`
- **FP Quantiles**: P5=`14.0`, P25=`20.0`, P50=`25.5`, P75=`39.0`, P95=`68.0`

### Feature: `fill_ratio`
- **TP Quantiles**: P5=`0.1807`, P25=`0.2687`, P50=`0.3566`, P75=`0.4865`, P95=`0.7037`
- **FP Quantiles**: P5=`0.1133`, P25=`0.2348`, P50=`0.3255`, P75=`0.4723`, P95=`0.6776`

### Feature: `contour_area`
- **TP Quantiles**: P5=`0.0`, P25=`0.0`, P50=`109.5`, P75=`309.5`, P95=`1299.7`
- **FP Quantiles**: P5=`0.0`, P25=`0.0`, P50=`52.5`, P75=`242.5`, P95=`672.225`

### Feature: `bbox_height`
- **TP Quantiles**: P5=`17.0`, P25=`25.0`, P50=`43.0`, P75=`88.0`, P95=`144.45`
- **FP Quantiles**: P5=`19.55`, P25=`29.0`, P50=`57.0`, P75=`81.25`, P95=`159.7`

### Feature: `mask_ratio`
- **TP Quantiles**: P5=`0.0`, P25=`0.0`, P50=`0.3097`, P75=`0.4355`, P95=`0.7054`
- **FP Quantiles**: P5=`0.0`, P25=`0.0`, P50=`0.2747`, P75=`0.4723`, P95=`0.6522`

### Feature: `bbox_area`
- **TP Quantiles**: P5=`284.7`, P25=`594.0`, P50=`1135.0`, P75=`3586.5`, P95=`6479.9`
- **FP Quantiles**: P5=`308.25`, P25=`814.5`, P50=`1268.0`, P75=`2833.25`, P95=`10410.8`
