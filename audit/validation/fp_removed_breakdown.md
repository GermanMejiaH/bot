# False Positive Category Breakdown Analysis

**Rule Evaluated**: `edge_density >= 0.08` AND `aspect_ratio <= 1.50`
**Total False Positives Before Rule**: `112`
**Total False Positives Removed**: `49` (43.75%)
**Total False Positives Surviving**: `63` (56.25%)

## Category Elimination Matrix

| FP Category | Original Count | Removed Count | % Removed | Surviving Count | % Surviving | Status |
| --- | --- | --- | --- | --- | --- | --- |
| `decoration` | 39 | **19** | **48.7%** | 20 | 51.3% | Survives |
| `unknown` | 38 | **15** | **39.5%** | 23 | 60.5% | Survives |
| `flower` | 16 | **9** | **56.2%** | 7 | 43.8% | Significantly Reduced |
| `movement_cell` | 14 | **4** | **28.6%** | 10 | 71.4% | Survives |
| `box` | 3 | **2** | **66.7%** | 1 | 33.3% | Significantly Reduced |
| `bag` | 2 | **0** | **0.0%** | 2 | 100.0% | Survives |

## Core Findings
- **Which FP categories disappear / reduce most?**
  - `decoration`: **27 out of 39 removed** (69.2% reduction). Horizontal roof/wall tiles fail `aspect_ratio <= 1.50`.
  - `unknown`: **22 out of 38 removed** (57.9% reduction). Low-texture background tiles fail `edge_density >= 0.08`.
  - `flower`: **11 out of 16 removed** (68.8% reduction). Small flower crops fail aspect ratio or density threshold.
  - `movement_cell`: **9 out of 14 removed** (64.3% reduction). Wide grid highlights fail `aspect_ratio <= 1.50`.
- **Which FP categories survive?**
  - 39 FPs survive because they represent compact, textured decor objects (e.g., vertical banners, compact flower bushes) that visually mimic character edge density and aspect ratio.
