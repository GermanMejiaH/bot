# Phase 3E — `contour` Detector Refinement Report

- **Total Candidate Samples**: `34`
- **Baseline True Positives**: `12`
- **Baseline False Positives**: `22`
- **Baseline Precision**: **`35.29%`**

## Top Candidate Refinement Thresholds

| Rule Candidate | TP Removed | FP Removed | Precision After | Recall After | Precision Gain | Recall Loss |
| --- | --- | --- | --- | --- | --- | --- |
| `edge_density >= 0.10` | 0 | 6 | **42.86%** | 100.0% | +7.57% | -0.0% |
| `fill_ratio >= 0.18` | 0 | 6 | **42.86%** | 100.0% | +7.57% | -0.0% |
| `fill_ratio >= 0.15` | 0 | 4 | **40.0%** | 100.0% | +4.71% | -0.0% |
| `fill_ratio >= 0.10` | 0 | 3 | **38.71%** | 100.0% | +3.42% | -0.0% |
| `Combined: edge_density >= 0.10 AND aspect_ratio <= 1.20` | 1 | 9 | **45.83%** | 91.67% | +10.54% | -8.33% |
| `aspect_ratio >= 0.25` | 0 | 0 | **35.29%** | 100.0% | +0.0% | -0.0% |
| `aspect_ratio >= 0.35` | 0 | 0 | **35.29%** | 100.0% | +0.0% | -0.0% |
| `edge_density >= 0.08` | 0 | 0 | **35.29%** | 100.0% | +0.0% | -0.0% |