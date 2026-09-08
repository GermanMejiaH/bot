# Phase 3E — `combat_base` Detector Refinement Report

- **Total Candidate Samples**: `67`
- **Baseline True Positives**: `16`
- **Baseline False Positives**: `51`
- **Baseline Precision**: **`23.88%`**

## Top Candidate Refinement Thresholds

| Rule Candidate | TP Removed | FP Removed | Precision After | Recall After | Precision Gain | Recall Loss |
| --- | --- | --- | --- | --- | --- | --- |
| `edge_density >= 0.08` | 0 | 7 | **26.67%** | 100.0% | +2.79% | -0.0% |
| `Combined: edge_density >= 0.08 AND aspect_ratio <= 1.50` | 0 | 7 | **26.67%** | 100.0% | +2.79% | -0.0% |
| `fill_ratio >= 0.18` | 0 | 2 | **24.62%** | 100.0% | +0.74% | -0.0% |
| `fill_ratio >= 0.20` | 0 | 2 | **24.62%** | 100.0% | +0.74% | -0.0% |
| `aspect_ratio <= 1.20` | 0 | 0 | **23.88%** | 100.0% | +0.0% | -0.0% |
| `fill_ratio >= 0.10` | 0 | 0 | **23.88%** | 100.0% | +0.0% | -0.0% |
| `fill_ratio >= 0.15` | 0 | 0 | **23.88%** | 100.0% | +0.0% | -0.0% |
| `bbox_area >= 150` | 0 | 0 | **23.88%** | 100.0% | +0.0% | -0.0% |