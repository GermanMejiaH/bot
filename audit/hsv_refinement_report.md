# Phase 3E — `hsv` Detector Refinement Report

- **Total Candidate Samples**: `51`
- **Baseline True Positives**: `12`
- **Baseline False Positives**: `39`
- **Baseline Precision**: **`23.53%`**

## Top Candidate Refinement Thresholds

| Rule Candidate | TP Removed | FP Removed | Precision After | Recall After | Precision Gain | Recall Loss |
| --- | --- | --- | --- | --- | --- | --- |
| `bbox_area <= 8000` | 0 | 1 | **24.0%** | 100.0% | +0.47% | -0.0% |
| `200 <= bbox_area <= 8000` | 0 | 1 | **24.0%** | 100.0% | +0.47% | -0.0% |
| `aspect_ratio >= 0.25` | 0 | 0 | **23.53%** | 100.0% | +0.0% | -0.0% |
| `bbox_area >= 150` | 0 | 0 | **23.53%** | 100.0% | +0.0% | -0.0% |
| `bbox_area >= 200` | 0 | 0 | **23.53%** | 100.0% | +0.0% | -0.0% |
| `aspect_ratio >= 0.35` | 1 | 2 | **22.92%** | 91.67% | +-0.61% | -8.33% |
| `fill_ratio >= 0.10` | 1 | 2 | **22.92%** | 91.67% | +-0.61% | -8.33% |
| `fill_ratio >= 0.18` | 2 | 9 | **25.0%** | 83.33% | +1.47% | -16.67% |