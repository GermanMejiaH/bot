# Phase 3F — `combat_base` Detector-Specific Threshold Sweep Report

- **Accepted Candidates**: `67`
- **True Positives**: `16`
- **False Positives**: `51`
- **Baseline Precision**: `23.88%`

## 1. Aspect Ratio Sweeps (Upper Ceiling: `aspect_ratio <= T`)

| Aspect Ratio Ceiling | TP Removed | FP Removed | Precision After | Recall After | Precision Gain | Recall Loss |
| --- | --- | --- | --- | --- | --- | --- |
| `<= 0.8` | 0 | 0 | **23.88%** | 100.0% | +0.0% | -0.0% |
| `<= 1.0` | 0 | 0 | **23.88%** | 100.0% | +0.0% | -0.0% |
| `<= 1.2` | 0 | 0 | **23.88%** | 100.0% | +0.0% | -0.0% |
| `<= 1.5` | 0 | 0 | **23.88%** | 100.0% | +0.0% | -0.0% |
| `<= 1.8` | 0 | 0 | **23.88%** | 100.0% | +0.0% | -0.0% |
| `<= 2.0` | 0 | 0 | **23.88%** | 100.0% | +0.0% | -0.0% |

## 2. Edge Density Sweeps (Lower Floor: `edge_density >= T`)

| Edge Density Floor | TP Removed | FP Removed | Precision After | Recall After | Precision Gain | Recall Loss |
| --- | --- | --- | --- | --- | --- | --- |
| `>= 0.00` | 0 | 0 | **23.88%** | 100.0% | +0.0% | -0.0% |
| `>= 0.02` | 0 | 0 | **23.88%** | 100.0% | +0.0% | -0.0% |
| `>= 0.04` | 0 | 0 | **23.88%** | 100.0% | +0.0% | -0.0% |
| `>= 0.06` | 0 | 2 | **24.62%** | 100.0% | +0.74% | -0.0% |
| `>= 0.08` | 0 | 7 | **26.67%** | 100.0% | +2.79% | -0.0% |
| `>= 0.10` | 1 | 9 | **26.32%** | 93.75% | +2.44% | -6.25% |
| `>= 0.12` | 1 | 13 | **28.3%** | 93.75% | +4.42% | -6.25% |
| `>= 0.15` | 3 | 19 | **28.89%** | 81.25% | +5.01% | -18.75% |

## 3. Fill Ratio Sweeps (Lower Floor: `fill_ratio >= T`)

| Fill Ratio Floor | TP Removed | FP Removed | Precision After | Recall After | Precision Gain | Recall Loss |
| --- | --- | --- | --- | --- | --- | --- |
| `>= 0.15` | 0 | 0 | **23.88%** | 100.0% | +0.0% | -0.0% |
| `>= 0.20` | 0 | 2 | **24.62%** | 100.0% | +0.74% | -0.0% |
| `>= 0.25` | 0 | 7 | **26.67%** | 100.0% | +2.79% | -0.0% |
| `>= 0.30` | 3 | 16 | **27.08%** | 81.25% | +3.2% | -18.75% |
| `>= 0.35` | 7 | 25 | **25.71%** | 56.25% | +1.83% | -43.75% |
| `>= 0.40` | 7 | 27 | **27.27%** | 56.25% | +3.39% | -43.75% |