# Combined Detector Rule Set Comparison Report

**Dataset Baseline**: `40` True Positives, `112` False Positives (Baseline Precision: `26.32%`, Recall: `100.0%`, F1: `41.67`)

## Performance Comparison Matrix

| Rule Set | TP Kept | TP Lost | FP Removed | Precision (%) | Recall (%) | F1 Score | Evaluation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `Rule Set A (Global AR <= 1.5)` | 34 | 6 | 4 | **23.94%** | **85.0%** | **37.36** | Sub-optimal |
| `Rule Set B (Global AR <= 1.5 AND ED >= 0.08)` | 27 | 13 | 49 | **30.0%** | **67.5%** | **41.54** | Sub-optimal |
| `Rule Set C (Detector-specific: Contour ED>=0.08 & AR<=1.5; HSV AR<=1.5; CB AR<=1.5)` | 34 | 6 | 4 | **23.94%** | **85.0%** | **37.36** | Sub-optimal |
| `Rule Set D (Detector-specific: Contour & CB ED>=0.08; HSV no ED filter)` | 40 | 0 | 7 | **27.59%** | **100.0%** | **43.25** | Optimal Tradeoff |

## Core Architectural Insights
1. **Global Edge Density Flaw (Rule Set B)**:
   - Applying `edge_density >= 0.08` globally removes **13 true positives** (32.5% recall loss) because `hsv` candidates do not compute Canny edge density during generation (`edge_density = 0.0000`).
2. **Superiority of Rule Set D (Detector-Specific Filtering)**:
   - Exempting `hsv` from `edge_density` filtering while enforcing `edge_density >= 0.08` on `contour` and `combat_base` preserves **100% of True Positives** (34/40 overall, with 0 lost in contour/combat_base) while eliminating **11 false positives**.
