# Detector Contribution Analysis Report

**Breakdown of True Positive Entity Detections by Candidate Generator Pipeline**

| Detector Method | Total Entities Detected | Player Count | Monster Count | NPC Count | Entity Share (%) |
| --- | --- | --- | --- | --- | --- |
| `contour` | 12 | 1 | 11 | 0 | **30.0%** |
| `hsv` | 12 | 1 | 11 | 0 | **30.0%** |
| `combat_base` | 16 | 4 | 12 | 0 | **40.0%** |
| **TOTAL** | **40** | **6** | **34** | **0** | **100.0%** |

## Summary of Findings
- `combat_base` contributed the highest raw number of true entity detections (**16 / 40**, **40.0%** share), successfully capturing 4 player instances and 12 monsters.
- `contour` and `hsv` each contributed **12 true entities** (**30.0%** share each), capturing 1 player and 11 monsters respectively.
- `contour` produced the cleanest, most closely wrapped bounding boxes around character silhouettes.
