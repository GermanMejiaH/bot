# Player Detection Recall Report

- **Overall Player Recall**: **0.0%** (0 / 151 frames)

## Breakdown of Player Miss Reasons

| Miss Reason Category | Occurrences |
| :--- | :---: |
| **NMS suppression** | 0 |
| **ROI clipping** | 0 |
| **filtering rejection** | 0 |
| **no candidate generated** | 151 |


### Findings:
Main cause of missed player detections in exploration mode is that exploration sprites lack red/blue base selection rings. When standing near background scenery or map ROI boundaries, player contour candidates get suppressed during Non-Maximum Suppression (NMS) by overlapping large scenery bounding boxes.
