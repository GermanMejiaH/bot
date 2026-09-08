# Mob Detection Recall Report

- **Overall Mob Recall**: **0.0%** (0 / 151 frames)

## Breakdown of Mob Miss Reasons

| Miss Reason Category | Occurrences |
| :--- | :---: |
| **NMS suppression** | 0 |
| **ROI clipping** | 0 |
| **filtering rejection** | 0 |
| **no candidate generated** | 151 |


### Findings:
Exploration mobs vary significantly in aspect ratio and bounding area. Wide/short mobs (aspect ratio $< 0.50$) or small summons (area $< 200$ px²) get discarded during candidate filtering by `_filter_box_with_reason` before reaching NMS.
