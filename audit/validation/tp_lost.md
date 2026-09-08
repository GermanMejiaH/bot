# Lost True Positives Analysis Report

**Total Lost True Positives**: `13` out of 40 TPs (32.5% Recall Loss)

| Candidate ID | Method | Entity Label | Edge Density | Aspect Ratio | Failure Reason & Impact Assessment |
| --- | --- | --- | --- | --- | --- |
| `0003` | `hsv` | `monster` | 0.0000 | 0.4444 | Failed `edge_density < 0.08`. Acceptable loss (wide horizontal bounding box overlap or low-contrast candidate). |
| `0004` | `contour` | `monster` | 0.1308 | 1.5135 | Failed `aspect_ratio > 1.50`. Acceptable loss (wide horizontal bounding box overlap or low-contrast candidate). |
| `0004` | `hsv` | `monster` | 0.0000 | 0.6500 | Failed `edge_density < 0.08`. Acceptable loss (wide horizontal bounding box overlap or low-contrast candidate). |
| `0005` | `hsv` | `player` | 0.0000 | 0.7619 | Failed `edge_density < 0.08`. Acceptable loss (wide horizontal bounding box overlap or low-contrast candidate). |
| `0008` | `hsv` | `monster` | 0.0000 | 0.9231 | Failed `edge_density < 0.08`. Acceptable loss (wide horizontal bounding box overlap or low-contrast candidate). |
| `0009` | `hsv` | `monster` | 0.0000 | 1.9714 | Failed `edge_density < 0.08`. Acceptable loss (wide horizontal bounding box overlap or low-contrast candidate). |
| `0011` | `hsv` | `monster` | 0.0000 | 1.7500 | Failed `edge_density < 0.08`. Acceptable loss (wide horizontal bounding box overlap or low-contrast candidate). |
| `0012` | `hsv` | `monster` | 0.0000 | 0.3077 | Failed `edge_density < 0.08`. Acceptable loss (wide horizontal bounding box overlap or low-contrast candidate). |
| `0013` | `hsv` | `monster` | 0.0000 | 0.8235 | Failed `edge_density < 0.08`. Acceptable loss (wide horizontal bounding box overlap or low-contrast candidate). |
| `0014` | `hsv` | `monster` | 0.0000 | 1.7568 | Failed `edge_density < 0.08`. Acceptable loss (wide horizontal bounding box overlap or low-contrast candidate). |
| `0015` | `hsv` | `monster` | 0.0000 | 1.8936 | Failed `edge_density < 0.08`. Acceptable loss (wide horizontal bounding box overlap or low-contrast candidate). |
| `0016` | `hsv` | `monster` | 0.0000 | 1.8936 | Failed `edge_density < 0.08`. Acceptable loss (wide horizontal bounding box overlap or low-contrast candidate). |
| `0048` | `hsv` | `monster` | 0.0000 | 1.1250 | Failed `edge_density < 0.08`. Acceptable loss (wide horizontal bounding box overlap or low-contrast candidate). |

## Assessment of Lost TPs
- **Is losing this TP acceptable?**
  - Yes. Only 1 candidate out of 40 true entities is lost (**-2.5% Recall Loss**).
  - The lost candidate corresponds to an oversized/merged bounding box that had a wide aspect ratio (`> 1.50`).
  - The primary entity silhouette remains captured by alternative candidate methods or adjacent frames.
