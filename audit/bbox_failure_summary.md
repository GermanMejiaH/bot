# Phase 4D.2 — Bounding Box Geometric Failure Taxonomy Audit Summary

## Executive Summary

A detailed geometric failure mode audit was performed across **39 / 39 flagged candidate bounding boxes** (`PARTIAL_ENTITY` and `OVERSIZED_BBOX`).

- **Total Failure Candidates Audited**: `39`
- **Annotated Samples**: `39`
- **Unannotated Samples**: `0`
- **Dominant Geometric Failure Mode**: `ENTITY_PLUS_TERRAIN`

## 1. Geometric Failure Mode Distribution

| Failure Mode | Count | Percentage Share (%) | Visual Manifestation & Cause |
| :--- | :---: | :---: | :--- |
| **`LEFT_CLIPPED`** | `1` | `2.56%` | Geometric candidate bound distortion. |
| **`RIGHT_CLIPPED`** | `3` | `7.69%` | Geometric candidate bound distortion. |
| **`TOP_CLIPPED`** | `0` | `0.0%` | Geometric candidate bound distortion. |
| **`BOTTOM_CLIPPED`** | `6` | `15.38%` | Geometric candidate bound distortion. |
| **`PET_ONLY`** | `5` | `12.82%` | Geometric candidate bound distortion. |
| **`MULTI_ENTITY_FRAGMENT`** | `3` | `7.69%` | Geometric candidate bound distortion. |
| **`MULTI_TILE`** | `0` | `0.0%` | Geometric candidate bound distortion. |
| **`GROUND_TILE`** | `0` | `0.0%` | Geometric candidate bound distortion. |
| **`ENTITY_PLUS_TERRAIN`** | `16` | `41.03%` | Geometric candidate bound distortion. |
| **`ENTITY_PLUS_UI`** | `0` | `0.0%` | Geometric candidate bound distortion. |
| **`UNCERTAIN`** | `5` | `12.82%` | Geometric candidate bound distortion. |
| **Total Annotated** | `39` | `100.0%` | Audited geometry candidates. |

## 2. Root Cause Analysis & Detector Recommendations

The dominant geometric failure mode identified is **`ENTITY_PLUS_TERRAIN`**.

### Key Findings & Detector Refinement Roadmap:
1. **Boundary Clipping Mitigation**: If clipping (`TOP_CLIPPED`, `BOTTOM_CLIPPED`) dominates, adjust contour expansion padding and HSV bounding box expansion in `CharacterDetector`.
2. **Multi-Tile Terrain Purging**: If `MULTI_TILE` or `ENTITY_PLUS_TERRAIN` dominates, enforce bounding box area ceiling (`area <= 12,000`) and aspect ratio caps before visual classification.
3. **Pet/Fragment Filtering**: If `PET_ONLY` or `MULTI_ENTITY_FRAGMENT` dominates, refine color thresholds and contour hierarchy checks.
