# Phase 4D.1C — Positive Candidate Contamination Audit Summary

## Executive Summary

A total of **100 / 100 accepted positive predictions** ($P \ge 0.50$) were annotated and categorized to measure dataset contamination sources.

- **Annotated Samples**: `100`
- **Unannotated Samples**: `0`
- **Dominant Contamination Source**: `PARTIAL_ENTITY`

## 1. Category Frequency Breakdown

| Review Category | Count | Percentage Share (%) | Visual Impact & Interpretation |
| :--- | :---: | :---: | :--- |
| **`VALID_ENTITY`** | `60` | `60.0%` | True valid player character, monster, or NPC sprite core. |
| **`PARTIAL_ENTITY`** | `23` | `23.0%` | Character crop with low target occupancy (<30% body core). |
| **`DECORATION`** | `0` | `0.0%` | Scenery prop, statue, lamp post, wall column, or plant flora. |
| **`OVERSIZED_BBOX`** | `16` | `16.0%` | Bounding box covering multiple ground cells or terrain tiles. |
| **`UNCERTAIN`** | `1` | `1.0%` | Ambiguous crop requiring secondary visual context. |
| **Total Annotated** | `100` | `100.0%` | Audited positive candidates. |

## 2. Root Cause Analysis & Primary Contamination Drivers

The audit indicates that **`PARTIAL_ENTITY` crops** dominate false positive acceptances. Bounding box candidate extraction often captures ground floor tiles with only a tiny fraction of an entity edge.

## 3. Recommended Remediation Strategy

1. **Deterministic BBox Pre-Filtering**: Add size floor (`area >= 400`), size ceiling (`area <= 12000`), and ROI floor limits in `CharacterDetector` to purge oversized multi-tiles and micro-glints prior to visual classification.
2. **Targeted Negative Retraining**: Expand negative dataset with explicit `DECORATION` and `PARTIAL_ENTITY` crops to refine classifier decision boundaries in Phase 4D.2.
