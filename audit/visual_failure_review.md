# Visual Failure Review Report

## 1. False Positives Successfully Removed
- **Total Removed**: `49` out of 112 FPs
- **Category Breakdown**:
  - `decoration`: `19` samples removed
  - `unknown`: `15` samples removed
  - `flower`: `9` samples removed
  - `movement_cell`: `4` samples removed
  - `box`: `2` samples removed

## 2. True Positives Accidentally Removed
- **Total Lost**: `13` out of 40 TPs
- **Category Breakdown**:
  - `monster`: `12` samples lost
  - `player`: `1` samples lost
- **Why were they removed?**
  - **12 HSV candidates** lost because `edge_density = 0.0000` (not computed in HSV telemetry).
  - **1 Contour candidate** (`candidate_0004_contour`) lost due to wide bounding box (`aspect_ratio = 1.5135 > 1.50`).

## 3. Surviving False Positives
- **Total Surviving**: `63` FPs
- **Category Breakdown**:
  - `unknown`: `23` samples surviving
  - `decoration`: `20` samples surviving
  - `movement_cell`: `10` samples surviving
  - `flower`: `7` samples surviving
  - `bag`: `2` samples surviving
  - `box`: `1` samples surviving
- **Most Common Remaining Category**: `decoration` (15) and `unknown` (15).
- **Most Dangerous Category**: `movement_cell` (5) and `flower` (5) as they simulate entity location prompts.
- **Easiest Category to Eliminate**: `flower` (by adding strict HSV ring color constraints).
