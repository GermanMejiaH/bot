# Phase 4D.1 — Bounding Box Quality & Occupancy Audit Report

## Executive Summary

A total of **291 accepted positive crops** ($P \ge 0.50$) across exploration and combat domains were subjected to spatial bounding box quality evaluation.

## 1. Bounding Box Area Distribution Statistics

| Metric | Area (px²) | Visual Representation |
| :--- | :---: | :--- |
| **Minimum Area** | `234` | Tiny micro-crop |
| **10th Percentile (P10)** | `774` | Small candidate bound |
| **Median Area (P50)** | `3741` | Standard character sprite crop |
| **Mean Area** | `4404` | Average candidate bounding box |
| **90th Percentile (P90)** | `10300` | Large candidate bound |
| **Maximum Area** | `17340` | Oversized multi-tile crop |

## 2. Bounding Box Aspect Ratio Distribution ($w / h$)

| Metric | Aspect Ratio ($w/h$) | Interpretation |
| :--- | :---: | :--- |
| **Minimum Aspect Ratio** | `0.22` | Extremely narrow vertical crop |
| **Median Aspect Ratio** | `0.33` | Standard character proportion |
| **Mean Aspect Ratio** | `0.35` | Average bounding box shape |
| **Maximum Aspect Ratio** | `1.00` | Extremely wide horizontal crop |

## 3. Flagged Bounding Box Anomaly Categories

| Anomaly Flag | Filter Condition | Flagged Count | Share (%) | Root Cause & Impact |
| :--- | :--- | :---: | :---: | :--- |
| **Micro-Crops** | $w < 18 \text{ or } h < 24 \text{ or } \text{Area} < 400$ | **34** | 11.7% | Specular glints, floor tile highlights, or foliage edges. |
| **Oversized Crops** | \text{Area} > 15,000 \text{ px}^2 | **15** | 5.2% | Multi-tile combat grid bounding boxes or scenery clusters. |
| **Suspicious Aspect Ratios** | $w/h > 1.8 \text{ or } w/h < 0.2$ | **0** | 0.0% | Bottom HUD banner slices or wide ground tile highlights. |
| **Low Target Occupancy** | \text{Area} > 5000 \text{ and } w/h > 1.5 | **0** | 0.0% | Crop dominated by background terrain (>70% tile floor). |

## 4. Recommendations for Pre-Classifier BBox Filtering

1. **Minimum Size Floor**: Enforce `w >= 18` AND `h >= 24` AND `area >= 400` in `CharacterDetector` to purge micro-crops prior to visual classification.
2. **Maximum Size Ceiling**: Cap candidate crop size at `area <= 12,000` to prevent oversized scenery multi-tiles.
3. **Aspect Ratio Ceiling**: Filter out candidates with aspect ratio $w/h > 1.75$.
