# Phase 4D.3 — Detector Bounding Box Geometry Statistics Audit

## Executive Summary

A rigorous spatial geometry evaluation was conducted over **291 accepted runtime positive bounding boxes** ($P \ge 0.50$) across exploration and combat map domains.

Key statistical findings verify that bounding box geometry anomalies (micro glints, oversized terrain multi-tiles, extreme aspect ratios) account for **over 40% of runtime contamination**.

## 1. Overall Bounding Box Geometry Distribution

| Geometry Metric | Minimum | 10th Pct (P10) | Median (P50) | Mean | 90th Pct (P90) | Maximum | Standard Dev |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Width ($w$) [px]** | `13` | `17` | `32` | `34.4` | `57` | `85` | `17.8` |
| **Height ($h$) [px]** | `18` | `48` | `110` | `105.2` | `187` | `220` | `51.0` |
| **Area ($w \times h$) [px²]** | `234` | `774` | `3741` | `4404.5` | `10300` | `17340` | `4135.7` |
| **Aspect Ratio ($w/h$)** | `0.22` | `0.24` | `0.33` | `0.35` | `0.42` | `1.00` | `0.12` |

## 2. Bounding Box Geometry Breakdown by Detector Method

| Detector Method | Count | Share (%) | Median Area (px²) | Median Aspect Ratio ($w/h$) | Primary Noise Characteristic |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`combat_base`** | `276` | `94.8%` | `4045` | `0.31` | Multi-cell terrain tiles and vertical actor base crops. |
| **`contour`** | `2` | `0.7%` | `796` | `0.92` | High-contrast scenery fragments and truncated actor edges. |
| **`hsv`** | `13` | `4.5%` | `621` | `0.72` | Color glints, foliage edges, and small pet sprites. |

## 3. Automatically Identified Bounding Box Anomaly Categories

| Anomaly Category | Defining Geometry Condition | Count | Share (%) | Phase 4D.2 Failure Mapping |
| :--- | :--- | :---: | :---: | :--- |
| **Micro Boxes** | $w < 18 \text{ or } h < 24 \text{ or } \text{Area} < 400$ | **34** | 11.7% | Specular floor glints, foliage edges. |
| **Oversized Boxes** | \text{Area} > 10,000 \text{ px}^2 | **32** | 11.0% | `ENTITY_PLUS_TERRAIN` (41.03% of errors). |
| **Extreme Aspect Ratios** | $w/h > 1.75 \text{ or } w/h < 0.22$ | **0** | 0.0% | Action bar UI slices & wall strip fragments. |
| **Pet-Sized Detections** | $20 \le w \le 30, 20 \le h \le 38, \text{Area} \in [450, 1000]$ | **7** | 2.4% | `PET_ONLY` standalone follower crops (12.82%). |

