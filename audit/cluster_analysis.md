# Phase 4A/4B — Unsupervised Cluster Analysis Report

## Executive Summary

Unsupervised clustering was applied to MobileNetV3 visual embeddings to test whether True Positives (TP) and False Positives (FP) form distinct visual manifolds without label supervision.

---

## 1. KMeans Clustering (k=2)

- **Overall Cluster Purity**: **73.7%**
- **Adjusted Rand Index (ARI)**: **-0.0229**
- **Normalized Mutual Information (NMI)**: **0.1919**
- **Silhouette Score**: **0.0808**

### Cluster Breakdown Table

| Cluster ID | Total Candidates | True Positives (TP) | False Positives (FP) | Dominant Class Purity |
| :---: | :---: | :---: | :---: | :---: |
| **Cluster 0** | 46 | 0 | 46 | **100.0%** (FP Dominant) |
| **Cluster 1** | 106 | 40 | 66 | **62.3%** (FP Dominant) |

---

## 2. DBSCAN Clustering

- **EPS**: 15.0 | **Min Samples**: 3
- **Clusters Discovered**: 25 (including noise cluster `-1`)
- **Overall Cluster Purity**: **96.7%**

### Cluster Breakdown Table

| Cluster ID | Total Candidates | True Positives (TP) | False Positives (FP) | Dominant Class Purity |
| :---: | :---: | :---: | :---: | :---: |
| **Noise Cluster** | 58 | 5 | 53 | **91.4%** |
| **Cluster 0** | 3 | 3 | 0 | **100.0%** |
| **Cluster 1** | 3 | 3 | 0 | **100.0%** |
| **Cluster 2** | 3 | 3 | 0 | **100.0%** |
| **Cluster 3** | 3 | 0 | 3 | **100.0%** |
| **Cluster 4** | 3 | 3 | 0 | **100.0%** |
| **Cluster 5** | 3 | 3 | 0 | **100.0%** |
| **Cluster 6** | 3 | 0 | 3 | **100.0%** |
| **Cluster 7** | 3 | 3 | 0 | **100.0%** |
| **Cluster 8** | 3 | 3 | 0 | **100.0%** |
| **Cluster 9** | 6 | 6 | 0 | **100.0%** |
| **Cluster 10** | 3 | 3 | 0 | **100.0%** |
| **Cluster 11** | 5 | 5 | 0 | **100.0%** |
| **Cluster 12** | 6 | 0 | 6 | **100.0%** |
| **Cluster 13** | 3 | 0 | 3 | **100.0%** |
| **Cluster 14** | 5 | 0 | 5 | **100.0%** |
| **Cluster 15** | 3 | 0 | 3 | **100.0%** |
| **Cluster 16** | 3 | 0 | 3 | **100.0%** |
| **Cluster 17** | 6 | 0 | 6 | **100.0%** |
| **Cluster 18** | 6 | 0 | 6 | **100.0%** |
| **Cluster 19** | 6 | 0 | 6 | **100.0%** |
| **Cluster 20** | 4 | 0 | 4 | **100.0%** |
| **Cluster 21** | 5 | 0 | 5 | **100.0%** |
| **Cluster 22** | 3 | 0 | 3 | **100.0%** |
| **Cluster 23** | 3 | 0 | 3 | **100.0%** |

---

## Key Clustering Insights

1. **High Visual Overlap**: Unsupervised clustering shows **significant overlap** between small entity crops (mobs/players) and complex map scenery props (flowers, wooden boxes, map borders).
2. **Scenery Diversity**: Scenery false positives do not form a single homogeneous cluster; rather, flowers, roofs, and wall fragments scatter across the feature space, wrapping around true character entity embeddings.
