# Phase 4C — Offline Visual Classifier Generalization & Evaluation Report

## Executive Summary

Phase 4C conducted an **offline generalization study** evaluating a secondary visual classifier pipeline built on **MobileNetV3 Small (576-d) embeddings** across 152 manually audited candidate crops (40 True Positives, 112 False Positives) originating from 41 unique source frames.

To ensure rigorous validation without data leakage, candidates were split using an **80/20 Group-Aware Stratified Partition (`StratifiedGroupKFold` on `frame_id`)**, isolating 6 complete source frames (37 candidate crops) into a **Holdout Test Set** that was never seen during model training, feature scaling, or hyperparameter selection.

### Key Results

| Metric | Existing Heuristic Baseline | Visual Classifier (Holdout) | Net Improvement |
| :--- | :---: | :---: | :---: |
| **Precision** | 29.7% | **100.0%** | **+70.3%** |
| **Recall** | 100.0% | **100.0%** | 0.0% |
| **F1 Score** | 0.4583 | **1.0000** | **+0.5417** |
| **PR-AUC** | N/A | **1.0000** | N/A |
| **ROC-AUC** | N/A | **1.0000** | N/A |
| **Accuracy** | 29.7% | **100.0%** | **+70.3%** |

---

## 1. Experimental Setup & Split Composition

- **Total Dataset Size**: 152 candidate crops across 41 unique `frame_id` screenshots.
- **Class Ratio**: 40 True Positives (players, monsters, NPCs) vs 112 False Positives (flowers, roofs, tree trunks, movement cells).
- **Partitioning Method**: `StratifiedGroupKFold(n_splits=5, random_state=42)` grouped strictly on `frame_id`.
- **Training Partition (80%)**: 115 candidates (29 TPs, 86 FPs) across 35 source frames.
- **Holdout Test Partition (20%)**: 37 candidates (11 TPs, 26 FPs) across 6 isolated source frames (`audit_0005`, `audit_0006`, `audit_0007`, `audit_0049`, `audit_0109`, `audit_0147`).
- **Reproducibility**: Split manifest exported to [`models/entity_classifier_split.json`](file:///c:/Users/Andres/Desktop/bot/models/entity_classifier_split.json).

---

## 2. Model Selection & Cross-Validation (Training Set)

Model selection was conducted on `X_train` using 5-Fold `StratifiedGroupKFold` cross-validation:

| Candidate Model Architecture | Mean Training CV F1 Score | Status |
| :--- | :---: | :---: |
| **Logistic Regression (L2 Penalty, $C=0.1$)** | **0.9333** | **SELECTED** |
| **Random Forest (`max_depth=4`)** | 0.9164 | Candidate |
| **XGBoost (`max_depth=3`, `subsample=0.7`)** | 0.8929 | Candidate |

- **Selection Rationale**: L2-regularized Logistic Regression ($C=0.1$) achieved the highest mean CV F1 score while natively controlling high-dimensional overfitting ($D=576, N=115$) without discarding semantic feature topology.

---

## 3. Holdout Test Set Performance

### Comparative Evaluation (Baseline Heuristic vs Visual Classifier)

On the 37 unseen candidate crops in the holdout partition:
- The **Existing Heuristic Baseline** accepted all 37 candidates, yielding 11 True Positives and 26 False Positives (**29.7% Precision**).
- The **Visual Classifier Pipeline** correctly classified all 11 True Positives while eliminating all 26 False Positive scenery crops (**100.0% Precision**, **100.0% Recall**, **1.0000 F1 Score**, **1.0000 PR-AUC**).

### Holdout Confusion Matrix
$$\begin{pmatrix} \text{True Negatives (TN) = 26} & \text{False Positives (FP) = 0} \\ \text{False Negatives (FN) = 0} & \text{True Positives (TP) = 11} \end{pmatrix}$$

### Per-Detector Source Breakdown (Holdout)

| Candidate Generator Source | Holdout Samples | Holdout TPs | Baseline Precision | Classifier Precision | Classifier Recall | Classifier F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `contour` | 12 | 4 | 33.3% | **100.0%** | **100.0%** | **1.0000** |
| `hsv` | 13 | 4 | 30.8% | **100.0%** | **100.0%** | **1.0000** |
| `combat_base` | 12 | 3 | 25.0% | **100.0%** | **100.0%** | **1.0000** |

---

## 4. CPU Latency & Resource Footprint Benchmark

Tested on CPU using PyTorch MobileNetV3 Small tensor feature extraction:

| Benchmark Scenario | Candidate Batch Size | Mean Latency | Median (p50) | p95 Latency | Throughput | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Single Candidate Crop** | 1 | **12.71 ms** | 12.70 ms | 13.65 ms | 78.7 crops/sec | **PASS (< 20 ms)** |
| **Candidate Batch** | 8 | **36.44 ms** (4.55 ms/crop) | 33.72 ms | 45.46 ms | 219.5 crops/sec | **PASS** |

- **RAM Memory Footprint**: **58.49 MB** total resident memory (MobileNetV3 Small weights + Scikit-Learn L2 pipeline).

---

## 5. Forensic Error Analysis

- **Holdout Error Count**: **0 errors (0 FPs, 0 FNs)**.
- Detailed audit logs exported to [`audit/holdout_error_review.md`](file:///c:/Users/Andres/Desktop/bot/audit/holdout_error_review.md).
- Visual montage generated at `audit/holdout_errors_montage.png`.

---

## 6. Qualitative Engineering Assessment & Phase 4D Recommendation

### Engineering Findings
1. **Out-of-Sample Generalization**: MobileNetV3 embeddings + L2 Logistic Regression generalized to 100% precision and recall on 6 completely unseen source frames.
2. **False Positive Elimination**: The classifier eliminated 100% of surviving scenery false positives (roof tiles, map flowers, movement cells, wall fragments).
3. **Acceptable CPU Latency**: Single crop processing takes **12.71 ms**, and batching 8 candidate crops takes **36.44 ms** (4.55 ms/crop), well within DTA's perception budget.

### Recommendation for Phase 4D
- **PROCEED TO PHASE 4D PROTOTYPE INTEGRATION**:
  - Integrate `VisualCandidateClassifier` as an optional secondary post-NMS candidate filter stage in `CharacterDetector` / `FramePipeline`.
  - Maintain `enable_visual_classifier: bool = False` as default setting to ensure zero breaking changes.
