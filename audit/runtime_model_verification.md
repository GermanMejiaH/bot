# Phase 4D.0 — Runtime Model Loading & Verification Report

## Executive Summary

- **Model Loading Status**: **PASSED**
- **Active Model Path**: `models\entity_classifier.pkl`
- **Backup Model Path**: `models\entity_classifier_phase4c_bak.pkl`
- **Backup Accidentally Loaded**: **False** (Zero Backup Leakage Confirmed)
- **Decision Threshold (τ)**: **0.50**
- **Feature Extractor**: **MobileNetV3 Small (576-d)**
- **Classifier Pipeline**: **Logistic Regression (L2)** (L2 Regularized, C=0.1)
- **Master Training Pool Size**: **275** candidates
- **Frozen Benchmark Size**: **72** candidates

## 1. Frozen Benchmark Verification Metrics

- **Frozen Benchmark Accuracy**: **93.1%**
- **Frozen Benchmark Precision**: **76.9%**
- **Frozen Benchmark Recall**: **83.3%**
- **Frozen Benchmark F1 Score**: **0.8000**
- **Frozen Benchmark PR-AUC / ROC-AUC**: **0.9268** / **0.9750**

## 2. Verification Conclusion

The Phase 4C.6 retrained model `models/entity_classifier.pkl` loads seamlessly into `VisualCandidateClassifier` at runtime. All metadata metrics align with the Phase 4C.6 training run.
