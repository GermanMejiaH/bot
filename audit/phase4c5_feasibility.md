# Phase 4C.5 — Stress Holdout Validation Feasibility Audit

## Executive Summary

A technical feasibility audit was conducted to verify whether a valid, independent, disjoint out-of-sample candidate dataset exists for Phase 4C.5 stress testing.

- **Phase 4C Dataset**: 152 manually labeled candidate crops (40 True Positives, 112 False Positives) across 41 source frames (`models/entity_classifier_split.json`).
- **Disjoint Labelled Candidates Available**: **0 candidates**.
- **Feasibility Verdict**: ⚠️ **DATA COLLECTION REQUIRED**

---

## 1. Candidate Repository Inventory

A workspace-wide scan of all candidate repositories yielded the following inventory:

| Repository Location | Item Count | Asset Type | Phase 4C Overlap Status |
| :--- | :---: | :--- | :--- |
| `audit/review/accepted/*.png` | 152 | Labeled Candidate Crops | **100% Used in Phase 4C** (115 Train, 37 Holdout) |
| `audit/review/accepted/*.json` | 152 | Candidate Sidecar Metadata | **100% Used in Phase 4C** |
| `audit/manual_labels.xlsx` | 152 | Ground Truth Labels | **100% Used in Phase 4C** |
| `audit/crops/*.png` | 422 | Detector Crops (`accepted` & `rejected`) | 152 accepted (used in 4C), 270 rejected (unlabelled FPs) |
| `dataset/exploration/*.png` | 86 | Full Exploration Screenshots | Unprocessed / Unextracted Frames |
| `dataset/combat/*.png` | 99 | Full Combat Screenshots | Unprocessed / Unextracted Frames |
| `dataset/raw/*.png` | 185 | Full Capture Screenshots | Unprocessed / Unextracted Frames |

---

## 2. Overlap & Disjoint Stress Candidate Analysis

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 4C.5 STRESS CANDIDATE POOL                     │
├──────────────────────────────────────┬─────────┬────────────────────────┤
│ Category                             │ Count   │ Status                 │
├──────────────────────────────────────┼─────────┼────────────────────────┤
│ Total Labelled Candidates            │ 152     │ 100% Used in Phase 4C  │
│ Labelled Disjoint Candidates         │ 0       │ Insufficient (< 100)   │
│ Unlabelled Rejected Crops (audit/)   │ 270     │ FPs only (No TPs)      │
│ Unprocessed Frame Pool (dataset/)    │ 370     │ Requires Extraction    │
└──────────────────────────────────────┴─────────┴────────────────────────┘
```

1. **Labelled Candidate Overlap**: Every single manually labeled candidate crop in `audit/manual_labels.xlsx` (152 total) was consumed by the Phase 4C 80/20 `StratifiedGroupKFold` split (115 in `X_train`, 37 in `X_holdout`).
2. **Unlabelled Crop Pool**: The remaining 270 crop files in `audit/crops/` consist exclusively of candidate detections rejected during perception stage filtering (`entity_XXXX_rejected.png`). While these provide false-positive scenery crops, they contain zero true-positive character entity candidates.
3. **Unprocessed Frames**: 370 full PNG screenshots exist in `dataset/` (`exploration`, `combat`, `raw`). Running candidate extraction across these frames will yield $\approx 800–1200$ candidate crops. However, these candidates must be annotated with ground-truth labels before out-of-sample generalization metrics (Precision, Recall, F1, PR-AUC) can be computed.

---

## 3. Recommendation & Next Steps

### Recommendation: **DATA COLLECTION REQUIRED**

Phase 4C.5 model evaluation cannot proceed immediately because fewer than 100 completely new, manually labeled candidates exist outside the Phase 4C split.

### Recommended Next Steps
1. **Execute Candidate Extraction Pipeline**: Run `CharacterDetector` across the 86 exploration frames in `dataset/exploration/` and 99 combat frames in `dataset/combat/` to generate a candidate pool of $\approx 300-500$ crops.
2. **Manual Labeling**: Annotate $\ge 150$ new candidate crops into `audit/manual_labels_stress.xlsx` with ground-truth entity labels (`player`, `monster`, `npc` vs scenery/props).
3. **Execute Phase 4C.5 Stress Evaluation**: Once `manual_labels_stress.xlsx` contains $\ge 100$ new disjoint candidates, execute Phase 4C.5 evaluation on `models/entity_classifier.pkl` unchanged.
