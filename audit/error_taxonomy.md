# Phase 4C.5 — Forensic Error Taxonomy & Cluster Audit

## Executive Summary

A total of **26 classification errors** (23 False Positives, 3 False Negatives) out of 144 stress candidate samples were subjected to forensic visual auditing.

### Root Cause Attribution Summary

- **Detector Generator Noise**: **10 errors (38.5%)** — Should be filtered *earlier* in `CharacterDetector` before reaching visual classification.
- **Classifier Model Confusion**: **13 errors (50.0%)** — False Positives caused by feature space overlap with green PM grid tiles and vertical scenery props, fixable via training data expansion.
- **Classifier Under-Representation**: **3 errors (11.5%)** — False Negatives caused by occluded or dark character sprites, fixable via targeted positive training data augmentation.

## 1. Visual Cluster Summary Table

| Cluster Key | Cluster Name | Count | Share (%) | Error Type | Root Cause Category | Primary Remediation Path |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| `tiny_specular_noise` | **Tiny Specular & Floor Noise Artifacts** | **8** | 30.8% | False Positive | `Detector Noise` | **Early Filtering in CharacterDetector** |
| `tactical_grid_overlay` | **Tactical Grid & PM Movement Overlays** | **7** | 26.9% | False Positive | `Classifier Confusion` | **Additional Classifier Training Data** |
| `scenery_prop_confusion` | **Vertical Scenery Prop Confusion** | **6** | 23.1% | False Positive | `Classifier Confusion` | **Additional Classifier Training Data** |
| `occluded_entity_fn` | **Occluded / Low-Contrast Entity Misses (FN)** | **3** | 11.5% | False Negative | `Classifier Under-Representation` | **Additional Classifier Training Data** |
| `ui_hud_fragment` | **UI Action Bar / HUD Edge Leakage** | **2** | 7.7% | False Positive | `Detector Noise` | **Early Filtering in CharacterDetector** |

## 2. Detailed Cluster Deep-Dive

### 2.1 Tiny Specular & Floor Noise Artifacts (`tiny_specular_noise`)

- **Sample Count**: **8** (30.8% of total errors)
- **Root Cause Category**: `Detector Noise`
- **Visual Description**: Micro-crops (sub-25px, area < 500 px) from bright floor tiles, specular glints, or foliage edges.
- **Primary Remediation**: **Early Filtering in CharacterDetector**
- **Actionable Strategy**: Implement minimum candidate bounding box dimensions (w >= 18, h >= 24) and area threshold (area >= 400 px).

| Candidate ID | Method | Folder | Frame ID | Bounding Box [x, y, w, h] | Crop Image |
| :---: | :---: | :---: | :---: | :---: | :--- |
| 101 | `hsv` | `exploration` | `frame_20260907_221504_300832_audit_0050` | `[520, 508, 16, 16]` | [`candidate_0101_hsv.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_0101_hsv.png) |
| 103 | `hsv` | `exploration` | `frame_20260907_221504_300832_audit_0050` | `[520, 291, 16, 17]` | [`candidate_0103_hsv.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_0103_hsv.png) |
| 182 | `contour` | `exploration` | `frame_20260907_221604_944340_audit_0126` | `[941, 193, 22, 23]` | [`candidate_0182_contour.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_0182_contour.png) |
| 183 | `contour` | `exploration` | `frame_20260907_221604_944340_audit_0126` | `[1000, 189, 27, 24]` | [`candidate_0183_contour.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_0183_contour.png) |
| 188 | `contour` | `exploration` | `frame_20260907_221607_111215_audit_0128` | `[941, 193, 22, 23]` | [`candidate_0188_contour.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_0188_contour.png) |
| 189 | `contour` | `exploration` | `frame_20260907_221607_111215_audit_0128` | `[1000, 189, 27, 24]` | [`candidate_0189_contour.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_0189_contour.png) |
| 268 | `hsv` | `exploration` | `frame_20260907_221752_001037_audit_0019` | `[931, 621, 15, 20]` | [`candidate_0268_hsv.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_0268_hsv.png) |
| 294 | `contour` | `exploration` | `frame_20260907_222057_827694_audit_0140` | `[781, 246, 18, 16]` | [`candidate_0294_contour.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_0294_contour.png) |

---

### 2.2 Tactical Grid & PM Movement Overlays (`tactical_grid_overlay`)

- **Sample Count**: **7** (26.9% of total errors)
- **Root Cause Category**: `Classifier Confusion`
- **Visual Description**: Translucent green PM movement grid cells or red combat range highlights on dark ground terrain.
- **Primary Remediation**: **Additional Classifier Training Data**
- **Actionable Strategy**: Augment training set with saturated green/red grid cell negative crops and adjust HSV saturation upper bounds.

| Candidate ID | Method | Folder | Frame ID | Bounding Box [x, y, w, h] | Crop Image |
| :---: | :---: | :---: | :---: | :---: | :--- |
| 314 | `contour` | `combat` | `frame_20260907_210147_065628_frame_000001` | `[295, 405, 28, 44]` | [`candidate_0314_contour.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_0314_contour.png) |
| 327 | `contour` | `combat` | `frame_20260907_211452_499105_frame_000001` | `[295, 402, 28, 47]` | [`candidate_0327_contour.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_0327_contour.png) |
| 423 | `contour` | `combat` | `frame_20260907_211559_236099_frame_000061` | `[385, 547, 32, 60]` | [`candidate_0423_contour.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_0423_contour.png) |
| 549 | `hsv` | `combat` | `frame_20260907_214902_361806_frame_000019` | `[403, 757, 24, 37]` | [`candidate_0549_hsv.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_0549_hsv.png) |
| 576 | `hsv` | `combat` | `frame_20260907_214911_263898_frame_000025` | `[403, 761, 21, 33]` | [`candidate_0576_hsv.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_0576_hsv.png) |
| 586 | `hsv` | `combat` | `frame_20260907_214914_146079_frame_000027` | `[403, 761, 21, 33]` | [`candidate_0586_hsv.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_0586_hsv.png) |
| 848 | `hsv` | `combat` | `frame_20260907_221829_085639_audit_0045` | `[403, 761, 20, 33]` | [`candidate_0848_hsv.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_0848_hsv.png) |

---

### 2.3 Vertical Scenery Prop Confusion (`scenery_prop_confusion`)

- **Sample Count**: **6** (23.1% of total errors)
- **Root Cause Category**: `Classifier Confusion`
- **Visual Description**: Tall static scenery props (statues, lamp posts, wall columns, flags, crates) sharing vertical aspect ratio and edge features with characters.
- **Primary Remediation**: **Additional Classifier Training Data**
- **Actionable Strategy**: Expand training set negative samples with diverse map background props and static scenery assets.

| Candidate ID | Method | Folder | Frame ID | Bounding Box [x, y, w, h] | Crop Image |
| :---: | :---: | :---: | :---: | :---: | :--- |
| 69 | `combat_base` | `exploration` | `frame_20260907_221442_574990_audit_0011` | `[834, 386, 16, 48]` | [`candidate_0069_combat_base.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_0069_combat_base.png) |
| 323 | `combat_base` | `combat` | `frame_20260907_210147_065628_frame_000001` | `[748, 292, 31, 100]` | [`candidate_0323_combat_base.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_0323_combat_base.png) |
| 366 | `combat_base` | `combat` | `frame_20260907_211520_707005_frame_000023` | `[703, 433, 32, 76]` | [`candidate_0366_combat_base.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_0366_combat_base.png) |
| 1039 | `combat_base` | `combat` | `frame_20260907_221956_041076_audit_0102` | `[867, 346, 53, 127]` | [`candidate_1039_combat_base.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_1039_combat_base.png) |
| 1111 | `combat_base` | `combat` | `frame_20260907_222032_592893_audit_0124` | `[506, 501, 67, 168]` | [`candidate_1111_combat_base.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_1111_combat_base.png) |
| 1115 | `combat_base` | `combat` | `frame_20260907_222032_592893_audit_0124` | `[399, 256, 22, 81]` | [`candidate_1115_combat_base.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_1115_combat_base.png) |

---

### 2.4 Occluded / Low-Contrast Entity Misses (FN) (`occluded_entity_fn`)

- **Sample Count**: **3** (11.5% of total errors)
- **Root Cause Category**: `Classifier Under-Representation`
- **Visual Description**: True character/monster entities partially hidden behind obstacles or presenting low color contrast against map terrain.
- **Primary Remediation**: **Additional Classifier Training Data**
- **Actionable Strategy**: Collect and add occluded, dark, and edge-of-screen character entity positive crops to the training split.

| Candidate ID | Method | Folder | Frame ID | Bounding Box [x, y, w, h] | Crop Image |
| :---: | :---: | :---: | :---: | :---: | :--- |
| 59 | `contour` | `exploration` | `frame_20260907_221026_763785_audit_synth_0001` | `[570, 320, 61, 61]` | [`candidate_0059_contour.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_0059_contour.png) |
| 60 | `hsv` | `exploration` | `frame_20260907_221026_763785_audit_synth_0001` | `[673, 420, 55, 93]` | [`candidate_0060_hsv.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_0060_hsv.png) |
| 1032 | `combat_base` | `combat` | `frame_20260907_221956_041076_audit_0102` | `[686, 594, 22, 86]` | [`candidate_1032_combat_base.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_1032_combat_base.png) |

---

### 2.5 UI Action Bar / HUD Edge Leakage (`ui_hud_fragment`)

- **Sample Count**: **2** (7.7% of total errors)
- **Root Cause Category**: `Detector Noise`
- **Visual Description**: Wide rectangular crops along the bottom screen boundary (y >= 600) capturing action bar icons or spell HUD borders.
- **Primary Remediation**: **Early Filtering in CharacterDetector**
- **Actionable Strategy**: Tighten MapROIExtractor bottom margin (y_max <= 600) to strictly exclude HUD interface boundaries.

| Candidate ID | Method | Folder | Frame ID | Bounding Box [x, y, w, h] | Crop Image |
| :---: | :---: | :---: | :---: | :---: | :--- |
| 286 | `hsv` | `exploration` | `frame_20260907_222051_557964_audit_0136` | `[0, 607, 89, 47]` | [`candidate_0286_hsv.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_0286_hsv.png) |
| 431 | `hsv` | `combat` | `frame_20260907_211613_390979_frame_000075` | `[0, 607, 86, 47]` | [`candidate_0431_hsv.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_0431_hsv.png) |

---

