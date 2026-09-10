# Phase 4C.5 — Stress Holdout Error Forensic Review

**Total Stress Candidates Audited**: 144 | **Total Classification Errors**: 26

## 1. Error Summary Table

| Candidate ID | Method | Folder | Ground Truth | Prediction | Error Type | P(Entity) | Filename |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 59 | `contour` | `exploration` | `1` (1) | `0` | **False Negative** | **0.4777** | [`candidate_0059_contour.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_0059_contour.png) |
| 60 | `hsv` | `exploration` | `1` (1) | `0` | **False Negative** | **0.0250** | [`candidate_0060_hsv.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_0060_hsv.png) |
| 69 | `combat_base` | `exploration` | `0` (0) | `1` | **False Positive** | **0.5251** | [`candidate_0069_combat_base.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_0069_combat_base.png) |
| 101 | `hsv` | `exploration` | `0` (0) | `1` | **False Positive** | **0.9484** | [`candidate_0101_hsv.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_0101_hsv.png) |
| 103 | `hsv` | `exploration` | `0` (0) | `1` | **False Positive** | **0.6761** | [`candidate_0103_hsv.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_0103_hsv.png) |
| 182 | `contour` | `exploration` | `0` (0) | `1` | **False Positive** | **0.9839** | [`candidate_0182_contour.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_0182_contour.png) |
| 183 | `contour` | `exploration` | `0` (0) | `1` | **False Positive** | **0.9825** | [`candidate_0183_contour.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_0183_contour.png) |
| 188 | `contour` | `exploration` | `0` (0) | `1` | **False Positive** | **0.9839** | [`candidate_0188_contour.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_0188_contour.png) |
| 189 | `contour` | `exploration` | `0` (0) | `1` | **False Positive** | **0.9825** | [`candidate_0189_contour.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_0189_contour.png) |
| 268 | `hsv` | `exploration` | `0` (0) | `1` | **False Positive** | **0.7934** | [`candidate_0268_hsv.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_0268_hsv.png) |
| 286 | `hsv` | `exploration` | `0` (0) | `1` | **False Positive** | **0.5287** | [`candidate_0286_hsv.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_0286_hsv.png) |
| 294 | `contour` | `exploration` | `0` (0) | `1` | **False Positive** | **0.8576** | [`candidate_0294_contour.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_0294_contour.png) |
| 314 | `contour` | `combat` | `0` (0) | `1` | **False Positive** | **1.0000** | [`candidate_0314_contour.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_0314_contour.png) |
| 323 | `combat_base` | `combat` | `0` (0) | `1` | **False Positive** | **0.7749** | [`candidate_0323_combat_base.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_0323_combat_base.png) |
| 327 | `contour` | `combat` | `0` (0) | `1` | **False Positive** | **1.0000** | [`candidate_0327_contour.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_0327_contour.png) |
| 366 | `combat_base` | `combat` | `0` (0) | `1` | **False Positive** | **0.7334** | [`candidate_0366_combat_base.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_0366_combat_base.png) |
| 423 | `contour` | `combat` | `0` (0) | `1` | **False Positive** | **0.7348** | [`candidate_0423_contour.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_0423_contour.png) |
| 431 | `hsv` | `combat` | `0` (0) | `1` | **False Positive** | **0.7828** | [`candidate_0431_hsv.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_0431_hsv.png) |
| 549 | `hsv` | `combat` | `0` (0) | `1` | **False Positive** | **0.5140** | [`candidate_0549_hsv.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_0549_hsv.png) |
| 576 | `hsv` | `combat` | `0` (0) | `1` | **False Positive** | **0.7817** | [`candidate_0576_hsv.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_0576_hsv.png) |
| 586 | `hsv` | `combat` | `0` (0) | `1` | **False Positive** | **0.6570** | [`candidate_0586_hsv.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_0586_hsv.png) |
| 848 | `hsv` | `combat` | `0` (0) | `1` | **False Positive** | **0.6590** | [`candidate_0848_hsv.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_0848_hsv.png) |
| 1032 | `combat_base` | `combat` | `1` (1) | `0` | **False Negative** | **0.0755** | [`candidate_1032_combat_base.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_1032_combat_base.png) |
| 1039 | `combat_base` | `combat` | `0` (0) | `1` | **False Positive** | **0.9868** | [`candidate_1039_combat_base.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_1039_combat_base.png) |
| 1111 | `combat_base` | `combat` | `0` (0) | `1` | **False Positive** | **0.5453** | [`candidate_1111_combat_base.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_1111_combat_base.png) |
| 1115 | `combat_base` | `combat` | `0` (0) | `1` | **False Positive** | **0.9642** | [`candidate_1115_combat_base.png`](file:///C:/Users/Andres/Desktop/bot/audit/stress_candidates/candidate_1115_combat_base.png) |
