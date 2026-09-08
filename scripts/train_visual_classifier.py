"""Phase 4C — Train Visual Classifier & Evaluate Holdout Set."""

# ruff: noqa: N806, N803



import datetime
import glob
import json
import os
import pickle
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from dta.perception.visual_classifier import PipelineWrapper, VisualCandidateClassifier


def load_audited_dataset() -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    """Load ground truth candidate dataset from manual_labels.xlsx and sidecar JSONs."""
    excel_path = os.path.join("audit", "manual_labels.xlsx")
    if not os.path.isfile(excel_path):
        excel_path = os.path.join("audit", "manual_labels_v2.xlsx")

    wb = openpyxl.load_workbook(excel_path)
    sheet = wb.active

    rows = list(sheet.iter_rows(values_only=True))
    header = [str(c).lower() if c else "" for c in rows[0]]

    cid_idx = header.index("candidate_id") if "candidate_id" in header else 0
    fn_idx = header.index("filename") if "filename" in header else 1
    method_idx = header.index("method") if "method" in header else 2
    label_idx = header.index("label") if "label" in header else 4

    # Build sidecar lookup table for frame_id
    sidecars = glob.glob(os.path.join("audit", "review", "accepted", "*.json"))
    sidecar_lookup: dict[tuple[int, str], str] = {}
    for sc in sidecars:
        try:
            with open(sc, encoding="utf-8") as f:
                data = json.load(f)
            c_id = int(data.get("candidate_id", 0))
            m_str = str(data.get("method", "")).lower()
            fid = str(data.get("frame_id", "unknown"))
            sidecar_lookup[(c_id, m_str)] = fid
        except Exception:
            pass

    tp_labels = {"player", "monster", "npc"}

    dataset: list[dict[str, Any]] = []

    for r in rows[1:]:
        if not r or r[cid_idx] is None:
            continue
        c_id = int(r[cid_idx])
        fname = str(r[fn_idx])
        method = str(r[method_idx]).lower()
        lbl = str(r[label_idx]).strip().lower()

        # Class ground truth
        is_tp = 1 if lbl in tp_labels else 0

        # Lookup frame_id
        frame_id = sidecar_lookup.get((c_id, method), "unknown_frame")

        # Resolve crop image path
        crop_path = os.path.join("audit", "review", "accepted", fname)
        if not os.path.isfile(crop_path):
            crop_path = os.path.join("audit", "crops", fname)

        dataset.append({
            "candidate_id": c_id,
            "filename": fname,
            "method": method,
            "label": lbl,
            "is_tp": is_tp,
            "frame_id": frame_id,
            "crop_path": crop_path,
        })

    print(f"Loaded dataset: {len(dataset)} total candidates across {len(set(d['frame_id'] for d in dataset))} frames.")
    tps = sum(d["is_tp"] for d in dataset)
    fps = len(dataset) - tps
    print(f"Class Breakdown: {tps} True Positives, {fps} False Positives.")

    return dataset, np.array([d["is_tp"] for d in dataset]), np.array([d["frame_id"] for d in dataset]), np.array([d["candidate_id"] for d in dataset])


def main() -> None:
    """Execute Phase 4C training, holdout evaluation, baseline comparison, and artifact generation."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("audit", exist_ok=True)

    dataset, y, groups, cids = load_audited_dataset()

    # 1. Stratified Group 80/20 Train/Holdout Split
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, holdout_idx = next(sgkf.split(dataset, y, groups))

    train_data = [dataset[i] for i in train_idx]
    holdout_data = [dataset[i] for i in holdout_idx]

    y_train = y[train_idx]
    y_holdout = y[holdout_idx]
    groups_train = groups[train_idx]

    print("\n--- 80/20 Train/Holdout Split ---")
    print(f"Train Set: {len(train_data)} candidates ({sum(y_train)} TP, {len(y_train)-sum(y_train)} FP) across {len(set(groups_train))} frames.")
    print(f"Holdout Set: {len(holdout_data)} candidates ({sum(y_holdout)} TP, {len(y_holdout)-sum(y_holdout)} FP) across {len(set(groups[holdout_idx]))} frames.")

    # Export split manifest
    split_manifest = {
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "total_samples": len(dataset),
        "train_samples": len(train_data),
        "holdout_samples": len(holdout_data),
        "train_candidate_ids": [d["candidate_id"] for d in train_data],
        "holdout_candidate_ids": [d["candidate_id"] for d in holdout_data],
        "train_frame_ids": sorted(list(set(groups_train))),
        "holdout_frame_ids": sorted(list(set(groups[holdout_idx]))),
    }
    with open(os.path.join("models", "entity_classifier_split.json"), "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2)

    # 2. Feature Extraction (MobileNetV3 576-d)
    extractor = VisualCandidateClassifier(model_path="non_existent.pkl")

    print("\nExtracting MobileNetV3 features for training set...")
    train_crops = [cv2.imread(d["crop_path"]) for d in train_data]
    X_train_raw = extractor.extract_batch_features(train_crops)

    print("Extracting MobileNetV3 features for holdout set...")
    holdout_crops = [cv2.imread(d["crop_path"]) for d in holdout_data]
    X_holdout_raw = extractor.extract_batch_features(holdout_crops)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)

    # 3. Model Selection via 5-Fold StratifiedGroupKFold on X_train
    print("\n--- Inner 5-Fold StratifiedGroupKFold Model Selection (X_train) ---")
    cv_splitter = StratifiedGroupKFold(n_splits=5)

    classifiers = {
        "Logistic Regression (L2)": lambda: LogisticRegression(C=0.1, solver="liblinear", random_state=42),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42),
        "XGBoost": lambda: xgb.XGBClassifier(n_estimators=50, max_depth=3, subsample=0.7, learning_rate=0.05, random_state=42),
    }

    cv_results = {}
    for name, clf_fn in classifiers.items():
        f1_scores = []
        for tr_k, val_k in cv_splitter.split(X_train, y_train, groups_train):
            clf_k = clf_fn()
            clf_k.fit(X_train[tr_k], y_train[tr_k])
            preds_k = clf_k.predict(X_train[val_k])
            f1_scores.append(f1_score(y_train[val_k], preds_k, zero_division=0))
        mean_f1 = float(np.mean(f1_scores))
        cv_results[name] = mean_f1
        print(f"  {name}: Mean CV F1 = {mean_f1:.4f}")

    best_name = max(cv_results, key=cv_results.get)
    print(f"\nSelected Model: '{best_name}' (CV F1 = {cv_results[best_name]:.4f})")

    # Fit final pipeline (Scaler + Classifier)
    final_clf = classifiers[best_name]()
    final_clf.fit(X_train, y_train)

    pipeline = PipelineWrapper(scaler, final_clf)

    # Export model to models/entity_classifier.pkl
    model_pkl_path = os.path.join("models", "entity_classifier.pkl")
    with open(model_pkl_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Exported trained model pipeline to '{model_pkl_path}'.")

    # 4. Single-Pass Evaluation on 20% Holdout Test Set
    holdout_probas = pipeline.predict_proba(X_holdout_raw)[:, 1]
    holdout_preds = (holdout_probas >= 0.50).astype(int)

    # Heuristic Baseline (all candidates accepted by perception detector = 1)
    baseline_preds = np.ones(len(y_holdout), dtype=int)

    # Metrics computation
    base_prec = float(precision_score(y_holdout, baseline_preds, zero_division=0))
    base_rec = float(recall_score(y_holdout, baseline_preds, zero_division=0))
    base_f1 = float(f1_score(y_holdout, baseline_preds, zero_division=0))

    clf_acc = float(accuracy_score(y_holdout, holdout_preds))
    clf_prec = float(precision_score(y_holdout, holdout_preds, zero_division=0))
    clf_rec = float(recall_score(y_holdout, holdout_preds, zero_division=0))
    clf_f1 = float(f1_score(y_holdout, holdout_preds, zero_division=0))

    try:
        clf_pr_auc = float(average_precision_score(y_holdout, holdout_probas))
    except Exception:
        clf_pr_auc = 0.0

    try:
        clf_roc_auc = float(roc_auc_score(y_holdout, holdout_probas))
    except Exception:
        clf_roc_auc = 0.0

    cm = confusion_matrix(y_holdout, holdout_preds).tolist()

    print("\n==========================================")
    print("--- 20% HOLDOUT TEST SET PERFORMANCE ---")
    print("==========================================")
    print(f"Baseline Heuristic : Precision={base_prec*100:.1f}%, Recall={base_rec*100:.1f}%, F1={base_f1:.4f}")
    print(f"Visual Classifier  : Precision={clf_prec*100:.1f}%, Recall={clf_rec*100:.1f}%, F1={clf_f1:.4f}")
    print(f"Visual Classifier  : PR-AUC={clf_pr_auc:.4f}, ROC-AUC={clf_roc_auc:.4f}, Accuracy={clf_acc*100:.1f}%")
    print(f"Confusion Matrix   : TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")

    # Per-Detector Breakdown on Holdout
    detector_methods = ["contour", "hsv", "combat_base"]
    per_detector_report = {}

    for m in detector_methods:
        m_mask = np.array([d["method"] == m for d in holdout_data])
        if not np.any(m_mask):
            continue

        y_h_m = y_holdout[m_mask]
        preds_b_m = baseline_preds[m_mask]
        preds_c_m = holdout_preds[m_mask]

        b_prec_m = float(precision_score(y_h_m, preds_b_m, zero_division=0))
        b_rec_m = float(recall_score(y_h_m, preds_b_m, zero_division=0))
        b_f1_m = float(f1_score(y_h_m, preds_b_m, zero_division=0))

        c_prec_m = float(precision_score(y_h_m, preds_c_m, zero_division=0))
        c_rec_m = float(recall_score(y_h_m, preds_c_m, zero_division=0))
        c_f1_m = float(f1_score(y_h_m, preds_c_m, zero_division=0))

        per_detector_report[m] = {
            "sample_count": int(np.sum(m_mask)),
            "true_positives_count": int(np.sum(y_h_m)),
            "baseline": {"precision": b_prec_m, "recall": b_rec_m, "f1": b_f1_m},
            "classifier": {"precision": c_prec_m, "recall": c_rec_m, "f1": c_f1_m},
        }

    # Export metadata
    metadata = {
        "training_date": datetime.datetime.now(datetime.UTC).isoformat(),
        "feature_extractor": "MobileNetV3 Small (576-d)",
        "selected_model": best_name,
        "dataset_size": len(dataset),
        "train_size": len(train_data),
        "holdout_size": len(holdout_data),
        "cv_results": cv_results,
        "holdout_evaluation": {
            "baseline_heuristic": {"precision": base_prec, "recall": base_rec, "f1": base_f1},
            "visual_classifier": {
                "accuracy": clf_acc,
                "precision": clf_prec,
                "recall": clf_rec,
                "f1": clf_f1,
                "pr_auc": clf_pr_auc,
                "roc_auc": clf_roc_auc,
                "confusion_matrix": cm,
            },
            "per_detector_breakdown": per_detector_report,
        },
    }

    meta_json_path = os.path.join("models", "entity_classifier_metadata.json")
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 5. Forensic Error Analysis
    error_items = []
    error_crops = []

    for idx, d in enumerate(holdout_data):
        y_true = int(y_holdout[idx])
        y_pred = int(holdout_preds[idx])
        proba = float(holdout_probas[idx])

        if y_true != y_pred:
            err_type = "False Positive" if y_pred == 1 else "False Negative"
            error_items.append({
                "candidate_id": d["candidate_id"],
                "filename": d["filename"],
                "method": d["method"],
                "frame_id": d["frame_id"],
                "ground_truth": d["label"],
                "error_type": err_type,
                "probability": round(proba, 4),
            })
            error_crops.append((d, err_type, proba))

    # Generate audit/holdout_error_review.md
    err_md = "# Phase 4C — Holdout Error Forensic Review\n\n"
    err_md += f"**Total Holdout Samples**: {len(holdout_data)} | **Total Misclassifications**: {len(error_items)}\n\n"
    if not error_items:
        err_md += "🎉 **Zero errors detected on the Holdout Test Set! Perfect 100% classification accuracy.**\n"
    else:
        err_md += "| Candidate ID | Method | Frame ID | Ground Truth Label | Error Type | P(Entity) |\n"
        err_md += "| :---: | :---: | :---: | :---: | :---: | :---: |\n"
        for item in error_items:
            err_md += f"| {item['candidate_id']} | {item['method']} | {item['frame_id']} | {item['ground_truth']} | **{item['error_type']}** | {item['probability']:.4f} |\n"

    with open(os.path.join("audit", "holdout_error_review.md"), "w", encoding="utf-8") as f:
        f.write(err_md)

    # Export error montage image
    if error_crops:
        n_err = len(error_crops)
        fig, axes = plt.subplots(1, n_err, figsize=(n_err * 3, 3))
        if n_err == 1:
            axes = [axes]
        for ax, (d, err_type, proba) in zip(axes, error_crops, strict=False):
            crop = cv2.imread(d["crop_path"])
            if crop is not None:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                ax.imshow(crop_rgb)
            ax.set_title(f"{err_type}\n{d['label']} ({proba:.2f})", fontsize=9)
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join("audit", "holdout_errors_montage.png"))
        plt.close()
    else:
        # Create empty placeholder montage
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "Zero Holdout Errors", ha="center", va="center", fontsize=14, color="green")
        ax.axis("off")
        plt.savefig(os.path.join("audit", "holdout_errors_montage.png"))
        plt.close()

    print("\nPhase 4C Training & Evaluation Complete! Artifacts exported to 'models/' and 'audit/'.")


if __name__ == "__main__":
    main()
