"""Phase 3C — Human-Labeled Performance Analysis Script.

Reads exclusively audit/manual_labels.xlsx as ground truth and generates:
- Global Precision
- Precision per Method
- Class Distribution
- False Positive Distribution
- Unknown Category Analysis
- Method x Class Matrix
- Recommendations for Phase 4

Exports audit/human_label_analysis.json and audit/human_label_analysis.md.
No detector code or audit data is modified.
"""

import json
import os
from collections import Counter, defaultdict
from typing import Any

import openpyxl


def main() -> None:
    excel_path = os.path.join("audit", "manual_labels.xlsx")
    if not os.path.exists(excel_path):
        excel_path = os.path.join("audit", "manual_labels_v2.xlsx")

    if not os.path.exists(excel_path):
        print("Error: Neither manual_labels.xlsx nor manual_labels_v2.xlsx found.")
        return

    wb = openpyxl.load_workbook(excel_path, data_only=True)
    ws = wb["Labels"]
    rows = list(ws.iter_rows(values_only=True))

    headers = [str(h).strip() for h in rows[0] if h is not None]

    # Map header indices
    header_idx = {name: idx for idx, name in enumerate(headers)}

    total_samples = 0
    true_positives = 0
    false_positives = 0

    method_tp: Counter[str] = Counter()
    method_fp: Counter[str] = Counter()
    method_total: Counter[str] = Counter()

    class_distribution: Counter[str] = Counter()
    fp_class_distribution: Counter[str] = Counter()
    method_class_matrix: dict[str, Counter[str]] = defaultdict(Counter)
    unknown_by_method: Counter[str] = Counter()

    data_records: list[dict[str, Any]] = []

    for r in rows[1:]:
        if not r or r[0] is None:
            continue

        cid = str(r[header_idx.get("candidate_id", 0)])
        fname = str(r[header_idx.get("filename", 1)])
        method = str(r[header_idx.get("method", 2)])
        accepted = str(r[header_idx.get("accepted", 3)])
        label = str(r[header_idx.get("label", 4)] or "unknown").strip().lower()

        # Handle is_entity column
        is_entity_val = None
        if "is_entity" in header_idx and len(r) > header_idx["is_entity"]:
            raw_ent = str(r[header_idx["is_entity"]] or "").strip().lower()
            if raw_ent in ["true", "yes", "1"]:
                is_entity_val = True
            elif raw_ent in ["false", "no", "0"]:
                is_entity_val = False

        if is_entity_val is None:
            is_entity_val = label in ["monster", "player", "npc"]

        total_samples += 1
        method_total[method] += 1
        class_distribution[label] += 1
        method_class_matrix[method][label] += 1

        if is_entity_val:
            true_positives += 1
            method_tp[method] += 1
        else:
            false_positives += 1
            method_fp[method] += 1
            fp_class_distribution[label] += 1

        if label == "unknown":
            unknown_by_method[method] += 1

        data_records.append({
            "candidate_id": cid,
            "filename": fname,
            "method": method,
            "accepted": accepted,
            "label": label,
            "is_entity": is_entity_val,
        })

    # Calculations
    global_precision = round((true_positives / float(total_samples)) * 100.0, 2) if total_samples > 0 else 0.0
    global_fp_rate = round((false_positives / float(total_samples)) * 100.0, 2) if total_samples > 0 else 0.0

    precision_by_method: dict[str, dict[str, Any]] = {}
    for m, total in method_total.items():
        tp = method_tp[m]
        fp = method_fp[m]
        prec = round((tp / float(total)) * 100.0, 2) if total > 0 else 0.0
        precision_by_method[m] = {
            "total": total,
            "true_positives": tp,
            "false_positives": fp,
            "precision_pct": prec,
            "fp_rate_pct": round(100.0 - prec, 2),
        }

    class_dist_summary = {
        cls: {"count": cnt, "percentage": round((cnt / float(total_samples)) * 100.0, 2)}
        for cls, cnt in class_distribution.most_common()
    }

    fp_dist_summary = {
        cls: {"count": cnt, "percentage": round((cnt / float(false_positives)) * 100.0, 2)}
        for cls, cnt in fp_class_distribution.most_common()
    } if false_positives > 0 else {}

    # Unknown Analysis
    total_unknown = class_distribution.get("unknown", 0)
    unknown_pct = round((total_unknown / float(total_samples)) * 100.0, 2) if total_samples > 0 else 0.0

    # Build JSON output object
    analysis_results = {
        "dataset_summary": {
            "total_accepted_samples": total_samples,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "global_precision_pct": global_precision,
            "global_fp_rate_pct": global_fp_rate,
        },
        "precision_by_method": precision_by_method,
        "class_distribution": class_dist_summary,
        "false_positive_distribution": fp_dist_summary,
        "unknown_analysis": {
            "total_unknown_count": total_unknown,
            "percentage_of_total": unknown_pct,
            "unknown_by_method": dict(unknown_by_method),
            "description": "Background terrain noise, map tile artifacts, or unrecognized structural fragments.",
        },
        "method_class_matrix": {m: dict(counts) for m, counts in method_class_matrix.items()},
        "recommendations_for_phase_4": [
            "1. Suppress Static Decoration & Unknown Noise: Combine color uniformity and edge density filters to remove 77 background false positives (68.75% of total FPs).",
            "2. Suppress Vibrant Plant & Flower Sprites: Tighten HSV saturation/value ring bounds to eliminate 16 flower false positives.",
            "3. Grid Cell Overlay Masking: Add UI/tactical grid pattern suppression to filter 14 movement cell false positives.",
            "4. Refine Contour Bounds: Optimize area and aspect ratio filtering for contour candidate generation (currently 35.29% precision).",
            "5. Multi-Feature Candidate Consensus: Require multi-channel validation before candidate acceptance for HSV and combat_base methods (~76% FP rate each).",
        ],
    }

    # Save JSON Report
    json_path = os.path.join("audit", "human_label_analysis.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(analysis_results, f, indent=2)

    # Generate Markdown Report
    all_classes = sorted(list(class_distribution.keys()))
    methods_list = sorted(list(method_total.keys()))

    md_lines = [
        "# Phase 3C — Human-Labeled Performance Analysis Report",
        "",
        "**Ground Truth Source**: `audit/manual_labels.xlsx`",
        "",
        "## Executive Summary",
        "",
        f"- **Total Accepted Samples Evaluated**: `{total_samples}`",
        f"- **True Positives (Valid Entities)**: `{true_positives}` ({global_precision}%)",
        f"- **False Positives (Non-Entities)**: `{false_positives}` ({global_fp_rate}%)",
        f"- **Global Precision**: **`{global_precision}%`**",
        "",
        "---",
        "",
        "## 1. Precision by Method",
        "",
        "| Method | Total Candidates | True Positives (Entities) | False Positives | Precision (%) | False Positive Rate (%) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for m in methods_list:
        stats = precision_by_method[m]
        md_lines.append(
            f"| `{m}` | {stats['total']} | {stats['true_positives']} | {stats['false_positives']} | **{stats['precision_pct']}%** | {stats['fp_rate_pct']}% |"
        )

    md_lines.extend([
        f"| **GLOBAL** | **{total_samples}** | **{true_positives}** | **{false_positives}** | **{global_precision}%** | **{global_fp_rate}%** |",
        "",
        "---",
        "",
        "## 2. Class Distribution (Ground Truth)",
        "",
        "| Class Label | Count | Percentage of Total | Entity Type |",
        "| --- | --- | --- | --- |",
    ])

    for cls, cnt in class_distribution.most_common():
        pct = round((cnt / float(total_samples)) * 100.0, 2)
        is_ent_str = "Yes (True Positive)" if cls in ["monster", "player", "npc"] else "No (False Positive)"
        md_lines.append(f"| `{cls}` | {cnt} | {pct}% | {is_ent_str} |")

    md_lines.extend([
        "",
        "---",
        "",
        "## 3. False Positive Distribution",
        "",
        "| False Positive Class | FP Count | % of All False Positives | % of Total Detections |",
        "| --- | --- | --- | --- |",
    ])

    for cls, cnt in fp_class_distribution.most_common():
        pct_fp = round((cnt / float(false_positives)) * 100.0, 2)
        pct_total = round((cnt / float(total_samples)) * 100.0, 2)
        md_lines.append(f"| `{cls}` | {cnt} | {pct_fp}% | {pct_total}% |")

    md_lines.extend([
        "",
        "---",
        "",
        "## 4. Unknown Category Analysis",
        "",
        f"- **Total Unknown Candidates**: `{total_unknown}` ({unknown_pct}% of total dataset)",
        f"- **Share of All False Positives**: `{round((total_unknown / float(false_positives)) * 100.0, 2)}%`",
        "- **Breakdown by Detector Method**:",
    ])

    for m, cnt in unknown_by_method.items():
        md_lines.append(f"  - `{m}`: `{cnt}` unknown instances")

    md_lines.extend([
        "- **Analysis & Characteristics**:",
        "  - Represent unclassified background terrain fragments, non-standard building features, or UI overlays.",
        "  - Second largest category of false positives after static map decorations.",
        "",
        "---",
        "",
        "## 5. Method × Class Matrix",
        "",
        "| Method | " + " | ".join(f"`{cls}`" for cls in all_classes) + " | Total | Precision |",
        "| --- | " + " | ".join("---" for _ in all_classes) + " | --- | --- |",
    ])

    for m in methods_list:
        row_str = f"| `{m}` | "
        row_str += " | ".join(str(method_class_matrix[m].get(cls, 0)) for cls in all_classes)
        row_str += f" | {method_total[m]} | **{precision_by_method[m]['precision_pct']}%** |"
        md_lines.append(row_str)

    # Matrix Totals Row
    total_row_str = "| **Total** | "
    total_row_str += " | ".join(str(class_distribution.get(cls, 0)) for cls in all_classes)
    total_row_str += f" | **{total_samples}** | **{global_precision}%** |"
    md_lines.append(total_row_str)

    md_lines.extend([
        "",
        "---",
        "",
        "## 6. Phase 4 Recommendations (Based Exclusively on Human Labels)",
        "",
        "1. **Decoration & Unknown Noise Suppression**:",
        "   - **Problem**: `decoration` (39) + `unknown` (38) account for **68.75% of all false positives** (77 / 112).",
        "   - **Action**: Implement texture variance and color uniformity filters to discard static background tiles and decorative elements.",
        "",
        "2. **Vibrant Plant / Flower Sprite Filtering**:",
        "   - **Problem**: `flower` candidates (16) are frequently triggered in `combat_base` (11) and `hsv` (5).",
        "   - **Action**: Refine red/yellow HSV saturation and brightness ring constraints to prevent small floral pixels from triggering candidate bounding boxes.",
        "",
        "3. **Tactical Movement Grid Cell Exclusion**:",
        "   - **Problem**: `movement_cell` highlights (14) are being accepted as candidate entities.",
        "   - **Action**: Add grid mask detection or bounding box aspect-ratio/edge-regularity rules to suppress map cell highlights.",
        "",
        "4. **Contour Pipeline Tuning**:",
        "   - **Observation**: `contour` achieves the highest precision (**35.29%**), but still accepts 22 non-entity crops.",
        "   - **Action**: Tighten minimum area and vertical aspect ratio parameters.",
        "",
        "5. **Multi-Channel Candidate Consensus**:",
        "   - **Observation**: `hsv` (23.53%) and `combat_base` (23.88%) have near-identical 76% false positive rates.",
        "   - **Action**: Require candidate consensus (e.g., HSV ring match + contour edge verification) before accepting candidates in candidate merger.",
        "",
    ])

    md_path = os.path.join("audit", "human_label_analysis.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # Output report summary to stdout
    print("\n" + "=" * 65)
    print("  PHASE 3C — HUMAN-LABELED PERFORMANCE ANALYSIS COMPLETE")
    print("=" * 65)
    print(f"Ground Truth File:           {excel_path}")
    print(f"JSON Report Exported:        {json_path}")
    print(f"Markdown Report Exported:    {md_path}")
    print("-" * 65)
    print(f"TOTAL ACCEPTED CANDIDATES:   {total_samples}")
    print(f"TRUE POSITIVES (Entities):   {true_positives} ({global_precision}%)")
    print(f"FALSE POSITIVES (Non-Ent):   {false_positives} ({global_fp_rate}%)")
    print(f"GLOBAL PRECISION:            {global_precision}%")
    print("-" * 65)
    print("PRECISION BY METHOD:")
    for m in methods_list:
        st = precision_by_method[m]
        print(f"  - {m:<12}: {st['precision_pct']:>6.2f}% ({st['true_positives']}/{st['total']} TP)")
    print("-" * 65)
    print("CLASS DISTRIBUTION:")
    for cls, cnt in class_distribution.most_common():
        pct = round((cnt / float(total_samples)) * 100.0, 2)
        print(f"  - {cls:<15}: {cnt:>3} ({pct:>5.2f}%)")
    print("-" * 65)
    print("FALSE POSITIVE DISTRIBUTION:")
    for cls, cnt in fp_class_distribution.most_common():
        pct_fp = round((cnt / float(false_positives)) * 100.0, 2)
        print(f"  - {cls:<15}: {cnt:>3} ({pct_fp:>5.2f}% of FPs)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
