"""Phase 4D.1C — Positive Audit Statistics Analyzer.

Reads audit/manual_positive_review_v2.xlsx, computes category frequencies,
and exports:
- audit/positive_audit_summary.json
- audit/positive_audit_summary.md

# ruff: noqa: N806, N803
"""

import json
import os

import openpyxl

ALLOWED_CATEGORIES = [
    "VALID_ENTITY",
    "PARTIAL_ENTITY",
    "DECORATION",
    "OVERSIZED_BBOX",
    "UNCERTAIN",
]


def analyze_positive_audit(excel_path: str) -> dict:
    """Analyze annotations in manual_positive_review_v2.xlsx."""
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel file not found at '{excel_path}'")

    wb = openpyxl.load_workbook(excel_path, data_only=True)
    ws = wb.active

    total_samples = 0
    annotated_samples = 0
    unannotated_samples = 0

    category_counts = {cat: 0 for cat in ALLOWED_CATEGORIES}
    category_counts["OTHER_UNCLASSIFIED"] = 0

    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or not row[0]:
            continue

        total_samples += 1
        review_label = str(row[5]).strip() if len(row) >= 6 and row[5] is not None else ""

        if review_label == "" or review_label.lower() == "none":
            unannotated_samples += 1
        elif review_label in category_counts:
            category_counts[review_label] += 1
            annotated_samples += 1
        else:
            category_counts["OTHER_UNCLASSIFIED"] += 1
            annotated_samples += 1

    denom = max(1, annotated_samples)
    category_percentages = {
        cat: round((count / float(denom)) * 100.0, 2)
        for cat, count in category_counts.items()
    }

    # Determine dominant error source (excluding VALID_ENTITY and UNCERTAIN)
    error_cats = {k: v for k, v in category_counts.items() if k not in {"VALID_ENTITY", "UNCERTAIN"}}
    dominant_error = max(error_cats.items(), key=lambda x: x[1])[0] if error_cats and max(error_cats.values()) > 0 else "NONE"

    summary_data = {
        "total_samples": total_samples,
        "annotated_samples": annotated_samples,
        "unannotated_samples": unannotated_samples,
        "category_counts": category_counts,
        "category_percentages": category_percentages,
        "dominant_error_source": dominant_error,
    }
    return summary_data


def export_summary_json(data: dict, json_path: str) -> None:
    """Export summary statistics to JSON."""
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Exported audit statistics JSON to '{json_path}'.")


def export_summary_md(data: dict, md_path: str) -> None:
    """Export audit statistics report to Markdown."""
    total = data["total_samples"]
    annotated = data["annotated_samples"]
    unannotated = data["unannotated_samples"]
    counts = data["category_counts"]
    pcts = data["category_percentages"]
    dominant = data["dominant_error_source"]

    md = "# Phase 4D.1C — Positive Candidate Contamination Audit Summary\n\n"
    md += "## Executive Summary\n\n"
    md += f"A total of **{annotated} / {total} accepted positive predictions** ($P \\ge 0.50$) were annotated and categorized to measure dataset contamination sources.\n\n"
    md += f"- **Annotated Samples**: `{annotated}`\n"
    md += f"- **Unannotated Samples**: `{unannotated}`\n"
    md += f"- **Dominant Contamination Source**: `{dominant}`\n\n"

    md += "## 1. Category Frequency Breakdown\n\n"
    md += "| Review Category | Count | Percentage Share (%) | Visual Impact & Interpretation |\n"
    md += "| :--- | :---: | :---: | :--- |\n"
    md += f"| **`VALID_ENTITY`** | `{counts.get('VALID_ENTITY', 0)}` | `{pcts.get('VALID_ENTITY', 0.0)}%` | True valid player character, monster, or NPC sprite core. |\n"
    md += f"| **`PARTIAL_ENTITY`** | `{counts.get('PARTIAL_ENTITY', 0)}` | `{pcts.get('PARTIAL_ENTITY', 0.0)}%` | Character crop with low target occupancy (<30% body core). |\n"
    md += f"| **`DECORATION`** | `{counts.get('DECORATION', 0)}` | `{pcts.get('DECORATION', 0.0)}%` | Scenery prop, statue, lamp post, wall column, or plant flora. |\n"
    md += f"| **`OVERSIZED_BBOX`** | `{counts.get('OVERSIZED_BBOX', 0)}` | `{pcts.get('OVERSIZED_BBOX', 0.0)}%` | Bounding box covering multiple ground cells or terrain tiles. |\n"
    md += f"| **`UNCERTAIN`** | `{counts.get('UNCERTAIN', 0)}` | `{pcts.get('UNCERTAIN', 0.0)}%` | Ambiguous crop requiring secondary visual context. |\n"
    md += f"| **Total Annotated** | `{annotated}` | `100.0%` | Audited positive candidates. |\n\n"

    md += "## 2. Root Cause Analysis & Primary Contamination Drivers\n\n"
    if dominant == "DECORATION":
        md += "The audit indicates that **`DECORATION` contamination** is the primary driver of false positive acceptances. High-contrast vertical scenery objects (statues, lamp posts, wall columns) produce visual feature embeddings that confuse the linear classifier.\n"
    elif dominant == "PARTIAL_ENTITY":
        md += "The audit indicates that **`PARTIAL_ENTITY` crops** dominate false positive acceptances. Bounding box candidate extraction often captures ground floor tiles with only a tiny fraction of an entity edge.\n"
    elif dominant == "OVERSIZED_BBOX":
        md += "The audit indicates that **`OVERSIZED_BBOX` candidates** dominate contamination. Large multi-tile bounding boxes encompassing floor terrain and multiple objects are being accepted as single entities.\n"
    else:
        md += "Contamination is distributed across multiple categories including decorations, partial entity crops, and oversized bounding boxes.\n"

    md += "\n## 3. Recommended Remediation Strategy\n\n"
    md += "1. **Deterministic BBox Pre-Filtering**: Add size floor (`area >= 400`), size ceiling (`area <= 12000`), and ROI floor limits in `CharacterDetector` to purge oversized multi-tiles and micro-glints prior to visual classification.\n"
    md += "2. **Targeted Negative Retraining**: Expand negative dataset with explicit `DECORATION` and `PARTIAL_ENTITY` crops to refine classifier decision boundaries in Phase 4D.2.\n"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Exported audit statistics Markdown summary to '{md_path}'.")


def main() -> None:
    excel_path = os.path.join("audit", "manual_positive_review_v2.xlsx")
    json_path = os.path.join("audit", "positive_audit_summary.json")
    md_path = os.path.join("audit", "positive_audit_summary.md")

    print("\n=======================================================")
    print("  PHASE 4D.1C POSITIVE AUDIT STATISTICS ANALYZER")
    print("=======================================================")

    summary_data = analyze_positive_audit(excel_path)
    export_summary_json(summary_data, json_path)
    export_summary_md(summary_data, md_path)

    print("\nPhase 4D.1C Statistics Analysis Successfully Completed!")


if __name__ == "__main__":
    main()
