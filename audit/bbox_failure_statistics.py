"""Phase 4D.2 — Bounding Box Failure Statistics Analyzer.

Reads audit/bbox_failure_review.xlsx, summarizes failure distributions,
and exports:
- audit/bbox_failure_summary.json
- audit/bbox_failure_summary.md

# ruff: noqa: N806, N803
"""

import json
import os

import openpyxl

ALLOWED_FAILURE_TYPES = [
    "LEFT_CLIPPED",
    "RIGHT_CLIPPED",
    "TOP_CLIPPED",
    "BOTTOM_CLIPPED",
    "PET_ONLY",
    "MULTI_ENTITY_FRAGMENT",
    "MULTI_TILE",
    "GROUND_TILE",
    "ENTITY_PLUS_TERRAIN",
    "ENTITY_PLUS_UI",
    "UNCERTAIN",
]


def analyze_bbox_failures(excel_path: str) -> dict:
    """Analyze annotations in bbox_failure_review.xlsx."""
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel file not found at '{excel_path}'")

    wb = openpyxl.load_workbook(excel_path, data_only=True)
    ws = wb.active

    total_samples = 0
    annotated_samples = 0
    unannotated_samples = 0

    failure_counts = {ft: 0 for ft in ALLOWED_FAILURE_TYPES}
    failure_counts["OTHER_UNCLASSIFIED"] = 0

    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or not row[0]:
            continue

        total_samples += 1
        failure_type = str(row[5]).strip() if len(row) >= 6 and row[5] is not None else ""

        if failure_type == "" or failure_type.lower() == "none":
            unannotated_samples += 1
        elif failure_type in failure_counts:
            failure_counts[failure_type] += 1
            annotated_samples += 1
        else:
            failure_counts["OTHER_UNCLASSIFIED"] += 1
            annotated_samples += 1

    denom = max(1, annotated_samples)
    failure_percentages = {
        ft: round((count / float(denom)) * 100.0, 2)
        for ft, count in failure_counts.items()
    }

    # Dominant failure mode
    non_uncertain = {k: v for k, v in failure_counts.items() if k != "UNCERTAIN"}
    dominant = max(non_uncertain.items(), key=lambda x: x[1])[0] if non_uncertain and max(non_uncertain.values()) > 0 else "NONE"

    return {
        "total_samples": total_samples,
        "annotated_samples": annotated_samples,
        "unannotated_samples": unannotated_samples,
        "failure_counts": failure_counts,
        "failure_percentages": failure_percentages,
        "dominant_failure_mode": dominant,
    }


def export_summary_json(data: dict, json_path: str) -> None:
    """Export failure summary to JSON."""
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Exported BBox failure summary JSON to '{json_path}'.")


def export_summary_md(data: dict, md_path: str) -> None:
    """Export failure summary report to Markdown."""
    total = data["total_samples"]
    annotated = data["annotated_samples"]
    unannotated = data["unannotated_samples"]
    counts = data["failure_counts"]
    pcts = data["failure_percentages"]
    dominant = data["dominant_failure_mode"]

    md = "# Phase 4D.2 — Bounding Box Geometric Failure Taxonomy Audit Summary\n\n"
    md += "## Executive Summary\n\n"
    md += f"A detailed geometric failure mode audit was performed across **{annotated} / {total} flagged candidate bounding boxes** (`PARTIAL_ENTITY` and `OVERSIZED_BBOX`).\n\n"
    md += f"- **Total Failure Candidates Audited**: `{total}`\n"
    md += f"- **Annotated Samples**: `{annotated}`\n"
    md += f"- **Unannotated Samples**: `{unannotated}`\n"
    md += f"- **Dominant Geometric Failure Mode**: `{dominant}`\n\n"

    md += "## 1. Geometric Failure Mode Distribution\n\n"
    md += "| Failure Mode | Count | Percentage Share (%) | Visual Manifestation & Cause |\n"
    md += "| :--- | :---: | :---: | :--- |\n"
    for ft in ALLOWED_FAILURE_TYPES:
        count = counts.get(ft, 0)
        pct = pcts.get(ft, 0.0)
        md += f"| **`{ft}`** | `{count}` | `{pct}%` | Geometric candidate bound distortion. |\n"
    md += f"| **Total Annotated** | `{annotated}` | `100.0%` | Audited geometry candidates. |\n\n"

    md += "## 2. Root Cause Analysis & Detector Recommendations\n\n"
    md += f"The dominant geometric failure mode identified is **`{dominant}`**.\n\n"
    md += "### Key Findings & Detector Refinement Roadmap:\n"
    md += "1. **Boundary Clipping Mitigation**: If clipping (`TOP_CLIPPED`, `BOTTOM_CLIPPED`) dominates, adjust contour expansion padding and HSV bounding box expansion in `CharacterDetector`.\n"
    md += "2. **Multi-Tile Terrain Purging**: If `MULTI_TILE` or `ENTITY_PLUS_TERRAIN` dominates, enforce bounding box area ceiling (`area <= 12,000`) and aspect ratio caps before visual classification.\n"
    md += "3. **Pet/Fragment Filtering**: If `PET_ONLY` or `MULTI_ENTITY_FRAGMENT` dominates, refine color thresholds and contour hierarchy checks.\n"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Exported BBox failure summary Markdown report to '{md_path}'.")


def main() -> None:
    excel_path = os.path.join("audit", "bbox_failure_review.xlsx")
    json_path = os.path.join("audit", "bbox_failure_summary.json")
    md_path = os.path.join("audit", "bbox_failure_summary.md")

    print("\n=======================================================")
    print("  PHASE 4D.2 BBOX FAILURE STATISTICS ANALYZER")
    print("=======================================================")

    summary_data = analyze_bbox_failures(excel_path)
    export_summary_json(summary_data, json_path)
    export_summary_md(summary_data, md_path)

    print("\nPhase 4D.2 BBox Failure Analysis Successfully Completed!")


if __name__ == "__main__":
    main()
