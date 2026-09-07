"""CLI script analyzing dataset statistics, classification distribution, and metadata metrics."""

import argparse
import json
import os
from collections import Counter
from typing import Any


def analyze_dataset(dataset_dir: str = "dataset") -> dict[str, Any]:
    """Inspect dataset directory structure, metadata JSON files, and session report."""
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' does not exist.")
        return {}

    subdirs = ["combat", "exploration", "raw", "validation"]
    category_counts: dict[str, dict[str, int]] = {}

    total_images = 0
    total_jsons = 0

    resolutions: Counter[str] = Counter()
    phashes: set[str] = set()

    combat_count = 0
    exploration_count = 0

    pa_values: list[int] = []
    pm_values: list[int] = []
    hp_values: list[int] = []
    entity_counts: list[int] = []
    frame_diffs: list[float] = []

    for sub in subdirs:
        subpath = os.path.join(dataset_dir, sub)
        if not os.path.exists(subpath):
            category_counts[sub] = {"images": 0, "jsons": 0}
            continue

        files = os.listdir(subpath)
        img_count = sum(1 for f in files if f.endswith(".png"))
        json_count = sum(1 for f in files if f.endswith(".json"))

        category_counts[sub] = {"images": img_count, "jsons": json_count}
        total_images += img_count
        total_jsons += json_count

        # Parse JSON sidecars (avoid duplicating raw folder stats in aggregates)
        if sub in ["combat", "exploration", "validation"]:
            for f in files:
                if f.endswith(".json"):
                    jpath = os.path.join(subpath, f)
                    try:
                        with open(jpath, encoding="utf-8") as jf:
                            data = json.load(jf)

                        if data.get("combat") is True:
                            combat_count += 1
                        else:
                            exploration_count += 1

                        res = data.get("resolution")
                        if res and len(res) == 2:
                            resolutions[f"{res[0]}x{res[1]}"] += 1

                        phash = data.get("phash")
                        if phash:
                            phashes.add(phash)

                        if data.get("pa") is not None:
                            pa_values.append(int(data["pa"]))
                        if data.get("pm") is not None:
                            pm_values.append(int(data["pm"]))
                        if data.get("hp") is not None:
                            hp_values.append(int(data["hp"]))
                        if data.get("detected_entities") is not None:
                            entity_counts.append(int(data["detected_entities"]))
                        if data.get("frame_difference") is not None:
                            frame_diffs.append(float(data["frame_difference"]))

                    except Exception:
                        pass

    # Read session report.json if present
    report_data: dict[str, Any] | None = None
    report_path = os.path.join(dataset_dir, "report.json")
    if os.path.exists(report_path):
        try:
            with open(report_path, encoding="utf-8") as rf:
                report_data = json.load(rf)
        except Exception:
            pass

    # Summary Statistics Output
    print("\n" + "=" * 60)
    print("           DOFUS 3 BOT - DATASET ANALYSIS REPORT           ")
    print("=" * 60)
    print(f"Dataset Root Directory : {os.path.abspath(dataset_dir)}")
    print(f"Total Saved Images     : {total_images}")
    print(f"Total Metadata Files   : {total_jsons}")
    print(f"Unique Perceptual Hashes: {len(phashes)}")

    print("\n" + "-" * 60)
    print(" CATEGORY BREAKDOWN")
    print("-" * 60)
    for sub, counts in category_counts.items():
        print(f"  • {sub:<12} : {counts['images']:>5} images | {counts['jsons']:>5} jsons")

    classified_total = combat_count + exploration_count
    if classified_total > 0:
        combat_pct = (combat_count / classified_total) * 100.0
        exp_pct = (exploration_count / classified_total) * 100.0
        print("\n" + "-" * 60)
        print(" STATE DISTRIBUTION")
        print("-" * 60)
        print(f"  • Combat      : {combat_count:>5} samples ({combat_pct:>5.1f}%)")
        print(f"  • Exploration : {exploration_count:>5} samples ({exp_pct:>5.1f}%)")

    if resolutions:
        print("\n" + "-" * 60)
        print(" RESOLUTION BREAKDOWN")
        print("-" * 60)
        for res_str, count in resolutions.most_common():
            pct = (count / max(1, classified_total)) * 100.0
            print(f"  • {res_str:<12} : {count:>5} samples ({pct:>5.1f}%)")

    if pa_values or pm_values or hp_values or entity_counts:
        print("\n" + "-" * 60)
        print(" OCR & PERCEPTION METRICS (AVERAGES)")
        print("-" * 60)
        if pa_values:
            print(f"  • Average PA           : {sum(pa_values)/len(pa_values):.1f}")
        if pm_values:
            print(f"  • Average PM           : {sum(pm_values)/len(pm_values):.1f}")
        if hp_values:
            print(f"  • Average HP           : {sum(hp_values)/len(hp_values):.1f}")
        if entity_counts:
            print(f"  • Avg Entities / Frame : {sum(entity_counts)/len(entity_counts):.2f}")
        if frame_diffs:
            print(f"  • Avg Frame Difference : {sum(frame_diffs)/len(frame_diffs):.4f}")

    if report_data and "metrics" in report_data:
        m = report_data["metrics"]
        print("\n" + "-" * 60)
        print(" LAST SESSION REPORT (report.json)")
        print("-" * 60)
        print(f"  • Session Duration     : {report_data.get('duration_seconds', 0.0):.1f} seconds")
        print(f"  • Total Frames Seen    : {m.get('total_frames_seen', 0)}")
        print(f"  • Total Frames Saved   : {m.get('total_frames_saved', 0)}")
        print(f"  • Duplicates Skipped   : {m.get('duplicate_frames_skipped', 0)}")
        print(f"  • Combat Saved         : {m.get('combat_frames_saved', 0)}")
        print(f"  • Exploration Saved    : {m.get('exploration_frames_saved', 0)}")
        print(f"  • Avg Frame Difference : {m.get('average_frame_difference', 0.0):.4f}")

    print("=" * 60 + "\n")

    return {
        "total_images": total_images,
        "category_counts": category_counts,
        "combat_count": combat_count,
        "exploration_count": exploration_count,
        "report_data": report_data,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Dofus 3 bot dataset statistics and metadata.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset",
        help="Path to dataset root directory (default: 'dataset')",
    )
    args = parser.parse_args()
    analyze_dataset(args.dataset_dir)


if __name__ == "__main__":
    main()
