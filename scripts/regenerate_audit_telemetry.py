"""Regenerate audit/detections.json and update sidecar JSON files with expanded telemetry."""

import json
import os
import sys

import cv2

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from dta.detectors.character_detector import CharacterDetector


def regenerate() -> None:
    detector = CharacterDetector()
    orig_dir = os.path.join("audit", "original")
    accepted_dir = os.path.join("audit", "review", "accepted")

    frame_files = sorted([f for f in os.listdir(orig_dir) if f.startswith("frame_") and f.endswith(".png")])
    print(f"Found {len(frame_files)} original frames in {orig_dir}")

    # Index existing sidecars in audit/review/accepted/
    sidecar_lookup: dict[tuple[str, str], str] = {}
    if os.path.exists(accepted_dir):
        for fname in os.listdir(accepted_dir):
            if fname.endswith(".json"):
                parts = fname.replace(".json", "").split("_")
                if len(parts) >= 3 and parts[0] == "candidate":
                    cid_fmt = parts[1]
                    method = "_".join(parts[2:])
                    sidecar_lookup[(cid_fmt, method)] = os.path.join(accepted_dir, fname)

    all_frames_data = []
    global_candidate_id = 1
    total_candidates = 0
    hsv_candidates = 0
    sidecars_updated = 0

    for fname in frame_files:
        seq_str = fname.replace("frame_", "").replace(".png", "")
        seq_num = int(seq_str)
        frame_id = f"audit_{seq_num:04d}"

        frame_path = os.path.join(orig_dir, fname)
        frame_bgr = cv2.imread(frame_path)
        if frame_bgr is None:
            continue

        trace = detector.detect_characters_with_trace(frame_bgr)
        cands = trace.get("all_candidates", [])

        cand_records = []
        for det in cands:
            cid = global_candidate_id
            global_candidate_id += 1
            det.candidate_id = cid

            status_tag = "accepted" if det.accepted else "rejected"
            crop_file = f"entity_{cid:04d}_{status_tag}.png"
            det.crop_file = crop_file

            record = {
                "candidate_id": cid,
                "method": det.method,
                "bbox": [det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h],
                "centroid": list(det.centroid),
                "area": det.area,
                "confidence": det.confidence,
                "accepted": det.accepted,
                "reason": det.reason,
                "crop_file": crop_file,
                "lifecycle": det.lifecycle,
                "diagnostics": det.diagnostics,
                "triggered_rules": det.triggered_rules,
            }
            cand_records.append(record)
            total_candidates += 1
            if det.method == "hsv":
                hsv_candidates += 1

            cid_fmt = f"{cid:04d}"
            sc_key = (cid_fmt, det.method)
            if sc_key in sidecar_lookup:
                sc_path = sidecar_lookup[sc_key]
                try:
                    with open(sc_path, encoding="utf-8") as sc_f:
                        sc_data = json.load(sc_f)
                    sc_data["diagnostics"] = det.diagnostics
                    sc_data["frame_id"] = frame_id
                    sc_data["bbox"] = [det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h]
                    sc_data["area"] = det.area
                    with open(sc_path, "w", encoding="utf-8") as sc_f:
                        json.dump(sc_data, sc_f, indent=2)
                    sidecars_updated += 1
                except Exception as err:
                    print(f"Error updating sidecar {sc_path}: {err}")

        frame_record = {
            "frame_id": frame_id,
            "audit_sequence": seq_num,
            "timestamp": "2026-09-07T21:00:00Z",
            "game_state": {
                "combat": False,
                "entities_count": len([c for c in cands if c.accepted]),
                "entities_by_method": {
                    "contour": sum(1 for c in cands if c.accepted and c.method == "contour"),
                    "hsv": sum(1 for c in cands if c.accepted and c.method == "hsv"),
                    "combat_base": sum(1 for c in cands if c.accepted and c.method == "combat_base"),
                },
            },
            "statistics": trace.get("statistics", {}),
            "candidates": cand_records,
        }
        all_frames_data.append(frame_record)

    audit_json = {
        "total_frames_audited": len(all_frames_data),
        "audit_timestamp": "2026-09-07T21:00:00Z",
        "frames": all_frames_data,
    }

    with open(os.path.join("audit", "detections.json"), "w", encoding="utf-8") as f:
        json.dump(audit_json, f, indent=2)

    print(f"Regenerated audit/detections.json with {len(all_frames_data)} frames and {total_candidates} candidates.")
    print(f"HSV candidates with valid edge_density: {hsv_candidates}")
    print(f"Sidecar JSON files updated: {sidecars_updated} / {len(sidecar_lookup)}")


if __name__ == "__main__":
    regenerate()
