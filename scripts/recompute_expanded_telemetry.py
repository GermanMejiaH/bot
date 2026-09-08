"""Recompute and populate expanded telemetry across audit/detections.json and sidecar JSON files."""

import json
import os
import sys

import cv2
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from dta.detectors.character_detector import CharacterDetector


def update_telemetry() -> None:
    detector = CharacterDetector()
    detections_path = os.path.join("audit", "detections.json")
    accepted_dir = os.path.join("audit", "review", "accepted")

    with open(detections_path, encoding="utf-8") as f:
        detections_data = json.load(f)

    frames = detections_data.get("frames", [])
    total_candidates_updated = 0
    hsv_candidates_updated = 0

    sidecar_lookup: dict[tuple[str, str], str] = {}
    if os.path.exists(accepted_dir):
        for fname in os.listdir(accepted_dir):
            if fname.endswith(".json"):
                # Format: candidate_0001_contour.json
                parts = fname.replace(".json", "").split("_")
                if len(parts) >= 3 and parts[0] == "candidate":
                    cid_fmt = parts[1] # e.g. 0001
                    method = "_".join(parts[2:]) # e.g. contour, hsv, combat_base
                    sidecar_lookup[(cid_fmt, method)] = os.path.join(accepted_dir, fname)

    for fr in frames:
        frame_seq = fr.get("audit_sequence", 1)
        orig_img_path = os.path.join("audit", "original", f"frame_{frame_seq:04d}.png")
        if not os.path.exists(orig_img_path):
            continue

        frame_bgr = cv2.imread(orig_img_path)
        if frame_bgr is None:
            continue

        hsv_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        gray_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_img, (3, 3), 0)
        edges_img = cv2.Canny(blurred, 30, 90)

        for candidate in fr.get("candidates", []):
            bbox = candidate.get("bbox", [0, 0, 0, 0])
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            method = candidate.get("method", "contour")
            cid = candidate.get("candidate_id", 0)

            # Compute mask crop if applicable
            mask_crop = None
            cnt_area = candidate.get("diagnostics", {}).get("contour_area", 0.0)

            if method == "hsv":
                hsv_mask = cv2.inRange(hsv_img, np.array([0, 15, 20]), np.array([180, 255, 255]))
                green_tile_mask = cv2.inRange(hsv_img, np.array([35, 60, 60]), np.array([85, 255, 255]))
                hsv_mask = cv2.bitwise_and(hsv_mask, cv2.bitwise_not(green_tile_mask))
                kernel_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
                closed_hsv = cv2.morphologyEx(hsv_mask, cv2.MORPH_CLOSE, kernel_vert)
                mask_crop = closed_hsv[max(0, y) : min(frame_bgr.shape[0], y + h), max(0, x) : min(frame_bgr.shape[1], x + w)]
            elif method == "contour":
                kernel_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
                closed_contours = cv2.morphologyEx(edges_img, cv2.MORPH_CLOSE, kernel_vert)
                mask_crop = closed_contours[max(0, y) : min(frame_bgr.shape[0], y + h), max(0, x) : min(frame_bgr.shape[1], x + w)]

            existing_diag = candidate.get("diagnostics", {})
            updated_diag = detector._compute_candidate_diagnostics(
                frame=frame_bgr,
                hsv=hsv_img,
                edges=edges_img,
                bbox=(x, y, w, h),
                mask_crop=mask_crop,
                cnt_area=cnt_area,
                base_diag=existing_diag,
            )

            candidate["diagnostics"] = updated_diag
            total_candidates_updated += 1
            if method == "hsv":
                hsv_candidates_updated += 1

            # Update sidecar JSON if present
            cid_fmt = f"{cid:04d}"
            sc_key = (cid_fmt, method)
            if sc_key in sidecar_lookup:
                sc_path = sidecar_lookup[sc_key]
                try:
                    with open(sc_path, encoding="utf-8") as sc_file:
                        sc_data = json.load(sc_file)
                    sc_data["diagnostics"] = updated_diag
                    with open(sc_path, "w", encoding="utf-8") as sc_file:
                        json.dump(sc_data, sc_file, indent=2)
                except Exception as err:
                    print(f"Warning: Could not update sidecar {sc_path}: {err}")

    # Save updated detections.json
    with open(detections_path, "w", encoding="utf-8") as f:
        json.dump(detections_data, f, indent=2)

    print(f"Successfully recomputed telemetry for {total_candidates_updated} candidates across {len(frames)} frames!")
    print(f"HSV candidates updated with Canny edge_density: {hsv_candidates_updated}")


if __name__ == "__main__":
    update_telemetry()
