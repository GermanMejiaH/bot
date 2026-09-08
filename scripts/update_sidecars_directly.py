"""Update all 152 sidecar JSON files in audit/review/accepted/ directly with expanded telemetry."""

import glob
import json
import os
import sys

import cv2
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from dta.detectors.character_detector import CharacterDetector


def update_sidecars() -> None:
    detector = CharacterDetector()
    accepted_dir = os.path.join("audit", "review", "accepted")
    json_paths = glob.glob(os.path.join(accepted_dir, "*.json"))

    print(f"Found {len(json_paths)} sidecar JSON files in {accepted_dir}")

    updated_count = 0
    hsv_count = 0

    for jpath in json_paths:
        with open(jpath, encoding="utf-8") as f:
            sc = json.load(f)

        frame_id = sc.get("frame_id", "")
        # e.g. audit_0148 -> frame_0148.png
        if frame_id.startswith("audit_"):
            seq_num = int(frame_id.replace("audit_", ""))
            frame_filename = f"frame_{seq_num:04d}.png"
        else:
            frame_filename = "frame_0001.png"

        orig_path = os.path.join("audit", "original", frame_filename)
        if not os.path.exists(orig_path):
            # Fallback to PNG crop if original frame doesn't exist
            png_crop_path = jpath.replace(".json", ".png")
            if os.path.exists(png_crop_path):
                crop_bgr = cv2.imread(png_crop_path)
                if crop_bgr is not None:
                    h, w = crop_bgr.shape[:2]
                    frame_bgr = crop_bgr
                    bbox = (0, 0, w, h)
                else:
                    continue
            else:
                continue
        else:
            frame_bgr = cv2.imread(orig_path)
            if frame_bgr is None:
                continue
            bbox_list = sc.get("bbox", [0, 0, 0, 0])
            bbox = (bbox_list[0], bbox_list[1], bbox_list[2], bbox_list[3])

        hsv_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV) if len(frame_bgr.shape) == 3 else frame_bgr
        gray_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if len(frame_bgr.shape) == 3 else frame_bgr
        blurred = cv2.GaussianBlur(gray_img, (3, 3), 0)
        edges_img = cv2.Canny(blurred, 30, 90)

        method = sc.get("method", "contour")
        x, y, w, h = bbox

        mask_crop = None
        cnt_area = sc.get("diagnostics", {}).get("contour_area", 0.0)

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

        existing_diag = sc.get("diagnostics", {})
        updated_diag = detector._compute_candidate_diagnostics(
            frame=frame_bgr,
            hsv=hsv_img,
            edges=edges_img,
            bbox=(x, y, w, h),
            mask_crop=mask_crop,
            cnt_area=cnt_area,
            base_diag=existing_diag,
        )

        sc["diagnostics"] = updated_diag
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(sc, f, indent=2)

        updated_count += 1
        if method == "hsv":
            hsv_count += 1

    print(f"Successfully updated {updated_count} / {len(json_paths)} sidecar JSON files.")
    print(f"HSV sidecars with updated edge_density: {hsv_count}")


if __name__ == "__main__":
    update_sidecars()
