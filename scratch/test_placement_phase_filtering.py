"""Script to analyze empty placement tiles vs real character entities in Image 1."""

import os

import cv2
import numpy as np

USER_IMAGES_DIR = r"C:\Users\Andres\.gemini\antigravity-ide\brain\aaee6007-432e-4086-9822-f16f6f0a3661\.user_uploaded"

def analyze_placement_tiles():
    path = os.path.join(USER_IMAGES_DIR, "media_1788756198739.jpg")
    frame = cv2.imread(path)
    if frame is None:
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red & Blue ring/cell masks
    r1 = cv2.inRange(hsv, np.array([0, 120, 120]), np.array([12, 255, 255]))
    r2 = cv2.inRange(hsv, np.array([165, 120, 120]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(r1, r2)
    blue_mask = cv2.inRange(hsv, np.array([100, 120, 120]), np.array([130, 255, 255]))
    cell_mask = cv2.bitwise_or(red_mask, blue_mask)

    cnts, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Total candidate placement tile contours: {len(cnts)}")

    for idx, c in enumerate(cnts):
        x, y, w, h = cv2.boundingRect(c)
        aspect = h / float(w) if w > 0 else 0
        if 18 <= w <= 80 and 10 <= h <= 50 and 0.30 <= aspect <= 0.90:
            # 1. Tile fill ratio
            crop_cell = cell_mask[y:y+h, x:x+w]
            fill = np.count_nonzero(crop_cell) / float(w * h)

            # 2. Region directly above tile (character body region: y-1.6h to y)
            body_y1 = max(0, y - int(h * 1.6))
            body_y2 = y
            body_edges = edges[body_y1:body_y2, x:x+w]
            edge_dens = np.count_nonzero(body_edges) / float(w * (body_y2 - body_y1)) if (body_y2 - body_y1) > 0 else 0

            # 3. HSV Saturation variance in body region (ground is uniform, character has clothing/skin/hat contrast)
            body_hsv = hsv[body_y1:body_y2, x:x+w]
            sat_std = float(np.std(body_hsv[:, :, 1])) if body_hsv.size > 0 else 0
            val_std = float(np.std(body_hsv[:, :, 2])) if body_hsv.size > 0 else 0

            print(f"Tile #{idx:2d}: Box=({x:3d},{y:3d},{w:2d},{h:2d}), Fill={fill:.2f}, EdgeDens={edge_dens:.3f}, SatStd={sat_std:.1f}, ValStd={val_std:.1f}")

if __name__ == "__main__":
    analyze_placement_tiles()
