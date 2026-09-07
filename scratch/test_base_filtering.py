"""Script to test hollow ring fill ratio and vertical body verification on combat base candidates."""

import cv2
import numpy as np


def analyze_combat_bases():
    path = "debug_captures/raw/raw_frame_000061.png"
    frame = cv2.imread(path)
    if frame is None:
        print("Frame not found")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    r1 = cv2.inRange(hsv, np.array([0, 140, 140]), np.array([10, 255, 255]))
    r2 = cv2.inRange(hsv, np.array([170, 140, 140]), np.array([180, 255, 255]))
    red_ring = cv2.bitwise_or(r1, r2)
    blue_ring = cv2.inRange(hsv, np.array([100, 140, 140]), np.array([125, 255, 255]))
    ring_mask = cv2.bitwise_or(red_ring, blue_ring)

    cnts, _ = cv2.findContours(ring_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"Total raw ring mask contours: {len(cnts)}")

    valid_bases = []
    for idx, c in enumerate(cnts):
        x, y, w, h = cv2.boundingRect(c)
        aspect = h / float(w) if w > 0 else 0.0
        if 18 <= w <= 85 and 10 <= h <= 55 and 0.30 <= aspect <= 0.90:
            # 1. Fill density of the ring mask inside bbox
            crop_mask = ring_mask[y:y+h, x:x+w]
            fill_ratio = np.count_nonzero(crop_mask) / float(w * h)

            # 2. Vertical sprite body check directly above base: y-1.5h to y
            body_y1 = max(0, y - int(h * 1.8))
            body_y2 = y
            sprite_region = edges[body_y1:body_y2, x:x+w]
            edge_density = np.count_nonzero(sprite_region) / float(w * (body_y2 - body_y1)) if (body_y2 - body_y1) > 0 else 0.0

            print(f"Candidate #{idx}: Box=({x},{y},{w},{h}), aspect={aspect:.2f}, fill_ratio={fill_ratio:.2f}, edge_density={edge_density:.3f}")

            # Filter:
            # - Hollow ring or small base border should have fill_ratio < 0.65 (solid tiles are >0.70)
            # - Sprite region above must have edge_density > 0.04 (contains character body edges)
            if fill_ratio < 0.65 and edge_density >= 0.04:
                valid_bases.append((x, y, w, h))

    print(f"\nFiltered valid character bases: {len(valid_bases)}")

if __name__ == "__main__":
    analyze_combat_bases()
