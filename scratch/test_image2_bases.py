"""Inspect all base contours in Image 2 with lowered HSV Saturation/Value thresholds."""

import os

import cv2
import numpy as np

USER_IMAGES_DIR = r"C:\Users\Andres\.gemini\antigravity-ide\brain\aaee6007-432e-4086-9822-f16f6f0a3661\.user_uploaded"

def inspect_img2():
    path = os.path.join(USER_IMAGES_DIR, "media_1788758300811.jpg")
    frame = cv2.imread(path)
    h, w = frame.shape[:2]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)

    # Red & Blue ring masks (S>=90, V>=90)
    r1 = cv2.inRange(hsv, np.array([0, 90, 90]), np.array([12, 255, 255]))
    r2 = cv2.inRange(hsv, np.array([165, 90, 90]), np.array([180, 255, 255]))
    red_ring = cv2.bitwise_or(r1, r2)
    blue_ring = cv2.inRange(hsv, np.array([95, 90, 90]), np.array([135, 255, 255]))
    ring_mask = cv2.bitwise_or(red_ring, blue_ring)

    cnts, _ = cv2.findContours(ring_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Total ring_mask contours in Image 2: {len(cnts)}")

    for idx, c in enumerate(cnts):
        bx, by, bw, bh = cv2.boundingRect(c)
        aspect = bh / float(bw) if bw > 0 else 0
        if 16 <= bw <= 80 and 8 <= bh <= 50 and 0.25 <= aspect <= 0.95:
            crop_mask = ring_mask[by:by+bh, bx:bx+bw]
            fill = np.count_nonzero(crop_mask) / float(bw * bh) if (bw * bh) > 0 else 1.0

            body_y1 = max(0, by - int(bh * 1.8))
            body_y2 = by
            sprite_region = edges[body_y1:body_y2, bx:bx+bw]
            edge_density = np.count_nonzero(sprite_region) / float(bw * (body_y2 - body_y1)) if (body_y2 - body_y1) > 0 else 0

            print(f"  Candidate #{idx:2d}: Box=({bx:3d},{by:3d},{bw:2d},{bh:2d}), aspect={aspect:.2f}, fill={fill:.2f}, edge_dens={edge_density:.3f}")

if __name__ == "__main__":
    inspect_img2()
