"""Script to inspect user's latest 3 feedback images and test character base detection & placement tile filtering."""

import os

import cv2
import numpy as np

USER_IMAGES_DIR = r"C:\Users\Andres\.gemini\antigravity-ide\brain\aaee6007-432e-4086-9822-f16f6f0a3661\.user_uploaded"

def analyze_new_images():
    img_files = [
        ("Image 1 (Placement Phase)", os.path.join(USER_IMAGES_DIR, "media_1788758171041.jpg")),
        ("Image 2 (Adjacent Combat Entities)", os.path.join(USER_IMAGES_DIR, "media_1788758300811.jpg")),
        ("Image 3 (4 Entities Close-up)", os.path.join(USER_IMAGES_DIR, "media_1788758352113.png")),
    ]

    for label, path in img_files:
        if not os.path.exists(path):
            print(f"Missing: {path}")
            continue

        frame = cv2.imread(path)
        h, w = frame.shape[:2]
        print(f"\n==================== {label} ({w}x{h}) ====================")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red & Blue ring masks
        r1 = cv2.inRange(hsv, np.array([0, 130, 120]), np.array([12, 255, 255]))
        r2 = cv2.inRange(hsv, np.array([165, 130, 120]), np.array([180, 255, 255]))
        red_ring = cv2.bitwise_or(r1, r2)
        blue_ring = cv2.inRange(hsv, np.array([100, 130, 120]), np.array([130, 255, 255]))
        ring_mask = cv2.bitwise_or(red_ring, blue_ring)

        cnts, _ = cv2.findContours(ring_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"Raw ring_mask contours: {len(cnts)}")

        candidates = []
        for idx, c in enumerate(cnts):
            bx, by, bw, bh = cv2.boundingRect(c)
            aspect = bh / float(bw) if bw > 0 else 0
            if 16 <= bw <= 80 and 8 <= bh <= 50 and 0.25 <= aspect <= 0.95:
                crop_mask = ring_mask[by:by+bh, bx:bx+bw]
                fill_ratio = np.count_nonzero(crop_mask) / float(bw * bh) if (bw * bh) > 0 else 1.0

                body_y1 = max(0, by - int(bh * 1.8))
                body_y2 = by
                sprite_region = edges[body_y1:body_y2, bx:bx+bw]
                edge_density = np.count_nonzero(sprite_region) / float(bw * (body_y2 - body_y1)) if (body_y2 - body_y1) > 0 else 0

                body_hsv = hsv[body_y1:body_y2, bx:bx+bw]
                sat_std = float(np.std(body_hsv[:, :, 1])) if body_hsv.size > 0 else 0.0
                val_std = float(np.std(body_hsv[:, :, 2])) if body_hsv.size > 0 else 0.0

                print(f"  Contour #{idx:2d}: Box=({bx:3d},{by:3d},{bw:2d},{bh:2d}), aspect={aspect:.2f}, fill={fill_ratio:.2f}, edge_dens={edge_density:.3f}, sat_std={sat_std:.1f}, val_std={val_std:.1f}")

                candidates.append((bx, by, bw, bh))

if __name__ == "__main__":
    analyze_new_images()
