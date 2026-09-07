"""Test morphological connection and white corner inclusion for player base detection."""

import os

import cv2
import numpy as np

USER_IMAGES_DIR = r"C:\Users\Andres\.gemini\antigravity-ide\brain\aaee6007-432e-4086-9822-f16f6f0a3661\.user_uploaded"

def test_connection():
    path = os.path.join(USER_IMAGES_DIR, "media_1788759305759.jpg")
    frame = cv2.imread(path)
    h, w = frame.shape[:2]

    # Player region: x (280..420), y (420..520)
    player_crop = frame[420:520, 280:420]
    hsv_player = cv2.cvtColor(player_crop, cv2.COLOR_BGR2HSV)

    # 1. Blue ring mask
    blue_mask = cv2.inRange(hsv_player, np.array([85, 40, 70]), np.array([135, 255, 255]))

    # 2. White corner brackets mask (S <= 60, V >= 180)
    white_mask = cv2.inRange(hsv_player, np.array([0, 0, 180]), np.array([180, 60, 255]))

    # Combined base ring mask
    combined_ring = cv2.bitwise_or(blue_mask, white_mask)

    # Apply morphological closing with (9, 5) rectangle kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
    closed_ring = cv2.morphologyEx(combined_ring, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(closed_ring, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Connected base ring contours in player region: {len(cnts)}")

    for c in cnts:
        bx, by, bw, bh = cv2.boundingRect(c)
        aspect = bh / float(bw) if bw > 0 else 0
        gx, gy = 280 + bx, 420 + by
        if 14 <= bw <= 80 and 6 <= bh <= 50 and 0.20 <= aspect <= 0.95:
            print(f"   Player Base Candidate: Local=({bx},{by},{bw},{bh}) -> Global ({gx},{gy},{bw},{bh}), aspect={aspect:.2f}")

if __name__ == "__main__":
    test_connection()
