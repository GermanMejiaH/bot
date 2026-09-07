"""Script to analyze why player on blue base was missed and why barn roof barrel was detected."""

import os

import cv2
import numpy as np

USER_IMAGES_DIR = r"C:\Users\Andres\.gemini\antigravity-ide\brain\aaee6007-432e-4086-9822-f16f6f0a3661\.user_uploaded"

def inspect_image1():
    path = os.path.join(USER_IMAGES_DIR, "media_1788759305759.jpg")
    frame = cv2.imread(path)
    if frame is None:
        print("Image 1 not found")
        return

    h, w = frame.shape[:2]
    print(f"Image 1 Resolution: {w}x{h}")

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _ = cv2.Canny(gray, 40, 120)

    # 1. Red Heart & HUD check
    # Top roof area (y < 0.20 * h)
    roof_crop = frame[0:int(h*0.20), :]
    cv2.imwrite("debug_captures/scratch/roof_crop.png", roof_crop)

    # 2. Player base area in Image 1: x from 30% to 45%, y from 50% to 70%
    px1, py1 = int(w * 0.30), int(h * 0.50)
    px2, py2 = int(w * 0.45), int(h * 0.70)
    player_region = frame[py1:py2, px1:px2]
    cv2.imwrite("debug_captures/scratch/player_region.png", player_region)

    # Check HSV colors in player region
    player_hsv = hsv[py1:py2, px1:px2]

    # Blue base mask (S>=50, V>=80)
    blue_mask = cv2.inRange(player_hsv, np.array([85, 50, 80]), np.array([135, 255, 255]))
    cv2.imwrite("debug_captures/scratch/player_blue_mask.png", blue_mask)

    cnts_blue, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Blue contours in player region: {len(cnts_blue)}")
    for c in cnts_blue:
        bx, by, bw, bh = cv2.boundingRect(c)
        aspect = bh / float(bw) if bw > 0 else 0
        global_x, global_y = px1 + bx, py1 + by
        print(f"   Local box: ({bx},{by},{bw},{bh}) -> Global ({global_x},{global_y},{bw},{bh}), aspect={aspect:.2f}")

    # Morphological closing on blue mask (3x3 kernel) to connect pet-broken fragments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    closed_blue = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    cnts_closed, _ = cv2.findContours(closed_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"\nClosed blue contours in player region: {len(cnts_closed)}")
    for c in cnts_closed:
        bx, by, bw, bh = cv2.boundingRect(c)
        aspect = bh / float(bw) if bw > 0 else 0
        global_x, global_y = px1 + bx, py1 + by
        print(f"   Closed Local box: ({bx},{by},{bw},{bh}) -> Global ({global_x},{global_y},{bw},{bh}), aspect={aspect:.2f}")

if __name__ == "__main__":
    inspect_image1()
