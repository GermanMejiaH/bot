"""Detailed inspection of character vs path/grid base markers."""

import cv2
import numpy as np


def inspect_ring_shapes():
    path = "debug_captures/raw/raw_frame_000061.png"
    frame = cv2.imread(path)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red & Blue ring masks
    r1 = cv2.inRange(hsv, np.array([0, 140, 140]), np.array([10, 255, 255]))
    r2 = cv2.inRange(hsv, np.array([170, 140, 140]), np.array([180, 255, 255]))
    red_ring = cv2.bitwise_or(r1, r2)
    blue_ring = cv2.inRange(hsv, np.array([100, 140, 140]), np.array([125, 255, 255]))

    # Analyze red contours
    cnts_red, _ = cv2.findContours(red_ring, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Red contours count: {len(cnts_red)}")
    for i, c in enumerate(cnts_red):
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        if 15 <= w <= 80 and 10 <= h <= 60:
            print(f"  Red Candidate #{i}: Box=({x},{y},{w},{h}), Area={area}, Solidity={solidity:.2f}")

    # Analyze blue contours
    cnts_blue, _ = cv2.findContours(blue_ring, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"\nBlue contours count: {len(cnts_blue)}")
    for i, c in enumerate(cnts_blue):
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        if 15 <= w <= 80 and 10 <= h <= 60:
            print(f"  Blue Candidate #{i}: Box=({x},{y},{w},{h}), Area={area}, Solidity={solidity:.2f}")

if __name__ == "__main__":
    inspect_ring_shapes()
