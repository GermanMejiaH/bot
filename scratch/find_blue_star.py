"""Find exact Blue Star PA icon coordinates in HUD crop."""

import os

import cv2
import numpy as np

USER_IMAGES_DIR = r"C:\Users\Andres\.gemini\antigravity-ide\brain\aaee6007-432e-4086-9822-f16f6f0a3661\.user_uploaded"

def find_star():
    for label, name in [("Image 1", "media_1788756198739.jpg"), ("Image 2", "media_1788756252003.jpg"), ("Image 3", "media_1788756427309.jpg")]:
        path = os.path.join(USER_IMAGES_DIR, name)
        frame = cv2.imread(path)
        h, w = frame.shape[:2]

        rx1, ry1 = int(w * 0.75), int(h * 0.65)
        crop_br = frame[ry1:h, rx1:w]
        hsv_br = cv2.cvtColor(crop_br, cv2.COLOR_BGR2HSV)

        # Red Heart
        r1 = cv2.inRange(hsv_br, np.array([0, 90, 90]), np.array([12, 255, 255]))
        r2 = cv2.inRange(hsv_br, np.array([165, 90, 90]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(r1, r2)
        cnts, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        heart_box = None
        for c in cnts:
            bx, by, bw, bh = cv2.boundingRect(c)
            aspect = bh / float(bw) if bw > 0 else 0
            if 35 <= bw <= 100 and 35 <= bh <= 100 and 0.8 <= aspect <= 1.25:
                heart_box = (bx, by, bw, bh)
                break

        print(f"\n--- {label} ---")
        if heart_box:
            hx, hy, hw, hh = heart_box
            print(f"Red Heart: local ({hx}, {hy}, {hw}, {hh})")

        # Blue Star: Hue 85..130, S 80..255, V 80..255
        blue_mask = cv2.inRange(hsv_br, np.array([85, 80, 80]), np.array([130, 255, 255]))
        cnts_b, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts_b:
            bx, by, bw, bh = cv2.boundingRect(c)
            if 15 <= bw <= 55 and 15 <= bh <= 55:
                print(f"Blue Star candidate: local ({bx}, {by}, {bw}, {bh}), dx_from_heart={bx-hx}, dy_from_heart={by-hy}")
                star_crop = crop_br[by:by+bh, bx:bx+bw]
                cv2.imwrite(f"debug_captures/scratch/star_{label.replace(' ', '_')}.png", star_crop)

        # Green Diamond: Hue 35..85, S 80..255, V 80..255
        green_mask = cv2.inRange(hsv_br, np.array([35, 80, 80]), np.array([85, 255, 255]))
        cnts_g, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts_g:
            bx, by, bw, bh = cv2.boundingRect(c)
            if 15 <= bw <= 55 and 15 <= bh <= 55:
                print(f"Green Diamond candidate: local ({bx}, {by}, {bw}, {bh}), dx_from_heart={bx-hx}, dy_from_heart={by-hy}")

if __name__ == "__main__":
    find_star()
