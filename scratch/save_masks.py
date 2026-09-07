"""Save upscaled PA and PM binarized masks to debug EasyOCR text reading."""

import os

import cv2
import numpy as np

USER_IMAGES_DIR = r"C:\Users\Andres\.gemini\antigravity-ide\brain\aaee6007-432e-4086-9822-f16f6f0a3661\.user_uploaded"

def save_up_masks():
    img_files = [
        ("Image 1", os.path.join(USER_IMAGES_DIR, "media_1788756198739.jpg")),
        ("Image 2", os.path.join(USER_IMAGES_DIR, "media_1788756252003.jpg")),
        ("Image 3", os.path.join(USER_IMAGES_DIR, "media_1788756427309.jpg")),
    ]

    for label, path in img_files:
        frame = cv2.imread(path)
        h, w = frame.shape[:2]

        search_x1 = int(w * 0.75)
        search_y1 = int(h * 0.65)
        hud_crop = frame[search_y1:h, search_x1:w]
        hsv_hud = cv2.cvtColor(hud_crop, cv2.COLOR_BGR2HSV)

        r1 = cv2.inRange(hsv_hud, np.array([0, 90, 90]), np.array([12, 255, 255]))
        r2 = cv2.inRange(hsv_hud, np.array([165, 90, 90]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(r1, r2)

        cnts, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        heart_box = None
        for c in cnts:
            bx, by, bw, bh = cv2.boundingRect(c)
            aspect = bh / float(bw) if bw > 0 else 0
            if 35 <= bw <= 100 and 35 <= bh <= 100 and 0.8 <= aspect <= 1.25:
                heart_box = (bx, by, bw, bh)
                break

        if not heart_box:
            continue

        hx, hy, hw, hh = heart_box
        ghx = search_x1 + hx
        ghy = search_y1 + hy

        pa_roi = (
            max(0, ghx - int(hw * 0.45)),
            ghy + int(hh * 0.75),
            ghx + int(hw * 0.55),
            ghy + int(hh * 1.50),
        )

        pa_crop = frame[pa_roi[1]:pa_roi[3], pa_roi[0]:pa_roi[2]]
        hsv_pa = cv2.cvtColor(pa_crop, cv2.COLOR_BGR2HSV)
        wmask_pa = cv2.inRange(hsv_pa, np.array([0, 0, 160]), np.array([180, 90, 255]))
        up_pa = cv2.resize(wmask_pa, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

        tag = label.replace(" ", "_")
        cv2.imwrite(f"debug_captures/scratch/pa_mask_{tag}.png", up_pa)
        cv2.imwrite(f"debug_captures/scratch/pa_raw_{tag}.png", pa_crop)
        print(f"Saved pa_mask_{tag}.png: shape={up_pa.shape}")

if __name__ == "__main__":
    save_up_masks()
