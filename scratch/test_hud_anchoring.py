"""Save visual crops of PA, PM, and HP relative to Red Heart."""

import os

import cv2
import numpy as np

USER_IMAGES_DIR = r"C:\Users\Andres\.gemini\antigravity-ide\brain\aaee6007-432e-4086-9822-f16f6f0a3661\.user_uploaded"

def save_hud_crops():
    img_files = [
        ("Image 1", os.path.join(USER_IMAGES_DIR, "media_1788756198739.jpg")),
        ("Image 2", os.path.join(USER_IMAGES_DIR, "media_1788756252003.jpg")),
        ("Image 3", os.path.join(USER_IMAGES_DIR, "media_1788756427309.jpg")),
    ]

    for label, path in img_files:
        frame = cv2.imread(path)
        h, w = frame.shape[:2]

        rx1, ry1 = int(w * 0.75), int(h * 0.65)
        crop_br = frame[ry1:h, rx1:w]
        hsv_br = cv2.cvtColor(crop_br, cv2.COLOR_BGR2HSV)

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

        if not heart_box:
            continue

        hx, hy, hw, hh = heart_box
        tag = label.replace(" ", "_")

        # Heart crop
        heart_crop = crop_br[hy:hy+hh, hx:hx+hw]
        cv2.imwrite(f"debug_captures/scratch/crop_heart_{tag}.png", heart_crop)

        # PA crop: left icon below heart
        pa_crop = crop_br[hy + hh - 8 : hy + int(hh * 1.8), max(0, hx - int(hw * 0.35)) : hx + int(hw * 0.55)]
        cv2.imwrite(f"debug_captures/scratch/crop_pa_{tag}.png", pa_crop)

        # PM crop: right icon below heart
        pm_crop = crop_br[hy + hh - 8 : hy + int(hh * 1.8), hx + int(hw * 0.45) : min(crop_br.shape[1], hx + int(hw * 1.35))]
        cv2.imwrite(f"debug_captures/scratch/crop_pm_{tag}.png", pm_crop)

        print(f"Saved crops for {tag}: heart=({hx},{hy},{hw},{hh}), PA box=({hx-int(hw*0.35)}, {hy+hh-8}), PM box=({hx+int(hw*0.45)}, {hy+hh-8})")

if __name__ == "__main__":
    save_hud_crops()
