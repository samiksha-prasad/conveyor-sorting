"""
The baseline detect_color() with one fix: red also matches hues 170-179.

OpenCV hue runs 0-179 and red sits at both ends of that range. The baseline
code only checks 0-10, so crimson or slightly bluish reds are reported as
'Unknown'. Everything else (100x100 resize, blur, ranges, 1000-pixel rule,
checking order) is unchanged, so this isolates that bug from the HSV
approach's sensitivity to lighting.
"""

import cv2
import numpy as np

COLOR_RANGES = {
    "Red": [[(0, 120, 70), (10, 255, 255)], [(170, 120, 70), (179, 255, 255)]],
    "Green": [[(36, 50, 70), (89, 255, 255)]],
    "Blue": [[(90, 50, 70), (128, 255, 255)]],
}


def detect_colour_fixed(image):
    resized = cv2.resize(image, (100, 100))
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    for colour, ranges in COLOR_RANGES.items():
        mask = sum(cv2.inRange(hsv, np.array(lo), np.array(hi)) > 0 for lo, hi in ranges)
        if np.count_nonzero(mask) > 1000:
            return colour
    return "Unknown"
