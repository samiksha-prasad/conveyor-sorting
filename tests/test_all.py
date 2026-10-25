"""Tests for the baseline colour rule, the frame generator, the CNN and the conveyor model."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sim"))
sys.path.insert(0, os.path.join(ROOT, "baseline"))

import cv2
import numpy as np
import torch

from cnn import SortingCNN, preprocess
from colour_detect import detect_colour
from conveyor import simulate
from hsv_fixed import detect_colour_fixed
from scene import AUGMENTED_TRAINING, CLASSES, CONDITIONS, HEIGHT, WIDTH, make_dataset


def solid_frame(bgr, patch=None, size=(480, 640)):
    """Grey frame with an optional coloured square of `patch` pixels per side."""
    frame = np.full((*size, 3), 60, np.uint8)
    if patch:
        frame[100:100 + patch, 100:100 + patch] = bgr
    return frame


def hsv_bgr(h, s=255, v=200):
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0])


def test_detector_is_the_baseline_code():
    # detect_colour is compiled from baseline/sorter.py itself, not a copy
    assert detect_colour.__code__.co_filename.endswith(os.path.join("baseline", "sorter.py"))
    assert detect_colour.__name__ == "detect_color"


def test_baseline_rule():
    assert detect_colour(solid_frame(hsv_bgr(5), 300)) == "Red"
    assert detect_colour(solid_frame(hsv_bgr(60), 300)) == "Green"
    assert detect_colour(solid_frame(hsv_bgr(110), 300)) == "Blue"
    assert detect_colour(solid_frame(None)) == "Unknown"
    # frames are resized to 100x100 first, so a colour must cover over 10% of the view:
    # 150x150 of 640x480 is 7.3% (too small); 200x200 is 13% (enough)
    assert detect_colour(solid_frame(hsv_bgr(60), 150)) == "Unknown"
    assert detect_colour(solid_frame(hsv_bgr(60), 200)) == "Green"
    # red is checked first, so orange-red (hue 9) counts as red
    assert detect_colour(solid_frame(hsv_bgr(9), 300)) == "Red"


def test_red_hue_bug_and_fix():
    # crimson (hue 175) is missed by the baseline range 0-10 but caught by the fix
    assert detect_colour(solid_frame(hsv_bgr(175), 300)) == "Unknown"
    assert detect_colour_fixed(solid_frame(hsv_bgr(175), 300)) == "Red"
    for h, expected in ((5, "Red"), (60, "Green"), (110, "Blue")):
        assert detect_colour_fixed(solid_frame(hsv_bgr(h), 300)) == expected
    assert detect_colour_fixed(solid_frame(hsv_bgr(60), 150)) == "Unknown"


def test_generator_is_reproducible_and_balanced():
    a, la = make_dataset(40, CONDITIONS["standard"], seed=5)
    b, lb = make_dataset(40, CONDITIONS["standard"], seed=5)
    assert a.shape == (40, HEIGHT, WIDTH, 3) and a.dtype == np.uint8
    assert np.array_equal(a, b) and np.array_equal(la, lb)
    assert np.bincount(la, minlength=len(CLASSES)).tolist() == [10, 10, 10, 10]
    for params in list(CONDITIONS.values()) + [AUGMENTED_TRAINING]:
        imgs, _ = make_dataset(4, params, seed=1)
        assert imgs.shape == (4, HEIGHT, WIDTH, 3)


def test_low_light_is_darker():
    std, _ = make_dataset(20, CONDITIONS["standard"], seed=3)
    low, _ = make_dataset(20, CONDITIONS["low_light"], seed=3)
    assert low.mean() < 0.6 * std.mean()


def test_cnn_shapes():
    frames, _ = make_dataset(3, CONDITIONS["standard"], seed=2)
    x = preprocess(frames)
    assert x.shape == (3, 3, 72, 96)
    with torch.no_grad():
        assert SortingCNN().eval()(x).shape == (3, 4)


def test_conveyor_model():
    # slow feed: both firmwares sort everything
    assert simulate(5, "blocking", 0.005) == 1.0
    assert simulate(5, "nonblocking", 0.005) == 1.0
    # the blocking 5 s hold cannot keep up at 30 items/min; the timed window can
    assert simulate(30, "blocking", 0.005) < 0.6
    assert simulate(30, "nonblocking", 0.005) == 1.0
    # if classification took longer than half the window, timed sorting would fail
    assert simulate(30, "nonblocking", 0.5) < 0.5


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("all tests passed")
