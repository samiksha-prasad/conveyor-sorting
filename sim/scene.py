"""
Synthetic camera frames of objects on a conveyor belt.

Each frame is a 320 x 240 BGR image (OpenCV's channel order), like a
downscaled webcam frame: a dark rubber belt, one object of a known colour,
and a lighting condition. Everything is drawn from fixed random seeds, so
every dataset is reproducible.

These are synthetic images, not photographs. They model effects known
to hurt colour detection (dim light, colour casts, shadows, partial
occlusion, glare and motion blur), but they do not
capture everything a real camera does. See the README for what this means
for the results.

Labels: 0 = Red, 1 = Green, 2 = Blue, 3 = Other (any colour that should not
be sorted, which the baseline detector reports as 'Unknown').
"""

import colorsys
import math

import cv2
import numpy as np

WIDTH, HEIGHT = 320, 240
CLASSES = ["Red", "Green", "Blue", "Other"]

# Object colours as (hue degrees, saturation, value) ranges.
# Real red objects span both sides of hue 0/360, so red is sampled across it.
PALETTE = {
    "Red":   [((-10, 10), (0.6, 0.95), (0.5, 0.95))],
    "Green": [((90, 150), (0.5, 0.95), (0.45, 0.9))],
    "Blue":  [((200, 245), (0.55, 0.95), (0.45, 0.9))],
    "Other": [
        ((20, 38), (0.7, 0.95), (0.7, 0.95)),    # orange
        ((45, 62), (0.6, 0.95), (0.75, 0.95)),   # yellow
        ((265, 300), (0.45, 0.85), (0.4, 0.8)),  # purple
        ((315, 340), (0.3, 0.6), (0.75, 0.95)),  # pink
        ((0, 360), (0.0, 0.1), (0.8, 0.95)),     # white
        ((0, 360), (0.0, 0.3), (0.06, 0.18)),    # black
        ((18, 35), (0.5, 0.8), (0.25, 0.45)),    # brown
    ],
}

# Lighting / camera conditions. Each is a dict of parameter ranges.
# "standard" also describes the training data of the standard-trained CNN.
CONDITIONS = {
    "standard":    dict(gain=(0.85, 1.15), cast=None, noise=0.01, blur=(1, 3)),
    "low_light":   dict(gain=(0.28, 0.42), cast=None, noise=0.035, blur=(1, 3)),
    "warm_light":  dict(gain=(0.85, 1.1), cast=(1.15, 1.0, 0.7), noise=0.01, blur=(1, 3)),
    "cool_light":  dict(gain=(0.85, 1.1), cast=(0.78, 0.95, 1.15), noise=0.01, blur=(1, 3)),
    "shadow":      dict(gain=(0.85, 1.1), cast=None, noise=0.01, blur=(1, 3), shadow=(0.3, 0.45)),
    "glare":       dict(gain=(0.85, 1.1), cast=None, noise=0.01, blur=(1, 3), glare=(0.5, 0.8)),
    "occlusion":   dict(gain=(0.85, 1.1), cast=None, noise=0.01, blur=(1, 3), occlusion=(0.35, 0.55)),
    "motion_blur": dict(gain=(0.85, 1.1), cast=None, noise=0.01, blur=(9, 15)),
}

# Training distribution for the lighting-augmented CNN: wide random lighting,
# casts, shadows, occlusion and blur. Glare is deliberately left out, so the
# glare test measures generalisation to a condition never seen in training.
AUGMENTED_TRAINING = dict(gain=(0.25, 1.3), cast="random", noise=(0.005, 0.05), blur=(1, 13),
                          shadow_p=0.3, shadow=(0.3, 0.7), occlusion_p=0.3, occlusion=(0.2, 0.5))


def _hsv_to_rgb(h_deg, s, v):
    return np.array(colorsys.hsv_to_rgb((h_deg % 360) / 360.0, s, v))


def _sample_colour(rng, label):
    ranges = PALETTE[CLASSES[label]]
    h, s, v = ranges[rng.integers(len(ranges))]
    return _hsv_to_rgb(rng.uniform(*h), rng.uniform(*s), rng.uniform(*v))


def _belt(rng):
    base = rng.uniform(0.16, 0.26)
    x = np.arange(WIDTH)[None, :]
    seams = 0.015 * np.sin(2 * math.pi * x / rng.uniform(25, 60) + rng.uniform(0, 6.3))
    texture = rng.normal(0, 0.012, (HEIGHT, WIDTH))
    belt = np.clip(base + seams + texture, 0, 1)
    tint = rng.uniform(0.97, 1.03, 3)
    return belt[:, :, None] * tint[None, None, :]


def _object_mask(rng):
    """Return (mask float32 HxW in [0,1], centre, radius) for a random shape."""
    mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    # Objects fill roughly 10-40% of the view. The baseline detector needs a colour
    # to cover over 10% of the (resized) frame, so the camera has to be mounted
    # close to the belt; this framing matches that.
    r = int(rng.uniform(70, 100))
    cx = int(rng.uniform(r + 10, WIDTH - r - 10))
    cy = int(rng.uniform(r + 10, HEIGHT - r - 10))
    shape = rng.integers(4)
    if shape == 0:
        cv2.circle(mask, (cx, cy), r, 255, -1, cv2.LINE_AA)
    elif shape == 1:
        cv2.rectangle(mask, (cx - r, cy - int(r * 0.7)), (cx + r, cy + int(r * 0.7)), 255, -1)
    elif shape == 2:
        pts = np.array([[cx, cy - r], [cx - r, cy + int(r * 0.8)], [cx + r, cy + int(r * 0.8)]])
        cv2.fillPoly(mask, [pts.astype(np.int32)], 255, cv2.LINE_AA)
    else:  # irregular blob
        angles = np.sort(rng.uniform(0, 2 * math.pi, 9))
        radii = r * rng.uniform(0.65, 1.0, 9)
        pts = np.stack([cx + radii * np.cos(angles), cy + radii * np.sin(angles)], 1)
        cv2.fillPoly(mask, [pts.astype(np.int32)], 255, cv2.LINE_AA)
    angle = rng.uniform(0, 180)
    rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    mask = cv2.warpAffine(mask, rot, (WIDTH, HEIGHT))
    return mask.astype(np.float32) / 255.0, (cx, cy), r


def _range(rng, v):
    return rng.uniform(*v) if isinstance(v, tuple) else v


def render(label, params, rng):
    """Render one frame. Returns a uint8 BGR image."""
    img = _belt(rng)
    mask, (cx, cy), r = _object_mask(rng)

    # Object colour with simple directional shading and surface texture.
    colour = _sample_colour(rng, label)
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    theta = rng.uniform(0, 2 * math.pi)
    shade = 1 + 0.12 * ((xx - cx) * math.cos(theta) + (yy - cy) * math.sin(theta)) / r
    shade = np.clip(shade, 0.8, 1.15) * (1 + rng.normal(0, 0.02, (HEIGHT, WIDTH)))
    obj = colour[None, None, :] * shade[:, :, None]
    img = img * (1 - mask[:, :, None]) + obj * mask[:, :, None]

    # Occlusion: a neutral grey flap covering part of the object.
    occ = params.get("occlusion")
    if occ and rng.random() < params.get("occlusion_p", 1.0):
        frac = rng.uniform(*occ)
        side = rng.integers(4)
        flap = np.zeros((HEIGHT, WIDTH), np.float32)
        span = 2 * r * frac
        # The flap covers a fraction `frac` of the object's width or height,
        # entering from one side, and extends a little beyond the object.
        x0, x1 = max(0, int(cx - 1.4 * r)), int(cx + 1.4 * r)
        y0, y1 = max(0, int(cy - 1.4 * r)), int(cy + 1.4 * r)
        if side == 0:
            flap[y0:y1, x0: int(cx - r + span)] = 1
        elif side == 1:
            flap[y0:y1, int(cx + r - span): x1] = 1
        elif side == 2:
            flap[y0: int(cy - r + span), x0:x1] = 1
        else:
            flap[int(cy + r - span): y1, x0:x1] = 1
        grey = rng.uniform(0.35, 0.6)
        img = img * (1 - flap[:, :, None]) + grey * flap[:, :, None]

    # Global lighting: brightness gain and colour cast.
    img = img * _range(rng, params["gain"])
    cast = params.get("cast")
    if cast == "random":
        img = img * rng.uniform(0.65, 1.25, 3)[None, None, :]
    elif cast is not None:
        img = img * (np.array(cast) * rng.uniform(0.97, 1.03, 3))[None, None, :]

    # Shadow: a soft-edged dark band across the object.
    sh = params.get("shadow")
    if sh and rng.random() < params.get("shadow_p", 1.0):
        depth = rng.uniform(*sh)
        phi = rng.uniform(0, 2 * math.pi)
        d = (xx - cx) * math.cos(phi) + (yy - cy) * math.sin(phi) + rng.uniform(-0.3, 0.3) * r
        band = 1 / (1 + np.exp(-d / 6.0))
        img = img * (depth + (1 - depth) * band)[:, :, None]

    # Glare: a bright specular highlight on the object.
    gl = params.get("glare")
    if gl:
        peak = rng.uniform(*gl)
        gx, gy = cx + rng.uniform(-0.4, 0.4) * r, cy + rng.uniform(-0.4, 0.4) * r
        sigma = rng.uniform(0.35, 0.6) * r
        blob = peak * np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * sigma ** 2))
        img = img + blob[:, :, None]

    # Camera: horizontal motion blur (belt movement), then sensor noise.
    k = int(round(_range(rng, params["blur"])))
    if k > 1:
        kernel = np.zeros((1, k), np.float32)
        kernel[0, :] = 1.0 / k
        img = cv2.filter2D(img.astype(np.float32), -1, kernel)
    noise = _range(rng, params["noise"])
    img = img + rng.normal(0, noise, img.shape)

    rgb = np.clip(img, 0, 1)
    return cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def make_dataset(n, params, seed):
    """n frames with balanced labels. Returns (images uint8 [n,H,W,3] BGR, labels [n])."""
    rng = np.random.default_rng(seed)
    labels = np.arange(n) % len(CLASSES)
    rng.shuffle(labels)
    images = np.stack([render(int(l), params, rng) for l in labels])
    return images, labels
