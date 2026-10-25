"""
Conveyor throughput simulation: how many items per minute can be sorted
before items start going into the wrong bin?

Model (all timings are stated assumptions, not measurements of the real belt):
  * Items reach the camera one at a time. Their spacing averages 60/r seconds
    for a feed rate of r items per minute, varying randomly by +/-30%, as hand-
    or hopper-fed items would.
  * Classification takes the latency measured on this machine for each method.
    A command ('R', 'G', 'B') is sent when classification finishes; nothing
    is sent for Other.
  * Three diverters sit along the belt after the camera. An item reaches the
    red, green and blue diverters 2.0, 2.5 and 3.0 s after the camera,
    passing them in that order. An open diverter pushes whatever item is
    passing into its bin. An item that passes no open diverter ends in the
    Other bin at the end of the belt.
  * Blocking firmware (baseline/sketch_dec23a.ino): commands are handled one
    at a time; each opens its diverter at once for 5 s, and the board can do
    nothing else meanwhile (delay(5000)).
  * Non-blocking firmware: each diverter opens only around the time its own
    item is due: a 0.6 s window centred on (command time + travel time), as
    in arduino/sorter_nonblocking.ino.

To isolate the mechanics, every item is classified correctly here. End-to-end
accuracy is then classification accuracy times the mechanical success rate.

Usage:
    python sim/conveyor.py
"""

import csv
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "baseline"))

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRAVEL = {0: 2.0, 1: 2.5, 2: 3.0}   # seconds from camera to the red/green/blue diverter
OTHER = 3
BLOCKING_HOLD = 5.0                  # delay(5000) in baseline/sketch_dec23a.ino
NONBLOCKING_HOLD = 0.6               # HOLD_MS in arduino/sorter_nonblocking.ino
SPACING_JITTER = 0.3
N_ITEMS = 2000
RATES = list(range(5, 201, 5))
TARGET = 0.99


def item_times(rate, n, rng):
    gaps = 60.0 / rate * rng.uniform(1 - SPACING_JITTER, 1 + SPACING_JITTER, n)
    return np.cumsum(gaps)


def simulate(rate, firmware, latency, seed=0, n=N_ITEMS):
    """Fraction of items that end in the right bin."""
    rng = np.random.default_rng(seed)
    t_cam = item_times(rate, n, rng)
    labels = rng.integers(0, 4, n)

    windows = {0: [], 1: [], 2: []}   # open intervals per diverter
    board_free = 0.0
    for t, c in zip(t_cam, labels):
        if c == OTHER:
            continue
        t_cmd = t + latency
        if firmware == "blocking":
            start = max(t_cmd, board_free)        # waits for any earlier delay(5000)
            board_free = start + BLOCKING_HOLD
            windows[c].append((start, start + BLOCKING_HOLD))
        else:
            # as in the firmware: timed from command arrival, centred on expected arrival
            open_at = t_cmd + TRAVEL[c] - NONBLOCKING_HOLD / 2
            windows[c].append((open_at, open_at + NONBLOCKING_HOLD))

    starts = {c: np.array([w[0] for w in windows[c]]) for c in windows}
    ends = {c: np.array([w[1] for w in windows[c]]) for c in windows}

    correct = 0
    for t, c in zip(t_cam, labels):
        landed = OTHER
        for gate in (0, 1, 2):                    # passes the diverters in order
            t_pass = t + TRAVEL[gate]
            if len(starts[gate]) and np.any((starts[gate] <= t_pass) & (t_pass <= ends[gate])):
                landed = gate
                break
        correct += landed == c
    return correct / n


def measure_latency():
    """Median per-frame classification time on this machine, at webcam resolution."""
    import torch
    from cnn import SortingCNN, preprocess
    from colour_detect import detect_colour
    from scene import CONDITIONS, make_dataset

    frames, _ = make_dataset(50, CONDITIONS["standard"], seed=7)
    import cv2
    frames = [cv2.resize(f, (640, 480)) for f in frames]   # webcam-sized frames

    def timeit(fn):
        ts = []
        for f in frames:
            s = time.perf_counter()
            fn(f)
            ts.append(time.perf_counter() - s)
        return float(np.median(ts))

    model = SortingCNN()
    path = os.path.join(ROOT, "models", "cnn_augmented.pt")
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
    model.eval()
    torch.set_num_threads(1)

    @torch.no_grad()
    def cnn(f):
        return model(preprocess([f])).argmax(1)

    return {"HSV (baseline)": timeit(detect_colour), "CNN": timeit(cnn)}


def max_rate(firmware, latency):
    ok = [r for r in RATES if simulate(r, firmware, latency) >= TARGET]
    # largest rate such that every rate up to it also meets the target
    best = 0
    for r in RATES:
        if r in ok:
            best = r
        else:
            break
    return best


def main():
    lat = measure_latency()
    print("Median classification time per 640x480 frame on this machine:")
    for k, v in lat.items():
        print(f"  {k}: {1000 * v:.1f} ms")

    rows = []
    curves = {}
    for firmware in ("blocking", "nonblocking"):
        for method, latency in lat.items():
            curve = [simulate(r, firmware, latency) for r in RATES]
            curves[(firmware, method)] = curve
            for r, s in zip(RATES, curve):
                rows.append({"firmware": firmware, "method": method, "rate_per_min": r,
                             "latency_s": latency, "mechanical_success": s})

    with open(os.path.join(ROOT, "throughput_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"\nHighest feed rate with at least {TARGET:.0%} of items in the right bin:")
    print("| Firmware | Classifier | Max items/min |")
    print("|---|---|---|")
    for firmware in ("blocking", "nonblocking"):
        for method, latency in lat.items():
            print(f"| {firmware} | {method} | {max_rate(firmware, latency)} |")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    styles = {"blocking": ("#8c8c8c", "Baseline firmware (blocking 5 s hold)"),
              "nonblocking": ("#2c6e3f", "Non-blocking firmware (timed 0.6 s window)")}
    for firmware, (colour, label) in styles.items():
        ax.plot(RATES, [100 * s for s in curves[(firmware, "CNN")]], color=colour,
                marker="o", markersize=3, label=label + ", CNN")
        ax.plot(RATES, [100 * s for s in curves[(firmware, "HSV (baseline)")]], color=colour,
                linestyle="--", linewidth=1, label=label + ", HSV")
    ax.axhline(100 * TARGET, color="#d62728", linewidth=1, linestyle=":")
    ax.text(RATES[-1], 100 * TARGET, f"{TARGET:.0%} target ", ha="right", va="bottom",
            fontsize=8, color="#d62728")
    ax.set_xlabel("Feed rate (items per minute)")
    ax.set_ylabel("Items in the correct bin (%), assuming correct classification")
    ax.set_title("Simulated sorting throughput: firmware timing is the bottleneck")
    ax.set_ylim(0, 102)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "throughput.png"), dpi=150)
    plt.close(fig)
    print("\nSaved throughput_results.csv, throughput.png")


if __name__ == "__main__":
    main()
