"""
Compare the baseline HSV colour rule with a CNN, under different lighting.

Methods
  HSV (baseline)       detect_color() from baseline/sorter.py, unchanged
  HSV (red fixed)      the same function with its red-hue bug fixed (hsv_fixed.py)
  CNN (standard light) trained only on standard-lighting frames
  CNN (augmented)      trained on frames with randomised lighting, casts,
                       shadows, occlusion and blur (but never glare)

All settings below were fixed before any test results were seen.

Usage (from the repository root):
    python sim/evaluate.py          # about 10 minutes on a laptop CPU
"""

import csv
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "baseline"))

import cv2
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cnn import INPUT_H, INPUT_W, SortingCNN, predict, to_tensor, train
from colour_detect import detect_colour
from hsv_fixed import detect_colour_fixed
from scene import AUGMENTED_TRAINING, CLASSES, CONDITIONS, make_dataset

N_TRAIN, N_VAL, N_TEST = 8000, 1000, 600     # N_TEST frames per condition
STD_SEEDS, AUG_SEEDS = (1, 3, 5), (2, 4, 6)   # one training run per seed
CHUNK = 200
SEEDS = {"train_std": 10_000, "val_std": 20_000, "train_aug": 30_000, "val_aug": 40_000,
         "test": 50_000}
HSV_LABEL = {"Red": 0, "Green": 1, "Blue": 2, "Unknown": 3}
OUT = ROOT


def _chunk(args):
    n, params, seed, with_hsv = args
    images, labels = make_dataset(n, params, seed)
    small = np.stack([cv2.resize(f, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
                      for f in images])
    hsv = (np.array([[HSV_LABEL[detect_colour(f)], HSV_LABEL[detect_colour_fixed(f)]]
                     for f in images]) if with_hsv else None)
    return small, labels, hsv


def build(n, params, seed, with_hsv=False, pool=None):
    jobs = [(min(CHUNK, n - i), params, seed + i // CHUNK, with_hsv) for i in range(0, n, CHUNK)]
    parts = pool.map(_chunk, jobs)
    small = np.concatenate([p[0] for p in parts])
    labels = np.concatenate([p[1] for p in parts])
    hsv = np.concatenate([p[2] for p in parts]) if with_hsv else None
    return small, labels, hsv


def outcome_counts(y_true, y_pred):
    """Sorting outcomes: correct, wrong bin, missed (coloured -> Other), false sort (Other -> bin)."""
    other = CLASSES.index("Other")
    correct = (y_true == y_pred)
    wrong_bin = (y_true != other) & (y_pred != other) & ~correct
    missed = (y_true != other) & (y_pred == other)
    false_sort = (y_true == other) & (y_pred != other)
    return {k: int(v.sum()) for k, v in
            dict(correct=correct, wrong_bin=wrong_bin, missed=missed, false_sort=false_sort).items()}


def main():
    t0 = time.time()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    with Pool(max(1, os.cpu_count() or 1)) as pool:
        print("Generating training data ...")
        xs_tr, ys_tr, _ = build(N_TRAIN, CONDITIONS["standard"], SEEDS["train_std"], pool=pool)
        xs_va, ys_va, _ = build(N_VAL, CONDITIONS["standard"], SEEDS["val_std"], pool=pool)
        xa_tr, ya_tr, _ = build(N_TRAIN, AUGMENTED_TRAINING, SEEDS["train_aug"], pool=pool)
        xa_va, ya_va, _ = build(N_VAL, AUGMENTED_TRAINING, SEEDS["val_aug"], pool=pool)
        print("Generating test data ...")
        tests = {}
        for k, (name, params) in enumerate(CONDITIONS.items()):
            tests[name] = build(N_TEST, params, SEEDS["test"] + 1000 * k, with_hsv=True, pool=pool)
    print(f"Data ready in {time.time() - t0:.0f}s")

    # Each CNN is trained TRAIN_RUNS times with different seeds: behaviour on
    # conditions outside the training data can vary a lot between runs.
    runs = {}
    for name, (x_tr, y_tr, x_va, y_va), seeds in [
            ("CNN (standard light)", (xs_tr, ys_tr, xs_va, ys_va), STD_SEEDS),
            ("CNN (augmented)", (xa_tr, ya_tr, xa_va, ya_va), AUG_SEEDS)]:
        runs[name] = []
        for k, seed in enumerate(seeds):
            print(f"Training {name}, run {k + 1}/{len(seeds)} (seed {seed}) ...")
            model = SortingCNN()
            best = train(model, to_tensor(x_tr), y_tr, to_tensor(x_va), y_va, seed=seed)
            print(f"  best validation accuracy {best:.4f}")
            runs[name].append(model)
        fname = "cnn_standard.pt" if "standard" in name else "cnn_augmented.pt"
        torch.save(runs[name][0].state_dict(), os.path.join(OUT, "models", fname))

    rows, preds_by = [], {}
    for cond, (x, y, hsv_pred) in tests.items():
        preds = {("HSV (baseline)", 1): hsv_pred[:, 0], ("HSV (red fixed)", 1): hsv_pred[:, 1]}
        xt = to_tensor(x)
        for name, models in runs.items():
            for k, model in enumerate(models):
                preds[(name, k + 1)] = predict(model, xt)
        preds_by[cond] = (y, preds)
        for (method, run), p in preds.items():
            row = {"condition": cond, "method": method, "run": run, "n": len(y),
                   "accuracy": float((p == y).mean())}
            row.update(outcome_counts(y, p))
            rows.append(row)

    with open(os.path.join(OUT, "classification_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # confusion matrices (first run of each CNN) for the standard and low-light conditions
    conf = {}
    for cond in ("standard", "low_light"):
        y, preds = preds_by[cond]
        for (method, run), p in preds.items():
            if run != 1:
                continue
            m = np.zeros((4, 4), int)
            for t, q in zip(y, p):
                m[t, q] += 1
            conf[(cond, method)] = m
    np.save(os.path.join(OUT, "models", "confusion.npy"), conf, allow_pickle=True)

    plot(rows)
    print_report(rows, conf)
    print(f"\nDone in {time.time() - t0:.0f}s")


METHODS = ["HSV (baseline)", "HSV (red fixed)", "CNN (standard light)", "CNN (augmented)"]
COLOURS = {"HSV (baseline)": "#8c8c8c", "HSV (red fixed)": "#c9a227",
           "CNN (standard light)": "#4c9be8", "CNN (augmented)": "#2c6e3f"}


def summarise(rows):
    """(condition, method) -> (mean, min, max) accuracy in %, over training runs."""
    out = {}
    for r in rows:
        out.setdefault((r["condition"], r["method"]), []).append(100 * r["accuracy"])
    return {k: (float(np.mean(v)), min(v), max(v)) for k, v in out.items()}


def plot(rows):
    conds = list(CONDITIONS)
    acc = summarise(rows)
    fig, ax = plt.subplots(figsize=(13, 5.8))
    width = 0.2
    for i, m in enumerate(METHODS):
        xs = [c + (i - 1.5) * width for c in range(len(conds))]
        means = [acc[(c, m)][0] for c in conds]
        err = None
        if m.startswith("CNN"):
            err = [[acc[(c, m)][0] - acc[(c, m)][1] for c in conds],
                   [acc[(c, m)][2] - acc[(c, m)][0] for c in conds]]
        bars = ax.bar(xs, means, width, label=m, color=COLOURS[m], yerr=err,
                      error_kw=dict(elinewidth=0.8, capsize=2, ecolor="#333333"))
        for b, c in zip(bars, conds):
            top = acc[(c, m)][2] if m.startswith("CNN") else b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, top + 0.8, f"{b.get_height():.0f}",
                    ha="center", fontsize=6)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([c.replace("_", " ") + ("\n(unseen in training)" if c == "glare" else "")
                        for c in conds], fontsize=9)
    ax.set_ylabel("Classification accuracy (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Colour classification under different lighting: baseline HSV code vs CNN "
                 f"({N_TEST} synthetic frames per condition; CNN bars: mean of "
                 f"{len(STD_SEEDS)} training runs, whiskers: min-max)", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "accuracy_by_condition.png"), dpi=150)
    plt.close(fig)


def print_report(rows, conf):
    acc = summarise(rows)
    conds = list(CONDITIONS)

    def cell(c, m):
        mean, lo, hi = acc[(c, m)]
        return f"{mean:.1f}%" if not m.startswith("CNN") else f"{mean:.1f}% ({lo:.0f}–{hi:.0f})"

    print("\nAccuracy (CNN: mean of training runs, with min–max)")
    print("| Condition | " + " | ".join(METHODS) + " |")
    print("|---|" + "---|" * len(METHODS))
    for c in conds:
        print(f"| {c} | " + " | ".join(cell(c, m) for m in METHODS) + " |")
    means = {}
    for m in METHODS:
        per_run = {}
        for r in rows:
            if r["method"] == m:
                per_run.setdefault(r["run"], []).append(100 * r["accuracy"])
        run_means = [float(np.mean(v)) for v in per_run.values()]
        means[m] = (float(np.mean(run_means)), min(run_means), max(run_means))
    print("| **mean of all conditions** | " + " | ".join(
        f"**{means[m][0]:.1f}%**" + (f" ({means[m][1]:.0f}–{means[m][2]:.0f})" if m.startswith("CNN") else "")
        for m in METHODS) + " |")

    print("\nSorting outcomes, all conditions and runs combined:")
    print("| Method | Correct | Wrong bin | Missed (coloured -> Other) | False sort (Other -> bin) |")
    print("|---|---|---|---|---|")
    for m in METHODS:
        tot = {k: sum(r[k] for r in rows if r["method"] == m)
               for k in ("n", "correct", "wrong_bin", "missed", "false_sort")}
        print(f"| {m} | " + " | ".join(f"{100 * tot[k] / tot['n']:.1f}%"
                                         for k in ("correct", "wrong_bin", "missed", "false_sort")) + " |")

    for (cond, m), mat in conf.items():
        print(f"\nConfusion matrix, {cond}, {m} (run 1; rows = true, cols = predicted; {CLASSES})")
        print(mat)


if __name__ == "__main__":
    os.makedirs(os.path.join(OUT, "models"), exist_ok=True)
    main()
