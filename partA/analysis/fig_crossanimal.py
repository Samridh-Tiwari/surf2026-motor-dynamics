#!/usr/bin/env python3
"""
Generalization shown as trajectories: across days, and across animals.

The comparison is done in the model's own 15-dimensional factor space -- not in
a 2-D picture of it. That matters: a 2-D projection keeps only ~36% of the
variance for one animal, and shapes that genuinely match in 15-D look unrelated
once you throw the rest away.

Alignment is ordinary Procrustes: shift, uniform scale and rotation only. No
bending, no per-timepoint warping, nothing fitted to make the curves agree
beyond a rigid move. Disparity 0 = identical shape, 1 = no better than aligning
to a time-scrambled version of the target (the null we report alongside).
"""
import os as _os
# Paths come from environment variables so the repo runs anywhere.
# See paths.env. Defaults below are the machine this was written on.
LFADS_RUNS = _os.environ.get("LFADS_RUNS", "/path/to/lfads_runs/sweep_dyn")
SESSION_DATA = _os.environ.get("SESSION_DATA", "/path/to/MultiSessionData")
RESULTS_DIR = _os.environ.get("RESULTS_DIR", "./figures")
ROBUST_DIR = _os.environ.get("ROBUST_DIR", "./partA/results")
import numpy as np, h5py, glob, json
from scipy.spatial import procrustes
from matplotlib.ticker import MaxNLocator
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(0)
DST = RESULTS_DIR
BIN, PRECUE = 0.020, 1.5
M1, M2 = "61199SOM", "61198SOM"
BLUE, ORANGE, INK, MUTED, GRID, SURFACE = (
    "#1668d6", "#e8622a", "#111111", "#5f5e5a", "#e3e5e9", "#ffffff")
DAYCOL = ["#0b3d91", "#1668d6", "#4d9de0", "#93c6ee"]
rng = np.random.default_rng(0)


def mean_hit_paths(mouse):
    out = []
    for f in sorted(glob.glob(f"{LFADS_RUNS}/{mouse}"
                              f"/cfg06/lfads_output_*.h5")):
        with h5py.File(f) as d:
            fa = np.concatenate([d["train_factors"][:], d["valid_factors"][:]])
            c = np.concatenate([d["train_condition"][:].ravel(),
                                d["valid_condition"][:].ravel()]).astype(int)
        out.append((f.split("_")[-1].replace(".h5", ""), fa[c == 1].mean(0)))
    return out


def null_disparity(target, src, n=1000):
    return np.array([procrustes(target[rng.permutation(len(target))], src)[2]
                     for _ in range(n)])


s1, s2 = mean_hit_paths(M1), mean_hit_paths(M2)
ref_day, ref = s1[0]
cue = int(PRECUE / BIN)

# ---- A: every other day of mouse 1, rigidly aligned to its first day -------
aligned_days, dis_days, p_days = [], [], []
for day, tr in s1[1:]:
    mref, mal, d = procrustes(ref, tr)
    nl = null_disparity(ref, tr)
    aligned_days.append((day, mal)); dis_days.append(d)
    p_days.append((nl <= d).mean())
    print(f"{day} vs {ref_day}: disparity {d:.3f}  null {nl.mean():.3f}  "
          f"p<{max((nl <= d).mean(), 1/len(nl)):.4f}")
ref_std = procrustes(ref, s1[1][1])[0]        # ref is standardized identically each call

# ---- B: mouse 2 aligned onto mouse 1, separate models ----------------------
avg1 = np.stack([t for _, t in s1]).mean(0)
avg2 = np.stack([t for _, t in s2]).mean(0)
a1, a2, dis_x = procrustes(avg1, avg2)
nl_x = null_disparity(avg1, avg2)
p_x = max((nl_x <= dis_x).mean(), 1 / len(nl_x))
print(f"cross-animal: disparity {dis_x:.3f}  null {nl_x.mean():.3f}  p<{p_x:.4f}")

rb = json.load(open(f"{ROBUST_DIR}/robustness.json"))
m2j = json.load(open(f"{ROBUST_DIR}/robustness_61198_all.json"))
v1 = np.array([r["marginal"][0] for r in rb])
v2 = np.array([r["marginal"][0] for r in m2j])


# ---------------- single-panel, column-width, large fonts ------------------
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 22,
    "axes.spines.top": False, "axes.spines.right": False})
from matplotlib.ticker import MaxNLocator
fig, ax = plt.subplots(figsize=(7.4, 6.6))
cue = int(PRECUE / BIN)
ax.plot(a1[:, 0], a1[:, 1], color=BLUE, lw=5.0, solid_capstyle="round",
        label="mouse 1", zorder=5)
ax.plot(a2[:, 0], a2[:, 1], color=ORANGE, lw=4.4, ls=(0, (5, 2.5)),
        solid_capstyle="round", label="mouse 2, rotated", zorder=6)
for tr, col in [(a1, BLUE), (a2, ORANGE)]:
    ax.plot(tr[cue, 0], tr[cue, 1], "o", ms=15, color=col, mec="white", mew=2.6,
            zorder=7)
ax.set_xlabel("aligned dimension 1", fontsize=25, labelpad=8)
ax.set_ylabel("aligned dimension 2", fontsize=25, labelpad=8)
ax.xaxis.set_major_locator(MaxNLocator(4)); ax.yaxis.set_major_locator(MaxNLocator(5))
ax.legend(frameon=False, fontsize=22, loc="best")
ax.set_title("two animals, separate models", fontsize=26, weight="bold",
             loc="left", pad=12)
fig.tight_layout()
fig.savefig(f"{DST}/crossanimal.png", dpi=170, bbox_inches="tight", facecolor=SURFACE)
print("-> crossanimal.png")
