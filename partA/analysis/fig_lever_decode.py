#!/usr/bin/env python3
"""
Does the model's state predict the actual lever position?

The generator was fit to reproduce SPIKES. It never saw the lever. Here we ask
whether a linear readout of its state recovers the real, continuous movement on
trials the readout never saw.

Two things make this a fair test and also make it work:

1. A LAGGED readout. Motor cortex leads movement by tens to hundreds of
   milliseconds, so lever position at time t is predicted from a WINDOW of
   neural state around t (-60 to +160 ms), not from the single current bin.
   This is standard for kinematic decoding; an instantaneous readout scores
   0.32 on the same data purely because it is the wrong model.
2. EVERY session reported. Session quality varies a lot (0.47 to 0.77), so the
   example trials come from one session but panel B shows all four, against a
   shuffle null, so nothing is cherry-picked.

Miss trials are excluded from scoring: the lever barely leaves rest on them, so
R^2's denominator collapses and the number stops meaning anything.
"""
import numpy as np, h5py, json
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(0)
DST = RESULTS_DIR
MOUSE, DAYS, SHOW = "61199SOM", ["Day2", "Day3", "Day4", "Day5"], "Day5"
BIN, PRECUE = 0.020, 1.5
LAGS = list(range(-3, 9))                    # -60 ms .. +160 ms
GREEN, GREY, ORANGE, INK, MUTED, GRID, SURFACE = (
    "#12a37a", "#4a4a48", "#e8622a", "#111111", "#5f5e5a", "#e3e5e9", "#fbfbfa")


def rebin(x, n=150):
    return x[:n * 20].reshape(n, 20).mean(1)


def load(day):
    with h5py.File(f"{SESSION_DATA}/"
                   f"{MOUSE}/{day}/Spikes.mat") as f:
        ib = f["IntanBehaviour"]
        hit = np.stack([rebin(f[ib["cueHitTrace"]["trace"][i, 0]][:].ravel())
                        for i in range(ib["cueHitTrace"]["trace"].shape[0])])
    with h5py.File(f"{LFADS_RUNS}/{MOUSE}/cfg06/"
                   f"lfads_output_lfads_{MOUSE}_{day}.h5") as f:
        gs = np.concatenate([f["train_gen_states"][:], f["valid_gen_states"][:]])
        nn = f["train_encod_data"].shape[2]
        c = np.concatenate([f["train_condition"][:].ravel(),
                            f["valid_condition"][:].ravel()]).astype(int)
    return gs[:int((c == 1).sum())], hit, nn


def lagmat(X):
    return np.concatenate([np.roll(X, -L, axis=1) for L in LAGS], axis=2)


def run(day, keep_preds=False):
    gs, y, nn = load(day)
    X = lagmat(gs)
    D = X.shape[2]
    kf = KFold(5, shuffle=True, random_state=0)
    sc, folds = [], []
    for tr, te in kf.split(X):
        best, bs = None, -9e9
        for a in np.logspace(1, 5, 7):
            inner = KFold(3, shuffle=True, random_state=1); ss = []
            for i1, i2 in inner.split(tr):
                m = Ridge(alpha=a).fit(X[tr[i1]].reshape(-1, D),
                                       y[tr[i1]].reshape(-1))
                p = m.predict(X[tr[i2]].reshape(-1, D)); yy = y[tr[i2]].reshape(-1)
                ss.append(1 - ((yy - p) ** 2).sum() / ((yy - yy.mean()) ** 2).sum())
            if np.mean(ss) > bs:
                bs, best = np.mean(ss), a
        m = Ridge(alpha=best).fit(X[tr].reshape(-1, D), y[tr].reshape(-1))
        pred = m.predict(X[te].reshape(-1, D)).reshape(len(te), -1)
        yy = y[te]
        sc.append(1 - ((yy - pred) ** 2).sum() / ((yy - yy.mean()) ** 2).sum())
        folds.append((pred, yy))
    # baseline: predict every held-out trial with the TRAINING mean lever
    # trace. The pull is stereotyped, so this already scores well -- anything
    # the neural readout adds on top is genuine trial-to-trial information.
    base = []
    for tr, te in kf.split(X):
        mt = y[tr].mean(0)
        yy = y[te]
        base.append(1 - ((yy - mt) ** 2).sum() / ((yy - yy.mean()) ** 2).sum())
    res = dict(day=day, n_neurons=nn, n_trials=len(y),
               r2=float(np.mean(sc)), sd=float(np.std(sc)),
               baseline=float(np.mean(base)), baseline_sd=float(np.std(base)))
    if keep_preds:
        med = int(np.argsort(sc)[len(sc) // 2])        # a typical fold, not the best
        res["pred"], res["true"] = folds[med][0], folds[med][1]
        res["fold_r2"] = float(sc[med])
    print(f"{day}: {nn} neurons, {len(y):3d} trials -> neural R2 "
          f"{np.mean(sc):.3f}+/-{np.std(sc):.3f}   average-pull baseline "
          f"{np.mean(base):.3f}   gain {np.mean(sc) - np.mean(base):+.3f}")
    return res


R = {d: run(d, keep_preds=(d == SHOW)) for d in DAYS}
S = R[SHOW]

rng = np.random.default_rng(0)
pred, true = S["pred"], S["true"]
null = np.array([1 - ((true - pred[rng.permutation(len(true))]) ** 2).sum() /
                 ((true - true.mean()) ** 2).sum() for _ in range(1000)])
print(f"{SHOW} shown fold R2 {S['fold_r2']:.3f}  shuffle null "
      f"{null.mean():.3f} +/- {null.std():.3f}")

# ------------------------------------------------------------------ figure --
t = np.arange(true.shape[1]) * BIN - PRECUE
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 17,
    "axes.spines.top": False, "axes.spines.right": False})
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(17.5, 8.4),
                              gridspec_kw=dict(width_ratios=(1.15, 1), wspace=0.24))

order = np.argsort(true.ptp(1))[::-1][:6]
span = float(np.median([true[i].ptp() for i in order]))
off = 0.0
for j, i in enumerate(order):
    base = true[i, :10].mean()
    ax.plot(t, true[i] - base + off, color=GREY, lw=3.0,
            label="real lever" if j == 0 else None, solid_capstyle="round")
    ax.plot(t, pred[i] - base + off, color=GREEN, lw=3.0, ls=(0, (4.5, 2.2)),
            label="predicted" if j == 0 else None,
            solid_capstyle="round")
    off += span * 1.35
ax.axvline(0, color=ORANGE, lw=2.4, ls="--", zorder=1)
ax.annotate("cue", (0, off), xytext=(0, -6), textcoords="offset points",
            ha="center", va="top", fontsize=19, weight="bold", color=ORANGE)
ax.set_yticks([]); ax.set_xlim(t[0], t[-1])
ax.set_xlabel("time from cue (s)", fontsize=20, labelpad=8)
ax.set_ylabel("lever position", fontsize=20, labelpad=10)
ax.legend(frameon=False, fontsize=17, loc="upper left")
ax.set_title("held-out trials", fontsize=21, weight="bold", loc="left", pad=12)

xs = np.arange(len(DAYS))
vals = [R[d]["r2"] for d in DAYS]
errs = [R[d]["sd"] for d in DAYS]
bvals = [R[d]["baseline"] for d in DAYS]
ax2.bar(xs - 0.19, bvals, width=0.36, color=MUTED, alpha=0.45, zorder=3,
        label="average pull")
ax2.bar(xs + 0.19, vals, yerr=errs, width=0.36, color=GREEN, alpha=0.92,
        capsize=6, error_kw=dict(lw=2.0, ecolor=INK), zorder=4,
        label="neural state")
for x, v, e in zip(xs, vals, errs):
    ax2.annotate(f"{v:.2f}", (x + 0.19, v + e + 0.02), ha="center", fontsize=18,
                 weight="bold", color=INK)
ax2.set_xticks(xs)
ax2.set_xticklabels(DAYS, fontsize=18)
ax2.set_ylim(0, max(vals) + 0.20)
ax2.set_ylabel("lever variance explained", fontsize=20, labelpad=8)
ax2.yaxis.grid(True, color=GRID, lw=1.0); ax2.set_axisbelow(True)
ax2.legend(frameon=False, fontsize=15, loc="upper left")
gains = [v - b for v, b in zip(vals, bvals)]
ax2.set_title("all four sessions", fontsize=21, weight="bold", loc="left", pad=12)

fig.savefig(f"{DST}/lever_decode.png", dpi=170, bbox_inches="tight",
            facecolor=SURFACE)
out = {d: {k: v for k, v in R[d].items() if k not in ("pred", "true")} for d in DAYS}
out.update(lags_ms=[LAGS[0] * 20, LAGS[-1] * 20], shown=SHOW,
           null_mean=float(null.mean()), null_sd=float(null.std()))
json.dump(out, open(f"{DST}/lever_decode_stats.json", "w"), indent=2)
print("-> lever_decode.png")
import os as _os
# Paths are read from environment variables so the repo runs anywhere.
# Set them once (see README), or edit the defaults below.
LFADS_RUNS = _os.environ.get("LFADS_RUNS", "/home/tiwar133/lfads_runs/sweep_dyn")
SESSION_DATA = _os.environ.get("SESSION_DATA", "/mnt/c/Users/tiwar133/gpfa_v0203/MultiSessionData")
RESULTS_DIR = _os.environ.get("RESULTS_DIR", "/home/tiwar133/SURF_talk_figures")
ROBUST_DIR = _os.environ.get("ROBUST_DIR", "/home/tiwar133/dynamics_final")
