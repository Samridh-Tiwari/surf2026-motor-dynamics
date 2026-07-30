#!/usr/bin/env python3
"""
Slide A9: what moves the state along the groove?

The model is never told when the cue happened, or which trials succeeded. It is
given spikes only. Yet the input it infers -- the part of the activity the
generator could not produce on its own -- spikes ~0.1 s after the cue, in both
animals, and is larger on successful trials.

Null: permute Hit/Miss labels 2000x and recompute the effect.
"""
import numpy as np, h5py, glob, json
from scipy.stats import mannwhitneyu
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DST = "/home/tiwar133/SURF_talk_figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdbd7", "#ffffff"
DT, ONSET, SKIP, CFG, NPERM = 0.020, 75, 12, "cfg06", 2000
rng = np.random.default_rng(0)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 13})

fig, axes = plt.subplots(1, 2, figsize=(14.6, 6.0), sharex=True,
                         gridspec_kw=dict(wspace=0.24))
stats = {}
for ax, (mouse, label) in zip(axes, [("61199SOM", "mouse 1"), ("61198SOM", "mouse 2")]):
    U, C = [], []
    for p in sorted(glob.glob(
            f"/home/tiwar133/lfads_runs/sweep_dyn/{mouse}/{CFG}/lfads_output_*.h5")):
        with h5py.File(p) as f:
            U.append(np.concatenate([f["train_gen_inputs"][:], f["valid_gen_inputs"][:]]))
            C.append(np.concatenate([f["train_condition"][:].ravel(),
                                     f["valid_condition"][:].ravel()]))
    U = np.concatenate(U); C = np.concatenate(C).astype(int)
    t = (np.arange(U.shape[1]) - ONSET) * DT
    nrm = np.linalg.norm(U, axis=2)

    post = (t > 0) & (t <= 0.4); pre = (t >= -1.0) & (t <= -0.1)
    resp = nrm[:, post].max(1) - nrm[:, pre].mean(1)
    rh, rm = resp[C == 1], resp[C == 0]
    u_st, p = mannwhitneyu(rh, rm, alternative="greater")
    cles = u_st / (len(rh) * len(rm))
    null = np.array([(lambda cp: mannwhitneyu(resp[cp == 1], resp[cp == 0],
                                              alternative="greater")[0]
                      / ((cp == 1).sum() * (cp == 0).sum()))(rng.permutation(C))
                     for _ in range(NPERM)])
    mu_all = nrm.mean(0)
    base = mu_all[pre].mean(); peak = mu_all[post].max()
    peak_t = t[post][mu_all[post].argmax()]
    stats[mouse] = dict(cles=float(cles), p=float(p),
                        null_95=float(np.percentile(null, 95)),
                        peak_over_baseline=float(peak / base), peak_t=float(peak_t),
                        n_hit=int(len(rh)), n_miss=int(len(rm)))
    print(f"{label}: peak {peak/base:.2f}x baseline at {peak_t:+.2f}s | "
          f"P(Hit>Miss) {cles:.3f} (null 95th {np.percentile(null,95):.3f}) p={p:.1e}")

    for lab, m, col in [("Hit", C == 1, ORANGE), ("Miss", C == 0, BLUE)]:
        mu = nrm[m].mean(0)[SKIP:]; se = (nrm[m].std(0) / np.sqrt(m.sum()))[SKIP:]
        ax.fill_between(t[SKIP:], mu - 1.96 * se, mu + 1.96 * se,
                        color=col, alpha=0.22, lw=0)
        ax.plot(t[SKIP:], mu, color=col, lw=2.8, zorder=4,
                label=f"{lab}  (n={int(m.sum())})")
    ax.axvline(0, color=INK, lw=2, ls="--", zorder=3)
    ax.annotate("cue", (0, ax.get_ylim()[1]), xytext=(0, -4),
                textcoords="offset points", ha="center", va="top",
                fontsize=14, weight="bold", color=INK)
    ax.set_xlim(t[SKIP], t[-1])
    ax.set_xlabel("time from cue (s)", fontsize=13.5, labelpad=7)
    ax.set_ylabel("inferred external input  $\\|u_t\\|$", fontsize=13.5, labelpad=7)
    ax.yaxis.grid(True, color=GRID, lw=0.9); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=12, loc="upper left")
    ax.set_title(f"{label}\npeak {peak/base:.1f}x baseline at {peak_t*1000:+.0f} ms   ·   "
                 f"P(Hit>Miss) {cles:.2f}",
                 fontsize=13.5, weight="bold", loc="left", pad=10)

fig.tight_layout()
fig.savefig(f"{DST}/SlideA9_cue_is_the_push.png", dpi=170,
            bbox_inches="tight", facecolor=SURFACE)
json.dump(stats, open(f"{DST}/cue_stats_cfg06.json", "w"), indent=2)
print("-> SlideA9_cue_is_the_push.png")
