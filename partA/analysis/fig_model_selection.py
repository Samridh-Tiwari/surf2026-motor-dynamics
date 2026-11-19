#!/usr/bin/env python3
"""
Why reconstruction loss is the wrong criterion.

For every trained configuration we measure two things:

  reconstruction   how well the full model reproduces held-out spikes
                   (valid/recon_smth, already logged during the sweep)

  autonomy         run the generator forward from its inferred initial state
                   with ALL inferred input set to zero, and correlate that
                   trajectory with the one the full model produced. High
                   autonomy means the dynamics carry the trial. Low autonomy
                   means the inferred inputs are doing the work.

Reconstruction cannot tell these apart, because a trial explained by injected
input fits the spikes just as well as one explained by the learned rule. Only
models with high autonomy are usable for a claim about intrinsic dynamics.
"""
import os as _os
LFADS_RUNS = _os.environ.get("LFADS_RUNS", "/path/to/lfads_runs/sweep_dyn")
RESULTS_DIR = _os.environ.get("RESULTS_DIR", "./figures")

import numpy as np, h5py, glob, json, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(0); np.random.seed(0)
CLIP = 5.0
dev = "cuda" if torch.cuda.is_available() else "cpu"
MICE = [("61199SOM", "mouse 1"), ("61198SOM", "mouse 2")]
BLUE, ORANGE, RED, INK, MUTED, GRID, SURFACE = (
    "#1668d6", "#e8622a", "#d12b2b", "#111111", "#5f5e5a", "#e3e5e9", "#ffffff")


def step(h, W_ih, W_hh, b_ih, b_hh, H):
    u = torch.zeros(len(h), W_ih.shape[1], device=dev)
    x = u @ W_ih.T + b_ih
    xz, xr, xn = torch.chunk(x, 3, -1)
    wzr, wn = torch.split(W_hh, [2 * H, H])
    bzr, bn = torch.split(b_hh, [2 * H, H])
    hz, hr = torch.chunk(h @ wzr.T + bzr, 2, -1)
    z = torch.sigmoid(xz + hz); r = torch.sigmoid(xr + hr)
    return torch.clamp(z * h + (1 - z) * torch.tanh(xn + (r * h) @ wn.T + bn),
                       -CLIP, CLIP)


def autonomy(mouse, cfg):
    root = f"{LFADS_RUNS}/{mouse}/{cfg}"
    outs = sorted(glob.glob(f"{root}/lfads_output_*.h5"))
    cks = [p for p in glob.glob(f"{root}/checkpoints/*.ckpt") if "last" not in p]
    if not outs or not cks:
        return None
    sd = torch.load(cks[0], map_location="cpu")["state_dict"]
    pf = "decoder.rnn.cell.gen_cell."
    W_ih, W_hh = sd[pf + "weight_ih"].to(dev), sd[pf + "weight_hh"].to(dev)
    b_ih, b_hh = sd[pf + "bias_ih"].to(dev), sd[pf + "bias_hh"].to(dev)
    H = W_hh.shape[1]
    cors = []
    for f in outs:
        with h5py.File(f) as d:
            gs = np.concatenate([d["train_gen_states"][:], d["valid_gen_states"][:]])
            g0 = np.concatenate([d["train_gen_init"][:], d["valid_gen_init"][:]])
        h = torch.tensor(g0, dtype=torch.float32, device=dev)
        out = []
        with torch.no_grad():
            for _ in range(gs.shape[1]):
                h = step(h, W_ih, W_hh, b_ih, b_hh, H)
                out.append(h.cpu().numpy())
        roll = np.stack(out, 1)
        cors.append(np.corrcoef(roll.ravel(), gs.ravel())[0, 1])
    return float(np.mean(cors))


rows = []
for mouse, label in MICE:
    # final validation reconstruction from each run's own training log:
    # complete for all 10 configs, unlike the batched sweep_results JSONs
    import csv
    recon = {}
    for c in sorted(glob.glob(f"{LFADS_RUNS}/{mouse}/cfg*/csv_logs/metrics.csv")):
        lg = [r for r in csv.DictReader(open(c)) if r.get("valid/recon_smth")]
        if lg:
            recon[c.split("/")[-3]] = float(lg[-1]["valid/recon_smth"])
    for cfg in sorted(recon):
        a = autonomy(mouse, cfg)
        if a is None:
            continue
        rows.append(dict(mouse=mouse, label=label, cfg=cfg,
                         recon=recon[cfg], autonomy=a))
        print(f"{label} {cfg}: recon {recon[cfg]:.4f}  autonomy {a:.3f}")

json.dump(rows, open(f"{RESULTS_DIR}/model_selection_stats.json", "w"), indent=2)

# ------------------------------------------------------------------ figure --
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 15,
    "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2), gridspec_kw=dict(wspace=0.24))

for ax, (mouse, label) in zip(axes, MICE):
    R = [r for r in rows if r["mouse"] == mouse]
    if not R:
        continue
    x = np.array([r["recon"] for r in R]); y = np.array([r["autonomy"] for r in R])
    names = [r["cfg"] for r in R]
    best_recon = names[int(np.argmin(x))]
    for xi, yi, nm in zip(x, y, names):
        if nm == "cfg06":
            ax.scatter(xi, yi, s=340, color=BLUE, edgecolor=INK, lw=2.2, zorder=6)
            ax.annotate("cfg06\nselected", (xi, yi), xytext=(12, -6),
                        textcoords="offset points", fontsize=15, weight="bold",
                        color=BLUE)
        elif nm == best_recon:
            ax.scatter(xi, yi, s=340, color=RED, edgecolor=INK, lw=2.2,
                       marker="s", zorder=6)
            ax.annotate(f"{nm}\nbest reconstruction", (xi, yi), xytext=(12, 4),
                        textcoords="offset points", fontsize=15, weight="bold",
                        color=RED)
        else:
            ax.scatter(xi, yi, s=150, color=MUTED, alpha=0.55, lw=0, zorder=4)
    ax.axhline(0.9, color=MUTED, lw=1.2, ls=":")
    ax.annotate("usable for a dynamics claim", (ax.get_xlim()[0], 0.905),
                xytext=(4, 3), textcoords="offset points", fontsize=13,
                color=MUTED)
    ax.set_xlabel("reconstruction loss  (lower is better)", fontsize=16, labelpad=8)
    ax.set_ylabel("autonomy: trajectory reproduced\nwith all input removed",
                  fontsize=16, labelpad=8)
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, color=GRID, lw=1.0); ax.set_axisbelow(True)
    ax.set_title(label, fontsize=18, weight="bold", loc="left", pad=10)

fig.suptitle("The best-reconstructing model is not the one that learned the rule",
             fontsize=20, weight="bold", y=1.0)
fig.savefig(f"{RESULTS_DIR}/model_selection.png", dpi=170, bbox_inches="tight",
            facecolor=SURFACE)
print("-> model_selection.png")
