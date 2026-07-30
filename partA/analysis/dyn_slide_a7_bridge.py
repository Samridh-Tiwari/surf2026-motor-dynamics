#!/usr/bin/env python3
"""
Bridge slide (A7): can the rule generate, or only describe?

Give the trained generator ONLY the inferred starting state of each trial, set
the external input to zero, and let it run the full trial on its own. Compare
the factor trajectory it produces to the one the full model inferred.

Null: give it another trial's starting state (shuffled g0). If the rollout still
matched, the dynamics would not be carrying trial-specific information.
"""
import numpy as np, h5py, glob, json, torch
import torch.nn.functional as Fn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(0); np.random.seed(0)
DST = "/home/tiwar133/SURF_talk_figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdbd7", "#ffffff"
BIN, PRECUE, CLIP = 0.020, 1.5, 5.0
dev = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = "/home/tiwar133/lfads_runs/sweep_dyn/61199SOM/cfg06"

ck = [p for p in glob.glob(f"{ROOT}/checkpoints/*.ckpt") if "last" not in p][0]
sd = torch.load(ck, map_location="cpu")["state_dict"]
p = "decoder.rnn.cell.gen_cell."
W_ih, W_hh = sd[p + "weight_ih"].to(dev), sd[p + "weight_hh"].to(dev)
b_ih, b_hh = sd[p + "bias_ih"].to(dev), sd[p + "bias_hh"].to(dev)
# readout is KernelNormalizedLinear: rows L2-normalized before use
W_fac = Fn.normalize(sd["decoder.rnn.cell.fac_linear.weight"], p=2, dim=1).to(dev)
H = W_hh.shape[1]


def step(h, u):
    x = u @ W_ih.T + b_ih
    x_z, x_r, x_n = torch.chunk(x, 3, -1)
    w_zr, w_n = torch.split(W_hh, [2 * H, H])
    b_zr, b_n = torch.split(b_hh, [2 * H, H])
    h_z, h_r = torch.chunk(h @ w_zr.T + b_zr, 2, -1)
    z = torch.sigmoid(x_z + h_z); r = torch.sigmoid(x_r + h_r)
    n = torch.tanh(x_n + (r * h) @ w_n.T + b_n)
    return torch.clamp(z * h + (1 - z) * n, -CLIP, CLIP)


G0, FAC = [], []
for f in sorted(glob.glob(f"{ROOT}/lfads_output_*.h5")):
    with h5py.File(f) as d:
        G0.append(np.concatenate([d["train_gen_init"][:], d["valid_gen_init"][:]]))
        FAC.append(np.concatenate([d["train_factors"][:], d["valid_factors"][:]]))
G0 = torch.tensor(np.concatenate(G0), dtype=torch.float32, device=dev)
F = np.concatenate(FAC).astype(np.float64)
n, T, K = F.shape
t = np.arange(T) * BIN - PRECUE
print(f"{n} trials, {T} bins, {K} factors, generator {H}-D")


def rollout(g0):
    h = g0.clone()
    u = torch.zeros(len(g0), W_ih.shape[1], device=dev)
    out = []
    with torch.no_grad():
        for _ in range(T):
            h = step(h, u)                      # decoder steps, then reads out
            out.append((h @ W_fac.T).cpu().numpy())
    return np.stack(out, 1)


R = rollout(G0)
perm = torch.randperm(len(G0))
Rn = rollout(G0[perm])


def r2_t(pred):
    ss_res = ((F - pred) ** 2).sum(2).mean(0)
    ss_tot = ((F - F.mean(0, keepdims=True)) ** 2).sum(2).mean(0)
    return 1 - ss_res / ss_tot


r2, r2n = r2_t(R), r2_t(Rn)
cue = int(PRECUE / BIN)
print(f"R2 at start {r2[0]:.3f} | at cue {r2[cue]:.3f} | end {r2[-1]:.3f}")
print(f"mean R2 pre-cue {r2[:cue].mean():.3f}   post-cue {r2[cue:].mean():.3f}")
print(f"shuffled-start null: pre-cue {r2n[:cue].mean():.3f}  post-cue {r2n[cue:].mean():.3f}")
json.dump(dict(r2_precue=float(r2[:cue].mean()), r2_postcue=float(r2[cue:].mean()),
               null_precue=float(r2n[:cue].mean()), null_postcue=float(r2n[cue:].mean())),
          open(f"{DST}/rollout_stats.json", "w"), indent=2)

# ------------------------------------------------------------------ figure --
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 13})

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.9),
                              gridspec_kw=dict(width_ratios=(1.05, 1), wspace=0.27))
tr = int(np.argmax(np.abs(F).sum((1, 2))))
top = np.argsort(F[tr].std(0))[::-1][:4]
off = 0.0
for i, k in enumerate(top):
    sc = max(F[tr, :, k].ptp(), 1e-6)
    ax.plot(t, F[tr, :, k] / sc + off, color=INK, lw=2.3,
            label="inferred by the full model" if i == 0 else None)
    ax.plot(t, R[tr, :, k] / sc + off, color=AQUA, lw=2.3, ls="--",
            label="generated from the start state alone" if i == 0 else None)
    off += 1.55
ax.axvline(0, color=ORANGE, lw=2, ls="--", zorder=2)
ax.annotate("cue", (0, off - 0.35), fontsize=13.5, weight="bold",
            color=ORANGE, ha="center")
ax.set_yticks([]); ax.set_xlim(t[0], t[-1])
ax.set_xlabel("time from cue (s)", fontsize=13.5, labelpad=7)
ax.set_ylabel("4 example dimensions", fontsize=13.5, labelpad=10)
ax.legend(frameon=False, fontsize=11.5, loc="upper center",
          bbox_to_anchor=(0.5, -0.13), ncol=2)
ax.set_title("A   One trial, run forward from its starting state",
             fontsize=14, weight="bold", loc="left", pad=10)

ax2.axhline(0, color=MUTED, lw=1, ls=":")
ax2.fill_between(t, -1.25, r2n, color=MUTED, alpha=0.25, lw=0,
                 label="wrong starting state")
ax2.plot(t, r2, color=AQUA, lw=3.0, zorder=4, label="correct starting state")
ax2.axvline(0, color=ORANGE, lw=2, ls="--", zorder=3)
ax2.annotate("cue", (0, 1.06), fontsize=13.5, weight="bold", color=ORANGE, ha="center")
ax2.set_xlim(t[0], t[-1]); ax2.set_ylim(-1.25, 1.12)
ax2.set_xlabel("time from cue (s)", fontsize=13.5, labelpad=7)
ax2.set_ylabel("variance of the real trajectory explained", fontsize=13, labelpad=7)
ax2.yaxis.grid(True, color=GRID, lw=0.9); ax2.set_axisbelow(True)
ax2.legend(frameon=False, fontsize=11.5, loc="lower left", bbox_to_anchor=(0.02, 0.10))
ax2.set_title(f"B   All {n} trials\npre-cue {r2[:cue].mean():.2f}   "
              f"post-cue {r2[cue:].mean():.2f}",
              fontsize=14, weight="bold", loc="left", pad=10)

fig.tight_layout()
fig.savefig(f"{DST}/SlideA7_rollout.png", dpi=170, bbox_inches="tight", facecolor=SURFACE)
print("-> SlideA7_rollout.png")
