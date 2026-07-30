#!/usr/bin/env python3
"""
Slide A7: what shape is the learned rule?

A  Jacobian eigenvalues at the network's fixed points, complex plane
B  distribution of |lambda|, with the decay time it implies
C  marginal-mode count across every trained model and both mice

Everything from exact autograd Jacobians of the trained generator. No cartoons.
"""
import numpy as np, h5py, glob, json, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

torch.manual_seed(0); np.random.seed(0)
DST = RESULTS_DIR
OUT = ROBUST_DIR
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdbd7", "#ffffff"
DT, MARG, CLIP = 0.020, 0.01, 5.0
dev = "cuda" if torch.cuda.is_available() else "cpu"


class GenCell(torch.nn.Module):
    def __init__(s, sd):
        super().__init__()
        p = "decoder.rnn.cell.gen_cell."
        s.w_ih = sd[p + "weight_ih"].to(dev); s.w_hh = sd[p + "weight_hh"].to(dev)
        s.b_ih = sd[p + "bias_ih"].to(dev);   s.b_hh = sd[p + "bias_hh"].to(dev)
        s.H = s.w_hh.shape[1]
    def forward(s, u, h):
        x = u @ s.w_ih.T + s.b_ih
        x_z, x_r, x_n = torch.chunk(x, 3, -1)
        w_zr, w_n = torch.split(s.w_hh, [2 * s.H, s.H])
        b_zr, b_n = torch.split(s.b_hh, [2 * s.H, s.H])
        h_z, h_r = torch.chunk(h @ w_zr.T + b_zr, 2, -1)
        z = torch.sigmoid(x_z + h_z); r = torch.sigmoid(x_r + h_r)
        n = torch.tanh(x_n + (r * h) @ w_n.T + b_n)
        return torch.clamp(z * h + (1 - z) * n, -CLIP, CLIP)


ROOT = f"{LFADS_RUNS}/61199SOM/cfg06"
ck = [p for p in glob.glob(f"{ROOT}/checkpoints/*.ckpt") if "last" not in p][0]
cell = GenCell(torch.load(ck, map_location="cpu")["state_dict"])
G = np.concatenate([np.concatenate([h5py.File(p)["train_gen_states"][:],
                                    h5py.File(p)["valid_gen_states"][:]])
                    for p in sorted(glob.glob(f"{ROOT}/lfads_output_*.h5"))])
flat = G.reshape(-1, cell.H).astype(np.float32)

NSEED, ITERS = 512, 12000
idx = np.random.default_rng(0).choice(len(flat), NSEED, replace=False)
g = torch.nn.Parameter(torch.tensor(flat[idx], device=dev)
                       + 0.05 * torch.randn(NSEED, cell.H, device=dev))
u = torch.zeros(NSEED, cell.w_ih.shape[1], device=dev)
opt = torch.optim.Adam([g], lr=2e-2)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ITERS, eta_min=1e-5)
for i in range(ITERS):
    opt.zero_grad()
    ((0.5 * ((cell(u, g) - g) ** 2).sum(-1)).sum()).backward()
    opt.step(); sch.step()
with torch.no_grad():
    keep = (torch.norm(cell(u, g) - g, dim=-1)
            / torch.norm(g, dim=-1).median()) < 1e-3
FP = g.detach()[keep]
print(f"{len(FP)} fixed points from {NSEED} seeds")

sel = np.random.default_rng(1).choice(len(FP), min(40, len(FP)), replace=False)
eig = []
for i in sel:
    J = torch.autograd.functional.jacobian(
        lambda h: cell(u[:1], h).squeeze(0), FP[i:i + 1], vectorize=True
    ).squeeze().cpu().numpy()
    eig.append(np.linalg.eigvals(J))
E = np.concatenate(eig)
mag = np.abs(E)
n_marg = (np.abs(mag - 1) < MARG).sum() / len(sel)
n_stab = (mag < 1 - MARG).sum() / len(sel)
n_unst = (mag > 1 + MARG).sum() / len(sel)
tau = -DT / np.log(np.clip(mag[mag < 1 - MARG], 1e-9, None))
print(f"per fixed point: {n_marg:.1f} marginal, {n_stab:.1f} stable, {n_unst:.2f} unstable")
print(f"stable decay time: median {np.median(tau)*1000:.0f} ms")

rb = json.load(open(f"{OUT}/robustness.json"))
m2 = json.load(open(f"{OUT}/robustness_61198_all.json"))

# ------------------------------------------------------------------ figure --
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 12.5})

fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(16.6, 5.7),
                                   gridspec_kw=dict(width_ratios=(1, 1.05, 0.95),
                                                    wspace=0.30))
is_m = np.abs(mag - 1) < MARG
ax.add_patch(Circle((0, 0), 1, fill=False, color=INK, lw=1.8, ls="--", zorder=3))
ax.scatter(E[~is_m].real, E[~is_m].imag, s=9, color=BLUE, alpha=0.30,
           lw=0, zorder=4, label="decays back")
ax.scatter(E[is_m].real, E[is_m].imag, s=22, color=AQUA, alpha=0.85,
           lw=0, zorder=5, label="persists")
ax.axhline(0, color=GRID, lw=0.9, zorder=1); ax.axvline(0, color=GRID, lw=0.9, zorder=1)
ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.25); ax.set_aspect("equal")
ax.set_xlabel("real part", fontsize=13, labelpad=6)
ax.set_ylabel("imaginary part", fontsize=13, labelpad=6)
ax.legend(frameon=False, fontsize=11.5, loc="upper left",
          markerscale=1.8, borderaxespad=0.2, handletextpad=0.4)
ax.set_title("A   Eigenvalues at the fixed points\nunit circle dashed",
             fontsize=13.5, weight="bold", loc="left", pad=10)

ax2.hist(mag, bins=np.linspace(0, 1.05, 70), color=BLUE, alpha=0.75, lw=0)
ax2.axvspan(1 - MARG, 1.05, color=AQUA, alpha=0.30, lw=0)
ymax = ax2.get_ylim()[1]
ax2.annotate(f"{n_marg:.0f} persist", (1.0, ymax * 0.97), ha="center", va="top",
             fontsize=13.5, weight="bold", color=AQUA,
             xytext=(-4, 0), textcoords="offset points")
ax2.annotate(f"{n_stab:.0f} decay back\nmedian {np.median(tau)*1000:.0f} ms",
             (0.30, ymax * 0.55), fontsize=13.5, weight="bold", color=BLUE)
ax2.set_xlim(0, 1.05)
ax2.set_xlabel("|eigenvalue|", fontsize=13, labelpad=6)
ax2.set_ylabel("count", fontsize=13, labelpad=6)
ax2.yaxis.grid(True, color=GRID, lw=0.9); ax2.set_axisbelow(True)
ax2.set_title("B   Two populations, no unstable ones\n"
              f"{n_unst:.2f} above 1 per fixed point",
              fontsize=13.5, weight="bold", loc="left", pad=10)

SHOWN = "cfg06"          # the model presented throughout Part A
groups = [("mouse 1", rb, AQUA, True), ("mouse 2", m2, ORANGE, False)]
rng2 = np.random.default_rng(3)
for gi, (name, recs, col, mark) in enumerate(groups):
    v = np.array([r["marginal"][0] for r in recs])
    is_shown = np.array([mark and r["config"] == SHOWN for r in recs])
    jit = rng2.uniform(-0.13, 0.13, len(v))
    ax3.scatter(gi + jit[~is_shown], v[~is_shown], s=115, color=col, alpha=0.55,
                edgecolor="white", lw=1.5, zorder=4)
    ax3.scatter(gi + jit[is_shown], v[is_shown], s=250, color=col, marker="D",
                edgecolor=INK, lw=1.8, zorder=6)
    ax3.plot([gi - 0.27, gi + 0.27], [np.median(v)] * 2, color=INK, lw=2.8, zorder=5)
    ax3.annotate(f"{np.median(v):.0f}", (gi + 0.30, np.median(v)), fontsize=15,
                 weight="bold", color=col, va="center")
ax3.set_xticks([0, 1]); ax3.set_xticklabels(["mouse 1", "mouse 2"], fontsize=14)
ax3.set_xlim(-0.55, 1.62); ax3.set_ylim(0, 60)
ax3.set_ylabel("directions that persist", fontsize=13, labelpad=6)
ax3.yaxis.grid(True, color=GRID, lw=0.9); ax3.set_axisbelow(True)
ax3.scatter([], [], s=150, color=MUTED, marker="D", edgecolor=INK, lw=1.6,
            label="model in panels A-B")
ax3.scatter([], [], s=90, color=MUTED, alpha=0.55, label="other models")
ax3.legend(frameon=False, fontsize=10.5, loc="upper right", handletextpad=0.3)
ax3.set_title("C   5 models per mouse\nbar = median", fontsize=13.5,
              weight="bold", loc="left", pad=10)

fig.tight_layout()
fig.savefig(f"{DST}/SlideA8_groove.png", dpi=170, bbox_inches="tight", facecolor=SURFACE)
print("-> SlideA8_groove.png")
json.dump(dict(n_marginal=float(n_marg), n_stable=float(n_stab),
               n_unstable=float(n_unst), tau_median_ms=float(np.median(tau) * 1000),
               n_fixed_points=int(len(FP)), ambient=int(cell.H)),
          open(f"{DST}/groove_stats.json", "w"), indent=2)
import os as _os
# Paths are read from environment variables so the repo runs anywhere.
# Set them once (see README), or edit the defaults below.
LFADS_RUNS = _os.environ.get("LFADS_RUNS", "/home/tiwar133/lfads_runs/sweep_dyn")
SESSION_DATA = _os.environ.get("SESSION_DATA", "/mnt/c/Users/tiwar133/gpfa_v0203/MultiSessionData")
RESULTS_DIR = _os.environ.get("RESULTS_DIR", "/home/tiwar133/SURF_talk_figures")
ROBUST_DIR = _os.environ.get("ROBUST_DIR", "/home/tiwar133/dynamics_final")
