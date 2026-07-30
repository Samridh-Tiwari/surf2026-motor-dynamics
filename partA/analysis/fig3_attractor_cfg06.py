#!/usr/bin/env python3
"""
Regenerate fig3_attractor_spectrum on cfg06, the SELECTED model.

The previous version was computed on cfg01 at 1024 fixed points, which
contradicts sections 3.3 (cfg01 rejected) and 3.4 (cfg06, 512 fixed points,
36.7 marginal) of the paper.

A  exact Jacobian eigenvalues at the converged fixed points, vs the unit circle
B  sorted |lambda| within each fixed point, mean +/- s.d.
C  the fixed-point manifold is low-dimensional, like the data
"""
import os as _os
# Paths come from environment variables so the repo runs anywhere.
# See paths.env. Defaults below are the machine this was written on.
LFADS_RUNS = _os.environ.get("LFADS_RUNS", "/home/tiwar133/lfads_runs/sweep_dyn")
SESSION_DATA = _os.environ.get("SESSION_DATA", "/mnt/c/Users/tiwar133/gpfa_v0203/MultiSessionData")
RESULTS_DIR = _os.environ.get("RESULTS_DIR", "/home/tiwar133/SURF_talk_figures")
ROBUST_DIR = _os.environ.get("ROBUST_DIR", "/home/tiwar133/dynamics_final")
import numpy as np, h5py, glob, json, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

torch.manual_seed(0); np.random.seed(0)
DST = RESULTS_DIR
ROOT = f"{LFADS_RUNS}/61199SOM/cfg06"
DT, MARG, CLIP, NSEED, ITERS = 0.020, 0.01, 5.0, 512, 12000
dev = "cuda" if torch.cuda.is_available() else "cpu"

BLUE, GREEN, ORANGE = "#2a6fd6", "#12a37a", "#e8622a"
INK, MUTED, GRID, SURFACE = "#111111", "#5a5955", "#dfe1e5", "#fbfbfa"


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


ck = [p for p in glob.glob(f"{ROOT}/checkpoints/*.ckpt") if "last" not in p][0]
cell = GenCell(torch.load(ck, map_location="cpu")["state_dict"])
G = np.concatenate([np.concatenate([h5py.File(p)["train_gen_states"][:],
                                    h5py.File(p)["valid_gen_states"][:]])
                    for p in sorted(glob.glob(f"{ROOT}/lfads_output_*.h5"))])
flat = G.reshape(-1, cell.H).astype(np.float32)

idx = np.random.default_rng(0).choice(len(flat), NSEED, replace=False)
g = torch.nn.Parameter(torch.tensor(flat[idx], device=dev)
                       + 0.05 * torch.randn(NSEED, cell.H, device=dev))
u = torch.zeros(NSEED, cell.w_ih.shape[1], device=dev)
opt = torch.optim.Adam([g], lr=2e-2)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ITERS, eta_min=1e-5)
for _ in range(ITERS):
    opt.zero_grad()
    ((0.5 * ((cell(u, g) - g) ** 2).sum(-1)).sum()).backward()
    opt.step(); sch.step()
with torch.no_grad():
    keep = (torch.norm(cell(u, g) - g, dim=-1)
            / torch.norm(g, dim=-1).median()) < 1e-3
FP = g.detach()[keep]
nfp = len(FP)
print(f"{nfp} fixed points from {NSEED} seeds")

sel = np.random.default_rng(1).choice(nfp, min(60, nfp), replace=False)
eig, sorted_mag = [], []
for i in sel:
    J = torch.autograd.functional.jacobian(
        lambda h: cell(u[:1], h).squeeze(0), FP[i:i + 1], vectorize=True
    ).squeeze().cpu().numpy()
    e = np.linalg.eigvals(J)
    eig.append(e); sorted_mag.append(np.sort(np.abs(e))[::-1])
E = np.concatenate(eig); mag = np.abs(E)
S = np.stack(sorted_mag)

n_marg = (np.abs(mag - 1) < MARG).sum() / len(sel)
n_stab = (mag < 1 - MARG).sum() / len(sel)
n_unst = (mag > 1 + MARG).sum() / len(sel)
tau = -DT / np.log(np.clip(mag[mag < 1 - MARG], 1e-9, None))
print(f"per fixed point: {n_marg:.1f} marginal, {n_stab:.1f} stable, {n_unst:.2f} unstable")
print(f"stable tau median {np.median(tau)*1000:.0f} ms")


def cumvar_pr(X):
    Xc = X - X.mean(0)
    ev = np.linalg.svd(Xc, compute_uv=False) ** 2
    ev = ev / ev.sum()
    return np.cumsum(ev), (ev.sum() ** 2) / (ev ** 2).sum()


cv_fp, pr_fp = cumvar_pr(FP.cpu().numpy())
cv_dat, pr_dat = cumvar_pr(flat[np.random.default_rng(2).choice(len(flat), 20000,
                                                                replace=False)])
print(f"participation ratio: fixed points {pr_fp:.1f}, data {pr_dat:.1f}")

# ------------------------------------------------------------------ figure --
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 12})
fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(16.3, 5.6),
                                   gridspec_kw=dict(wspace=0.28))
fig.subplots_adjust(top=0.78)

ax.add_patch(Circle((0, 0), 1, fill=False, color=INK, lw=1.6, zorder=3))
ax.scatter(E.real, E.imag, s=7, color=BLUE, alpha=0.28, lw=0, zorder=4)
ax.axhline(0, color=GRID, lw=0.9); ax.axvline(0, color=GRID, lw=0.9)
ax.set_xlim(-1.28, 1.28); ax.set_ylim(-1.28, 1.28); ax.set_aspect("equal")
ax.set_xlabel(r"Re($\lambda$)", fontsize=13); ax.set_ylabel(r"Im($\lambda$)", fontsize=13)
ax.set_title(f"Exact Jacobian spectrum\nat the fixed points ({len(sel)} shown)",
             fontsize=13, weight="bold", pad=10)

m, sdv = S.mean(0), S.std(0)
ax2.fill_between(np.arange(S.shape[1]), m - sdv, m + sdv, color=BLUE, alpha=0.22, lw=0)
ax2.plot(m, color=BLUE, lw=2.6, zorder=4)
ax2.axhline(1, color=ORANGE, lw=1.6, ls="--", zorder=3)
ax2.annotate(r"$|\lambda| = 1$", (S.shape[1], 1.0), xytext=(-6, 6),
             textcoords="offset points", ha="right", fontsize=12, color=ORANGE)
ax2.axvline(n_marg, color=GREEN, lw=1.8, ls=":", zorder=3)
ax2.annotate(f"{n_marg:.0f} marginal modes", (n_marg, 0.52), xytext=(9, 0),
             textcoords="offset points", rotation=90, fontsize=12.5,
             weight="bold", color=GREEN, va="center")
ax2.set_xlim(0, S.shape[1]); ax2.set_ylim(0, 1.08)
ax2.set_xlabel("mode (sorted within each fixed point)", fontsize=13)
ax2.set_ylabel(r"$|\lambda|$", fontsize=13)
ax2.yaxis.grid(True, color=GRID, lw=0.8); ax2.set_axisbelow(True)
ax2.set_title(f"{n_marg:.0f} of {cell.H} modes are marginal\n"
              f"{n_unst:.2f} unstable per fixed point",
              fontsize=13, weight="bold", pad=10)

ax3.plot(np.arange(1, len(cv_dat) + 1), cv_dat, color=MUTED, lw=2.4, ls="--",
         label=f"real data (PR = {pr_dat:.0f})")
ax3.plot(np.arange(1, len(cv_fp) + 1), cv_fp, color=GREEN, lw=2.8,
         label=f"fixed-point manifold (PR = {pr_fp:.0f})")
ax3.axhline(0.9, color=GRID, lw=1.0, ls=":")
ax3.set_xlim(1, cell.H); ax3.set_ylim(0, 1.03)
ax3.set_xlabel("number of dimensions", fontsize=13)
ax3.set_ylabel("cumulative variance explained", fontsize=13)
ax3.yaxis.grid(True, color=GRID, lw=0.8); ax3.set_axisbelow(True)
ax3.legend(frameon=False, fontsize=11.5, loc="lower right")
ax3.set_title("The attractor manifold is\nlow-dimensional, like the data",
              fontsize=13, weight="bold", pad=10)

fig.suptitle("The trained generator implements a continuous attractor "
             "— a manifold of marginally stable states",
             fontsize=15.5, weight="bold", y=1.045)
fig.text(0.5, -0.035,
         f"61199SOM, selected model cfg06. Exact autograd Jacobians of the trained "
         f"128-D ClippedGRU at {nfp} converged fixed points (u=0); {len(sel)} sampled "
         f"for the spectrum. 'Marginal' = ||λ|−1| < {MARG}. PR = participation ratio.",
         ha="center", fontsize=10, color=MUTED)

fig.savefig(f"{DST}/fig3_attractor_spectrum.png", dpi=200,
            bbox_inches="tight", facecolor=SURFACE)
json.dump(dict(model="cfg06", n_fixed_points=int(nfp), n_sampled=int(len(sel)),
               marginal=float(n_marg), stable=float(n_stab), unstable=float(n_unst),
               tau_median_ms=float(np.median(tau) * 1000),
               pr_fixed_points=float(pr_fp), pr_data=float(pr_dat)),
          open(f"{DST}/fig3_attractor_stats.json", "w"), indent=2)
print("-> fig3_attractor_spectrum.png (cfg06)")
