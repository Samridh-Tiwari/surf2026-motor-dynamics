#!/usr/bin/env python3
"""
Slide A10: what would it cost to steer?

The groove says some directions hold a displacement and others fight it. That is
directly a statement about the cost of intervening. Quantify it two ways:

A  empirical -- displace the state of the FULL nonlinear network by a fixed
   amount along a groove direction vs an off-groove direction, release, and
   watch how much of the displacement survives.
B  analytic -- sustained input needed to HOLD a unit displacement, ||(I-A)v||,
   for every direction, sorted by how fast that direction decays.

A displacement along the groove is free to maintain. Off it, you pay forever.
"""
import os as _os
# Paths come from environment variables so the repo runs anywhere.
# See paths.env. Defaults below are the machine this was written on.
LFADS_RUNS = _os.environ.get("LFADS_RUNS", "/path/to/lfads_runs/sweep_dyn")
SESSION_DATA = _os.environ.get("SESSION_DATA", "/path/to/MultiSessionData")
RESULTS_DIR = _os.environ.get("RESULTS_DIR", "./figures")
ROBUST_DIR = _os.environ.get("ROBUST_DIR", "./partA/results")
import numpy as np, h5py, glob, json, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(0); np.random.seed(0)
DST = RESULTS_DIR
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdbd7", "#ffffff"
DT, MARG, CLIP, AMP = 0.020, 0.01, 5.0, 1.0
dev = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = f"{LFADS_RUNS}/61199SOM/cfg06"


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

NSEED, ITERS = 256, 12000
idx = np.random.default_rng(0).choice(len(flat), NSEED, replace=False)
g = torch.nn.Parameter(torch.tensor(flat[idx], device=dev)
                       + 0.05 * torch.randn(NSEED, cell.H, device=dev))
u0 = torch.zeros(NSEED, cell.w_ih.shape[1], device=dev)
opt = torch.optim.Adam([g], lr=2e-2)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ITERS, eta_min=1e-5)
for _ in range(ITERS):
    opt.zero_grad()
    ((0.5 * ((cell(u0, g) - g) ** 2).sum(-1)).sum()).backward()
    opt.step(); sch.step()
with torch.no_grad():
    keep = (torch.norm(cell(u0, g) - g, dim=-1) / torch.norm(g, dim=-1).median()) < 1e-3
FP = g.detach()[keep]
print(f"{len(FP)} fixed points")

T = 60
sel = np.random.default_rng(1).choice(len(FP), min(30, len(FP)), replace=False)
keep_groove, keep_off, hold_cost, lam_all = [], [], [], []
for i in sel:
    x0 = FP[i:i + 1]
    J = torch.autograd.functional.jacobian(
        lambda h: cell(u0[:1], h).squeeze(0), x0, vectorize=True
    ).squeeze().cpu().numpy()
    w, V = np.linalg.eig(J)
    mag = np.abs(w)
    # sustained input to hold a unit displacement along each eigendirection
    for j in range(len(w)):
        v = np.real(V[:, j]); nv = np.linalg.norm(v)
        if nv < 1e-8:
            continue
        v = v / nv
        hold_cost.append(float(np.linalg.norm((np.eye(len(v)) - J) @ v)))
        lam_all.append(float(mag[j]))
    gm = np.where(np.abs(mag - 1) < MARG)[0]
    st = np.where(mag < 1 - MARG)[0]
    if len(gm) == 0 or len(st) == 0:
        continue

    def release(dirs, n=4):
        out = []
        for j in dirs[:n]:
            v = np.real(V[:, j])
            nv = np.linalg.norm(v)
            if nv < 1e-8:
                continue
            d = torch.tensor(AMP * v / nv, dtype=torch.float32, device=dev)[None]
            h, hr = x0 + d, x0.clone()
            tr = []
            with torch.no_grad():
                for _ in range(T):
                    h = cell(u0[:1], h); hr = cell(u0[:1], hr)
                    tr.append(float(torch.norm(h - hr)))
            out.append(tr)
        return out

    keep_groove += release(gm)
    keep_off += release(np.random.default_rng(i).permutation(st))   # typical off-groove

Kg = np.array(keep_groove) / AMP
Ko = np.array(keep_off) / AMP
t = (np.arange(T) + 1) * DT
lam_all, hold_cost = np.array(lam_all), np.array(hold_cost)
is_g = np.abs(lam_all - 1) < MARG
ratio = np.median(hold_cost[~is_g]) / max(np.median(hold_cost[is_g]), 1e-9)
print(f"groove:     {Kg.mean(0)[-1]*100:.0f}% of the push still there after {t[-1]:.1f}s")
print(f"off-groove: {Ko.mean(0)[-1]*100:.0f}%")
print(f"hold cost  groove {np.median(hold_cost[is_g]):.4f} vs "
      f"off-groove {np.median(hold_cost[~is_g]):.4f}  -> {ratio:.0f}x")


# ---- does the success-vs-failure direction lie in the groove? --------------
CD = np.concatenate([np.concatenate([h5py.File(q)["train_condition"][:].ravel(),
                                     h5py.File(q)["valid_condition"][:].ravel()])
                     for q in sorted(glob.glob(f"{ROOT}/lfads_output_*.h5"))]).astype(int)
post = slice(80, 120)
dvec = G[CD == 1][:, post].mean((0, 1)) - G[CD == 0][:, post].mean((0, 1))
dvec = dvec / np.linalg.norm(dvec)
rng3 = np.random.default_rng(2)
obs, nulls = [], []
for i in sel:
    J = torch.autograd.functional.jacobian(
        lambda h: cell(u0[:1], h).squeeze(0), FP[i:i + 1], vectorize=True
    ).squeeze().cpu().numpy()
    w, V = np.linalg.eig(J)
    gm = np.where(np.abs(np.abs(w) - 1) < MARG)[0]
    if len(gm) == 0:
        continue
    Q, _ = np.linalg.qr(np.real(V[:, gm]))
    obs.append(float(np.sum((Q.T @ dvec) ** 2)))
    r = rng3.normal(size=(len(dvec), 400)); r /= np.linalg.norm(r, axis=0)
    nulls.append(np.sum((Q.T @ r) ** 2, axis=0))
obs_align = float(np.mean(obs))
null_align = np.mean(np.stack(nulls), axis=0)
n_ge = int((null_align >= obs_align).sum())
p_align = (n_ge + 1) / (len(null_align) + 1)      # never report p = 0
p_txt = f"p = {p_align:.3f}" if n_ge else f"p < {1/len(null_align):.4f}"
print(f"behavioural direction in groove: {obs_align:.3f} vs chance "
      f"{null_align.mean():.3f} ({obs_align/null_align.mean():.2f}x), {p_txt}")

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 13})
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14.6, 6.0),
                              gridspec_kw=dict(wspace=0.26))

for arr, col, lab in [(Kg, AQUA, "along the groove"), (Ko, BLUE, "off the groove")]:
    m, se = arr.mean(0), arr.std(0) / np.sqrt(len(arr))
    ax.fill_between(t, m - 1.96 * se, m + 1.96 * se, color=col, alpha=0.22, lw=0)
    ax.plot(t, m, color=col, lw=3.2, zorder=4, label=lab)
ax.axhline(1, color=MUTED, lw=1, ls=":")
ax.set_xlim(0, t[-1]); ax.set_ylim(0, 1.15)
ax.set_xlabel("time after the push (s)", fontsize=13.5, labelpad=7)
ax.set_ylabel("fraction of the push still there", fontsize=13.5, labelpad=7)
ax.yaxis.grid(True, color=GRID, lw=0.9); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=13, loc="center right")
ax.annotate(f"{Kg.mean(0)[-1]*100:.0f}%", (t[-1], Kg.mean(0)[-1]), xytext=(-6, 10),
            textcoords="offset points", ha="right", fontsize=17, weight="bold", color=AQUA)
ax.annotate(f"{Ko.mean(0)[-1]*100:.0f}%", (t[-1], Ko.mean(0)[-1]), xytext=(-6, 10),
            textcoords="offset points", ha="right", fontsize=17, weight="bold", color=BLUE)
ax.set_title("A   Push the network, then let go\nfull nonlinear model, 30 states",
             fontsize=13.5, weight="bold", loc="left", pad=10)

ax2.hist(null_align, bins=40, color=MUTED, alpha=0.45, lw=0,
         label="a random direction")
ax2.axvline(obs_align, color=ORANGE, lw=3.4, zorder=5)
ax2.annotate(f"success vs failure\n{obs_align:.2f}", (obs_align, ax2.get_ylim()[1] * 0.88),
             xytext=(10, 0), textcoords="offset points", fontsize=14,
             weight="bold", color=ORANGE, va="top")
ax2.set_xlabel("fraction of the direction lying in the groove", fontsize=13.5, labelpad=7)
ax2.set_ylabel("count", fontsize=13.5, labelpad=7)
ax2.yaxis.grid(True, color=GRID, lw=0.9); ax2.set_axisbelow(True)
ax2.legend(frameon=False, fontsize=12.5, loc="upper left")
ax2.set_title(f"B   Is the behavioural direction cheap?\n"
              f"{obs_align/null_align.mean():.2f}x chance   ·   {p_txt}",
              fontsize=13.5, weight="bold", loc="left", pad=10)

fig.tight_layout()
fig.savefig(f"{DST}/SlideA10_cost_of_steering.png", dpi=170,
            bbox_inches="tight", facecolor=SURFACE)
json.dump(dict(groove_retained=float(Kg.mean(0)[-1]), off_retained=float(Ko.mean(0)[-1]),
               hold_ratio=float(ratio), align_obs=obs_align,
               align_chance=float(null_align.mean()), align_p=p_align,
               n_fp=int(len(FP))),
          open(f"{DST}/steering_stats.json", "w"), indent=2)
print("-> SlideA10_cost_of_steering.png")
