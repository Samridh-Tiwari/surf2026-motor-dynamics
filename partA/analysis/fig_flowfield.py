#!/usr/bin/env python3
"""
Neural state space with the learned flow field, both mice.

At each grid location we take the AVERAGE REAL 128-D generator state of the
nearby data, push it one step through the trained rule, and project the change
back to 2D. So every arrow is a real one-step prediction of the fitted model at
a place the data actually visits -- not an extrapolation into empty space.
Grid cells with too little nearby data are left blank rather than invented.

Outputs
  flowfield.png          static, both mice side by side
  clip_flow/f_*.png      frames: mouse 1, state moving along the flow
"""
import os, glob, json
import numpy as np, h5py, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

torch.manual_seed(0); np.random.seed(0)
DST = "/home/tiwar133/SURF_talk_figures"
BIN, PRECUE, CLIP = 0.020, 1.5, 5.0
dev = "cuda" if torch.cuda.is_available() else "cpu"
MICE = [("61199SOM", "mouse 1"), ("61198SOM", "mouse 2")]

HIT, MISS, ARROW, INK, MUTED, GRID, SURFACE = (
    "#1668d6", "#d12b2b", "#8e9199", "#111111", "#5f5e5a", "#e3e5e9", "#ffffff")


def load(mouse):
    root = f"/home/tiwar133/lfads_runs/sweep_dyn/{mouse}/cfg06"
    ck = [p for p in glob.glob(f"{root}/checkpoints/*.ckpt") if "last" not in p][0]
    sd = torch.load(ck, map_location="cpu")["state_dict"]
    pf = "decoder.rnn.cell.gen_cell."
    W_ih, W_hh = sd[pf + "weight_ih"].to(dev), sd[pf + "weight_hh"].to(dev)
    b_ih, b_hh = sd[pf + "bias_ih"].to(dev), sd[pf + "bias_hh"].to(dev)
    H = W_hh.shape[1]

    def step(h):
        u = torch.zeros(len(h), W_ih.shape[1], device=dev)
        x = u @ W_ih.T + b_ih
        xz, xr, xn = torch.chunk(x, 3, -1)
        wzr, wn = torch.split(W_hh, [2 * H, H])
        bzr, bn = torch.split(b_hh, [2 * H, H])
        hz, hr = torch.chunk(h @ wzr.T + bzr, 2, -1)
        z = torch.sigmoid(xz + hz); r = torch.sigmoid(xr + hr)
        return torch.clamp(z * h + (1 - z) * torch.tanh(xn + (r * h) @ wn.T + bn),
                           -CLIP, CLIP)

    GS, C = [], []
    for f in sorted(glob.glob(f"{root}/lfads_output_*.h5")):
        with h5py.File(f) as d:
            GS.append(np.concatenate([d["train_gen_states"][:], d["valid_gen_states"][:]]))
            C.append(np.concatenate([d["train_condition"][:].ravel(),
                                     d["valid_condition"][:].ravel()]))
    return step, np.concatenate(GS), np.concatenate(C).astype(int), H


def flow(mouse, ngrid=26, kmin=2):
    step, GS, C, H = load(mouse)
    T = GS.shape[1]
    pca = PCA(n_components=2).fit(GS.reshape(-1, H))
    var2 = pca.explained_variance_ratio_.sum()
    flat = GS.reshape(-1, H)
    P2 = pca.transform(flat)

    # a representative single trial of each type: closest to that type's mean.
    # single trials have real excursion; averaging across trials cancels it.
    P3 = P2.reshape(len(GS), T, 2)
    hit_i = np.where(C == 1)[0]; miss_i = np.where(C == 0)[0]
    mh = P3[hit_i].mean(0); mm = P3[miss_i].mean(0)
    tr_hit = P3[hit_i[np.argmin(((P3[hit_i] - mh) ** 2).sum((1, 2)))]]
    tr_miss = P3[miss_i[np.argmin(((P3[miss_i] - mm) ** 2).sum((1, 2)))]]

    # frame on the trajectories, not the whole cloud
    both = np.vstack([tr_hit, tr_miss])
    lo, hi = both.min(0), both.max(0)
    pad = (hi - lo) * 0.28
    lo, hi = lo - pad, hi + pad
    gx = np.linspace(lo[0], hi[0], ngrid)
    gy = np.linspace(lo[1], hi[1], ngrid)
    dx, dy = gx[1] - gx[0], gy[1] - gy[0]

    X, Y, U, V, S, N = [], [], [], [], [], []
    for cx in gx:
        for cy in gy:
            # every real sample whose 2D projection falls in this cell
            m = ((np.abs(P2[:, 0] - cx) < dx * 1.4) &
                 (np.abs(P2[:, 1] - cy) < dy * 1.4))
            if m.sum() < kmin:
                continue                      # truly no data: draw nothing
            N.append(int(m.sum()))
            rep = flat[m].mean(0)             # an on-manifold representative
            with torch.no_grad():
                nxt = step(torch.tensor(rep[None], dtype=torch.float32,
                                        device=dev)).cpu().numpy()[0]
            d2 = pca.transform(nxt[None])[0] - pca.transform(rep[None])[0]
            X.append(cx); Y.append(cy); U.append(d2[0]); V.append(d2[1])
            S.append(np.linalg.norm(d2))
    X, Y, U, V, S, N = map(np.array, (X, Y, U, V, S, N))
    L = np.sqrt(U ** 2 + V ** 2)
    keep = L > 1e-9
    X, Y, U, V, S, L, N = (X[keep], Y[keep], U[keep], V[keep], S[keep],
                           L[keep], N[keep])
    arrow_len = 0.62 * min(dx, dy)
    U, V = U / L * arrow_len, V / L * arrow_len      # direction only
    print(f"{mouse}: {len(X)} arrows, 2 PCs hold {var2:.1%}, "
          f"speed {S.min():.3f}-{S.max():.3f}")
    return dict(X=X, Y=Y, U=U, V=V, S=S, N=N, lo=lo, hi=hi, var2=var2,
                hit=tr_hit, miss=tr_miss, T=T, pca=pca, GS=GS, C=C)


F = {m: flow(m) for m, _ in MICE}

# --------------------------------------------------------------- static fig --
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 17,
    "axes.spines.top": False, "axes.spines.right": False})


def draw(ax, d, title, label_pts=True, cbar=False):
    import matplotlib as mpl
    sp = np.log10(d["S"])
    norm = mpl.colors.Normalize(sp.min(), sp.max())
    cmap = mpl.cm.viridis_r
    rgba = cmap(norm(sp))
    sup = np.log10(d["N"] + 1)                    # data support behind each arrow
    rgba[:, 3] = 0.16 + 0.79 * (sup - sup.min()) / max(sup.ptp(), 1e-9)
    q = ax.quiver(d["X"], d["Y"], d["U"], d["V"], color=rgba,
                  angles="xy", scale_units="xy", scale=1.0,
                  width=0.0058, headwidth=3.6, headlength=4.0,
                  headaxislength=3.4, zorder=2)
    if cbar:
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, fraction=0.043, pad=0.02)
        cb.set_label("how fast the state is changing\n(pale = slow, dark = fast)", fontsize=15)
        cb.ax.tick_params(labelsize=13)
    cue = int(PRECUE / BIN)
    for tr, col, lab, lw in [(d["miss"], MISS, "one miss trial", 3.2),
                             (d["hit"], HIT, "one hit trial", 4.4)]:
        ax.plot(tr[:, 0], tr[:, 1], color=col, lw=lw, zorder=5,
                solid_capstyle="round", label=lab)
    h = d["hit"]
    if label_pts:
        for idx, name, dxy in [(0, "trial start", (14, 12)),
                               (cue, "cue", (14, -20)),
                               (d["T"] - 1, "trial end", (-16, 14))]:
            ax.plot(h[idx, 0], h[idx, 1], "o", ms=13, color=INK, zorder=7)
            ax.annotate(name, (h[idx, 0], h[idx, 1]), xytext=dxy,
                        textcoords="offset points", fontsize=17, weight="bold",
                        color=INK, zorder=8)
    ax.set_xlim(d["lo"][0], d["hi"][0]); ax.set_ylim(d["lo"][1], d["hi"][1])
    ax.set_xlabel("latent dimension 1", fontsize=19, labelpad=8)
    ax.set_ylabel("latent dimension 2", fontsize=19, labelpad=8)
    ax.set_title(title, fontsize=21, weight="bold", pad=14)


fig, axes = plt.subplots(1, 2, figsize=(20.5, 8.6), gridspec_kw=dict(wspace=0.24))
for ax, (mouse, name) in zip(axes, MICE):
    draw(ax, F[mouse], name, cbar=(mouse == MICE[-1][0]))
axes[0].legend(frameon=True, framealpha=0.92, edgecolor="none",
               fontsize=17, loc="lower left")
fig.suptitle("The rule the population follows, measured in both animals",
             fontsize=26, weight="bold", y=1.005)
fig.text(0.5, -0.055,
         "Arrows: one step of the fitted rule from the average real state in that "
         "region, projected to the top two\nlatent dimensions "
         f"({F['61199SOM']['var2']:.0%} and {F['61198SOM']['var2']:.0%} of state "
         "variance). Arrows fade where fewer recordings\npass through, so pale "
         "regions are the least certain.",
         ha="center", fontsize=14, color=MUTED, linespacing=1.6)
fig.savefig(f"{DST}/flowfield.png", dpi=170, bbox_inches="tight", facecolor=SURFACE)
print("-> flowfield.png")
plt.close(fig)

# ------------------------------------------------------------------- clip ----
d = F["61199SOM"]
outdir = f"{DST}/clip_flow"
os.makedirs(outdir, exist_ok=True)
T = d["T"]; cue = int(PRECUE / BIN); HOLD = 30
t = np.arange(T) * BIN - PRECUE
h = d["hit"]

fig = plt.figure(figsize=(10.8, 10.8), facecolor=SURFACE)
ax = fig.add_subplot(111)
draw(ax, d, "", label_pts=False)
ax.set_title("")
sub = fig.text(0.5, 0.955, "", ha="center", fontsize=25, weight="bold", color=INK)
clock = fig.text(0.5, 0.035, "", ha="center", fontsize=22, weight="bold",
                 color=INK, family="monospace")
trail, = ax.plot([], [], color=HIT, lw=5.0, zorder=6, solid_capstyle="round")
glow, = ax.plot([], [], "o", ms=34, color=HIT, alpha=0.20, zorder=7)
head, = ax.plot([], [], "o", ms=17, color=HIT, zorder=8, mec="white", mew=3)
for line in ax.get_lines()[:2]:
    line.set_alpha(0.22)                     # fade the two static mean paths
ax.legend().set_visible(False)
fig.subplots_adjust(left=0.11, right=0.98, top=0.91, bottom=0.10)

for k in range(T + HOLD):
    i = min(k, T - 1)
    trail.set_data(h[:i + 1, 0], h[:i + 1, 1])
    head.set_data([h[i, 0]], [h[i, 1]])
    glow.set_data([h[i, 0]], [h[i, 1]])
    sub.set_text("the cue arrives" if cue <= i < cue + 26 else
                 ("preparing" if i < cue else "movement"))
    clock.set_text(f"{t[i]:+.2f} s")
    fig.savefig(f"{outdir}/f_{k:04d}.png", dpi=100, facecolor=SURFACE)
    if k % 45 == 0:
        print(f"   frame {k}/{T + HOLD}")
plt.close(fig)
json.dump({m: dict(var2=float(F[m]["var2"]), arrows=int(len(F[m]["X"])))
           for m, _ in MICE}, open(f"{DST}/flowfield_stats.json", "w"), indent=2)
print("done")
