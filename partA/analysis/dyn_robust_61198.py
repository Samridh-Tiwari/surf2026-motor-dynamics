#!/usr/bin/env python3
"""Marginal-mode count across mouse-2 configs, same settings as mouse 1."""
import numpy as np, h5py, glob, json, torch
torch.manual_seed(0); np.random.seed(0)
MARG, CLIP = 0.01, 5.0
OUT = ROBUST_DIR
ROOT = f"{LFADS_RUNS}/61198SOM"
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


def analyse(cfg, nseed=256, iters=8000):
    ck = [p for p in glob.glob(f"{ROOT}/{cfg}/checkpoints/*.ckpt") if "last" not in p]
    h5 = sorted(glob.glob(f"{ROOT}/{cfg}/lfads_output_*.h5"))
    if not ck or not h5:
        return None
    cell = GenCell(torch.load(ck[0], map_location="cpu")["state_dict"])
    G = np.concatenate([np.concatenate([h5py.File(p)["train_gen_states"][:],
                                        h5py.File(p)["valid_gen_states"][:]]) for p in h5])
    flat = G.reshape(-1, cell.H).astype(np.float32)
    idx = np.random.default_rng(0).choice(len(flat), nseed, replace=False)
    g = torch.nn.Parameter(torch.tensor(flat[idx], device=dev)
                           + 0.05 * torch.randn(nseed, cell.H, device=dev))
    u = torch.zeros(nseed, cell.w_ih.shape[1], device=dev)
    opt = torch.optim.Adam([g], lr=2e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=1e-5)
    for _ in range(iters):
        opt.zero_grad()
        ((0.5 * ((cell(u, g) - g) ** 2).sum(-1)).sum()).backward()
        opt.step(); sch.step()
    with torch.no_grad():
        keep = (torch.norm(cell(u, g) - g, dim=-1)
                / torch.norm(g, dim=-1).median()) < 1e-3
    FP = g.detach()[keep]
    if len(FP) == 0:
        return dict(config=cfg, n_fp=0)
    sel = np.random.default_rng(1).choice(len(FP), min(30, len(FP)), replace=False)
    nm, ns, nu = [], [], []
    for i in sel:
        J = torch.autograd.functional.jacobian(
            lambda h: cell(u[:1], h).squeeze(0), FP[i:i + 1], vectorize=True
        ).squeeze().cpu().numpy()
        m = np.abs(np.linalg.eigvals(J))
        nm.append(int((np.abs(m - 1) < MARG).sum()))
        ns.append(int((m < 1 - MARG).sum()))
        nu.append(int((m > 1 + MARG).sum()))
    return dict(config=cfg, n_fp=int(len(FP)),
                marginal=[float(np.mean(nm)), float(np.std(nm))],
                stable=[float(np.mean(ns)), float(np.std(ns))],
                unstable=[float(np.mean(nu)), float(np.std(nu))])


res = []
print(f"{'config':<8}{'FPs':>6}{'marginal':>15}{'unstable':>11}")
for cfg in ["cfg01", "cfg00", "cfg06", "cfg02", "cfg09"]:
    r = analyse(cfg)
    if r and r.get("n_fp"):
        res.append(r)
        print(f"{cfg:<8}{r['n_fp']:>6}{r['marginal'][0]:>10.1f}+/-{r['marginal'][1]:<4.1f}"
              f"{r['unstable'][0]:>11.2f}")
mg = [r["marginal"][0] for r in res]
print(f"\nmouse 2 marginal modes: {np.mean(mg):.1f} +/- {np.std(mg):.1f} "
      f"(range {min(mg):.0f}-{max(mg):.0f}) across {len(res)} models")
json.dump(res, open(f"{OUT}/robustness_61198_all.json", "w"), indent=2)
import os as _os
# Paths are read from environment variables so the repo runs anywhere.
# Set them once (see README), or edit the defaults below.
LFADS_RUNS = _os.environ.get("LFADS_RUNS", "/home/tiwar133/lfads_runs/sweep_dyn")
SESSION_DATA = _os.environ.get("SESSION_DATA", "/mnt/c/Users/tiwar133/gpfa_v0203/MultiSessionData")
RESULTS_DIR = _os.environ.get("RESULTS_DIR", "/home/tiwar133/SURF_talk_figures")
ROBUST_DIR = _os.environ.get("ROBUST_DIR", "/home/tiwar133/dynamics_final")
