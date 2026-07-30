#!/usr/bin/env python3
r"""
reproduce_lfads_paper.py  —  Pandarinath 2018 (Nature Methods) key figures
                             reproduced on YOUR data (LFADS vs GPFA).
===========================================================================
Reproduces, per session, the figures that are meaningful for your cued
lever-press task:

  Fig 2b : example-neuron rates (LFADS vs GPFA vs Gaussian smoothing),
           condition-averaged and single-trial.
  Fig 2c : t-SNE of LFADS initial conditions (g0), colored by Hit/Miss.
  Fig 2d/e: OLE (ridge) decoding of lever position, LFADS vs GPFA vs
           smoothing vs binning, plus R^2 vs neuron-subsample size.
  Fig 2f : held-out-neuron co-smoothing (bits/spike), LFADS vs GPFA — the
           paper's central quantitative LFADS>GPFA claim.
  Fig 3  : jPCA rotational dynamics, condition-averaged + single-trial.

NOT reproduced (your task/data lack the needed structure):
  Fig 5 (inferred-input perturbation) — no perturbation task.
  Fig 6 (rate-LFP phase locking)      — no LFP recordings.

Inputs per session (a manifest entry):
  lfads_h5   : LFADS output .h5 (train/valid_factors, *_output_params,
               *_recon_data, *_ic_mean, *_condition)
  gpfa_mat   : GPFA_results.mat (trainDat/testDat/seqTrain/seqTest/estParams/
               IntanBehaviour)  — optional; GPFA panels skipped if absent
  spikes_mat : Spikes.mat for lever alignment (IntanBehaviour)

Usage:
  python reproduce_lfads_paper.py --manifest sessions.json --out_dir paper_figs
"""

import argparse, json, os, glob, warnings, itertools
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.special import gammaln
from sklearn.linear_model import Ridge, RidgeCV, PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
warnings.filterwarnings('ignore')

COL = {'LFADS': '#3B6EA5', 'GPFA': '#D9843B', 'Smooth': '#6BAF6B',
       'Bin': '#9AA0A6', 'Hit': '#2E9E68', 'Miss': '#D65C5C', 'ink': '#2b2b2b'}

def _style():
    plt.rcParams.update({
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'axes.edgecolor': '#cccccc', 'axes.grid': True, 'grid.color': '#ECECEC',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.titlesize': 11, 'axes.titleweight': 'semibold',
        'font.size': 9, 'legend.fontsize': 8, 'legend.frameon': False})

# ======================================================================
# LForm loaders: LFADS h5
# ======================================================================
def load_lfads(h5_path):
    """Return dict with pooled train+valid: factors (n,T,F), rates (n,T,N)
    from output_params, spikes (n,T,N) from recon_data, ic (n,ic_dim),
    cond ('Hit'/'Miss')."""
    with h5py.File(h5_path, 'r') as f:
        def cat(a, b): return np.concatenate([np.asarray(f[a], np.float64),
                                              np.asarray(f[b], np.float64)], 0)
        factors = cat('train_factors', 'valid_factors')
        rates = cat('train_output_params', 'valid_output_params')
        spikes = cat('train_recon_data', 'valid_recon_data')
        ic = cat('train_ic_mean', 'valid_ic_mean') if 'train_ic_mean' in f else None
        cnum = np.concatenate([np.asarray(f['train_condition']).ravel(),
                               np.asarray(f['valid_condition']).ravel()])
    if np.nanmin(rates) < 0:      # log-rate -> rate
        rates = np.exp(rates)
    cond = np.where(cnum == 1, 'Hit', 'Miss')
    return dict(factors=factors, rates=rates, spikes=spikes, ic=ic, cond=cond)

# ======================================================================
# GPFA loader (v7.3 .mat) — reuse validated helpers, pool train+test
# ======================================================================
def _ref(ds):
    return ds[()].flatten() if h5py.check_dtype(ref=ds.dtype) is not None else None
def _nstruct(g):
    if isinstance(g, h5py.Dataset): return 0
    for k in g.keys():
        it = g[k]
        if isinstance(it, h5py.Dataset):
            r = _ref(it); return len(r) if r is not None else it.size
    return 0
def _numfield(f, g, field, n):
    out = np.full(n, np.nan)
    if field not in g: return out
    ds = g[field]; r = _ref(ds)
    if r is not None:
        for i in range(min(n, len(r))): out[i] = np.asarray(f[r[i]][()]).ravel()[0]
    else:
        a = np.asarray(ds[()]).ravel(); m = min(n, len(a)); out[:m] = a[:m]
    return out
def _strfield(f, g, field, n):
    if field not in g: return [None]*n
    ds = g[field]; r = _ref(ds); out = []
    if r is not None:
        for i in range(n):
            if i >= len(r): out.append(None); continue
            raw = np.asarray(f[r[i]][()]).ravel()
            out.append(''.join(chr(c) for c in raw if c != 0).strip() if raw.dtype.kind in 'ui'
                       else str(raw[0]))
    else:
        out = [str(x) for x in np.asarray(ds[()]).ravel()[:n]]
    return out
def _rotation(f, nf):
    if 'estParams' not in f or 'C' not in f['estParams']: return None
    C = np.asarray(f['estParams']['C'][()], np.float64)
    if C.shape[1] != nf and C.shape[0] == nf: C = C.T
    if C.shape[1] != nf: return None
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    return np.diag(S) @ Vt
def _latents(f, sg, n, rot=None):
    if n == 0 or isinstance(sg, h5py.Dataset): return []
    lat = [None]*n; used = None
    for field in ('xorth', 'xsm', 'xpost', 'y'):
        if field not in sg: continue
        r = _ref(sg[field])
        if r is None: continue
        for i in range(min(n, len(r))):
            if lat[i] is not None: continue
            a = np.asarray(f[r[i]][()], np.float64)
            if a.ndim != 2: continue
            lat[i] = a.T if a.shape[0] > a.shape[1] else a
        if all(l is not None for l in lat): used = field; break
    if used and used != 'xorth' and rot is not None:
        lat = [rot @ l if l is not None else None for l in lat]
    return lat
def _condmap(f, cues):
    ib = f['IntanBehaviour']
    def fr(raw): return raw if raw.ndim == 1 else (raw[0] if raw.shape[0] < raw.shape[1] else raw[:,0])
    hit = fr(np.asarray(ib['cueHit'][()], np.float64))/1000.0 if 'cueHit' in ib else np.array([])
    miss = fr(np.asarray(ib['cueMiss'][()], np.float64))/1000.0 if 'cueMiss' in ib else np.array([])
    out = []
    for ct in cues:
        if hit.size and np.any(np.abs(hit-ct) < 0.005): out.append('Hit')
        elif miss.size and np.any(np.abs(miss-ct) < 0.005): out.append('Miss')
        else: out.append('Unknown')
    return out

def load_gpfa_cosmoothing(gpfa_mat):
    """Load GPFA factors, own spikes (y), and trialId for FAIR co-smoothing.
    Returns (factors (n,T,F), y_spikes (n,T,N), trialId (n,)) or None."""
    with h5py.File(gpfa_mat, 'r') as f:
        if 'seqTrain' not in f or 'xorth' not in f['seqTrain']:
            return None
        def stack(fld):
            refs = np.array(f['seqTrain'][fld]).ravel()
            return np.stack([np.array(f[r][()], np.float64) for r in refs])
        Xg = stack('xorth')      # (n,T,F) — xorth is already (T,F) per trial
        Yg = stack('y') if 'y' in f['seqTrain'] else None
        tid = np.array([np.array(f[r][()]).ravel()[0]
                        for r in np.array(f['seqTrain']['trialId']).ravel()]).astype(int)
    return Xg, Yg, tid

def fair_cosmoothing(gpfa_mat, lfads_h5, seed=0):
    """FAIR co-smoothing: align GPFA & LFADS to the SAME trials via trialId,
    score BOTH models' factors against a COMMON spike target (LFADS recon_data),
    held-out neurons x held-out trials. Returns {'LFADS':bps,'GPFA':bps,'n_trials':k}
    or None if GPFA unavailable."""
    g = load_gpfa_cosmoothing(gpfa_mat)
    if g is None:
        return None
    Xg, Yg, tid = g
    with h5py.File(lfads_h5, 'r') as f:
        Xl = np.concatenate([np.asarray(f['train_factors'], np.float64),
                             np.asarray(f['valid_factors'], np.float64)])
        Yl = np.concatenate([np.asarray(f['train_recon_data'], np.float64),
                             np.asarray(f['valid_recon_data'], np.float64)])
    # align LFADS to GPFA's trial subset via trialId (1-indexed)
    idx = tid - 1
    if idx.max() >= Xl.shape[0]:
        return None  # trialId out of range -> can't align safely
    Xl_sub, Yl_sub = Xl[idx], Yl[idx]
    # match T and neuron count
    T = min(Xg.shape[1], Xl_sub.shape[1]); N = min(Yg.shape[2] if Yg is not None else Yl_sub.shape[2], Yl_sub.shape[2])
    F = min(Xg.shape[2], Xl_sub.shape[2])
    Xg, Xl_sub, Yl_sub = Xg[:, :T, :F], Xl_sub[:, :T, :F], Yl_sub[:, :T, :N]
    # common target = LFADS recon spikes (full resolution); score each model's factors against it
    lfads_bps = cosmooth_core(Xl_sub, Yl_sub, seed)
    gpfa_bps  = cosmooth_core(Xg,     Yl_sub, seed)
    return {'LFADS': lfads_bps, 'GPFA': gpfa_bps, 'n_trials': int(len(idx))}

def cosmooth_core(X, Y, seed=0, n_folds=5):
    """held-out-neuron co-smoothing bits/spike. X (n,T,F) factors, Y (n,T,N) spikes."""
    n, T, F = X.shape; N = Y.shape[2]
    rng = np.random.default_rng(seed)
    neuron_folds = np.array_split(rng.permutation(N), n_folds)
    trial_kf = KFold(n_folds, shuffle=True, random_state=seed)
    scores = []; sc = StandardScaler()
    for held in neuron_folds:
        for tr, te in trial_kf.split(np.arange(n)):
            sc.fit(X[tr].reshape(-1, F))
            Xtr_s = sc.transform(X[tr].reshape(-1, F)); Xte_s = sc.transform(X[te].reshape(-1, F))
            pred = np.zeros((len(te)*T, len(held)))
            for j, nrn in enumerate(held):
                ytr = Y[tr][:, :, nrn].reshape(-1)
                try:
                    pred[:, j] = PoissonRegressor(alpha=1.0, max_iter=300).fit(Xtr_s, ytr).predict(Xte_s)
                except Exception:
                    pred[:, j] = ytr.mean()
            scores.append(bits_per_spike(Y[te][:, :, held].reshape(-1, len(held)), pred))
    return float(np.nanmean(scores))

def load_gpfa(gpfa_mat):
    """Return dict with pooled latents (n,T,F) and cond. GPFA gives latents
    only (no per-neuron rates unless a GLM is fit); rates handled separately."""
    with h5py.File(gpfa_mat, 'r') as f:
        def split(dat_key, seq_key, rot=None):
            dk = dat_key if dat_key in f else ('dat' if 'dat' in f else dat_key)
            sk = seq_key if seq_key in f else ('seqTrain' if seq_key == 'seqTest' and 'seqTrain' in f else seq_key)
            if dk not in f or sk not in f: return None
            g = f[dk]; n = _nstruct(g)
            if n == 0: return None
            cue = _numfield(f, g, 'cueTime', n)
            cond = _strfield(f, g, 'condition', n)
            if all(c is None for c in cond): cond = _condmap(f, cue)
            lat = _latents(f, f[sk], _nstruct(f[sk]), rot=rot)
            nu = min(n, len(lat)); Xs, Cs = [], []
            for i in range(nu):
                if lat[i] is None: continue
                Xs.append(lat[i].T); Cs.append(cond[i] or 'Unknown')
            if not Xs: return None
            mt = min(x.shape[0] for x in Xs)
            return np.stack([x[:mt] for x in Xs]), np.array(Cs)
        tr = split('trainDat', 'seqTrain')
        if tr is None: raise RuntimeError("GPFA: no train latents")
        rot = _rotation(f, tr[0].shape[2])
        te = split('testDat', 'seqTest', rot=rot)
    if te is None:
        X, C = tr
    else:
        mt = min(tr[0].shape[1], te[0].shape[1]); mf = min(tr[0].shape[2], te[0].shape[2])
        X = np.concatenate([tr[0][:,:mt,:mf], te[0][:,:mt,:mf]], 0)
        C = np.concatenate([tr[1], te[1]])
    keep = C != 'Unknown'
    return dict(factors=X[keep], cond=C[keep])

# ======================================================================
# Lever behavior alignment (shared)
# ======================================================================
def load_lever(spikes_mat, cond, T, t_before=1.5, t_after=1.5):
    """Align lever position to each trial. Trials assumed Hit-then-Miss
    order (LFADS/GPFA convention). Returns V (n,T) or None per bad trial."""
    with h5py.File(spikes_mat, 'r') as f:
        ib = f['IntanBehaviour']
        tv = np.asarray(ib['time'][()], np.float64).ravel()
        lv = np.asarray(ib['leverTrace'][()], np.float64).ravel()
        hr = np.asarray(ib['cueHit'][()], np.float64); mr = np.asarray(ib['cueMiss'][()], np.float64)
        hs = (hr[0] if hr.ndim == 2 else hr.ravel())/1000.0
        ms = (mr[0] if mr.ndim == 2 else mr.ravel())/1000.0
    n = len(cond); cues = np.concatenate([hs, ms])
    n_cues = len(cues)
    if n_cues < n:
        print(f"    [warn] {n} factor trials but only {n_cues} cues; "
              f"decoding first {n_cues} trials, dropping {n-n_cues}")
    V = np.full((n, T), np.nan); ok = np.ones(n, bool)
    for i in range(n):
        if i >= n_cues:            # no cue for this trial -> drop
            ok[i] = False; continue
        edges = np.linspace(cues[i]-t_before, cues[i]+t_after, T+1)
        idx = np.searchsorted(tv, edges)
        for b in range(T):
            s, e = idx[b], idx[b+1]
            if e > s: V[i, b] = lv[s:e].mean()
        if np.isnan(V[i]).any():
            good = ~np.isnan(V[i])
            if good.sum() >= 2: V[i, ~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), V[i, good])
            elif good.sum() == 1: V[i, :] = V[i, good][0]
            else: ok[i] = False
    return V, ok

def gaussian_smooth_spikes(spikes, sigma_bins=3):
    """Gaussian-smoothed firing rate estimate from spike counts (paper's
    'smoothing' baseline). spikes: (n,T,N) -> smoothed (n,T,N)."""
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(spikes, sigma_bins, axis=1)

def bits_per_spike(spikes, rate):
    """Poisson bits/spike of rate vs per-neuron-mean null. spikes,rate:(...,N)."""
    rate = np.clip(rate, 1e-9, None)
    ll = spikes*np.log(rate) - rate - gammaln(spikes+1)
    null = np.clip(spikes.mean(axis=tuple(range(spikes.ndim-1)), keepdims=True), 1e-9, None)
    lln = spikes*np.log(null) - null - gammaln(spikes+1)
    nsp = spikes.sum()
    return float((ll.sum()-lln.sum())/(nsp*np.log(2))) if nsp > 0 else np.nan

# ======================================================================
# Fig 2f — held-out-neuron co-smoothing (bits/spike), LFADS vs GPFA
# ======================================================================
def cosmoothing(spikes, factors_by_model, n_folds=5, seed=0):
    """Hold out neurons; fit Poisson GLM from each model's factors ->
    held-out neurons' spikes on train trials; score bits/spike on held-out
    trials (proper double cross-val over trials AND neurons).
    spikes: (n,T,N). factors_by_model: {name:(n,T,F)}. Returns {name:bps}."""
    n, T, N = spikes.shape
    rng = np.random.default_rng(seed)
    neuron_folds = np.array_split(rng.permutation(N), n_folds)
    trial_kf = KFold(n_folds, shuffle=True, random_state=seed)
    out = {name: [] for name in factors_by_model}
    for held_neurons in neuron_folds:
        keep_mask = np.ones(N, bool); keep_mask[held_neurons] = False
        for tr_idx, te_idx in trial_kf.split(np.arange(n)):
            for name, F in factors_by_model.items():
                Ftr = F[tr_idx].reshape(-1, F.shape[2])
                Fte = F[te_idx].reshape(-1, F.shape[2])
                sc = StandardScaler().fit(Ftr)
                Ftr_s, Fte_s = sc.transform(Ftr), sc.transform(Fte)
                pred = np.zeros((len(te_idx)*T, len(held_neurons)))
                for j, nrn in enumerate(held_neurons):
                    ytr = spikes[tr_idx][:, :, nrn].reshape(-1)
                    try:
                        glm = PoissonRegressor(alpha=1.0, max_iter=300).fit(Ftr_s, ytr)
                        pred[:, j] = glm.predict(Fte_s)
                    except Exception:
                        pred[:, j] = ytr.mean()
                yte = spikes[te_idx][:, :, held_neurons].reshape(-1, len(held_neurons))
                out[name].append(bits_per_spike(yte, pred))
    return {name: float(np.nanmean(v)) for name, v in out.items()}

# ======================================================================
# Fig 2d/e — OLE decoding + population-size sweep
# ======================================================================
def _ridge_cv_r2(X, V, cond, n_folds=5, seed=42):
    n, T, F = X.shape
    y = (cond == 'Hit').astype(int)
    strat = (y.sum() >= n_folds) and ((n-y.sum()) >= n_folds)
    a = RidgeCV(alphas=np.logspace(-3, 6, 30), cv=5).fit(
        StandardScaler().fit_transform(X.reshape(-1, F)), V.reshape(-1)).alpha_
    sp = StratifiedKFold(n_folds, shuffle=True, random_state=seed) if strat else KFold(n_folds, shuffle=True, random_state=seed)
    it = sp.split(np.arange(n), y) if strat else sp.split(np.arange(n))
    preds = np.zeros_like(V); sc = StandardScaler()
    for tri, tei in it:
        A = X[tri].reshape(-1, F); b = V[tri].reshape(-1); C = X[tei].reshape(-1, F)
        sc.fit(A); preds[tei] = Ridge(alpha=a).fit(sc.transform(A), b).predict(sc.transform(C)).reshape(len(tei), T)
    # excl-flat R^2 (honest headline)
    k = (V.max(1)-V.min(1)) >= 0.10*(V.max()-V.min())
    r2 = 1 - np.sum((V[k]-preds[k])**2)/(np.sum((V[k]-V[k].mean())**2)+1e-12)
    return float(r2), preds

def decoding_comparison(V, cond, feature_sets):
    """feature_sets: {name:(n,T,dim)} incl LFADS, GPFA, Smooth, Bin.
    Returns {name: r2_excl_flat}."""
    return {name: _ridge_cv_r2(X, V, cond)[0] for name, X in feature_sets.items()}

def population_sweep(V, cond, smoothed, binned, sizes=(5, 10, 25, 50, 100), seed=0):
    """Paper Fig 2e spirit: R^2 vs #neurons for smoothing/binning (which we
    CAN subsample). LFADS/GPFA use full factors (can't refit here) -> drawn
    as horizontal reference lines by the caller. Returns {'Smooth':[(size,r2)],
    'Bin':[...]}."""
    N = smoothed.shape[2]; rng = np.random.default_rng(seed)
    out = {'Smooth': [], 'Bin': []}
    for s in sizes:
        if s > N: continue
        idx = rng.choice(N, s, replace=False)
        out['Smooth'].append((s, _ridge_cv_r2(smoothed[:, :, idx], V, cond)[0]))
        out['Bin'].append((s, _ridge_cv_r2(binned[:, :, idx], V, cond)[0]))
    return out

# ======================================================================
# Fig 2b — example-neuron rates (condition-averaged), model comparison
# GPFA rates via GLM from GPFA factors (paper does the same).
# ======================================================================
def glm_rates(factors, spikes, seed=0):
    """Fit Poisson GLM factors->each neuron (cross-validated over trials) to
    get model rates for a model that only provides factors (GPFA)."""
    n, T, N = spikes.shape; F = factors.shape[2]
    kf = KFold(5, shuffle=True, random_state=seed); rates = np.zeros_like(spikes)
    for tri, tei in kf.split(np.arange(n)):
        sc = StandardScaler().fit(factors[tri].reshape(-1, F))
        Ftr = sc.transform(factors[tri].reshape(-1, F)); Fte = sc.transform(factors[tei].reshape(-1, F))
        for nrn in range(N):
            ytr = spikes[tri][:, :, nrn].reshape(-1)
            try:
                g = PoissonRegressor(alpha=1.0, max_iter=300).fit(Ftr, ytr)
                rates[tei][:, :, nrn] = g.predict(Fte).reshape(len(tei), T)
            except Exception:
                rates[tei][:, :, nrn] = ytr.mean()
    return rates

# ======================================================================
# Fig 3 — jPCA (Churchland 2012): skew-symmetric fit to PCA'd rate deriv
# ======================================================================
def jpca(trajs, n_pca=6):
    """trajs: (C,T,N) condition/trial-averaged rate trajectories.
    Returns 2D projection onto the top rotational plane: (C,T,2)."""
    C, T, N = trajs.shape
    X = trajs.reshape(-1, N)
    X = X - X.mean(0)
    n_pca = min(n_pca, N, X.shape[0]-1)
    pcs = PCA(n_pca).fit(X)
    Z = pcs.transform(trajs.reshape(-1, N)).reshape(C, T, n_pca)
    # state and derivative pairs
    Xstate = Z[:, :-1, :].reshape(-1, n_pca)
    Xdot = (Z[:, 1:, :] - Z[:, :-1, :]).reshape(-1, n_pca)
    # fit Xdot = M Xstate with M skew-symmetric, via least squares on the
    # skew-basis (Churchland 2012 appendix): solve for k in M=sum k_i B_i
    idx = [(i, j) for i in range(n_pca) for j in range(i+1, n_pca)]
    Bs = []
    for (i, j) in idx:
        B = np.zeros((n_pca, n_pca)); B[i, j] = 1; B[j, i] = -1; Bs.append(B)
    # design: for each sample, dot = M x -> stack
    A = np.zeros((Xdot.size, len(Bs)))
    for b, B in enumerate(Bs):
        A[:, b] = (Xstate @ B.T).reshape(-1)
    k, *_ = np.linalg.lstsq(A, Xdot.reshape(-1), rcond=None)
    M = sum(kb*B for kb, B in zip(k, Bs))
    evals, evecs = np.linalg.eig(M)
    order = np.argsort(-np.abs(evals.imag))
    v = evecs[:, order[0]]
    plane = np.column_stack([v.real, v.imag])
    q, _ = np.linalg.qr(plane)
    proj = (Z.reshape(-1, n_pca) @ q).reshape(C, T, 2)
    return proj

# ======================================================================
# PER-SESSION FIGURE ASSEMBLY
# ======================================================================
def cond_avg(arr, cond, c):
    m = cond == c
    return arr[m].mean(0) if m.sum() else None

def gpfa_decoding_r2(gpfa_mat, spikes_mat, t_before=1.5, t_after=1.5):
    """Decode lever from GPFA factors, aligning GPFA's own trials (via trialId)
    to behavior. Returns r2_excl_flat or None."""
    g = load_gpfa_cosmoothing(gpfa_mat)
    if g is None: return None
    Xg, _, tid = g
    with h5py.File(spikes_mat, 'r') as f:
        ib = f['IntanBehaviour']
        tv = np.asarray(ib['time'][()], np.float64).ravel()
        lv = np.asarray(ib['leverTrace'][()], np.float64).ravel()
        hr = np.asarray(ib['cueHit'][()], np.float64); mr = np.asarray(ib['cueMiss'][()], np.float64)
        hs = (hr[0] if hr.ndim == 2 else hr.ravel())/1000.0
        ms = (mr[0] if mr.ndim == 2 else mr.ravel())/1000.0
    all_cues = np.concatenate([hs, ms])
    Tg = Xg.shape[1]
    V = np.full((len(tid), Tg), np.nan); ok = np.ones(len(tid), bool)
    for i, t in enumerate(tid):
        ci = int(t) - 1
        if ci < 0 or ci >= len(all_cues): ok[i] = False; continue
        edges = np.linspace(all_cues[ci]-t_before, all_cues[ci]+t_after, Tg+1)
        idx = np.searchsorted(tv, edges)
        for b in range(Tg):
            s, e = idx[b], idx[b+1]
            if e > s: V[i, b] = lv[s:e].mean()
        good = ~np.isnan(V[i])
        if good.sum() >= 2: V[i, ~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), V[i, good])
        elif good.sum() < 1: ok[i] = False
    Xg, V = Xg[ok], V[ok]
    if len(V) < 10: return None
    r2, _ = _ridge_cv_r2(Xg, V, np.array(['x']*len(V)))
    return float(r2)

def build_session_figures(label, L, G, V, ok, out_dir, t_before=1.5, t_after=1.5,
                          gpfa_mat=None, lfads_h5=None, spikes_mat=None):
    _style()
    os.makedirs(out_dir, exist_ok=True)
    T = L['factors'].shape[1]
    tvec = np.linspace(-t_before, t_after, T)
    spikes = L['spikes']; smoothed = gaussian_smooth_spikes(spikes, 3)
    binned = spikes.copy()
    has_g = G is not None and G['factors'].shape[0] == L['factors'].shape[0]

    # ---------- FIG 2 (composite) ----------
    fig = plt.figure(figsize=(16, 11)); gs = gridspec.GridSpec(3, 3, hspace=0.5, wspace=0.35)

    # 2f co-smoothing — FAIR: trialId-aligned, common spike target
    bps = None
    if gpfa_mat and lfads_h5 and os.path.exists(gpfa_mat):
        try:
            fair = fair_cosmoothing(gpfa_mat, lfads_h5)
            if fair: bps = {'LFADS': fair['LFADS'], 'GPFA': fair['GPFA']}
        except Exception as e:
            print(f"    fair co-smoothing failed ({e}); LFADS-only")
    if bps is None:  # LFADS-only fallback (no GPFA or failed)
        bps = {'LFADS': cosmooth_core(L['factors'], spikes)}
    ax = fig.add_subplot(gs[0, 0])
    ax.bar(list(bps.keys()), list(bps.values()), color=[COL[k] for k in bps], edgecolor='white')
    ax.axhline(0, color='k', lw=0.8); ax.set_ylabel('bits/spike')
    ax.set_title('Fig 2f: held-out-neuron co-smoothing\n(same trials, common target)')

    # 2d decoding bars — GPFA decoded on its OWN trialId-aligned behavior
    feats = {'LFADS': L['factors'], 'Smooth': smoothed, 'Bin': binned}
    r2s = decoding_comparison(V[ok], L['cond'][ok],
                              {k: v[ok] for k, v in feats.items()})
    if gpfa_mat and spikes_mat and os.path.exists(gpfa_mat):
        try:
            g_r2 = gpfa_decoding_r2(gpfa_mat, spikes_mat, t_before, t_after)
            if g_r2 is not None: r2s['GPFA'] = g_r2
        except Exception as e:
            print(f"    GPFA decoding failed ({e})")
    ax = fig.add_subplot(gs[0, 1])
    order = [k for k in ('LFADS', 'GPFA', 'Smooth', 'Bin') if k in r2s]
    ax.bar(order, [r2s[k] for k in order], color=[COL[k] for k in order], edgecolor='white')
    ax.axhline(0, color='k', lw=0.8); ax.set_ylabel('R² (excl-flat)')
    ax.set_title('Fig 2d: lever decoding R²')

    # 2e population sweep
    ax = fig.add_subplot(gs[0, 2])
    sweep = population_sweep(V[ok], L['cond'][ok], smoothed[ok], binned[ok])
    for name in ('Smooth', 'Bin'):
        if sweep[name]:
            xs, ys = zip(*sweep[name]); ax.plot(xs, ys, 'o-', color=COL[name], label=name)
    ax.axhline(r2s['LFADS'], color=COL['LFADS'], ls='--', label='LFADS (full)')
    if 'GPFA' in r2s: ax.axhline(r2s['GPFA'], color=COL['GPFA'], ls='--', label='GPFA (full)')
    ax.set_xlabel('# neurons'); ax.set_ylabel('R²'); ax.set_title('Fig 2e: R² vs population'); ax.legend()

    # 2c t-SNE of g0
    ax = fig.add_subplot(gs[1, 0])
    if L['ic'] is not None and L['ic'].shape[0] >= 10:
        emb = TSNE(2, perplexity=min(30, (L['ic'].shape[0]-1)//3), random_state=0,
                   init='pca').fit_transform(L['ic'])
        for c in ('Hit', 'Miss'):
            m = L['cond'] == c
            if m.sum(): ax.scatter(emb[m, 0], emb[m, 1], s=18, c=COL[c], label=c, alpha=0.7, edgecolors='none')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'no g0 / too few trials', ha='center')
    ax.set_title('Fig 2c: g0 (IC) t-SNE'); ax.set_xticks([]); ax.set_yticks([])

    # 2b example neurons: condition-averaged rates, LFADS vs smoothed (+GPFA GLM)
    Lrates = L['rates']
    gpfa_rates = glm_rates(G['factors'], spikes) if has_g else None
    # pick 2 most Hit-modulated neurons
    hit_mod = np.abs(cond_avg(Lrates, L['cond'], 'Hit') - cond_avg(Lrates, L['cond'], 'Miss')).max(0) \
        if (L['cond'] == 'Miss').any() else Lrates.mean((0, 1))
    ex = np.argsort(-hit_mod)[:2]
    for r, nrn in enumerate(ex):
        ax = fig.add_subplot(gs[1, 1+r])
        for c in ('Hit', 'Miss'):
            if not (L['cond'] == c).any(): continue
            ax.plot(tvec, cond_avg(Lrates, L['cond'], c)[:, nrn], color=COL[c], lw=2, label=f'LFADS {c}')
            ax.plot(tvec, cond_avg(smoothed, L['cond'], c)[:, nrn], color=COL[c], lw=1, ls=':', alpha=0.7)
            if gpfa_rates is not None:
                ax.plot(tvec, cond_avg(gpfa_rates, L['cond'], c)[:, nrn], color=COL[c], lw=1, ls='--', alpha=0.7)
        ax.set_title(f'Fig 2b: neuron {nrn}\n(solid=LFADS, dot=smooth, dash=GPFA)')
        ax.set_xlabel('Time rel. cue (s)'); ax.set_ylabel('rate')
        if r == 0: ax.legend(fontsize=7)

    # 3 jPCA (LFADS rates, per-trial trajectories)
    ax = fig.add_subplot(gs[2, 0])
    try:
        proj = jpca(Lrates)  # (n,T,2)
        for c in ('Hit', 'Miss'):
            m = L['cond'] == c
            if m.sum():
                mean_traj = proj[m].mean(0)
                ax.plot(mean_traj[:, 0], mean_traj[:, 1], color=COL[c], lw=2, label=f'{c} mean')
                ax.plot(mean_traj[0, 0], mean_traj[0, 1], 'o', color=COL[c])
        ax.legend()
    except Exception as e:
        ax.text(0.5, 0.5, f'jPCA failed:\n{e}', ha='center', fontsize=7)
    ax.set_title('Fig 3: jPCA (LFADS rates)\ncondition-averaged rotation')
    ax.set_xlabel('jPC1'); ax.set_ylabel('jPC2')

    # jPCA single-trial (a few trials)
    ax = fig.add_subplot(gs[2, 1])
    try:
        for c in ('Hit', 'Miss'):
            m = np.where(L['cond'] == c)[0][:8]
            for i in m:
                ax.plot(proj[i, :, 0], proj[i, :, 1], color=COL[c], lw=0.6, alpha=0.5)
        ax.set_title('Fig 3f: jPCA single trials')
    except Exception:
        ax.text(0.5, 0.5, 'n/a', ha='center')
    ax.set_xlabel('jPC1'); ax.set_ylabel('jPC2')

    # summary text
    ax = fig.add_subplot(gs[2, 2]); ax.axis('off')
    lines = [f"Session: {label}", "",
             f"Co-smoothing bits/spike:",
             f"  LFADS = {bps.get('LFADS'):.4f}" if 'LFADS' in bps else "",
             (f"  GPFA  = {bps.get('GPFA'):.4f}" if 'GPFA' in bps else "  GPFA  = n/a"), "",
             "Decoding R² (excl-flat):"] + \
            [f"  {k:6s}= {r2s[k]:.3f}" for k in order] + \
            ["", f"n trials={L['factors'].shape[0]}, F={L['factors'].shape[2]}, N={spikes.shape[2]}"]
    ax.text(0, 1, "\n".join(l for l in lines if l is not None), va='top', family='monospace', fontsize=9)

    fig.suptitle(f'LFADS paper figures (2 & 3) reproduced — {label}', fontsize=14, weight='bold')
    png = os.path.join(out_dir, f'paper_figs_{label}.png')
    fig.savefig(png, dpi=140, bbox_inches='tight'); plt.close(fig)
    return png, {'bits_per_spike': bps, 'decoding_r2': r2s}

# ======================================================================
# DRIVER
# ======================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', required=True, help='JSON: [{label,lfads_h5,gpfa_mat,spikes_mat}, ...]')
    ap.add_argument('--out_dir', default='paper_figs')
    ap.add_argument('--t_before', type=float, default=1.5)
    ap.add_argument('--t_after', type=float, default=1.5)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    items = json.load(open(args.manifest))
    allres = {}
    for it in items:
        lbl = it['label']; print(f"\n=== {lbl} ===")
        try:
            L = load_lfads(it['lfads_h5'])
        except Exception as e:
            print(f"  LFADS load failed: {e}"); continue
        G = None
        if it.get('gpfa_mat') and os.path.exists(it['gpfa_mat']):
            try: G = load_gpfa(it['gpfa_mat'])
            except Exception as e: print(f"  GPFA load failed ({e}); LFADS-only")
        V, ok = load_lever(it['spikes_mat'], L['cond'], L['factors'].shape[1], args.t_before, args.t_after)
        png, res = build_session_figures(lbl, L, G, V, ok, args.out_dir, args.t_before, args.t_after,
                                         gpfa_mat=it.get('gpfa_mat'), lfads_h5=it['lfads_h5'],
                                         spikes_mat=it.get('spikes_mat'))
        print(f"  -> {png}")
        allres[lbl] = res
    json.dump(allres, open(os.path.join(args.out_dir, 'paper_figs_metrics.json'), 'w'), indent=2, default=str)
    print(f"\nMetrics -> {os.path.join(args.out_dir, 'paper_figs_metrics.json')}")

if __name__ == '__main__':
    main()