#!/usr/bin/env python3
"""
Recompute held-out-neuron co-smoothing against GPFA using the SWEEP-WINNING
models, so slide A6 refers to the same models as the rest of Part A.

The stored paper_figs_metrics.json was produced 2026-07-09 from
lfads_runs/61199SOM_CUE and lfads_runs/61198SOM -- neither is a sweep winner.
Model: cfg06 both mice (best autonomy at near-best recon).

Reuses fair_cosmoothing() from reproduce_lfads_paper.py unchanged: same trials
via trialId, common spike target, held-out neurons x held-out trials.
"""
import os as _os
# Paths come from environment variables so the repo runs anywhere.
# See paths.env. Defaults below are the machine this was written on.
LFADS_RUNS = _os.environ.get("LFADS_RUNS", "/path/to/lfads_runs/sweep_dyn")
SESSION_DATA = _os.environ.get("SESSION_DATA", "/path/to/MultiSessionData")
RESULTS_DIR = _os.environ.get("RESULTS_DIR", "./figures")
ROBUST_DIR = _os.environ.get("ROBUST_DIR", "./partA/results")
import json, importlib.util, numpy as np

spec = importlib.util.spec_from_file_location(
    "rp", _os.path.join(_os.path.dirname(__file__), "reproduce_lfads_paper.py"))
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)

GPFA = (f"{SESSION_DATA}/"
        "{mouse}/{day}/GPFA_fixedXDim15/GPFA_results.mat")
SWEEP = (f"{LFADS_RUNS}/{mouse}/{cfg}/"
         "lfads_output_lfads_{mouse}_{day}.h5")
SESSIONS = [("61199SOM", d, "cfg06") for d in ["Day2", "Day3", "Day4", "Day5"]] + \
           [("61198SOM", d, "cfg06") for d in ["Day2", "Day3", "Day5", "Day7"]]

old = json.load(open(_os.environ.get("OLD_METRICS", f"{RESULTS_DIR}/paper_figs_metrics.json")))
rows = []
print(f"{'session':<15}{'GPFA':>8}{'LFADS':>9}{'ratio':>8}{'(old LFADS)':>14}")
for mouse, day, cfg in SESSIONS:
    g = GPFA.format(mouse=mouse, day=day)
    l = SWEEP.format(mouse=mouse, cfg=cfg, day=day)
    try:
        r = rp.fair_cosmoothing(g, l)
    except Exception as e:
        print(f"{mouse[:5]} {day:<9} FAILED: {type(e).__name__}: {e}"); continue
    if r is None:
        print(f"{mouse[:5]} {day:<9} skipped (alignment)"); continue
    o = old.get(f"{mouse}_{day}", {}).get("bits_per_spike", {}).get("LFADS")
    rows.append(dict(sess=f"{mouse[:5]} {day}", gpfa=r["GPFA"], lfads=r["LFADS"],
                     n_trials=r["n_trials"], old_lfads=o))
    print(f"{mouse[:5]} {day:<9}{r['GPFA']:>8.3f}{r['LFADS']:>9.3f}"
          f"{r['LFADS']/r['GPFA']:>7.1f}x{(o if o else float('nan')):>14.3f}")

if rows:
    L = np.array([r["lfads"] for r in rows]); G = np.array([r["gpfa"] for r in rows])
    from scipy.stats import wilcoxon
    st, p = wilcoxon(L, G)
    print(f"\nmean GPFA {G.mean():.3f}   LFADS {L.mean():.3f}")
    print(f"LFADS higher in {int((L>G).sum())}/{len(L)} sessions")
    print(f"ratio median {np.median(L/G):.1f}x  range {(L/G).min():.1f}-{(L/G).max():.1f}x")
    print(f"Wilcoxon p = {p:.4f}")
    json.dump(rows, open(f"{RESULTS_DIR}/cosmooth_sweep.json", "w"),
              indent=2)
    print("-> cosmooth_sweep.json")
