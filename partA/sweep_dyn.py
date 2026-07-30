#!/usr/bin/env python3
"""Manual AutoLFADS-style hyperparameter sweep using the PROVEN standard trainer.
Trains N full LFADS models with different regularization knobs on cue-aligned
(FIXED) data, ranks by validation reconstruction (valid/recon_smth).

Usage:  python sweep_dyn.py <mouse>       e.g. 61199SOM
Env:    SMOKE=1  -> 1 config, 3 epochs, no posterior sampling (pipeline check).
"""
import os, os as _os, sys, json, random, glob
from pathlib import Path
import pandas as pd
from lfads_torch.run_model import run_model

MOUSE = sys.argv[1] if len(sys.argv) > 1 else "61199SOM"
CFG_START = int(sys.argv[2]) if len(sys.argv) > 2 else 0
CFG_END = int(sys.argv[3]) if len(sys.argv) > 3 else None
SMOKE = os.environ.get("SMOKE", "0") == "1"
DATA = _os.environ.get("BINNED_DIR", f"/home/tiwar133/lfads_sessions_{MOUSE}_20ms_FIXED") + "/lfads_*.h5"
CFG  = f"../../lfads-torch/configs/lfads_{MOUSE}.yaml"   # relative to run_model.py location
ROOT = Path.home()/"lfads_runs"/"sweep_dyn"/MOUSE
ROOT.mkdir(parents=True, exist_ok=True)

# AutoLFADS knobs (log-uniform-ish discrete choices)
GRID = {
    "model.dropout_rate":                          [0.0, 0.1, 0.3],
    "model.train_aug_stack.transforms.0.cd_rate":  [0.2, 0.3, 0.5],
    "model.kl_co_scale":                           [1e-5, 1e-4, 5e-4],
    "model.kl_ic_scale":                           [1e-5, 1e-4],
    "model.l2_gen_scale":                          [1e-4, 1e-3, 1e-2],
    "model.l2_con_scale":                          [1e-4, 1e-3],
}
random.seed(0)
N = 1 if SMOKE else 10
configs = [{}]                                   # cfg00 = current defaults (baseline)
for _ in range(N-1):
    configs.append({k: random.choice(v) for k, v in GRID.items()})

# support parallel batches: if CFG_START/CFG_END provided, run only that range
if CFG_END is None: CFG_END = N
configs = configs[CFG_START:CFG_END]
cfg_ids = range(CFG_START, CFG_END)

results = []
for batch_idx, (i, hp) in enumerate(zip(cfg_ids, configs)):
    rd = ROOT/f"cfg{i:02d}"; (rd/"checkpoints").mkdir(parents=True, exist_ok=True)
    ov = {
        "datamodule.datafile_pattern":        DATA,
        "callbacks.model_checkpoint.dirpath": str(rd/"checkpoints"),
        "logger.csv_logger.save_dir":         str(rd/"csv_logs"),
        "logger.tensorboard_logger.save_dir": str(rd),
        "posterior_sampling.fn.filename":     str(rd/"lfads_output.h5"),
        "trainer.max_epochs":                 3 if SMOKE else 600,
        **{k: (f"{v:.1e}" if isinstance(v, float) and v < 0.01 else v) for k, v in hp.items()},
    }
    print(f"\n===== cfg{i:02d}  {hp if hp else '(baseline defaults)'} =====", flush=True)
    # resumability: skip a config that already finished (posterior output present)
    if (rd/"lfads_output.h5").exists() and not SMOKE:
        m = pd.read_csv(rd/"csv_logs"/"metrics.csv") if (rd/"csv_logs"/"metrics.csv").exists() else None
        best = float(m["valid/recon_smth"].min()) if m is not None and "valid/recon_smth" in m else None
        print(f"cfg{i:02d} already done (skip), best={best}", flush=True)
        results.append({"cfg": i, "hp": hp, "valid_recon_smth": best})
        json.dump(results, open(ROOT/f"sweep_results_b{CFG_START:02d}.json", "w"), indent=2); continue
    os.chdir(rd)
    best = None
    # power-cut / restart resilience: resume from last.ckpt if this config was mid-training
    resume_dir = str(rd/"checkpoints") if list((rd/"checkpoints").glob("*.ckpt")) else None
    if resume_dir: print(f"cfg{i:02d} resuming from checkpoint", flush=True)
    try:
        run_model(overrides=ov, config_path=CFG, do_train=True, do_posterior_sample=(not SMOKE),
                  checkpoint_dir=resume_dir)
        mcsv = rd/"csv_logs"/"metrics.csv"
        if mcsv.exists():
            m = pd.read_csv(mcsv)
            if "valid/recon_smth" in m: best = float(m["valid/recon_smth"].min())
    except Exception as e:
        print(f"cfg{i:02d} FAILED: {repr(e)[:200]}", flush=True)
    results.append({"cfg": i, "hp": hp, "valid_recon_smth": best})
    json.dump(results, open(ROOT/f"sweep_results_b{CFG_START:02d}.json", "w"), indent=2)
    print(f"cfg{i:02d} best valid/recon_smth = {best}", flush=True)

ok = [r for r in results if r["valid_recon_smth"] is not None]
if ok:
    winner = min(ok, key=lambda r: r["valid_recon_smth"])
    print(f"\n==== SWEEP DONE. WINNER: cfg{winner['cfg']:02d} "
          f"recon={winner['valid_recon_smth']:.5f}  hp={winner['hp']} ====", flush=True)
    json.dump(winner, open(ROOT/"winner.json", "w"), indent=2)
