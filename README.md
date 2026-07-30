# Motor cortex dynamics and a causal readout for closed-loop intervention

Code, trained models, and derived results for the SURF 2026 technical report
*A Continuous Attractor in Mouse Motor Cortex, and a Causal Readout for
Closed-Loop Intervention* (Tiwari, Kolhe, Jayant).

Two mice, 9 recording sessions, 39 to 118 simultaneously recorded neurons in
forelimb motor cortex during a cued lever-pull task.

- **Part A** fits LFADS to the spiking and analyses the learned dynamics.
- **Part B** builds a causal, real-time detector of impending movement.

---

## Where each number in the paper comes from

| Reported value | Script | Output |
|---|---|---|
| 8/8 sessions, 0.303 vs 0.080 bits/spike, median 4.1× | `partA/analysis/dyn_recompute_cosmooth.py` | `results/cosmooth_sweep.json` |
| Lever decode R² 0.47–0.77, baseline 0.26–0.68 | `partA/analysis/fig_lever_decode.py` | `results/lever_decode_stats.json` |
| Traces figure used in the paper | `partA/analysis/fig_lever_traces.py` | `figures/lever_decode_traces.png` |
| Autonomous rollout 94% pre-cue, 82% post-cue | `partA/analysis/dyn_slide_a7_bridge.py` | `results/rollout_stats.json` |
| 512 fixed points, 33 marginal, 95 stable, τ 84 ms | `partA/analysis/fig3_attractor_cfg06.py` | `results/fig3_attractor_stats.json` |
| Marginal counts across 5 models × 2 animals | `partA/analysis/dyn_robust_61198.py` | `results/robustness*.json` |
| Cue-locked input, 1.73× and 2.18× baseline | `partA/analysis/dyn_slide_a9.py` | `results/cue_stats_cfg06.json` |
| Steering 75% along vs 8% across, 1.38× chance | `partA/analysis/dyn_slide_a10.py` | `results/steering_stats.json` |
| Cross-session 0.02–0.21, cross-animal 0.11 | `partA/analysis/fig_generalize2.py` | `results/generalization_stats.json` |
| Cross-animal figure used in the paper | `partA/analysis/fig_crossanimal.py` | `figures/crossanimal.png` |
| Flow field, 2 PCs hold 36% and 59% | `partA/analysis/fig_flowfield.py` | `results/flowfield_stats.json` |
| PCA dimensionality, Spearman r = −0.16 | `partA/analysis/reproduce_lfads_paper.py` | `results/dimensionality_*.json`, `scree_table.json` |
| Detector AUC 0.877, 60% at ~5 FA/min | `partB/m1_realtime_test.m` | |
| Live: 80 ms, 90% early, 71% precision | `partB/closedloop_live_gui.m` | per-run `run_stats.mat` |
| Circular-shift null, z = +6.7 and +8.2 | `partB/shuffle_control.m` | |
| Cross-day 0.485 → 0.740, ceiling 0.768 | `partB/m1_crossday.m`, `m1_deploy_crossday.m` | |
| M1 vs M2, 0 ms lag, r = 0.98–0.99 | `partB/m1m2_timing.m`, `m1m2_cuefree_test.m` | |
| Reward threshold 2.8197 / 2.8188 V | `partB/find_reward_threshold.m`, `genuine_pulls.m` | |

---

## Layout

```
paper/          the technical report (LaTeX) and its three figures
partA/
  preprocessing/  1 ms spikes to 20 ms bins, per animal
  analysis/       every Part A analysis and figure script
  models/         trained cfg06 checkpoint and hyperparameters, per animal
  results/        JSON emitted by each script, one value per reported number
partB/          real-time detection, validation, cross-day, cross-area (MATLAB)
  pipeline/       causal front end: CMR, filter, adaptive threshold, PCA, CUSUM
figures/        final figures
partA/sweep_dyn.py  LFADS hyperparameter sweep and training entry point
```

## Labelling scheme

- Animals: `61199SOM`, `61198SOM`
- Sessions: `Day2` to `Day7` within an animal
- Stitched archive indexes the same sessions as `dataset_01` to `dataset_08`
- LFADS configurations: `cfg01` to `cfg10`. **`cfg06` is the selected model
  throughout.** See the paper on why the reconstruction winner was rejected.

## Environment

Part A: conda environment `lfads-torch`, PyTorch with CUDA, `h5py`,
`scikit-learn`, `scipy`, `matplotlib`.
Part B: MATLAB R2026a.

All stochastic procedures are seeded (`torch.manual_seed(0)`,
`np.random.seed(0)`, `rng(7)`), so the reported numbers reproduce exactly.

## Part A pipeline: fitting and analysing the dynamics

```
lfads_stitch_data.h5                     raw spike counts, 1 ms, all sessions
   |
   |  partA/preprocessing/rebin_lfads_<animal>_FIXED.py
   v
lfads_sessions_<animal>_20ms_FIXED/      20 ms bins, 150 per trial, 80/20 split
   |
   |  partA/sweep_dyn.py <animal>        trains 10 configs, ranks by valid recon
   v
lfads_runs/sweep_dyn/<animal>/cfg06/     checkpoint + lfads_output_<session>.h5
   |
   |  partA/analysis/*.py
   v
figures/  and  partA/results/*.json
```

**1. Rebin.** Converts 1 ms counts to 20 ms bins and writes one HDF5 per session.

```bash
python partA/preprocessing/rebin_lfads_61199SOM_FIXED.py
```

**2. Train.** Runs the hyperparameter sweep. `SMOKE=1` does one config for three
epochs first, to check the pipeline before committing to a full run.

```bash
SMOKE=1 python partA/sweep_dyn.py 61199SOM     # pipeline check
python partA/sweep_dyn.py 61199SOM             # full sweep, 10 configs
```

To skip training entirely, use the committed `partA/models/<animal>/cfg06.ckpt`.
That is the selected model and it reproduces every number in the paper.

**3. Analyse.** Each script is standalone, prints its numbers, and writes a
`*_stats.json`. Order does not matter.

```bash
python partA/analysis/fig3_attractor_cfg06.py   # fixed points, attractor spectrum
python partA/analysis/fig_lever_traces.py       # lever decode, paper figure
python partA/analysis/fig_crossanimal.py        # cross-animal alignment
python partA/analysis/fig_flowfield.py          # flow field, both animals
python partA/analysis/dyn_slide_a9.py           # cue-locked inferred input
python partA/analysis/dyn_slide_a10.py          # steering cost, 75% vs 8%
python partA/analysis/dyn_recompute_cosmooth.py # held-out-neuron comparison
```

Fixed-point scripts (`fig3_attractor_cfg06.py`, `dyn_slide_a10.py`) run 12,000
Adam iterations from 512 seeds and take several minutes on a GPU. The rest run
in under a minute.

**Before running any of them**, edit the path constants at the top of the file:
`ROOT` (LFADS output directory), `DST` (where figures go), and in
`fig_lever_decode.py` / `fig_lever_traces.py` the `MultiSessionData` path to the
behavioural `.mat` files.

## Part B pipeline: real-time detection

`partB/pipeline/` holds the causal front end. Every stage only looks backward in
time, and filter and baseline state carry across file boundaries, so replaying a
recording gives bit-identical output to the live stream.

```
.rhd files                               raw 20 kHz, 64 ch + lever + TTL
   |
   |  extractBehavioralEvents.m          -> behavioral_events.mat
   |  find_reward_threshold.m            recovers the hardware reward voltage
   |  genuine_pulls.m                    -> validated pull onsets
   v
labelled events
   |
   |  bin_both_ports.m                   -> binned_M1_M2.mat (skips raw re-reads)
   |  train_best_model.m                 fits the readout
   |  export_live_model.m                freezes it for the GUI
   v
trained detector
   |
   |  m1_realtime_test.m                 held-out + cross-day validation
   |  closedloop_replay_real.m           full causal replay on a real session
   |  closedloop_live_gui.m              live, streaming
   |  shuffle_control.m                  base-rate-corrected null
   v
run_stats.mat  and  reported numbers
```

Typical order, from MATLAB with `partB/` and `partB/pipeline/` on the path:

```matlab
extractBehavioralEvents                  % script, walks the .rhd files
find_reward_threshold                    % prints the hardware threshold
[onset, thrV, amp] = genuine_pulls(sessionDir, 1.0);
bin_both_ports(sessionDir, 64);          % Port B = M1, Port A = M2
m1_realtime_test                         % AUC, lead times, cross-day
shuffle_control('runs/run_<timestamp>/run_stats.mat')
```

Live detector:

```matlab
closedloop_live_gui                      % opens the GUI, RHD file mode
closedloop_live_gui('selftest')          % headless, checks live == offline
```

Three data sources are selectable in the GUI: **RHD files** (raw files replayed
through the full causal pipeline, no preprocessing needed), **RHX live** (TCP to
the Intan RHX waveform port), and a synthetic source for testing. Each run writes
its own `run_stats.mat`, which is what `shuffle_control.m` consumes.

Cross-day and cross-area analyses are standalone:

```matlab
m1_crossday                              % Day1 model on Day2, aligned vs not
m1_deploy_crossday                       % seeded calibration, early-session recall
m1m2_timing                              % cross-correlation lag, ramp latency
m1m2_cuefree_test                        % self-initiated only, M1 vs M2
verify_m1m2                              % controls on the M2 comparison
```

## Data

The recordings are too large for GitHub: the stitched spike archive is 1.7 GB,
the per-session behavioural files total 33 GB, and the LFADS per-session outputs
total 528 MB, against a 100 MB per-file limit here.

**The full dataset is hosted on Purdue Box. Access is granted on request to the
corresponding author.** This covers the raw `.rhd` recordings with their
behavioural channels, the stitched spike archive, the rebinned model inputs, and
the LFADS per-session outputs.

Included in this repository instead:

- **Trained model checkpoints** (`partA/models/`, 2.3 MB each) with their
  `hparams.yaml`. These regenerate every Part A result without retraining.
- **All derived numbers** (`partA/results/`), so every value quoted in the paper
  can be checked without running anything or obtaining the data.

## Contact

Samridh Tiwari, `tiwar133@purdue.edu`
Nano Neurotechnology Laboratory, Weldon School of Biomedical Engineering,
Purdue University.
