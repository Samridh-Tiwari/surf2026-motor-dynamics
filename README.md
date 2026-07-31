# surf2026-motor-dynamics

Code, trained models and derived results for the SURF 2026 technical report
*A Continuous Attractor in Mouse Motor Cortex, and a Causal Readout for
Closed-Loop Intervention* (Tiwari, Kolhe, Jayant).

Two mice, 9 recording sessions, 39 to 118 simultaneously recorded neurons in
forelimb motor cortex during a cued lever-pull task.

Part A fits LFADS to the spiking and analyses the learned dynamics.
Part B builds a causal, real-time detector of impending movement.

## Data

Data is provided on request. Contact `samridhtiwari3005@gmail.com`.

The recordings are too large for GitHub. The stitched spike archive is 1.7 GB,
the per-session behavioural files total 33 GB, and the LFADS per-session outputs
total 528 MB. GitHub limits files to 100 MB.

This repository contains the trained checkpoints (2.3 MB each) and every derived
number, so results can be checked or regenerated without the raw data.

## Quick start

```bash
git clone https://github.com/Samridh-Tiwari/surf2026-motor-dynamics.git
cd surf2026-motor-dynamics
pip install -r requirements.txt

cp paths.env paths.local.env      # edit the paths
source paths.local.env
```

To check a number without running anything, open `partA/results/`. Every value
quoted in the paper is stored there as JSON. The table below says which file
holds which number.

To regenerate a figure, set `LFADS_RUNS` and run the script, for example:

```bash
python partA/analysis/fig_crossanimal.py
```

All paths are read from environment variables, listed in `paths.env`. Nothing
inside the scripts needs editing.

## Where each number comes from

| Value | Script | Output |
|---|---|---|
| 8/8 sessions, 0.303 vs 0.080 bits/spike, median 4.1x | `partA/analysis/dyn_recompute_cosmooth.py` | `results/cosmooth_sweep.json` |
| Lever decode R2 0.47-0.77, baseline 0.26-0.68 | `partA/analysis/fig_lever_decode.py` | `results/lever_decode_stats.json` |
| Lever traces figure in the paper | `partA/analysis/fig_lever_traces.py` | `figures/lever_decode_traces.png` |
| Reconstruction and autonomy for every fitted config | `partA/analysis/fig_model_selection.py` | `results/model_selection_stats.json` |
| Autonomous rollout 94% pre-cue, 82% post-cue | `partA/analysis/dyn_slide_a7_bridge.py` | `results/rollout_stats.json` |
| 512 fixed points, 33 marginal, 95 stable, tau 84 ms | `partA/analysis/fig3_attractor_cfg06.py` | `results/fig3_attractor_stats.json` |
| Marginal counts, 5 models x 2 animals | `partA/analysis/dyn_robust_61198.py` | `results/robustness*.json` |
| Cue-locked input, 1.73x and 2.18x baseline | `partA/analysis/dyn_slide_a9.py` | `results/cue_stats_cfg06.json` |
| Steering 75% along vs 8% across, 1.38x chance | `partA/analysis/dyn_slide_a10.py` | `results/steering_stats.json` |
| Cross-session 0.02-0.21, cross-animal 0.11 | `partA/analysis/fig_generalize2.py` | `results/generalization_stats.json` |
| Cross-animal figure in the paper | `partA/analysis/fig_crossanimal.py` | `figures/crossanimal.png` |
| Flow field, 2 PCs hold 36% and 59% | `partA/analysis/fig_flowfield.py` | `results/flowfield_stats.json` |
| PCA dimensionality, Spearman r = -0.16 | `partA/analysis/reproduce_lfads_paper.py` | `results/dimensionality_*.json` |
| Detector AUC 0.877, 60% at ~5 false alarms/min | `partB/m1_realtime_test.m` | |
| Live: 80 ms lead, 90% early, 71% precision | `partB/closedloop_live_gui.m` | `run_stats.mat` |
| Circular-shift null, z = +6.7 and +8.2 | `partB/shuffle_control.m` | |
| Cross-day 0.485 to 0.740, ceiling 0.768 | `partB/m1_crossday.m`, `m1_deploy_crossday.m` | |
| M1 vs M2, 0 ms lag, r = 0.98-0.99 | `partB/m1m2_timing.m`, `m1m2_cuefree_test.m` | |
| Reward threshold 2.8197 / 2.8188 V | `partB/find_reward_threshold.m`, `genuine_pulls.m` | |

## Layout

```
paper/                the technical report (LaTeX) and its figures
partA/
  sweep_dyn.py        LFADS training and hyperparameter sweep
  preprocessing/      1 ms spikes to 20 ms bins, per animal
  analysis/           analysis and figure scripts
  models/             trained cfg06 checkpoint and hparams, per animal
  results/            JSON written by each script
partB/
  pipeline/           causal front end: CMR, filter, threshold, PCA, CUSUM
  *.m                 detection, validation, cross-day, cross-area
figures/              final figures
```

## Labelling

Animals are `61199SOM` and `61198SOM`. Sessions are `Day2` to `Day7` within an
animal. The stitched archive indexes the same sessions as `dataset_01` to
`dataset_08`. LFADS configurations are `cfg01` to `cfg10`; `cfg06` is the
selected model throughout.

`cfg06` is not the reconstruction winner, and that is deliberate. Reconstruction
loss cannot tell dynamics apart from injected input. Run `cfg01`, the
reconstruction winner, with all inferred input switched off and its generator
reproduces only 47% of the trajectory in mouse 1 and 46% in mouse 2. `cfg06`
reproduces 97% and 87%, at 2.9% and 4.6% worse reconstruction.
`partA/analysis/fig_model_selection.py` measures both for every trained config.

## Environment

Part A needs Python with `numpy`, `scipy`, `h5py`, `scikit-learn`, `matplotlib`,
`pandas`, and `lfads-torch` with PyTorch. See `requirements.txt`.
Part B needs MATLAB R2026a.

All stochastic procedures are seeded, so the reported numbers reproduce exactly.

## Part A pipeline

```
lfads_stitch_data.h5                    raw spike counts, 1 ms
   |  partA/preprocessing/rebin_lfads_<animal>_FIXED.py
   v
20 ms binned sessions                   150 bins per trial, 80/20 split
   |  partA/sweep_dyn.py <animal>
   v
lfads_runs/<animal>/cfg06/              checkpoint and per-session outputs
   |  partA/analysis/*.py
   v
figures/ and partA/results/*.json
```

Rebin:

```bash
python partA/preprocessing/rebin_lfads_61199SOM_FIXED.py
```

Train. `SMOKE=1` runs one config for three epochs as a pipeline check.

```bash
SMOKE=1 python partA/sweep_dyn.py 61199SOM
python partA/sweep_dyn.py 61199SOM
```

Training can be skipped. The committed `partA/models/<animal>/cfg06.ckpt` is the
selected model and reproduces every number in the paper.

Analyse. Each script is standalone, prints its numbers, and writes a JSON file.

```bash
python partA/analysis/fig3_attractor_cfg06.py   # fixed points, attractor
python partA/analysis/fig_lever_traces.py       # lever decode
python partA/analysis/fig_crossanimal.py        # cross-animal alignment
python partA/analysis/fig_flowfield.py          # flow field, both animals
python partA/analysis/dyn_slide_a9.py           # cue-locked inferred input
python partA/analysis/dyn_slide_a10.py          # steering cost
python partA/analysis/dyn_recompute_cosmooth.py # held-out-neuron comparison
```

`fig3_attractor_cfg06.py` and `dyn_slide_a10.py` run 12,000 Adam iterations from
512 seeds and take several minutes on a GPU. The others take under a minute.

## Part B pipeline

`partB/pipeline/` holds the causal front end. Every stage looks only backward in
time, and filter and baseline state carry across file boundaries, so replaying a
recording gives the same output as the live stream.

```
.rhd files                              20 kHz, 64 ch, lever, TTL
   |  extractBehavioralEvents.m         -> behavioral_events.mat
   |  find_reward_threshold.m           recovers the hardware reward voltage
   |  genuine_pulls.m                   -> validated pull onsets
   v
labelled events
   |  bin_both_ports.m                  -> binned_M1_M2.mat
   |  train_best_model.m                fits the readout
   |  export_live_model.m               freezes it for the GUI
   v
trained detector
   |  m1_realtime_test.m                held-out and cross-day validation
   |  closedloop_replay_real.m          causal replay on a real session
   |  closedloop_live_gui.m             live
   |  shuffle_control.m                 base-rate-corrected null
   v
run_stats.mat and reported numbers
```

From MATLAB, with `partB/` and `partB/pipeline/` on the path:

```matlab
extractBehavioralEvents
find_reward_threshold
[onset, thrV, amp] = genuine_pulls(sessionDir, 1.0);
bin_both_ports(sessionDir, 64);          % Port B = M1, Port A = M2
m1_realtime_test
shuffle_control('runs/run_<timestamp>/run_stats.mat')
```

Live detector:

```matlab
closedloop_live_gui                      % GUI, RHD file mode
closedloop_live_gui('selftest')          % headless, checks live == offline
```

The GUI takes three data sources: RHD files replayed through the full causal
pipeline, RHX live over TCP to the Intan waveform port, and a synthetic source
for testing. Each run writes its own `run_stats.mat`, which `shuffle_control.m`
reads.

Cross-day and cross-area analyses:

```matlab
m1_crossday                              % Day1 model on Day2, aligned and not
m1_deploy_crossday                       % seeded calibration
m1m2_timing                              % cross-correlation lag, ramp latency
m1m2_cuefree_test                        % self-initiated trials only
verify_m1m2                              % controls on the M2 comparison
```

## Contact

Samridh Tiwari, `samridhtiwari3005@gmail.com`
Nano Neurotechnology Laboratory, Weldon School of Biomedical Engineering,
Purdue University.
