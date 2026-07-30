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
figures/        final figures
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

## Running the Part A scripts

The scripts contain absolute paths from the machine they were written on. Before
running, edit the path constants at the top of each file:

- `ROOT` or the `lfads_runs/...` path to point at your LFADS output directory
- the `MultiSessionData/...` path in `fig_lever_decode.py`,
  `fig_lever_traces.py` to point at the behavioural `.mat` files
- `DST` to your output directory

Each script prints its numbers to stdout and writes a `*_stats.json`.

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
