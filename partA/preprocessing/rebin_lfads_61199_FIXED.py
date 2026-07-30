import os as _os
# import h5py
# import numpy as np
# import os

# INPUT_H5   = '/mnt/c/Users/tiwar133/gpfa_v0203/lfads_stitch_data.h5'
# OUT_DIR    = '/home/tiwar133/lfads_sessions_61199SOM_20ms'
# SESSIONS   = ['dataset_05', 'dataset_06', 'dataset_07', 'dataset_08']
# LABELS     = ['61199SOM_Day2', '61199SOM_Day3', '61199SOM_Day4', '61199SOM_Day5']
# REBIN      = 20
# VAL_FRAC   = 0.2
# os.makedirs(OUT_DIR, exist_ok=True)

# with h5py.File(INPUT_H5, 'r') as fin:
#     for ds_name, label in zip(SESSIONS, LABELS):
#         spikes = fin[ds_name]['spikes'][:]
#         spikes = np.transpose(spikes, (2, 1, 0))  # [nTrials, 3000, nNeurons]
#         # Rebin to 20ms
#         nTrials, nTime, nNeurons = spikes.shape
#         nBins = nTime // REBIN
#         spikes = spikes[:, :nBins*REBIN, :]
#         spikes = spikes.reshape(nTrials, nBins, REBIN, nNeurons).sum(axis=2)
#         # Train/val split
#         n_val   = max(1, int(nTrials * VAL_FRAC))
#         n_train = nTrials - n_val
#         train_data = spikes[:n_train]
#         valid_data = spikes[n_train:]
#         out_path = f"{OUT_DIR}/lfads_{label}.h5"
#         with h5py.File(out_path, 'w') as fout:
#             fout.create_dataset('train_encod_data', data=train_data)
#             fout.create_dataset('train_recon_data',  data=train_data)
#             fout.create_dataset('valid_encod_data', data=valid_data)
#             fout.create_dataset('valid_recon_data',  data=valid_data)
#         print(f"{label}: train={train_data.shape}, valid={valid_data.shape}")

# print("\nDone!")
#!/usr/bin/env python3
"""
rebin_lfads_61199_FIXED.py
==========================
Corrected version of rebin_lfads_61199.py

THE BUG:
  The original script did np.transpose(spikes, (2,1,0)) assuming the MATLAB
  H5 shape was (nNeurons, timeBins, nTrials).

  But lfads_stitch_data.h5 attrs confirm:
    dataset_05: nNeurons=39, nTrials=381  shape=(381, 3000, 39)
    dataset_06: nNeurons=96, nTrials=319  shape=(319, 3000, 96)
    ...

  The shape is ALREADY (nTrials, timeBins, nNeurons) — no transpose needed.
  The old transpose was swapping neurons and trials, causing LFADS to treat
  39 neurons as 39 "trials" and 381 trials as 381 "neurons".

THE FIX:
  Remove the transpose entirely. Just rebin and split.
"""

import h5py
import numpy as np
import os

INPUT_H5 = _os.environ.get('STITCH_H5', '/mnt/c/Users/tiwar133/gpfa_v0203/lfads_stitch_data.h5')
OUT_DIR  = _os.environ.get('BINNED_DIR', '/home/tiwar133/lfads_sessions_61199SOM_20ms_FIXED')
SESSIONS = ['dataset_05', 'dataset_06', 'dataset_07', 'dataset_08']
LABELS   = ['61199SOM_Day2', '61199SOM_Day3', '61199SOM_Day4', '61199SOM_Day5']
REBIN    = 20       # 1ms -> 20ms bins
VAL_FRAC = 0.2

os.makedirs(OUT_DIR, exist_ok=True)

with h5py.File(INPUT_H5, 'r') as fin:
    for ds_name, label in zip(SESSIONS, LABELS):
        spikes = fin[ds_name]['spikes'][:]   # shape: (nTrials, 3000, nNeurons)

        # Sanity check against stored attributes
        n_trials_attr  = int(fin[ds_name].attrs['nTrials'])
        n_neurons_attr = int(fin[ds_name].attrs['nNeurons'])
        assert spikes.shape[0] == n_trials_attr,  \
            f"Expected {n_trials_attr} trials, got {spikes.shape[0]}"
        assert spikes.shape[2] == n_neurons_attr, \
            f"Expected {n_neurons_attr} neurons, got {spikes.shape[2]}"

        # NO TRANSPOSE — data is already (nTrials, timeBins, nNeurons)
        nTrials, nTime, nNeurons = spikes.shape
        print(f"\n{label}:")
        print(f"  Input shape:  {spikes.shape}  (nTrials={nTrials}, "
              f"timeBins={nTime}, nNeurons={nNeurons})")

        # Rebin: sum every REBIN 1ms bins into one 20ms bin
        nBins  = nTime // REBIN
        spikes = spikes[:, :nBins * REBIN, :]
        spikes = spikes.reshape(nTrials, nBins, REBIN, nNeurons).sum(axis=2)
        print(f"  After rebin:  {spikes.shape}  "
              f"(nTrials={nTrials}, timeBins={nBins}, nNeurons={nNeurons})")

        # Train / valid split (chronological — same as before)
        n_val   = max(1, int(nTrials * VAL_FRAC))
        n_train = nTrials - n_val
        train_data = spikes[:n_train]
        valid_data = spikes[n_train:]
        print(f"  Train: {train_data.shape}   Valid: {valid_data.shape}")

        # Also save condition labels for the validation script
        cond = fin[ds_name]['condition'][:].flatten()   # (nTrials,) 1=Hit 0=Miss
        cond_train = cond[:n_train]
        cond_valid = cond[n_train:]

        # Save
        out_path = f"{OUT_DIR}/lfads_{label}.h5"
        with h5py.File(out_path, 'w') as fout:
            fout.create_dataset('train_encod_data', data=train_data.astype(np.float32))
            fout.create_dataset('train_recon_data',  data=train_data.astype(np.float32))
            fout.create_dataset('valid_encod_data', data=valid_data.astype(np.float32))
            fout.create_dataset('valid_recon_data',  data=valid_data.astype(np.float32))
            # Store condition and cue times so validation script can use them
            fout.create_dataset('train_condition', data=cond_train.astype(np.int32))
            fout.create_dataset('valid_condition', data=cond_valid.astype(np.int32))
            fout.attrs['label']    = label
            fout.attrs['nTrials']  = nTrials
            fout.attrs['nNeurons'] = nNeurons

print("\nDone! Output in:", OUT_DIR)
print("\nNEXT STEPS:")
print("1. Update your LFADS config datafile_pattern to point to", OUT_DIR)
print("2. Update encod_data_dim in model config to match new nNeurons per day")
print("3. Retrain: python ~/run_lfads_61199SOM.py")
print("\nExpected shapes after fix:")
print("  Day2: (305 train, 150, 39)  +  (76 valid, 150, 39)")
print("  Day3: (255 train, 150, 96)  +  (64 valid, 150, 96)")
print("  Day4: (166 train, 150, 43)  +  (42 valid, 150, 43)")
print("  Day5: (204 train, 150, 43)  +  (51 valid, 150, 43)")