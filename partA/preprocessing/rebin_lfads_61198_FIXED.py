#!/usr/bin/env python3
"""
rebin_lfads_61198_FIXED.py
===========================
Corrected single-step version replacing prepare_lfads_61198.py + split_lfads_61198.py
+ rebin_lfads_61198.py.
THE BUG (same as 61199SOM):
  np.transpose(spikes, (2,1,0)) assumed shape (nNeurons, timeBins, nTrials)
  but lfads_stitch_data.h5 stores (nTrials, timeBins, nNeurons) already.
THE FIX:
  No transpose. Rebin 1ms -> 20ms, then split train/valid, write directly
  to the per-session output folder LFADS reads from.
"""

import os as _os
import h5py
import numpy as np
import os

INPUT_H5 = _os.environ.get('STITCH_H5', '/path/to/lfads_stitch_data.h5')
OUT_DIR  = _os.environ.get('BINNED_DIR', '/path/to/lfads_sessions_61198SOM_20ms_FIXED')
SESSIONS = ['dataset_01', 'dataset_02', 'dataset_03', 'dataset_04']
LABELS   = ['61198SOM_Day2', '61198SOM_Day3', '61198SOM_Day5', '61198SOM_Day7']
REBIN    = 20
VAL_FRAC = 0.2

os.makedirs(OUT_DIR, exist_ok=True)

with h5py.File(INPUT_H5, 'r') as fin:
    for ds_name, label in zip(SESSIONS, LABELS):
        spikes = fin[ds_name]['spikes'][:]   # (nTrials, 3000, nNeurons)
        n_trials_attr  = int(np.array(fin[ds_name].attrs['nTrials']).item())
        n_neurons_attr = int(np.array(fin[ds_name].attrs['nNeurons']).item())
        assert spikes.shape[0] == n_trials_attr
        assert spikes.shape[2] == n_neurons_attr
        nTrials, nTime, nNeurons = spikes.shape
        print(f"\n{label}:")
        print(f"  Input shape:  {spikes.shape}  (nTrials={nTrials}, "
              f"timeBins={nTime}, nNeurons={nNeurons})")
        nBins  = nTime // REBIN
        spikes = spikes[:, :nBins * REBIN, :]
        spikes = spikes.reshape(nTrials, nBins, REBIN, nNeurons).sum(axis=2)
        print(f"  After rebin:  {spikes.shape}")
        n_val   = max(1, int(nTrials * VAL_FRAC))
        n_train = nTrials - n_val
        train_data = spikes[:n_train]
        valid_data = spikes[n_train:]
        print(f"  Train: {train_data.shape}   Valid: {valid_data.shape}")
        cond = fin[ds_name]['condition'][:].flatten()
        cond_train = cond[:n_train]
        cond_valid = cond[n_train:]
        out_path = f"{OUT_DIR}/lfads_{label}.h5"
        with h5py.File(out_path, 'w') as fout:
            fout.create_dataset('train_encod_data', data=train_data.astype(np.float32))
            fout.create_dataset('train_recon_data',  data=train_data.astype(np.float32))
            fout.create_dataset('valid_encod_data', data=valid_data.astype(np.float32))
            fout.create_dataset('valid_recon_data',  data=valid_data.astype(np.float32))
            fout.create_dataset('train_condition', data=cond_train.astype(np.int32))
            fout.create_dataset('valid_condition', data=cond_valid.astype(np.int32))
            fout.attrs['label']    = label
            fout.attrs['nTrials']  = nTrials
            fout.attrs['nNeurons'] = nNeurons
print("\nDone! Output in:", OUT_DIR)

