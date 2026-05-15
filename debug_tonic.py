import h5py
import numpy as np
from pathlib import Path

# Open one of the HDF5 files and inspect the tonic_activations structure
script_dir = Path(r'c:\work_hard\byte\data\first_ea\replays')
hdf5_path = script_dir / '2026-05-05_17-42-25_ea_from_random_genomes_all_runs_all.h5'
with h5py.File(hdf5_path, 'r') as f:
    vg = f['variant_0']
    if 'tonic_activations' in vg:
        ta = vg['tonic_activations'][:]
        print(f"Shape: {ta.shape}")
        print(f"Dtype: {ta.dtype}")
        print(f"Dtype names: {ta.dtype.names}")
        if ta.dtype.names:
            print(f"\nField names: {list(ta.dtype.names)}")
            print(f"\nSample row:")
            print(ta[0])
        else:
            print(f"\nFirst few values: {ta[:5]}")
