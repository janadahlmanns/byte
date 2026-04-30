#!/usr/bin/env python3
"""Quick script to check HDF5 file structure"""
import h5py

hdf5_file = 'data/first_ea/2026-04-30_14-32-28_first_ea.h5'

with h5py.File(hdf5_file, 'r') as f:
    print("Top-level attributes:")
    for k, v in f.attrs.items():
        if 'neuron' in k.lower():
            print(f"  {k}: {v}")
    
    print("\nElite genomes:")
    elite_group = f['elite_genomes']
    elite_list = sorted([k for k in elite_group.keys() if k.startswith("elite_")])
    print(f"  Available: {elite_list[:5]}{'...' if len(elite_list) > 5 else ''}")
    
    if elite_list:
        elite_0 = elite_group[elite_list[0]]
        print(f"\n{elite_list[0]} datasets:")
        for k in elite_0.keys():
            ds = elite_0[k]
            print(f"  {k}: shape={ds.shape}, dtype={ds.dtype}")
            if k == 'connection_weights' and len(ds.shape) == 3:
                print(f"    -> Network has {ds.shape[0]} neurons")
