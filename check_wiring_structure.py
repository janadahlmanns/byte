import h5py
from pathlib import Path

hdf5_file = Path('data/pipeline_check/rawdata/2026-03-12_13-04-06_random_lookup.h5')
with h5py.File(hdf5_file, 'r') as f:
    variant = f['variant_01']
    print("variant_01 children:")
    for key in sorted(variant.keys())[:10]:  # First 10
        item = variant[key]
        if isinstance(item, h5py.Dataset):
            print(f"  {key}: Dataset (shape={item.shape}, dtype={item.dtype})")
        elif isinstance(item, h5py.Group):
            print(f"  {key}: Group (children={list(item.keys())})")
    
    # Check the 'wiring' item specifically
    if 'wiring' in variant:
        wiring_item = variant['wiring']
        print(f"\nwiring is a {type(wiring_item).__name__}")
        if isinstance(wiring_item, h5py.Group):
            print(f"  wiring Group children: {list(wiring_item.keys())}")
        else:
            print(f"  wiring Dataset shape: {wiring_item.shape}, dtype: {wiring_item.dtype}")
