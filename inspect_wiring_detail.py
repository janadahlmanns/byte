import h5py
from pathlib import Path

hdf5_file = Path('data/pipeline_check/rawdata/2026-03-12_13-04-06_random_lookup.h5')
with h5py.File(hdf5_file, 'r') as f:
    variant = f['variant_01']
    
    if 'wiring' in variant:
        wiring_item = variant['wiring']
        print(f"wiring is a: {type(wiring_item).__name__}")
        
        if isinstance(wiring_item, h5py.Group):
            print(f"wiring is a Group with contents: {list(wiring_item.keys())}")
            for key in list(wiring_item.keys())[:5]:
                item = wiring_item[key]
                print(f"  {key}: {type(item).__name__}, shape={item.shape if hasattr(item, 'shape') else 'N/A'}")
        elif isinstance(wiring_item, h5py.Dataset):
            print(f"wiring is a Dataset:")
            print(f"  shape: {wiring_item.shape}")
            print(f"  dtype: {wiring_item.dtype}")
