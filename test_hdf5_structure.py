import h5py
from pathlib import Path
import pandas as pd

hdf5_file = Path('data/pipeline_check/rawdata/2026-03-12_13-04-06_random_lookup.h5')
with h5py.File(hdf5_file, 'r') as f:
    print('Root keys:', list(f.keys()))
    if 'variant_01' in f:
        print('variant_01 contents:')
        for key in f['variant_01'].keys():
            item = f['variant_01'][key]
            if isinstance(item, h5py.Dataset):
                print(f'  {key}: Dataset, shape={item.shape}, dtype={item.dtype}')
                if 'wiring' in key or 'summary' in key:
                    df = pd.DataFrame(item[()])
                    print(f'    DataFrame columns: {df.columns.tolist()}')
                    print(f'    DataFrame shape: {df.shape}')
            elif isinstance(item, h5py.Group):
                print(f'  {key}: Group, contents={list(item.keys())}')
