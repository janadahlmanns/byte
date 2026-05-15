import pandas as pd
import h5py
from pathlib import Path
import sys
sys.path.insert(0, r'c:\work_hard\byte\data\first_ea\replays')

# Quick test to show variant_data structure
_variant_parts = []
script_dir = Path(r'c:\work_hard\byte\data\first_ea\replays')
hdf5_path = script_dir / '2026-05-05_17-42-25_ea_from_random_genomes_all_runs_all.h5'

with h5py.File(hdf5_path, 'r') as f:
    variant_keys = sorted([k for k in f.keys() if k.startswith('variant_')])
    for variant_key in variant_keys[:3]:  # Just first 3 for demo
        variant_id = int(variant_key.split('_')[1])
        vg = f[variant_key]
        eta_value = vg['eta'][()] if 'eta' in vg else None
        _variant_parts.append({
            'group': 'EA from Random',
            'type': 'experiment',
            'variant': variant_id,
            'eta': eta_value,
        })

variant_data = pd.DataFrame(_variant_parts)
print("variant_data structure:")
print(variant_data)
print(f"\nColumns: {list(variant_data.columns)}")
print(f"Dtypes:\n{variant_data.dtypes}")
