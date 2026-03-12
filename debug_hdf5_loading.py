import h5py
from pathlib import Path
import pandas as pd
import sys

# Add workspace to path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

# Test the loading functions
def load_wiring_from_hdf5(hdf5_file: Path, variant_name: str = "variant_01") -> pd.DataFrame:
    """Load wiring data from HDF5 file for a specific variant."""
    with h5py.File(hdf5_file, 'r') as f:
        if variant_name not in f:
            variants = [key for key in f.keys() if key.startswith('variant_')]
            if variants:
                variant_name = sorted(variants)[0]
            else:
                raise ValueError(f"No variants found in {hdf5_file}")
        
        if 'wiring' not in f[variant_name]:
            raise ValueError(f"No wiring dataset in {variant_name}")
        
        wiring_dataset = f[variant_name]['wiring']
        df_wiring = pd.DataFrame(wiring_dataset[()])
    
    return df_wiring

def load_modulation_from_hdf5(hdf5_file: Path, variant_name: str = "variant_01") -> pd.DataFrame:
    """Load modulation data from HDF5 file for a specific variant."""
    with h5py.File(hdf5_file, 'r') as f:
        if variant_name not in f:
            variants = [key for key in f.keys() if key.startswith('variant_')]
            if variants:
                variant_name = sorted(variants)[0]
            else:
                raise ValueError(f"No variants found in {hdf5_file}")
        
        if 'modulation' not in f[variant_name]:
            return pd.DataFrame()  # Return empty DF if no modulation
        
        modulation_dataset = f[variant_name]['modulation']
        df_modulation = pd.DataFrame(modulation_dataset[()])
    
    return df_modulation

# Test with actual file
hdf5_file = Path('data/pipeline_check/rawdata/2026-03-12_13-04-06_random_lookup.h5')
print(f"Testing with: {hdf5_file}")

try:
    df_wiring = load_wiring_from_hdf5(hdf5_file, "variant_01")
    print(f"✓ Wiring loaded: {df_wiring.shape}")
    print(f"  Columns: {df_wiring.columns.tolist()}")
except Exception as e:
    print(f"✗ Error loading wiring: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

try:
    df_modulation = load_modulation_from_hdf5(hdf5_file, "variant_01")
    print(f"✓ Modulation loaded: {df_modulation.shape}")
except Exception as e:
    print(f"✗ Error loading modulation: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Now test with the network viz functions
import sys
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from analysis_tools.network_visualization import network_viz
    print("\nTesting network_viz functions...")
    
    neuron_positions, neuron_types = network_viz.load_network_viz_config('11')
    print(f"✓ Network config loaded")
    print(f"  Neuron positions: {len(neuron_positions)} neurons")
    print(f"  Neuron types: {len(neuron_types)} neurons")
except Exception as e:
    print(f"✗ Error loading network config: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
