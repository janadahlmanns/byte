import h5py
from pathlib import Path
import pandas as pd
import sys
import tempfile

# Add workspace to path
WORKSPACE_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(WORKSPACE_ROOT))

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
print(f"Testing network visualization pipeline...\n")

try:
    from analysis_tools.network_visualization import network_viz
    
    # Load config
    neuron_positions, neuron_types = network_viz.load_network_viz_config('11')
    print(f"✓ Network config loaded")
    
    # Load wiring and modulation
    df_wiring = load_wiring_from_hdf5(hdf5_file, "variant_01")
    df_modulation = load_modulation_from_hdf5(hdf5_file, "variant_01")
    print(f"✓ Wiring and modulation loaded")
    
    # Create temp directory and save to CSV
    with tempfile.TemporaryDirectory() as tmpdir:
        wiring_csv = Path(tmpdir) / "wiring.csv"
        modulation_csv = Path(tmpdir) / "modulation.csv"
        
        df_wiring.to_csv(wiring_csv, index=False)
        if len(df_modulation) > 0:
            df_modulation.to_csv(modulation_csv, index=False)
        
        print(f"✓ CSV files created")
        
        # Define weight columns
        weight_columns = [
            'weight_initial',
            'weight_final_run_0001',
            'weight_final_run_0002',
            'weight_final_run_0003'
        ]
        panel_labels = [
            'Initial Wiring',
            'Run 1 - Final',
            'Run 2 - Final',
            'Run 3 - Final'
        ]
        
        # Try to draw networks
        output_path = Path(tmpdir) / "network_test.png"
        print(f"\nAttempting to call draw_and_combine_networks...")
        print(f"  wiring_csv: {wiring_csv}")
        print(f"  modulation_csv: {modulation_csv}")
        print(f"  output_path: {output_path}")
        
        result = network_viz.draw_and_combine_networks(
            wiring_csv=str(wiring_csv),
            weight_columns=weight_columns,
            panel_labels=panel_labels,
            output_path=str(output_path),
            modulation_csv=str(modulation_csv) if len(df_modulation) > 0 else None,
            neuron_positions=neuron_positions,
            neuron_types=neuron_types,
            title=f'Network Development - Benchmark Test'
        )
        
        print(f"✓ Network visualization completed: {result}")
        
except Exception as e:
    print(f"\n✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
