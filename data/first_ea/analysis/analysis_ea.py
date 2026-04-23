"""
Analysis script for generating EA (Evolutionary Algorithm) simulation results report.

This script generates a Word document report with analysis of EA simulation data,
including statistical comparisons, visualizations, and descriptive statistics.
"""

# ==================================================================================================================================================
# SECTION A) IMPORTS AND INPUTS
# ==================================================================================================================================================

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import skew, kurtosis, shapiro, f_oneway, kruskal, mannwhitneyu, ttest_ind
try:
    from scipy.integrate import trapezoid as trapz
except ImportError:
    from scipy.integrate import trapz
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Add workspace root to path for imports
_current_path = Path(__file__).resolve()
_workspace_root = None
while _current_path.parent != _current_path:
    if (_current_path / "data").exists() and (_current_path / "simulate").exists():
        _workspace_root = _current_path
        break
    _current_path = _current_path.parent

if _workspace_root:
    sys.path.insert(0, str(_workspace_root))
    # from analysis_tools.network_visualization import network_viz

# =====================================================================
# User Configuration and Data Selection
# =====================================================================

# Experiment name
EXPERIMENT_NAME = "slow_mutation"  # Used for file naming and report titles

# HDF5 file containing EA variant data (omit .h5 extension)
EXPERIMENT_HDF5 = "2026-04-22_20-57-56_slow_mutation"

# Benchmark data (optional): List of tuples (benchmark_display_name, hdf5_filename_without_extension)
# Leave as empty list [] if no benchmarks to compare
BENCHMARK_HDF5_FILES = [
    ("Hard-wired Lookup", "2026-04-22_20-24-58_lookup"),
    ("Random networks", "2026-04-22_20-38-37_random"),
]



# Color scheme for visualizations
PRIMARY_COLOR = "#0B3D2E"      # Dark green for best performing variants
SECONDARY_COLOR = "#8B3A3A"    # Wine red for worst performing variants
TERTIARY_COLOR = "#4A7C8C"     # Grayish ice blue for benchmarks
HIGHLIGHT_COLOR = "#D4AF37"     # Gold for highlights

# ==================================================================================================================================================
# SECTION B) HELPER FUNCTIONS
# ==================================================================================================================================================


def _find_hdf5_file(hdf5_name: str, search_dir: Path = None) -> Path:
    """
    Locate HDF5 file by name (with or without .h5 extension).
    
    Args:
        hdf5_name: Filename without .h5 extension
        search_dir: Directory to search in. If None, searches parent directory of script's directory
    
    Returns:
        Path to the HDF5 file
        
    Raises:
        FileNotFoundError: If file not found
    """
    if search_dir is None:
        # Script is in data/first_ea/analysis/, search in data/first_ea/
        search_dir = Path(__file__).resolve().parent.parent
    
    # Try with .h5 extension
    hdf5_path = search_dir / f"{hdf5_name}.h5"
    if hdf5_path.exists():
        return hdf5_path
    
    # Try without modification (in case user included extension)
    hdf5_path = search_dir / hdf5_name
    if hdf5_path.exists():
        return hdf5_path
    
    raise FileNotFoundError(f"HDF5 file not found: {hdf5_name} in {search_dir}")


def _load_ea_attributes(hdf5_path: Path) -> dict:
    """
    Load experiment parameters from HDF5 file top-level attributes.
    
    Args:
        hdf5_path: Path to HDF5 file
    
    Returns:
        Dictionary of attributes
    """
    with h5py.File(hdf5_path, 'r') as f:
        attrs = dict(f.attrs)
    return attrs


def _load_generation_stats(hdf5_path: Path) -> pd.DataFrame:
    """
    Load generation_stats dataset into DataFrame.
    
    Expected columns: generation, mean, median, min, max, std, iqr
    
    Args:
        hdf5_path: Path to HDF5 file
    
    Returns:
        DataFrame with generation statistics
    """
    with h5py.File(hdf5_path, 'r') as f:
        if 'generation_stats' not in f:
            raise ValueError("generation_stats dataset not found in HDF5 file")
        data = f['generation_stats'][:]
        df = pd.DataFrame(data)
    return df


def _load_elite_lifespans(hdf5_path: Path) -> pd.DataFrame:
    """
    Load elite_genomes/lifespans dataset into DataFrame.
    
    Args:
        hdf5_path: Path to HDF5 file
    
    Returns:
        DataFrame with elite lifespan data
    """
    with h5py.File(hdf5_path, 'r') as f:
        if 'elite_genomes/lifespans' not in f:
            raise ValueError("elite_genomes/lifespans dataset not found in HDF5 file")
        data = f['elite_genomes/lifespans'][:]
        df = pd.DataFrame(data)
    return df


def _load_elite_genomes(hdf5_path: Path) -> pd.DataFrame:
    """
    Load elite genome data into DataFrame.
    
    Each row represents one elite genome (elite_0, elite_1, etc.)
    Columns: elite_id (N), eta, and tonic_activation_0, tonic_activation_1, ...
    
    Args:
        hdf5_path: Path to HDF5 file
    
    Returns:
        DataFrame with elite genome data
    """
    all_data = []
    
    with h5py.File(hdf5_path, 'r') as f:
        if 'elite_genomes' not in f:
            raise ValueError("elite_genomes folder not found in HDF5 file")
        
        elite_group = f['elite_genomes']
        # Find all elite_N folders
        elite_folders = sorted([key for key in elite_group.keys() if key.startswith('elite_')])
        
        for elite_folder in elite_folders:
            try:
                elite_id = int(elite_folder.split('_')[1])
                folder = elite_group[elite_folder]
                
                # Load eta (it's stored as a 1D array with one element)
                if 'eta' not in folder:
                    print(f"  Warning: eta not found in {elite_folder}")
                    continue
                eta_array = folder['eta'][:]
                eta = eta_array[0] if len(eta_array) > 0 else eta_array[()]
                
                # Load tonic_activations
                if 'tonic_activations' not in folder:
                    print(f"  Warning: tonic_activations not found in {elite_folder}")
                    continue
                tonic_acts = folder['tonic_activations'][:]
                
                # Create row
                row = {'elite_id': elite_id, 'eta': eta}
                for i, val in enumerate(tonic_acts):
                    row[f'tonic_activation_{i}'] = val
                all_data.append(row)
            except Exception as e:
                print(f"  Warning: Error loading {elite_folder}: {e}")
                continue
    
    if not all_data:
        raise ValueError(f"No elite genome data found in {hdf5_path}")
    
    df = pd.DataFrame(all_data)
    return df 


def _load_connection_weights(hdf5_path: Path) -> dict:
    """
    Load connection weights for all elite genomes from HDF5 file.
    
    Returns a dictionary mapping elite_id -> connection_weights array.
    
    Args:
        hdf5_path: Path to HDF5 file
    
    Returns:
        Dictionary {elite_id: connection_weights_array, ...}
    """
    connection_weights = {}
    
    with h5py.File(hdf5_path, 'r') as f:
        if 'elite_genomes' not in f:
            raise ValueError("elite_genomes folder not found in HDF5 file")
        
        elite_group = f['elite_genomes']
        # Find all elite_N folders
        elite_folders = sorted([key for key in elite_group.keys() if key.startswith('elite_')])
        
        for elite_folder in elite_folders:
            try:
                elite_id = int(elite_folder.split('_')[1])
                folder = elite_group[elite_folder]
                
                # Load connection_weights
                if 'connection_weights' not in folder:
                    print(f"  Warning: connection_weights not found in {elite_folder}")
                    continue
                
                weights = folder['connection_weights'][:]
                connection_weights[elite_id] = weights
            except Exception as e:
                print(f"  Warning: Error loading connection_weights from {elite_folder}: {e}")
                continue
    
    return connection_weights


def _load_modulation_specs(hdf5_path: Path, source_label: str) -> pd.DataFrame:
    """
    Load modulation specifications for all elite genomes from HDF5 file.
    
    Each row represents one modulation entry (source neuron → target neuron, modulated by modulating_neuron).
    
    Args:
        hdf5_path: Path to HDF5 file
        source_label: Label to add as 'source' column (e.g., 'experiment', benchmark name)
    
    Returns:
        DataFrame with columns: source, elite, source, target, modulating_neuron, modulation_weight
        (Note: 'source' at front is the experiment/benchmark label)
    """
    all_data = []
    
    with h5py.File(hdf5_path, 'r') as f:
        if 'elite_genomes' not in f:
            raise ValueError("elite_genomes folder not found in HDF5 file")
        
        elite_group = f['elite_genomes']
        # Find all elite_N folders
        elite_folders = sorted([key for key in elite_group.keys() if key.startswith('elite_')])
        
        for elite_folder in elite_folders:
            try:
                elite_id = int(elite_folder.split('_')[1])
                folder = elite_group[elite_folder]
                
                # Load modulation_spec
                if 'modulation_spec' not in folder:
                    print(f"  Warning: modulation_spec not found in {elite_folder}")
                    continue
                
                mod_data = folder['modulation_spec'][:]
                
                # Convert structured array to DataFrame
                df_mod = pd.DataFrame(mod_data)
                
                # Add elite identifier
                df_mod['elite'] = elite_id
                
                # Add source label
                df_mod['source'] = source_label
                
                all_data.append(df_mod)
            except Exception as e:
                print(f"  Warning: Error loading modulation_spec from {elite_folder}: {e}")
                continue
    
    if not all_data:
        # Return empty DataFrame with correct structure
        return pd.DataFrame(columns=['source', 'elite', 'source_neuron', 'target_neuron', 'modulating_neuron', 'modulation_weight'])
    
    df = pd.concat(all_data, ignore_index=True)
    
    # Reorder columns: source at front, then elite, then the modulation columns
    # Get the modulation columns (exclude 'source' and 'elite')
    mod_cols = [col for col in df.columns if col not in ['source', 'elite']]
    df = df[['source', 'elite'] + mod_cols]
    
    return df



# ==================================================================================================================================================
# SECTION C) DATA LOADING
# ==================================================================================================================================================

print("\n=== SECTION C: DATA LOADING ===")
print("Loading experiment data...")
print(f"Loading experiment data from: {EXPERIMENT_HDF5}")
experiment_hdf5_path = _find_hdf5_file(EXPERIMENT_HDF5)

# Load experiment attributes
print("  Loading experiment parameters...")
experiment_attrs = _load_ea_attributes(experiment_hdf5_path)
print(f"  Found {len(experiment_attrs)} experiment parameters")

# Load experiment generation stats
print("  Loading generation_stats...")
df_generation_stats_exp = _load_generation_stats(experiment_hdf5_path)
print(f"  Loaded generation_stats with {len(df_generation_stats_exp)} generations")
print("\nExperiment generation_stats head:")
print(df_generation_stats_exp.head())

# Load experiment elite lifespans
print("  Loading elite lifespans...")
df_elite_lifespans_exp = _load_elite_lifespans(experiment_hdf5_path)
print(f"  Loaded elite_lifespans with shape {df_elite_lifespans_exp.shape}")
print("\nExperiment elite_lifespans head:")
print(df_elite_lifespans_exp.head())

# Load experiment elite genomes
print("  Loading elite genomes...")
df_elite_genomes_exp = _load_elite_genomes(experiment_hdf5_path)
print(f"  Loaded {len(df_elite_genomes_exp)} elite genomes with {len(df_elite_genomes_exp.columns)} columns")
print("\nExperiment elite_genomes head:")
print(df_elite_genomes_exp.head())

# Load experiment connection weights
print("  Loading connection weights...")
connection_weights_experiment = _load_connection_weights(experiment_hdf5_path)
print(f"  Loaded connection weights for {len(connection_weights_experiment)} elite genomes")
for elite_id, weights in connection_weights_experiment.items():
    print(f"    elite_{elite_id}: shape {weights.shape}")

# Load experiment modulation specs
print("  Loading modulation specs...")
df_modulation_specs_exp = _load_modulation_specs(experiment_hdf5_path, 'experiment')
print(f"  Loaded modulation specs with {len(df_modulation_specs_exp)} total entries")
print("\nExperiment modulation_specs head:")
print(df_modulation_specs_exp.head())

print("\nExperiment data loaded.")
print("Loading benchmark data...")

# Load benchmark data if specified
df_benchmarks_generations_stats = None
df_benchmarks_elite_lifespans = None
df_benchmarks_elite_genomes = None
df_benchmarks_modulation_specs = None
benchmark_attrs = {}
connection_weights_collection = {}  # Will collect all connection_weights structures

if BENCHMARK_HDF5_FILES:
    for bench_name, bench_hdf5 in BENCHMARK_HDF5_FILES:
        try:
            bench_path = _find_hdf5_file(bench_hdf5)
            print(f"\n  Loading benchmark '{bench_name}'...")
            
            # Load benchmark attributes
            bench_attrs = _load_ea_attributes(bench_path)
            benchmark_attrs[bench_name] = bench_attrs
            print(f"    Found {len(bench_attrs)} benchmark parameters")
            
            # Load benchmark generation stats
            df_gen_stats = _load_generation_stats(bench_path)
            df_gen_stats['source'] = bench_name
            if df_benchmarks_generations_stats is None:
                df_benchmarks_generations_stats = df_gen_stats
            else:
                df_benchmarks_generations_stats = pd.concat([df_benchmarks_generations_stats, df_gen_stats], ignore_index=True)
            print(f"    Loaded generation_stats with {len(df_gen_stats)} generations")
            
            # Load benchmark elite lifespans
            df_lifespans = _load_elite_lifespans(bench_path)
            print(f"    Loaded elite_lifespans with shape {df_lifespans.shape}")
            df_lifespans['source'] = bench_name
            if df_benchmarks_elite_lifespans is None:
                df_benchmarks_elite_lifespans = df_lifespans
            else:
                df_benchmarks_elite_lifespans = pd.concat([df_benchmarks_elite_lifespans, df_lifespans], ignore_index=True)
            
            # Load benchmark elite genomes
            df_genomes = _load_elite_genomes(bench_path)
            df_genomes['source'] = bench_name
            if df_benchmarks_elite_genomes is None:
                df_benchmarks_elite_genomes = df_genomes
            else:
                df_benchmarks_elite_genomes = pd.concat([df_benchmarks_elite_genomes, df_genomes], ignore_index=True)
            print(f"    Loaded {len(df_genomes)} elite genomes")
            
            # Load benchmark connection weights
            # Create a sanitized name for the dictionary key (replace spaces with underscores)
            bench_key = bench_name.replace(" ", "_").replace("-", "_").lower()
            connection_weights_key = f"connection_weights_{bench_key}"
            connection_weights_data = _load_connection_weights(bench_path)
            connection_weights_collection[connection_weights_key] = connection_weights_data
            print(f"    Loaded connection weights for {len(connection_weights_data)} elite genomes")
            
            # Load benchmark modulation specs
            df_mod_specs = _load_modulation_specs(bench_path, bench_name)
            if df_benchmarks_modulation_specs is None:
                df_benchmarks_modulation_specs = df_mod_specs
            else:
                df_benchmarks_modulation_specs = pd.concat([df_benchmarks_modulation_specs, df_mod_specs], ignore_index=True)
            print(f"    Loaded modulation specs with {len(df_mod_specs)} entries")
            
        except FileNotFoundError as e:
            print(f"  Warning: Could not load benchmark '{bench_name}': {e}")
        except Exception as e:
            print(f"  Warning: Error loading benchmark '{bench_name}': {e}")
else:
    print("  No benchmarks specified.")

print("\nAll HDF5 data loaded.")

# Print summary of connection weights collections
print("\nConnection Weights Collections:")
print(f"  connection_weights_experiment: {len(connection_weights_experiment)} elite genomes")
for elite_id, weights in connection_weights_experiment.items():
    print(f"    elite_{elite_id}: shape {weights.shape}")

if connection_weights_collection:
    for cw_key, cw_data in connection_weights_collection.items():
        print(f"  {cw_key}: {len(cw_data)} elite genomes")
        for elite_id, weights in cw_data.items():
            print(f"    elite_{elite_id}: shape {weights.shape}")

# Print summary of modulation specs
print("\nModulation Specs Collections:")
print(f"  df_modulation_specs_exp: {len(df_modulation_specs_exp)} entries")
if df_benchmarks_modulation_specs is not None:
    print(f"  df_benchmarks_modulation_specs: {len(df_benchmarks_modulation_specs)} total entries")
    for bench_name in df_benchmarks_modulation_specs['source'].unique():
        count = len(df_benchmarks_modulation_specs[df_benchmarks_modulation_specs['source'] == bench_name])
        print(f"    {bench_name}: {count} entries")

print("=== END SECTION C ===")


# ==================================================================================================================================================
# SECTION D) ANALYSIS
# ==================================================================================================================================================

print("\n=== SECTION D: ANALYSIS ===")
print("Creating document...")

from docx import Document

doc = Document()
doc.add_heading(f"EA Analysis Report: {EXPERIMENT_NAME}", level=0)
print("Document created.")



# endregion 5

# ==================================================================================================================================================
# SAVE REPORT
# ==================================================================================================================================================

print("Saving document...")

output_dir = Path(__file__).resolve().parent / 'reports'
output_dir.mkdir(exist_ok=True)
output_path = output_dir / f'{EXPERIMENT_NAME}_analysis_report.docx'

doc.save(str(output_path))
print(f"Report saved to: {output_path}")

print("=== END SECTION D ===")
print("\nAnalysis complete!")
