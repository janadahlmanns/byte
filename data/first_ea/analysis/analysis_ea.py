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
EXPERIMENT_NAME = "ea_analysis"  # Used for file naming and report titles

# HDF5 file containing EA variant data (omit .h5 extension)
EXPERIMENT_HDF5 = "first_ea"

# Benchmark data (optional): List of tuples (benchmark_display_name, hdf5_filename_without_extension)
# Leave as empty list [] if no benchmarks to compare
BENCHMARK_HDF5_FILES = []

# Runs to show in detail (for visualizations)
RUNS_TO_SHOW_IN_DETAIL = [1, 2, 3]

# Color scheme for visualizations
PRIMARY_COLOR = "#0B3D2E"      # Dark green for best performing variants
SECONDARY_COLOR = "#8B3A3A"    # Wine red for worst performing variants
TERTIARY_COLOR = "#4A7C8C"     # Grayish ice blue for benchmarks


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


def _load_hdf5_variants_to_dataframe(hdf5_path: Path, source_label: str = None) -> pd.DataFrame:
    """
    Load all variants from HDF5 file into a single DataFrame.
    
    Args:
        hdf5_path: Path to HDF5 file
        source_label: Optional label to add as 'source' column (e.g., 'experiment', 'benchmark_name')
    
    Returns:
        DataFrame with columns from summary dataset, plus 'variant' and optional 'source' columns
    """
    all_data = []
    
    with h5py.File(hdf5_path, 'r') as f:
        # Find all variant groups (variant_01, variant_02, etc.)
        variant_names = sorted([key for key in f.keys() if key.startswith('variant_')])
        
        for variant_name in variant_names:
            variant_group = f[variant_name]
            
            # Load summary dataset if it exists
            if 'summary' in variant_group:
                summary_data = variant_group['summary'][:]
                
                # Convert structured array to DataFrame
                df_variant = pd.DataFrame(summary_data)
                
                # Add variant identifier
                df_variant['variant'] = variant_name
                
                # Add source label if provided
                if source_label is not None:
                    df_variant['source'] = source_label
                
                all_data.append(df_variant)
    
    if not all_data:
        raise ValueError(f"No variant data found in {hdf5_path}")
    
    df = pd.concat(all_data, ignore_index=True)
    return df 


def _calculate_overview_statistics(data_series: pd.Series) -> dict:
    """
    Calculate comprehensive overview statistics for a data series.
    
    Args:
        data_series: Pandas Series of numeric data
    
    Returns:
        Dictionary with all statistics
    """
    values = data_series.values
    
    return {
        'mean': values.mean(),
        'median': np.median(values),
        'std': values.std(),
        'min': values.min(),
        'max': values.max(),
        'range': values.max() - values.min(),
        'iqr': np.percentile(values, 75) - np.percentile(values, 25),
        'p5': np.percentile(values, 5),
        'p25': np.percentile(values, 25),
        'p75': np.percentile(values, 75),
        'p95': np.percentile(values, 95),
        'skewness': skew(values),
        'kurtosis': kurtosis(values),
        'cv': (values.std() / values.mean()) * 100 if values.mean() != 0 else 0,
    }



# ==================================================================================================================================================
# SECTION C) DATA LOADING
# ==================================================================================================================================================

print("\n=== SECTION C: DATA LOADING ===")
print("Loading experiment data...")
print(f"Loading experiment data from: {EXPERIMENT_HDF5}")
experiment_hdf5_path = _find_hdf5_file(EXPERIMENT_HDF5)
df_experiment = _load_hdf5_variants_to_dataframe(experiment_hdf5_path, source_label="experiment")
print(f"  Loaded {len(df_experiment)} runs across {df_experiment['variant'].nunique()} variants")
df_all = df_experiment.copy()

print("Experiment data loaded.")
print("Loading benchmark data...")

# Load benchmark data if specified
df_benchmarks = None
if BENCHMARK_HDF5_FILES:
    benchmark_dfs = []
    for bench_name, bench_hdf5 in BENCHMARK_HDF5_FILES:
        try:
            bench_path = _find_hdf5_file(bench_hdf5)
            df_bench = _load_hdf5_variants_to_dataframe(bench_path, source_label=bench_name)
            benchmark_dfs.append(df_bench)
            print(f"  Loaded benchmark '{bench_name}': {len(df_bench)} runs")
        except FileNotFoundError as e:
            print(f"  Warning: Could not load benchmark '{bench_name}': {e}")
    
    if benchmark_dfs:
        df_benchmarks = pd.concat(benchmark_dfs, ignore_index=True)
        df_all = pd.concat([df_experiment, df_benchmarks], ignore_index=True)
else:
    print("  No benchmarks specified.")

print("All HDF5 data loaded.")
print("=== END SECTION C ===")

# HDF5 files dictionary for potential per-tick analyses
hdf5_files_dict = {'experiment': experiment_hdf5_path}
if BENCHMARK_HDF5_FILES:
    for bench_name, bench_hdf5 in BENCHMARK_HDF5_FILES:
        try:
            hdf5_files_dict[bench_name] = _find_hdf5_file(bench_hdf5)
        except FileNotFoundError:
            pass


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
