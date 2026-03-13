"""
Analysis script for generating comprehensive simulation results report.

This script generates a Word document report with analysis of simulation data,
including statistical comparisons, visualizations, and descriptive statistics.
"""

# ==================================================================================================================================================
# SECTION A) IMPORTS AND INPUTS
# ==================================================================================================================================================

import h5py
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt

# Add workspace root to path for imports
_current_path = Path(__file__).resolve()
_workspace_root = None
while _current_path.parent != _current_path:
    if (_current_path / "data").exists() and (_current_path / "simulate").exists():
        _workspace_root = _current_path
        break
    _current_path = _current_path.parent

if _workspace_root:
    import sys
    sys.path.insert(0, str(_workspace_root))
    from analysis_tools.network_visualization import network_viz

# =====================================================================
# User Configuration and Data Selection
# =====================================================================

# Experiment name
EXPERIMENT_NAME = "random_wiring"  # Used for file naming and report titles

# HDF5 file containing variant data (omit .h5 extension)
EXPERIMENT_HDF5 = "2026-03-13_09-51-21_random"

# Benchmark data (optional): List of tuples (benchmark_display_name, hdf5_filename_without_extension)
# Leave as empty list [] if no benchmarks to compare
BENCHMARK_HDF5_FILES = [
    ("Hard-wired Lookup", "2026-03-13_09-50-50_random_lookup"),
]

# Network visualization configuration (e.g., '11' for network_viz_11.yaml)
NETWORK_VIZ_CONFIG = "11"
RUNS_TO_SHOW_IN_DETAIL = [1,2,3]

# Color scheme for visualizations
PRIMARY_COLOR = "#0B3D2E"      # Dark green for successful variants
SECONDARY_COLOR = "#8B3A3A"    # Wine red for unsuccessful variants
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
        # Script is in data/temp/analysis/, search in data/temp/
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
        source_label: Optional label to add as 'source' column (e.g., 'random', 'benchmark_name')
    
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
        'cv': (values.std() / values.mean()) * 100,
    }


def _calculate_per_variant_statistics(df: pd.DataFrame, metric_col: str) -> dict:
    """
    Calculate statistics for each variant separately, returning min/max/avg of those values.
    
    Args:
        df: DataFrame with variant column and metric column
        metric_col: Column name to calculate statistics on
    
    Returns:
        Dictionary with per-variant stats: {stat_key: {'min': (value, variant_name), 'max': (value, variant_name), 'avg': value}}
    """
    per_variant_stats = {}
    
    # Calculate each statistic for each variant
    for variant_name in sorted(df['variant'].unique()):
        variant_data = df[df['variant'] == variant_name][metric_col]
        stats = _calculate_overview_statistics(variant_data)
        per_variant_stats[variant_name] = stats
    
    # Aggregate: for each stat, find min/max/avg across variants
    all_stat_keys = ['mean', 'median', 'std', 'min', 'max', 'range', 'iqr', 'p5', 'p25', 'p75', 'p95', 'skewness', 'kurtosis', 'cv']
    aggregated = {}
    
    for stat_key in all_stat_keys:
        values = [per_variant_stats[v][stat_key] for v in per_variant_stats.keys()]
        variant_names = sorted(per_variant_stats.keys())
        
        min_value = min(values)
        max_value = max(values)
        avg_value = np.mean(values)
        
        min_variant = variant_names[values.index(min_value)]
        max_variant = variant_names[values.index(max_value)]
        
        aggregated[stat_key] = {
            'min': (min_value, min_variant),
            'max': (max_value, max_variant),
            'avg': avg_value,
        }
    
    return aggregated


def _get_variant_numbers(variant_list: list) -> str:
    """
    Convert variant names to simplified numbers for display.
    
    Args:
        variant_list: List of variant names (e.g., ['variant_01', 'variant_03'])
    
    Returns:
        Comma-separated string of variant numbers without leading zeros (e.g., '1, 3')
    """
    numbers = [v.replace('variant_', '') for v in variant_list]
    # Remove leading zeros for cleaner display
    numbers = [str(int(n)) for n in numbers]
    return ", ".join(numbers)


def analyze_survival_race(df_data: pd.DataFrame) -> None:
    """
    Plot cumulative survival for each variant.
    Each line shows how many runs are still alive at each tick.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create unique identifier combining source and variant
    df_data['unique_variant'] = df_data['source'] + '_' + df_data['variant']
    
    # Color mapping by group
    color_map = {
        'successful': PRIMARY_COLOR,
        'unsuccessful': SECONDARY_COLOR,
        'other': '#808080',
        'benchmark': TERTIARY_COLOR
    }
    
    # Z-order mapping (back to front)
    zorder_map = {
        'other': 1,
        'unsuccessful': 2,
        'successful': 3,
        'benchmark': 4
    }
    
    # Plot one line per unique variant
    for unique_var in sorted(df_data['unique_variant'].unique()):
        variant_data = df_data[df_data['unique_variant'] == unique_var]
        survival_times = variant_data['lifetime_ticks'].values
        group = variant_data['group'].iloc[0]
        color = color_map[group]
        zorder = zorder_map[group]
        
        # Create tick array from 0 to max lifetime
        max_tick = int(survival_times.max())
        ticks = np.arange(0, max_tick + 1)
        
        # Count how many runs are still alive at each tick
        alive_counts = np.array([(survival_times >= t).sum() for t in ticks])
        
        # Plot this variant's survival curve with z-order
        ax.plot(ticks, alive_counts, color=color, zorder=zorder)
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=PRIMARY_COLOR, linewidth=2, label='Successful'),
        Line2D([0], [0], color=SECONDARY_COLOR, linewidth=2, label='Unsuccessful'),
        Line2D([0], [0], color='#808080', linewidth=2, label='Other'),
        Line2D([0], [0], color=TERTIARY_COLOR, linewidth=2, label='Benchmark'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_xlabel('Ticks')
    ax.set_ylabel('Runs Alive')
    ax.set_title('Survival Race')
    ax.grid(True, alpha=0.3)
    
    # Save figure
    figures_dir = Path(__file__).resolve().parent / 'figures'
    figures_dir.mkdir(exist_ok=True)
    output_path = figures_dir / 'survival_race.png'
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Add to document
    doc.add_picture(str(output_path), width=6.5 * 914400)
    doc.add_paragraph()


def analyze_wiring(wiring_data: dict, modulation_data: dict, runs_to_show: list, benchmark_name: str = "benchmark") -> None:
    """
    Analyze and visualize wiring data for specified runs.
    
    Generates a 2x2 grid of network plots (initial + 3 runs) with shared legend,
    saves to file, and adds to the report.
    
    Args:
        wiring_data: Dictionary mapping variant_name -> structured array with wiring
        modulation_data: Dictionary mapping variant_name -> structured array with modulation
        runs_to_show: List of run IDs to include in detailed analysis (e.g., [1, 2, 3])
        benchmark_name: Name of the benchmark for file naming
    """
    import tempfile
    
    # Extract wiring data - assume single variant in dict
    if not wiring_data:
        return
    
    variant_name = list(wiring_data.keys())[0]
    wiring_array = wiring_data[variant_name]
    
    # Convert structured array to DataFrame
    df_wiring = pd.DataFrame(wiring_array)
    
    # Load modulation data if available
    df_modulation = None
    if modulation_data and variant_name in modulation_data:
        modulation_array = modulation_data[variant_name]
        df_modulation = pd.DataFrame(modulation_array)
    
    # Load network configuration
    try:
        neuron_positions, neuron_types = network_viz.load_network_viz_config(NETWORK_VIZ_CONFIG)
    except Exception:
        # Fallback if config loading fails
        return
    
    # Define weight columns and labels for the 2x2 grid
    weight_columns = ['weight_initial']
    panel_labels = ['Initial Wiring']
    
    for run_id in runs_to_show[:3]:  # Up to 3 additional runs
        weight_col = f'weight_final_run_{run_id:04d}'
        if weight_col in df_wiring.columns:
            weight_columns.append(weight_col)
            panel_labels.append(f'Run {run_id} - Final')
    
    # Ensure we have exactly 4 panels (initial + up to 3 runs)
    weight_columns = weight_columns[:4]
    panel_labels = panel_labels[:4]
    
    # Create output directory for figures
    figures_dir = Path(__file__).resolve().parent / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    # Determine output filename - use the benchmark name
    output_filename = f'wiring_{benchmark_name.replace(" ", "_")}.png'
    output_path = figures_dir / output_filename
    
    # Temporarily save wiring and modulation data to CSV for network_viz function
    with tempfile.TemporaryDirectory() as tmpdir:
        wiring_csv = Path(tmpdir) / "wiring.csv"
        df_wiring.to_csv(wiring_csv, index=False)
        
        # Save modulation data if available
        modulation_csv = None
        if df_modulation is not None and len(df_modulation) > 0:
            modulation_csv = Path(tmpdir) / "modulation.csv"
            df_modulation.to_csv(modulation_csv, index=False)
            modulation_csv = str(modulation_csv)
        
        # Generate network visualization with 2x2 grid and shared legend
        network_viz.draw_and_combine_networks(
            wiring_csv=str(wiring_csv),
            weight_columns=weight_columns,
            panel_labels=panel_labels,
            output_path=str(output_path),
            modulation_csv=modulation_csv,
            neuron_positions=neuron_positions,
            neuron_types=neuron_types,
            title=f'Network Wiring - {benchmark_name}'
        )
    
    doc.add_picture(str(output_path), width=6.5 * 914400)  # ~6.5 inches in EMUs
    doc.add_paragraph()


# ==================================================================================================================================================
# SECTION C) DATA LOADING
# ==================================================================================================================================================

# Load experiment data
print(f"Loading experiment data from: {EXPERIMENT_HDF5}")
experiment_hdf5_path = _find_hdf5_file(EXPERIMENT_HDF5)
df_experiment = _load_hdf5_variants_to_dataframe(experiment_hdf5_path, source_label="experiment")
print(f"  Loaded {len(df_experiment)} runs across {df_experiment['variant'].nunique()} variants")
df_all = df_experiment.copy()

# Load benchmark data if specified
df_benchmarks = None
if BENCHMARK_HDF5_FILES:
    benchmark_dfs = []
    for benchmark_name, benchmark_hdf5 in BENCHMARK_HDF5_FILES:
        print(f"Loading benchmark '{benchmark_name}' from: {benchmark_hdf5}")
        benchmark_path = _find_hdf5_file(benchmark_hdf5)
        df_bench = _load_hdf5_variants_to_dataframe(benchmark_path, source_label=benchmark_name)
        print(f"  Loaded {len(df_bench)} runs across {df_bench['variant'].nunique()} variants")
        benchmark_dfs.append(df_bench)
    
    df_benchmarks = pd.concat(benchmark_dfs, ignore_index=True)
    df_all = pd.concat([df_experiment, df_benchmarks], ignore_index=True)
    print(f"\nCombined dataset: {len(df_all)} total runs")
else:
    df_all = df_experiment.copy()
    print(f"\nDataset (experiment only): {len(df_all)} total runs")

# Wiring data: {source: {variant: {run: data}}}
wiring_data_experiment = {}
modulation_data_experiment = {}

with h5py.File(experiment_hdf5_path, 'r') as f:
    variant_names = sorted([key for key in f.keys() if key.startswith('variant_')])
    for variant_name in variant_names:
        variant_group = f[variant_name]
        if 'wiring' in variant_group:
            wiring_data_experiment[variant_name] = variant_group['wiring'][:]
        if 'modulation' in variant_group:
            modulation_data_experiment[variant_name] = variant_group['modulation'][:]

# Wiring data for benchmarks
wiring_data_benchmarks = {}
modulation_data_benchmarks = {}

if BENCHMARK_HDF5_FILES:
    for benchmark_name, benchmark_hdf5 in BENCHMARK_HDF5_FILES:
        benchmark_path = _find_hdf5_file(benchmark_hdf5)
        wiring_data_benchmarks[benchmark_name] = {}
        modulation_data_benchmarks[benchmark_name] = {}
        
        with h5py.File(benchmark_path, 'r') as f:
            variant_names = sorted([key for key in f.keys() if key.startswith('variant_')])
            for variant_name in variant_names:
                variant_group = f[variant_name]
                if 'wiring' in variant_group:
                    wiring_data_benchmarks[benchmark_name][variant_name] = variant_group['wiring'][:]
                if 'modulation' in variant_group:
                    modulation_data_benchmarks[benchmark_name][variant_name] = variant_group['modulation'][:]


# ==================================================================================================================================================
# SECTION D) ANALYSIS
# ==================================================================================================================================================

from docx import Document

doc = Document()
doc.add_heading(f"Analysis Report: {EXPERIMENT_NAME}", level=0)

#region 1 Experiment Information

doc.add_heading("1. Experiment Information", level=1)

# Experiment overview
doc.add_paragraph("Experiment:", style="Heading 3")
experiment_n_variants = df_experiment['variant'].nunique()
experiment_total_runs = len(df_experiment)
experiment_runs_per_variant = experiment_total_runs // experiment_n_variants if experiment_n_variants > 0 else 0
doc.add_paragraph(f"Dataset: {EXPERIMENT_HDF5}.h5")
doc.add_paragraph(f"Number of variants: {experiment_n_variants}")
doc.add_paragraph(f"Runs per variant: {experiment_runs_per_variant}")
doc.add_paragraph(f"Total data points: {experiment_total_runs}")



if df_benchmarks is not None and len(df_benchmarks) > 0:
    for benchmark_name, benchmark_hdf5 in BENCHMARK_HDF5_FILES:
        doc.add_paragraph(f"{benchmark_name}:", style="Heading 3")
        benchmark_df = df_benchmarks[df_benchmarks['source'] == benchmark_name]
        benchmark_n_variants = benchmark_df['variant'].nunique()
        benchmark_total_runs = len(benchmark_df)
        benchmark_runs_per_variant = benchmark_total_runs // benchmark_n_variants if benchmark_n_variants > 0 else 0
        doc.add_paragraph(f"Dataset: {benchmark_hdf5}.h5")
        doc.add_paragraph(f"Number of variants: {benchmark_n_variants}")
        doc.add_paragraph(f"Runs per variant: {benchmark_runs_per_variant}")
        doc.add_paragraph(f"Total data points: {benchmark_total_runs}")
else:
    doc.add_paragraph("No benchmark data loaded.")

#region 1.1 Overview Variants

doc.add_heading("1.1. Overview Variants", level=2)

# Calculate overview statistics for experiment variants
df_experiment_data = df_all[df_all['source'] == 'experiment']
stats_experiment = _calculate_overview_statistics(df_experiment_data['lifetime_ticks'])

# Create statistics table for experiment variants
stats_labels_and_units = [
    ("Mean", stats_experiment['mean'], " ticks"),
    ("Median", stats_experiment['median'], " ticks"),
    ("Std Dev", stats_experiment['std'], " ticks"),
    ("Min", stats_experiment['min'], " ticks"),
    ("Max", stats_experiment['max'], " ticks"),
    ("Range", stats_experiment['range'], " ticks"),
    ("IQR (25th-75th percentile)", stats_experiment['iqr'], " ticks"),
    ("5th Percentile", stats_experiment['p5'], " ticks"),
    ("25th Percentile", stats_experiment['p25'], " ticks"),
    ("75th Percentile", stats_experiment['p75'], " ticks"),
    ("95th Percentile", stats_experiment['p95'], " ticks"),
    ("Skewness", stats_experiment['skewness'], ""),
    ("Kurtosis (excess)", stats_experiment['kurtosis'], ""),
    ("Coefficient of Variation", stats_experiment['cv'], "%"),
]

# Build table
table = doc.add_table(rows=len(stats_labels_and_units) + 1, cols=2)
table.style = "Light Grid Accent 1"

# Header row
header_cells = table.rows[0].cells
header_cells[0].text = "Statistic"
header_cells[1].text = "Value"

# Data rows
for row_idx, (label, value, unit) in enumerate(stats_labels_and_units, 1):
    cells = table.rows[row_idx].cells
    cells[0].text = label
    if unit == "%":
        cells[1].text = f"{value:.2f} %"
    elif label in ["Skewness", "Kurtosis (excess)"]:
        cells[1].text = f"{value:.3f}"
    else:
        cells[1].text = f"{value:.2f}{unit}"

# Per-variant statistics table
doc.add_paragraph("Per-Variant Summary (Min, Max, Average across variants):", style="Heading 3")

per_variant_agg = _calculate_per_variant_statistics(df_experiment_data, 'lifetime_ticks')

# Define statistics to display
stats_labels_and_units = [
    ("Mean", 'mean', " ticks"),
    ("Median", 'median', " ticks"),
    ("Std Dev", 'std', " ticks"),
    ("Min", 'min', " ticks"),
    ("Max", 'max', " ticks"),
    ("Range", 'range', " ticks"),
    ("IQR (25th-75th percentile)", 'iqr', " ticks"),
    ("5th Percentile", 'p5', " ticks"),
    ("25th Percentile", 'p25', " ticks"),
    ("75th Percentile", 'p75', " ticks"),
    ("95th Percentile", 'p95', " ticks"),
    ("Skewness", 'skewness', ""),
    ("Kurtosis (excess)", 'kurtosis', ""),
    ("Coefficient of Variation", 'cv', "%"),
]

# Build table with 4 columns: Statistic | Min | Max | Average
table = doc.add_table(rows=len(stats_labels_and_units) + 1, cols=4)
table.style = "Light Grid Accent 1"

# Header row
header_cells = table.rows[0].cells
header_cells[0].text = "Statistic"
header_cells[1].text = "Min (Variant)"
header_cells[2].text = "Max (Variant)"
header_cells[3].text = "Average"

# Data rows
for row_idx, (label, stat_key, unit) in enumerate(stats_labels_and_units, 1):
    cells = table.rows[row_idx].cells
    cells[0].text = label
    
    min_value, min_variant = per_variant_agg[stat_key]['min']
    max_value, max_variant = per_variant_agg[stat_key]['max']
    avg_value = per_variant_agg[stat_key]['avg']
    
    # Format min value
    if unit == "%":
        cells[1].text = f"{min_value:.2f} % ({min_variant})"
    elif label in ["Skewness", "Kurtosis (excess)"]:
        cells[1].text = f"{min_value:.3f} ({min_variant})"
    else:
        cells[1].text = f"{min_value:.2f}{unit} ({min_variant})"
    
    # Format max value
    if unit == "%":
        cells[2].text = f"{max_value:.2f} % ({max_variant})"
    elif label in ["Skewness", "Kurtosis (excess)"]:
        cells[2].text = f"{max_value:.3f} ({max_variant})"
    else:
        cells[2].text = f"{max_value:.2f}{unit} ({max_variant})"
    
    # Format average value
    if unit == "%":
        cells[3].text = f"{avg_value:.2f} %"
    elif label in ["Skewness", "Kurtosis (excess)"]:
        cells[3].text = f"{avg_value:.3f}"
    else:
        cells[3].text = f"{avg_value:.2f}{unit}"

#table with statistics calculated per variant goes here

#endregion # closes 1.1

#region 1.2 Overview Benchmarks

doc.add_heading("1.2. Overview Benchmarks", level=2)

if df_benchmarks is not None and len(df_benchmarks) > 0:
    # Calculate statistics for each benchmark
    benchmark_stats_all = {}
    for benchmark_name, _ in BENCHMARK_HDF5_FILES:
        df_bench_data = df_all[df_all['source'] == benchmark_name]
        if len(df_bench_data) > 0:
            benchmark_stats_all[benchmark_name] = _calculate_overview_statistics(df_bench_data['lifetime_ticks'])
    
    if benchmark_stats_all:
        # Define statistics to display
        stats_labels_and_units = [
            ("Mean", 'mean', " ticks"),
            ("Median", 'median', " ticks"),
            ("Std Dev", 'std', " ticks"),
            ("Min", 'min', " ticks"),
            ("Max", 'max', " ticks"),
            ("Range", 'range', " ticks"),
            ("IQR (25th-75th percentile)", 'iqr', " ticks"),
            ("5th Percentile", 'p5', " ticks"),
            ("25th Percentile", 'p25', " ticks"),
            ("75th Percentile", 'p75', " ticks"),
            ("95th Percentile", 'p95', " ticks"),
            ("Skewness", 'skewness', ""),
            ("Kurtosis (excess)", 'kurtosis', ""),
            ("Coefficient of Variation", 'cv', "%"),
        ]
        
        # Create table with one column per benchmark
        n_benchmarks = len(benchmark_stats_all)
        table = doc.add_table(rows=len(stats_labels_and_units) + 1, cols=n_benchmarks + 1)
        table.style = "Light Grid Accent 1"
        
        # Header row: metric name + benchmark names
        header_cells = table.rows[0].cells
        header_cells[0].text = "Statistic"
        for col_idx, benchmark_name in enumerate(sorted(benchmark_stats_all.keys()), 1):
            header_cells[col_idx].text = benchmark_name
        
        # Data rows
        for row_idx, (label, stat_key, unit) in enumerate(stats_labels_and_units, 1):
            cells = table.rows[row_idx].cells
            cells[0].text = label
            
            for col_idx, benchmark_name in enumerate(sorted(benchmark_stats_all.keys()), 1):
                value = benchmark_stats_all[benchmark_name][stat_key]
                if unit == "%":
                    cells[col_idx].text = f"{value:.2f} %"
                elif label in ["Skewness", "Kurtosis (excess)"]:
                    cells[col_idx].text = f"{value:.3f}"
                else:
                    cells[col_idx].text = f"{value:.2f}{unit}"

    # Analyze wiring for each benchmark
    for benchmark_name, _ in BENCHMARK_HDF5_FILES:
        if benchmark_name in wiring_data_benchmarks:
            modulation_dict = modulation_data_benchmarks.get(benchmark_name, {})
            analyze_wiring(wiring_data_benchmarks[benchmark_name], modulation_dict, RUNS_TO_SHOW_IN_DETAIL, benchmark_name)
                    
else:
    doc.add_paragraph("No benchmark data loaded.")



#endregion # closes 1.2

#endregion # closes 1

#region 2 Group Selection

doc.add_heading("2. Group Selection", level=1)

# Calculate median survival for each random variant
median_by_variant = df_experiment.groupby('variant')['lifetime_ticks'].median().sort_values()

# Get min and max medians
min_median = median_by_variant.min()
max_median = median_by_variant.max()
median_range = max_median - min_median

# Calculate thresholds at 10% and 90% of the range
threshold_10_pct = min_median + (0.10 * median_range)
threshold_90_pct = min_median + (0.90 * median_range)

# Assign groups to experiment variants based on median survival
variant_groups = {}
successful_variants = []
unsuccessful_variants = []
other_variants = []

for variant_name, median_val in median_by_variant.items():
    if median_val < threshold_10_pct:
        variant_groups[variant_name] = 'unsuccessful'
        unsuccessful_variants.append(variant_name)
    elif median_val > threshold_90_pct:
        variant_groups[variant_name] = 'successful'
        successful_variants.append(variant_name)
    else:
        variant_groups[variant_name] = 'other'
        other_variants.append(variant_name)

# Add group assignment to df_all
def assign_group(row):
    if row['source'] == 'experiment':
        return variant_groups.get(row['variant'], 'unknown')
    else:
        return 'benchmark'

df_all['group'] = df_all.apply(assign_group, axis=1)

# Make df_all immutable to prevent accidental modifications
df_all.flags.writeable = False

# Add summary to document
doc.add_paragraph(f"Thresholds based on median survival time (lifetime_ticks):")
doc.add_paragraph(f"  Unsuccessful (bottom 10%): median < {threshold_10_pct:.2f} ticks", style="List Bullet")
doc.add_paragraph(f"  Successful (top 10%): median > {threshold_90_pct:.2f} ticks", style="List Bullet")

# Create table showing variant assignments
table = doc.add_table(rows=1, cols=3)
table.style = "Light Grid Accent 1"

# Header row
header_cells = table.rows[0].cells
header_cells[0].text = "Group"
header_cells[1].text = "Count"
header_cells[2].text = "Variants"

# Add successful variants row
table.add_row()
row = table.rows[1].cells
row[0].text = "Successful (Top 10%)"
row[1].text = str(len(successful_variants))
row[2].text = _get_variant_numbers(successful_variants)

# Add unsuccessful variants row
table.add_row()
row = table.rows[2].cells
row[0].text = "Unsuccessful (Bottom 10%)"
row[1].text = str(len(unsuccessful_variants))
row[2].text = _get_variant_numbers(unsuccessful_variants)

doc.add_paragraph()
analyze_survival_race(df_all)
doc.add_paragraph()

#add a call to the analyze_wiring function for the first variant in the successful and unsuccessful groups

# Get first variant from successful and unsuccessful groups
first_successful = successful_variants[0] if successful_variants else None
first_unsuccessful = unsuccessful_variants[0] if unsuccessful_variants else None

# Analyze wiring for first successful variant
if first_successful:
    wiring_dict_success = {first_successful: wiring_data_experiment[first_successful]}
    modulation_dict_success = {first_successful: modulation_data_experiment.get(first_successful, np.array([]))}
    analyze_wiring(wiring_dict_success, modulation_dict_success, RUNS_TO_SHOW_IN_DETAIL, f"Exemplary successful variant - {first_successful}")

# Analyze wiring for first unsuccessful variant
if first_unsuccessful:
    wiring_dict_unsuccess = {first_unsuccessful: wiring_data_experiment[first_unsuccessful]}
    modulation_dict_unsuccess = {first_unsuccessful: modulation_data_experiment.get(first_unsuccessful, np.array([]))}
    analyze_wiring(wiring_dict_unsuccess, modulation_dict_unsuccess, RUNS_TO_SHOW_IN_DETAIL, f"Exemplary unsuccessful variant - {first_unsuccessful}")

#endregion # closes 2

#region 3 Comparison Successful vs. Unsuccessful (vs. Benchmarks)

doc.add_heading("3. Comparison Successful vs. Unsuccessful (vs. Benchmarks)", level=1)

#region 3.1 Survival

doc.add_heading("3.1. Survival", level=2)


#endregion # closes 3.1

#region 3.2 Food Consumption

doc.add_heading("3.2. Food Consumption", level=2)

doc.add_paragraph("TBD")

#endregion # closes 3.2

#region 3.3 Movement

doc.add_heading("3.3. Movement", level=2)

#region 3.3.1 Movements Made

doc.add_heading("3.3.1. Movements Made", level=3)

doc.add_paragraph("TBD")

#endregion # closes 3.3.1

#region 3.3.2 Ground Covered

doc.add_heading("3.3.2. Ground Covered", level=3)

doc.add_paragraph("TBD")

#endregion # closes 3.3.2

#endregion # closes 3.3

#region 3.4 Decisions

doc.add_heading("3.4. Decisions", level=2)

#region 3.4.1 Decisions Made

doc.add_heading("3.4.1. Decisions Made", level=3)

doc.add_paragraph("TBD")

#endregion # closes 3.4.1

#region 3.4.2 Correct Decisions

doc.add_heading("3.4.2. Correct Decisions", level=3)

doc.add_paragraph("TBD")

#endregion # closes 3.4.2

#endregion # closes 3.4

#endregion # closes 3

#region 4 Summary

doc.add_heading("4. Summary", level=1)

doc.add_paragraph("TBD")

#endregion # closes 4


# ==================================================================================================================================================
# SECTION E) WRAP UP
# ==================================================================================================================================================

# Save the report
report_path = Path(__file__).resolve().parent / f"report_{EXPERIMENT_NAME}.docx"
doc.save(report_path)
print(f"\nReport saved to: {report_path}")
