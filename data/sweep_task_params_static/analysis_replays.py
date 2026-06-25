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
from scipy.stats import skew, kurtosis, shapiro, f_oneway, kruskal, mannwhitneyu, ttest_ind, studentized_range, ks_2samp
from scipy.ndimage import gaussian_filter1d
try:
    from scipy.integrate import trapezoid as trapz
except ImportError:
    from scipy.integrate import trapz
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
    import sys
    sys.path.insert(0, str(_workspace_root))
    from analysis_tools.network_visualization import network_viz

# =====================================================================
# User Configuration and Data Selection
# =====================================================================

# Experiment name
EXPERIMENT_NAME = "inital_235_no_regrow_replay"  # Used for file naming and report titles

# Experiment data: list of tuples (display_name, hdf5_filename_without_extension, color_index)
# The FIRST entry is the primary experiment. All experiments are compared as whole groups.
# Additional entries are treated the same way — each is its own group.
# Leave all but the first commented out if running a single-experiment analysis.
# color_index references EXPERIMENT_COLORS; if the index is out of range a color is auto-generated.
EXPERIMENT_HDF5_FILES = [
    ("EA 20 generations", "2026-06-22_18-07-42_no_regrow_initial_0.235_genomes_all_runs_all", 1),
    #("EA from Soft-Coded", "2026-05-06_09-34-55_ea_from_lookup_soft_genomes_all_runs_all", 1),
    #("EA from Hard-Coded", "2026-05-06_11-31-03_ea_from_lookup_hard_genomes_all_runs_all", 2),
]

# Benchmark data (optional): List of tuples (benchmark_display_name, hdf5_filename_without_extension, color_index)
# Leave as empty list [] if no benchmarks to compare
# color_index references BENCHMARK_COLORS; if the index is out of range a color is auto-generated.
BENCHMARK_HDF5_FILES = [
     ("1st generation (Random)", "2026-06-22_18-07-42_no_regrow_initial_0.235_BM_random", 0),
    # #("Soft-Coded", "2026-05-27_10-00-35_2026-05-05_17-42-25_ea_from_random_BM_lookup_soft", 1),
    # ("Hard-Coded", "2026-05-27_10-00-48_2026-05-05_17-42-25_ea_from_random_BM_lookup_hard", 1),
]

# Network visualization configuration (e.g., '11' for network_viz_11.yaml)
NETWORK_VIZ_CONFIG = "11"

# Color scheme for visualizations
# Experiment colors:
EXPERIMENT_COLORS = [
    "#ddd55f", 
    "#b38219",   
    "#705915",  
]
# Benchmark colors:
BENCHMARK_COLORS = [
    "#aad5ee",   
    "#336f99",   
    "#2b3b5f",  
]


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
        # Default: search in the script's own directory
        search_dir = Path(__file__).resolve().parent
    
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


# Row spec used by both the helper and the docx writer:
# (label, stat_key, format_spec)  where format_spec is one of 'ticks' | 'dimless' | 'pct'
_LIFETIME_STAT_ROWS = [
    ("Mean",                        'mean',     'ticks'),
    ("Median",                      'median',   'ticks'),
    ("Std Dev",                     'std',      'ticks'),
    ("Min",                         'min',      'ticks'),
    ("Max",                         'max',      'ticks'),
    ("Range",                       'range',    'ticks'),
    ("IQR (25th–75th percentile)",  'iqr',      'ticks'),
    ("5th Percentile",              'p5',       'ticks'),
    ("25th Percentile",             'p25',      'ticks'),
    ("75th Percentile",             'p75',      'ticks'),
    ("95th Percentile",             'p95',      'ticks'),
    ("Skewness",                    'skewness', 'dimless'),
    ("Kurtosis (excess)",           'kurtosis', 'dimless'),
    ("Coefficient of Variation",    'cv',       'pct'),
]


def _fmt_stat(value: float, fmt: str) -> str:
    """Format a single statistic value for a Word table cell."""
    if fmt == 'ticks':   return f"{value:.2f} ticks"
    if fmt == 'dimless': return f"{value:.3f}"
    if fmt == 'pct':     return f"{value:.2f} %"
    return str(value)


def _lifetime_stats_by_group(df_summary: pd.DataFrame) -> dict:
    """
    Compute lifetime_ticks summary statistics for every group in df_summary.

    Returns
    -------
    dict  {group_name: {stat_key: float}}
          stat_keys: mean, median, std, min, max, range, iqr,
                     p5, p25, p75, p95, skewness, kurtosis, cv
    """
    result = {}
    for group_name in df_summary['group'].unique():
        vals = df_summary.loc[df_summary['group'] == group_name, 'lifetime_ticks'].to_numpy(dtype=float)
        result[group_name] = {
            'mean':     vals.mean(),
            'median':   np.median(vals),
            'std':      vals.std(),
            'min':      vals.min(),
            'max':      vals.max(),
            'range':    vals.max() - vals.min(),
            'iqr':      np.percentile(vals, 75) - np.percentile(vals, 25),
            'p5':       np.percentile(vals, 5),
            'p25':      np.percentile(vals, 25),
            'p75':      np.percentile(vals, 75),
            'p95':      np.percentile(vals, 95),
            'skewness': float(skew(vals)),
            'kurtosis': float(kurtosis(vals)),
            'cv':       (vals.std() / vals.mean()) * 100,
        }
    return result


def describe_lifetime_statistics(df_summary: pd.DataFrame, doc) -> None:
    """
    Generate and add descriptive statistics table to the document.
    
    Parameters:
    -----------
    df_summary : pd.DataFrame
        Summary data with 'group' and 'lifetime_ticks' columns.
    doc : Word document object
        Document to add the statistics table to.
    """
    _lifetime_stats = _lifetime_stats_by_group(df_summary)
    _groups_ordered = list(df_summary['group'].unique())

    table = doc.add_table(rows=len(_LIFETIME_STAT_ROWS) + 1, cols=len(_groups_ordered) + 1)
    table.style = "Light Grid Accent 1"
    _hdr = table.rows[0].cells
    _hdr[0].text = "Statistic"
    for _ci, _gname in enumerate(_groups_ordered, 1):
        _hdr[_ci].text = _gname
    for _ri, (label, stat_key, fmt) in enumerate(_LIFETIME_STAT_ROWS, 1):
        _row = table.rows[_ri].cells
        _row[0].text = label
        for _ci, _gname in enumerate(_groups_ordered, 1):
            _row[_ci].text = _fmt_stat(_lifetime_stats[_gname][stat_key], fmt)


def _group_overview_stats(df_summary: pd.DataFrame) -> list[dict]:
    """
    Compute per-group overview counts from df_summary.

    Returns a list of dicts (one per group, ordered by appearance in df_summary), each with:
        group, type, hdf5_filename, n_variants, n_runs, runs_per_variant
    """
    # Build filename lookup: group_name -> hdf5 filename string (from user config)
    _filename_lookup = {name: hdf5 for name, hdf5, *_ in EXPERIMENT_HDF5_FILES + BENCHMARK_HDF5_FILES}

    rows = []
    for group_name in df_summary['group'].unique():
        gdf        = df_summary[df_summary['group'] == group_name]
        n_variants = gdf['variant'].nunique()
        n_runs     = len(gdf)
        runs_pv    = n_runs // n_variants if n_variants > 0 else 0
        rows.append({
            'group':           group_name,
            'type':            gdf['type'].iloc[0],
            'hdf5_filename':   _filename_lookup.get(group_name, ''),
            'n_variants':      n_variants,
            'n_runs':          n_runs,
            'runs_per_variant': runs_pv,
        })
    return rows


def classify_effect_size(effect_size_value: float, effect_size_type: str) -> str:
    """
    Classify effect size as insignificant, small, medium, or large based on type.
    Supports: "Cohen's d", "Rank-Biserial r", "Eta-squared", "Epsilon-squared".
    """
    es_abs = abs(effect_size_value)

    if effect_size_type == "Cohen's d":
        if es_abs < 0.2:
            return "insignificant"
        elif es_abs < 0.5:
            return "small"
        elif es_abs < 0.8:
            return "medium"
        else:
            return "large"
    elif effect_size_type == "Rank-Biserial r":
        if es_abs < 0.11:
            return "insignificant"
        elif es_abs < 0.28:
            return "small"
        elif es_abs < 0.43:
            return "medium"
        else:
            return "large"
    elif effect_size_type == "Eta-squared":
        if es_abs < 0.01:
            return "insignificant"
        elif es_abs < 0.06:
            return "small"
        elif es_abs < 0.14:
            return "medium"
        else:
            return "large"
    elif effect_size_type == "Epsilon-squared":
        if es_abs < 0.01:
            return "insignificant"
        elif es_abs < 0.08:
            return "small"
        elif es_abs < 0.26:
            return "medium"
        else:
            return "large"
    else:
        return "unknown"


def _get_ordered_groups_with_zorder(groups_input) -> tuple:
    """
    Order groups so benchmarks come first (in their BENCHMARK_HDF5_FILES order),
    then experiments (in their EXPERIMENT_HDF5_FILES order).
    
    Returns a tuple: (ordered_groups_list, zorder_map_dict)
    where zorder_map maps group_name -> z-order value (benchmarks low, experiments high).
    
    Args:
        groups_input: List or array-like of group names
    
    Returns:
        Tuple of (ordered_groups, zorder_dict)
    """
    # Convert to list if needed
    if hasattr(groups_input, 'unique'):
        groups_list = list(groups_input.unique())
    else:
        groups_list = list(groups_input)
    
    # Get benchmark and experiment names
    benchmark_names = [name for name, *_ in BENCHMARK_HDF5_FILES]
    experiment_names = [name for name, *_ in EXPERIMENT_HDF5_FILES]
    
    # Separate groups into benchmarks and experiments
    benchmarks_ordered = []
    experiments_ordered = []
    other_groups = []
    
    for group in groups_list:
        if group in benchmark_names:
            benchmarks_ordered.append(group)
        elif group in experiment_names:
            experiments_ordered.append(group)
        else:
            other_groups.append(group)
    
    # Sort within each category by their original list order
    benchmarks_ordered.sort(key=lambda g: benchmark_names.index(g))
    experiments_ordered.sort(key=lambda g: experiment_names.index(g))
    
    # Combine: benchmarks first, then experiments, then any others
    ordered_groups = benchmarks_ordered + experiments_ordered + other_groups
    
    # Create z-order map: benchmarks have low z-order (drawn first), experiments high (drawn last)
    zorder_map = {}
    for i, group in enumerate(benchmarks_ordered):
        zorder_map[group] = 1 + i
    for i, group in enumerate(experiments_ordered):
        zorder_map[group] = 100 + i
    for i, group in enumerate(other_groups):
        zorder_map[group] = 200 + i
    
    return ordered_groups, zorder_map


def analyze_per_run(df_data: pd.DataFrame, metric_col: str, y_label: str, filename_str: str, group_color_map: dict = None) -> None:
    """
    Box plot with jitter overlay per group, summary stats table, and comparative stats table.
    Groups: unsuccessful, successful, plus each benchmark/additional-experiment source as its own group.
    group_color_map: dict mapping original (lowercase) group names to hex color strings.
    """
    # Dynamically discover and extract all groups from the data
    # Order: benchmarks first, then experiments (so experiments are drawn on top)
    all_unique_groups, zorder_map = _get_ordered_groups_with_zorder(df_data['group'].unique())
    
    group_data = {}
    group_order = []
    
    for group_name in all_unique_groups:
        df_group = df_data[df_data['group'] == group_name]
        if len(df_group) > 0:
            group_data[group_name] = df_group[metric_col].values
            group_order.append(group_name)
    
    # --- Plot: box plot with jitter ---
    _gcm = group_color_map or {}

    def _color_for(group_name: str) -> str:
        """Resolve a color for a group using the provided color map."""
        return _gcm.get(group_name, '#808080')
    fig, ax = plt.subplots(figsize=(10, 7))
    positions = list(range(len(group_order)))
    
    # Jitter overlay (draw first so it appears behind boxplots)
    for i, g in enumerate(group_order):
        values = group_data[g]
        color = _color_for(g)
        z = zorder_map.get(g, 50)
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(values))
        ax.scatter(np.full(len(values), i) + jitter, values, color=color, alpha=0.4, s=8, zorder=z)
    
    # Box plots
    bp = ax.boxplot(
        [group_data[g] for g in group_order],
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
    )
    for i, g in enumerate(group_order):
        color = _color_for(g)
        z = zorder_map.get(g, 50)
        bp['boxes'][i].set_facecolor(color)
        bp['boxes'][i].set_alpha(0.3)
        bp['boxes'][i].set_zorder(z)
        bp['medians'][i].set_color('black')
        bp['medians'][i].set_zorder(z)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(group_order, fontsize=11)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(f'{y_label} by Group', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    figures_dir = Path(__file__).resolve().parent / f'figures_{EXPERIMENT_NAME}'
    figures_dir.mkdir(exist_ok=True)
    output_path = figures_dir / f'groups_{filename_str}.png'
    fig.tight_layout()
    _ymin, _ymax = ax.get_ylim()
    ax.set_ylim(_ymin, _ymax + (_ymax - _ymin) / 3)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    
    doc.add_picture(str(output_path), width=6.5 * 914400)
    doc.add_paragraph()
    
    # --- Summary statistics table ---
    from scipy.stats import sem as scipy_sem
    
    stat_names = ['N', 'Mean', 'Std Dev', 'SEM', 'Median', 'Min', 'Max', 'IQR', 'P25', 'P75', 'Skewness', 'Kurtosis']
    stats_per_group = {}
    for g in group_order:
        v = group_data[g]
        stats_per_group[g] = {
            'N': len(v),
            'Mean': np.mean(v),
            'Std Dev': np.std(v, ddof=1),
            'SEM': scipy_sem(v),
            'Median': np.median(v),
            'Min': np.min(v),
            'Max': np.max(v),
            'IQR': np.percentile(v, 75) - np.percentile(v, 25),
            'P25': np.percentile(v, 25),
            'P75': np.percentile(v, 75),
            'Skewness': skew(v),
            'Kurtosis': kurtosis(v),
        }
    
    table = doc.add_table(rows=len(stat_names) + 1, cols=len(group_order) + 1)
    table.style = 'Light Grid Accent 1'
    table.rows[0].cells[0].text = 'Statistic'
    for col_idx, g in enumerate(group_order, 1):
        table.rows[0].cells[col_idx].text = g
    for row_idx, stat_name in enumerate(stat_names, 1):
        table.rows[row_idx].cells[0].text = stat_name
        for col_idx, g in enumerate(group_order, 1):
            val = stats_per_group[g][stat_name]
            if stat_name == 'N':
                table.rows[row_idx].cells[col_idx].text = str(int(val))
            elif stat_name in ['Skewness', 'Kurtosis']:
                table.rows[row_idx].cells[col_idx].text = f'{val:.3f}'
            else:
                table.rows[row_idx].cells[col_idx].text = f'{val:.2f}'
    
    doc.add_paragraph()

    if len(group_order) < 2:
        doc.add_paragraph('Only one group — no comparative statistics.')
        doc.add_paragraph()
        return

    # --- Comparative statistics ---
    # Test normality per group (Shapiro-Wilk)
    normality_results = {}
    all_normal = True
    for g in group_order:
        v = group_data[g]
        if len(v) >= 3:
            stat_val, p_val = shapiro(v)
            normality_results[g] = (stat_val, p_val)
            if p_val < 0.05:
                all_normal = False
        else:
            normality_results[g] = (np.nan, np.nan)
            all_normal = False
    
    # Build normality results table
    doc.add_paragraph('Normality Testing (Shapiro-Wilk):', style='Heading 3')
    norm_table = doc.add_table(rows=len(group_order) + 1, cols=3)
    norm_table.style = 'Light Grid Accent 1'
    norm_table.rows[0].cells[0].text = 'Group'
    norm_table.rows[0].cells[1].text = 'W statistic'
    norm_table.rows[0].cells[2].text = 'p-value'
    for row_idx, g in enumerate(group_order, 1):
        norm_table.rows[row_idx].cells[0].text = g
        w, p = normality_results[g]
        norm_table.rows[row_idx].cells[1].text = f'{w:.4f}' if not np.isnan(w) else 'N/A'
        norm_table.rows[row_idx].cells[2].text = f'{p:.2e}' if not np.isnan(p) else 'N/A'
        
    # Determine which tests to use based on group count and normality
    group_arrays = [group_data[g] for g in group_order]
    n_groups = len(group_order)
    
    if n_groups == 2:
        # Two groups: t-test (normal) or Mann-Whitney U (non-normal)
        if all_normal:
            omnibus_name = "Welch's t-test"
            omnibus_stat, omnibus_p = ttest_ind(group_arrays[0], group_arrays[1], equal_var=False)
            
            # Cohen's d effect size
            mean_diff = np.mean(group_arrays[0]) - np.mean(group_arrays[1])
            n1, n2 = len(group_arrays[0]), len(group_arrays[1])
            var1, var2 = np.var(group_arrays[0], ddof=1), np.var(group_arrays[1], ddof=1)
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            effect_size = mean_diff / pooled_std if pooled_std > 0 else 0
            effect_size_label = "Cohen's d"
        else:
            omnibus_name = 'Mann-Whitney U'
            omnibus_stat, omnibus_p = mannwhitneyu(group_arrays[0], group_arrays[1], alternative='two-sided')
            
            # Rank-biserial correlation as effect size
            n1, n2 = len(group_arrays[0]), len(group_arrays[1])
            effect_size = 1 - (2 * omnibus_stat) / (n1 * n2)
            effect_size_label = 'Rank-Biserial r'
    else:
        # More than two groups: ANOVA (normal) or Kruskal-Wallis (non-normal)
        if all_normal:
            omnibus_name = 'One-way ANOVA'
            omnibus_stat, omnibus_p = f_oneway(*group_arrays)
            
            # Eta-squared effect size
            grand_mean = np.mean(np.concatenate(group_arrays))
            ss_between = sum(len(arr) * (np.mean(arr) - grand_mean) ** 2 for arr in group_arrays)
            ss_total = sum(np.sum((arr - grand_mean) ** 2) for arr in group_arrays)
            effect_size = ss_between / ss_total if ss_total > 0 else 0
            effect_size_label = "Eta-squared"
        else:
            omnibus_name = 'Kruskal-Wallis'
            omnibus_stat, omnibus_p = kruskal(*group_arrays)
            
            # Epsilon-squared effect size
            N = sum(len(arr) for arr in group_arrays)
            k = len(group_arrays)
            effect_size = (omnibus_stat - k + 1) / (N - k) if (N - k) > 0 else 0
            effect_size_label = "Epsilon-squared"
    
    distribution_str = 'normally distributed' if all_normal else 'not normally distributed'
    doc.add_paragraph(f'Data is {distribution_str} (alpha=0.05).')
    
    doc.add_paragraph(f'{omnibus_name} (omnibus test):', style='Heading 3')
    omnibus_table = doc.add_table(rows=2, cols=4)
    omnibus_table.style = 'Light Grid Accent 1'
    omnibus_table.rows[0].cells[0].text = 'Test Name'
    omnibus_table.rows[0].cells[1].text = 'Test Statistic'
    omnibus_table.rows[0].cells[2].text = 'p-value'
    omnibus_table.rows[0].cells[3].text = effect_size_label
    omnibus_table.rows[1].cells[0].text = omnibus_name
    omnibus_table.rows[1].cells[1].text = f'{omnibus_stat:.4f}'
    omnibus_table.rows[1].cells[2].text = f'{omnibus_p:.2e}'
    omnibus_table.rows[1].cells[3].text = f'{effect_size:.4f} ({classify_effect_size(effect_size, effect_size_label)})'
    doc.add_paragraph()
    
    # Post-hoc testing only if omnibus test is significant
    if omnibus_p < 0.05:
        # Determine post-hoc test
        if n_groups == 2:
            # For 2 groups, omnibus test IS the pairwise test
            posthoc_name = omnibus_name
            pairwise_results = [(group_order[0], group_order[1], omnibus_stat, omnibus_p, effect_size)]
        else:
            # For >2 groups, do pairwise comparisons
            if all_normal:
                posthoc_name = "Welch's t-test"
                posthoc_func = lambda a, b: ttest_ind(a, b, equal_var=False)
                
                # Function to compute Cohen's d for pairwise
                def compute_cohens_d(a, b):
                    mean_diff = np.mean(a) - np.mean(b)
                    n1, n2 = len(a), len(b)
                    var1, var2 = np.var(a, ddof=1), np.var(b, ddof=1)
                    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
                    return mean_diff / pooled_std if pooled_std > 0 else 0
                
                def effect_size_func(a, b, stat):
                    return compute_cohens_d(a, b)
                
                effect_size_label_posthoc = "Cohen's d"
            else:
                posthoc_name = 'Mann-Whitney U'
                posthoc_func = lambda a, b: mannwhitneyu(a, b, alternative='two-sided')
                
                # Function to compute rank-biserial for pairwise
                def effect_size_func(a, b, stat):
                    n1, n2 = len(a), len(b)
                    return 1 - (2 * stat) / (n1 * n2)
                
                effect_size_label_posthoc = 'Rank-Biserial r'
            
            pairwise_results = []
            for g1, g2 in combinations(group_order, 2):
                stat_val, p_val = posthoc_func(group_data[g1], group_data[g2])
                es = effect_size_func(group_data[g1], group_data[g2], stat_val)
                pairwise_results.append((g1, g2, stat_val, p_val, es))
            
            # Apply appropriate multiple comparisons correction for >2 groups
            if n_groups > 2:
                if all_normal:
                    # For parametric tests: Use Tukey's HSD
                    k = n_groups
                    N = sum(len(group_data[g]) for g in group_order)
                    df_error = N - k
                    
                    corrected_results = []
                    for (g1, g2, stat_val, p_val, es) in pairwise_results:
                        # Apply Tukey's HSD correction using studentized range distribution
                        corrected_p = studentized_range.sf(abs(stat_val), k, df_error)
                        corrected_p = min(corrected_p, 1.0)
                        corrected_results.append((g1, g2, stat_val, corrected_p, es))
                    
                    pairwise_results = corrected_results
                    posthoc_name = f"{posthoc_name} (Tukey's HSD correction)"
                else:
                    # For non-parametric tests: Use Holm-Bonferroni correction
                    p_values = [result[3] for result in pairwise_results]
                    reject, corrected_p_values, _, _ = multipletests(p_values, alpha=0.05, method='holm')
                    
                    corrected_results = []
                    for idx, (g1, g2, stat_val, p_val, es) in enumerate(pairwise_results):
                        corrected_results.append((g1, g2, stat_val, corrected_p_values[idx], es))
                    
                    pairwise_results = corrected_results
                    posthoc_name = f"{posthoc_name} (Holm-Bonferroni correction)"
        
        doc.add_paragraph(f'Pairwise Post-Hoc ({posthoc_name}):', style='Heading 3')
        posthoc_table = doc.add_table(rows=len(pairwise_results) + 1, cols=5)
        posthoc_table.style = 'Light Grid Accent 1'
        posthoc_table.rows[0].cells[0].text = 'Group 1'
        posthoc_table.rows[0].cells[1].text = 'Group 2'
        posthoc_table.rows[0].cells[2].text = 'Statistic'
        posthoc_table.rows[0].cells[3].text = 'p-value'
        posthoc_table.rows[0].cells[4].text = effect_size_label_posthoc if n_groups > 2 else effect_size_label
        for row_idx, result in enumerate(pairwise_results, 1):
            g1, g2, stat_val, p_val, es = result
            posthoc_table.rows[row_idx].cells[0].text = g1
            posthoc_table.rows[row_idx].cells[1].text = g2
            posthoc_table.rows[row_idx].cells[2].text = f'{stat_val:.4f}'
            posthoc_table.rows[row_idx].cells[3].text = f'{p_val:.2e}'
            es_label = effect_size_label_posthoc if n_groups > 2 else effect_size_label
            posthoc_table.rows[row_idx].cells[4].text = f'{es:.4f} ({classify_effect_size(es, es_label)})'
        doc.add_paragraph()
    else:
        doc.add_paragraph(f'Omnibus test not significant (p >= 0.05). No post-hoc testing performed.')
        doc.add_paragraph()


def plot_wiring(wiring_data: dict, modulation_data: dict) -> None:
    """
    Visualize wiring data for all groups, with two passes: effective weight and raw initial weight.
    
    Expects wiring_data arrays to already contain a 'weight_effective' column
    (computed during data loading as weight_initial × reliability).
    Creates a 2x2 grid of network plots per group, sampling up to 4 variants.
    
    Args:
        wiring_data: Dict mapping group_name -> {variant_id -> structured array}
        modulation_data: Dict mapping group_name -> {variant_id -> structured array}
    """
    if not wiring_data:
        return
    
    net_positions, net_neuron_types = network_viz.load_network_viz_config(NETWORK_VIZ_CONFIG)
    
    net_figures_dir = Path(__file__).resolve().parent / f'figures_{EXPERIMENT_NAME}'
    net_figures_dir.mkdir(exist_ok=True)
    
    for _weight_col, _pass_label, _pass_tag in [
        ('weight_effective', 'Effective Connection Strength (weight_initial × reliability)', 'effective'),
        ('weight_initial',   'Raw Initial Weights (weight_initial)',                         'raw'),
    ]:
        doc.add_heading(_pass_label, level=3)
        
        for _grp_name, _grp_variants in wiring_data.items():
            # Sample up to 4 variants evenly distributed across the group
            _all_vids = sorted(_grp_variants.keys())
            _n_all = len(_all_vids)
            _n_pick = min(4, _n_all)
            _indices = [int(np.floor(i)) for i in np.linspace(0, _n_all - 1, _n_pick)]
            _vids = [_all_vids[i] for i in _indices]
            
            # Create 2x2 subplot figure
            fig, axes = plt.subplots(2, 2, figsize=(13, 10))
            fig.subplots_adjust(hspace=0.05, wspace=0.1)
            axes = axes.flatten()
            
            # Draw each sampled variant
            for _i, _vid in enumerate(_vids):
                _w = pd.DataFrame(_grp_variants[_vid])
                _w['weight'] = _w[_weight_col]
                _draw_wiring_panel(
                    axes[_i], _w, modulation_data[_grp_name][_vid],
                    net_positions, net_neuron_types,
                    title=f'Variant {_vid}',
                )
            
            # Hide unused subplots
            for _i in range(len(_vids), 4):
                axes[_i].axis('off')
            
            # Add shared legend at figure bottom
            _legend_elements = [
                mpatches.Patch(facecolor=network_viz.COLORS['neuron_fill'], edgecolor=network_viz.COLORS['input_edge'],  linewidth=2, label='Input'),
                mpatches.Patch(facecolor=network_viz.COLORS['neuron_fill'], edgecolor=network_viz.COLORS['output_edge'], linewidth=2, label='Output'),
                mpatches.Patch(facecolor=network_viz.COLORS['neuron_fill'], edgecolor=network_viz.COLORS['hidden_edge'], linewidth=2, label='Hidden'),
                plt.Line2D([0], [0], color=network_viz.COLORS['excitatory'], linewidth=2, linestyle='-', label='Excitatory'),
                plt.Line2D([0], [0], color=network_viz.COLORS['inhibitory'], linewidth=2, linestyle='-', label='Inhibitory'),
            ]
            fig.legend(handles=_legend_elements, loc='lower center', ncol=5, fontsize=9,
                       frameon=True, bbox_to_anchor=(0.5, 0.01))
            fig.subplots_adjust(bottom=0.07)
            
            # Save figure and add to document
            _fig_path = net_figures_dir / f'network_{_pass_tag}_{_grp_name.replace(" ", "_")}.png'
            fig.savefig(str(_fig_path), dpi=150, bbox_inches='tight')
            plt.close(fig)
            doc.add_picture(str(_fig_path), width=6.5 * 914400)
            doc.add_paragraph()


def analyze_distribution_across_variants(wiring_data: dict, metric_col: str, x_label: str,
                                         n_bins: int, group_color_map: dict = None) -> None:
    """
    Plot normalized frequency distributions of a metric pooled across all variants per group,
    and run pairwise KS tests with Holm-Bonferroni correction between all group pairs.

    The KS statistic D is used as the effect size (max absolute CDF difference, 0-1):
      small ≥ 0.1, medium ≥ 0.3, large ≥ 0.5

    Args:
        wiring_data:     {group_name: {variant_id: structured array}} — arrays must contain metric_col.
        metric_col:      Column name to extract from each variant's array. Also used for filenames.
        x_label:         X-axis label.
        n_bins:          Number of histogram bins.
        group_color_map: Dict mapping group name -> hex color string.
    """
    all_groups, zorder_map = _get_ordered_groups_with_zorder(list(wiring_data.keys()))
    color_map  = group_color_map or {}

    # --- Pool all values per group across variants ---
    group_values = {}
    for grp, variants in wiring_data.items():
        group_values[grp] = np.concatenate(
            [pd.DataFrame(arr)[metric_col].values for arr in variants.values()]
        )

    # --- Shared bin edges so all groups are comparable ---
    all_vals    = np.concatenate(list(group_values.values()))
    bin_edges   = np.linspace(all_vals.min(), all_vals.max(), n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # --- Plot ---
    figures_dir = Path(__file__).resolve().parent / f'figures_{EXPERIMENT_NAME}'
    figures_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for grp in all_groups:
        counts, _ = np.histogram(group_values[grp], bins=bin_edges)
        freq      = counts / counts.sum()                    # normalize to relative frequency
        smoothed  = gaussian_filter1d(freq, sigma=1.0)       # slight smoothing
        ax.plot(bin_centers, smoothed, linewidth=2, label=grp,
                color=color_map.get(grp, None), zorder=zorder_map.get(grp, 50))

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('Relative Frequency', fontsize=12)
    ax.set_title(f'Distribution of {x_label} across Variants', fontsize=14)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    fig_path = figures_dir / f'distribution_{metric_col}.png'
    fig.tight_layout()
    fig.savefig(str(fig_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    doc.add_picture(str(fig_path), width=6.5 * 914400)
    doc.add_paragraph()

    # --- Pairwise KS tests with Holm-Bonferroni correction ---
    if len(all_groups) < 2:
        return

    def _ks_effect_label(d: float) -> str:
        if d >= 0.5:  return 'large'
        if d >= 0.3:  return 'medium'
        if d >= 0.1:  return 'small'
        return 'negligible'

    _pairs      = list(combinations(all_groups, 2))
    _ks_results = []
    for g0, g1 in _pairs:
        stat, p = ks_2samp(group_values[g0], group_values[g1])
        _ks_results.append({'group_a': g0, 'group_b': g1, 'ks': stat, 'p_raw': p})

    _raw_ps                  = [r['p_raw'] for r in _ks_results]
    _rejected, _p_corr, _, _ = multipletests(_raw_ps, method='holm')
    for r, p_c, rej in zip(_ks_results, _p_corr, _rejected):
        r['p_corrected'] = p_c
        r['rejected']    = rej

    doc.add_heading("Distribution Comparison (Pairwise KS tests, Holm-Bonferroni corrected)", level=3)
    _ks_tbl = doc.add_table(rows=len(_ks_results) + 1, cols=6)
    _ks_tbl.style = "Light Grid Accent 1"
    _ks_hdr = _ks_tbl.rows[0].cells
    _ks_hdr[0].text = "Group A"
    _ks_hdr[1].text = "Group B"
    _ks_hdr[2].text = "D (effect size)"
    _ks_hdr[3].text = "Effect magnitude"
    _ks_hdr[4].text = "p (raw)"
    _ks_hdr[5].text = "p (corrected)"
    for _ri, r in enumerate(_ks_results, 1):
        _ks_row = _ks_tbl.rows[_ri].cells
        _ks_row[0].text = r['group_a']
        _ks_row[1].text = r['group_b']
        _ks_row[2].text = f"{r['ks']:.4f}"
        _ks_row[3].text = _ks_effect_label(r['ks'])
        _ks_row[4].text = _fmt_pvalue(r['p_raw'])
        _ks_row[5].text = _fmt_pvalue(r['p_corrected']) + (" *" if r['rejected'] else "")
    doc.add_paragraph()

    # --- Per-variant distributions (all variants from all groups in one figure, no pooling, no statistical tests) ---
    doc.add_heading("Per-Variant Distributions", level=3)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for grp in all_groups:
        variants = wiring_data[grp]
        grp_color = color_map.get(grp, None)
        
        # Plot each variant as a separate line using global bin edges
        for var_id, arr in sorted(variants.items()):
            var_values = pd.DataFrame(arr)[metric_col].values
            counts, _ = np.histogram(var_values, bins=bin_edges)
            freq = counts / counts.sum()
            smoothed = gaussian_filter1d(freq, sigma=1.0)
            ax.plot(bin_centers, smoothed, linewidth=1, alpha=0.5, color=grp_color,
                    zorder=zorder_map.get(grp, 50))
    
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('Relative Frequency', fontsize=12)
    ax.set_title(f'Per-Variant Distribution of {x_label}', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend for group colors
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=color_map.get(grp, '#808080'), linewidth=2, label=grp)
                       for grp in all_groups]
    ax.legend(handles=legend_elements, loc='best', fontsize=10)
    
    fig_path = figures_dir / f'distribution_variants_{metric_col}.png'
    fig.tight_layout()
    fig.savefig(str(fig_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    doc.add_picture(str(fig_path), width=6.5 * 914400)
    doc.add_paragraph()


def analyze_per_variant(variant_data: pd.DataFrame, metric_col: str, y_label: str,
                        group_color_map: dict = None) -> None:
    """
    Analyze variant-level metrics with jitter plot and optional boxplot overlay.
    
    For each group, plots individual variant values as jittered points.
    If a group has more than 3 variants, overlays a boxplot.
    Performs adaptive statistical tests comparing groups.
    
    Args:
        variant_data: DataFrame with columns: group, type, variant, and metric_col
        metric_col: Column name to analyze (e.g., 'eta')
        y_label: Label for y-axis
        group_color_map: Dict mapping group name -> hex color string
    """
    if variant_data is None or len(variant_data) == 0:
        doc.add_paragraph(f"{y_label} data not available")
        return
    
    if metric_col not in variant_data.columns:
        doc.add_paragraph(f"{y_label} ({metric_col}) not found in variant data")
        return
    
    all_groups, zorder_map = _get_ordered_groups_with_zorder(variant_data['group'].unique())
    color_map = group_color_map or {}
    
    # --- Create jitter + boxplot figure ---
    figures_dir = Path(__file__).resolve().parent / f'figures_{EXPERIMENT_NAME}'
    figures_dir.mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Position each group on x-axis
    group_positions = {grp: i for i, grp in enumerate(all_groups)}
    
    for grp in all_groups:
        grp_data = variant_data[variant_data['group'] == grp][metric_col].values
        x_pos = group_positions[grp]
        color = color_map.get(grp, '#808080')
        z = zorder_map.get(grp, 50)
        
        # Jitter points
        x_jitter = np.random.normal(x_pos, 0.04, size=len(grp_data))
        ax.scatter(x_jitter, grp_data, alpha=0.6, s=80, color=color, label=grp, zorder=z)
        
        # Boxplot if more than 3 variants
        if len(grp_data) > 3:
            bp = ax.boxplot([grp_data], positions=[x_pos], widths=0.2,
                           patch_artist=True, showfliers=False, zorder=z-1)
            for patch in bp['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.3)
            for whisker in bp['whiskers']:
                whisker.set(color=color, linewidth=1.5)
            for median in bp['medians']:
                median.set(color=color, linewidth=2)
    
    ax.set_xticks(list(group_positions.values()))
    ax.set_xticklabels(all_groups, fontsize=11)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(f'{y_label} per Variant', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='best', fontsize=10)
    
    fig_path = figures_dir / f'per_variant_{metric_col}.png'
    fig.tight_layout()
    _ymin, _ymax = ax.get_ylim()
    ax.set_ylim(_ymin, _ymax + (_ymax - _ymin) / 3)
    fig.savefig(str(fig_path), dpi=150)
    plt.close(fig)
    
    doc.add_picture(str(fig_path), width=6.5 * 914400)
    doc.add_paragraph()
    
    # --- Statistics: adaptive tests across groups ---
    if len(all_groups) < 2:
        return
    
    doc.add_heading(f"Statistical Comparison of {y_label}", level=3)
    
    group_values = {grp: variant_data[variant_data['group'] == grp][metric_col].values
                    for grp in all_groups}
    
    # Shapiro-Wilk normality test per group
    normality_results = {}
    for grp in all_groups:
        vals = group_values[grp]
        if len(vals) < 3:
            normality_results[grp] = {'stat': np.nan, 'p': np.nan, 'normal': False}
        else:
            stat, p = shapiro(vals)
            normality_results[grp] = {'stat': stat, 'p': p, 'normal': p >= 0.05}
    
    all_normal = all(normality_results[grp]['normal'] for grp in all_groups)
    
    # Omnibus test
    if len(all_groups) == 2:
        # Two groups: Welch's t-test or Mann-Whitney U
        g0, g1 = all_groups
        if all_normal:
            stat, p_omnibus = ttest_ind(group_values[g0], group_values[g1], equal_var=False)
            test_name = "Welch's t-test"
            effect_col = "Cohen's d"
            # Cohen's d
            n0, n1 = len(group_values[g0]), len(group_values[g1])
            pooled_std = np.sqrt(((n0-1)*np.std(group_values[g0], ddof=1)**2 + 
                                 (n1-1)*np.std(group_values[g1], ddof=1)**2) / (n0+n1-2))
            effect = (np.mean(group_values[g0]) - np.mean(group_values[g1])) / pooled_std if pooled_std > 0 else 0
        else:
            stat, p_omnibus = mannwhitneyu(group_values[g0], group_values[g1])
            test_name = "Mann-Whitney U"
            effect_col = "Rank-Biserial r"
            n = len(group_values[g0]) + len(group_values[g1])
            effect = 1 - (2 * stat) / (len(group_values[g0]) * len(group_values[g1]))
        
        doc.add_paragraph(f"{test_name}: p = {_fmt_pvalue(p_omnibus)}, {effect_col} = {effect:.4f}")
    else:
        # More than two groups: ANOVA or Kruskal-Wallis
        if all_normal:
            stat, p_omnibus = f_oneway(*[group_values[grp] for grp in all_groups])
            test_name = "One-way ANOVA"
            effect_col = "Eta-squared η²"
            # Eta-squared
            grand_mean = np.concatenate([group_values[grp] for grp in all_groups]).mean()
            ss_between = sum(len(group_values[grp]) * (np.mean(group_values[grp]) - grand_mean)**2 
                           for grp in all_groups)
            ss_total = sum(np.sum((group_values[grp] - grand_mean)**2) for grp in all_groups)
            effect = ss_between / ss_total if ss_total > 0 else 0
        else:
            stat, p_omnibus = kruskal(*[group_values[grp] for grp in all_groups])
            test_name = "Kruskal-Wallis"
            effect_col = "Ordinal Epsilon-squared ε²R"
            n = sum(len(group_values[grp]) for grp in all_groups)
            # Ordinal epsilon-squared (simplified)
            effect = (stat - len(all_groups) + 1) / (n - len(all_groups))
        
        doc.add_paragraph(f"{test_name}: p = {_fmt_pvalue(p_omnibus)}, {effect_col} = {effect:.4f}")
        
        # Pairwise post-hoc tests if omnibus significant
        if p_omnibus < 0.05:
            doc.add_heading("Pairwise Comparisons (Holm-Bonferroni corrected)", level=4)
            
            _pairs = list(combinations(all_groups, 2))
            _results = []
            for g0, g1 in _pairs:
                if all_normal:
                    stat, p = ttest_ind(group_values[g0], group_values[g1], equal_var=False)
                    n0, n1 = len(group_values[g0]), len(group_values[g1])
                    pooled_std = np.sqrt(((n0-1)*np.std(group_values[g0], ddof=1)**2 + 
                                         (n1-1)*np.std(group_values[g1], ddof=1)**2) / (n0+n1-2))
                    effect = (np.mean(group_values[g0]) - np.mean(group_values[g1])) / pooled_std if pooled_std > 0 else 0
                    effect_name = "Cohen's d"
                else:
                    stat, p = mannwhitneyu(group_values[g0], group_values[g1])
                    n = len(group_values[g0]) + len(group_values[g1])
                    effect = 1 - (2 * stat) / (len(group_values[g0]) * len(group_values[g1]))
                    effect_name = "Rank-Biserial r"
                
                _results.append({'group_a': g0, 'group_b': g1, 'p_raw': p, 'effect': effect})
            
            _raw_ps = [r['p_raw'] for r in _results]
            _rejected, _p_corr, _, _ = multipletests(_raw_ps, method='holm')
            for r, p_c, rej in zip(_results, _p_corr, _rejected):
                r['p_corrected'] = p_c
                r['rejected'] = rej
            
            _tbl = doc.add_table(rows=len(_results) + 1, cols=5)
            _tbl.style = "Light Grid Accent 1"
            _hdr = _tbl.rows[0].cells
            _hdr[0].text = "Group A"
            _hdr[1].text = "Group B"
            _hdr[2].text = effect_name
            _hdr[3].text = "p (raw)"
            _hdr[4].text = "p (corrected)"
            
            for _ri, r in enumerate(_results, 1):
                _row = _tbl.rows[_ri].cells
                _row[0].text = r['group_a']
                _row[1].text = r['group_b']
                _row[2].text = f"{r['effect']:.4f}"
                _row[3].text = _fmt_pvalue(r['p_raw'])
                _row[4].text = _fmt_pvalue(r['p_corrected']) + (" *" if r['rejected'] else "")
            
            doc.add_paragraph()


def analyze_per_run_direction(df_data: pd.DataFrame, metric_base: str, y_label: str, filename_str: str, group_color_map: dict = None) -> None:
    """
    Analyze a metric stratified by direction within groups.
    Creates box plots and statistics for each group vs. direction combination.
    
    Args:
        df_data: DataFrame with 'group' column and direction columns (e.g., 'food_sensed_north', 'food_sensed_east', etc.)
        metric_base: Base name for metric columns (e.g., 'food_sensed' to match 'food_sensed_north', 'food_sensed_east', etc.)
        y_label: Label for y-axis (e.g., 'Food Sensed per Tick')
        filename_str: Base name for output files
        group_color_map: dict mapping original (lowercase) group names to hex color strings.
    """
    
    directions = ['north', 'east', 'south', 'west', 'stay']
    
    # Dynamically discover and extract all groups from the data
    # Order: benchmarks first, then experiments (so experiments are drawn on top)
    all_unique_groups, zorder_map = _get_ordered_groups_with_zorder(df_data['group'].unique())
    
    # Prepare data: for each group and direction, collect normalized values
    group_direction_data = {}  # {(group_name, direction): [values]}
    group_order = []
    
    for group_name in all_unique_groups:
        if group_name not in group_order:
            group_order.append(group_name)
        
        df_group = df_data[df_data['group'] == group_name]
        
        for direction in directions:
            col_name = f"{metric_base}_{direction}"
            if col_name in df_data.columns:
                group_direction_data[(group_name, direction)] = df_group[col_name].values
    
    if not group_direction_data:
        doc.add_paragraph(f"No direction data found for {metric_base}.")
        return
    
    # --- Plot: box plot with directions stratified within groups ---
    _gcm_dir = group_color_map or {}

    def _color_for_dir(group_name: str) -> str:
        return _gcm_dir.get(group_name, '#808080')

    fig, ax = plt.subplots(figsize=(14, 7))
    
    # First pass: determine which directions actually have data
    available_directions = set()
    for group in group_order:
        df_group = df_data[df_data['group'] == group]
        for direction in directions:
            col_name = f"{metric_base}_{direction}"
            if col_name in df_data.columns and col_name in df_group.columns:
                available_directions.add(direction)
    
    # Use only available directions
    direction_order = [d for d in directions if d in available_directions]
    
    if not direction_order:
        doc.add_paragraph(f"No direction data found for {metric_base}.")
        return
    
    # Flatten data for boxplot and create positions
    all_box_data = []
    all_positions = []
    all_labels = []
    pos = 0
    
    for group_idx, group in enumerate(group_order):
        group_start_pos = pos
        for dir_idx, direction in enumerate(direction_order):
            key = (group, direction)
            if key in group_direction_data:
                all_box_data.append(group_direction_data[key])
                all_positions.append(pos)
                # Create label with direction abbreviation
                dir_abbr = direction[0].upper()  # N, E, S, W
                if direction == 'stay':
                    dir_abbr = 'St'
                all_labels.append(dir_abbr)
                pos += 1
        pos += 1  # Add spacing between groups
    
    # Jitter overlay (draw first so it appears behind boxplots)
    box_idx = 0
    for group_idx, group in enumerate(group_order):
        for dir_idx, direction in enumerate(direction_order):
            key = (group, direction)
            if key in group_direction_data:
                values = group_direction_data[key]
                color = _color_for_dir(group)
                jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(values))
                ax.scatter(np.full(len(values), all_positions[box_idx]) + jitter, values, color=color, alpha=0.4, s=8, zorder=1)
                box_idx += 1
    
    # Box plots (draw second so they appear on top)
    bp = ax.boxplot(
        all_box_data,
        positions=all_positions,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
    )
    
    # Color boxes by group
    box_idx = 0
    for group_idx, group in enumerate(group_order):
        color = _color_for_dir(group)
        for direction in direction_order:
            key = (group, direction)
            if key in group_direction_data:
                if box_idx < len(bp['boxes']):
                    bp['boxes'][box_idx].set_facecolor(color)
                    bp['boxes'][box_idx].set_alpha(0.3)
                    bp['medians'][box_idx].set_color('black')
                box_idx += 1
    
    # Set x-axis labels: direction abbreviations
    ax.set_xticks(all_positions)
    ax.set_xticklabels(all_labels, fontsize=10)
    
    # Set group labels on a secondary level
    group_positions = []
    group_labels_text = []
    pos = 0
    for group_idx, group in enumerate(group_order):
        group_start = pos
        pos += len(direction_order)  # Move past this group's directions
        group_center = (group_start + pos - 1) / 2
        group_positions.append(group_center)
        group_labels_text.append(group)
        pos += 1  # Add spacing between groups
    
    # Add group labels as text above the plot
    for group_pos, group_label in zip(group_positions, group_labels_text):
        ax.text(group_pos, ax.get_ylim()[1] * 0.95, group_label, 
                ha='center', va='top', fontsize=11, fontweight='bold')
    
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(f'{y_label} by Group and Direction', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add vertical separators between groups
    pos = 0
    for group_idx in range(len(group_order) - 1):
        pos += len(direction_order) + 0.5  # Move to end of group + spacing
        ax.axvline(x=pos - 0.5, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    # Add legend for directions
    from matplotlib.lines import Line2D
    direction_full_names = {
        'north': 'North (N)',
        'east': 'East (E)',
        'south': 'South (S)',
        'west': 'West (W)',
        'stay': 'Stay (St)'
    }
    
    # Save figure
    figures_dir = Path(__file__).resolve().parent / f'figures_{EXPERIMENT_NAME}'
    figures_dir.mkdir(exist_ok=True)
    output_path = figures_dir / f'groups_direction_{filename_str}.png'
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    doc.add_picture(str(output_path), width=6.5 * 914400)
    doc.add_paragraph()
    
    # --- Summary statistics table ---
    from scipy.stats import sem as scipy_sem
    
    stat_names = ['N', 'Mean', 'Std Dev', 'Median', 'Min', 'Max']
    
    # Create table with columns for each direction within each group
    table = doc.add_table(rows=len(stat_names) + 1, cols=len(direction_order) + 1)
    table.style = 'Light Grid Accent 1'
    
    # Header row
    table.rows[0].cells[0].text = 'Statistic'
    for dir_idx, direction in enumerate(direction_order, 1):
        table.rows[0].cells[dir_idx].text = direction.capitalize()
    
    # Data rows
    for stat_idx, stat_name in enumerate(stat_names, 1):
        table.rows[stat_idx].cells[0].text = stat_name
        for dir_idx, direction in enumerate(direction_order, 1):
            # Average across all groups for this direction
            values_all_groups = []
            for group in group_order:
                key = (group, direction)
                if key in group_direction_data:
                    values_all_groups.extend(group_direction_data[key])
            
            if values_all_groups:
                v = np.array(values_all_groups)
                if stat_name == 'N':
                    # Count non-NaN values
                    table.rows[stat_idx].cells[dir_idx].text = str(np.sum(~np.isnan(v)))
                elif stat_name == 'Mean':
                    table.rows[stat_idx].cells[dir_idx].text = f'{np.nanmean(v):.2f}'
                elif stat_name == 'Std Dev':
                    table.rows[stat_idx].cells[dir_idx].text = f'{np.nanstd(v, ddof=1):.2f}'
                elif stat_name == 'Median':
                    table.rows[stat_idx].cells[dir_idx].text = f'{np.nanmedian(v):.2f}'
                elif stat_name == 'Min':
                    table.rows[stat_idx].cells[dir_idx].text = f'{np.nanmin(v):.2f}'
                elif stat_name == 'Max':
                    table.rows[stat_idx].cells[dir_idx].text = f'{np.nanmax(v):.2f}'
    
    doc.add_paragraph()
    
    # --- Second Plot: Variant-level lines stratified by direction ---
    # Prepare variant-level data: for each group, variant, and direction, store values and calculate stats
    variant_direction_values = {}  # {(group, variant, direction): [values]}
    variant_direction_stats = {}  # {(group, variant, direction): (mean, ci_lower, ci_upper)}
    variants_per_group = {}  # {group: [list of variants]}
    variant_to_group = {}  # {variant: group} for color mapping
    
    # Build the mapping from original data
    for group_name in all_unique_groups:
        df_group = df_data[df_data['group'] == group_name]
        
        variants_in_group = sorted(df_group['variant'].unique())
        variants_per_group[group_name] = variants_in_group
        
        for variant in variants_in_group:
            variant_to_group[variant] = group_name
            df_variant = df_group[df_group['variant'] == variant]
            
            for direction in direction_order:
                col_name = f"{metric_base}_{direction}"
                if col_name in df_variant.columns:
                    values = df_variant[col_name].values
                    values_clean = values[~np.isnan(values)]
                    
                    # Store raw values
                    variant_direction_values[(group_name, variant, direction)] = values_clean
                    
                    # Calculate mean and 95% CI
                    mean_val = np.mean(values_clean) if len(values_clean) > 0 else np.nan
                    if len(values_clean) > 1:
                        se = np.std(values_clean, ddof=1) / np.sqrt(len(values_clean))
                        ci_delta = 1.96 * se  # 95% CI
                        ci_lower = mean_val - ci_delta
                        ci_upper = mean_val + ci_delta
                    else:
                        ci_lower = ci_upper = mean_val
                    
                    variant_direction_stats[(group_name, variant, direction)] = (mean_val, ci_lower, ci_upper)
    
    if not variant_direction_stats:
        doc.add_paragraph(f"No variant-level direction data found for {metric_base}.")
        return
    
    # Create faceted plot: one subplot per group
    n_groups = len(group_order)
    fig, axes = plt.subplots(1, n_groups, figsize=(5 * n_groups, 5), sharey=True)
    
    # Handle case where there's only one group (axes would be a 1D array)
    if n_groups == 1:
        axes = [axes]
    
    # Direction abbreviations for x-axis labels
    direction_abbr_map = {
        'north': 'N',
        'east': 'E',
        'south': 'S',
        'west': 'W',
        'stay': 'Stay'
    }
    
    direction_labels = [direction_abbr_map.get(d, d) for d in direction_order]
    x_positions = np.arange(len(direction_order))
    
    # Color mapping by group (build from passed-in color map)
    _facet_color_map = {g: _color_for_dir(g) for g in group_order}
    
    # Plot each group in its own facet
    for group_idx, group in enumerate(group_order):
        ax = axes[group_idx]
        variants = sorted(variants_per_group.get(group, []))
        
        # Plot one line per variant with confidence band
        for variant in variants:
            means = []
            ci_lowers = []
            ci_uppers = []
            
            for direction in direction_order:
                key = (group, variant, direction)
                if key in variant_direction_stats:
                    mean_val, ci_lower, ci_upper = variant_direction_stats[key]
                    means.append(mean_val)
                    ci_lowers.append(ci_lower)
                    ci_uppers.append(ci_upper)
                else:
                    means.append(np.nan)
                    ci_lowers.append(np.nan)
                    ci_uppers.append(np.nan)
            
            # Use group color for the variant line
            color = _facet_color_map.get(group, '#808080')
            
            # Plot confidence band
            ax.fill_between(x_positions, ci_lowers, ci_uppers, color=color, alpha=0.15, zorder=1)
            
            # Plot line for this variant
            ax.plot(x_positions, means, color=color, marker='o', linewidth=1.5, markersize=4, alpha=0.8, zorder=2)
        
        # Formatting
        ax.set_xlabel('Direction', fontsize=10)
        if group_idx == 0:
            ax.set_ylabel(y_label, fontsize=10)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(direction_labels, fontsize=9)
        ax.set_title(f'{group}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Overall title
    fig.suptitle(f'{y_label} per Variant', 
                 fontsize=13, fontweight='bold', y=1.00)
    fig.tight_layout()
    
    # Save figure
    output_path_variants = figures_dir / f'groups_direction_variants_{filename_str}.png'
    fig.savefig(output_path_variants, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Add to document
    doc.add_picture(str(output_path_variants), width=6.5 * 914400)
    doc.add_paragraph()


def analyze_heatmaps(heatmap_data: dict, df_groups: pd.DataFrame) -> None:
    """
    Generate and display staying heatmaps for groups and variants.
    
    Skips 'other' group. For each variant in selected groups, creates a figure showing:
    - Top-left: Averaged heatmap across all runs (with own colorbar)
    - Other 3 panels: Raw data heatmaps from first 3 runs (with shared colorbar, if available)
    
    Args:
        heatmap_data: Dict mapping source -> variant -> [arrays...]
        df_groups: DataFrame with group assignments (must have 'source', 'variant', 'group' columns)
    """
    
    # Return early if no heatmap data
    if not heatmap_data:
        return
    
    # Build group -> (source, variants) mapping
    group_variants = {}  # {group_name: [(source, variant_name), ...]}
    for source in sorted(heatmap_data.keys()):
        for variant_name in sorted(heatmap_data[source].keys()):
            # Find group for this source/variant from the dataframe
            mask = (df_groups['source'] == source) & (df_groups['variant'] == variant_name)
            if mask.any():
                group = df_groups.loc[mask, 'group'].iloc[0]
            else:
                group = source
            
            if group not in group_variants:
                group_variants[group] = []
            group_variants[group].append((source, variant_name))
    
    # For each group and variant, create a figure
    figures_dir = Path(__file__).resolve().parent / f'figures_{EXPERIMENT_NAME}'
    figures_dir.mkdir(exist_ok=True)
    
    for group_name in sorted(group_variants.keys()):
        source_variant_pairs = group_variants[group_name]
        
        # Limit to first 3 variants per group
        variants_to_process = source_variant_pairs[:3]
        
        # Process each variant in the group (create separate figure for each)
        for source, variant_name in variants_to_process:
            
            # Get staying heatmap arrays for this variant
            arrays = heatmap_data[source][variant_name]
            if len(arrays) == 0:
                continue
            
            fig = plt.figure(figsize=(14, 12))
            
            # Panel 0: Averaged heatmap
            averaged_heatmap = np.mean(arrays, axis=0)
            avg_values = averaged_heatmap.flatten()
            avg_vmin = 0  # Minimum is always 0 for counts
            avg_vmax = np.max(avg_values)  # Use actual maximum, not percentile
            
            # Panels 1-3: Raw run data (first 3 runs)
            raw_arrays = []
            for run_idx in range(min(3, len(arrays))):
                raw_arrays.append(arrays[run_idx])
            
            # Calculate shared colormap for raw data
            raw_values = np.concatenate([arr.flatten() for arr in raw_arrays])
            raw_vmin = 0  # Minimum is always 0 for counts
            raw_vmax = np.max(raw_values)  # Use actual maximum, not percentile
            
            # Create panels
            for pane_idx in range(4):
                ax = fig.add_subplot(2, 2, pane_idx + 1)
                
                if pane_idx == 0:
                    # Averaged heatmap
                    im = ax.imshow(averaged_heatmap, cmap='RdYlBu_r', vmin=avg_vmin, vmax=avg_vmax, aspect='auto', origin='upper')
                    variant_num = variant_name.replace('variant_', '')
                    ax.set_title(f'{group_name} (variant {variant_num}, all runs averaged)', fontsize=11, fontweight='bold')
                    
                    # Add colorbar below this panel
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("bottom", size="10%", pad=0.1)
                    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
                    cbar.set_label('Value', fontsize=9)
                
                else:
                    # Raw run data
                    run_idx = pane_idx - 1
                    if run_idx < len(raw_arrays):
                        im = ax.imshow(raw_arrays[run_idx], cmap='RdYlBu_r', vmin=raw_vmin, vmax=raw_vmax, aspect='auto', origin='upper')
                        variant_num = variant_name.replace('variant_', '')
                        ax.set_title(f'{group_name} (variant {variant_num}, run {run_idx + 1})', fontsize=11, fontweight='bold')
                        
                        # Add colorbar below this panel
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("bottom", size="10%", pad=0.1)
                        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
                        cbar.set_label('Value', fontsize=9)
                    else:
                        # Empty panel
                        ax.set_title(f'Run {run_idx + 1} (N/A)', fontsize=11)
                
                # Remove axes labels and ticks
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            
            # Adjust layout
            fig.subplots_adjust(left=0.08, right=0.95, top=0.92, hspace=0.5, wspace=0.25)
            
            # Save figure
            group_display_name = group_name.replace(' ', '_')
            output_filename = f'heatmap_{group_display_name}_{variant_name}_staying.png'
            output_path = figures_dir / output_filename
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # Add to document
            doc.add_picture(str(output_path), width=5.0 * 914400)
    doc.add_paragraph()


# Unicode superscript digits + minus for scientific p-value formatting
_SUPERSCRIPT_DIGITS = str.maketrans('0123456789', '\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079')


def _fmt_pvalue(p: float) -> str:
    """Format p-value as decimal (>= 0.001) or scientific notation (< 0.001).
    p == 0.0 means float underflow; reported as < 10⁻³⁰⁰."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "N/A"
    if p == 0.0:
        return "< 10\u207b\u00b3\u2070\u2070"  # < 10⁻³⁰⁰
    if p >= 0.001:
        return f"{p:.4f}"
    exp = int(np.floor(np.log10(p)))   # e.g. -15
    mantissa = p / (10.0 ** exp)
    exp_str = str(abs(exp)).translate(_SUPERSCRIPT_DIGITS)
    return f"{mantissa:.2f} \u00d7 10\u207b{exp_str}"  # e.g. 2.85 × 10⁻¹⁵


def _logrank_pairwise(df_summary: pd.DataFrame) -> list:
    """
    Pairwise log-rank tests: all possible combinations of groups.
    Applies Holm-Bonferroni correction across all pairs.

    Returns list of dicts sorted by (group_a, group_b):
        group_a, group_b, chi2, p_raw, p_corrected, rejected (bool)
    Returns empty list when fewer than two groups exist.
    """
    from scipy.stats import chi2 as chi2_dist

    loaded = sorted(df_summary['group'].unique())
    pairs  = list(combinations(loaded, 2))
    if not pairs:
        return []

    results = []
    for a, b in pairs:
        df_pair     = df_summary[df_summary['group'].isin([a, b])]
        event_times = np.sort(df_pair['lifetime_ticks'].unique())
        O = {a: 0.0, b: 0.0}
        E = {a: 0.0, b: 0.0}
        V = {a: 0.0, b: 0.0}
        for t in event_times:
            n_j = int((df_pair['lifetime_ticks'] >= t).sum())
            d_j = int((df_pair['lifetime_ticks'] == t).sum())
            if n_j <= 1 or d_j == 0:
                continue
            for g in [a, b]:
                mask = df_pair['group'] == g
                n_ij = int((df_pair.loc[mask, 'lifetime_ticks'] >= t).sum())
                d_ij = int((df_pair.loc[mask, 'lifetime_ticks'] == t).sum())
                e_ij = n_ij * d_j / n_j
                O[g] += d_ij
                E[g] += e_ij
                V[g] += (n_ij / n_j) * (1.0 - n_ij / n_j) * d_j * (n_j - d_j) / max(n_j - 1, 1)
        stat  = sum((O[g] - E[g]) ** 2 / V[g] for g in [a, b] if V[g] > 0.0)
        p_raw = float(chi2_dist.sf(stat, df=1))
        results.append({'group_a': a, 'group_b': b, 'chi2': stat, 'p_raw': p_raw})

    p_raws = [r['p_raw'] for r in results]
    rejected, p_corrected, _, _ = multipletests(p_raws, alpha=0.05, method='holm')
    for r, p_corr, rej in zip(results, p_corrected, rejected):
        r['p_corrected'] = float(p_corr)
        r['rejected']    = bool(rej)

    return results


def _logrank_test(df_summary: pd.DataFrame) -> dict:
    """
    k-sample log-rank test comparing lifetime_ticks distributions across groups.
    All runs are fully observed (no censoring).

    Returns dict with:
        chi2, df, p_value, groups (list), observed (dict), expected (dict)
    """
    from scipy.stats import chi2 as chi2_dist

    groups = list(df_summary['group'].unique())
    event_times = np.sort(df_summary['lifetime_ticks'].unique())

    O = {g: 0.0 for g in groups}
    E = {g: 0.0 for g in groups}
    V = {g: 0.0 for g in groups}

    for t in event_times:
        n_j = int((df_summary['lifetime_ticks'] >= t).sum())
        d_j = int((df_summary['lifetime_ticks'] == t).sum())
        if n_j <= 1 or d_j == 0:
            continue
        for g in groups:
            mask = df_summary['group'] == g
            n_ij = int((df_summary.loc[mask, 'lifetime_ticks'] >= t).sum())
            d_ij = int((df_summary.loc[mask, 'lifetime_ticks'] == t).sum())
            e_ij = n_ij * d_j / n_j
            O[g] += d_ij
            E[g] += e_ij
            V[g] += (n_ij / n_j) * (1.0 - n_ij / n_j) * d_j * (n_j - d_j) / (n_j - 1)

    stat = sum((O[g] - E[g]) ** 2 / V[g] for g in groups if V[g] > 0.0)
    df_val = len(groups) - 1
    p_val = float(chi2_dist.sf(stat, df=df_val))

    return {
        'chi2': stat,
        'df': df_val,
        'p_value': p_val,
        'groups': groups,
        'observed': O,
        'expected': E,
    }


def analyze_survival_race(df_summary: pd.DataFrame, doc = None, group_color_map: dict = None) -> None:
    """
    Analyze and report survival data: visualization, log-rank statistics (omnibus and pairwise),
    and comparative tables.
    
    Parameters:
    -----------
    df_summary : pd.DataFrame
        Summary data with 'group', 'variant', 'run_id', and 'lifetime_ticks' columns.
    doc : Word document object (optional)
        If provided, adds visualization and statistical tables to the document.
    group_color_map : dict (optional)
        Maps group names to hex color strings for visualization.
    """
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(12, 8))

    all_groups, zorder_map = _get_ordered_groups_with_zorder(df_summary['group'].unique())
    _gcm = group_color_map or {}

    for (group_name, variant_id), grp in df_summary.groupby(['group', 'variant']):
        survival_times = grp['lifetime_ticks'].to_numpy(dtype=float)
        color  = _gcm.get(group_name, '#808080')
        zorder = zorder_map.get(group_name, 1)
        max_tick = int(survival_times.max())
        ticks  = np.arange(0, max_tick + 1)
        alive  = np.array([(survival_times >= t).sum() for t in ticks])
        ax.plot(ticks, alive, color=color, zorder=zorder)

    legend_elements = [
        Line2D([0], [0], color=_gcm.get(g, '#808080'), linewidth=2, label=g)
        for g in all_groups
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    ax.set_xlabel('Ticks')
    ax.set_ylabel('Runs Alive')
    ax.set_title('Survival Race')
    ax.grid(True, alpha=0.3)

    figures_dir = Path(__file__).resolve().parent / f'figures_{EXPERIMENT_NAME}'
    figures_dir.mkdir(exist_ok=True)
    output_path = figures_dir / 'survival_race.png'
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Add visualization to document if provided
    doc.add_picture(str(output_path), width=6.5 * 914400)
    doc.add_paragraph()

    # --- Omnibus log-rank test ---
    if df_summary['group'].nunique() < 2:
        doc.add_paragraph('Only one group — no log-rank statistics.')
        return

    _lr = _logrank_test(df_summary)

    doc.add_heading("Omnibus", level=3)
    _lr_table = doc.add_table(rows=len(_lr['groups']) + 1, cols=4)
    _lr_table.style = "Light Grid Accent 1"
    _lr_hdr = _lr_table.rows[0].cells
    _lr_hdr[0].text = "Group"
    _lr_hdr[1].text = "Observed deaths"
    _lr_hdr[2].text = "Expected deaths"
    _lr_hdr[3].text = "(O - E) / E"
    for _ri, _gname in enumerate(_lr['groups'], 1):
        _o = _lr['observed'][_gname]
        _e = _lr['expected'][_gname]
        _r = _lr_table.rows[_ri].cells
        _r[0].text = _gname
        _r[1].text = f"{_o:.0f}"
        _r[2].text = f"{_e:.2f}"
        _r[3].text = f"{(_o - _e) / _e * 100:.2f} %"

    doc.add_paragraph()
    _sig_omni = "significant" if _lr['p_value'] < 0.05 else "not significant"
    doc.add_paragraph(
        f"Omnibus log-rank test: \u03c7\u00b2({_lr['df']}) = {_lr['chi2']:.3f}, "
        f"p = {_fmt_pvalue(_lr['p_value'])}. The difference between survival curves is {_sig_omni} "
        f"at \u03b1 = 0.05."
    )

    # --- Pairwise log-rank tests (Holm-Bonferroni corrected) ---
    _lr_pairs = _logrank_pairwise(df_summary)
    if _lr_pairs:
        doc.add_heading("Pairwise Comparisons (Holm-Bonferroni corrected)", level=3)
        _pw_table = doc.add_table(rows=len(_lr_pairs) + 1, cols=5)
        _pw_table.style = "Light Grid Accent 1"
        _pw_hdr = _pw_table.rows[0].cells
        _pw_hdr[0].text = "Group A"
        _pw_hdr[1].text = "Group B"
        _pw_hdr[2].text = "\u03c7\u00b2"
        _pw_hdr[3].text = "p (raw)"
        _pw_hdr[4].text = "p (corrected)"
        for _ri, _pw in enumerate(_lr_pairs, 1):
            _r = _pw_table.rows[_ri].cells
            _r[0].text = _pw['group_a']
            _r[1].text = _pw['group_b']
            _r[2].text = f"{_pw['chi2']:.3f}"
            _r[3].text = _fmt_pvalue(_pw['p_raw'])
            _r[4].text = _fmt_pvalue(_pw['p_corrected']) + (" *" if _pw['rejected'] else "")
        doc.add_paragraph()
        print(f"  Pairwise: {len(_lr_pairs)} pair(s) tested with Holm-Bonferroni correction.")


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
    figures_dir = Path(__file__).resolve().parent / f'figures_{EXPERIMENT_NAME}'
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


def track_decision_precision(df_per_tick: pd.DataFrame, df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate decision precision per direction for each run (ULTRA-OPTIMIZED).
    
    Uses the fact that food can only be consumed during stay decisions.
    Skips first tick to avoid boundary checks and ensure safe lookback within runs.
    Pure counting approach: no filtering, just vectorized aggregation.
    
    Args:
        df_per_tick: Per-tick DataFrame with columns: tick, decision_made, movement, food_consumed, run_id, variant, group, source
        df_all: Run-level DataFrame to add precision columns to
    
    Returns:
        df_all with added decision_precision_{direction} columns (north, east, south, west, stay)
    """
    
    # Initialize precision columns with NaN
    df_all.flags.writeable = True
    for direction in ['north', 'east', 'south', 'west', 'stay']:
        df_all[f'decision_precision_{direction}'] = np.nan
    
    # Skip first tick (no decisions in tick 0, ensures safe lookback)
    df = df_per_tick.iloc[1:].copy()
    
    if len(df) == 0:
        df_all.flags.writeable = False
        return df_all
    
    # === Create lookahead column for food in next tick (within each run) ===
    df['food_next'] = df.groupby(['variant', 'source', 'run_id'])['food_consumed'].shift(-1).fillna(0).astype(int)
    
    # === Vectorize movement decoding ===
    def decode_movement_vec(movement_series):
        decoded = movement_series.apply(
            lambda x: x.decode('utf-8').lower() if isinstance(x, bytes) else str(x).lower()
        )
        direction_map = {'n': 'north', 'e': 'east', 's': 'south', 'w': 'west', 'stay': 'stay'}
        return decoded.map(direction_map)
    
    df['direction'] = decode_movement_vec(df['movement'])
    df = df[df['direction'].notna()]  # Remove unmapped directions
    
    # === Process each direction ===
    results = {}
    
    for direction in ['north', 'east', 'south', 'west', 'stay']:
        if direction == 'stay':
            # Food can only be consumed during stay: food_consumed = correct stays
            total_counts = df.groupby(['variant', 'source', 'run_id'])['decision_made'].sum()
            correct_counts = df.groupby(['variant', 'source', 'run_id'])['food_consumed'].sum()
        else:
            # Movement decisions followed by food consumption in next tick
            df_dir = df[df['direction'] == direction]
            
            if len(df_dir) == 0:
                continue
            
            total_counts = df_dir.groupby(['variant', 'source', 'run_id']).size()
            correct_counts = df_dir.groupby(['variant', 'source', 'run_id'])['food_next'].sum()
        
        # Calculate precision (avoid division by zero)
        precision = correct_counts / total_counts.replace(0, np.nan)
        results[direction] = precision
    
    # === Batch update df_all using index alignment ===
    df_all_indexed = df_all.set_index(['variant', 'source', 'run_id'])
    
    for direction, precision_series in results.items():
        df_all_indexed[f'decision_precision_{direction}'] = precision_series
    
    # Reset index to return to original format
    df_all = df_all_indexed.reset_index()
    
    df_all.flags.writeable = False
    return df_all


def load_per_tick_data(experiment_hdf5_path: Path, variant_groups: dict, hdf5_files_dict: dict) -> pd.DataFrame:
    """
    Load per_tick data from HDF5 files for successful, unsuccessful, and benchmark variants.
    
    Args:
        experiment_hdf5_path: Path to experiment HDF5 file
        variant_groups: Dict mapping variant names to their group assignments (successful/unsuccessful/other)
        hdf5_files_dict: Dict mapping source labels to HDF5 file paths
    
    Returns:
        DataFrame with per_tick data, including source, variant, group, and run_id columns.
        DataFrame is sealed immutable.
    
    Raises:
        ValueError: If no per_tick data found in any of the specified groups
    """
    all_per_tick_data = []
    
    # Load per_tick data from all experiment sources
    for source_label, source_path in hdf5_files_dict.items():
        # Determine group name from source label
        # Experiments use source label directly as group name; benchmarks too
        try:
            with h5py.File(source_path, 'r') as f:
                vnames = sorted([k for k in f.keys() if k.startswith('variant_')])
                for variant_name in vnames:
                    # Look up group assignment from variant_groups if available, else use source_label
                    group_assignment = variant_groups.get(variant_name, source_label)
                    vg = f[variant_name]
                    run_names = sorted([k for k in vg.keys() if k.startswith('run_')])
                    for run_name in run_names:
                        if 'per_tick' not in vg[run_name]:
                            continue
                        run_group = vg[run_name]
                        per_tick_table = run_group['per_tick']
                        if isinstance(per_tick_table, h5py.Dataset):
                            per_tick_df = pd.DataFrame(per_tick_table[:])
                            per_tick_df['source'] = source_label
                            per_tick_df['variant'] = variant_name
                            per_tick_df['group'] = group_assignment
                            run_id_numeric = int(run_name.split('_')[1])
                            per_tick_df['run_id'] = run_id_numeric
                            all_per_tick_data.append(per_tick_df)
        except Exception as e:
            print(f"Warning: Error loading per_tick data for '{source_label}': {e}")
            raise ValueError(f"Failed to load per_tick data for '{source_label}'") from e
    
    # (legacy loop kept as placeholder - logic merged into unified loop above)
    if False:  # dead code guard
        pass
    
    # (benchmark loading merged into unified loop above)
    
    # Combine all per_tick data
    if not all_per_tick_data:
        raise ValueError("No per_tick data found")
    
    df_all_per_tick = pd.concat(all_per_tick_data, ignore_index=True)
    # Make immutable
    df_all_per_tick.flags.writeable = False
    print(f"\nPer-tick dataset: {len(df_all_per_tick)} total ticks across {df_all_per_tick['variant'].nunique()} variants")
    
    return df_all_per_tick


def load_heatmap_data(experiment_hdf5_path: Path, variant_groups: dict, hdf5_files_dict: dict) -> dict:
    """
    Load staying heatmap data from all HDF5 sources (experiments and benchmarks).
    
    Args:
        experiment_hdf5_path: Path to primary experiment HDF5 file (unused, kept for API compat)
        variant_groups: Dict mapping variant names to their group assignments
        hdf5_files_dict: Dict mapping source labels to HDF5 file paths
    
    Returns:
        Dict: {source: {variant: [arrays...]}}
              Returns empty dict if no heatmap data exists.
    """
    heatmap_data = {}  # {source: {variant: [arrays...]}}
    
    for source_label, source_path in hdf5_files_dict.items():
        try:
            with h5py.File(source_path, 'r') as f:
                variant_keys = sorted([k for k in f.keys() if k.startswith('variant_')])
                for variant_name in variant_keys:
                    vg = f[variant_name]
                    run_keys = sorted([k for k in vg.keys() if k.startswith('run_')])
                    
                    staying_arrays = []
                    for run_name in run_keys:
                        run_group = vg[run_name]
                        if 'staying' in run_group:
                            staying_arrays.append(run_group['staying'][:])
                    
                    if staying_arrays:
                        if source_label not in heatmap_data:
                            heatmap_data[source_label] = {}
                        heatmap_data[source_label][variant_name] = staying_arrays
        except Exception as e:
            print(f"Warning: Error loading heatmap data for '{source_label}': {e}")
    
    # Print summary
    total_variants = sum(len(variants_dict) for variants_dict in heatmap_data.values())
    if total_variants > 0:
        print(f"\nHeatmap dataset: {total_variants} variants loaded (staying data)")
    
    return heatmap_data


def analyze_per_tick_metric(df_per_tick: pd.DataFrame, metric_name: str, y_label: str, filename_str: str, group_color_map: dict = None) -> None:
    """
    Plot a per-tick metric as two time-series figures and run adaptive statistical tests.

    Steps:
      1. Silently return if per_tick_available is False or metric column is missing.
      2. Plot 1: absolute ticks on x-axis: per-group mean line + 95% CI band.
      3. Adaptive stats on per-run means (calls analyze_per_run).
      4. Plot 2: normalized % lifetime on x-axis (uses ticks_norm column from df_per_tick):
         same aggregation as Plot 1 but with x=ticks_norm.
    """
    if not per_tick_available or df_per_tick is None:
        return
    if metric_name not in df_per_tick.columns:
        return

    all_groups, zorder_map = _get_ordered_groups_with_zorder(df_per_tick['group'].unique())
    figures_dir = Path(__file__).resolve().parent / f'figures_{EXPERIMENT_NAME}'
    figures_dir.mkdir(exist_ok=True)

    # ── Plot 1: absolute ticks ─────────────────────────────────────────────────
    tick_stats = (
        df_per_tick.groupby(['group', 'tick'])[metric_name]
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )
    tick_stats['sem'] = tick_stats['std'] / np.sqrt(tick_stats['count'])
    tick_stats['ci_lo'] = tick_stats['mean'] - 1.96 * tick_stats['sem']
    tick_stats['ci_hi'] = tick_stats['mean'] + 1.96 * tick_stats['sem']

    fig, ax = plt.subplots(figsize=(14, 7))
    for g in all_groups:
        gdf = tick_stats[tick_stats['group'] == g].sort_values('tick')
        color = COLOR_MAP.get(g, '#808080')
        z = zorder_map.get(g, 50)
        ax.plot(gdf['tick'], gdf['mean'], color=color, linewidth=2.5, label=g, zorder=z)
        ax.fill_between(gdf['tick'], gdf['ci_lo'], gdf['ci_hi'], color=color, alpha=0.2, zorder=z-1)
    ax.set_xlabel('Tick', fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(f'{y_label} over Time', fontsize=14)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    out_abs = figures_dir / f'ticks_abs_{filename_str}.png'
    fig.tight_layout()
    fig.savefig(out_abs, dpi=150, bbox_inches='tight')
    plt.close(fig)

    doc.add_picture(str(out_abs), width=6.5 * 914400)
    doc.add_paragraph()

    # ── Stats: per-run mean → adaptive tests via analyze_per_run ──────────────
    df_run_means = (
        df_per_tick.groupby(['group', 'variant', 'run'])[metric_name]
        .mean()
        .reset_index()
    )
    analyze_per_run(df_run_means, metric_name, f'Mean {y_label} per Run',
                    f'{filename_str}_stats', group_color_map=group_color_map)

    # ── Plot 2: normalised % lifetime (ticks_norm column) ─────────────────────
    norm_stats = (
        df_per_tick.groupby(['group', 'ticks_norm'])[metric_name]
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )
    norm_stats['sem'] = norm_stats['std'] / np.sqrt(norm_stats['count'])
    norm_stats['ci_lo'] = norm_stats['mean'] - 1.96 * norm_stats['sem']
    norm_stats['ci_hi'] = norm_stats['mean'] + 1.96 * norm_stats['sem']

    fig, ax = plt.subplots(figsize=(14, 7))
    for g in all_groups:
        gdf = norm_stats[norm_stats['group'] == g].sort_values('ticks_norm')
        color = COLOR_MAP.get(g, '#808080')
        z = zorder_map.get(g, 50)
        ax.plot(gdf['ticks_norm'], gdf['mean'], color=color, linewidth=2.5, label=g, zorder=z)
        ax.fill_between(gdf['ticks_norm'], gdf['ci_lo'], gdf['ci_hi'], color=color, alpha=0.2, zorder=z-1)
    ax.set_xlabel('Percentage of Lifespan (%)', fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(f'{y_label} across Lifespan (normalized)', fontsize=14)
    ax.set_xlim(0, 100)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    out_norm = figures_dir / f'ticks_norm_{filename_str}.png'
    fig.tight_layout()
    fig.savefig(out_norm, dpi=150, bbox_inches='tight')
    plt.close(fig)

    doc.add_picture(str(out_norm), width=6.5 * 914400)
    doc.add_paragraph()


def _draw_wiring_panel(ax, wiring_df, modulation_array, neuron_positions, neuron_types,
                       title: str) -> None:
    """Draw one network panel into ax. wiring_df must already contain a 'weight' column."""
    conns = wiring_df[['src', 'tgt', 'weight']].copy()
    conns = conns[conns['weight'] != 0.0].reset_index(drop=True)

    mod_df = None
    if modulation_array is not None and len(modulation_array) > 0:
        mod_df = pd.DataFrame(modulation_array)

    neurons = {
        nid: {'pos': pos, 'type': neuron_types.get(nid, 'hidden')}
        for nid, pos in neuron_positions.items()
    }

    network_viz.draw_network(
        neurons=neurons,
        connections=conns,
        modulatory=mod_df,
        ax=ax,
        title=title,
        show_weights=True,
        weight_column='weight',
    )


def calculate_connectivity(per_neuron_data: dict, wiring_data: dict) -> None:
    """
    Add 'in_degree' and 'out_degree' fields to per_neuron_data in-place.

    wiring_data already contains only active connections (abs(weight) > 0), so
    in-degree = number of times a neuron ID appears in 'tgt',
    out-degree = number of times it appears in 'src'.

    The plain 1-D tonic-activation array stored for each variant is replaced with a
    structured numpy array containing three fields:
        'tonic_activation' (f4), 'in_degree' (i4), 'out_degree' (i4).

    Args:
        per_neuron_data: {group_name: {variant_id: np.ndarray}} — per-neuron tonic values.
        wiring_data:     {group_name: {variant_id: structured array}} — active connections
                         with at minimum fields 'src' and 'tgt'.
    """
    for grp, variants in per_neuron_data.items():
        wiring_variants = wiring_data.get(grp, {})
        for vid, tonic_arr in variants.items():
            n_neurons  = len(tonic_arr)
            in_degree  = np.zeros(n_neurons, dtype=np.int32)
            out_degree = np.zeros(n_neurons, dtype=np.int32)

            wiring_arr = wiring_variants.get(vid)
            if wiring_arr is not None and len(wiring_arr) > 0:
                wdf  = pd.DataFrame(wiring_arr)
                tgts = wdf['tgt'].values.astype(int)
                srcs = wdf['src'].values.astype(int)
                np.add.at(in_degree,  tgts[(tgts >= 0) & (tgts < n_neurons)], 1)
                np.add.at(out_degree, srcs[(srcs >= 0) & (srcs < n_neurons)], 1)

            new_arr = np.zeros(n_neurons, dtype=[
                ('tonic_activation', 'f4'),
                ('in_degree',        'i4'),
                ('out_degree',       'i4'),
            ])
            if n_neurons > 0:
                new_arr['tonic_activation'] = np.asarray(tonic_arr, dtype='f4')
            new_arr['in_degree']  = in_degree
            new_arr['out_degree'] = out_degree
            variants[vid] = new_arr


def plot_heatmap_overview() -> None:
    """
    For each group in heatmap_data, generate a composite figure with up to three panels
    (first / middle / last variant selected via linspace).  Each panel contains:
      - one average heatmap (square) + colorbar, centered on top
      - four individual run heatmaps side-by-side, centered below
    All panels use a fixed size; empty space is left when fewer than three variants exist.
    One composite image per group is saved and inserted into the report.
    """
    if not heatmap_data:
        return

    figures_dir = Path(__file__).resolve().parent / f'figures_{EXPERIMENT_NAME}'
    figures_dir.mkdir(exist_ok=True)

    # --- Layout constants (all in inches) ---
    N_PANELS  = 3       # always 3 panel columns (empty for missing variants)
    N_RUNS    = 4       # individual run cells per panel

    RUN_CELL  = 2.0     # each individual run plot: RUN_CELL × RUN_CELL inches (square)
    GAP       = 30 / 150            # ~30 px gap at 150 dpi ≈ 0.20 in
    PANEL_GAP = GAP                 # horizontal gap between variant panels
    PANEL_W   = N_RUNS * RUN_CELL + (N_RUNS - 1) * GAP   # runs row total width incl. gaps
    AVG_SIZE  = PANEL_W * 0.80             # avg heatmap square side
    CBAR_W    = 0.22                        # colorbar width
    CBAR_PAD  = 0.10                        # gap between avg heatmap and colorbar
    AVG_SEC_W = AVG_SIZE + CBAR_PAD + CBAR_W  # total width of avg + colorbar section
    AVG_X_OFF = (PANEL_W - AVG_SEC_W) / 2  # horizontal offset to center avg section

    TITLE_H   = 0.90    # space above avg heatmap for title (tripled font needs more room)
    LABEL_H   = 0.55    # space below run plots for run labels
    GAP_V     = GAP     # vertical gap between runs row and avg plot

    fig_w = N_PANELS * PANEL_W + (N_PANELS - 1) * PANEL_GAP
    fig_h = TITLE_H + AVG_SIZE + GAP_V + RUN_CELL + LABEL_H

    # Normalised y-coordinates (bottom-up):
    run_b   = LABEL_H / fig_h
    avg_b   = (LABEL_H + RUN_CELL + GAP_V) / fig_h
    run_h_n = RUN_CELL / fig_h
    avg_h_n = AVG_SIZE / fig_h
    run_w_n = RUN_CELL / fig_w
    avg_w_n = AVG_SIZE / fig_w
    cbar_w_n = CBAR_W  / fig_w
    cbar_h_n = AVG_SIZE / fig_h

    for group_name, group_variants in sorted(heatmap_data.items()):
        all_vids = sorted(group_variants.keys())
        n_avail  = len(all_vids)
        if n_avail == 0:
            continue

        # Linspace selection: first, floor-middle, last (deduplicated, order preserved)
        if n_avail == 1:
            selected_vids = [all_vids[0]]
        elif n_avail == 2:
            selected_vids = [all_vids[0], all_vids[-1]]
        else:
            mid_idx = int(np.floor((n_avail - 1) / 2))
            selected_vids = [all_vids[0], all_vids[mid_idx], all_vids[-1]]

        fig = plt.figure(figsize=(fig_w, fig_h))

        for pi, vid in enumerate(selected_vids):
            runs_dict = group_variants[vid]
            run_ids   = sorted(runs_dict.keys())

            panel_x0 = pi * (PANEL_W + PANEL_GAP)   # inches from figure left

            # --- average heatmap (square, centered above runs row) ---
            stack   = np.stack([runs_dict[r].astype(float) for r in run_ids], axis=0)
            avg_map = stack.mean(axis=0)

            avg_l = (panel_x0 + AVG_X_OFF) / fig_w
            ax_avg = fig.add_axes([avg_l, avg_b, avg_w_n, avg_h_n])
            im_avg = ax_avg.imshow(avg_map, origin='upper', aspect='equal',
                                   cmap='hot', interpolation='nearest')
            ax_avg.set_title(f'Average — variant {vid}\n({group_name})',
                             fontsize=24, pad=6)
            ax_avg.axis('off')

            # colorbar flush against the avg heatmap
            cbar_l = avg_l + avg_w_n + CBAR_PAD / fig_w
            ax_cb  = fig.add_axes([cbar_l, avg_b, cbar_w_n, cbar_h_n])
            fig.colorbar(im_avg, cax=ax_cb)
            ax_cb.tick_params(labelsize=18)

            # --- individual run heatmaps (small gap between cells) ---
            for ri in range(N_RUNS):
                run_l  = (panel_x0 + ri * (RUN_CELL + GAP)) / fig_w
                ax_run = fig.add_axes([run_l, run_b, run_w_n, run_h_n])
                if ri < len(run_ids):
                    rid = run_ids[ri]
                    ax_run.imshow(runs_dict[rid].astype(float), origin='upper',
                                  aspect='equal', cmap='hot', interpolation='nearest')
                    ax_run.set_xlabel(f'run {rid}', fontsize=21)
                    ax_run.set_xticks([])
                    ax_run.set_yticks([])
                else:
                    ax_run.axis('off')

        # Save and insert into report
        safe_group = group_name.replace(' ', '_').replace('/', '-')
        fig_path = figures_dir / f'heatmap_{safe_group}.png'
        fig.savefig(str(fig_path), dpi=150)
        plt.close(fig)

        doc.add_heading(f'Heatmaps — {group_name}', level=3)
        doc.add_picture(str(fig_path), width=6.5 * 914400)
        doc.add_paragraph()


def analyze_foods_consumed_per_direction(df_per_tick: pd.DataFrame, group_color_map: dict = None) -> None:
    """
    Analyse the direction from which the worm approached food before each consumption event.

    Each row where food_consumed == 1 marks a consumption event; the 'movement' column on
    that row is always 'stay'.  The function temporarily shifts 'movement' forward by one
    tick within each (group, variant, run) so that the preceding movement aligns with the
    consumption row.  It then counts how often N / E / S / W was the preceding move, per run.

    Plot: jitter + box-plot, x-axis grouped by direction, hue = group.

    Statistics: per-direction one-way tests (ANOVA if normally distributed, Kruskal-Wallis
    otherwise) with Holm-Bonferroni correction across the four directions, plus pairwise
    post-hoc for significant directions.

    Note on two-way ANOVA: the fully correct omnibus test would be a two-way ANOVA
    (group × direction) or its non-parametric equivalent, the Scheirer-Ray-Hare test.
    Both require statsmodels / pingouin (not in this project).  The approach below —
    per-direction one-way tests with Holm correction — provides equivalent interpretable
    information at the cost of not having a single interaction p-value.

    Args:
        df_per_tick:     Per-tick tracking DataFrame (global df_per_tick).
        group_color_map: Dict mapping group name -> hex colour string.
    """
    _DIRS = ['N', 'E', 'S', 'W']
    _gcm  = group_color_map or {}

    def _color_for(g: str) -> str:
        return _gcm.get(g, '#808080')

    # ── 1.  Build per-run direction counts ──────────────────────────────────────
    # Shift movement within each run so that consumption rows show preceding move.
    _tmp = df_per_tick[['group', 'variant', 'run', 'tick', 'movement', 'food_consumed']].copy()
    # movement is stored as bytes in HDF5 — decode to str if needed
    if _tmp['movement'].dtype == object and len(_tmp) > 0 and isinstance(_tmp['movement'].iloc[0], bytes):
        _tmp['movement'] = _tmp['movement'].str.decode('utf-8')
    _tmp['movement_prev'] = (
        _tmp.groupby(['group', 'variant', 'run'])['movement']
        .shift(1)
    )
    _consumed = _tmp[_tmp['food_consumed'] == 1].copy()

    # Count per direction per run
    _rows = []
    for (_grp, _var, _run), _grp_df in _consumed.groupby(['group', 'variant', 'run']):
        for _d in _DIRS:
            _rows.append({
                'group':     _grp,
                'variant':   _var,
                'run':       _run,
                'direction': _d,
                'count':     int((_grp_df['movement_prev'] == _d).sum()),
            })
    _df_counts = pd.DataFrame(_rows)

    all_groups, zorder_map = _get_ordered_groups_with_zorder(_df_counts['group'].unique())
    n_groups   = len(all_groups)

    # ── 2.  Plot ─────────────────────────────────────────────────────────────────
    figures_dir = Path(__file__).resolve().parent / f'figures_{EXPERIMENT_NAME}'
    figures_dir.mkdir(exist_ok=True)

    # x positions: groups in blocks, directions side-by-side within each group block
    _dir_colors  = {'N': '#4e79a7', 'E': '#f28e2b', 'S': '#59a14f', 'W': '#e15759'}
    _dir_width   = 0.7 / len(_DIRS)
    _block_gap   = 1.0
    _x_positions  = {}   # (group, direction) -> x
    _block_centers = {}
    for _gi, _g in enumerate(all_groups):
        _block_start = _gi * (_block_gap + 0.7)
        _block_centers[_g] = _block_start + 0.35 - _dir_width / 2
        for _di, _d in enumerate(_DIRS):
            _x_positions[(_g, _d)] = _block_start + _di * _dir_width

    fig, ax = plt.subplots(figsize=(12, 7))
    _rng = np.random.default_rng(42)

    for _g in all_groups:
        for _d in _DIRS:
            _vals = _df_counts[(_df_counts['group'] == _g) & (_df_counts['direction'] == _d)]['count'].values
            _x    = _x_positions[(_g, _d)]
            _col  = _dir_colors[_d]
            # jitter
            _jit = _rng.uniform(-_dir_width * 0.3, _dir_width * 0.3, size=len(_vals))
            ax.scatter(_x + _jit, _vals, color=_col, alpha=0.4, s=10, zorder=1)
            # box
            if len(_vals) >= 2:
                _bp = ax.boxplot(
                    _vals,
                    positions=[_x],
                    widths=_dir_width * 0.8,
                    patch_artist=True,
                    showfliers=False,
                    manage_ticks=False,
                )
                _bp['boxes'][0].set_facecolor(_col)
                _bp['boxes'][0].set_alpha(0.35)
                _bp['medians'][0].set_color('black')

    # Legend: directions
    _legend_handles = [mpatches.Patch(color=_dir_colors[_d], label=_d) for _d in _DIRS]
    ax.legend(handles=_legend_handles, loc='upper right', fontsize=10, title='Direction')

    # x-axis ticks at group block centres
    ax.set_xticks([_block_centers[_g] for _g in all_groups])
    ax.set_xticklabels(all_groups, fontsize=11)
    ax.set_ylabel('Consumption Events per Run', fontsize=12)
    ax.set_title('Food Consumed — Preceding Direction of Approach (by Group)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    _fig_path = figures_dir / 'food_direction_counts.png'
    fig.tight_layout()
    _ymin, _ymax = ax.get_ylim()
    ax.set_ylim(_ymin, _ymax + (_ymax - _ymin) / 3)
    fig.savefig(str(_fig_path), dpi=150)
    plt.close(fig)
    doc.add_picture(str(_fig_path), width=6.5 * 914400)
    doc.add_paragraph()

    # ── 3.  Statistics: per-group one-way tests across directions ────────────────
    doc.add_heading('Statistical Comparison — Directional Preference per Group', level=3)
    doc.add_paragraph(
        'For each group, the four directions (N/E/S/W) are compared as levels. '
        'One-way ANOVA (normally distributed data) or Kruskal-Wallis (non-normal) tests '
        'whether the group shows a directional preference. Pairwise post-hoc tests with '
        'Tukey\'s HSD (normal) or Holm-Bonferroni (non-normal) correction are reported for '
        'significant omnibus results.'
    )

    for _g in all_groups:
        doc.add_heading(f'Group: {_g}', level=4)

        # Arrays: one per direction, values = per-run counts for this group
        _arrs = [
            _df_counts[(_df_counts['group'] == _g) & (_df_counts['direction'] == _d)]['count'].values
            for _d in _DIRS
        ]


        # ── Normality testing (Shapiro-Wilk) ─────────────────────────────────
        _norm_results = {}
        _all_normal = True
        for _d, _v in zip(_DIRS, _arrs):
            if len(_v) < 3 or np.ptp(_v) == 0:
                _norm_results[_d] = (np.nan, np.nan)
                _all_normal = False
            else:
                _w_sw, _p_sw = shapiro(_v)
                _norm_results[_d] = (_w_sw, _p_sw)
                if _p_sw < 0.05:
                    _all_normal = False

        doc.add_paragraph('Normality Testing (Shapiro-Wilk):', style='Heading 5')
        _norm_tbl = doc.add_table(rows=len(_DIRS) + 1, cols=3)
        _norm_tbl.style = 'Light Grid Accent 1'
        _norm_tbl.rows[0].cells[0].text = 'Direction'
        _norm_tbl.rows[0].cells[1].text = 'W statistic'
        _norm_tbl.rows[0].cells[2].text = 'p-value'
        for _ri, _d in enumerate(_DIRS, 1):
            _w_sw, _p_sw = _norm_results[_d]
            _norm_tbl.rows[_ri].cells[0].text = _d
            _norm_tbl.rows[_ri].cells[1].text = f'{_w_sw:.4f}' if not np.isnan(_w_sw) else 'N/A'
            _norm_tbl.rows[_ri].cells[2].text = f'{_p_sw:.2e}' if not np.isnan(_p_sw) else 'N/A'

        _dist_str = 'normally distributed' if _all_normal else 'not normally distributed'
        doc.add_paragraph(f'Data is {_dist_str} (alpha=0.05).')

        # ── Omnibus test ──────────────────────────────────────────────────────
        if _all_normal:
            _stat, _p = f_oneway(*_arrs)
            _test_name    = 'One-way ANOVA'
            _grand        = np.mean(np.concatenate(_arrs))
            _ss_b         = sum(len(a) * (np.mean(a) - _grand) ** 2 for a in _arrs)
            _ss_t         = sum(np.sum((a - _grand) ** 2) for a in _arrs)
            _es_omni      = _ss_b / _ss_t if _ss_t > 0 else 0.0
            _es_type_omni = 'Eta-squared'
            _es_col_omni  = 'η²'
        else:
            _all_combined = np.concatenate(_arrs)
            if np.ptp(_all_combined) == 0:
                _stat, _p = 0.0, 1.0
            else:
                _stat, _p = kruskal(*_arrs)
            _test_name    = 'Kruskal-Wallis'
            _N            = len(_all_combined)
            _k            = len(_arrs)
            _es_omni      = (_stat - _k + 1) / (_N - _k) if (_N - _k) > 0 else 0.0
            _es_type_omni = 'Epsilon-squared'
            _es_col_omni  = 'ε²'

        doc.add_paragraph(f'{_test_name} (omnibus test):', style='Heading 5')
        _omni_tbl = doc.add_table(rows=2, cols=4)
        _omni_tbl.style = 'Light Grid Accent 1'
        _oh = _omni_tbl.rows[0].cells
        _oh[0].text = 'Test Name'
        _oh[1].text = 'Test Statistic'
        _oh[2].text = 'p-value'
        _oh[3].text = _es_col_omni
        _ov = _omni_tbl.rows[1].cells
        _ov[0].text = _test_name
        _ov[1].text = f'{_stat:.4f}'
        _ov[2].text = f'{_p:.2e}' + (' *' if _p < 0.05 else '')
        _ov[3].text = f'{_es_omni:.4f} ({classify_effect_size(_es_omni, _es_type_omni)})'

        if _p >= 0.05:
            doc.add_paragraph('Omnibus test not significant (p ≥ 0.05). No post-hoc testing performed.')
            continue

        # ── Pairwise post-hoc ─────────────────────────────────────────────────
        _k_dirs = len(_DIRS)  # 4
        _N_dirs = sum(len(a) for a in _arrs)
        _df_err = _N_dirs - _k_dirs

        _ph_raw = []
        for (_di, _da), (_dj, _db) in combinations(enumerate(_DIRS), 2):
            _a, _b = _arrs[_di], _arrs[_dj]
            if _all_normal:
                _s, _pp = ttest_ind(_a, _b, equal_var=False)
                _n0, _n1 = len(_a), len(_b)
                _v0, _v1 = np.var(_a, ddof=1), np.var(_b, ddof=1)
                _ps = np.sqrt(((_n0-1)*_v0 + (_n1-1)*_v1) / (_n0+_n1-2))
                _es = (np.mean(_a) - np.mean(_b)) / _ps if _ps > 0 else 0.0
                _ph_es_type  = "Cohen's d"
                _ph_es_label = "Cohen's d"
            else:
                _s, _pp = mannwhitneyu(_a, _b, alternative='two-sided')
                _es = 1 - (2 * _s) / (len(_a) * len(_b))
                _ph_es_type  = 'Rank-Biserial r'
                _ph_es_label = 'Rank-Biserial r'
            _ph_raw.append((_da, _db, _s, _pp, _es))

        if _all_normal:
            _ph_rows = []
            for (_da, _db, _s, _pp, _es) in _ph_raw:
                _cp = float(studentized_range.sf(abs(_s), _k_dirs, _df_err))
                _cp = min(_cp, 1.0)
                _ph_rows.append((_da, _db, _s, _cp, _es))
            _ph_name = "Welch's t-test (Tukey's HSD correction)"
        else:
            _ph_rej, _ph_pc, _, _ = multipletests([x[3] for x in _ph_raw], method='holm')
            _ph_rows = [(_da, _db, _s, _pc, _es)
                        for (_da, _db, _s, _, _es), _pc in zip(_ph_raw, _ph_pc)]
            _ph_name = 'Mann-Whitney U (Holm-Bonferroni correction)'

        doc.add_paragraph(f'Pairwise Post-Hoc ({_ph_name}):', style='Heading 5')
        _ph_tbl = doc.add_table(rows=len(_ph_rows) + 1, cols=5)
        _ph_tbl.style = 'Light Grid Accent 1'
        _ph_h = _ph_tbl.rows[0].cells
        _ph_h[0].text = 'Direction A'
        _ph_h[1].text = 'Direction B'
        _ph_h[2].text = 'Statistic'
        _ph_h[3].text = 'p-value'
        _ph_h[4].text = _ph_es_label
        for _ri, (_da, _db, _s, _pp, _es) in enumerate(_ph_rows, 1):
            _c = _ph_tbl.rows[_ri].cells
            _c[0].text = _da
            _c[1].text = _db
            _c[2].text = f'{_s:.4f}'
            _c[3].text = f'{_pp:.2e}' + (' *' if _pp < 0.05 else '')
            _c[4].text = f'{_es:.4f} ({classify_effect_size(_es, _ph_es_type)})'

    # ── 4.  Normalised deviation plot — per variant ──────────────────────────────
    # For each variant: pool all runs, sum direction counts, compute % deviation
    # from expected (variant_total / 4).  One data point per (group, variant, direction).

    # x-position layout: same block structure as the raw-counts plot above
    _x_pos2 = {}
    _block_ctrs2 = {}
    for _gi, _g in enumerate(all_groups):
        _bs = _gi * (_block_gap + 0.7)
        _block_ctrs2[_g] = _bs + 0.35 - _dir_width / 2
        for _di, _d in enumerate(_DIRS):
            _x_pos2[(_g, _d)] = _bs + _di * _dir_width

    # Sum direction counts across all runs within each (group, variant)
    _df_var = (
        _df_counts
        .groupby(['group', 'variant', 'direction'])['count']
        .sum()
        .reset_index()
    )
    _df_var_pivot = _df_var.pivot_table(
        index=['group', 'variant'],
        columns='direction',
        values='count',
        fill_value=0,
    ).reset_index()
    _df_var_pivot['_total'] = _df_var_pivot[_DIRS].sum(axis=1)

    _dev_var_rows = []
    for _, _row in _df_var_pivot.iterrows():
        _tot = _row['_total']
        if _tot == 0:
            continue
        _expected = _tot / 4.0
        for _d in _DIRS:
            _dev_var_rows.append({
                'group':     _row['group'],
                'variant':   _row['variant'],
                'direction': _d,
                'deviation': (_row[_d] - _expected) / _expected * 100.0,
            })
    _df_dev_var = pd.DataFrame(_dev_var_rows)

    fig3, ax3 = plt.subplots(figsize=(12, 7))
    _rng3 = np.random.default_rng(42)

    for _g in all_groups:
        for _d in _DIRS:
            _mask = (_df_dev_var['group'] == _g) & (_df_dev_var['direction'] == _d)
            _vals = _df_dev_var.loc[_mask, 'deviation'].values
            _x    = _x_pos2[(_g, _d)]   # same layout as first plot
            _col  = _dir_colors[_d]

            if len(_vals) == 0:
                continue

            # Jitter first (lower z-order), then bar on top
            _jit = _rng3.uniform(-_dir_width * 0.3, _dir_width * 0.3, size=len(_vals))
            ax3.scatter(_x + _jit, _vals, color=_col, alpha=0.5, s=14, zorder=1)

            _mean_dev = float(np.mean(_vals))
            ax3.bar(
                _x, _mean_dev,
                width=_dir_width * 0.85,
                color=_col, alpha=0.45,
                zorder=2,
            )

    ax3.axhline(0, color='black', linewidth=0.8, linestyle='--', zorder=0)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'+{v:.0f}%' if v > 0 else f'{v:.0f}%'))
    ax3.legend(handles=_legend_handles, loc='upper right', fontsize=10, title='Direction')
    ax3.set_xticks([_block_ctrs2[_g] for _g in all_groups])
    ax3.set_xticklabels(all_groups, fontsize=11)
    ax3.set_ylabel('Deviation from Expected Frequency (%)', fontsize=12)
    ax3.set_title('Directional Approach Bias — Deviation from Equal Distribution, pooled per Variant (by Group)', fontsize=14)
    ax3.grid(True, alpha=0.3, axis='y')

    _fig3_path = figures_dir / 'food_direction_deviation_per_variant.png'
    fig3.tight_layout()
    # Add 2x padding AFTER tight_layout, then save (without bbox_inches='tight' which would crop it)
    _ymin, _ymax = ax3.get_ylim()
    _yrange = _ymax - _ymin
    ax3.set_ylim(_ymin, _ymax + _yrange / 3)
    fig3.savefig(str(_fig3_path), dpi=150)
    plt.close(fig3)

    doc.add_heading('Directional Approach Bias (% Deviation from Expected) — per Variant', level=3)
    doc.add_paragraph(
        'For each variant, all runs are pooled and the total food count per direction is summed. '
        'The expected count per direction is the variant\'s total food count ÷ 4. '
        'Bars show the mean deviation across variants; jitter shows individual variants.'
    )
    doc.add_picture(str(_fig3_path), width=6.5 * 914400)
    doc.add_paragraph()


def analyze_movements_per_direction(df_per_tick: pd.DataFrame, group_color_map: dict = None) -> None:
    """
    Analyse movement patterns: count and compare directional preferences (N/E/S/W only).

    Steps:
      1. Count occurrences of each movement type (N/E/S/W) per run
      2. Plot: jitter + box-plot, x-axis grouped by direction (4 directions), hue = group
      3. Statistics: per-group one-way tests across all 4 directions
      4. Directional bias plot: N/E/S/W, deviation from expected (total/4)

    Args:
        df_per_tick:     Per-tick tracking DataFrame (global df_per_tick).
        group_color_map: Dict mapping group name -> hex colour string.
    """
    _DIRS = ['N', 'E', 'S', 'W']
    _gcm  = group_color_map or {}

    def _color_for(g: str) -> str:
        return _gcm.get(g, '#808080')

    if df_per_tick is None or len(df_per_tick) == 0:
        return

    figures_dir = Path(__file__).resolve().parent / f'figures_{EXPERIMENT_NAME}'
    figures_dir.mkdir(exist_ok=True)

    all_groups, zorder_map = _get_ordered_groups_with_zorder(df_per_tick['group'].unique())

    doc.add_heading('Movement Analysis — Directional Preferences', level=2)

    # ── 1.  Build per-run direction counts ──────────────────────────────────────
    doc.add_heading('1. Directional Movement Distribution', level=3)

    _tmp = df_per_tick[['group', 'variant', 'run', 'movement']].copy()
    # Decode movement to str if needed
    if _tmp['movement'].dtype == object and len(_tmp) > 0 and isinstance(_tmp['movement'].iloc[0], bytes):
        _tmp['movement'] = _tmp['movement'].str.decode('utf-8')

    # Count per direction per run
    _rows = []
    for (_grp, _var, _run), _grp_df in _tmp.groupby(['group', 'variant', 'run']):
        for _d in _DIRS:
            _rows.append({
                'group':     _grp,
                'variant':   _var,
                'run':       _run,
                'direction': _d,
                'count':     int((_grp_df['movement'] == _d).sum()),
            })
    _df_counts = pd.DataFrame(_rows)

    n_groups = len(all_groups)

    # ── 3.  Plot ─────────────────────────────────────────────────────────────────
    # x positions: groups in blocks, directions side-by-side within each group block
    _dir_colors  = {'N': '#4e79a7', 'E': '#f28e2b', 'S': '#59a14f', 'W': '#e15759'}
    _dir_width   = 0.7 / len(_DIRS)
    _block_gap   = 1.0
    _x_positions  = {}   # (group, direction) -> x
    _block_centers = {}
    for _gi, _g in enumerate(all_groups):
        _block_start = _gi * (_block_gap + 0.7)
        _block_centers[_g] = _block_start + 0.35 - _dir_width / 2
        for _di, _d in enumerate(_DIRS):
            _x_positions[(_g, _d)] = _block_start + _di * _dir_width

    fig, ax = plt.subplots(figsize=(12, 7))
    _rng = np.random.default_rng(42)

    for _g in all_groups:
        for _d in _DIRS:
            _vals = _df_counts[(_df_counts['group'] == _g) & (_df_counts['direction'] == _d)]['count'].values
            _x    = _x_positions[(_g, _d)]
            _col  = _dir_colors[_d]
            # jitter
            _jit = _rng.uniform(-_dir_width * 0.3, _dir_width * 0.3, size=len(_vals))
            ax.scatter(_x + _jit, _vals, color=_col, alpha=0.4, s=10, zorder=1)
            # box
            if len(_vals) >= 2:
                _bp = ax.boxplot(
                    _vals,
                    positions=[_x],
                    widths=_dir_width * 0.8,
                    patch_artist=True,
                    showfliers=False,
                    manage_ticks=False,
                )
                _bp['boxes'][0].set_facecolor(_col)
                _bp['boxes'][0].set_alpha(0.35)
                _bp['medians'][0].set_color('black')

    # Legend: directions
    _legend_handles = [mpatches.Patch(color=_dir_colors[_d], label=_d) for _d in _DIRS]
    ax.legend(handles=_legend_handles, loc='upper right', fontsize=10, title='Direction')

    # x-axis ticks at group block centres
    ax.set_xticks([_block_centers[_g] for _g in all_groups])
    ax.set_xticklabels(all_groups, fontsize=11)
    ax.set_ylabel('Movement Count per Run', fontsize=12)
    ax.set_title('Movement Distribution by Direction (by Group)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    _fig_path = figures_dir / 'movement_direction_counts.png'
    fig.tight_layout()
    _ymin, _ymax = ax.get_ylim()
    ax.set_ylim(_ymin, _ymax + (_ymax - _ymin) / 3)
    fig.savefig(str(_fig_path), dpi=150)
    plt.close(fig)
    doc.add_picture(str(_fig_path), width=6.5 * 914400)
    doc.add_paragraph()

    # ── 2.  Statistics: per-group one-way tests across all directions ──────────────────
    doc.add_heading('3. Statistical Comparison — Directional Preference per Group', level=3)
    doc.add_paragraph(
        'For each group, the five movement types (N/E/S/W/stay) are compared as levels. '
        'One-way ANOVA (normally distributed data) or Kruskal-Wallis (non-normal) tests '
        'whether the group shows a directional preference. Pairwise post-hoc tests with '
        'Tukey\'s HSD (normal) or Holm-Bonferroni (non-normal) correction are reported for '
        'significant omnibus results.'
    )

    for _g in all_groups:
        doc.add_heading(f'Group: {_g}', level=4)

        # Arrays: one per direction, values = per-run counts for this group
        _arrs = [
            _df_counts[(_df_counts['group'] == _g) & (_df_counts['direction'] == _d)]['count'].values
            for _d in _DIRS
        ]

        # ── Normality testing (Shapiro-Wilk) ─────────────────────────────────
        _norm_results = {}
        _all_normal = True
        for _d, _v in zip(_DIRS, _arrs):
            if len(_v) < 3 or np.ptp(_v) == 0:
                _norm_results[_d] = (np.nan, np.nan)
                _all_normal = False
            else:
                _w_sw, _p_sw = shapiro(_v)
                _norm_results[_d] = (_w_sw, _p_sw)
                if _p_sw < 0.05:
                    _all_normal = False

        doc.add_paragraph('Normality Testing (Shapiro-Wilk):', style='Heading 5')
        _norm_tbl = doc.add_table(rows=len(_DIRS) + 1, cols=3)
        _norm_tbl.style = 'Light Grid Accent 1'
        _norm_tbl.rows[0].cells[0].text = 'Direction'
        _norm_tbl.rows[0].cells[1].text = 'W statistic'
        _norm_tbl.rows[0].cells[2].text = 'p-value'
        for _ri, _d in enumerate(_DIRS, 1):
            _w_sw, _p_sw = _norm_results[_d]
            _norm_tbl.rows[_ri].cells[0].text = _d
            _norm_tbl.rows[_ri].cells[1].text = f'{_w_sw:.4f}' if not np.isnan(_w_sw) else 'N/A'
            _norm_tbl.rows[_ri].cells[2].text = f'{_p_sw:.2e}' if not np.isnan(_p_sw) else 'N/A'

        _dist_str = 'normally distributed' if _all_normal else 'not normally distributed'
        doc.add_paragraph(f'Data is {_dist_str} (alpha=0.05).')

        # ── Omnibus test ──────────────────────────────────────────────────────
        if _all_normal:
            _stat, _p = f_oneway(*_arrs)
            _test_name    = 'One-way ANOVA'
            _grand        = np.mean(np.concatenate(_arrs))
            _ss_b         = sum(len(a) * (np.mean(a) - _grand) ** 2 for a in _arrs)
            _ss_t         = sum(np.sum((a - _grand) ** 2) for a in _arrs)
            _es_omni      = _ss_b / _ss_t if _ss_t > 0 else 0.0
            _es_type_omni = 'Eta-squared'
            _es_col_omni  = 'η²'
        else:
            _all_combined = np.concatenate(_arrs)
            if np.ptp(_all_combined) == 0:
                _stat, _p = 0.0, 1.0
            else:
                _stat, _p = kruskal(*_arrs)
            _test_name    = 'Kruskal-Wallis'
            _N            = len(_all_combined)
            _k            = len(_arrs)
            _es_omni      = (_stat - _k + 1) / (_N - _k) if (_N - _k) > 0 else 0.0
            _es_type_omni = 'Epsilon-squared'
            _es_col_omni  = 'ε²'

        doc.add_paragraph(f'{_test_name} (omnibus test):', style='Heading 5')
        _omni_tbl = doc.add_table(rows=2, cols=4)
        _omni_tbl.style = 'Light Grid Accent 1'
        _oh = _omni_tbl.rows[0].cells
        _oh[0].text = 'Test Name'
        _oh[1].text = 'Test Statistic'
        _oh[2].text = 'p-value'
        _oh[3].text = _es_col_omni
        _ov = _omni_tbl.rows[1].cells
        _ov[0].text = _test_name
        _ov[1].text = f'{_stat:.4f}'
        _ov[2].text = f'{_p:.2e}' + (' *' if _p < 0.05 else '')
        _ov[3].text = f'{_es_omni:.4f} ({classify_effect_size(_es_omni, _es_type_omni)})'

        if _p >= 0.05:
            doc.add_paragraph('Omnibus test not significant (p ≥ 0.05). No post-hoc testing performed.')
            continue

        # ── Pairwise post-hoc ─────────────────────────────────────────────────
        _k_dirs = len(_DIRS)  # 5
        _N_dirs = sum(len(a) for a in _arrs)
        _df_err = _N_dirs - _k_dirs

        _ph_raw = []
        for (_di, _da), (_dj, _db) in combinations(enumerate(_DIRS), 2):
            _a, _b = _arrs[_di], _arrs[_dj]
            if _all_normal:
                _s, _pp = ttest_ind(_a, _b, equal_var=False)
                _n0, _n1 = len(_a), len(_b)
                _v0, _v1 = np.var(_a, ddof=1), np.var(_b, ddof=1)
                _ps = np.sqrt(((_n0-1)*_v0 + (_n1-1)*_v1) / (_n0+_n1-2))
                _es = (np.mean(_a) - np.mean(_b)) / _ps if _ps > 0 else 0.0
                _ph_es_type  = "Cohen's d"
                _ph_es_label = "Cohen's d"
            else:
                _s, _pp = mannwhitneyu(_a, _b, alternative='two-sided')
                _es = 1 - (2 * _s) / (len(_a) * len(_b))
                _ph_es_type  = 'Rank-Biserial r'
                _ph_es_label = 'Rank-Biserial r'
            _ph_raw.append((_da, _db, _s, _pp, _es))

        if _all_normal:
            _ph_rows = []
            for (_da, _db, _s, _pp, _es) in _ph_raw:
                _cp = float(studentized_range.sf(abs(_s), _k_dirs, _df_err))
                _cp = min(_cp, 1.0)
                _ph_rows.append((_da, _db, _s, _cp, _es))
            _ph_name = "Welch's t-test (Tukey's HSD correction)"
        else:
            _ph_rej, _ph_pc, _, _ = multipletests([x[3] for x in _ph_raw], method='holm')
            _ph_rows = [(_da, _db, _s, _pc, _es)
                        for (_da, _db, _s, _, _es), _pc in zip(_ph_raw, _ph_pc)]
            _ph_name = 'Mann-Whitney U (Holm-Bonferroni correction)'

        doc.add_paragraph(f'Pairwise Post-Hoc ({_ph_name}):', style='Heading 5')
        _ph_tbl = doc.add_table(rows=len(_ph_rows) + 1, cols=5)
        _ph_tbl.style = 'Light Grid Accent 1'
        _ph_h = _ph_tbl.rows[0].cells
        _ph_h[0].text = 'Movement A'
        _ph_h[1].text = 'Movement B'
        _ph_h[2].text = 'Statistic'
        _ph_h[3].text = 'p-value'
        _ph_h[4].text = _ph_es_label
        for _ri, (_da, _db, _s, _pp, _es) in enumerate(_ph_rows, 1):
            _c = _ph_tbl.rows[_ri].cells
            _c[0].text = _da
            _c[1].text = _db
            _c[2].text = f'{_s:.4f}'
            _c[3].text = f'{_pp:.2e}' + (' *' if _pp < 0.05 else '')
            _c[4].text = f'{_es:.4f} ({classify_effect_size(_es, _ph_es_type)})'

    # ── 3.  Directional bias plot — per variant (excluding 'stay') ─────────────
    # For each variant: pool all runs, sum direction counts (only N/E/S/W),
    # compute % deviation from expected (variant_total_motion / 4).

    doc.add_heading('4. Directional Motion Bias (excluding "stay")', level=3)

    # x-position layout: same block structure as the raw-counts plot above
    _x_pos2 = {}
    _block_ctrs2 = {}
    for _gi, _g in enumerate(all_groups):
        _bs = _gi * (_block_gap + 0.7)
        _block_ctrs2[_g] = _bs + 0.35 - (_dir_width * 4) / 2  # center for 4 directions
        for _di, _d in enumerate(_DIRS):
            _x_pos2[(_g, _d)] = _bs + _di * _dir_width

    # Sum direction counts across all runs within each (group, variant)
    _df_var = (
        _df_counts.groupby(['group', 'variant', 'direction'])['count']
        .sum()
        .reset_index()
    )
    _df_var_pivot = _df_var.pivot_table(
        index=['group', 'variant'],
        columns='direction',
        values='count',
        fill_value=0,
    ).reset_index()
    _df_var_pivot['_total'] = _df_var_pivot[_DIRS].sum(axis=1)

    _dev_var_rows = []
    for _, _row in _df_var_pivot.iterrows():
        _tot = _row['_total']
        if _tot == 0:
            continue
        _expected = _tot / 4.0
        for _d in _DIRS:
            _dev_var_rows.append({
                'group':     _row['group'],
                'variant':   _row['variant'],
                'direction': _d,
                'deviation': (_row[_d] - _expected) / _expected * 100.0,
            })
    _df_dev_var = pd.DataFrame(_dev_var_rows)

    fig3, ax3 = plt.subplots(figsize=(12, 7))
    _rng3 = np.random.default_rng(42)

    for _g in all_groups:
        for _d in _DIRS:
            _mask = (_df_dev_var['group'] == _g) & (_df_dev_var['direction'] == _d)
            _vals = _df_dev_var.loc[_mask, 'deviation'].values
            _x    = _x_pos2[(_g, _d)]
            _col  = _dir_colors[_d]

            if len(_vals) == 0:
                continue

            # Jitter first (lower z-order), then bar on top
            _jit = _rng3.uniform(-_dir_width * 0.3, _dir_width * 0.3, size=len(_vals))
            ax3.scatter(_x + _jit, _vals, color=_col, alpha=0.5, s=14, zorder=1)

            _mean_dev = float(np.mean(_vals))
            ax3.bar(
                _x, _mean_dev,
                width=_dir_width * 0.85,
                color=_col, alpha=0.45,
                zorder=2,
            )

    ax3.axhline(0, color='black', linewidth=0.8, linestyle='--', zorder=0)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'+{v:.0f}%' if v > 0 else f'{v:.0f}%'))
    _legend_handles = [mpatches.Patch(color=_dir_colors[_d], label=_d) for _d in _DIRS]
    ax3.legend(handles=_legend_handles, loc='upper right', fontsize=10, title='Direction')
    ax3.set_xticks([_block_ctrs2[_g] for _g in all_groups])
    ax3.set_xticklabels(all_groups, fontsize=11)
    ax3.set_ylabel('Deviation from Expected Frequency (%)', fontsize=12)
    ax3.set_title('Directional Motion Bias — Deviation from Equal Distribution, pooled per Variant (by Group)', fontsize=14)
    ax3.grid(True, alpha=0.3, axis='y')

    _fig3_path = figures_dir / 'movement_direction_deviation_per_variant.png'
    fig3.tight_layout()
    # Add 2x padding AFTER tight_layout, then save (without bbox_inches='tight' which would crop it)
    _ymin, _ymax = ax3.get_ylim()
    _yrange = _ymax - _ymin
    ax3.set_ylim(_ymin, _ymax + _yrange / 3)
    fig3.savefig(str(_fig3_path), dpi=150)
    plt.close(fig3)

    doc.add_paragraph(
        'For each variant, all runs are pooled and the total movement count per direction is summed. '
        'The expected count per direction is the variant\'s total movement count ÷ 4. '
        'Bars show the mean deviation across variants; jitter shows individual variants.'
    )
    doc.add_picture(str(_fig3_path), width=6.5 * 914400)
    doc.add_paragraph()


def analyze_decisions(df_per_tick: pd.DataFrame, group_color_map: dict = None) -> None:
    """
    Analyze decision correctness across groups and ticks (excluding tick 0).

    A decision can be:
      - 'correct': appropriate action for sensory input
      - 'nothing_sensed': moved but no food sensed anywhere (uninformed)
      - 'incorrect': wrong action given the sensory input

    Steps:
      1. Plot counts of each decision type per run, stratified by group
      2. Plot counts excluding 'nothing_sensed'
      3. Plot % distribution per variant (pooled runs)
      4. Plot % distribution excluding 'nothing_sensed' from totals
      5. Statistical tests: do groups differ in decision quality?

    Args:
        df_per_tick:     Per-tick tracking DataFrame with 'correct' column (global df_per_tick).
        group_color_map: Dict mapping group name -> hex colour string.
    """
    if df_per_tick is None or len(df_per_tick) == 0 or 'correct' not in df_per_tick.columns:
        return

    figures_dir = Path(__file__).resolve().parent / f'figures_{EXPERIMENT_NAME}'
    figures_dir.mkdir(exist_ok=True)

    # Exclude tick 0 (no real decision made then)
    df = df_per_tick[df_per_tick['tick'] > 0].copy()
    
    _DECISION_TYPES = ['correct', 'nothing_sensed', 'incorrect']
    _DECISION_TYPES_INFORMED = ['correct', 'incorrect']
    
    _decision_colors = {
        'correct': '#59a14f',
        'nothing_sensed': '#999999',
        'incorrect': '#e15759'
    }

    all_groups, zorder_map = _get_ordered_groups_with_zorder(df['group'].unique())
    n_groups = len(all_groups)

    doc.add_heading('Decision Analysis', level=2)

    # ── 1. Build per-run decision counts ──────────────────────────────────────
    doc.add_heading('1. Decision Counts per Run (All Decision Types)', level=3)

    # Count per decision type per run
    _rows = []
    for (_grp, _var, _run), _grp_df in df.groupby(['group', 'variant', 'run']):
        for _dt in _DECISION_TYPES:
            _rows.append({
                'group':        _grp,
                'variant':      _var,
                'run':          _run,
                'decision_type': _dt,
                'count':        int((_grp_df['correct'] == _dt).sum()),
            })
    _df_counts_all = pd.DataFrame(_rows)

    # Plot 1: All decision types
    _dir_width   = 0.7 / len(_DECISION_TYPES)
    _block_gap   = 1.0
    _x_positions  = {}
    _block_centers = {}
    for _gi, _g in enumerate(all_groups):
        _block_start = _gi * (_block_gap + 0.7)
        _block_centers[_g] = _block_start + 0.35 - _dir_width / 2
        for _di, _dt in enumerate(_DECISION_TYPES):
            _x_positions[(_g, _dt)] = _block_start + _di * _dir_width

    fig, ax = plt.subplots(figsize=(12, 7))
    _rng = np.random.default_rng(42)

    for _g in all_groups:
        for _dt in _DECISION_TYPES:
            _vals = _df_counts_all[(_df_counts_all['group'] == _g) & (_df_counts_all['decision_type'] == _dt)]['count'].values
            _x    = _x_positions[(_g, _dt)]
            _col  = _decision_colors[_dt]
            # jitter
            _jit = _rng.uniform(-_dir_width * 0.3, _dir_width * 0.3, size=len(_vals))
            ax.scatter(_x + _jit, _vals, color=_col, alpha=0.4, s=10, zorder=1)
            # box
            if len(_vals) >= 2:
                _bp = ax.boxplot(
                    _vals,
                    positions=[_x],
                    widths=_dir_width * 0.8,
                    patch_artist=True,
                    showfliers=False,
                    manage_ticks=False,
                )
                _bp['boxes'][0].set_facecolor(_col)
                _bp['boxes'][0].set_alpha(0.35)
                _bp['medians'][0].set_color('black')

    # Legend
    _legend_handles = [mpatches.Patch(color=_decision_colors[_dt], label=_dt) for _dt in _DECISION_TYPES]
    ax.legend(handles=_legend_handles, loc='upper right', fontsize=10, title='Decision Type')
    ax.set_xticks([_block_centers[_g] for _g in all_groups])
    ax.set_xticklabels(all_groups, fontsize=11)
    ax.set_ylabel('Decision Count per Run', fontsize=12)
    ax.set_title('Decision Type Distribution by Group (All Ticks ≥ 1)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    _fig_path = figures_dir / 'decisions_n_all.png'
    fig.tight_layout()
    _ymin, _ymax = ax.get_ylim()
    ax.set_ylim(_ymin, _ymax + (_ymax - _ymin) / 3)
    fig.savefig(str(_fig_path), dpi=150)
    plt.close(fig)
    doc.add_picture(str(_fig_path), width=6.5 * 914400)
    doc.add_paragraph()

    # ── 2. Plot informed decisions only (excluding nothing_sensed) ──────────────
    doc.add_heading('2. Decision Counts per Run (Informed Decisions Only)', level=3)

    _dir_width_inf   = 0.7 / len(_DECISION_TYPES_INFORMED)
    _x_positions_inf = {}
    _block_centers_inf = {}
    for _gi, _g in enumerate(all_groups):
        _block_start = _gi * (_block_gap + 0.7)
        _block_centers_inf[_g] = _block_start + 0.35 - _dir_width_inf / 2
        for _di, _dt in enumerate(_DECISION_TYPES_INFORMED):
            _x_positions_inf[(_g, _dt)] = _block_start + _di * _dir_width_inf

    fig, ax = plt.subplots(figsize=(12, 7))
    _rng = np.random.default_rng(42)

    for _g in all_groups:
        for _dt in _DECISION_TYPES_INFORMED:
            _vals = _df_counts_all[(_df_counts_all['group'] == _g) & (_df_counts_all['decision_type'] == _dt)]['count'].values
            _x    = _x_positions_inf[(_g, _dt)]
            _col  = _decision_colors[_dt]
            # jitter
            _jit = _rng.uniform(-_dir_width_inf * 0.3, _dir_width_inf * 0.3, size=len(_vals))
            ax.scatter(_x + _jit, _vals, color=_col, alpha=0.4, s=10, zorder=1)
            # box
            if len(_vals) >= 2:
                _bp = ax.boxplot(
                    _vals,
                    positions=[_x],
                    widths=_dir_width_inf * 0.8,
                    patch_artist=True,
                    showfliers=False,
                    manage_ticks=False,
                )
                _bp['boxes'][0].set_facecolor(_col)
                _bp['boxes'][0].set_alpha(0.35)
                _bp['medians'][0].set_color('black')

    _legend_handles_inf = [mpatches.Patch(color=_decision_colors[_dt], label=_dt) for _dt in _DECISION_TYPES_INFORMED]
    ax.legend(handles=_legend_handles_inf, loc='upper right', fontsize=10, title='Decision Type')
    ax.set_xticks([_block_centers_inf[_g] for _g in all_groups])
    ax.set_xticklabels(all_groups, fontsize=11)
    ax.set_ylabel('Decision Count per Run', fontsize=12)
    ax.set_title('Informed Decision Distribution by Group (Excluding "Nothing Sensed")', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    _fig_path = figures_dir / 'decisions_n_informed.png'
    fig.tight_layout()
    _ymin, _ymax = ax.get_ylim()
    ax.set_ylim(_ymin, _ymax + (_ymax - _ymin) / 3)
    fig.savefig(str(_fig_path), dpi=150)
    plt.close(fig)
    doc.add_picture(str(_fig_path), width=6.5 * 914400)
    doc.add_paragraph()

    # ── 3. Percentage distribution (all decision types, pooled per variant) ─────
    doc.add_heading('3. Decision Type Distribution (% Per Variant, All Types)', level=3)

    # Sum decision counts per variant across all runs
    _df_var_all = (
        _df_counts_all.groupby(['group', 'variant', 'decision_type'])['count']
        .sum()
        .reset_index()
    )
    _df_var_pivot_all = _df_var_all.pivot_table(
        index=['group', 'variant'],
        columns='decision_type',
        values='count',
        fill_value=0,
    ).reset_index()
    _df_var_pivot_all['_total'] = _df_var_pivot_all[_DECISION_TYPES].sum(axis=1)

    _pct_rows_all = []
    for _, _row in _df_var_pivot_all.iterrows():
        _tot = _row['_total']
        if _tot == 0:
            continue
        for _dt in _DECISION_TYPES:
            _pct_rows_all.append({
                'group':        _row['group'],
                'variant':      _row['variant'],
                'decision_type': _dt,
                'percentage':   (_row[_dt] / _tot * 100.0) if _tot > 0 else 0.0,
            })
    _df_pct_all = pd.DataFrame(_pct_rows_all)

    fig, ax = plt.subplots(figsize=(12, 7))
    _rng = np.random.default_rng(42)

    for _g in all_groups:
        for _dt in _DECISION_TYPES:
            _mask = (_df_pct_all['group'] == _g) & (_df_pct_all['decision_type'] == _dt)
            _vals = _df_pct_all.loc[_mask, 'percentage'].values
            _x    = _x_positions[(_g, _dt)]
            _col  = _decision_colors[_dt]

            if len(_vals) == 0:
                continue

            _jit = _rng.uniform(-_dir_width * 0.3, _dir_width * 0.3, size=len(_vals))
            ax.scatter(_x + _jit, _vals, color=_col, alpha=0.5, s=14, zorder=1)

            _mean_pct = float(np.mean(_vals))
            ax.bar(
                _x, _mean_pct,
                width=_dir_width * 0.85,
                color=_col, alpha=0.45,
                zorder=2,
            )

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))
    ax.legend(handles=_legend_handles, loc='upper right', fontsize=10, title='Decision Type')
    ax.set_xticks([_block_centers[_g] for _g in all_groups])
    ax.set_xticklabels(all_groups, fontsize=11)
    ax.set_ylabel('Percentage of Decisions (%)', fontsize=12)
    ax.set_title('Decision Type Distribution by Group (% per Variant, All Types)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    _fig_path = figures_dir / 'decisions_ratio_all.png'
    fig.tight_layout()
    # Add 2x padding AFTER tight_layout, then save (without bbox_inches='tight' which would crop it)
    _ymin, _ymax = ax.get_ylim()
    _yrange = _ymax - _ymin
    ax.set_ylim(_ymin, _ymax + _yrange / 3)
    fig.savefig(str(_fig_path), dpi=150)
    plt.close(fig)
    doc.add_picture(str(_fig_path), width=6.5 * 914400)
    doc.add_paragraph()

    # ── 4. Percentage distribution (informed only, excluding nothing_sensed from totals) ─
    doc.add_heading('4. Informed Decision Distribution (% Per Variant, Excluding "Nothing Sensed")', level=3)

    # Filter to only informed decisions, then recalculate percentages
    _df_var_inf = (
        _df_counts_all[_df_counts_all['decision_type'].isin(_DECISION_TYPES_INFORMED)]
        .groupby(['group', 'variant', 'decision_type'])['count']
        .sum()
        .reset_index()
    )
    _df_var_pivot_inf = _df_var_inf.pivot_table(
        index=['group', 'variant'],
        columns='decision_type',
        values='count',
        fill_value=0,
    ).reset_index()
    _df_var_pivot_inf['_total'] = _df_var_pivot_inf[_DECISION_TYPES_INFORMED].sum(axis=1)

    _pct_rows_inf = []
    for _, _row in _df_var_pivot_inf.iterrows():
        _tot = _row['_total']
        if _tot == 0:
            continue
        for _dt in _DECISION_TYPES_INFORMED:
            _pct_rows_inf.append({
                'group':        _row['group'],
                'variant':      _row['variant'],
                'decision_type': _dt,
                'percentage':   (_row[_dt] / _tot * 100.0) if _tot > 0 else 0.0,
            })
    _df_pct_inf = pd.DataFrame(_pct_rows_inf)

    fig, ax = plt.subplots(figsize=(12, 7))
    _rng = np.random.default_rng(42)

    for _g in all_groups:
        for _dt in _DECISION_TYPES_INFORMED:
            _mask = (_df_pct_inf['group'] == _g) & (_df_pct_inf['decision_type'] == _dt)
            _vals = _df_pct_inf.loc[_mask, 'percentage'].values
            _x    = _x_positions_inf[(_g, _dt)]
            _col  = _decision_colors[_dt]

            if len(_vals) == 0:
                continue

            _jit = _rng.uniform(-_dir_width_inf * 0.3, _dir_width_inf * 0.3, size=len(_vals))
            ax.scatter(_x + _jit, _vals, color=_col, alpha=0.5, s=14, zorder=1)

            _mean_pct = float(np.mean(_vals))
            ax.bar(
                _x, _mean_pct,
                width=_dir_width_inf * 0.85,
                color=_col, alpha=0.45,
                zorder=2,
            )

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))
    ax.legend(handles=_legend_handles_inf, loc='upper right', fontsize=10, title='Decision Type')
    ax.set_xticks([_block_centers_inf[_g] for _g in all_groups])
    ax.set_xticklabels(all_groups, fontsize=11)
    ax.set_ylabel('Percentage of Informed Decisions (%)', fontsize=12)
    ax.set_title('Informed Decision Distribution by Group (% of Informed Decisions Only)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    _fig_path = figures_dir / 'decisions_ratio_informed.png'
    fig.tight_layout()
    # Add 2x padding AFTER tight_layout, then save (without bbox_inches='tight' which would crop it)
    _ymin, _ymax = ax.get_ylim()
    _yrange = _ymax - _ymin
    ax.set_ylim(_ymin, _ymax + _yrange / 3)
    fig.savefig(str(_fig_path), dpi=150)
    plt.close(fig)
    doc.add_picture(str(_fig_path), width=6.5 * 914400)
    doc.add_paragraph()

    # ── 5. Statistical testing: Do groups differ in decision quality? ───────────
    doc.add_heading('5. Statistical Comparison — Decision Quality Between Groups', level=3)

    # Test 1: All three decision types
    doc.add_heading('Test 1: All Decision Types (Correct vs Nothing_Sensed vs Incorrect)', level=4)
    
    doc.add_paragraph(
        'We test whether the three groups show different distributions across all decision types. '
        'Data are the % of each decision type per variant. '
    )

    for _dt in _DECISION_TYPES:
        doc.add_heading(f'Decision Type: {_dt.upper()}', level=5)

        _arrs = []
        for _g in all_groups:
            _mask = (_df_pct_all['group'] == _g) & (_df_pct_all['decision_type'] == _dt)
            _vals = _df_pct_all.loc[_mask, 'percentage'].values
            _arrs.append(_vals)

        # Normality testing
        _norm_results = {}
        _all_normal = True
        for _g, _v in zip(all_groups, _arrs):
            if len(_v) < 3 or np.ptp(_v) == 0:
                _norm_results[_g] = (np.nan, np.nan)
                _all_normal = False
            else:
                _w_sw, _p_sw = shapiro(_v)
                _norm_results[_g] = (_w_sw, _p_sw)
                if _p_sw < 0.05:
                    _all_normal = False

        # Omnibus test
        if _all_normal:
            _stat, _p = f_oneway(*_arrs)
            _test_name = 'One-way ANOVA'
            _grand = np.mean(np.concatenate(_arrs))
            _ss_b = sum(len(a) * (np.mean(a) - _grand) ** 2 for a in _arrs)
            _ss_t = sum(np.sum((a - _grand) ** 2) for a in _arrs)
            _es_omni = _ss_b / _ss_t if _ss_t > 0 else 0.0
            _es_type_omni = 'Eta-squared'
            _es_col_omni = 'η²'
        else:
            _all_combined = np.concatenate(_arrs)
            if np.ptp(_all_combined) == 0:
                _stat, _p = 0.0, 1.0
            else:
                _stat, _p = kruskal(*_arrs)
            _test_name = 'Kruskal-Wallis'
            _N = len(_all_combined)
            _k = len(_arrs)
            _es_omni = (_stat - _k + 1) / (_N - _k) if (_N - _k) > 0 else 0.0
            _es_type_omni = 'Epsilon-squared'
            _es_col_omni = 'ε²'

        _omni_tbl = doc.add_table(rows=2, cols=4)
        _omni_tbl.style = 'Light Grid Accent 1'
        _oh = _omni_tbl.rows[0].cells
        _oh[0].text = 'Test Name'
        _oh[1].text = 'Test Statistic'
        _oh[2].text = 'p-value'
        _oh[3].text = _es_col_omni
        _ov = _omni_tbl.rows[1].cells
        _ov[0].text = _test_name
        _ov[1].text = f'{_stat:.4f}'
        _ov[2].text = f'{_p:.2e}' + (' *' if _p < 0.05 else '')
        _ov[3].text = f'{_es_omni:.4f} ({classify_effect_size(_es_omni, _es_type_omni)})'

        if _p < 0.05:
            doc.add_paragraph(f'Result: Groups show SIGNIFICANT differences in {_dt} rates (p < 0.05) ✓')
            
            # ── Pairwise post-hoc (only if >2 groups) ────────────────────────────
            if len(all_groups) > 2:
                _k_groups = len(all_groups)
                _N_groups = sum(len(a) for a in _arrs)
                _df_err = _N_groups - _k_groups

                _ph_raw = []
                for (_gi, _ga), (_gj, _gb) in combinations(enumerate(all_groups), 2):
                    _a, _b = _arrs[_gi], _arrs[_gj]
                    if _all_normal:
                        _s, _pp = ttest_ind(_a, _b, equal_var=False)
                        _n0, _n1 = len(_a), len(_b)
                        _v0, _v1 = np.var(_a, ddof=1), np.var(_b, ddof=1)
                        _ps = np.sqrt(((_n0-1)*_v0 + (_n1-1)*_v1) / (_n0+_n1-2))
                        _es = (np.mean(_a) - np.mean(_b)) / _ps if _ps > 0 else 0.0
                        _ph_es_type  = "Cohen's d"
                        _ph_es_label = "Cohen's d"
                    else:
                        _s, _pp = mannwhitneyu(_a, _b, alternative='two-sided')
                        _es = 1 - (2 * _s) / (len(_a) * len(_b))
                        _ph_es_type  = 'Rank-Biserial r'
                        _ph_es_label = 'Rank-Biserial r'
                    _ph_raw.append((_ga, _gb, _s, _pp, _es))

                if _all_normal:
                    _ph_rows = []
                    for (_ga, _gb, _s, _pp, _es) in _ph_raw:
                        _cp = float(studentized_range.sf(abs(_s), _k_groups, _df_err))
                        _cp = min(_cp, 1.0)
                        _ph_rows.append((_ga, _gb, _s, _cp, _es))
                    _ph_name = "Welch's t-test (Tukey's HSD correction)"
                else:
                    _ph_rej, _ph_pc, _, _ = multipletests([x[3] for x in _ph_raw], method='holm')
                    _ph_rows = [(_ga, _gb, _s, _pc, _es)
                                for (_ga, _gb, _s, _, _es), _pc in zip(_ph_raw, _ph_pc)]
                    _ph_name = 'Mann-Whitney U (Holm-Bonferroni correction)'

                doc.add_paragraph(f'Pairwise Post-Hoc ({_ph_name}):', style='Heading 5')
                _ph_tbl = doc.add_table(rows=len(_ph_rows) + 1, cols=5)
                _ph_tbl.style = 'Light Grid Accent 1'
                _ph_h = _ph_tbl.rows[0].cells
                _ph_h[0].text = 'Group A'
                _ph_h[1].text = 'Group B'
                _ph_h[2].text = 'Statistic'
                _ph_h[3].text = 'p-value'
                _ph_h[4].text = _ph_es_label
                for _ri, (_ga, _gb, _s, _pp, _es) in enumerate(_ph_rows, 1):
                    _c = _ph_tbl.rows[_ri].cells
                    _c[0].text = _ga
                    _c[1].text = _gb
                    _c[2].text = f'{_s:.4f}'
                    _c[3].text = f'{_pp:.2e}' + (' *' if _pp < 0.05 else '')
                    _c[4].text = f'{_es:.4f} ({classify_effect_size(_es, _ph_es_type)})'
        else:
            doc.add_paragraph(f'Result: No significant difference in {_dt} rates between groups (p ≥ 0.05)')

        doc.add_paragraph()

    # Test 2: Informed decisions only (correct vs incorrect)
    doc.add_heading('Test 2: Informed Decisions Only (Correct vs Incorrect)', level=4)

    doc.add_paragraph(
        'We test whether the three groups show different rates of correct decisions among informed (non-uncertain) decisions only. '
        'Data are the % of correct decisions among informed decisions, per variant. '
    )

    # Extract correct % from informed only
    _df_correct_pct = _df_pct_inf[_df_pct_inf['decision_type'] == 'correct'].copy()

    _arrs_correct = []
    for _g in all_groups:
        _mask = (_df_correct_pct['group'] == _g)
        _vals = _df_correct_pct.loc[_mask, 'percentage'].values
        _arrs_correct.append(_vals)

    # Normality testing
    _norm_results_correct = {}
    _all_normal_correct = True
    for _g, _v in zip(all_groups, _arrs_correct):
        if len(_v) < 3 or np.ptp(_v) == 0:
            _norm_results_correct[_g] = (np.nan, np.nan)
            _all_normal_correct = False
        else:
            _w_sw, _p_sw = shapiro(_v)
            _norm_results_correct[_g] = (_w_sw, _p_sw)
            if _p_sw < 0.05:
                _all_normal_correct = False

    # Omnibus test
    if _all_normal_correct:
        _stat_c, _p_c = f_oneway(*_arrs_correct)
        _test_name_c = 'One-way ANOVA'
        _grand_c = np.mean(np.concatenate(_arrs_correct))
        _ss_b_c = sum(len(a) * (np.mean(a) - _grand_c) ** 2 for a in _arrs_correct)
        _ss_t_c = sum(np.sum((a - _grand_c) ** 2) for a in _arrs_correct)
        _es_c = _ss_b_c / _ss_t_c if _ss_t_c > 0 else 0.0
        _es_type_c = 'Eta-squared'
        _es_col_c = 'η²'
    else:
        _all_combined_c = np.concatenate(_arrs_correct)
        if np.ptp(_all_combined_c) == 0:
            _stat_c, _p_c = 0.0, 1.0
        else:
            _stat_c, _p_c = kruskal(*_arrs_correct)
        _test_name_c = 'Kruskal-Wallis'
        _N_c = len(_all_combined_c)
        _k_c = len(_arrs_correct)
        _es_c = (_stat_c - _k_c + 1) / (_N_c - _k_c) if (_N_c - _k_c) > 0 else 0.0
        _es_type_c = 'Epsilon-squared'
        _es_col_c = 'ε²'

    _omni_tbl_c = doc.add_table(rows=2, cols=4)
    _omni_tbl_c.style = 'Light Grid Accent 1'
    _oh_c = _omni_tbl_c.rows[0].cells
    _oh_c[0].text = 'Test Name'
    _oh_c[1].text = 'Test Statistic'
    _oh_c[2].text = 'p-value'
    _oh_c[3].text = _es_col_c
    _ov_c = _omni_tbl_c.rows[1].cells
    _ov_c[0].text = _test_name_c
    _ov_c[1].text = f'{_stat_c:.4f}'
    _ov_c[2].text = f'{_p_c:.2e}' + (' *' if _p_c < 0.05 else '')
    _ov_c[3].text = f'{_es_c:.4f} ({classify_effect_size(_es_c, _es_type_c)})'

    if _p_c < 0.05:
        doc.add_paragraph('Result: Groups show SIGNIFICANT differences in correct decision rates (p < 0.05) ✓')
        
        # ── Pairwise post-hoc (only if >2 groups) ────────────────────────────
        if len(all_groups) > 2:
            _k_groups_c = len(all_groups)
            _N_groups_c = sum(len(a) for a in _arrs_correct)
            _df_err_c = _N_groups_c - _k_groups_c

            _ph_raw_c = []
            for (_gi, _ga), (_gj, _gb) in combinations(enumerate(all_groups), 2):
                _a, _b = _arrs_correct[_gi], _arrs_correct[_gj]
                if _all_normal_correct:
                    _s, _pp = ttest_ind(_a, _b, equal_var=False)
                    _n0, _n1 = len(_a), len(_b)
                    _v0, _v1 = np.var(_a, ddof=1), np.var(_b, ddof=1)
                    _ps = np.sqrt(((_n0-1)*_v0 + (_n1-1)*_v1) / (_n0+_n1-2))
                    _es = (np.mean(_a) - np.mean(_b)) / _ps if _ps > 0 else 0.0
                    _ph_es_type_c  = "Cohen's d"
                    _ph_es_label_c = "Cohen's d"
                else:
                    _s, _pp = mannwhitneyu(_a, _b, alternative='two-sided')
                    _es = 1 - (2 * _s) / (len(_a) * len(_b))
                    _ph_es_type_c  = 'Rank-Biserial r'
                    _ph_es_label_c = 'Rank-Biserial r'
                _ph_raw_c.append((_ga, _gb, _s, _pp, _es))

            if _all_normal_correct:
                _ph_rows_c = []
                for (_ga, _gb, _s, _pp, _es) in _ph_raw_c:
                    _cp = float(studentized_range.sf(abs(_s), _k_groups_c, _df_err_c))
                    _cp = min(_cp, 1.0)
                    _ph_rows_c.append((_ga, _gb, _s, _cp, _es))
                _ph_name_c = "Welch's t-test (Tukey's HSD correction)"
            else:
                _ph_rej_c, _ph_pc_c, _, _ = multipletests([x[3] for x in _ph_raw_c], method='holm')
                _ph_rows_c = [(_ga, _gb, _s, _pc, _es)
                              for (_ga, _gb, _s, _, _es), _pc in zip(_ph_raw_c, _ph_pc_c)]
                _ph_name_c = 'Mann-Whitney U (Holm-Bonferroni correction)'

            doc.add_paragraph(f'Pairwise Post-Hoc ({_ph_name_c}):', style='Heading 5')
            _ph_tbl_c = doc.add_table(rows=len(_ph_rows_c) + 1, cols=5)
            _ph_tbl_c.style = 'Light Grid Accent 1'
            _ph_h_c = _ph_tbl_c.rows[0].cells
            _ph_h_c[0].text = 'Group A'
            _ph_h_c[1].text = 'Group B'
            _ph_h_c[2].text = 'Statistic'
            _ph_h_c[3].text = 'p-value'
            _ph_h_c[4].text = _ph_es_label_c
            for _ri, (_ga, _gb, _s, _pp, _es) in enumerate(_ph_rows_c, 1):
                _c = _ph_tbl_c.rows[_ri].cells
                _c[0].text = _ga
                _c[1].text = _gb
                _c[2].text = f'{_s:.4f}'
                _c[3].text = f'{_pp:.2e}' + (' *' if _pp < 0.05 else '')
                _c[4].text = f'{_es:.4f} ({classify_effect_size(_es, _ph_es_type_c)})'
    else:
        doc.add_paragraph('Result: No significant difference in correct decision rates between groups (p ≥ 0.05)')

    doc.add_paragraph()


# ==================================================================================================================================================
# SECTION C) DATA LOADING
# ==================================================================================================================================================

# --- Column subsets loaded from each dataset ---
_SUMMARY_COLS  = ['run_id', 'lifetime_ticks', 'foods', 'distance', 'final_energy']
_PER_TICK_COLS = ['tick', 'food_sensed_N', 'food_sensed_E', 'food_sensed_S', 'food_sensed_W',
                  'movement', 'food_consumed', 'energy', 'manhattan_dist', 'decision_made']

# --- Availability flags (determined by probing the first file) ---
per_tick_available = False
heatmap_available  = False

# --- Accumulators (assembled into final structures after all files are loaded) ---
_summary_parts  = []   # list[pd.DataFrame]  -->  df_summary
_variant_parts  = []   # list[dict]  -->  variant_data (group, type, variant, eta, ...)
_per_tick_parts = []   # list[pd.DataFrame]  -->  df_per_tick
modulation_data = {}   # {group_name: {variant_id (int): np.ndarray}}
wiring_data     = {}   # {group_name: {variant_id (int): np.ndarray}}
per_neuron_data = {}   # {group_name: {variant_id (int): np.ndarray}}
heatmap_data    = {}   # {group_name: {variant_id (int): {run_id (int): np.ndarray}}}


def _process_hdf5_file(hdf5_path: Path, group_name: str, data_type: str,
                       check_availability: bool = False) -> None:
    """
    Load all data from one HDF5 file and append to the module-level accumulators.

    Always loads dataset a) summary for every variant.
    If check_availability=True, probes variant_0/run_0 for d) per_tick and e) staying
    and sets the module-level per_tick_available / heatmap_available flags accordingly.
    For subsequent files those flags are already set; raises loudly if expected data is absent.

    Heatmaps are limited to the first 5 variants per group to reduce memory usage.

    Args:
        hdf5_path:          Absolute path to the .h5 file.
        group_name:         Human-readable group label (from user config at the top).
        data_type:          'experiment' or 'benchmark'.
        check_availability: True only for the very first file loaded.
    """
    global per_tick_available, heatmap_available

    print(f"  Loading: {hdf5_path.name}")

    with h5py.File(hdf5_path, 'r') as f:
        variant_keys = sorted([k for k in f.keys() if k.startswith('variant_')])
        if not variant_keys:
            raise RuntimeError(f"No variant_ groups found in {hdf5_path}")

        # --- 1.2  Probe d) and e) availability (first file only) ---
        if check_availability:
            _probe_vg   = f[variant_keys[0]]
            _probe_runs = sorted([k for k in _probe_vg.keys() if k.startswith('run_')])
            if _probe_runs:
                _probe_run      = _probe_vg[_probe_runs[0]]
                per_tick_available = 'per_tick' in _probe_run
                heatmap_available  = 'staying'  in _probe_run
            # if no run_ subfolders exist, both flags remain False
            print(f"    Availability probe  ->  per_tick={per_tick_available},  heatmap={heatmap_available}")

        modulation_data[group_name] = {}
        wiring_data[group_name]     = {}
        per_neuron_data[group_name] = {}
        if heatmap_available:
            heatmap_data[group_name] = {}

        # --- Limit heatmap loading to first, middle (floor), and last variant ---
        _n_vk = len(variant_keys)
        if _n_vk == 1:
            _hm_indices = {0}
        elif _n_vk == 2:
            _hm_indices = {0, _n_vk - 1}
        else:
            _hm_indices = {0, int(np.floor((_n_vk - 1) / 2)), _n_vk - 1}

        for _vi, variant_key in enumerate(variant_keys):
            variant_id = int(variant_key.split('_')[1])
            vg = f[variant_key]

            # --- 1.1  Load dataset a) summary ---
            df_var = pd.DataFrame(vg['summary'][:])[_SUMMARY_COLS]
            df_var.insert(0, 'variant', variant_id)
            df_var.insert(0, 'type',    data_type)
            df_var.insert(0, 'group',   group_name)
            _summary_parts.append(df_var)

            # --- 1.1b  Load dataset b) modulation, c) eta, d) wiring, e) tonic_activations ---
            modulation_data[group_name][variant_id] = vg['modulation'][:]
            eta_value = vg['eta'][()] if 'eta' in vg else None  # Use [()] for scalar HDF5 datasets
            # Append variant-level metadata (eta, and future metrics) to _variant_parts
            _variant_parts.append({
                'group': group_name,
                'type': data_type,
                'variant': variant_id,
                'eta': eta_value,
            })
            wiring_data[group_name][variant_id]     = vg['wiring'][:]
            per_neuron_data[group_name][variant_id] = vg['tonic_activations'][:] if 'tonic_activations' in vg else np.array([])

            # Only load heatmap for the selected first / middle / last variants
            load_heatmap_for_this_variant = heatmap_available and _vi in _hm_indices

            # --- 1.3 / 1.4  Load d) per_tick and/or e) staying ---
            if per_tick_available or load_heatmap_for_this_variant:
                run_keys = sorted([k for k in vg.keys() if k.startswith('run_')])
                if not run_keys:
                    raise RuntimeError(
                        f"per_tick/heatmap data expected (flags are set) but no run_ subfolders "
                        f"found in  {hdf5_path.name}/{variant_key}"
                    )

                if load_heatmap_for_this_variant:
                    heatmap_data[group_name][variant_id] = {}

                for run_key in run_keys:
                    run_id = int(run_key.split('_')[1])
                    rg     = vg[run_key]

                    if per_tick_available:
                        if 'per_tick' not in rg:
                            raise RuntimeError(
                                f"per_tick expected but missing in "
                                f"{hdf5_path.name}/{variant_key}/{run_key}"
                            )
                        df_tick = pd.DataFrame(rg['per_tick'][:])[_PER_TICK_COLS]
                        df_tick.insert(0, 'run',     run_id)
                        df_tick.insert(0, 'variant', variant_id)
                        df_tick.insert(0, 'type',    data_type)
                        df_tick.insert(0, 'group',   group_name)
                        _per_tick_parts.append(df_tick)

                    if load_heatmap_for_this_variant:
                        if 'staying' not in rg:
                            raise RuntimeError(
                                f"staying expected but missing in "
                                f"{hdf5_path.name}/{variant_key}/{run_key}"
                            )
                        heatmap_data[group_name][variant_id][run_id] = rg['staying'][:]

    print(f"    Done: {len(variant_keys)} variants.")


# --- Step 1: First experiment file — probes for d/e availability ---
_first_exp_name, _first_exp_hdf5, *_ = EXPERIMENT_HDF5_FILES[0]
print(f"Loading first experiment '{_first_exp_name}' (availability probe)...")
_process_hdf5_file(_find_hdf5_file(_first_exp_hdf5), _first_exp_name, 'experiment',
                   check_availability=True)

# --- Step 2: Remaining experiment files ---
for _exp_name, _exp_hdf5, *_ in EXPERIMENT_HDF5_FILES[1:]:
    print(f"Loading experiment '{_exp_name}'...")
    _process_hdf5_file(_find_hdf5_file(_exp_hdf5), _exp_name, 'experiment')

# --- Step 3: Benchmark files ---
for _bench_name, _bench_hdf5, *_ in BENCHMARK_HDF5_FILES:
    print(f"Loading benchmark '{_bench_name}'...")
    _process_hdf5_file(_find_hdf5_file(_bench_hdf5), _bench_name, 'benchmark')

# --- Assemble final data structures ---
df_summary = pd.concat(_summary_parts, ignore_index=True)
df_summary['distance_norm'] = df_summary['distance'] / df_summary['lifetime_ticks']
print(f"\ndf_summary:    {len(df_summary)} rows  |  groups: {df_summary['group'].unique().tolist()}")

_n_modulation = sum(len(v) for v in modulation_data.values())
_n_wiring     = sum(len(v) for v in wiring_data.values())
_n_tonic      = sum(len(v) for v in per_neuron_data.values())
print(f"modulation_data: {len(modulation_data)} groups  |  {_n_modulation} total arrays")
print(f"wiring_data:     {len(wiring_data)} groups  |  {_n_wiring} total arrays")
print(f"tonic_activation_data: {len(per_neuron_data)} groups  |  {_n_tonic} total arrays")

variant_data = pd.DataFrame(_variant_parts) if _variant_parts else None
if variant_data is not None:
    print(f"variant_data:    {len(variant_data)} total variants")
else:
    print("variant_data:    not available")

# --- Add weight_effective column to all wiring arrays ---
# weight_effective = weight_initial × reliability; computed once here so all
# downstream analysis functions can access it without recomputing.
for _grp_variants in wiring_data.values():
    for _vid, _arr in _grp_variants.items():
        _df = pd.DataFrame(_arr)
        _df['weight_effective'] = _df['weight_initial'] * _df['reliability']
        _grp_variants[_vid] = _df.to_records(index=False)

if per_tick_available:
    df_per_tick = pd.concat(_per_tick_parts, ignore_index=True)
    # Normalised tick position per run: tick 0 → 0 %, last tick → 100 %
    _max_tick_per_run = df_per_tick.groupby(['group', 'variant', 'run'])['tick'].transform('max')
    df_per_tick['ticks_norm'] = (df_per_tick['tick'] / _max_tick_per_run * 100).round(1)
    
    # --- Add 'correct' column: classify each decision as correct / nothing_sensed / incorrect ---
    # Decode movement to str if needed
    if df_per_tick['movement'].dtype == object and len(df_per_tick) > 0 and isinstance(df_per_tick['movement'].iloc[0], bytes):
        df_per_tick['movement'] = df_per_tick['movement'].str.decode('utf-8')
    
    # Create lagged columns for sensory data (previous tick's sensory input drives current tick's decision)
    for _dir in ['N', 'E', 'S', 'W']:
        _col = f'food_sensed_{_dir}'
        df_per_tick[f'{_col}_prev'] = (
            df_per_tick.groupby(['group', 'variant', 'run'])[_col].shift(1).fillna(0).astype(int)
        )
    
    # Initialize correct column with 'incorrect' (default)
    df_per_tick['correct'] = 'incorrect'
    
    # Vectorized classification (much faster than row-by-row loop)
    # For 'stay' movements: correct if food_consumed == 1 in the same tick
    _is_stay = df_per_tick['movement'] == 'stay'
    df_per_tick.loc[_is_stay & (df_per_tick['food_consumed'] == 1), 'correct'] = 'correct'
    
    # For directional movements: check previous tick's sensory data
    # First, identify which rows have nothing sensed in any direction (in prev tick)
    _nothing_sensed_mask = (
        (df_per_tick['food_sensed_N_prev'] == 0) &
        (df_per_tick['food_sensed_E_prev'] == 0) &
        (df_per_tick['food_sensed_S_prev'] == 0) &
        (df_per_tick['food_sensed_W_prev'] == 0)
    )
    
    # Process each direction
    for _dir in ['N', 'E', 'S', 'W']:
        _is_this_dir = df_per_tick['movement'] == _dir
        _sensed_col = f'food_sensed_{_dir}_prev'
        
        # Correct: movement direction matches sensed food (in prev tick)
        df_per_tick.loc[_is_this_dir & (df_per_tick[_sensed_col] == 1), 'correct'] = 'correct'
        
        # Nothing sensed: direction movement when nothing was sensed in any direction
        df_per_tick.loc[_is_this_dir & _nothing_sensed_mask, 'correct'] = 'nothing_sensed'
        # (else: stays 'incorrect' by default)
    
    # Drop the temporary lagged columns
    df_per_tick.drop(columns=[f'food_sensed_{d}_prev' for d in ['N', 'E', 'S', 'W']], inplace=True)
    
    
    print(f"df_per_tick:  {len(df_per_tick)} rows")
else:
    df_per_tick = None
    print("df_per_tick:  not available")

if heatmap_available:
    _n_heatmaps = sum(
        len(runs) for variants in heatmap_data.values() for runs in variants.values()
    )
    print(f"heatmap_data: {len(heatmap_data)} groups  |  {_n_heatmaps} total heatmaps")
else:
    print("heatmap_data: not available")




# ==================================================================================================================================================
# Build COLOR_MAP: maps every group name to a hex color string.
# Each entry in EXPERIMENT_HDF5_FILES / BENCHMARK_HDF5_FILES carries a color_index that
# selects directly from EXPERIMENT_COLORS / BENCHMARK_COLORS.
# If color_index is out of range, a color is auto-generated in the same hue family.
# ==================================================================================================================================================
import colorsys as _colorsys

COLOR_MAP: dict = {}

# All experiments (EXPERIMENT_HDF5_FILES)
_all_exp_names = [name for name, *_ in EXPERIMENT_HDF5_FILES]
for _i, (_exp_name, _exp_hdf5, _exp_color_idx) in enumerate(EXPERIMENT_HDF5_FILES):
    if _exp_color_idx < len(EXPERIMENT_COLORS):
        COLOR_MAP[_exp_name] = EXPERIMENT_COLORS[_exp_color_idx]
    else:
        # Auto-generate a greenish color (hue ~120° / 0.33)
        _h = 0.33
        _l = max(0.25, 0.45 - (_i - len(EXPERIMENT_COLORS)) * 0.05)
        _s = 0.6
        _r, _g, _b = _colorsys.hls_to_rgb(_h, _l, _s)
        COLOR_MAP[_exp_name] = f'#{int(_r*255):02x}{int(_g*255):02x}{int(_b*255):02x}'

# Also map the primary experiment's source label 'experiment' to its color
COLOR_MAP['experiment'] = COLOR_MAP.get(_all_exp_names[0], EXPERIMENT_COLORS[0])

# Benchmarks (BENCHMARK_HDF5_FILES)
for _i, (_bench_name, _bench_hdf5, _bench_color_idx) in enumerate(BENCHMARK_HDF5_FILES):
    if _bench_color_idx < len(BENCHMARK_COLORS):
        COLOR_MAP[_bench_name] = BENCHMARK_COLORS[_bench_color_idx]
    else:
        # Auto-generate a reddish color (hue ~0° / 0.0)
        _h = 0.0
        _l = max(0.30, 0.50 - (_i - len(BENCHMARK_COLORS)) * 0.05)
        _s = 0.55
        _r, _g, _b = _colorsys.hls_to_rgb(_h, _l, _s)
        COLOR_MAP[_bench_name] = f'#{int(_r*255):02x}{int(_g*255):02x}{int(_b*255):02x}'




# ==================================================================================================================================================
# SECTION D) ANALYSIS
# ==================================================================================================================================================

print("\n=== SECTION D: ANALYSIS ===")

from docx import Document

doc = Document()
doc.add_heading(f"Analysis Report: {EXPERIMENT_NAME}", level=0)


#region 1 Experiment Information

doc.add_heading("1. Experiment Information", level=1)

doc.add_heading("1.1. Data overview", level=2)
# Overview table: one row per group
_overview = _group_overview_stats(df_summary)
table = doc.add_table(rows=len(_overview) + 1, cols=5)
table.style = "Light Grid Accent 1"
_hdr = table.rows[0].cells
_hdr[0].text = "Group"
_hdr[1].text = "Type"
_hdr[2].text = "Dataset"
_hdr[3].text = "Variants"
_hdr[4].text = "Runs (total / per variant)"
for _row_data, _row in zip(_overview, table.rows[1:]):
    _row.cells[0].text = _row_data['group']
    _row.cells[1].text = _row_data['type']
    _row.cells[2].text = f"{_row_data['hdf5_filename']}.h5"
    _row.cells[3].text = str(_row_data['n_variants'])
    _row.cells[4].text = f"{_row_data['n_runs']} / {_row_data['runs_per_variant']}"

doc.add_paragraph()

# Add statistical methodology section
doc.add_heading("1.2.Statistical Analysis Methodology", level=2)
methodology_text = """
This report employs adaptive statistical analysis based on data distribution and group count. Four analytical scenarios are used:

1. TWO GROUPS, NORMALLY DISTRIBUTED DATA
   • Omnibus Test: Welch's t-test (compares means)
   • Effect Size: Cohen's d (interpretation: small ≥ 0.2, medium ≥ 0.5, large ≥ 0.8)
   • Post-hoc: Not applicable (omnibus test serves as pairwise comparison)
   • Significance threshold: p < 0.05

2. TWO GROUPS, NON-NORMALLY DISTRIBUTED DATA
   • Omnibus Test: Mann-Whitney U test (compares distributions via ranks)
   • Effect Size: Rank-Biserial r (interpretation: small ≥ 0.11, medium ≥ 0.28, large ≥ 0.43)
   • Post-hoc: Not applicable (omnibus test serves as pairwise comparison)
   • Significance threshold: p < 0.05

3. MORE THAN TWO GROUPS, NORMALLY DISTRIBUTED DATA
   • Omnibus Test: One-way ANOVA (tests if any group means differ)
   • Effect Size: Eta-squared η² (interpretation: small ≥ 0.01, medium ≥ 0.06, large ≥ 0.14)
   • Post-hoc Pairwise Test: Welch's t-test (all pairs compared)
   • Multiple Comparisons Correction: Tukey's HSD (controls Type I error across pairwise comparisons)
   • Post-hoc Effect Size: Cohen's d (interpretation: small ≥ 0.2, medium ≥ 0.5, large ≥ 0.8)
   • Significance threshold: p < 0.05 (omnibus and post-hoc)

4. MORE THAN TWO GROUPS, NON-NORMALLY DISTRIBUTED DATA
   • Omnibus Test: Kruskal-Wallis (tests if any group distributions differ via ranks)
   • Effect Size: Ordinal Epsilon-squared ε²R (interpretation: small ≥ 0.01, medium ≥ 0.08, large ≥ 0.26)
   • Post-hoc Pairwise Test: Mann-Whitney U (all pairs compared)
   • Multiple Comparisons Correction: Holm-Bonferroni (sequential adjustment of p-value thresholds)
   • Post-hoc Effect Size: Rank-Biserial r (interpretation: small ≥ 0.11, medium ≥ 0.28, large ≥ 0.43)
   • Significance threshold: p < 0.05 (omnibus and post-hoc)

Normality is assessed using the Shapiro-Wilk test (alpha = 0.05) for each group independently. If any group fails the normality test, the analysis uses non-parametric methods.

TIME SERIES

For metrics tracked per simulation tick (e.g., energy), per-tick data are analyzed using the Area Under the Curve (AUC) method:
   • Each run's per-tick metric values (tick 0 to end of run) are integrated using the trapezoidal rule to produce a single AUC value per run
   • AUC values are then compared across groups using the same adaptive statistical framework as above (Welch's t-test/Mann-Whitney U for 2 groups, ANOVA/Kruskal-Wallis for >2 groups)
   • This approach is conservative and avoids temporal autocorrelation issues by summarizing the time-series into a single magnitude metric
   • Visualizations show individual run curves (low opacity) overlaid with group mean curves (bold) and 95% confidence bands (shaded)

DISTRIBUTIONS

Distrinutions are compared using the Kolmogorov-Smirnov (KS) test: with Holm-Bonferroni post hoc corrections, 
and effect size measured by the KS D statistic (interpreted as small ≥ 0.1, medium ≥ 0.3, large ≥ 0.5).

SURVIVAL CURVES

All groups' survival distributions are compared using the log-rank test:
   • Omnibus Test: Log-rank test (rank-based non-parametric test comparing survival distributions)
   • Post-hoc Pairwise Test: Pairwise log-rank tests (all group pairs compared)
   • Multiple Comparisons Correction: Holm-Bonferroni (sequential adjustment of p-value thresholds)
   • Post-hoc Effect Size: χ² per pair (interpretation depends on degrees of freedom and sample sizes)
   • Significance threshold: p < 0.05 (omnibus and post-hoc)
   • Assumptions: All data fully observed with no censoring; does not assume normality or constant hazards
"""
doc.add_paragraph(methodology_text)

doc.add_paragraph()

doc.add_heading("1.3. Descriptive Statistics per Group", level=2)

describe_lifetime_statistics(df_summary, doc)

doc.add_paragraph()

doc.add_heading("1.4. EA results", level=2)

# Add all PNG images from the parent data/first_ea folder with descriptions
_ea_images = {
    '2026-05-05_17-42-25_ea_from_random.png': 'EA evolved from random network initialization. Population size = 250, elite = 10. In generation 50: mean = 69.0, median = 74.22, max = 131.59.',
    '2026-05-06_09-34-55_ea_from_lookup_soft.png': 'EA evolved from lookup soft initialization. Population size = 260, elite = 10. In generation 50: mean = 129.3, median = 136.4, max = 195.9.',
    '2026-05-06_11-31-03_ea_from_lookup_hard.png': 'EA evolved from lookup hard initialization. Population size = 260, elite = 10. In generation 50: mean = 125.2, median = 134.5, max = 230.2.',
}

_ea_dir = Path(__file__).resolve().parent.parent  # Go up from replays/ to first_ea/

for _img_file, _description in _ea_images.items():
    _img_path = _ea_dir / _img_file
    if _img_path.exists():
        doc.add_picture(str(_img_path), width=6.5 * 914400)
        doc.add_paragraph(_description)
        doc.add_paragraph()  # Add spacing between images


#endregion # closes 1

#region 2 Survival

doc.add_heading("2. Survival", level=1)

analyze_survival_race(df_summary, doc=doc, group_color_map=COLOR_MAP)

doc.add_paragraph()

analyze_per_run(df_summary, 'lifetime_ticks', 'Survival Time [ticks]', 'lifetime', group_color_map=COLOR_MAP)

analyze_per_tick_metric(df_per_tick, 'energy', 'Energy [units]', 'energy', group_color_map=COLOR_MAP)

#endregion closes 2

#region 3 Genomes
doc.add_heading("3. Genomes", level=1)

doc.add_heading("3.1. Wiring", level=2)

plot_wiring(wiring_data, modulation_data)

analyze_distribution_across_variants(wiring_data, 'weight_effective', 'Effective Connection Weight', n_bins=40, group_color_map=COLOR_MAP)
analyze_distribution_across_variants(wiring_data, 'weight_initial', 'Raw Connection Weight', n_bins=40, group_color_map=COLOR_MAP)

calculate_connectivity(per_neuron_data, wiring_data)
analyze_distribution_across_variants(per_neuron_data, 'in_degree',  'In-Degree',  n_bins=20, group_color_map=COLOR_MAP)
analyze_distribution_across_variants(per_neuron_data, 'out_degree', 'Out-Degree', n_bins=20, group_color_map=COLOR_MAP)

doc.add_heading("3.2. Tonic Activation", level=2)

analyze_distribution_across_variants(per_neuron_data, 'tonic_activation', 'Tonic Activation', n_bins=20, group_color_map=COLOR_MAP)

doc.add_heading("3.3. Plasticity", level=2)

analyze_distribution_across_variants(modulation_data, 'modulation_weight', 'Modulatory Weight', n_bins=40, group_color_map=COLOR_MAP)
analyze_per_variant(variant_data, 'eta', 'Learning rate eta', group_color_map=COLOR_MAP)

#endregion 3 Genomes

#region 4 Food

doc.add_heading("4. Food", level=1)
analyze_per_run(df_summary, 'foods', 'Foods Consumed', 'foods', group_color_map=COLOR_MAP)
analyze_per_tick_metric(df_per_tick, 'food_consumed', 'Food Consumed [per tick]', 'food_consumed', group_color_map=COLOR_MAP)
analyze_foods_consumed_per_direction(df_per_tick, group_color_map=COLOR_MAP)

#endregion 4 Food

#region 5 Movement

doc.add_heading("5. Movement", level=1)
analyze_per_run(df_summary, 'distance', 'Distance Traveled', 'distance', group_color_map=COLOR_MAP)
analyze_per_run(df_summary, 'distance_norm', 'Distance Traveled (normalized by Lifetime)', 'distance_norm', group_color_map=COLOR_MAP)
analyze_per_tick_metric(df_per_tick, 'manhattan_dist', 'Manhattan Distance from origin', 'manhattan_dist', group_color_map=COLOR_MAP)
plot_heatmap_overview()
analyze_movements_per_direction(df_per_tick, group_color_map=COLOR_MAP)

#endregion 5 Movement


#region 6 Decisions

doc.add_heading("6. Decisions", level=1)
analyze_decisions(df_per_tick, group_color_map=COLOR_MAP)

#endregion 6 Decisions



# # ==================================================================================================================================================
# # SECTION E) WRAP UP
# # ==================================================================================================================================================


# Save the report
report_path = Path(__file__).resolve().parent / f"report_{EXPERIMENT_NAME}.docx"
doc.save(report_path)
print(f"Report saved to: {report_path}")
print("\n=== SCRIPT COMPLETE ===")
