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
from scipy.stats import skew, kurtosis, shapiro, f_oneway, kruskal, mannwhitneyu, ttest_ind, studentized_range
try:
    from scipy.integrate import trapezoid as trapz
except ImportError:
    from scipy.integrate import trapz
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import matplotlib.pyplot as plt
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
EXPERIMENT_NAME = "first_ea"  # Used for file naming and report titles

# Experiment data: list of tuples (display_name, hdf5_filename_without_extension)
# The FIRST entry is the primary experiment. All experiments are compared as whole groups.
# Additional entries are treated the same way — each is its own group.
# Leave all but the first commented out if running a single-experiment analysis.
EXPERIMENT_HDF5_FILES = [
    ("EA from Random", "2026-05-05_17-42-25_ea_from_random_genomes_all_runs_all"),
    ("EA from Soft-Coded", "2026-05-06_09-34-55_ea_from_lookup_soft_genomes_all_runs_all"),
   # ("EA from Hard-Coded", "2026-05-06_11-31-03_ea_from_lookup_hard_genomes_all_runs_all"),
]

# Benchmark data (optional): List of tuples (benchmark_display_name, hdf5_filename_without_extension)
# Leave as empty list [] if no benchmarks to compare
BENCHMARK_HDF5_FILES = [
   # ("Random", "2026-05-05_17-29-22_random_genomes_all_runs_all"),
    ("Soft-Coded", "2026-05-05_17-28-28_lookup_soft_genomes_all_runs_all"),
    #("Hard-Coded", "2026-05-05_17-28-53_lookup_hard_genomes_all_runs_all"),
]

# Network visualization configuration (e.g., '11' for network_viz_11.yaml)
NETWORK_VIZ_CONFIG = "11"
RUNS_TO_SHOW_IN_DETAIL = [1,2,3]

# Color scheme for visualizations
# Experiment colors (green family): index 0 = 1st experiment, 1 = 2nd experiment, 2 = 3rd experiment
EXPERIMENT_COLORS = [
    "#0B3D2E",   # 1st experiment: dark forest green
    "#1A6B4A",   # 2nd experiment: mid green
    "#2D9E6B",   # 3rd experiment: lighter green
]
# Benchmark colors (red/pink family): index 0 = 1st benchmark, 1 = 2nd benchmark, 2 = 3rd benchmark
BENCHMARK_COLORS = [
    "#8B3A3A",   # 1st benchmark: wine red
    "#C47070",   # 2nd benchmark: muted rose
    "#F0E2E7",   # 3rd benchmark: light blush pink
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


def _group_overview_stats(df_summary: pd.DataFrame) -> list[dict]:
    """
    Compute per-group overview counts from df_summary.

    Returns a list of dicts (one per group, ordered by appearance in df_summary), each with:
        group, type, hdf5_filename, n_variants, n_runs, runs_per_variant
    """
    # Build filename lookup: group_name -> hdf5 filename string (from user config)
    _filename_lookup = {name: hdf5 for name, hdf5 in EXPERIMENT_HDF5_FILES + BENCHMARK_HDF5_FILES}

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


def analyze_per_run(df_data: pd.DataFrame, metric_col: str, y_label: str, filename_str: str, group_color_map: dict = None) -> None:
    """
    Box plot with jitter overlay per group, summary stats table, and comparative stats table.
    Groups: unsuccessful, successful, plus each benchmark/additional-experiment source as its own group.
    group_color_map: dict mapping original (lowercase) group names to hex color strings.
    """
    # Helper function to classify effect sizes
    def classify_effect_size(effect_size_value: float, effect_size_type: str) -> str:
        """
        Classify effect size as insignificant, small, medium, or large based on type.
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
    
    # Dynamically discover and extract all groups from the data
    all_unique_groups = sorted(df_data['group'].unique())
    
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
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(values))
        ax.scatter(np.full(len(values), i) + jitter, values, color=color, alpha=0.4, s=8, zorder=1)
    
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
        bp['boxes'][i].set_facecolor(color)
        bp['boxes'][i].set_alpha(0.3)
        bp['medians'][i].set_color('black')
    
    ax.set_xticks(positions)
    ax.set_xticklabels(group_order, fontsize=11)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(f'{y_label} by Group', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    figures_dir = Path(__file__).resolve().parent / f'figures_{EXPERIMENT_NAME}'
    figures_dir.mkdir(exist_ok=True)
    output_path = figures_dir / f'groups_{filename_str}.png'
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
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
    
    doc.add_paragraph()
    
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
    doc.add_paragraph()
    
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


def analyze_per_run_direction(df_data: pd.DataFrame, metric_base: str, y_label: str, filename_str: str, group_color_map: dict = None) -> None:
    """
    Analyze a metric stratified by direction within groups.
    Creates box plots and statistics for each group × direction combination.
    
    Args:
        df_data: DataFrame with 'group' column and direction columns (e.g., 'food_sensed_north', 'food_sensed_east', etc.)
        metric_base: Base name for metric columns (e.g., 'food_sensed' to match 'food_sensed_north', 'food_sensed_east', etc.)
        y_label: Label for y-axis (e.g., 'Food Sensed per Tick')
        filename_str: Base name for output files
        group_color_map: dict mapping original (lowercase) group names to hex color strings.
    """
    
    directions = ['north', 'east', 'south', 'west', 'stay']
    
    # Dynamically discover and extract all groups from the data
    all_unique_groups = sorted(df_data['group'].unique())
    
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


def analyze_survival_race(df_summary: pd.DataFrame, group_color_map: dict = None) -> None:
    """
    Plot cumulative survival for each (group, variant) pair.
    Each line shows how many runs are still alive at each tick.
    """
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(12, 8))

    all_groups = list(df_summary['group'].unique())
    _gcm = group_color_map or {}

    # Z-order: experiments on top
    _exp_names = [name for name, _ in EXPERIMENT_HDF5_FILES]
    zorder_map = {
        g: (3 + _exp_names.index(g)) if g in _exp_names else 1
        for g in all_groups
    }

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

    all_groups = sorted(df_per_tick['group'].unique())
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
        ax.plot(gdf['tick'], gdf['mean'], color=color, linewidth=2.5, label=g, zorder=3)
        ax.fill_between(gdf['tick'], gdf['ci_lo'], gdf['ci_hi'], color=color, alpha=0.2, zorder=2)
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
        ax.plot(gdf['ticks_norm'], gdf['mean'], color=color, linewidth=2.5, label=g, zorder=3)
        ax.fill_between(gdf['ticks_norm'], gdf['ci_lo'], gdf['ci_hi'], color=color, alpha=0.2, zorder=2)
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
_per_tick_parts = []   # list[pd.DataFrame]  -->  df_per_tick
modulation_data = {}   # {group_name: {variant_id (int): np.ndarray}}
wiring_data     = {}   # {group_name: {variant_id (int): np.ndarray}}
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
        if heatmap_available:
            heatmap_data[group_name] = {}

        # --- Limit heatmap loading to first 5 variants per group ---
        heatmap_variant_count = 0

        for variant_key in variant_keys:
            variant_id = int(variant_key.split('_')[1])
            vg = f[variant_key]

            # --- 1.1  Load dataset a) summary ---
            df_var = pd.DataFrame(vg['summary'][:])[_SUMMARY_COLS]
            df_var.insert(0, 'variant', variant_id)
            df_var.insert(0, 'type',    data_type)
            df_var.insert(0, 'group',   group_name)
            _summary_parts.append(df_var)

            # --- 1.1b  Load dataset b) modulation and c) wiring ---
            modulation_data[group_name][variant_id] = vg['modulation'][:]
            wiring_data[group_name][variant_id]     = vg['wiring'][:]

            # Check if we should load heatmap for this variant (limit to first 5)
            load_heatmap_for_this_variant = heatmap_available and heatmap_variant_count < 5

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
                    heatmap_variant_count += 1

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
_first_exp_name, _first_exp_hdf5 = EXPERIMENT_HDF5_FILES[0]
print(f"Loading first experiment '{_first_exp_name}' (availability probe)...")
_process_hdf5_file(_find_hdf5_file(_first_exp_hdf5), _first_exp_name, 'experiment',
                   check_availability=True)

# --- Step 2: Remaining experiment files ---
for _exp_name, _exp_hdf5 in EXPERIMENT_HDF5_FILES[1:]:
    print(f"Loading experiment '{_exp_name}'...")
    _process_hdf5_file(_find_hdf5_file(_exp_hdf5), _exp_name, 'experiment')

# --- Step 3: Benchmark files ---
for _bench_name, _bench_hdf5 in BENCHMARK_HDF5_FILES:
    print(f"Loading benchmark '{_bench_name}'...")
    _process_hdf5_file(_find_hdf5_file(_bench_hdf5), _bench_name, 'benchmark')

# --- Assemble final data structures ---
df_summary = pd.concat(_summary_parts, ignore_index=True)
print(f"\ndf_summary:    {len(df_summary)} rows  |  groups: {df_summary['group'].unique().tolist()}")

_n_modulation = sum(len(v) for v in modulation_data.values())
_n_wiring     = sum(len(v) for v in wiring_data.values())
print(f"modulation_data: {len(modulation_data)} groups  |  {_n_modulation} total arrays")
print(f"wiring_data:     {len(wiring_data)} groups  |  {_n_wiring} total arrays")

if per_tick_available:
    df_per_tick = pd.concat(_per_tick_parts, ignore_index=True)
    # Normalised tick position per run: tick 0 → 0 %, last tick → 100 %
    _max_tick_per_run = df_per_tick.groupby(['group', 'variant', 'run'])['tick'].transform('max')
    df_per_tick['ticks_norm'] = (df_per_tick['tick'] / _max_tick_per_run * 100).round(1)
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
# Experiments (all entries in EXPERIMENT_HDF5_FILES) use EXPERIMENT_COLORS[0], [1], [2], ...
# Benchmarks use BENCHMARK_COLORS[0], [1], [2], ...
# Beyond 3 entries of either type, colors are auto-generated in the same hue family.
# ==================================================================================================================================================
import colorsys as _colorsys

COLOR_MAP: dict = {}

# All experiments (EXPERIMENT_HDF5_FILES)
_all_exp_names = [name for name, _ in EXPERIMENT_HDF5_FILES]
for _i, _exp_name in enumerate(_all_exp_names):
    if _i < len(EXPERIMENT_COLORS):
        COLOR_MAP[_exp_name] = EXPERIMENT_COLORS[_i]
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
_benchmark_names = [name for name, _ in BENCHMARK_HDF5_FILES]
for _i, _bench_name in enumerate(_benchmark_names):
    if _i < len(BENCHMARK_COLORS):
        COLOR_MAP[_bench_name] = BENCHMARK_COLORS[_i]
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
print("Creating document...")
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

PER-TICK METRICS (TEMPORAL ANALYSIS)

For metrics tracked per simulation tick (e.g., energy), per-tick data are analyzed using the Area Under the Curve (AUC) method:
   • Each run's per-tick metric values (tick 0 to end of run) are integrated using the trapezoidal rule to produce a single AUC value per run
   • AUC values are then compared across groups using the same adaptive statistical framework as above (Welch's t-test/Mann-Whitney U for 2 groups, ANOVA/Kruskal-Wallis for >2 groups)
   • This approach is conservative and avoids temporal autocorrelation issues by summarizing the time-series into a single magnitude metric
   • Visualizations show individual run curves (low opacity) overlaid with group mean curves (bold) and 95% confidence bands (shaded)

SURVIVAL CURVES

All groups' survival distributions are compared using the log-rank test:
   • Omnibus Test: Log-rank test (rank-based non-parametric test comparing survival distributions)
   • Post-hoc Pairwise Test: Pairwise log-rank tests (all group pairs compared)
   • Multiple Comparisons Correction: Holm–Bonferroni (sequential adjustment of p-value thresholds)
   • Post-hoc Effect Size: χ² per pair (interpretation depends on degrees of freedom and sample sizes)
   • Significance threshold: p < 0.05 (omnibus and post-hoc)
   • Assumptions: All data fully observed with no censoring; does not assume normality or constant hazards
"""
doc.add_paragraph(methodology_text)

doc.add_paragraph()

doc.add_heading("1.3. Descriptive Statistics per Group", level=2)

# One wide table: Statistic | group_0 | group_1 | ...
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

doc.add_paragraph()

doc.add_heading("1.4. EA results", level=2)

doc.add_paragraph("space for EA results here")


#endregion # closes 1

#region 2 Survival


doc.add_heading("2. Survival", level=1)

analyze_survival_race(df_summary, group_color_map=COLOR_MAP)
_lr = _logrank_test(df_summary)

# Omnibus: observed vs. expected table
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


# Pairwise (all group combinations, Holm-Bonferroni corrected)
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

doc.add_paragraph()


analyze_per_run(df_summary, 'lifetime_ticks', 'Survival Time [ticks]', 'lifetime', group_color_map=COLOR_MAP)

analyze_per_tick_metric(df_per_tick, 'energy', 'Energy [units]', 'energy', group_color_map=COLOR_MAP)

#endregion closes 2








#endregion # closes 3.1

# #region 3.2 Food Consumption

# print("  Analyzing food consumption...")
# doc.add_heading("3.2. Food Consumption", level=2)

# analyze_per_run(df_all, 'foods', 'Foods Consumed', 'foods', group_color_map=COLOR_MAP)
# print("  Food consumption analysis done.")


# # Calculate normalized food consumption (foods per tick)
# # Temporarily make df_all writable to add new column
# df_all.flags.writeable = True
# df_all['foods_norm'] = df_all['foods'] / df_all['lifetime_ticks']
# df_all.flags.writeable = False

# # Analyze normalized food consumption
# analyze_per_run(df_all, 'foods_norm', 'Foods Consumed (normalized to life time)', 'foods_norm', group_color_map=COLOR_MAP)

# doc.add_paragraph('The food consumption per tick shows, that EA solution don\'t just randomly find food at the end of their lifes and then life another 17 ticks. The random rate to survive would be 1 food / 17 ticks, which equals 0.05 foods/tick. The exact rate at which random surviving variants feed by chance. The hand-wired solution finds food with about double, the EA solutions with more than tripple that frquency. None of the networks have information about the current energy levels.', style='Normal')

# # Food sensing per direction (from per-tick data)
# if per_tick_included and len(df_all_per_tick) > 0:
#     print("  Aggregating food sensing by direction from per-tick data...")
#     # Check which direction columns exist in per-tick data
#     per_tick_cols = df_all_per_tick.columns.tolist()
#     # Map short direction names (N/E/S/W) to long names (north/east/south/west)
#     direction_map = {'N': 'north', 'E': 'east', 'S': 'south', 'W': 'west'}
#     available_directions_pt = []
#     for short_dir, long_dir in direction_map.items():
#         col_name = f"food_sensed_{short_dir}"
#         if col_name in per_tick_cols:
#             available_directions_pt.append((short_dir, long_dir))
    
#     if available_directions_pt:
#         # Aggregate per_tick food_sensed by direction, per run
#         for short_dir, long_dir in available_directions_pt:
#             col_name = f"food_sensed_{short_dir}"
#             # Sum across all ticks for each (source, variant, run_id)
#             agg_data = df_all_per_tick.groupby(['source', 'variant', 'run_id'])[col_name].sum().reset_index()
#             agg_data.rename(columns={col_name: f'food_sensed_sum_{long_dir}'}, inplace=True)
            
#             # Merge back into df_all
#             df_all.flags.writeable = True
#             # Merge on source, variant, run_id (which are the index keys in df_all)
#             df_all = df_all.merge(agg_data, on=['source', 'variant', 'run_id'], how='left')
#             # Normalize by lifetime_ticks
#             df_all[f'food_sensed_norm_{long_dir}'] = df_all[f'food_sensed_sum_{long_dir}'] / df_all['lifetime_ticks']
#             df_all.flags.writeable = False
        
#         # Analyze food sensing per direction
#         print("  Analyzing food sensing by direction...")
#         analyze_per_run_direction(df_all, 'food_sensed_norm', 'Food Sensed per direction (normalized to life time)', 'food_sensed', group_color_map=COLOR_MAP)
#         print("  Food sensing analysis done.")
#     else:
#         print("  No direction-specific food_sensed columns found in per-tick data.")
#         doc.add_paragraph("TBD - Food sensing by direction data not available")
# else:
#     print("  Per-tick data not available for food sensing direction analysis.")
#     doc.add_paragraph("TBD - Per-tick data required for direction analysis")

# doc.add_paragraph()

# #endregion # closes 3.2

# #region 3.3 Movement

# print("  Analyzing movement...")
# doc.add_heading("3.3. Movement", level=2)

# #region 3.3.1 Movements Made

# doc.add_heading("3.3.1. Movements Made", level=3)

# # Calculate total movements (sum across all directions)
# print("    Calculating total movements...")
# df_all.flags.writeable = True
# if 'moves_north' in df_all.columns:
#     df_all['moves_total'] = df_all['moves_north'] + df_all['moves_south'] + df_all['moves_east'] + df_all['moves_west']
# df_all.flags.writeable = False

# # Analyze total movements
# if 'moves_total' in df_all.columns:
#     analyze_per_run(df_all, 'moves_total', 'Total Movements Made', 'moves_total', group_color_map=COLOR_MAP)
# print("    Total movements analysis done.")

# # Calculate normalized movements (movements per tick)
# print("    Calculating normalized movements...")
# df_all.flags.writeable = True
# if 'moves_total' in df_all.columns:
#     df_all['moves_norm'] = df_all['moves_total'] / df_all['lifetime_ticks']
# df_all.flags.writeable = False

# # Analyze normalized movements
# if 'moves_norm' in df_all.columns:
#     analyze_per_run(df_all, 'moves_norm', 'Movements Made (normalized to life time)', 'moves_norm', group_color_map=COLOR_MAP)
# print("    Normalized movements analysis done.")
# doc.add_paragraph("TBD")

# # Movements per direction (from per-tick data)
# if per_tick_included and len(df_all_per_tick) > 0:
#     print("    Aggregating movements by direction from per-tick data...")
#     # Movement data is in a single 'movement' column with values b'N'/b'E'/b'S'/b'W'/b'stay' (bytes)
#     if 'movement' in df_all_per_tick.columns:
#         # Convert movement column to string if it contains bytes
#         if df_all_per_tick['movement'].dtype == 'object' and isinstance(df_all_per_tick['movement'].iloc[0], bytes):
#             df_all_per_tick['movement'] = df_all_per_tick['movement'].str.decode('utf-8')
        
#         # Map short direction names (N/E/S/W) to long names (north/east/south/west)
#         direction_map = {'N': 'north', 'E': 'east', 'S': 'south', 'W': 'west', 'stay': 'stay'}
#         print(f"      df_all_per_tick shape: {df_all_per_tick.shape}")
#         print(f"      df_all shape before aggregation: {df_all.shape}")
#         print(f"      movement values: {sorted(df_all_per_tick['movement'].unique())}")
        
#         # Aggregate movements by direction, per run
#         for short_dir, long_dir in direction_map.items():
#             # Count occurrences of this direction across all ticks for each (source, variant, run_id)
#             agg_data = df_all_per_tick[df_all_per_tick['movement'] == short_dir].groupby(['source', 'variant', 'run_id']).size().reset_index(name=f'moves_sum_{long_dir}')
#             print(f"      {short_dir} ({long_dir}): {len(agg_data)} rows with data")
            
#             # Merge back into df_all
#             df_all.flags.writeable = True
#             df_all = df_all.merge(agg_data, on=['source', 'variant', 'run_id'], how='left')
#             # Fill NaN (no movements in that direction) with 0
#             df_all[f'moves_sum_{long_dir}'].fillna(0, inplace=True)
#             # Normalize by lifetime_ticks
#             df_all[f'moves_norm_{long_dir}'] = df_all[f'moves_sum_{long_dir}'] / df_all['lifetime_ticks']
#             df_all.flags.writeable = False
#             print(f"      Created moves_norm_{long_dir}: min={df_all[f'moves_norm_{long_dir}'].min():.4f}, max={df_all[f'moves_norm_{long_dir}'].max():.4f}, mean={df_all[f'moves_norm_{long_dir}'].mean():.4f}")
        
#         print(f"      df_all shape after aggregation: {df_all.shape}")
#         print(f"      Columns: {[c for c in df_all.columns if 'moves_norm' in c]}")
        
#         # Analyze movements per direction
#         print("    Analyzing movements by direction...")
#         analyze_per_run_direction(df_all, 'moves_norm', 'Movements Made per direction (normalized to life time)', 'moves', group_color_map=COLOR_MAP)
#         print("    Movements by direction analysis done.")
#     else:
#         print("    'movement' column not found in per-tick data.")
#         doc.add_paragraph("TBD - Movements by direction data not available")
# else:
#     print("    Per-tick data not available for movements direction analysis.")
#     doc.add_paragraph("TBD - Per-tick data required for direction analysis")

# #endregion # closes 3.3.1

# #region 3.3.2 Ground Covered

# doc.add_heading("3.3.2. Ground Covered", level=3)

# # Per-tick manhattan distance analysis
# if per_tick_included:
#     print("    Analyzing per-tick distance...")
#     analyze_per_tick_metric(df_all_per_tick, 'manhattan_dist', 'Manhattan Distance [units]', 'distance', group_color_map=COLOR_MAP)
#     print("    Per-tick distance analysis done.")
# doc.add_paragraph("TBD")

# # Heatmaps: staying and entering
# if heatmap_data:
#     print("    Analyzing heatmaps...")
#     analyze_heatmaps(heatmap_data, df_all)
#     print("    Heatmap analysis done.")
# doc.add_paragraph("TBD")

# #endregion # closes 3.3.2

# #endregion # closes 3.3

# #region 3.4 Decisions

# print("  Analyzing decisions...")
# doc.add_heading("3.4. Decisions", level=2)

# #region 3.4.1 Decisions Made

# doc.add_heading("3.4.1. Decisions Made", level=3)

# # Analyze total decisions
# print("    Analyzing total decisions...")
# if 'decisions' in df_all.columns:
#     analyze_per_run(df_all, 'decisions', 'no. decisions', 'decisions', group_color_map=COLOR_MAP)
# print("    Total decisions analysis done.")

# # Calculate normalized decisions (decisions per tick)
# print("    Calculating normalized decisions...")
# df_all.flags.writeable = True
# if 'decisions' in df_all.columns:
#     df_all['decisions_norm'] = df_all['decisions'] / df_all['lifetime_ticks']
# df_all.flags.writeable = False

# # Analyze normalized decisions
# if 'decisions_norm' in df_all.columns:
#     analyze_per_run(df_all, 'decisions_norm', 'decisions per tick', 'decisions_norm', group_color_map=COLOR_MAP)
# print("    Normalized decisions analysis done.")
# doc.add_paragraph("TBD")

# #endregion # closes 3.4.1

# #region 3.4.2 Correct Decisions

# doc.add_heading("3.4.2. Correct Decisions", level=3)

# # Analyze total correct decisions
# print("    Analyzing correct decisions...")
# if 'correct_decisions' in df_all.columns:
#     analyze_per_run(df_all, 'correct_decisions', 'no. \'correct decisions\'', 'correct_decisions', group_color_map=COLOR_MAP)
# print("    Correct decisions analysis done.")

# # Calculate normalized correct decisions (correct decisions per tick)
# print("    Calculating normalized correct decisions...")
# df_all.flags.writeable = True
# if 'correct_decisions' in df_all.columns:
#     df_all['correct_decisions_norm'] = df_all['correct_decisions'] / df_all['lifetime_ticks']
# df_all.flags.writeable = False

# # Analyze normalized correct decisions
# if 'correct_decisions_norm' in df_all.columns:
#     analyze_per_run(df_all, 'correct_decisions_norm', '\'correct\' decisions per tick', 'correct_decisions_norm', group_color_map=COLOR_MAP)
# print("    Normalized correct decisions analysis done.")
# doc.add_paragraph("TBD")

# # Analyze correct decisions by direction
# if per_tick_included:
#     print("    Analyzing decision precision by direction...")
#     df_all = track_decision_precision(df_all_per_tick, df_all)
#     analyze_per_run_direction(df_all, 'decision_precision', 'Decision Precision by Direction (fraction correct)', 'decision_precision', group_color_map=COLOR_MAP)
#     print("    Decision precision analysis done.")
#     doc.add_paragraph("TBD")

# #endregion # closes 3.4.2

# #endregion # closes 3.4

# #endregion # closes 3 (remaining sub-regions TBD)

#endregion # closes 3

# #region 4 Summary

# doc.add_heading("4. Summary", level=1)

# doc.add_paragraph("TBD")

# #endregion # closes 4


# # ==================================================================================================================================================
# # SECTION E) WRAP UP
# # ==================================================================================================================================================

# print("\n=== SECTION E: WRAP UP ===")
print("Saving report...")
# Save the report
report_path = Path(__file__).resolve().parent / f"report_{EXPERIMENT_NAME}.docx"
doc.save(report_path)
print(f"Report saved to: {report_path}")
print("\n=== SCRIPT COMPLETE ===")
