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


def analyze_per_run(df_data: pd.DataFrame, metric_col: str, y_label: str, filename_str: str) -> None:
    """
    Box plot with jitter overlay per group, summary stats table, and comparative stats table.
    Groups: unsuccessful, successful, plus each benchmark source as its own group.
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
    
    # Dynamically discover and extract all groups from the data, excluding 'other'
    all_unique_groups = [g for g in sorted(df_data['group'].unique()) if g != 'other']
    
    # Mapping for display names - capitalize experiment groups, keep benchmark names as-is
    display_name_map = {
        'unsuccessful': 'Unsuccessful',
        'successful': 'Successful',
        'other': 'Other',
    }
    
    group_data = {}
    group_order = []
    
    for group_name in all_unique_groups:
        df_group = df_data[df_data['group'] == group_name]
        if len(df_group) > 0:
            # Use display name if available, otherwise use group name as-is (for benchmarks)
            display_name = display_name_map.get(group_name, group_name)
            group_data[display_name] = df_group[metric_col].values
            group_order.append(display_name)
    
    # --- Plot: box plot with jitter ---
    color_map = {
        'Unsuccessful': SECONDARY_COLOR,
        'Successful': PRIMARY_COLOR,
    }
    default_bench_color = TERTIARY_COLOR
    
    fig, ax = plt.subplots(figsize=(10, 7))
    positions = list(range(len(group_order)))
    
    # Box plots
    bp = ax.boxplot(
        [group_data[g] for g in group_order],
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
    )
    for i, g in enumerate(group_order):
        color = color_map.get(g, default_bench_color)
        bp['boxes'][i].set_facecolor(color)
        bp['boxes'][i].set_alpha(0.3)
        bp['medians'][i].set_color('black')
    
    # Jitter overlay
    for i, g in enumerate(group_order):
        values = group_data[g]
        color = color_map.get(g, default_bench_color)
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(values))
        ax.scatter(np.full(len(values), i) + jitter, values, color=color, alpha=0.4, s=8, zorder=3)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(group_order, fontsize=11)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(f'{y_label} by Group', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    figures_dir = Path(__file__).resolve().parent / 'figures'
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


def analyze_per_run_direction(df_data: pd.DataFrame, metric_base: str, y_label: str, filename_str: str) -> None:
    """
    Analyze a metric stratified by direction within groups.
    Creates box plots and statistics for each group × direction combination.
    
    Args:
        df_data: DataFrame with 'group' column and direction columns (e.g., 'food_sensed_north', 'food_sensed_east', etc.)
        metric_base: Base name for metric columns (e.g., 'food_sensed' to match 'food_sensed_north', 'food_sensed_east', etc.)
        y_label: Label for y-axis (e.g., 'Food Sensed per Tick')
        filename_str: Base name for output files
    """
    
    directions = ['north', 'east', 'south', 'west', 'stay']
    
    # Dynamically discover and extract all groups from the data, excluding 'other'
    all_unique_groups = [g for g in sorted(df_data['group'].unique()) if g != 'other']
    
    # Mapping for display names - capitalize experiment groups, keep benchmark names as-is
    display_name_map = {
        'unsuccessful': 'Unsuccessful',
        'successful': 'Successful',
        'other': 'Other',
    }
    
    # Prepare data: for each group and direction, collect normalized values
    group_direction_data = {}  # {(display_name, direction): [values]}
    group_order = []
    
    for group_name in all_unique_groups:
        display_name = display_name_map.get(group_name, group_name)
        if display_name not in group_order:
            group_order.append(display_name)
        
        df_group = df_data[df_data['group'] == group_name]
        
        for direction in directions:
            col_name = f"{metric_base}_{direction}"
            if col_name in df_data.columns:
                group_direction_data[(display_name, direction)] = df_group[col_name].values
    
    if not group_direction_data:
        doc.add_paragraph(f"No direction data found for {metric_base}.")
        return
    
    # --- Plot: box plot with directions stratified within groups ---
    color_map = {
        'Unsuccessful': SECONDARY_COLOR,
        'Successful': PRIMARY_COLOR,
        'Other': '#808080',
    }
    default_bench_color = TERTIARY_COLOR
    
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
    
    # Box plots
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
        color = color_map.get(group, default_bench_color)
        for direction in direction_order:
            key = (group, direction)
            if key in group_direction_data:
                if box_idx < len(bp['boxes']):
                    bp['boxes'][box_idx].set_facecolor(color)
                    bp['boxes'][box_idx].set_alpha(0.3)
                    bp['medians'][box_idx].set_color('black')
                box_idx += 1
    
    # Jitter overlay
    box_idx = 0
    for group_idx, group in enumerate(group_order):
        for dir_idx, direction in enumerate(direction_order):
            key = (group, direction)
            if key in group_direction_data:
                values = group_direction_data[key]
                color = color_map.get(group, default_bench_color)
                jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(values))
                ax.scatter(np.full(len(values), all_positions[box_idx]) + jitter, values, color=color, alpha=0.4, s=8, zorder=3)
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
    figures_dir = Path(__file__).resolve().parent / 'figures'
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


def analyze_heatmaps(heatmap_data: dict, df_groups: pd.DataFrame) -> None:
    """
    Generate and display heatmaps (staying and entering) for groups and variants.
    
    Skips 'other' group. For each variant in selected groups, creates a figure showing:
    - Top-left: Averaged heatmap across all runs (with own colorbar)
    - Other 3 panels: Raw data heatmaps from first 3 runs (with shared colorbar, if available)
    
    Args:
        heatmap_data: Dict mapping source -> variant -> {'staying': [...], 'entering': [...]}
        df_groups: DataFrame with group assignments (must have 'source', 'variant', 'group' columns)
    """
    
    # Return early if no heatmap data
    if not heatmap_data:
        return
    
    # Build group -> (source, variants) mapping, excluding 'other' group
    group_variants = {}  # {group_name: [(source, variant_name), ...]}
    for source in sorted(heatmap_data.keys()):
        for variant_name in sorted(heatmap_data[source].keys()):
            # Find group for this source/variant
            if source == 'experiment':
                mask = (df_groups['source'] == source) & (df_groups['variant'] == variant_name)
                if mask.any():
                    group = df_groups.loc[mask, 'group'].iloc[0]
                else:
                    continue
            else:
                # For benchmarks, the group is the source name
                group = source
            
            # Skip 'other' group entirely
            if group == 'other':
                continue
            
            if group not in group_variants:
                group_variants[group] = []
            group_variants[group].append((source, variant_name))
    
    # For each group and variant, create a figure
    figures_dir = Path(__file__).resolve().parent / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    for group_name in sorted(group_variants.keys()):
        source_variant_pairs = group_variants[group_name]
        
        # Limit to first 3 variants per group
        variants_to_process = source_variant_pairs[:3]
        
        # Process each variant in the group (create separate figure for each)
        for source, variant_name in variants_to_process:
            
            # Create figures for both staying and entering
            for heatmap_type in ['staying', 'entering']:
                
                # Get all runs for this variant
                arrays = heatmap_data[source][variant_name].get(heatmap_type, [])
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
                output_filename = f'heatmap_{group_display_name}_{variant_name}_{heatmap_type}.png'
                output_path = figures_dir / output_filename
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                # Add to document
                doc.add_picture(str(output_path), width=5.0 * 914400)
    doc.add_paragraph()


def analyze_survival_race(df_data: pd.DataFrame) -> None:
    """
    Plot cumulative survival for each variant.
    Each line shows how many runs are still alive at each tick.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create unique identifier combining source and variant
    df_data['unique_variant'] = df_data['source'] + '_' + df_data['variant']
    
    # Get all unique groups and separate into experiment groups and benchmarks
    all_groups = sorted(df_data['group'].unique())
    experiment_groups = [g for g in all_groups if g in ['successful', 'unsuccessful', 'other']]
    benchmark_groups = [g for g in all_groups if g not in ['successful', 'unsuccessful', 'other', 'unknown']]
    
    # Color mapping by group - dynamic based on actual groups
    color_map = {
        'successful': PRIMARY_COLOR,
        'unsuccessful': SECONDARY_COLOR,
        'other': '#808080',
    }
    for benchmark_group in benchmark_groups:
        color_map[benchmark_group] = TERTIARY_COLOR
    
    # Z-order mapping: higher for more important groups (benchmarks are always top)
    zorder_map = {
        'other': 1,
        'unsuccessful': 2,
        'successful': 3,
    }
    for idx, benchmark_group in enumerate(sorted(benchmark_groups)):
        zorder_map[benchmark_group] = 4 + idx
    
    # Plot one line per unique variant
    for unique_var in sorted(df_data['unique_variant'].unique()):
        variant_data = df_data[df_data['unique_variant'] == unique_var]
        survival_times = variant_data['lifetime_ticks'].values
        group = variant_data['group'].iloc[0]
        color = color_map.get(group, '#808080')
        zorder = zorder_map.get(group, 1)
        
        # Create tick array from 0 to max lifetime
        max_tick = int(survival_times.max())
        ticks = np.arange(0, max_tick + 1)
        
        # Count how many runs are still alive at each tick
        alive_counts = np.array([(survival_times >= t).sum() for t in ticks])
        
        # Plot this variant's survival curve with z-order
        ax.plot(ticks, alive_counts, color=color, zorder=zorder)
    
    # Create custom legend dynamically based on actual groups
    from matplotlib.lines import Line2D
    legend_elements = []
    
    # Add experiment groups in order
    if 'successful' in color_map:
        legend_elements.append(Line2D([0], [0], color=PRIMARY_COLOR, linewidth=2, label='Successful'))
    if 'unsuccessful' in color_map:
        legend_elements.append(Line2D([0], [0], color=SECONDARY_COLOR, linewidth=2, label='Unsuccessful'))
    if 'other' in color_map:
        legend_elements.append(Line2D([0], [0], color='#808080', linewidth=2, label='Other'))
    
    # Add benchmark groups
    for benchmark_group in sorted(benchmark_groups):
        legend_elements.append(Line2D([0], [0], color=TERTIARY_COLOR, linewidth=2, label=benchmark_group))
    
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


def track_decision_precision(df_per_tick: pd.DataFrame, df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate decision precision per direction for each run.
    
    For each run in df_all_per_tick, tracks decisions and their correctness.
    Adds decision_precision_{direction} columns to df_all.
    
    Args:
        df_per_tick: Per-tick DataFrame with columns: tick, decision_made, movement, food_consumed, run_id, variant, group, source
        df_all: Run-level DataFrame to add precision columns to (must have run_id column)
    
    Returns:
        df_all with added decision_precision_{direction} columns (north, east, south, west, stay)
        Unmatched rows in df_all remain NaN (intentional)
    """
    
    # Initialize precision columns with NaN
    df_all.flags.writeable = True
    for direction in ['north', 'east', 'south', 'west', 'stay']:
        df_all[f'decision_precision_{direction}'] = np.nan
    
    # Process all runs in df_per_tick, grouping by (variant, source, run_id) 
    # because run_id 1-300 appears in each variant/source combo
    for (variant, source, run_id), group_data in df_per_tick.groupby(['variant', 'source', 'run_id']):
        run_data = group_data.reset_index(drop=True)
        
        # Track decisions and correctness by direction
        decision_counts = {'north': 0, 'east': 0, 'south': 0, 'west': 0, 'stay': 0}
        correct_counts = {'north': 0, 'east': 0, 'south': 0, 'west': 0, 'stay': 0}
        
        # Step through ticks
        for tick_idx in range(len(run_data)):
            tick_row = run_data.iloc[tick_idx]
            
            # Check if a decision was made
            if tick_row['decision_made'] == 1:
                # Get the movement/decision type
                movement = tick_row['movement']
                
                # Decode if bytes, convert to string and lowercase
                if isinstance(movement, bytes):
                    movement = movement.decode('utf-8')
                direction = movement.lower()
                
                # Map single letters to full direction names
                direction_map = {'n': 'north', 'e': 'east', 's': 'south', 'w': 'west', 'stay': 'stay'}
                direction = direction_map.get(direction, direction)
                
                # Ensure valid direction
                if direction not in decision_counts:
                    continue
                
                # Tally the decision
                decision_counts[direction] += 1
                
                # Check correctness
                is_correct = False
                if direction == 'stay':
                    # Stay is correct if food was consumed in same tick
                    is_correct = tick_row['food_consumed'] == 1
                else:
                    # Movement is correct if food was consumed in NEXT tick
                    if tick_idx + 1 < len(run_data):
                        next_row = run_data.iloc[tick_idx + 1]
                        is_correct = next_row['food_consumed'] == 1
                
                # Increment correct counter if applicable
                if is_correct:
                    correct_counts[direction] += 1
        
        # Calculate precision for each direction and add to df_all
        for direction in ['north', 'east', 'south', 'west', 'stay']:
            if decision_counts[direction] > 0:
                precision = correct_counts[direction] / decision_counts[direction]
                # Match by variant, source, AND run_id to update the specific run
                mask = (df_all['variant'] == variant) & (df_all['source'] == source) & (df_all['run_id'] == run_id)
                df_all.loc[mask, f'decision_precision_{direction}'] = precision
    
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
    
    # Define which groups to include
    groups_to_include = {'successful', 'unsuccessful'}
    
    # Load per_tick data from experiment
    try:
        with h5py.File(experiment_hdf5_path, 'r') as f:
            variant_names = sorted([k for k in f.keys() if k.startswith('variant_')])
            for variant_name in variant_names:
                # Check if this variant is in one of the groups we want
                variant_group_assignment = variant_groups.get(variant_name)
                if variant_group_assignment not in groups_to_include:
                    continue  # Skip 'other' variants
                
                variant_group = f[variant_name]
                run_names = sorted([k for k in variant_group.keys() if k.startswith('run_')])
                
                for run_name in run_names:
                    if 'per_tick' not in variant_group[run_name]:
                        continue
                    
                    run_group = variant_group[run_name]
                    per_tick_table = run_group['per_tick']
                    
                    # Convert to DataFrame
                    if isinstance(per_tick_table, h5py.Dataset):
                        per_tick_df = pd.DataFrame(per_tick_table[:])
                        per_tick_df['source'] = 'experiment'
                        per_tick_df['variant'] = variant_name
                        per_tick_df['group'] = variant_group_assignment
                        # Extract numeric run_id from run_name (e.g., "run_1" -> 1) to match df_all
                        run_id_numeric = int(run_name.split('_')[1])
                        per_tick_df['run_id'] = run_id_numeric
                        all_per_tick_data.append(per_tick_df)
    except Exception as e:
        print(f"Warning: Error loading experiment per_tick data: {e}")
        raise ValueError("Failed to load experiment per_tick data") from e
    
    # Load per_tick data from benchmarks
    for source, benchmark_path in hdf5_files_dict.items():
        if source == 'experiment':
            continue
        
        try:
            with h5py.File(benchmark_path, 'r') as f:
                variant_names = sorted([k for k in f.keys() if k.startswith('variant_')])
                for variant_name in variant_names:
                    # Benchmarks are their own group
                    variant_group = f[variant_name]
                    run_names = sorted([k for k in variant_group.keys() if k.startswith('run_')])
                    
                    for run_name in run_names:
                        if 'per_tick' not in variant_group[run_name]:
                            continue
                        
                        run_group = variant_group[run_name]
                        per_tick_table = run_group['per_tick']
                        
                        # Convert to DataFrame
                        if isinstance(per_tick_table, h5py.Dataset):
                            per_tick_df = pd.DataFrame(per_tick_table[:])
                            per_tick_df['source'] = source
                            per_tick_df['variant'] = variant_name
                            per_tick_df['group'] = source
                            # Extract numeric run_id from run_name (e.g., "run_1" -> 1) to match df_all
                            run_id_numeric = int(run_name.split('_')[1])
                            per_tick_df['run_id'] = run_id_numeric
                            all_per_tick_data.append(per_tick_df)
        except Exception as e:
            print(f"Warning: Error loading benchmark '{source}' per_tick data: {e}")
    
    # Combine all per_tick data
    if not all_per_tick_data:
        raise ValueError("No per_tick data found")
    
    df_all_per_tick = pd.concat(all_per_tick_data, ignore_index=True)
    # Make immutable
    df_all_per_tick.flags.writeable = False
    print(f"\nPer-tick dataset: {len(df_all_per_tick)} total ticks across {df_all_per_tick['variant'].nunique()} variants")
    
    return df_all_per_tick


def analyze_per_tick_metric(df_per_tick: pd.DataFrame, metric_name: str, y_label: str, filename_str: str) -> bool:
    """
    Analyze per-tick metrics (e.g., energy per tick) for different groups.
    
    Uses pre-loaded per_tick data from df_all_per_tick.
    Plots individual run curves with group means and 95% CI bands.
    Calculates AUC for each run and performs statistical comparison between groups.
    
    Args:
        df_per_tick: Pre-loaded per-tick DataFrame with columns: tick-level metrics, source, variant, group, run_id
        metric_name: Name of the metric column in per_tick data (e.g., 'energy', 'manhattan_dist')
        y_label: Label for y-axis in plots
        filename_str: Base name for output files
    
    Returns:
        True if metric successfully analyzed, False if metric column not found
    """
    
    # Check if metric column exists in the dataframe
    if metric_name not in df_per_tick.columns:
        print(f"Warning: Metric '{metric_name}' not found in per_tick data columns")
        return False
    
    # Build per_tick_auc_data and per_tick_curves from the dataframe
    per_tick_auc_data = []  # List of dicts: {run_id, source, variant, group, auc}
    per_tick_curves = {}  # Dict: {(source, variant, run_id): (ticks, metric_values)}
    
    # Group by (source, variant, run_id) to extract curves and calculate AUC
    # This avoids collisions when same run_id appears in different sources/variants
    for (source, variant, run_id), group_data in df_per_tick.groupby(['source', 'variant', 'run_id']):
        # Sort by tick to ensure correct order (if there's a tick column)
        group_data = group_data.sort_values('tick') if 'tick' in group_data.columns else group_data
        
        # Extract metric values
        metric_values = group_data[metric_name].values
        ticks = np.arange(len(metric_values))
        
        # Calculate AUC using trapezoidal rule
        auc_value = trapz(metric_values, ticks)
        
        # Get group info from first row
        first_row = group_data.iloc[0]
        group = first_row['group']
        
        per_tick_auc_data.append({
            'run_id': run_id,
            'source': source,
            'variant': variant,
            'group': group,
            'auc': auc_value
        })
        
        # Use composite key to store curves
        per_tick_curves[(source, variant, run_id)] = (ticks, metric_values)
    
    if len(per_tick_auc_data) == 0:
        print(f"Warning: No per_tick curves found for metric '{metric_name}'")
        return False
    
    # Create DataFrame of AUC values for statistical analysis
    df_auc = pd.DataFrame(per_tick_auc_data)
    
    # Exclude 'other' group from per-tick analysis
    df_auc = df_auc[df_auc['group'] != 'other']
    
    # Discover all unique groups in df_auc
    all_groups = sorted(df_auc['group'].unique())
    
    # Map display names for experiment groups
    display_name_map = {
        'unsuccessful': 'Unsuccessful',
        'successful': 'Successful',
        'other': 'Other',
    }
    
    # Create color map dynamically for all groups
    color_map = {
        'unsuccessful': SECONDARY_COLOR,
        'successful': PRIMARY_COLOR,
        'other': '#808080',
    }
    default_bench_color = TERTIARY_COLOR
    for group in all_groups:
        if group not in color_map:
            color_map[group] = default_bench_color
    
    # Plot: individual curves + group means with CI
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot individual curves with low alpha, grouped by group
    for group in all_groups:
        group_data = df_auc[df_auc['group'] == group]
        color = color_map.get(group, '#808080')
        
        for _, row in group_data.iterrows():
            source = row['source']
            variant = row['variant']
            run_id = row['run_id']
            curve_key = (source, variant, run_id)
            
            if curve_key in per_tick_curves:
                ticks, metric_values = per_tick_curves[curve_key]
                ax.plot(ticks, metric_values, color=color, alpha=0.15, linewidth=0.8, zorder=1)
    
    # Calculate and plot group means with 95% CI (NO INTERPOLATION - raw data only)
    for group in all_groups:
        group_data = df_auc[df_auc['group'] == group]
        
        # Collect curves for this group (raw data only)
        curves = []
        for _, row in group_data.iterrows():
            source = row['source']
            variant = row['variant']
            run_id = row['run_id']
            curve_key = (source, variant, run_id)
            
            if curve_key in per_tick_curves:
                ticks, metric_values = per_tick_curves[curve_key]
                curves.append(metric_values)
        
        if len(curves) == 0:
            continue
        
        # Calculate mean at each tick without any interpolation
        # For each tick position, average only runs that have data at that tick
        max_length = max(len(c) for c in curves)
        mean_curve = []
        ci_lower_list = []
        ci_upper_list = []
        
        for tick_idx in range(max_length):
            # Get values from all runs that have this tick (some runs may have ended earlier)
            values_at_tick = [c[tick_idx] for c in curves if tick_idx < len(c)]
            
            if values_at_tick:
                mean_val = np.mean(values_at_tick)
                std_val = np.std(values_at_tick)
                n = len(values_at_tick)
                sem = std_val / np.sqrt(n)
                
                mean_curve.append(mean_val)
                ci_lower_list.append(mean_val - 1.96 * sem)
                ci_upper_list.append(mean_val + 1.96 * sem)
        
        mean_curve = np.array(mean_curve)
        ci_lower = np.array(ci_lower_list)
        ci_upper = np.array(ci_upper_list)
        group_ticks = np.arange(len(mean_curve))
        
        color = color_map.get(group, '#808080')
        display_label = display_name_map.get(group, group)
        
        # Plot CI as shaded region
        ax.fill_between(group_ticks, ci_lower, ci_upper, color=color, alpha=0.2, zorder=2)
        
        # Plot mean curve
        ax.plot(group_ticks, mean_curve, color=color, linewidth=2.5, label=display_label, zorder=3)
    
    ax.set_xlabel('Ticks', fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(f'{y_label} per Tick by Group', fontsize=14)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Save figure
    figures_dir = Path(__file__).resolve().parent / 'figures'
    figures_dir.mkdir(exist_ok=True)
    output_path = figures_dir / f'ticks_{filename_str}.png'
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Add to document
    doc.add_picture(str(output_path), width=6.5 * 914400)
    doc.add_paragraph()
    
    # Perform statistical analysis on AUC values using analyze_per_run
    # Filter out 'other' group before statistical analysis
    df_auc_for_stats = df_auc[df_auc['group'] != 'other']
    if len(df_auc_for_stats) > 0:
        analyze_per_run(df_auc_for_stats, 'auc', f'Area Under Curve (AUC) of {y_label}', filename_str)
    
    return True


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

# HDF5 files dictionary for per-tick analyses
hdf5_files_dict = {'experiment': experiment_hdf5_path}
if BENCHMARK_HDF5_FILES:
    for benchmark_name, benchmark_hdf5 in BENCHMARK_HDF5_FILES:
        benchmark_path = _find_hdf5_file(benchmark_hdf5)
        hdf5_files_dict[benchmark_name] = benchmark_path

# Heatmap data: {source: {variant: {'staying': [arrays...], 'entering': [arrays...]}}}
# Check if heatmap data exists first
heatmap_data_exists = False
try:
    with h5py.File(experiment_hdf5_path, 'r') as f:
        variant_keys = sorted([k for k in f.keys() if k.startswith('variant_')])
        if variant_keys:
            first_variant = variant_keys[0]
            variant_group = f[first_variant]
            run_keys = sorted([k for k in variant_group.keys() if k.startswith('run_')])
            if run_keys:
                first_run = run_keys[0]
                run_group = variant_group[first_run]
                if 'staying' in run_group and 'entering' in run_group:
                    heatmap_data_exists = True
except Exception:
    heatmap_data_exists = False

# Load heatmap data if it exists
heatmap_data = {}  # {source: {variant: {'staying': [arrays...], 'entering': [arrays...]}}}
if heatmap_data_exists:
    for source, hdf5_path in hdf5_files_dict.items():
        heatmap_data[source] = {}
        try:
            with h5py.File(hdf5_path, 'r') as f:
                variant_keys = sorted([k for k in f.keys() if k.startswith('variant_')])
                for variant_name in variant_keys:
                    variant_group = f[variant_name]
                    run_keys = sorted([k for k in variant_group.keys() if k.startswith('run_')])
                    
                    staying_arrays = []
                    entering_arrays = []
                    
                    for run_name in run_keys:
                        run_group = variant_group[run_name]
                        if 'staying' in run_group:
                            staying_arrays.append(run_group['staying'][:])
                        if 'entering' in run_group:
                            entering_arrays.append(run_group['entering'][:])
                    
                    if staying_arrays or entering_arrays:
                        heatmap_data[source][variant_name] = {
                            'staying': staying_arrays,
                            'entering': entering_arrays
                        }
        except Exception:
            continue


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

doc.add_paragraph()

# Add statistical methodology section
doc.add_heading("Statistical Analysis Methodology", level=2)
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
"""
doc.add_paragraph(methodology_text)

doc.add_paragraph()

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
        return row['source']  # Use benchmark source name as the group

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

# ==================================================================================================================================================
# Check if per_tick data is available and load it
# ==================================================================================================================================================

per_tick_included = False

# Check if per_tick data exists by looking at first successful variant
if first_successful:
    try:
        with h5py.File(experiment_hdf5_path, 'r') as f:
            if first_successful in f:
                variant_group = f[first_successful]
                # Get first run
                run_names = sorted([k for k in variant_group.keys() if k.startswith('run_')])
                if run_names:
                    first_run = run_names[0]
                    run_group = variant_group[first_run]
                    # Check for per_tick table
                    if 'per_tick' in run_group:
                        per_tick_included = True
    except Exception:
        per_tick_included = False

# Load per_tick data if available
if per_tick_included:
    try:
        df_all_per_tick = load_per_tick_data(experiment_hdf5_path, variant_groups, hdf5_files_dict)
    except ValueError:
        per_tick_included = False

#endregion # closes 2

#region 3 Comparison Successful vs. Unsuccessful (vs. Benchmarks)

doc.add_heading("3. Comparison Successful vs. Unsuccessful (vs. Benchmarks)", level=1)

#region 3.1 Survival

doc.add_heading("3.1. Survival", level=2)

analyze_per_run(df_all, 'lifetime_ticks', 'Survival Time [ticks]', 'survival_times')
doc.add_paragraph('Description of survival time differences between groups, statistical test results, and interpretation goes here.', style='Normal')
doc.add_paragraph()

# Per-tick energy analysis
if per_tick_included:
    analyze_per_tick_metric(df_all_per_tick, 'energy', 'Energy [units]', 'energy')
doc.add_paragraph("TBD")

#endregion # closes 3.1

#region 3.2 Food Consumption

doc.add_heading("3.2. Food Consumption", level=2)

analyze_per_run(df_all, 'foods', 'Foods Consumed', 'foods')


# Calculate normalized food consumption (foods per tick)
# Temporarily make df_all writable to add new column
df_all.flags.writeable = True
df_all['foods_norm'] = df_all['foods'] / df_all['lifetime_ticks']
df_all.flags.writeable = False

# Analyze normalized food consumption
analyze_per_run(df_all, 'foods_norm', 'Foods Consumed (normalized to life time)', 'foods_norm')

doc.add_paragraph('Description of food consumption differences between groups, statistical test results, and interpretation goes here.', style='Normal')

# Food sensing per direction (normalized by lifetime_ticks)
df_all.flags.writeable = True
for direction in ['north', 'east', 'south', 'west', 'stay']:
    col_name = f"food_sensed_{direction}"
    if col_name in df_all.columns:
        df_all[f"food_sensed_norm_{direction}"] = df_all[col_name] / df_all['lifetime_ticks']
df_all.flags.writeable = False

# Analyze food sensing per direction
analyze_per_run_direction(df_all, 'food_sensed_norm', 'Food Sensed per direction (normalized to life time)', 'food_sensed')
doc.add_paragraph("TBD")

doc.add_paragraph()

#endregion # closes 3.2

#region 3.3 Movement

doc.add_heading("3.3. Movement", level=2)

#region 3.3.1 Movements Made

doc.add_heading("3.3.1. Movements Made", level=3)

# Calculate total movements (sum across all directions)
df_all.flags.writeable = True
df_all['moves_total'] = df_all['moves_north'] + df_all['moves_south'] + df_all['moves_east'] + df_all['moves_west']
df_all.flags.writeable = False

# Analyze total movements
analyze_per_run(df_all, 'moves_total', 'Total Movements Made', 'moves_total')

# Calculate normalized movements (movements per tick)
df_all.flags.writeable = True
df_all['moves_norm'] = df_all['moves_total'] / df_all['lifetime_ticks']
df_all.flags.writeable = False

# Analyze normalized movements
analyze_per_run(df_all, 'moves_norm', 'Movements Made (normalized to life time)', 'moves_norm')
doc.add_paragraph("TBD")

# Movements per direction (normalized by lifetime_ticks)
df_all.flags.writeable = True
for direction in ['north', 'east', 'south', 'west', 'stay']:
    col_name = f"moves_{direction}"
    if col_name in df_all.columns:
        df_all[f"moves_norm_{direction}"] = df_all[col_name] / df_all['lifetime_ticks']
df_all.flags.writeable = False

# Analyze movements per direction
analyze_per_run_direction(df_all, 'moves_norm', 'Movements Made per direction (normalized to life time)', 'moves')
doc.add_paragraph("TBD")

#endregion # closes 3.3.1

#region 3.3.2 Ground Covered

doc.add_heading("3.3.2. Ground Covered", level=3)

# Per-tick manhattan distance analysis
if per_tick_included:
    analyze_per_tick_metric(df_all_per_tick, 'manhattan_dist', 'Manhattan Distance [units]', 'distance')
doc.add_paragraph("TBD")

# Heatmaps: staying and entering
analyze_heatmaps(heatmap_data, df_all)
doc.add_paragraph("TBD")

#endregion # closes 3.3.2

#endregion # closes 3.3

#region 3.4 Decisions

doc.add_heading("3.4. Decisions", level=2)

#region 3.4.1 Decisions Made

doc.add_heading("3.4.1. Decisions Made", level=3)

# Analyze total decisions
analyze_per_run(df_all, 'decisions', 'no. decisions', 'decisions')

# Calculate normalized decisions (decisions per tick)
df_all.flags.writeable = True
df_all['decisions_norm'] = df_all['decisions'] / df_all['lifetime_ticks']
df_all.flags.writeable = False

# Analyze normalized decisions
analyze_per_run(df_all, 'decisions_norm', 'decisions per tick', 'decisions_norm')
doc.add_paragraph("TBD")

#endregion # closes 3.4.1

#region 3.4.2 Correct Decisions

doc.add_heading("3.4.2. Correct Decisions", level=3)

# Analyze total correct decisions
analyze_per_run(df_all, 'correct_decisions', 'no. \'correct decisions\'', 'correct_decisions')

# Calculate normalized correct decisions (correct decisions per tick)
df_all.flags.writeable = True
df_all['correct_decisions_norm'] = df_all['correct_decisions'] / df_all['lifetime_ticks']
df_all.flags.writeable = False

# Analyze normalized correct decisions
analyze_per_run(df_all, 'correct_decisions_norm', '\'correct\' decisions per tick', 'correct_decisions_norm')
doc.add_paragraph("TBD")

# Analyze correct decisions by direction
if per_tick_included:
    df_all = track_decision_precision(df_all_per_tick, df_all)
    analyze_per_run_direction(df_all, 'decision_precision', 'Decision Precision by Direction (fraction correct)', 'decision_precision')
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
