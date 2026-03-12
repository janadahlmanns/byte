import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from scipy.stats import mannwhitneyu, kruskal
import scikit_posthocs as sp
import sys
import h5py

# Find workspace root by locating parent directory containing 'data' and 'simulate' folders
current_path = Path(__file__).resolve()
WORKSPACE_ROOT = None
while current_path.parent != current_path:  # While not at filesystem root
    if (current_path / "data").exists() and (current_path / "simulate").exists():
        WORKSPACE_ROOT = current_path
        break
    current_path = current_path.parent

if WORKSPACE_ROOT is None:
    raise RuntimeError("Could not find workspace root. Searched parent directories for 'data' and 'simulate' folders.")

sys.path.insert(0, str(WORKSPACE_ROOT))
from analysis_tools.network_visualization import network_viz



plt.ion()  # Enable interactive mode

# =====================================================================  A - CONFIGURATION  ================================================
# ==========================================================================================================================================


# Point to data directory by walking up until we find the "analysis" folder
# From there, go up one more level to find the experiment data files
current_path = Path(__file__).resolve().parent  # Start at script's directory
analysis_dir = None

while current_path.parent != current_path:  # While not at filesystem root
    if current_path.name == "analysis":
        analysis_dir = current_path
        break
    current_path = current_path.parent

if analysis_dir is None:
    raise RuntimeError("Analysis script must be placed in an 'analysis' folder")

BASE_DIR = analysis_dir.parent

# Manually specify the experiment HDF5 file name (without .h5 extension)
# or set to None to auto-detect most recent .h5 file
EXPERIMENT_NAME_OR_FILEPATH = "2026-03-12_19-19-06_random_lookup"  # HDF5 filename (with or without .h5)

# Locate the HDF5 file
if EXPERIMENT_NAME_OR_FILEPATH.endswith('.h5'):
    HDF5_FILE = Path(EXPERIMENT_NAME_OR_FILEPATH)
else:
    # Try with .h5 extension
    hdf5_candidates = list(BASE_DIR.glob(f"{EXPERIMENT_NAME_OR_FILEPATH}*.h5"))
    if hdf5_candidates:
        HDF5_FILE = hdf5_candidates[0]
    else:
        HDF5_FILE = BASE_DIR / f"{EXPERIMENT_NAME_OR_FILEPATH}.h5"

if not HDF5_FILE.exists():
    raise FileNotFoundError(f"HDF5 file not found: {HDF5_FILE}")

# Extract experiment name from script filename (not from HDF5 filename)
script_name = Path(__file__).stem  # e.g., "analysis_overall_survival" -> "overall_survival"
EXPERIMENT_NAME = script_name.replace("analysis_", "")

# Display name for the experiment (customize this for a nice report title and filenames)
# This is used in the report title and appended to all saved figures
EXPERIMENT_DISPLAY_NAME = "Convergence Check on hard-wired lookup variant"  


# Point to benchmark data (optional)
# Format: [(display_name, hdf5_file_or_name), (display_name2, hdf5_file_or_name2), ...]
# Example: [("Hard-wired Lookup", BASE_DIR / "2026-03-08_hardwired_lookup.h5"), ("Algorithmic", BASE_DIR / "2026-02-15_algo.h5")]
#BENCHMARK_HDF5_FILES = [
#    ("Hard-wired Lookup", BASE_DIR / "2026-03-12_13-04-06_random_lookup.h5"),
#]

BENCHMARK_HDF5_FILES = []

# =====================================================================
# Specification of visuals
# =====================================================================
GREEN_COLOR = "#0B3D2E"  # Dark green for successful variants
RED_COLOR = "#8B3A3A"    # Wine red for unsuccessful variants
# Network visualization configuration
NETWORK_VIZ_CONFIG = '11'  # Corresponds to network_viz_11.yaml in configs/ folder


# =====================================================================  HELPER FUNCTIONS FOR HDF5 LOADING  ====================================
# ==========================================================================================================================================

def load_variant_data_from_hdf5(hdf5_file: Path) -> pd.DataFrame:
    """
    Load variant summary data from HDF5 file.
    
    Args:
        hdf5_file: Path to HDF5 file
    
    Returns:
        DataFrame with all variant summary data (one row per run, variant column added)
    """
    all_variants = []
    
    with h5py.File(hdf5_file, 'r') as f:
        # Find all variant_XX groups
        variant_groups = [key for key in f.keys() if key.startswith('variant_')]
        
        for variant_group_name in sorted(variant_groups):
            variant_group = f[variant_group_name]
            
            # Load summary dataset
            if 'summary' not in variant_group:
                print(f"Warning: No summary dataset in {variant_group_name}")
                continue
            
            summary_dataset = variant_group['summary']
            # Convert structured array to DataFrame
            df_variant = pd.DataFrame(summary_dataset[()])
            df_variant['variant'] = variant_group_name
            all_variants.append(df_variant)
    
    if not all_variants:
        raise ValueError(f"No variant summary data found in {hdf5_file}")
    
    return pd.concat(all_variants, ignore_index=True)


def load_wiring_from_hdf5(hdf5_file: Path, variant_name: str = "variant_01") -> pd.DataFrame:
    """
    Load wiring data from HDF5 file for a specific variant.
    
    Args:
        hdf5_file: Path to HDF5 file
        variant_name: Variant group name (e.g., "variant_01")
    
    Returns:
        DataFrame with wiring data
    """
    with h5py.File(hdf5_file, 'r') as f:
        if variant_name not in f:
            # Try first variant if specified one doesn't exist
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
    """
    Load modulation data from HDF5 file for a specific variant.
    
    Args:
        hdf5_file: Path to HDF5 file
        variant_name: Variant group name (e.g., "variant_01")
    
    Returns:
        DataFrame with modulation data
    """
    with h5py.File(hdf5_file, 'r') as f:
        if variant_name not in f:
            # Try first variant if specified one doesn't exist
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


# =====================================================================  B - LOAD DATA AND INITIALIZE VARIABLES  ================================================
# ==========================================================================================================================================

# Load all variants from HDF5 file
print(f"Loading data from: {HDF5_FILE}")
df_random = load_variant_data_from_hdf5(HDF5_FILE)
print(f"Total rows loaded: {len(df_random)}")
print(f"Unique variants: {df_random['variant'].nunique()}")

# df_all will be the combined dataframe with benchmarks added later
df_all = df_random.copy()

# Load benchmark(s) if any
benchmark_data = {}
benchmark_names_map = {}  # Map HDF5 file to display name
df_benchmarks = None

if BENCHMARK_HDF5_FILES:
    benchmark_dfs = []
    for display_name, benchmark_hdf5_file in BENCHMARK_HDF5_FILES:
        if isinstance(benchmark_hdf5_file, str):
            benchmark_hdf5_file = Path(benchmark_hdf5_file)
        
        # Ensure .h5 extension
        if not benchmark_hdf5_file.suffix == '.h5':
            benchmark_hdf5_file = benchmark_hdf5_file.with_suffix('.h5')
        
        if not benchmark_hdf5_file.exists():
            print(f"Warning: Benchmark file not found: {benchmark_hdf5_file}")
            continue
        
        print(f"\nLoading benchmark '{display_name}' from: {benchmark_hdf5_file.resolve()}")
        
        try:
            df_benchmark = load_variant_data_from_hdf5(benchmark_hdf5_file)
            benchmark_group = benchmark_hdf5_file.stem
            df_benchmark["benchmark_group"] = benchmark_group
            df_benchmark["benchmark_display_name"] = display_name
            df_benchmark["benchmark_folder_path"] = str(benchmark_hdf5_file.parent.resolve())
            benchmark_dfs.append(df_benchmark)
            benchmark_names_map[benchmark_group] = display_name  # Map for later lookup
            print(f"  Loaded {df_benchmark['variant'].nunique()} variants with {len(df_benchmark)} total runs")
        except Exception as e:
            print(f"  Error loading benchmark: {e}")
            continue
    
    # Combine all benchmark data
    if benchmark_dfs:
        df_benchmarks = pd.concat(benchmark_dfs, ignore_index=True)
        df_all = pd.concat([df_random, df_benchmarks], ignore_index=True)
        print(f"\nTotal benchmark data loaded: {len(df_benchmarks)} rows")
    else:
        print("No valid benchmark files loaded")
        df_benchmarks = None
else:
    print("No benchmarks specified")
    df_benchmarks = None

# =====================================================================
# Calculate variant statistics and group thresholds
# =====================================================================

variant_stats = []

for variant_name in sorted(df_random["variant"].unique()):
    df_variant = df_random[df_random["variant"] == variant_name]
    lifetime_ticks = df_variant["lifetime_ticks"].values
    
    variant_stats.append({
        "variant": variant_name,
        "mean_survival": lifetime_ticks.mean(),
        "std_survival": lifetime_ticks.std(),
        "median_survival": np.median(lifetime_ticks),
        "min_survival": lifetime_ticks.min(),
        "max_survival": lifetime_ticks.max(),
        "n_runs": len(lifetime_ticks),
    })

df_stats = pd.DataFrame(variant_stats)

# Calculate threshold values for group classification
median_min = df_stats["median_survival"].min()
median_max = df_stats["median_survival"].max()
median_range = median_max - median_min

# Successful: top 10% of performance range (max - 10% of range)
median_successful_threshold = median_max - (0.10 * median_range)

# Unsuccessful: bottom 10% of performance range (min + 10% of range)
median_unsuccessful_threshold = median_min + (0.10 * median_range)

# Initialize output directory and figure tracking
results_dir = Path(__file__).resolve().parent
results_dir.mkdir(exist_ok=True)
fig_paths = []  # Track figures for report


# =====================================================================  C - HELPER FUNCTIONS ================================================
# ==========================================================================================================================================


# Perform appropriate statistical test
def perform_statistical_test(unsuccessful_group, successful_group, benchmark_groups=None, group_names=None):
    """
    Perform Kruskal-Wallis test with Dunn's post-hoc if multiple groups, Mann-Whitney U if two groups.
    
    Parameters:
    -----------
    unsuccessful_group : np.ndarray
        Data for unsuccessful/bottom 10% group
    successful_group : np.ndarray
        Data for successful/top 10% group
    benchmark_groups : dict or None
        Dict mapping benchmark name to np.ndarray of data
    group_names : dict or None
        Optional dict to customize group display names
    
    Returns:
    --------
    (stat, p_value, test_name, stat_name, posthoc_df or None)
    """
    if benchmark_groups is None or len(benchmark_groups) == 0:
        # Two-group Mann-Whitney U test
        stat, p_value = mannwhitneyu(unsuccessful_group, successful_group, alternative='two-sided')
        return stat, p_value, "Mann-Whitney U", "U-statistic", None
    else:
        # Multi-group Kruskal-Wallis test
        groups = [unsuccessful_group, successful_group]
        groups.extend(benchmark_groups.values())
        stat, p_value = kruskal(*groups)
        
        # Prepare data for Dunn's post-hoc test if available
        posthoc_df = None
        if p_value < 0.05:
            # Create DataFrame for post-hoc analysis
            data_list = []
            for val in unsuccessful_group:
                data_list.append({'metric': val, 'group': 'Unsuccessful'})
            for val in successful_group:
                data_list.append({'metric': val, 'group': 'Successful'})
            for bench_name, bench_data in benchmark_groups.items():
                for val in bench_data:
                    data_list.append({'metric': val, 'group': bench_name})
            
            df_posthoc = pd.DataFrame(data_list)
            posthoc_df = sp.posthoc_dunn(df_posthoc, val_col='metric', group_col='group', p_adjust='bonferroni')
        
        return stat, p_value, "Kruskal-Wallis", "H-statistic", posthoc_df


# Plot metric comparison (jitter + box plot)
def plot_metric_comparison(data_df, metric_col, group_col, title, y_label, output_path, 
                           unsuccessful_group_label="Unsuccessful\n(Bottom 10%)", 
                           successful_group_label="Successful\n(Top 10%)",
                           doc=None):
    """
    Create and save a jitter + box plot for metric comparison.
    
    Parameters:
    -----------
    data_df : pd.DataFrame
        DataFrame with data
    metric_col : str
        Column name for the metric values
    group_col : str
        Column name for group labels
    title : str
        Figure title
    y_label : str
        Y-axis label
    output_path : Path
        Where to save the figure
    unsuccessful_group_label : str
        Label for unsuccessful group (default includes newline for legend formatting)
    successful_group_label : str
        Label for successful group (default includes newline for legend formatting)
    doc : docx.Document, optional
        Word document to add the figure to. If provided, figure will be embedded in document.
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object
    """
    
    # Create color palette
    palette = {
        unsuccessful_group_label: RED_COLOR,
        successful_group_label: GREEN_COLOR
    }
    
    # Add benchmark colors if present
    unique_groups = data_df[group_col].unique()
    benchmark_colors = ["#2E8B9E", "#9E2E8B", "#8B9E2E", "#2E8B57"]
    color_idx = 0
    for group in sorted(unique_groups):
        if group not in palette:
            palette[group] = benchmark_colors[color_idx % len(benchmark_colors)]
            color_idx += 1
    
    # Create jitter + box plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Jitter plot
    sns.stripplot(
        data=data_df,
        x=group_col,
        y=metric_col,
        hue=group_col,
        palette=palette,
        size=4,
        alpha=0.7,
        jitter=True,
        ax=ax,
        dodge=False
    )
    
    # Box plot
    sns.boxplot(
        data=data_df,
        x=group_col,
        y=metric_col,
        hue=group_col,
        palette=palette,
        width=0.3,
        ax=ax,
        showcaps=True,
        whiskerprops={'linewidth': 2},
        boxprops={'linewidth': 2},
        medianprops={'color': 'black', 'linewidth': 2},
        fliersize=0,
        legend=False
    )
    
    # Make box patches more translucent
    for patch in ax.patches:
        patch.set_alpha(0.3)
    
    ax.set_xlabel("Variant Group", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Remove legend if it exists
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    
    plt.tight_layout()
    
    # Save figure
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    
    # Add to Word document if provided
    if doc is not None:
        if output_path.exists():
            doc.add_picture(str(output_path), width=Inches(6))
        doc.add_paragraph()
    
    return fig


# Calculate statistics for metric comparison
def calculate_statistics(unsuccessful_data, successful_data, benchmark_groups=None):
    """
    Calculate descriptive statistics and perform statistical tests.
    
    Parameters:
    -----------
    unsuccessful_data : np.ndarray
        Data values for unsuccessful group
    successful_data : np.ndarray
        Data values for successful group
    benchmark_groups : dict or None
        Dict mapping benchmark name to np.ndarray
    
    Returns:
    --------
    stats_dict : dict
        Dictionary with all calculated statistics
    """
    
    stats_dict = {
        "unsuccessful_n": len(unsuccessful_data),
        "unsuccessful_mean": unsuccessful_data.mean(),
        "unsuccessful_median": np.median(unsuccessful_data),
        "unsuccessful_std": unsuccessful_data.std(),
        "unsuccessful_min": unsuccessful_data.min(),
        "unsuccessful_max": unsuccessful_data.max(),
        "successful_n": len(successful_data),
        "successful_mean": successful_data.mean(),
        "successful_median": np.median(successful_data),
        "successful_std": successful_data.std(),
        "successful_min": successful_data.min(),
        "successful_max": successful_data.max(),
    }
    
    # Perform statistical test
    stat, p_value, test_name, stat_name, posthoc = perform_statistical_test(
        unsuccessful_data, successful_data, benchmark_groups
    )
    
    stats_dict["stat"] = stat
    stats_dict["p_value"] = p_value
    stats_dict["test_name"] = test_name
    stats_dict["stat_name"] = stat_name
    stats_dict["posthoc"] = posthoc
    
    return stats_dict


# Report statistics to Word document
def report_statistics(doc, stats_dict, metric_name):
    """
    Add statistics tables and results to Word document.
    
    Parameters:
    -----------
    doc : docx.Document
        The Word document object
    stats_dict : dict
        Dictionary with calculated statistics (from calculate_statistics)
    metric_name : str
        Name of the metric for header formatting
    """
    
    # Add statistics table
    table = doc.add_table(rows=7, cols=3)
    table.style = "Light Grid Accent 1"
    cells = table.rows[0].cells
    cells[0].text = "Metric"
    cells[1].text = "Successful (Top 10%)"
    cells[2].text = "Unsuccessful (Bottom 10%)"
    
    cells = table.rows[1].cells
    cells[0].text = "Number of runs"
    cells[1].text = f"{stats_dict['successful_n']}"
    cells[2].text = f"{stats_dict['unsuccessful_n']}"
    
    cells = table.rows[2].cells
    cells[0].text = "Mean"
    cells[1].text = f"{stats_dict['successful_mean']:.3f}"
    cells[2].text = f"{stats_dict['unsuccessful_mean']:.3f}"
    
    cells = table.rows[3].cells
    cells[0].text = "Median"
    cells[1].text = f"{stats_dict['successful_median']:.3f}"
    cells[2].text = f"{stats_dict['unsuccessful_median']:.3f}"
    
    cells = table.rows[4].cells
    cells[0].text = "Std Dev"
    cells[1].text = f"{stats_dict['successful_std']:.3f}"
    cells[2].text = f"{stats_dict['unsuccessful_std']:.3f}"
    
    cells = table.rows[5].cells
    cells[0].text = "Min"
    cells[1].text = f"{stats_dict['successful_min']:.3f}"
    cells[2].text = f"{stats_dict['unsuccessful_min']:.3f}"
    
    cells = table.rows[6].cells
    cells[0].text = "Max"
    cells[1].text = f"{stats_dict['successful_max']:.3f}"
    cells[2].text = f"{stats_dict['unsuccessful_max']:.3f}"
    
    # Add statistical test header
    test_name = stats_dict.get('test_name', 'Mann-Whitney U')
    if test_name == "Mann-Whitney U":
        header_text = "Differences between successful and unsuccessful random variants (Mann-Whitney U):"
    else:
        header_text = "Differences between successful, unsuccessful, and benchmark variants (Kruskal-Wallis):"
    
    doc.add_paragraph(header_text, style="Heading 3")
    
    # Add test results table
    table = doc.add_table(rows=3, cols=2)
    table.style = "Light Grid Accent 1"
    cells = table.rows[0].cells
    cells[0].text = "Test Statistic"
    cells[1].text = "Value"
    cells = table.rows[1].cells
    cells[0].text = stats_dict.get('stat_name', 'U-statistic')
    cells[1].text = f"{stats_dict['stat']:.4f}"
    cells = table.rows[2].cells
    cells[0].text = "p-value"
    cells[1].text = f"{stats_dict['p_value']:.4e}"
    
    doc.add_paragraph()
    
    # Add post-hoc results if available
    if stats_dict.get('posthoc') is not None:
        doc.add_paragraph("Dunn's Post-hoc Test Results (pairwise comparisons, Bonferroni-adjusted):", style="Heading 3")
        posthoc_df = stats_dict['posthoc']
        posthoc_table = doc.add_table(rows=len(posthoc_df) + 1, cols=len(posthoc_df.columns) + 1)
        posthoc_table.style = "Light Grid Accent 1"
        
        # Header row
        header_cells = posthoc_table.rows[0].cells
        header_cells[0].text = "Comparison"
        for col_idx, col_name in enumerate(posthoc_df.columns):
            header_cells[col_idx + 1].text = str(col_name)
        
        # Data rows
        for row_idx, (idx_name, row_data) in enumerate(posthoc_df.iterrows(), 1):
            data_cells = posthoc_table.rows[row_idx].cells
            data_cells[0].text = str(idx_name)
            for col_idx, val in enumerate(row_data):
                data_cells[col_idx + 1].text = f"{float(val):.4f}"


# Convergence analysis plot
def convergence_plot(data, metric_name, output_path, doc=None, n_bootstrap=1000):
    """
    Plot convergence of a metric over runs showing both central tendency and uncertainty.
    
    Parameters:
    -----------
    data : np.ndarray
        Array of metric values (one per run)
    metric_name : str
        Name of the metric for labels
    output_path : Path
        Where to save the figure
    doc : docx.Document, optional
        Word document to add the figure to
    n_bootstrap : int
        Number of bootstrap samples for SE calculation (default: 1000)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object
    
    Notes:
    ------
    The plot has two y-axes:
    
    Primary y-axis (left): Central tendency and spread
        - Rolling mean: average value up to current run
        - Rolling median: median value up to current run
        - Rolling IQR: interquartile range (distribution-free measure of data spread)
    
    Secondary y-axis (right): Uncertainty measures
        - Bootstrap SE of median: standard error from bootstrap resampling (distribution-free)
        - SE of mean: standard error of the mean (scales as SD/sqrt(N))
    
    When both central tendency and SE have flattened, you have sufficient runs.
    The SE curve is typically the more sensitive convergence indicator.
    """
    data = np.asarray(data).flatten()
    n_runs = len(data)
    
    # Calculate rolling statistics (cumulatively up to each run)
    rolling_mean = np.array([data[:i+1].mean() for i in range(n_runs)])
    rolling_median = np.array([np.median(data[:i+1]) for i in range(n_runs)])
    rolling_iqr = np.array([
        np.percentile(data[:i+1], 75) - np.percentile(data[:i+1], 25) 
        for i in range(n_runs)
    ])
    
    # Calculate rolling bootstrap SE of median
    bootstrap_se_median = []
    np.random.seed(42)  # For reproducibility
    for i in range(n_runs):
        subset = data[:i+1]
        if len(subset) < 2:
            bootstrap_se_median.append(np.nan)
        else:
            # Bootstrap: resample with replacement, compute median each time
            bootstrap_medians = []
            for _ in range(n_bootstrap):
                resampled = np.random.choice(subset, size=len(subset), replace=True)
                bootstrap_medians.append(np.median(resampled))
            bootstrap_se_median.append(np.std(bootstrap_medians))
    bootstrap_se_median = np.array(bootstrap_se_median)
    
    # Calculate rolling SE of mean (standard error of the mean)
    se_mean = np.array([
        data[:i+1].std() / np.sqrt(i+1) if i > 0 else np.inf
        for i in range(n_runs)
    ])
    
    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(13, 7))
    
    # Primary y-axis: central tendency and spread
    ax1.set_xlabel('Number of Runs', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Central Tendency & Data Spread', fontsize=12, fontweight='bold', color=GREEN_COLOR)
    ax1.tick_params(axis='y', labelcolor=GREEN_COLOR)
    
    lns1 = ax1.plot(range(1, n_runs+1), rolling_mean, label='Rolling Mean', 
                    linewidth=1.25, color=GREEN_COLOR, marker='o', markersize=6, alpha=0.7)
    lns2 = ax1.plot(range(1, n_runs+1), rolling_median, label='Rolling Median', 
                    linewidth=1.25, color=GREEN_COLOR, marker='s', markersize=6, alpha=0.7)
    lns3 = ax1.plot(range(1, n_runs+1), rolling_iqr, label='Rolling IQR (data spread)', 
                    linewidth=1.25, color=GREEN_COLOR, marker='^', markersize=6, alpha=0.7)
    ax1.grid(False)
    
    # Secondary y-axis: uncertainty measures
    ax2 = ax1.twinx()
    ax2.set_ylabel('Uncertainty (Standard Error)', fontsize=12, fontweight='bold', color=RED_COLOR)
    ax2.tick_params(axis='y', labelcolor=RED_COLOR)
    
    lns4 = ax2.plot(range(1, n_runs+1), bootstrap_se_median, label='Bootstrap SE of Median', 
                    linewidth=1.25, color=RED_COLOR, marker='o', markersize=6, 
                    alpha=0.7, linestyle='-')
    lns5 = ax2.plot(range(1, n_runs+1), se_mean, label='SE of Mean', 
                    linewidth=1.25, color=RED_COLOR, marker='D', markersize=6, 
                    alpha=0.7, linestyle='-')
    ax2.grid(False)
    
    # Add dashed vertical line at x=300 as reference
    ax1.axvline(x=300, color='black', linestyle='--', linewidth=1, alpha=0.5, zorder=0)
    
    # Title
    title = f'Convergence Analysis: {metric_name}'
    ax1.set_title(title, fontsize=13, fontweight='bold', pad=15)
    
    # Combined legend
    lns = lns1 + lns2 + lns3 + lns4 + lns5
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='center right', fontsize=10, frameon=True, fancybox=True)
    
    fig.tight_layout()
    
    # Save figure
    fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    # Add to Word document if provided
    if doc is not None:
        if output_path.exists():
            doc.add_picture(str(output_path), width=Inches(6))
        doc.add_paragraph()
    
    return fig


# =====================================================================  D - ANALYSIS ================================================
# ==========================================================================================================================================


from scipy.stats import skew, kurtosis

doc = Document()

#region 1. EXPERIMENT INFORMATION ========================================================================================================================================================================

doc.add_heading("Analysis Report: Random Wiring Variants", level=0)
doc.add_heading(f"{EXPERIMENT_DISPLAY_NAME}", level=1)

doc.add_heading("1. Experiment Information", level=2)

doc.add_paragraph("Random Wiring Variants:", style="Heading 3")
doc.add_paragraph(f"Experiment file: {HDF5_FILE.name}")
n_variants = df_random['variant'].nunique()
runs_per_variant = len(df_random) // n_variants if n_variants > 0 else 0
doc.add_paragraph(f"Total random variants analyzed: {n_variants}")
doc.add_paragraph(f"Runs per variant: {runs_per_variant}")
doc.add_paragraph(f"Total data points (random): {len(df_random)}")

if df_benchmarks is not None and len(df_benchmarks) > 0:
    doc.add_paragraph("Benchmark Variants:", style="Heading 3")
    for benchmark_name in sorted(df_benchmarks['benchmark_group'].unique()):
        df_bench = df_benchmarks[df_benchmarks['benchmark_group'] == benchmark_name]
        num_bench_variants = len(df_bench['variant'].unique())
        display_name = benchmark_names_map.get(benchmark_name, benchmark_name.replace("_", " ").title())
        doc.add_paragraph(f"{display_name}: {len(df_bench)} data points ({num_bench_variants} variants)")

#region 1.1. SUMMARY STATISTICS - RANDOM VARIANTS ==========================================================================================================================================================

doc.add_heading("1.1. Summary Statistics - Random Wiring Variants", level=2)

doc.add_paragraph(
    f"Summary of how long random wiring variants survive in the experimental environment. "
    f"We measure survival time in simulation ticks across all {n_variants} variants and {runs_per_variant} runs per variant. "
    f"The statistics below provide a comprehensive overview of survival time distribution across all measurements."
)

doc.add_paragraph("Overall Descriptive Statistics (Across All Variants):", style="Heading 3")

# =====================================================================
# Calculate comprehensive statistics on ALL raw lifetime data
# =====================================================================
all_lifetimes = df_random["lifetime_ticks"].values

overall_mean = all_lifetimes.mean()
overall_median = np.median(all_lifetimes)
overall_std = all_lifetimes.std()
overall_min = all_lifetimes.min()
overall_max = all_lifetimes.max()
overall_range = overall_max - overall_min
overall_iqr = np.percentile(all_lifetimes, 75) - np.percentile(all_lifetimes, 25)
overall_p5 = np.percentile(all_lifetimes, 5)
overall_p25 = np.percentile(all_lifetimes, 25)
overall_p75 = np.percentile(all_lifetimes, 75)
overall_p95 = np.percentile(all_lifetimes, 95)
overall_skewness = skew(all_lifetimes)
overall_kurtosis = kurtosis(all_lifetimes)
overall_cv = (overall_std / overall_mean) * 100

table = doc.add_table(rows=15, cols=2)
table.style = "Light Grid Accent 1"
cells = table.rows[0].cells
cells[0].text = "Statistic"
cells[1].text = "Value"

metrics_stats_table = [
    ("Mean", f"{overall_mean:.2f} ticks"),
    ("Median", f"{overall_median:.2f} ticks"),
    ("Std Dev", f"{overall_std:.2f} ticks"),
    ("Min", f"{overall_min:.2f} ticks"),
    ("Max", f"{overall_max:.2f} ticks"),
    ("Range", f"{overall_range:.2f} ticks"),
    ("IQR (25th-75th percentile)", f"{overall_iqr:.2f} ticks"),
    ("5th Percentile", f"{overall_p5:.2f} ticks"),
    ("25th Percentile", f"{overall_p25:.2f} ticks"),
    ("75th Percentile", f"{overall_p75:.2f} ticks"),
    ("95th Percentile", f"{overall_p95:.2f} ticks"),
    ("Skewness", f"{overall_skewness:.3f}"),
    ("Kurtosis (excess)", f"{overall_kurtosis:.3f}"),
    ("Coefficient of Variation", f"{overall_cv:.2f} %"),
]

for row_idx, (metric_name, value_str) in enumerate(metrics_stats_table, 1):
    cells = table.rows[row_idx].cells
    cells[0].text = metric_name
    cells[1].text = value_str


#endregion
#endregion  # closes 1
#region 2. GROUP SELECTION ================================================================================================================================================================================

doc.add_heading("2. Survival Race", level=2)


median_min = df_stats["median_survival"].min()
median_max = df_stats["median_survival"].max()
median_range = median_max - median_min

# =====================================================================
# Calculate threshold values for group classification
# =====================================================================
df_stats = pd.DataFrame(variant_stats)
median_min = df_stats["median_survival"].min()
median_max = df_stats["median_survival"].max()
median_range = median_max - median_min

# Successful: top 10% of performance range (max - 10% of range)
median_successful_threshold = median_max - (0.10 * median_range)

# Unsuccessful: bottom 10% of performance range (min + 10% of range)
median_unsuccessful_threshold = median_min + (0.10 * median_range)

# Initialize output directory and figure tracking
results_dir = Path(__file__).resolve().parent
results_dir.mkdir(exist_ok=True)
fig_paths = []  # Track figures for report

# Successful: top 10% of performance range (max - 10% of range)
median_successful_threshold = median_max - (0.10 * median_range)

# Unsuccessful: bottom 10% of performance range (min + 10% of range)
median_unsuccessful_threshold = median_min + (0.10 * median_range)

# Identify variants in each group
top_10_pct_variants = set(df_stats[df_stats["median_survival"] >= median_successful_threshold]["variant"].values)
bottom_10_pct_variants = set(df_stats[df_stats["median_survival"] <= median_unsuccessful_threshold]["variant"].values)




# =====================================================================
# IDENTIFY SUCCESSFUL AND UNSUCCESSFUL VARIANT GROUPS
# =====================================================================

successful_variants = set(df_stats[df_stats["median_survival"] >= median_successful_threshold]["variant"].values)
unsuccessful_variants = set(df_stats[df_stats["median_survival"] <= median_unsuccessful_threshold]["variant"].values)

# Assign group labels to the random variants dataframe
def _assign_group_label(var_name):
    if var_name in successful_variants:
        return "Successful\n(Top 10%)"
    if var_name in unsuccessful_variants:
        return "Unsuccessful\n(Bottom 10%)"
    return "Random Middle"

df_random["group"] = df_random["variant"].apply(_assign_group_label)

# For df_all, preserve benchmark group labels (which were already set) and only assign labels to random variants
if "benchmark_group" in df_all.columns:
    # Only update rows where benchmark_group is null (i.e., random variants)
    mask = df_all["benchmark_group"].isna()
    df_all.loc[mask, "group"] = df_all[mask]["variant"].apply(_assign_group_label)
else:
    # No benchmarks - df_all is just df_random with group labels already assigned
    df_all["group"] = df_all["variant"].apply(_assign_group_label)

# Make df_all immutable so that any attempts to modify df_all after this point will raise an error, catching unintended mutations of our data
df_all.flags.writeable = False

# =====================================================================
# Create survival race plot
# =====================================================================

doc.add_paragraph(
    f"This plot shows the survival curves for all {n_variants} wiring variants. "
)


sns.set_theme(style="whitegrid", context="talk")

fig = plt.figure(figsize=(14, 8))

# For each variant, compute survival race curve (using random variants only)
for variant_name in sorted(df_random["variant"].unique()):
    df_variant = df_random[df_random["variant"] == variant_name]
    survival_times = df_variant["lifetime_ticks"].values
    
    # Compute survival race: at each tick, how many are still alive
    max_t = survival_times.max()
    ticks = np.arange(max_t + 1)
     
    # Count survivors at each tick
    alive_count = np.array([(survival_times >= t).sum() for t in ticks])
    
    # Determine color and alpha based on performance
    if variant_name in successful_variants:
        color = GREEN_COLOR
        linewidth = 1.5
        alpha = 0.8
    elif variant_name in unsuccessful_variants:
        color = RED_COLOR
        linewidth = 1.5
        alpha = 0.8
    else:
        color = "gray"
        linewidth = 0.8
        alpha = 0.3
    
    plt.plot(
        ticks,
        alive_count,
        linewidth=linewidth,
        linestyle="-",
        alpha=alpha,
        color=color
    )

# Add benchmark variants if available
if df_benchmarks is not None:
    benchmark_colors_list = ["#2E8B9E", "#9E2E8B", "#8B9E2E", "#2E8B57"]
    for bench_idx, benchmark_name in enumerate(sorted(df_benchmarks['benchmark_group'].unique())):
        df_bench_group = df_benchmarks[df_benchmarks["benchmark_group"] == benchmark_name]
        bench_color = benchmark_colors_list[bench_idx % len(benchmark_colors_list)]
        
        for variant_name in sorted(df_bench_group["variant"].unique()):
            df_variant = df_bench_group[df_bench_group["variant"] == variant_name]
            survival_times = df_variant["lifetime_ticks"].values
            
            max_t = survival_times.max()
            ticks = np.arange(max_t + 1)
            alive_count = np.array([(survival_times >= t).sum() for t in ticks])
            
            plt.plot(
                ticks,
                alive_count,
                linewidth=2.0,
                linestyle="-",
                alpha=0.9,
                color=bench_color
            )

plt.xlabel("Tick", fontsize=12)
plt.ylabel("Number of Bytes alive", fontsize=12)
title_suffix = ""
if df_benchmarks is not None and len(df_benchmarks) > 0:
    title_suffix = f" + {len(list(df_benchmarks['benchmark_group'].unique()))} Benchmark(s)"
plt.title(f"Survival Race: All {n_variants} Random Variants{title_suffix}\n({runs_per_variant} world seeds per variant)")
plt.grid(True, alpha=0.3)

# Create custom legend
legend_elements = [
    Line2D([0], [0], color=GREEN_COLOR, linewidth=1.5, label=f"Random Top 10% (median ≥ {median_successful_threshold:.1f} ticks)"),
    Line2D([0], [0], color=RED_COLOR, linewidth=1.5, label=f"Random Bottom 10% (median ≤ {median_unsuccessful_threshold:.1f} ticks)"),
    Line2D([0], [0], color="gray", linewidth=0.8, label="Random Middle 80%"),
]

# Add benchmark legend entries
if df_benchmarks is not None:
    benchmark_colors_list = ["#2E8B9E", "#9E2E8B", "#8B9E2E", "#2E8B57"]
    for bench_idx, benchmark_name in enumerate(sorted(df_benchmarks['benchmark_group'].unique())):
        display_name = benchmark_names_map.get(benchmark_name, benchmark_name.replace("_", " ").title())
        bench_color = benchmark_colors_list[bench_idx % len(benchmark_colors_list)]
        legend_elements.append(Line2D([0], [0], color=bench_color, linewidth=2.0, label=f"Benchmark: {display_name}"))

plt.legend(handles=legend_elements, loc="upper right", fontsize=10, frameon=True, fancybox=True)

plt.tight_layout()

# Save figure
fig_path = results_dir / f"survival_race_{EXPERIMENT_NAME}_{EXPERIMENT_DISPLAY_NAME.replace(' ', '_')}.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {fig_path}")
fig_paths.append(("Survival Race: All Variants", fig_path))

plt.show()

if fig_path.exists():
    doc.add_picture(str(fig_path), width=Inches(6))
doc.add_paragraph()


# =====================================================================
# Network visualizations for group representatives
# =====================================================================

doc.add_paragraph()
doc.add_paragraph("Network Development", style="Heading 4")
doc.add_paragraph(
    "Below are combined network visualizations showing initial wiring and the first three runtime states."
)

# Load network configuration once
neuron_positions, neuron_types = network_viz.load_network_viz_config(NETWORK_VIZ_CONFIG)

# Helper function to extract numeric part of variant name for sorting
def _get_variant_number(variant_name):
    """Extract numeric portion from variant name (e.g., 'variant_001' -> 1)"""
    if "_" in variant_name:
        s = variant_name.split("_", 1)[1]
    else:
        s = variant_name
    try:
        return int(s)
    except:
        return float('inf')

# Define weight columns and labels for visualization
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

# Process successful variants (first up to 3)
if len(successful_variants) > 0:
    
    successful_sorted = sorted(successful_variants, key=_get_variant_number)[:3]
    
    # Create network plots for each successful variant
    for variant_name in successful_sorted:
        try:
            # Extract variant number for filename
            variant_num = _get_variant_number(variant_name)
            
            # Load wiring and modulation from HDF5
            df_wiring = load_wiring_from_hdf5(HDF5_FILE, variant_name)
            df_modulation = load_modulation_from_hdf5(HDF5_FILE, variant_name)
            
            # Create output path
            safe_exp_name = EXPERIMENT_DISPLAY_NAME.replace(" ", "_")
            network_fig_path = results_dir / f"network_{safe_exp_name}_variant_{variant_num:03d}.png"
            
            # Temporarily save to CSV for network viz function
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                wiring_csv = Path(tmpdir) / "wiring.csv"
                modulation_csv = Path(tmpdir) / "modulation.csv"
                
                df_wiring.to_csv(wiring_csv, index=False)
                if len(df_modulation) > 0:
                    df_modulation.to_csv(modulation_csv, index=False)
                
                # Generate network visualization
                network_viz.draw_and_combine_networks(
                    wiring_csv=str(wiring_csv),
                    weight_columns=weight_columns,
                    panel_labels=panel_labels,
                    output_path=str(network_fig_path),
                    modulation_csv=str(modulation_csv) if len(df_modulation) > 0 else None,
                    neuron_positions=neuron_positions,
                    neuron_types=neuron_types,
                    title=f'Network Development'
                )
                
                # Add to document
                doc.add_picture(str(network_fig_path), width=Inches(6.5))
                print(f"Saved network visualization: {network_fig_path}")
                
        except Exception as e:
            print(f"Error creating network visualization for {variant_name}: {e}")

#endregion # closes 2.
#region 3. CONVERGENCE =====================================================================================================================================================================================

doc.add_heading("3. Convergence", level=2)

doc.add_paragraph(
    "We analyze the convergence of simulation metrics and of variance of simulation metrics over the number of runs. "
    "Each run subjects the same variant to a different world seed, which can be thought of as a different environment. "
    "Simulations should run a large enough number of runs to allow metrics to converge to stable values that are reliable readouts of the variant's performance, "
    "rather than dominated by chance from rng world seeds."
)

#region 3.1. SURVIVAL TIMES ================================================================================================================================================================================

doc.add_heading("3.1. Survival Times", level=3)

doc.add_paragraph(
    "Convergence analysis of survival times (lifetime_ticks) across all runs. "
    "The left y-axis shows central tendency (rolling mean and median) and data spread (rolling IQR), "
    "while the right y-axis shows uncertainty estimates (bootstrap SE of median and SE of mean). "
    "Convergence is achieved when both the central tendency and uncertainty curves have stabilized."
)

# Extract all survival times for convergence analysis
all_survival_times = df_random["lifetime_ticks"].values

# Generate convergence plot for survival times
convergence_fig_path = results_dir / f"convergence_survival_times.png"
convergence_plot(all_survival_times, "Survival Times", convergence_fig_path, doc=doc)
fig_paths.append(("Convergence: Survival Times", convergence_fig_path))

#endregion

#region 3.2. FOOD CONSUMED ================================================================================================================================================================================

doc.add_heading("3.2. Food Consumed", level=3)

doc.add_paragraph(
    "Convergence analysis of food consumed across all runs. "
    "The left y-axis shows central tendency (rolling mean and median) and data spread (rolling IQR), "
    "while the right y-axis shows uncertainty estimates (bootstrap SE of median and SE of mean). "
    "Convergence is achieved when both the central tendency and uncertainty curves have stabilized."
)

# Extract all food consumed values for convergence analysis
all_food_consumed = df_random["foods"].values

# Generate convergence plot for food consumed
convergence_fig_path = results_dir / f"convergence_food_consumed.png"
convergence_plot(all_food_consumed, "Food Consumed", convergence_fig_path, doc=doc)
fig_paths.append(("Convergence: Food Consumed", convergence_fig_path))

#endregion

#region 3.3. DISTANCE TRAVELLED ================================================================================================================================================================================

doc.add_heading("3.3. Distance Travelled", level=3)

doc.add_paragraph(
    "Convergence analysis of distance travelled across all runs. "
    "The left y-axis shows central tendency (rolling mean and median) and data spread (rolling IQR), "
    "while the right y-axis shows uncertainty estimates (bootstrap SE of median and SE of mean). "
    "Convergence is achieved when both the central tendency and uncertainty curves have stabilized."
)

# Extract all distance travelled values for convergence analysis
all_distance_travelled = df_random["distance"].values

# Generate convergence plot for distance travelled
convergence_fig_path = results_dir / f"convergence_distance_travelled.png"
convergence_plot(all_distance_travelled, "Distance Travelled", convergence_fig_path, doc=doc)
fig_paths.append(("Convergence: Distance Travelled", convergence_fig_path))

#endregion

#region 3.4. DECISIONS MADE ================================================================================================================================================================================

doc.add_heading("3.4. Decisions Made", level=3)

doc.add_paragraph(
    "Convergence analysis of decisions made across all runs. "
    "The left y-axis shows central tendency (rolling mean and median) and data spread (rolling IQR), "
    "while the right y-axis shows uncertainty estimates (bootstrap SE of median and SE of mean). "
    "Convergence is achieved when both the central tendency and uncertainty curves have stabilized."
)

# Extract all decisions made values for convergence analysis
all_decisions_made = df_random["decisions"].values

# Generate convergence plot for decisions made
convergence_fig_path = results_dir / f"convergence_decisions_made.png"
convergence_plot(all_decisions_made, "Decisions Made", convergence_fig_path, doc=doc)
fig_paths.append(("Convergence: Decisions Made", convergence_fig_path))

#endregion

#region 3.5. FOOD CONSUMPTION PER TICK ==============================================================================================================================================================================

doc.add_heading("3.5. Food Consumption Per Tick", level=3)

doc.add_paragraph(
    "Convergence analysis of food consumption per tick (total food consumed divided by survival time). "
    "The left y-axis shows central tendency (rolling mean and median) and data spread (rolling IQR), "
    "while the right y-axis shows uncertainty estimates (bootstrap SE of median and SE of mean). "
    "Convergence is achieved when both the central tendency and uncertainty curves have stabilized."
)

# Calculate food consumption per tick (handling division by zero)
food_per_tick = np.divide(
    df_random["foods"].values, 
    df_random["lifetime_ticks"].values,
    out=np.zeros_like(df_random["foods"].values, dtype=float),
    where=df_random["lifetime_ticks"].values != 0
)

# Generate convergence plot for food per tick
convergence_fig_path = results_dir / f"convergence_food_per_tick.png"
convergence_plot(food_per_tick, "Food Consumption Per Tick", convergence_fig_path, doc=doc)
fig_paths.append(("Convergence: Food Per Tick", convergence_fig_path))

#endregion

#region 3.6. DISTANCE PER TICK ================================================================================================================================================================================

doc.add_heading("3.6. Distance Per Tick", level=3)

doc.add_paragraph(
    "Convergence analysis of distance travelled per tick (total distance divided by survival time). "
    "The left y-axis shows central tendency (rolling mean and median) and data spread (rolling IQR), "
    "while the right y-axis shows uncertainty estimates (bootstrap SE of median and SE of mean). "
    "Convergence is achieved when both the central tendency and uncertainty curves have stabilized."
)

# Calculate distance per tick (handling division by zero)
distance_per_tick = np.divide(
    df_random["distance"].values,
    df_random["lifetime_ticks"].values,
    out=np.zeros_like(df_random["distance"].values, dtype=float),
    where=df_random["lifetime_ticks"].values != 0
)

# Generate convergence plot for distance per tick
convergence_fig_path = results_dir / f"convergence_distance_per_tick.png"
convergence_plot(distance_per_tick, "Distance Per Tick", convergence_fig_path, doc=doc)
fig_paths.append(("Convergence: Distance Per Tick", convergence_fig_path))

#endregion

#region 3.7. DECISIONS PER TICK ===============================================================================================================================================================================

doc.add_heading("3.7. Decisions Per Tick", level=3)

doc.add_paragraph(
    "Convergence analysis of decisions made per tick (total decisions divided by survival time). "
    "The left y-axis shows central tendency (rolling mean and median) and data spread (rolling IQR), "
    "while the right y-axis shows uncertainty estimates (bootstrap SE of median and SE of mean). "
    "Convergence is achieved when both the central tendency and uncertainty curves have stabilized."
)

# Calculate decisions per tick (handling division by zero)
decisions_per_tick = np.divide(
    df_random["decisions"].values,
    df_random["lifetime_ticks"].values,
    out=np.zeros_like(df_random["decisions"].values, dtype=float),
    where=df_random["lifetime_ticks"].values != 0
)

# Generate convergence plot for decisions per tick
convergence_fig_path = results_dir / f"convergence_decisions_per_tick.png"
convergence_plot(decisions_per_tick, "Decisions Per Tick", convergence_fig_path, doc=doc)
fig_paths.append(("Convergence: Decisions Per Tick", convergence_fig_path))

#endregion

#region 3.8. CONNECTION WEIGHTS ===============================================================================================================================================================================

doc.add_heading("3.8. Connection Weights", level=3)

doc.add_paragraph(
    "Convergence analysis of synaptic connection weights for representative connections. "
    "Two representative connections are shown: Connection 1→6 and Connection 3→10. "
    "Each plot shows the left y-axis with central tendency (rolling mean and median) and data spread (rolling IQR), "
    "while the right y-axis shows uncertainty estimates (bootstrap SE of median and SE of mean). "
    "Convergence is achieved when both the central tendency and uncertainty curves have stabilized."
)

# Load wiring data for the first variant
df_wiring = load_wiring_from_hdf5(HDF5_FILE, "variant_01")

# Helper function to extract connection weights across runs
def extract_connection_weights(df_wiring, src, tgt):
    """Extract weight values for a specific connection across all runs."""
    connection = df_wiring[(df_wiring['src'] == src) & (df_wiring['tgt'] == tgt)]
    if len(connection) == 0:
        return None
    
    connection_row = connection.iloc[0]
    # Extract all weight_final_run_XXXX columns (in order)
    weight_cols = [col for col in df_wiring.columns if col.startswith('weight_final_run_')]
    weight_cols.sort()  # Ensure correct order
    weights = np.array([connection_row[col] for col in weight_cols])
    return weights

# Connection 1 → 6
weights_1_6 = extract_connection_weights(df_wiring, 1, 6)
if weights_1_6 is not None:
    convergence_fig_path = results_dir / f"convergence_connection_1_to_6.png"
    convergence_plot(weights_1_6, "Connection 1→6 Weight", convergence_fig_path, doc=doc)
    fig_paths.append(("Convergence: Connection 1→6", convergence_fig_path))
else:
    doc.add_paragraph("Connection 1→6 not found in wiring data.")

# Connection 3 → 10
weights_3_10 = extract_connection_weights(df_wiring, 3, 10)
if weights_3_10 is not None:
    convergence_fig_path = results_dir / f"convergence_connection_3_to_10.png"
    convergence_plot(weights_3_10, "Connection 3→10 Weight", convergence_fig_path, doc=doc)
    fig_paths.append(("Convergence: Connection 3→10", convergence_fig_path))
else:
    doc.add_paragraph("Connection 3→10 not found in wiring data.")

#endregion


#endregion # close 3
# =====================================================================  E - FINALIZE DOCUMENT AND SAVE ================================================
# ==========================================================================================================================================


print("\nFinalizing document and saving...")

report_path = results_dir / f"report_{EXPERIMENT_NAME}_{EXPERIMENT_DISPLAY_NAME.replace(' ', '_')}.docx"
doc.save(report_path)
print(f"Report saved to: {report_path}")

print("\nAnalysis complete!")
input("Press Enter to close all figures...")
plt.ioff()