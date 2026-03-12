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


# Point to raw data directory
BASE_DIR = Path(__file__).resolve().parents[2] / "rawdata"

# Manually specify the experiment HDF5 file name (without .h5 extension)
# or set to None to auto-detect most recent .h5 file
EXPERIMENT_NAME_OR_FILEPATH = "2026-03-12_13-04-06_random_lookup"  # HDF5 filename (with or without .h5)

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
EXPERIMENT_DISPLAY_NAME = "Random Wiring"  


# Point to benchmark data (optional)
# Format: [(display_name, hdf5_file_or_name), (display_name2, hdf5_file_or_name2), ...]
# Example: [("Hard-wired Lookup", BASE_DIR / "2026-03-08_hardwired_lookup.h5"), ("Algorithmic", BASE_DIR / "2026-02-15_algo.h5")]
BENCHMARK_HDF5_FILES = [
    ("Hard-wired Lookup", BASE_DIR / "2026-03-12_13-04-06_random_lookup.h5"),
]

#BENCHMARK_HDF5_FILES = []

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

# =====================================================================
# Per-variant summary: Highest, Lowest, Average across variants
# =====================================================================

doc.add_paragraph("Per-variant Summary (Highest, Lowest, Average across variants):", style="Heading 3")

# Build per-variant metrics
variant_metrics = []
for variant_name in sorted(df_random["variant"].unique()):
    vdf = df_random[df_random["variant"] == variant_name]["lifetime_ticks"].values
    if len(vdf) == 0:
        continue
    v_mean = vdf.mean()
    v_median = np.median(vdf)
    v_std = vdf.std()
    v_min = vdf.min()
    v_max = vdf.max()
    v_range = v_max - v_min
    v_iqr = np.percentile(vdf, 75) - np.percentile(vdf, 25)
    v_p5 = np.percentile(vdf, 5)
    v_p25 = np.percentile(vdf, 25)
    v_p75 = np.percentile(vdf, 75)
    v_p95 = np.percentile(vdf, 95)
    v_skew = skew(vdf)
    v_kurt = kurtosis(vdf)
    v_cv = (v_std / v_mean) * 100 if v_mean != 0 else float('nan')

    variant_metrics.append({
        "variant": variant_name,
        "mean": v_mean,
        "median": v_median,
        "std": v_std,
        "min": v_min,
        "max": v_max,
        "range": v_range,
        "iqr": v_iqr,
        "p5": v_p5,
        "p25": v_p25,
        "p75": v_p75,
        "p95": v_p95,
        "skewness": v_skew,
        "kurtosis": v_kurt,
        "cv": v_cv,
    })

# Metrics to summarize across variants (name, key, format)
variant_metrics_list = [
    ("Mean", "mean", "{:.2f} ticks"),
    ("Median", "median", "{:.2f} ticks"),
    ("Std Dev", "std", "{:.2f} ticks"),
    ("Min", "min", "{:.2f} ticks"),
    ("Max", "max", "{:.2f} ticks"),
    ("Range", "range", "{:.2f} ticks"),
    ("IQR (25th-75th)", "iqr", "{:.2f} ticks"),
    ("5th Percentile", "p5", "{:.2f} ticks"),
    ("25th Percentile", "p25", "{:.2f} ticks"),
    ("75th Percentile", "p75", "{:.2f} ticks"),
    ("95th Percentile", "p95", "{:.2f} ticks"),
    ("Skewness", "skewness", "{:.3f}"),
    ("Kurtosis (excess)", "kurtosis", "{:.3f}"),
    ("Coefficient of Variation", "cv", "{:.2f} %"),
]

if variant_metrics:
    tbl = doc.add_table(rows=len(variant_metrics_list) + 1, cols=4)
    tbl.style = "Light Grid Accent 1"
    hdr = tbl.rows[0].cells
    hdr[0].text = "Statistic"
    hdr[1].text = "Highest (variant)"
    hdr[2].text = "Lowest (variant)"
    hdr[3].text = "Average (across variants)"

    for r_idx, (label, key, fmt) in enumerate(variant_metrics_list, 1):
        vals = np.array([vm[key] for vm in variant_metrics], dtype=float)
        # handle NaNs gracefully
        try:
            max_idx = int(np.nanargmax(vals))
            min_idx = int(np.nanargmin(vals))
            max_val = float(vals[max_idx])
            min_val = float(vals[min_idx])
            avg_val = float(np.nanmean(vals))
            max_var = variant_metrics[max_idx]["variant"]
            min_var = variant_metrics[min_idx]["variant"]
        except ValueError:
            max_val = min_val = avg_val = float('nan')
            max_var = min_var = ""

        cells = tbl.rows[r_idx].cells
        cells[0].text = label
        # Format highest and lowest with variant names in parentheses
        try:
            cells[1].text = f"{fmt.format(max_val)} ({max_var})"
        except Exception:
            cells[1].text = f"{max_val} ({max_var})"
        try:
            cells[2].text = f"{fmt.format(min_val)} ({min_var})"
        except Exception:
            cells[2].text = f"{min_val} ({min_var})"
        try:
            cells[3].text = fmt.format(avg_val)
        except Exception:
            cells[3].text = str(avg_val)

#endregion
#region 1.2. SUMMARY STATISTICS - BENCHMARK VARIANTS (if available) =======================================================================================================================================

if df_benchmarks is not None and len(df_benchmarks) > 0:
    doc.add_heading("1.2. Summary Statistics - Benchmark Variants", level=2)
    
    for benchmark_name in sorted(df_benchmarks['benchmark_group'].unique()):
        df_bench = df_benchmarks[df_benchmarks['benchmark_group'] == benchmark_name]
        display_name = benchmark_names_map.get(benchmark_name, benchmark_name.replace("_", " ").title())
        
        doc.add_paragraph(display_name, style="Heading 3")
        
        bench_lifetimes = df_bench["lifetime_ticks"].values
        
        bench_mean = bench_lifetimes.mean()
        bench_median = np.median(bench_lifetimes)
        bench_std = bench_lifetimes.std()
        bench_min = bench_lifetimes.min()
        bench_max = bench_lifetimes.max()
        bench_range = bench_max - bench_min
        bench_iqr = np.percentile(bench_lifetimes, 75) - np.percentile(bench_lifetimes, 25)
        bench_skewness = skew(bench_lifetimes)
        bench_kurtosis = kurtosis(bench_lifetimes)
        
        table = doc.add_table(rows=15, cols=2)
        table.style = "Light Grid Accent 1"
        cells = table.rows[0].cells
        cells[0].text = "Statistic"
        cells[1].text = "Value"
        bench_p5 = np.percentile(bench_lifetimes, 5)
        bench_p25 = np.percentile(bench_lifetimes, 25)
        bench_p75 = np.percentile(bench_lifetimes, 75)
        bench_p95 = np.percentile(bench_lifetimes, 95)
        bench_cv = (bench_std / bench_mean) * 100 if bench_mean != 0 else float('nan')

        bench_metrics = [
            ("Mean", f"{bench_mean:.2f} ticks"),
            ("Median", f"{bench_median:.2f} ticks"),
            ("Std Dev", f"{bench_std:.2f} ticks"),
            ("Min", f"{bench_min:.2f} ticks"),
            ("Max", f"{bench_max:.2f} ticks"),
            ("Range", f"{bench_range:.2f} ticks"),
            ("IQR (25th-75th percentile)", f"{bench_iqr:.2f} ticks"),
            ("5th Percentile", f"{bench_p5:.2f} ticks"),
            ("25th Percentile", f"{bench_p25:.2f} ticks"),
            ("75th Percentile", f"{bench_p75:.2f} ticks"),
            ("95th Percentile", f"{bench_p95:.2f} ticks"),
            ("Skewness", f"{bench_skewness:.3f}"),
            ("Kurtosis (excess)", f"{bench_kurtosis:.3f}"),
            ("Coefficient of Variation", f"{bench_cv:.2f} %"),
        ]
        
        for row_idx, (metric_name, value_str) in enumerate(bench_metrics, 1):
            cells = table.rows[row_idx].cells
            cells[0].text = metric_name
            cells[1].text = value_str
        
        # Add network visualization for this benchmark
        doc.add_paragraph()  # Add spacing
        
        # Try to find and load benchmark HDF5 file
        benchmark_hdf5_file = None
        if df_benchmarks is not None and benchmark_name in df_benchmarks['benchmark_group'].values:
            # Find the HDF5 file path from benchmark data
            bench_sample = df_benchmarks[df_benchmarks['benchmark_group'] == benchmark_name].iloc[0]
            if 'benchmark_folder_path' in bench_sample:
                bench_folder = Path(bench_sample['benchmark_folder_path'])
                # Look for .h5 file in that folder
                h5_files = list(bench_folder.glob("*.h5"))
                if h5_files:
                    benchmark_hdf5_file = h5_files[0]
        
        if benchmark_hdf5_file and benchmark_hdf5_file.exists():
            try:
                # Load network configuration
                neuron_positions, neuron_types = network_viz.load_network_viz_config(NETWORK_VIZ_CONFIG)
                
                # Load wiring and modulation from HDF5
                df_wiring = load_wiring_from_hdf5(benchmark_hdf5_file, "variant_01")
                df_modulation = load_modulation_from_hdf5(benchmark_hdf5_file, "variant_01")
                
                # Define which weight columns and labels to show
                # Creates separate full-size figures then combines them
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
                
                # Temporarily save to CSV for network viz function
                import tempfile
                with tempfile.TemporaryDirectory() as tmpdir:
                    wiring_csv = Path(tmpdir) / "wiring.csv"
                    modulation_csv = Path(tmpdir) / "modulation.csv"
                    
                    df_wiring.to_csv(wiring_csv, index=False)
                    if len(df_modulation) > 0:
                        df_modulation.to_csv(modulation_csv, index=False)
                    
                    # Draw networks and combine
                    # Sanitize display_name for use in filename
                    safe_display_name = display_name.replace(" ", "_").replace("-", "_")
                    network_fig_path = results_dir / f"network_{safe_display_name}.png"
                    network_viz.draw_and_combine_networks(
                        wiring_csv=str(wiring_csv),
                        weight_columns=weight_columns,
                        panel_labels=panel_labels,
                        output_path=str(network_fig_path),
                        modulation_csv=str(modulation_csv) if len(df_modulation) > 0 else None,
                        neuron_positions=neuron_positions,
                        neuron_types=neuron_types,
                        title=f'Network Development - {display_name}'
                    )
                    
                    # Add to document
                    doc.add_paragraph("Network Development (Initial vs. Final States):", style="Heading 4")
                    doc.add_picture(str(network_fig_path), width=Inches(6.5))
                
            except FileNotFoundError as e:
                print(f"Warning: Network config not found for benchmark {display_name}: {e}")
            except Exception as e:
                print(f"Error loading network visualization for benchmark {display_name}: {e}")
        else:
            print(f"Warning: Could not find HDF5 file for benchmark {display_name}")

#endregion  # closes 1.2
#endregion  # closes 1
#region 2. GROUP SELECTION ================================================================================================================================================================================

doc.add_heading("2. Group Selection Based on Survival Race", level=2)

doc.add_paragraph(
    f"Success is measured based on median survival across {runs_per_variant:.0f} runs with different randomizer seeds. "
    f"The {n_variants} different wiring variants are split into successful and unsuccessful groups based on their performance. "
    f"The achievable range of median liefetimes is split into percentiles. Variants in the 'successful' group had median lifetimes in the top 10% of that range, "
    f"while variants in the 'unsuccessful' group had median lifetimes in the bottom 10% of that range. This approach respects natural performance gaps "
    f"and identifies only the variants that truly solve the task well regardless of the number of variants that achieve this."
)

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

table = doc.add_table(rows=4, cols=2)
table.style = "Light Grid Accent 1"
cells = table.rows[0].cells
cells[0].text = "Metric"
cells[1].text = "Value (ticks)"
cells = table.rows[1].cells
cells[0].text = "Successful threshold (top 10% of range)"
cells[1].text = f">= {median_successful_threshold:.1f}"
cells = table.rows[2].cells
cells[0].text = "Unsuccessful threshold (bottom 10% of range)"
cells[1].text = f"<= {median_unsuccessful_threshold:.1f}"
cells = table.rows[3].cells
cells[0].text = "Variants identified in each group"
cells[1].text = f"Successful: {len(successful_variants)}, Unsuccessful: {len(unsuccessful_variants)}"
doc.add_paragraph()

# Compact display of successful / unsuccessful variant lists (numbers only)
def _extract_key(v):
    if "_" in v:
        s = v.split("_", 1)[1]
    else:
        s = v
    try:
        return (0, int(s))  # Integers sort first, then by numeric value
    except Exception:
        return (1, s)  # Non-integers sort second, then lexically

success_sorted = sorted(successful_variants, key=_extract_key)
unsuccess_sorted = sorted(unsuccessful_variants, key=_extract_key)

success_nums = ", ".join(str(_extract_key(v)[1]) for v in success_sorted) if success_sorted else "None"
unsuccess_nums = ", ".join(str(_extract_key(v)[1]) for v in unsuccess_sorted) if unsuccess_sorted else "None"

# Create a compact two-row table
tbl = doc.add_table(rows=2, cols=1)
tbl.style = "Light Grid Accent 1"

# Row 1: Successful
cell = tbl.rows[0].cells[0]
p = cell.paragraphs[0]
run = p.add_run("Successful Variant Group (Top 10%): ")
run.font.size = Pt(8)
run.bold = True
run2 = p.add_run(success_nums)
run2.font.size = Pt(8)

# Row 2: Unsuccessful
cell = tbl.rows[1].cells[0]
p = cell.paragraphs[0]
run = p.add_run("Unsuccessful Variant Group (Bottom 10%): ")
run.font.size = Pt(8)
run.bold = True
run2 = p.add_run(unsuccess_nums)
run2.font.size = Pt(8)
doc.add_paragraph()

# =====================================================================
# Create survival race plot
# =====================================================================
doc.add_paragraph(
    "The survival race plot tracks how many individuals from each wiring variant survive over time. "
    "Each curve represents a variant's cumulative survival, showing how fitness changes across the simulation. "
    "We use this visualization to identify which variants are robustly long-lived. The comparison table shows key statistics "
    "for successful versus unsuccessful variants, helping quantify the performance gap."
)

doc.add_paragraph(
    f"This plot shows the survival curves for all {n_variants} wiring variants. "
    f"Variants are color-coded by their performance achievement (range-based thresholds):"
)
doc.add_paragraph(f"  • Green: Top 10% of variants (median survival ≥ {median_successful_threshold:.1f} ticks) — {len(successful_variants)} variants")
doc.add_paragraph(f"  • Red: Bottom 10% of variants (median survival ≤ {median_unsuccessful_threshold:.1f} ticks) — {len(unsuccessful_variants)} variants")
doc.add_paragraph(f"  • Gray: Middle range (other variants)")
if df_benchmarks is not None:
    doc.add_paragraph("  • Colored lines: Benchmark variants")

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

# Prepare successful/unsuccessful lifetime arrays from raw data so the comparison table can use them
successful_times = df_all[df_all["variant"].isin(successful_variants)]["lifetime_ticks"].values
unsuccessful_times = df_all[df_all["variant"].isin(unsuccessful_variants)]["lifetime_ticks"].values

doc.add_paragraph("Successful vs Unsuccessful Comparison:")
table = doc.add_table(rows=7, cols=3)
table.style = "Light Grid Accent 1"
cells = table.rows[0].cells
cells[0].text = "Metric"
cells[1].text = f"Successful (Top 10% of Variants)"
cells[2].text = f"Unsuccessful (Bottom 10% of Variants)"
cells = table.rows[1].cells
cells[0].text = "Number of runs"
cells[1].text = f"{len(successful_times)}"
cells[2].text = f"{len(unsuccessful_times)}"
cells = table.rows[2].cells
cells[0].text = "Mean (ticks)"
cells[1].text = f"{successful_times.mean():.1f}"
cells[2].text = f"{unsuccessful_times.mean():.1f}"
cells = table.rows[3].cells
cells[0].text = "Median (ticks)"
cells[1].text = f"{np.median(successful_times):.1f}"
cells[2].text = f"{np.median(unsuccessful_times):.1f}"
cells = table.rows[4].cells
cells[0].text = "Std Dev (ticks)"
cells[1].text = f"{successful_times.std():.1f}"
cells[2].text = f"{unsuccessful_times.std():.1f}"
cells = table.rows[5].cells
cells[0].text = "Min (ticks)"
cells[1].text = f"{successful_times.min():.1f}"
cells[2].text = f"{unsuccessful_times.min():.1f}"
cells = table.rows[6].cells
cells[0].text = "Max (ticks)"
cells[1].text = f"{successful_times.max():.1f}"
cells[2].text = f"{unsuccessful_times.max():.1f}"

# =====================================================================
# Network visualizations for group representatives
# =====================================================================

doc.add_paragraph()
doc.add_paragraph("Network Development: Successful vs Unsuccessful Variants", style="Heading 4")
doc.add_paragraph(
    "Below are combined network visualizations showing initial wiring and the first three runtime states "
    "for representative variants from the successful and unsuccessful groups."
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
    doc.add_paragraph("Successful Variants (Top 10%):", style="Heading 5")
    
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
                    title=f'Network Development - {variant_name} (Successful)'
                )
                
                # Add to document
                doc.add_picture(str(network_fig_path), width=Inches(6.5))
                print(f"Saved network visualization: {network_fig_path}")
                
        except Exception as e:
            print(f"Error creating network visualization for {variant_name}: {e}")

# Process unsuccessful variants (first up to 3)
if len(unsuccessful_variants) > 0:
    doc.add_paragraph()
    doc.add_paragraph("Unsuccessful Variants (Bottom 10%):", style="Heading 5")
    
    unsuccessful_sorted = sorted(unsuccessful_variants, key=_get_variant_number)[:3]
    
    # Create network plots for each unsuccessful variant
    for variant_name in unsuccessful_sorted:
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
                    title=f'Network Development - {variant_name} (Unsuccessful)'
                )
                
                # Add to document
                doc.add_picture(str(network_fig_path), width=Inches(6.5))
                print(f"Saved network visualization: {network_fig_path}")
            
        except Exception as e:
            print(f"Error creating network visualization for {variant_name}: {e}")

#endregion # closes 2.
#region 3. COMPARISON =====================================================================================================================================================================================


doc.add_heading("3. Comparison", level=2)

doc.add_paragraph(
    "This section compares metrics between successful and unsuccessful random variants, and against benchmark implementations where available. "
    "Survival times comparison helps validate our group selection; subsequent sections examine behavioral and performance metrics."
)

#region 3.1. SURVIVAL TIMES ========================================================================================================================================================================

doc.add_heading("3.1. Survival Times", level=3)

doc.add_paragraph(
    "Comparison of lifetime survival between successful and unsuccessful variants. "
    "Note: This metric is somewhat redundant since we selected groups specifically based on survival time, "
    "but it provides useful verification of group separation and reveals distribution characteristics."
)

# Prepare data for survival times comparison
plot_data = []

for variant_name in unsuccessful_variants:
    df_variant = df_all[df_all["variant"] == variant_name]
    for _, row in df_variant.iterrows():
        plot_data.append({
            "variant": variant_name,
            "group": "Unsuccessful\n(Bottom 10%)",
            "lifetime_ticks": row["lifetime_ticks"]
        })

for variant_name in successful_variants:
    df_variant = df_all[df_all["variant"] == variant_name]
    for _, row in df_variant.iterrows():
        plot_data.append({
            "variant": variant_name,
            "group": "Successful\n(Top 10%)",
            "lifetime_ticks": row["lifetime_ticks"]
        })

if df_benchmarks is not None:
    for benchmark_name in sorted(df_benchmarks['benchmark_group'].unique()):
        df_benchmark = df_benchmarks[df_benchmarks["benchmark_group"] == benchmark_name]
        display_name = benchmark_names_map.get(benchmark_name, benchmark_name.replace("_", " ").title())
        for _, row in df_benchmark.iterrows():
            plot_data.append({
                "variant": row["variant"],
                "group": f"{display_name}\n(Benchmark)",
                "lifetime_ticks": row["lifetime_ticks"]
            })

df_survival = pd.DataFrame(plot_data)

# Plot survival times comparison
title_survival = "Survival Times: Unsuccessful vs Successful Wiring Variants"
if df_benchmarks is not None:
    title_survival = "Survival Times: Random Variants vs Benchmarks"

output_path_survival = results_dir / f"comp_survival_{EXPERIMENT_NAME}_{EXPERIMENT_DISPLAY_NAME.replace(' ', '_')}.png"
fig = plot_metric_comparison(
    df_survival, 
    "lifetime_ticks", 
    "group",
    title=title_survival,
    y_label="Lifetime (ticks)",
    output_path=output_path_survival,
    doc=doc
)
fig_paths.append(("Survival Times Comparison", output_path_survival))
plt.show()

# Calculate statistics for survival times (df_survival-based arrays are available later if needed)

benchmark_times_dict = {}
if df_benchmarks is not None:
    for benchmark_name in sorted(df_benchmarks['benchmark_group'].unique()):
        display_name = benchmark_names_map.get(benchmark_name, benchmark_name.replace("_", " ").title())
        group_label = f"{display_name}\n(Benchmark)"
        group_times = df_survival[df_survival['group'] == group_label]["lifetime_ticks"].values
        if len(group_times) > 0:
            benchmark_times_dict[display_name] = group_times

stats_survival = calculate_statistics(unsuccessful_times, successful_times, benchmark_times_dict)

# Report statistics for survival times
report_statistics(doc, stats_survival, "Survival Times")

doc.add_paragraph("TBD - Interpretation of survival time differences.", style="Normal")

#endregion  # closes 3.1
#region 3.2. MOVEMENT EFFICIENCY ========================================================================================================================================================================

doc.add_heading("3.2. Movement Efficiency", level=3)

doc.add_paragraph(
    "Movement efficiency quantifies how effectively variants use sensory information to guide movement. "
    "Calculated as movements divided by sensing events in each cardinal direction (N, E, S, W). "
    "A ratio close to 1.0 indicates tight coupling between sensing and acting; higher ratios suggest "
    "either deliberate multi-movement strategies or noisy sensor integration."
)


# Direction mapping
direction_map = {
    'N': 'north',
    'E': 'east',
    'S': 'south',
    'W': 'west'
}
directions = ['N', 'E', 'S', 'W']

doc.add_paragraph("Movement Efficiency = Movement Count ÷ Sensing Count per direction")

# Note: summary statistics table for movement efficiency will be created
# after computing `df_efficiency` (below) so it is derived from `df_all`.

# ==================================================================
# Movement efficiency plot
# ==================================================================

# Calculate efficiency for each run: movement / sensing per direction
efficiency_data = []

# Process unsuccessful variants
for variant_name in unsuccessful_variants:
    df_variant = df_all[df_all["variant"] == variant_name]
    for _, row in df_variant.iterrows():
        for direction in directions:
            move_col = f"moves_{direction_map[direction]}"
            sense_col = f"food_sensed_{direction_map[direction]}"
            
            if move_col in row.index and sense_col in row.index:
                moves = row[move_col]
                senses = row[sense_col]
                # Avoid division by zero: if no sensing events, set efficiency to 0
                efficiency = moves / senses if senses > 0 else 0
                
                efficiency_data.append({
                    "group": "Unsuccessful",
                    "direction": direction,
                    "efficiency": efficiency
                })

# Process successful variants
for variant_name in successful_variants:
    df_variant = df_all[df_all["variant"] == variant_name]
    for _, row in df_variant.iterrows():
        for direction in directions:
            move_col = f"moves_{direction_map[direction]}"
            sense_col = f"food_sensed_{direction_map[direction]}"
            
            if move_col in row.index and sense_col in row.index:
                moves = row[move_col]
                senses = row[sense_col]
                efficiency = moves / senses if senses > 0 else 0
                
                efficiency_data.append({
                    "group": "Successful",
                    "direction": direction,
                    "efficiency": efficiency
                })

# Add benchmark data if available
if df_benchmarks is not None:
    for benchmark_name in sorted(df_benchmarks['benchmark_group'].unique()):
        df_bench = df_benchmarks[df_benchmarks["benchmark_group"] == benchmark_name]
        display_name = benchmark_names_map.get(benchmark_name, benchmark_name.replace("_", " ").title())
        
        for variant_name in sorted(df_bench["variant"].unique()):
            df_variant = df_bench[df_bench["variant"] == variant_name]
            for _, row in df_variant.iterrows():
                for direction in directions:
                    move_col = f"moves_{direction_map[direction]}"
                    sense_col = f"food_sensed_{direction_map[direction]}"
                    
                    if move_col in row.index and sense_col in row.index:
                        moves = row[move_col]
                        senses = row[sense_col]
                        efficiency = moves / senses if senses > 0 else 0
                        
                        efficiency_data.append({
                            "group": display_name,
                            "direction": direction,
                            "efficiency": efficiency
                        })

df_efficiency = pd.DataFrame(efficiency_data)

if not df_efficiency.empty:
    # Compute summary statistics for bar plots
    summary_stats = df_efficiency.groupby(['group', 'direction'])['efficiency'].agg(['mean', 'std', 'sem']).reset_index()
    
    # Get unique groups and determine layout
    groups = ['Unsuccessful', 'Successful']
    
    if df_benchmarks is not None:
        for benchmark_name in sorted(df_benchmarks['benchmark_group'].unique()):
            display_name = benchmark_names_map.get(benchmark_name, benchmark_name.replace("_", " ").title())
            groups.append(display_name)
    
    # Calculate number of rows needed (max 3 plots per row)
    num_groups = len(groups)
    num_cols = min(3, num_groups)
    num_rows = (num_groups + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 5*num_rows), sharey=True)
    
    # Handle case where there's only one subplot
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_rows == 1 or num_cols == 1:
        axes = axes.reshape(num_rows, num_cols)
    
    fig.suptitle("Movement Efficiency by Direction (Movements / Sensing Events)", fontsize=14, fontweight='bold')
    
    # Color mapping for groups
    group_colors = {
        'Unsuccessful': RED_COLOR,
        'Successful': GREEN_COLOR
    }
    
    if df_benchmarks is not None:
        benchmark_colors_list = ["#2E8B9E", "#9E2E8B", "#8B9E2E", "#2E8B57"]
        for bench_idx, benchmark_name in enumerate(sorted(df_benchmarks['benchmark_group'].unique())):
            display_name = benchmark_names_map.get(benchmark_name, benchmark_name.replace("_", " ").title())
            group_colors[display_name] = benchmark_colors_list[bench_idx % len(benchmark_colors_list)]
    
    # Plot each group
    x_pos = np.arange(len(directions))
    # Build and insert a summary table into the Word document using values
    try:
        unsuccessful_eff = summary_stats[summary_stats['group'] == 'Unsuccessful'].sort_values('direction')
        successful_eff = summary_stats[summary_stats['group'] == 'Successful'].sort_values('direction')

        table = doc.add_table(rows=5, cols=3)
        table.style = "Light Grid Accent 1"
        cells = table.rows[0].cells
        cells[0].text = "Direction"
        cells[1].text = "Unsuccessful (Mean ± Std)"
        cells[2].text = "Successful (Mean ± Std)"

        for idx, direction in enumerate(directions, 1):
            cells = table.rows[idx].cells
            cells[0].text = direction
            # If group-direction row is missing, fill with N/A
            try:
                unsucc_row = unsuccessful_eff[unsuccessful_eff['direction'] == direction].iloc[0]
                cells[1].text = f"{unsucc_row['mean']:.3f} ± {unsucc_row['std']:.3f}"
            except Exception:
                cells[1].text = "N/A"
            try:
                succ_row = successful_eff[successful_eff['direction'] == direction].iloc[0]
                cells[2].text = f"{succ_row['mean']:.3f} ± {succ_row['std']:.3f}"
            except Exception:
                cells[2].text = "N/A"

        doc.add_paragraph()
    except Exception:
        # If table creation fails, continue without interrupting the analysis
        doc.add_paragraph("Movement efficiency summary table could not be created.")
   
    for plot_idx, group_name in enumerate(groups):
        row = plot_idx // num_cols
        col = plot_idx % num_cols
        
        ax = axes[row, col]
        
        group_data = summary_stats[summary_stats['group'] == group_name].sort_values('direction')
        
        if len(group_data) > 0:
            ax.bar(
                x_pos,
                group_data['mean'].values,
                yerr=group_data['sem'].values,
                capsize=5,
                color=group_colors.get(group_name, '#808080'),
                alpha=0.7,
                edgecolor='black',
                linewidth=1.5
            )
        
        # Add dashed line at y=1 to mark correct decision threshold
        ax.axhline(y=1, color='black', linestyle='--', linewidth=1.5, alpha=0.6, label='Correct Decision (1:1)')
        
        ax.set_xlabel("Direction", fontsize=11)
        if col == 0:
            ax.set_ylabel("Movement Efficiency", fontsize=11)
        title = group_name
        if "Bottom 10%" in group_name or "Top 10%" in group_name:
            # Keep original naming for successful/unsuccessful
            title = group_name.replace("\n(Bottom 10%)", "").replace("\n(Top 10%)", "")
            title = "Unsuccessful (Bottom 10%)" if "Unsuccessful" in title else "Successful (Top 10%)"
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(directions)
        ax.grid(axis='y', alpha=0.3)
    
    # Hide unused subplots
    for plot_idx in range(num_groups, num_rows * num_cols):
        row = plot_idx // num_cols
        col = plot_idx % num_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure
    fig_efficiency_path = results_dir / f"movement_efficiency_{EXPERIMENT_NAME}_{EXPERIMENT_DISPLAY_NAME.replace(' ', '_')}.png"
    fig.savefig(fig_efficiency_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {fig_efficiency_path}")
    fig_paths.append(("Movement Efficiency by Direction", fig_efficiency_path))
    plt.show()
    
    if fig_efficiency_path.exists():
        doc.add_picture(str(fig_efficiency_path), width=Inches(6))
    doc.add_paragraph()
    
    doc.add_paragraph("TBD - Interpretation of movement efficiency by direction.", style="Normal")

#endregion  # closes 3.2
#region 3.3. DECISION ACCURACY ========================================================================================================================================================================


doc.add_heading("3.3. Decision Accuracy", level=3)

doc.add_paragraph(
    "Decision accuracy quantifies how often choices are aligned with food source locations. "
    "This measures, whether the wiring interprets sensory input to direct movement toward food. "
    "Higher accuracy indicates more direct coupling of decisions. We compare the distribution of accuracy values between successful and unsuccessful "
    "variants to determine whether fitness correlates with decision-making precision, or if other factors like movement efficiency play a larger role. " 
    "Note that moving onto sensed food is a very shortsighted strategy that might not lead to an optimal solution in more complex tasks e.g. with food " 
    "regrow in previous positions."
)

# Prepare data for decision accuracy
decision_accuracy_data = []

for variant_name in unsuccessful_variants:
    df_variant = df_all[df_all["variant"] == variant_name]
    for _, row in df_variant.iterrows():
        if 'decisions' in row.index and 'correct_decisions' in row.index:
            total_decisions = row['decisions']
            correct_decisions = row['correct_decisions']
            accuracy = correct_decisions / total_decisions if total_decisions > 0 else 0
            decision_accuracy_data.append({
                "variant": variant_name,
                "group": "Unsuccessful\n(Bottom 10%)",
                "accuracy": accuracy
            })

for variant_name in successful_variants:
    df_variant = df_all[df_all["variant"] == variant_name]
    for _, row in df_variant.iterrows():
        if 'decisions' in row.index and 'correct_decisions' in row.index:
            total_decisions = row['decisions']
            correct_decisions = row['correct_decisions']
            accuracy = correct_decisions / total_decisions if total_decisions > 0 else 0
            decision_accuracy_data.append({
                "variant": variant_name,
                "group": "Successful\n(Top 10%)",
                "accuracy": accuracy
            })

if df_benchmarks is not None:
    for benchmark_name in sorted(df_benchmarks['benchmark_group'].unique()):
        df_benchmark = df_benchmarks[df_benchmarks["benchmark_group"] == benchmark_name]
        display_name = benchmark_names_map.get(benchmark_name, benchmark_name.replace("_", " ").title())
        for _, row in df_benchmark.iterrows():
            if 'decisions' in row.index and 'correct_decisions' in row.index:
                total_decisions = row['decisions']
                correct_decisions = row['correct_decisions']
                accuracy = correct_decisions / total_decisions if total_decisions > 0 else 0
                decision_accuracy_data.append({
                    "variant": row["variant"],
                    "group": f"{display_name}\n(Benchmark)",
                    "accuracy": accuracy
                })

df_accuracy = pd.DataFrame(decision_accuracy_data)

if not df_accuracy.empty:
    # Plot decision accuracy comparison
    output_path_accuracy = results_dir / f"decision_accuracy_{EXPERIMENT_NAME}_{EXPERIMENT_DISPLAY_NAME.replace(' ', '_')}.png"
    fig = plot_metric_comparison(
        df_accuracy,
        "accuracy",
        "group",
        title="Decision Accuracy: Variants Comparison",
        y_label="Accuracy (Correct / Total)",
        output_path=output_path_accuracy,
        doc=doc
    )
    fig_paths.append(("Decision Accuracy Comparison", output_path_accuracy))
    plt.show()
    
    doc.add_paragraph("Decision Accuracy = Correct Decisions ÷ Total Decisions")
    # Calculate statistics for decision accuracy
    unsuccessful_accuracy = df_accuracy[df_accuracy["group"] == "Unsuccessful\n(Bottom 10%)"]["accuracy"].values
    successful_accuracy = df_accuracy[df_accuracy["group"] == "Successful\n(Top 10%)"]["accuracy"].values
    
    benchmark_accuracy_dict = {}
    if df_benchmarks is not None:
        for benchmark_name in sorted(df_benchmarks['benchmark_group'].unique()):
            display_name = benchmark_names_map.get(benchmark_name, benchmark_name.replace("_", " ").title())
            group_label = f"{display_name}\n(Benchmark)"
            group_accuracy = df_accuracy[df_accuracy['group'] == group_label]["accuracy"].values
            if len(group_accuracy) > 0:
                benchmark_accuracy_dict[display_name] = group_accuracy
    
    stats_accuracy = calculate_statistics(unsuccessful_accuracy, successful_accuracy, benchmark_accuracy_dict)
    
    # Report statistics for decision accuracy
    report_statistics(doc, stats_accuracy, "Decision Accuracy")
    
    doc.add_paragraph("TBD - Interpretation of decision accuracy differences.", style="Normal")
else:
    doc.add_paragraph("No decision accuracy data found in summary files.")

print("\nDecision Accuracy Analysis Complete")

#endregion  # closes 3.3
#region 3.4. DISTANCE TRAVELED ========================================================================================================================================================================

doc.add_heading("3.4. Distance Traveled", level=3)

doc.add_paragraph(
    "Distance travelled total and normalized per tick for lifespan differences. "
    "We compare movement patterns between successful and unsuccessful variants to understand how spatial exploration versus local exploitation correlates with fitness."
)

#region 3.4.1 Absolute Distance ========================================================================================================================================================================

doc.add_heading("3.4.1. Distance Traveled (Absolute)", level=4)

# Prepare data for absolute distance comparison
plot_data = []

for variant_name in unsuccessful_variants:
    df_variant = df_all[df_all["variant"] == variant_name]
    for _, row in df_variant.iterrows():
        if 'distance' in row.index:
            plot_data.append({
                "variant": variant_name,
                "group": "Unsuccessful\n(Bottom 10%)",
                "distance": row["distance"]
            })

for variant_name in successful_variants:
    df_variant = df_all[df_all["variant"] == variant_name]
    for _, row in df_variant.iterrows():
        if 'distance' in row.index:
            plot_data.append({
                "variant": variant_name,
                "group": "Successful\n(Top 10%)",
                "distance": row["distance"]
            })

# Filter benchmarks from df_all (benchmarks were concatenated into df_all earlier)
if 'benchmark_group' in df_all.columns:
    for benchmark_name in sorted(df_all[df_all['benchmark_group'].notna()]['benchmark_group'].unique()):
        df_benchmark = df_all[df_all['benchmark_group'] == benchmark_name]
        display_name = benchmark_names_map.get(benchmark_name, benchmark_name.replace("_", " ").title())
        for _, row in df_benchmark.iterrows():
            if 'distance' in row.index:
                plot_data.append({
                    "variant": row["variant"],
                    "group": f"{display_name}\n(Benchmark)",
                    "distance": row["distance"]
                })

df_distance = pd.DataFrame(plot_data)

if not df_distance.empty:
    output_path_distance = results_dir / f"comp_distance_{EXPERIMENT_NAME}_{EXPERIMENT_DISPLAY_NAME.replace(' ', '_')}.png"
    fig = plot_metric_comparison(
        df_distance,
        "distance",
        "group",
        title="Distance Traveled: Unsuccessful vs Successful Wiring Variants",
        y_label="Distance (units)",
        output_path=output_path_distance,
        doc=doc
    )
    fig_paths.append(("Distance (Absolute) Comparison", output_path_distance))
    plt.show()
    
    unsuccessful_distance = df_distance[df_distance["group"] == "Unsuccessful\n(Bottom 10%)"]["distance"].values
    successful_distance = df_distance[df_distance["group"] == "Successful\n(Top 10%)"]["distance"].values
    
    benchmark_distance_dict = {}
    if df_benchmarks is not None:
        for benchmark_name in sorted(df_benchmarks['benchmark_group'].unique()):
            display_name = benchmark_names_map.get(benchmark_name, benchmark_name.replace("_", " ").title())
            group_label = f"{display_name}\n(Benchmark)"
            group_distance = df_distance[df_distance['group'] == group_label]["distance"].values
            if len(group_distance) > 0:
                benchmark_distance_dict[display_name] = group_distance
    
    stats_distance = calculate_statistics(unsuccessful_distance, successful_distance, benchmark_distance_dict)
    report_statistics(doc, stats_distance, "Distance Traveled")
    
    doc.add_paragraph("TBD - Interpretation of distance differences.", style="Normal")

#endregion  # closes 3.4.1
#region 3.4.2 Distance Per Tick (Normalized) ========================================================================================================================================================================

doc.add_heading("3.4.2. Distance Per Tick (Normalized)", level=4)

# Prepare data for distance per tick
plot_data = []

for variant_name in unsuccessful_variants:
    df_variant = df_all[df_all["variant"] == variant_name]
    for _, row in df_variant.iterrows():
        # Derive distance_per_tick from raw tracked data (distance and lifetime_ticks)
        if 'distance' in row.index and 'lifetime_ticks' in row.index and row['lifetime_ticks'] > 0:
            val = row['distance'] / row['lifetime_ticks']
            plot_data.append({
                "variant": variant_name,
                "group": "Unsuccessful\n(Bottom 10%)",
                "distance_per_tick": val
            })

for variant_name in successful_variants:
    df_variant = df_all[df_all["variant"] == variant_name]
    for _, row in df_variant.iterrows():
        if 'distance' in row.index and 'lifetime_ticks' in row.index and row['lifetime_ticks'] > 0:
            val = row['distance'] / row['lifetime_ticks']
            plot_data.append({
                "variant": variant_name,
                "group": "Successful\n(Top 10%)",
                "distance_per_tick": val
            })

if df_benchmarks is not None:
    for benchmark_name in sorted(df_benchmarks['benchmark_group'].unique()):
        df_benchmark = df_benchmarks[df_benchmarks["benchmark_group"] == benchmark_name]
        display_name = benchmark_names_map.get(benchmark_name, benchmark_name.replace("_", " ").title())
        for _, row in df_benchmark.iterrows():
                if 'distance' in row.index and 'lifetime_ticks' in row.index and row['lifetime_ticks'] > 0:
                    val = row['distance'] / row['lifetime_ticks']
                    plot_data.append({
                        "variant": row["variant"],
                        "group": f"{display_name}\n(Benchmark)",
                        "distance_per_tick": val
                    })

df_distance_per_tick = pd.DataFrame(plot_data)

# Plot and comparative statistics (plot_metric_comparison + report_statistics)
output_path_distance_per_tick = results_dir / f"comp_distance_per_tick_{EXPERIMENT_NAME}_{EXPERIMENT_DISPLAY_NAME.replace(' ', '_')}.png"
fig = plot_metric_comparison(
    df_distance_per_tick,
    "distance_per_tick",
    "group",
    title="Distance Per Tick: Unsuccessful vs Successful Wiring Variants",
    y_label="Distance / Tick",
    output_path=output_path_distance_per_tick,
    doc=doc
)
fig_paths.append(("Distance Per Tick Comparison", output_path_distance_per_tick))
plt.show()

unsuccessful_dist_pt = df_distance_per_tick[df_distance_per_tick["group"] == "Unsuccessful\n(Bottom 10%)"]["distance_per_tick"].values
successful_dist_pt = df_distance_per_tick[df_distance_per_tick["group"] == "Successful\n(Top 10%)"]["distance_per_tick"].values

benchmark_dist_pt_dict = {}
if df_benchmarks is not None:
    for benchmark_name in sorted(df_benchmarks['benchmark_group'].unique()):
        display_name = benchmark_names_map.get(benchmark_name, benchmark_name.replace("_", " ").title())
        group_label = f"{display_name}\n(Benchmark)"
        group_dist_pt = df_distance_per_tick[df_distance_per_tick['group'] == group_label]["distance_per_tick"].values
        if len(group_dist_pt) > 0:
            benchmark_dist_pt_dict[display_name] = group_dist_pt

stats_dist_pt = calculate_statistics(unsuccessful_dist_pt, successful_dist_pt, benchmark_dist_pt_dict)
report_statistics(doc, stats_dist_pt, "Distance Per Tick")

doc.add_paragraph("TBD - Interpretation of distance per tick differences.", style="Normal")

#endregion  # closes 3.4.2
#endregion  # closes 3.4
#region 3.5. FOOD CONSUMPTION ========================================================================================================================================================================


doc.add_heading("3.5. Food Consumption", level=3)

doc.add_paragraph(
    "Food consumption is the ultimate solution of the task. The more food consumed, the longer the survival. "
    "Normalized per tick food consumption may be the metric most directly linked to survival fitness."
)

#region 3.5.1 Food Consumption (Absolute) ========================================================================================================================================================================
doc.add_heading("3.5.1. Food Consumption (Absolute)", level=4)

# Prepare data for absolute food consumption
plot_data = []

for variant_name in unsuccessful_variants:
    df_variant = df_all[df_all["variant"] == variant_name]
    for _, row in df_variant.iterrows():
        if 'foods' in row.index:
            plot_data.append({
                "variant": variant_name,
                "group": "Unsuccessful\n(Bottom 10%)",
                "foods": row["foods"]
            })

for variant_name in successful_variants:
    df_variant = df_all[df_all["variant"] == variant_name]
    for _, row in df_variant.iterrows():
        if 'foods' in row.index:
            plot_data.append({
                "variant": variant_name,
                "group": "Successful\n(Top 10%)",
                "foods": row["foods"]
            })

# Filter benchmarks from df_all (benchmarks were concatenated into df_all earlier)
if 'benchmark_group' in df_all.columns:
    for benchmark_name in sorted(df_all[df_all['benchmark_group'].notna()]['benchmark_group'].unique()):
        df_benchmark = df_all[df_all['benchmark_group'] == benchmark_name]
        display_name = benchmark_names_map.get(benchmark_name, benchmark_name.replace("_", " ").title())
        for _, row in df_benchmark.iterrows():
            if 'foods' in row.index:
                plot_data.append({
                    "variant": row["variant"],
                    "group": f"{display_name}\n(Benchmark)",
                    "foods": row["foods"]
                })

df_foods = pd.DataFrame(plot_data)

if not df_foods.empty:
    output_path_foods = results_dir / f"comp_foods_{EXPERIMENT_NAME}_{EXPERIMENT_DISPLAY_NAME.replace(' ', '_')}.png"
    fig = plot_metric_comparison(
        df_foods,
        "foods",
        "group",
        title="Food Consumption: Unsuccessful vs Successful Wiring Variants",
        y_label="Foods Consumed",
        output_path=output_path_foods,
        doc=doc
    )
    fig_paths.append(("Food Consumption Comparison", output_path_foods))
    plt.show()
    
    unsuccessful_foods = df_foods[df_foods["group"] == "Unsuccessful\n(Bottom 10%)"]["foods"].values
    successful_foods = df_foods[df_foods["group"] == "Successful\n(Top 10%)"]["foods"].values
    
    benchmark_foods_dict = {}
    if df_benchmarks is not None:
        for benchmark_name in sorted(df_benchmarks['benchmark_group'].unique()):
            display_name = benchmark_names_map.get(benchmark_name, benchmark_name.replace("_", " ").title())
            group_label = f"{display_name}\n(Benchmark)"
            group_foods = df_foods[df_foods['group'] == group_label]["foods"].values
            if len(group_foods) > 0:
                benchmark_foods_dict[display_name] = group_foods
    
    stats_foods = calculate_statistics(unsuccessful_foods, successful_foods, benchmark_foods_dict)
    report_statistics(doc, stats_foods, "Food Consumption")
    
    doc.add_paragraph("TBD - Interpretation of food consumption differences.", style="Normal")

#endregion  # closes 3.5.1
#region 3.5.2 Food Per Tick (Normalized) ========================================================================================================================================================================

doc.add_heading("3.5.2. Food Per Tick (Normalized)", level=4)

# Prepare data for food per tick
plot_data = []

for variant_name in unsuccessful_variants:
    df_variant = df_all[df_all["variant"] == variant_name]
    for _, row in df_variant.iterrows():
        # Derive foods_per_tick from raw tracked data (foods and lifetime_ticks)
        if 'foods' in row.index and 'lifetime_ticks' in row.index and row['lifetime_ticks'] > 0:
            val = row['foods'] / row['lifetime_ticks']
            plot_data.append({
                "variant": variant_name,
                "group": "Unsuccessful\n(Bottom 10%)",
                "foods_per_tick": val
            })

for variant_name in successful_variants:
    df_variant = df_all[df_all["variant"] == variant_name]
    for _, row in df_variant.iterrows():
        if 'foods' in row.index and 'lifetime_ticks' in row.index and row['lifetime_ticks'] > 0:
            val = row['foods'] / row['lifetime_ticks']
            plot_data.append({
                "variant": variant_name,
                "group": "Successful\n(Top 10%)",
                "foods_per_tick": val
            })

# Filter benchmarks from df_all (benchmarks were concatenated into df_all earlier)
if 'benchmark_group' in df_all.columns:
    for benchmark_name in sorted(df_all[df_all['benchmark_group'].notna()]['benchmark_group'].unique()):
        df_benchmark = df_all[df_all['benchmark_group'] == benchmark_name]
        display_name = benchmark_names_map.get(benchmark_name, benchmark_name.replace("_", " ").title())
        for _, row in df_benchmark.iterrows():
            if 'foods' in row.index and 'lifetime_ticks' in row.index and row['lifetime_ticks'] > 0:
                val = row['foods'] / row['lifetime_ticks']
                plot_data.append({
                    "variant": row["variant"],
                    "group": f"{display_name}\n(Benchmark)",
                    "foods_per_tick": val
                })

df_foods_per_tick = pd.DataFrame(plot_data)

if not df_foods_per_tick.empty:
    output_path_foods_per_tick = results_dir / f"comp_foods_per_tick_{EXPERIMENT_NAME}_{EXPERIMENT_DISPLAY_NAME.replace(' ', '_')}.png"
    fig = plot_metric_comparison(
        df_foods_per_tick,
        "foods_per_tick",
        "group",
        title="Food Per Tick: Unsuccessful vs Successful Wiring Variants",
        y_label="Foods / Tick",
        output_path=output_path_foods_per_tick,
        doc=doc
    )
    fig_paths.append(("Food Per Tick Comparison", output_path_foods_per_tick))
    plt.show()
    
    unsuccessful_foods_pt = df_foods_per_tick[df_foods_per_tick["group"] == "Unsuccessful\n(Bottom 10%)"]["foods_per_tick"].values
    successful_foods_pt = df_foods_per_tick[df_foods_per_tick["group"] == "Successful\n(Top 10%)"]["foods_per_tick"].values
    
    benchmark_foods_pt_dict = {}
    if df_benchmarks is not None:
        for benchmark_name in sorted(df_benchmarks['benchmark_group'].unique()):
            display_name = benchmark_names_map.get(benchmark_name, benchmark_name.replace("_", " ").title())
            group_label = f"{display_name}\n(Benchmark)"
            group_foods_pt = df_foods_per_tick[df_foods_per_tick['group'] == group_label]["foods_per_tick"].values
            if len(group_foods_pt) > 0:
                benchmark_foods_pt_dict[display_name] = group_foods_pt
    
    stats_foods_pt = calculate_statistics(unsuccessful_foods_pt, successful_foods_pt, benchmark_foods_pt_dict)
    report_statistics(doc, stats_foods_pt, "Food Per Tick")
    
    doc.add_paragraph("TBD - Interpretation of food per tick differences.", style="Normal")

#endregion  # closes 3.5.2
#endregion  # closes 3.5
#endregion  # closes 3

# =====================================================================  E - FINALIZE DOCUMENT AND SAVE ================================================
# ==========================================================================================================================================


print("\nFinalizing document and saving...")

report_path = results_dir / f"report_{EXPERIMENT_NAME}_{EXPERIMENT_DISPLAY_NAME.replace(' ', '_')}.docx"
doc.save(report_path)
print(f"Report saved to: {report_path}")

print("\nAnalysis complete!")
input("Press Enter to close all figures...")
plt.ioff()