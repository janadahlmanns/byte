"""
Plot survival race for all 500 wiring variants on a single figure.
Purpose: Quick visual sanity check on consistency across variants.
Each line = one variant, showing mean survival ± 95% CI across 100 world seeds.
"""

import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from matplotlib.lines import Line2D
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from scipy.stats import mannwhitneyu, kruskal
try:
    import scikit_posthocs as sp
    POSTHOC_AVAILABLE = True
except ImportError:
    POSTHOC_AVAILABLE = False
    print("Warning: scikit_posthocs not installed. Install with: pip install scikit-posthocs")

plt.ion()  # Enable interactive mode

# =====================================================================
# CONFIGURATION
# =====================================================================

# Point to your variant data
BASE_DIR = Path(__file__).resolve().parents[2] / "rawdata"

# Manually specify the experiment folder (or set to None to auto-detect most recent)
EXPERIMENT_FOLDER = "2026-03-08_15-39-28_random_no_regrow_all_tracked"  # Change this to your experiment folder name

EXPERIMENT_DIR = BASE_DIR / EXPERIMENT_FOLDER
if not EXPERIMENT_DIR.exists():
    raise FileNotFoundError(f"Experiment folder not found: {EXPERIMENT_DIR}")

# Extract experiment name from script filename (not EXPERIMENT_FOLDER)
script_name = Path(__file__).stem  # e.g., "analysis_overall_survival" -> "overall_survival"
EXPERIMENT_NAME = script_name.replace("analysis_", "")

# =====================================================================
# BENCHMARK VARIANTS (Optional)
# =====================================================================
# Add one or more benchmark variant folders with names to compare against the top/bottom 10% groups
# Each benchmark folder should contain variant_* subdirectories with summary_*.csv files
# Set to empty list [] to skip benchmark comparison
# Format: [(display_name, folder_path), (display_name2, folder_path2), ...]
# Example: [("Hard-wired Lookup", BASE_DIR / "2026-03-08_hardwired_lookup"), ("Algorithmic", BASE_DIR / "2026-02-15_algo")]

# BENCHMARK_FOLDERS = [
#     ("Hard-wired Lookup", BASE_DIR / "2026-03-08_12-51-56_hardwired_lookup"),
# ]

BENCHMARK_FOLDERS = []


# =====================================================================
# Load all variants
# =====================================================================

variant_dirs = sorted([d for d in EXPERIMENT_DIR.iterdir() if d.is_dir() and d.name.startswith("variant_")])
print(f"Found {len(variant_dirs)} variants")

all_data = []

for variant_dir in variant_dirs:
    variant_name = variant_dir.name  # e.g., "variant_001"
    
    # Find any summary_*.csv file in this variant directory (more flexible)
    summary_files = list(variant_dir.glob("summary_*.csv"))
    
    if summary_files:
        summary_file = summary_files[0]  # Take the first (and should be only) summary file
        df = pd.read_csv(summary_file)
        df["variant"] = variant_name
        all_data.append(df)
    else:
        print(f"Warning: No summary file in {variant_dir}")

if not all_data:
    raise ValueError("No variant summary data found!")

df_all = pd.concat(all_data, ignore_index=True)
print(f"Total rows loaded: {len(df_all)}")
print(f"Unique variants: {df_all['variant'].nunique()}")

# =====================================================================
# Load benchmark variants (optional)
# =====================================================================

benchmark_data = {}
benchmark_names_map = {}  # Map folder name to display name
benchmark_paths_map = {}  # Map folder name to full folder path

if BENCHMARK_FOLDERS:
    for display_name, benchmark_dir in BENCHMARK_FOLDERS:
        if not benchmark_dir.exists():
            print(f"Warning: Benchmark folder not found: {benchmark_dir}")
            continue
        
        folder_name = benchmark_dir.name
        benchmark_names_map[folder_name] = display_name
        benchmark_paths_map[folder_name] = str(benchmark_dir.resolve())
        
        print(f"\nLoading benchmark '{display_name}' from: {benchmark_dir.resolve()}")
        
        benchmark_variants = sorted([d for d in benchmark_dir.iterdir() if d.is_dir() and d.name.startswith("variant_")])
        print(f"  Found {len(benchmark_variants)} variants")
        
        benchmark_data[folder_name] = []
        
        for variant_dir in benchmark_variants:
            variant_name = variant_dir.name
            summary_files = list(variant_dir.glob("summary_*.csv"))
            
            if summary_files:
                summary_file = summary_files[0]
                df = pd.read_csv(summary_file)
                df["variant"] = f"{folder_name}_{variant_name}"
                df["benchmark_group"] = folder_name
                df["benchmark_display_name"] = display_name
                df["benchmark_folder_path"] = str(benchmark_dir.resolve())
                benchmark_data[folder_name].append(df)
            else:
                print(f"    Warning: No summary file in {variant_dir}")
        
        print(f"  Loaded {len(benchmark_data[folder_name])} variants from {folder_name}")
    
    # Combine all benchmark data
    all_benchmark_data = []
    for benchmark_list in benchmark_data.values():
        all_benchmark_data.extend(benchmark_list)
    
    if all_benchmark_data:
        df_benchmarks = pd.concat(all_benchmark_data, ignore_index=True)
        print(f"\nTotal benchmark rows loaded: {len(df_benchmarks)}")
        print(f"Unique benchmark groups: {df_benchmarks['benchmark_group'].nunique()}")
    else:
        df_benchmarks = None
else:
    df_benchmarks = None
    print("\nNo benchmark folders specified (BENCHMARK_FOLDERS is empty)")

# =====================================================================
# =====================================================================
# Compute summary statistics first
# =====================================================================

variant_stats = []

for variant_name in sorted(df_all["variant"].unique()):
    df_variant = df_all[df_all["variant"] == variant_name]
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

# Identify variants in top and bottom 10% of the median survival distribution
median_90_pct_threshold = df_stats["median_survival"].quantile(0.90)
median_10_pct_threshold = df_stats["median_survival"].quantile(0.10)

# Top 10%: variants above 90th percentile of median survival
top_10_pct_variants = set(df_stats[df_stats["median_survival"] >= median_90_pct_threshold]["variant"].values)

# Bottom 10%: variants below 10th percentile of median survival
bottom_10_pct_variants = set(df_stats[df_stats["median_survival"] <= median_10_pct_threshold]["variant"].values)

# =====================================================================
# HELPER FUNCTION: Perform appropriate statistical test
# =====================================================================

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
        if POSTHOC_AVAILABLE and p_value < 0.05:
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

# =====================================================================
# Create survival race plot with color/style coding
# =====================================================================

sns.set_theme(style="whitegrid", context="talk")

# Define colors
green_color = "#0B3D2E"  # Dark green from color map
red_color = "#8B3A3A"    # Wine red from color map

fig = plt.figure(figsize=(14, 8))

# For each variant, compute survival race curve
for variant_name in sorted(df_all["variant"].unique()):
    df_variant = df_all[df_all["variant"] == variant_name]
    survival_times = df_variant["lifetime_ticks"].values
    
    # Compute survival race: at each tick, how many are still alive
    max_t = survival_times.max()
    ticks = np.arange(max_t + 1)
    
    # Count survivors at each tick
    alive_count = np.array([(survival_times >= t).sum() for t in ticks])
    
    # Determine color and alpha based on performance
    if variant_name in top_10_pct_variants:
        color = green_color
        linewidth = 1.5
        alpha = 0.8
    elif variant_name in bottom_10_pct_variants:
        color = red_color
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
if df_benchmarks is not None:
    title_suffix = f" + {len(list(df_benchmarks['benchmark_group'].unique()))} Benchmark(s)"
plt.title(f"Survival Race: All {len(variant_dirs)} Random Variants{title_suffix}\n(100 world seeds per variant)")
plt.grid(True, alpha=0.3)

# Create custom legend
legend_elements = [
    Line2D([0], [0], color=green_color, linewidth=1.5, label=f"Random Top 10% (median ≥ {median_90_pct_threshold:.1f} ticks)"),
    Line2D([0], [0], color=red_color, linewidth=1.5, label=f"Random Bottom 10% (median ≤ {median_10_pct_threshold:.1f} ticks)"),
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

# Save figure to this same folder
results_dir = Path(__file__).resolve().parent
results_dir.mkdir(exist_ok=True)
fig_paths = []  # Track figures for report
fig_path = results_dir / f"survival_race_{EXPERIMENT_NAME}.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {fig_path}")
fig_paths.append(("Survival Race: All Variants", fig_path))

plt.show()

# =====================================================================
# Compile and save statistics
# =====================================================================

# Prepare summary dictionary
summary_dict = {
    "experiment_folder": str(EXPERIMENT_DIR.name),
    "num_variants": len(variant_dirs),
    "num_runs_per_variant": int(df_stats["n_runs"].iloc[0]) if len(df_stats) > 0 else 0,
    "mean_survival": {
        "across_variants_mean": float(df_stats['mean_survival'].mean()),
        "across_variants_std": float(df_stats['mean_survival'].std()),
    },
    "median_survival": {
        "across_variants_mean": float(df_stats['median_survival'].mean()),
        "across_variants_std": float(df_stats['median_survival'].std()),
    },
    "percentile_ranking": {
        "metric": "median_survival",
        "90th_percentile_threshold": float(median_90_pct_threshold),
        "10th_percentile_threshold": float(median_10_pct_threshold),
    },
    "top_10_pct_variants": df_stats[df_stats["median_survival"] >= median_90_pct_threshold].sort_values("median_survival", ascending=False)[["variant", "median_survival", "std_survival"]].to_dict('records'),
    "bottom_10_pct_variants": df_stats[df_stats["median_survival"] <= median_10_pct_threshold].sort_values("median_survival")[["variant", "median_survival", "std_survival"]].to_dict('records'),
}

# Add benchmark information if available
if df_benchmarks is not None:
    benchmark_summary = {}
    for benchmark_name in sorted(df_benchmarks['benchmark_group'].unique()):
        df_bench = df_benchmarks[df_benchmarks["benchmark_group"] == benchmark_name]
        benchmark_times = df_bench["lifetime_ticks"].values
        benchmark_summary[benchmark_name] = {
            "num_runs": int(len(benchmark_times)),
            "mean_survival": float(benchmark_times.mean()),
            "median_survival": float(np.median(benchmark_times)),
            "std_survival": float(benchmark_times.std()),
            "min_survival": float(benchmark_times.min()),
            "max_survival": float(benchmark_times.max()),
        }
    summary_dict["benchmarks"] = benchmark_summary

# Convert float64 values to float for JSON serialization
def convert_to_serializable(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    return obj

summary_dict = convert_to_serializable(summary_dict)

# Save to JSON file alongside the figure
stats_file = results_dir / f"statistical_summary_groups_{EXPERIMENT_NAME}.json"
with open(stats_file, "w") as f:
    json.dump(summary_dict, f, indent=2)
print(f"Saved: {stats_file}")

# =====================================================================
# Print summary statistics
# =====================================================================

print("\n" + "="*60)
print("STATISTICAL SUMMARY ACROSS ALL VARIANTS")
print("="*60)

# =====================================================================
# Print summary statistics
# =====================================================================

print("\n" + "="*60)
print("STATISTICAL SUMMARY ACROSS ALL VARIANTS")
print("="*60)
print(f"\nMean survival time (ticks):")
print(f"  Across variants - Mean: {df_stats['mean_survival'].mean():.1f}")
print(f"  Across variants - Std:  {df_stats['mean_survival'].std():.1f}")

print(f"\nMedian survival time (ticks):")
print(f"  Across variants - Mean: {df_stats['median_survival'].mean():.1f}")
print(f"  Across variants - Std:  {df_stats['median_survival'].std():.1f}")

print(f"\nVariants in TOP 10% (90th percentile) of median survival (>= {median_90_pct_threshold:.1f} ticks):")
print(df_stats[df_stats["median_survival"] >= median_90_pct_threshold].sort_values("median_survival", ascending=False)[["variant", "median_survival", "std_survival"]])

print(f"\nVariants in BOTTOM 10% (10th percentile) of median survival (<= {median_10_pct_threshold:.1f} ticks):")
print(df_stats[df_stats["median_survival"] <= median_10_pct_threshold].sort_values("median_survival")[["variant", "median_survival", "std_survival"]])

print("\nDone with survival race analysis!")

# =====================================================================
# Comparison: Successful vs Unsuccessful variants (with optional benchmarks)
# =====================================================================

print("\n" + "="*60)
print("COMPARISON: SUCCESSFUL VS UNSUCCESSFUL VARIANTS")
if df_benchmarks is not None:
    print(f"(+ {df_benchmarks['benchmark_group'].nunique()} BENCHMARK GROUP(S))")
print("="*60)

# Identify groups
successful_variants = set(df_stats[df_stats["median_survival"] >= median_90_pct_threshold]["variant"].values)
unsuccessful_variants = set(df_stats[df_stats["median_survival"] <= median_10_pct_threshold]["variant"].values)

print(f"\nSuccessful variants (top 10%): {len(successful_variants)}")
print(f"Unsuccessful variants (bottom 10%): {len(unsuccessful_variants)}")

# Prepare data for jitter plot
plot_data = []

# Order: unsuccessful first (left), then successful (right), then benchmarks
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

# Add benchmark data if available
if df_benchmarks is not None:
    for benchmark_name in sorted(df_benchmarks['benchmark_group'].unique()):
        df_benchmark = df_benchmarks[df_benchmarks["benchmark_group"] == benchmark_name]
        # Use the mapped display name
        display_name = benchmark_names_map.get(benchmark_name, benchmark_name.replace("_", " ").title())
        for _, row in df_benchmark.iterrows():
            plot_data.append({
                "variant": row["variant"],
                "group": f"{display_name}\n(Benchmark)",
                "lifetime_ticks": row["lifetime_ticks"]
            })

df_plot = pd.DataFrame(plot_data)

print(f"\nPlot data points: {len(df_plot)}")
print(f"  Unsuccessful: {len(df_plot[df_plot['group'] == 'Unsuccessful\n(Bottom 10%)'])}")
print(f"  Successful: {len(df_plot[df_plot['group'] == 'Successful\n(Top 10%)'])}")
if df_benchmarks is not None:
    for benchmark_name in sorted(df_benchmarks['benchmark_group'].unique()):
        display_name = benchmark_names_map.get(benchmark_name, benchmark_name.replace("_", " ").title())
        count = len(df_plot[df_plot['group'] == f"{display_name}\n(Benchmark)"])
        print(f"  {display_name} (Benchmark): {count}")

# Get data for statistics
unsuccessful_times = df_plot[df_plot["group"] == "Unsuccessful\n(Bottom 10%)"]["lifetime_ticks"].values
successful_times = df_plot[df_plot["group"] == "Successful\n(Top 10%)"]["lifetime_ticks"].values

# Always perform Mann-Whitney U for the two main groups
stat_mw, p_value_mw = mannwhitneyu(unsuccessful_times, successful_times, alternative='two-sided')
test_name_mw = "Mann-Whitney U"
stat_name_mw = "U-statistic"

# Perform Kruskal-Wallis if benchmarks exist
stat, p_value, test_name, stat_name = stat_mw, p_value_mw, test_name_mw, stat_name_mw
posthoc_survival = None
if df_benchmarks is not None:
    # Collect all group data for Kruskal-Wallis
    groups_data = [unsuccessful_times, successful_times]
    group_labels = ["Unsuccessful (Bottom 10%)", "Successful (Top 10%)"]
    
    for benchmark_name in sorted(df_benchmarks['benchmark_group'].unique()):
        display_name = benchmark_names_map.get(benchmark_name, benchmark_name.replace("_", " ").title())
        group_label = f"{display_name} (Benchmark)"
        group_times = df_plot[df_plot['group'] == f"{display_name}\n(Benchmark)"]["lifetime_ticks"].values
        groups_data.append(group_times)
        group_labels.append(group_label)
    
    # Kruskal-Wallis test (non-parametric test for multiple groups)
    stat, p_value = kruskal(*groups_data)
    test_name = "Kruskal-Wallis"
    stat_name = "H-statistic"
    
    # Perform Dunn's post-hoc test if available
    if POSTHOC_AVAILABLE and p_value < 0.05:
        # Create DataFrame for post-hoc analysis
        data_list = []
        for val in unsuccessful_times:
            data_list.append({'metric': val, 'group': 'Unsuccessful'})
        for val in successful_times:
            data_list.append({'metric': val, 'group': 'Successful'})
        for benchmark_name in sorted(df_benchmarks['benchmark_group'].unique()):
            display_name = benchmark_names_map.get(benchmark_name, benchmark_name.replace("_", " ").title())
            group_times = df_plot[df_plot['group'] == f"{display_name}\n(Benchmark)"]["lifetime_ticks"].values
            for val in group_times:
                data_list.append({'metric': val, 'group': display_name})
        
        df_posthoc = pd.DataFrame(data_list)
        posthoc_survival = sp.posthoc_dunn(df_posthoc, val_col='metric', group_col='group', p_adjust='bonferroni')

# Create jitter + box plot
fig, ax = plt.subplots(figsize=(12, 8))

# Define color palette
color_palette = {
    "Unsuccessful\n(Bottom 10%)": red_color,
    "Successful\n(Top 10%)": green_color
}

# Add benchmark colors if present
if df_benchmarks is not None:
    benchmark_colors = ["#2E8B9E", "#9E2E8B", "#8B9E2E", "#2E8B57"]  # Additional colors for benchmarks
    for idx, benchmark_name in enumerate(sorted(df_benchmarks['benchmark_group'].unique())):
        display_name = benchmark_names_map.get(benchmark_name, benchmark_name.replace("_", " ").title())
        color_palette[f"{display_name}\n(Benchmark)"] = benchmark_colors[idx % len(benchmark_colors)]

# Create jitter plot
sns.stripplot(
    data=df_plot,
    x="group",
    y="lifetime_ticks",
    hue="group",
    palette=color_palette,
    size=4,
    alpha=0.7,
    jitter=True,
    ax=ax,
    dodge=False
)

# Add box plot
sns.boxplot(
    data=df_plot,
    x="group",
    y="lifetime_ticks",
    hue="group",
    palette=color_palette,
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
ax.set_ylabel("Lifetime (ticks)", fontsize=12)
if df_benchmarks is None:
    title_text = f"Survival Times: Unsuccessful vs Successful Wiring Variants\n(Mann-Whitney U: U={stat:.4f}, p={p_value:.4e})"
else:
    title_text = f"Survival Times: Random Variants vs Benchmarks\n({test_name}: {stat_name}={stat:.4f}, p={p_value:.4e})"
ax.set_title(title_text)

# Remove legend if it exists
if ax.get_legend() is not None:
    ax.get_legend().remove()

plt.tight_layout()

# Save comparison figure
fig_compare_path = results_dir / f"comp_survival_{EXPERIMENT_NAME}.png"
fig.savefig(fig_compare_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {fig_compare_path}")
fig_paths.append(("Successful vs Unsuccessful Variants Comparison", fig_compare_path))

plt.show()

# Print comparison statistics
print("\n" + "="*60)
print("STATISTICAL COMPARISON: RANDOM VARIANTS")
print("="*60)

# Always print Mann-Whitney U for successful vs unsuccessful
print(f"\nMann-Whitney U Test (Successful vs Unsuccessful Random Variants):")
print(f"  U-statistic: {stat_mw:.4f}")
print(f"  p-value: {p_value_mw:.4e}")

if df_benchmarks is not None:
    # Also print Kruskal-Wallis for all groups
    print(f"\nKruskal-Wallis Test (All Groups: Successful, Unsuccessful + Benchmarks):")
    print(f"  H-statistic: {stat:.4f}")
    print(f"  p-value: {p_value:.4e}")
    
    # Print post-hoc results if available
    if posthoc_survival is not None:
        print(f"\nDunn's Post-hoc Test Results (Survival Time, p_adjust='bonferroni'):")
        print(posthoc_survival.round(4))

print(f"\nSuccessful variants (Median >= {median_90_pct_threshold:.1f} ticks):")
print(f"  N runs: {len(successful_times)}")
print(f"  Mean: {successful_times.mean():.1f} ticks")
print(f"  Median: {np.median(successful_times):.1f} ticks")
print(f"  Std: {successful_times.std():.1f} ticks")
print(f"  Min: {successful_times.min():.1f} ticks")
print(f"  Max: {successful_times.max():.1f} ticks")

print(f"\nUnsuccessful variants (Median <= {median_10_pct_threshold:.1f} ticks):")
print(f"  N runs: {len(unsuccessful_times)}")
print(f"  Mean: {unsuccessful_times.mean():.1f} ticks")
print(f"  Median: {np.median(unsuccessful_times):.1f} ticks")
print(f"  Std: {unsuccessful_times.std():.1f} ticks")
print(f"  Min: {unsuccessful_times.min():.1f} ticks")
print(f"  Max: {unsuccessful_times.max():.1f} ticks")

# Print benchmark statistics if available
if df_benchmarks is not None:
    print("\n" + "="*60)
    print("BENCHMARK VARIANT STATISTICS")
    print("="*60)
    for benchmark_name in sorted(df_benchmarks['benchmark_group'].unique()):
        df_bench = df_benchmarks[df_benchmarks["benchmark_group"] == benchmark_name]
        benchmark_times = df_bench["lifetime_ticks"].values
        display_name = benchmark_names_map.get(benchmark_name, benchmark_name.replace("_", " ").title())
        print(f"\n{display_name}:")
        print(f"  N runs: {len(benchmark_times)}")
        print(f"  Mean: {benchmark_times.mean():.1f} ticks")
        print(f"  Median: {np.median(benchmark_times):.1f} ticks")
        print(f"  Std: {benchmark_times.std():.1f} ticks")
        print(f"  Min: {benchmark_times.min():.1f} ticks")
        print(f"  Max: {benchmark_times.max():.1f} ticks")

# =====================================================================
# Movement Efficiency by Direction (Movement / Sensing)
# =====================================================================

print("\n" + "="*60)
print("MOVEMENT EFFICIENCY ANALYSIS (Movement / Sensing)")
print("="*60)

# Direction mapping
direction_map = {
    'N': 'north',
    'E': 'east',
    'S': 'south',
    'W': 'west'
}
directions = ['N', 'E', 'S', 'W']

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
        'Unsuccessful': red_color,
        'Successful': green_color
    }
    
    if df_benchmarks is not None:
        benchmark_colors_list = ["#2E8B9E", "#9E2E8B", "#8B9E2E", "#2E8B57"]
        for bench_idx, benchmark_name in enumerate(sorted(df_benchmarks['benchmark_group'].unique())):
            display_name = benchmark_names_map.get(benchmark_name, benchmark_name.replace("_", " ").title())
            group_colors[display_name] = benchmark_colors_list[bench_idx % len(benchmark_colors_list)]
    
    # Plot each group
    x_pos = np.arange(len(directions))
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
    fig_efficiency_path = results_dir / f"movement_efficiency_{EXPERIMENT_NAME}.png"
    fig.savefig(fig_efficiency_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {fig_efficiency_path}")
    fig_paths.append(("Movement Efficiency by Direction", fig_efficiency_path))
    
    # Print statistics
    print("\nMovement Efficiency Statistics (Movement / Sensing):\n")
    for group in groups:
        print(f"\n{group} Variants:")
        group_data = summary_stats[summary_stats['group'] == group].sort_values('direction')
        if len(group_data) > 0:
            for _, row in group_data.iterrows():
                print(f"  {row['direction']}: {row['mean']:.3f} ± {row['sem']:.3f}")
    
    plt.show()

print("\n" + "="*60)
print("DECISION ACCURACY ANALYSIS")
print("="*60)

# Extract decision accuracy data for unsuccessful and successful variants
decision_accuracy_data = []

# Process unsuccessful variants
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

# Process successful variants
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

# Add benchmark data if available
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
    print(f"\nDecision Accuracy data points: {len(df_accuracy)}")
    print(f"  Unsuccessful: {len(df_accuracy[df_accuracy['group'] == 'Unsuccessful\n(Bottom 10%)'])}")
    print(f"  Successful: {len(df_accuracy[df_accuracy['group'] == 'Successful\n(Top 10%)'])}")
    if df_benchmarks is not None:
        for benchmark_name in sorted(df_benchmarks['benchmark_group'].unique()):
            display_name = benchmark_names_map.get(benchmark_name, benchmark_name.replace("_", " ").title())
            count = len(df_accuracy[df_accuracy['group'] == f"{display_name}\n(Benchmark)"])
            print(f"  {display_name} (Benchmark): {count}")
    
    # Get data for statistics
    unsuccessful_accuracy = df_accuracy[df_accuracy["group"] == "Unsuccessful\n(Bottom 10%)"]["accuracy"].values
    successful_accuracy = df_accuracy[df_accuracy["group"] == "Successful\n(Top 10%)"]["accuracy"].values
    
    # Prepare benchmark data for statistical test
    benchmark_accuracy_groups = {}
    if df_benchmarks is not None:
        for benchmark_name in sorted(df_benchmarks['benchmark_group'].unique()):
            display_name = benchmark_names_map.get(benchmark_name, benchmark_name.replace("_", " ").title())
            group_label = f"{display_name}\n(Benchmark)"
            group_data = df_accuracy[df_accuracy['group'] == group_label]["accuracy"].values
            if len(group_data) > 0:
                benchmark_accuracy_groups[display_name] = group_data
    
    # Perform statistical test
    stat_acc, p_value_acc, test_name_acc, stat_name_acc, posthoc_acc = perform_statistical_test(
        unsuccessful_accuracy, successful_accuracy, benchmark_accuracy_groups
    )
    
    # Create color palette
    accuracy_palette = {
        "Unsuccessful\n(Bottom 10%)": red_color,
        "Successful\n(Top 10%)": green_color
    }
    
    if df_benchmarks is not None:
        benchmark_colors = ["#2E8B9E", "#9E2E8B", "#8B9E2E", "#2E8B57"]
        for idx, benchmark_name in enumerate(sorted(df_benchmarks['benchmark_group'].unique())):
            display_name = benchmark_names_map.get(benchmark_name, benchmark_name.replace("_", " ").title())
            accuracy_palette[f"{display_name}\n(Benchmark)"] = benchmark_colors[idx % len(benchmark_colors)]
    
    # Create jitter + box plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create jitter plot
    sns.stripplot(
        data=df_accuracy,
        x="group",
        y="accuracy",
        hue="group",
        palette=accuracy_palette,
        size=4,
        alpha=0.7,
        jitter=True,
        ax=ax,
        dodge=False
    )
    
    # Add box plot
    sns.boxplot(
        data=df_accuracy,
        x="group",
        y="accuracy",
        hue="group",
        palette=accuracy_palette,
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
    ax.set_ylabel("Decision Accuracy (Correct / Total)", fontsize=12)
    title_text = f"Decision Accuracy: Variants Comparison\n({test_name_acc}: {stat_name_acc}={stat_acc:.4f}, p={p_value_acc:.4e})"
    ax.set_title(title_text)
    
    # Remove legend if it exists
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    
    plt.tight_layout()
    
    # Save comparison figure
    fig_accuracy_path = results_dir / f"decision_accuracy_{EXPERIMENT_NAME}.png"
    fig.savefig(fig_accuracy_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {fig_accuracy_path}")
    fig_paths.append(("Decision Accuracy Comparison", fig_accuracy_path))
    
    plt.show()
    
    # Print comparison statistics
    print(f"\n{test_name_acc} Test Results (Decision Accuracy):")
    print(f"  {stat_name_acc}: {stat_acc:.4f}")
    print(f"  p-value: {p_value_acc:.4e}")
    
    # Print post-hoc results if available
    if posthoc_acc is not None:
        print(f"\nDunn's Post-hoc Test Results (Decision Accuracy, p_adjust='bonferroni'):")
        print(posthoc_acc.round(4))
    
    print(f"\nSuccessful variants (Decision Accuracy):")
    print(f"  N runs: {len(successful_accuracy)}")
    print(f"  Mean: {successful_accuracy.mean():.3f}")
    print(f"  Median: {np.median(successful_accuracy):.3f}")
    print(f"  Std: {successful_accuracy.std():.3f}")
    print(f"  Min: {successful_accuracy.min():.3f}")
    print(f"  Max: {successful_accuracy.max():.3f}")
    
    print(f"\nUnsuccessful variants (Decision Accuracy):")
    print(f"  N runs: {len(unsuccessful_accuracy)}")
    print(f"  Mean: {unsuccessful_accuracy.mean():.3f}")
    print(f"  Median: {np.median(unsuccessful_accuracy):.3f}")
    print(f"  Std: {unsuccessful_accuracy.std():.3f}")
    print(f"  Min: {unsuccessful_accuracy.min():.3f}")
    print(f"  Max: {unsuccessful_accuracy.max():.3f}")
else:
    print("No decision accuracy data found in summary files")

# =====================================================================
# DIAGNOSTIC: Decision Count vs Survival & Decision Patterns
# =====================================================================

print("\n" + "="*60)
print("DECISION DIAGNOSTIC ANALYSIS")
print("="*60)

# Calculate decisions per tick for each run
diagnostic_data = []
for variant_name in df_all["variant"].unique():
    df_variant = df_all[df_all["variant"] == variant_name]
    for _, row in df_variant.iterrows():
        lifetime = row["lifetime_ticks"]
        total_decisions = row.get("decisions", 0)
        decisions_per_tick = total_decisions / lifetime if lifetime > 0 else 0
        
        variant_group = None
        if variant_name in successful_variants:
            variant_group = "Successful"
        elif variant_name in unsuccessful_variants:
            variant_group = "Unsuccessful"
        else:
            variant_group = "Middle"
        
        diagnostic_data.append({
            "group": variant_group,
            "lifetime": lifetime,
            "total_decisions": total_decisions,
            "decisions_per_tick": decisions_per_tick
        })

df_diagnostic = pd.DataFrame(diagnostic_data)

# Determine number of rows needed for the combined figure
num_benchmarks = len(df_benchmarks['benchmark_group'].unique()) if df_benchmarks is not None else 0
num_rows = 1 + num_benchmarks  # 1 for random variants, 1 per benchmark
fig, axes = plt.subplots(num_rows, 3, figsize=(18, 5 * num_rows))

# Handle single row case (axes is 1D) vs multiple rows (axes is 2D)
if num_rows == 1:
    axes = axes.reshape(1, -1)

fig.suptitle("Decision Diagnostic Analysis", fontsize=16, fontweight='bold', y=0.995)

# ====== ROW 0: Random Variants ======
row_axes = axes[0]
row_axes[0].text(0.5, 1.08, "Random Variants (Unsuccessful, Middle, Successful)", 
                 transform=row_axes[0].transAxes, ha='center', fontsize=12, fontweight='bold')

# Plot 1: Total decisions vs Lifetime
for group, color in [("Middle", "gray"), ("Successful", green_color), ("Unsuccessful", red_color)]:
    data = df_diagnostic[df_diagnostic["group"] == group]
    row_axes[0].scatter(data["total_decisions"], data["lifetime"], alpha=0.4, s=20, color=color, label=group)

row_axes[0].set_xlabel("Total Decisions Made", fontsize=11)
row_axes[0].set_ylabel("Lifetime (ticks)", fontsize=11)
row_axes[0].set_title("Decision Count vs Survival Time", fontsize=12, fontweight='bold')
row_axes[0].legend(fontsize=9)
row_axes[0].grid(alpha=0.3)

# Plot 2: Decisions per tick vs Lifetime
for group, color in [("Middle", "gray"), ("Successful", green_color), ("Unsuccessful", red_color)]:
    data = df_diagnostic[df_diagnostic["group"] == group]
    row_axes[1].scatter(data["decisions_per_tick"], data["lifetime"], alpha=0.4, s=20, color=color, label=group)

row_axes[1].set_xlabel("Decisions per Tick", fontsize=11)
row_axes[1].set_ylabel("Lifetime (ticks)", fontsize=11)
row_axes[1].set_title("Decision Density vs Survival Time", fontsize=12, fontweight='bold')
row_axes[1].legend(fontsize=9)
row_axes[1].grid(alpha=0.3)

# Plot 3: Box plot of decisions per tick by group
diagnostic_data_for_box = []
for group in ["Unsuccessful", "Successful"]:
    data = df_diagnostic[df_diagnostic["group"] == group]
    for val in data["decisions_per_tick"].values:
        diagnostic_data_for_box.append({"group": group, "decisions_per_tick": val})

df_diagnostic_box = pd.DataFrame(diagnostic_data_for_box)

sns.stripplot(
    data=df_diagnostic_box,
    x="group",
    y="decisions_per_tick",
    hue="group",
    palette={"Unsuccessful": red_color, "Successful": green_color},
    size=4,
    alpha=0.7,
    jitter=True,
    ax=row_axes[2],
    dodge=False
)

sns.boxplot(
    data=df_diagnostic_box,
    x="group",
    y="decisions_per_tick",
    hue="group",
    palette={"Unsuccessful": red_color, "Successful": green_color},
    width=0.3,
    ax=row_axes[2],
    showcaps=True,
    whiskerprops={'linewidth': 1.5},
    boxprops={'linewidth': 1.5},
    medianprops={'color': 'black', 'linewidth': 1.5},
    fliersize=0,
    legend=False
)

for patch in row_axes[2].patches:
    patch.set_alpha(0.3)

row_axes[2].set_xlabel("Variant Group", fontsize=11)
row_axes[2].set_ylabel("Decisions per Tick", fontsize=11)
row_axes[2].set_title("Decision Density Distribution", fontsize=12, fontweight='bold')
if row_axes[2].get_legend() is not None:
    row_axes[2].get_legend().remove()

# ====== Additional ROWS: Benchmark Variants ======
if df_benchmarks is not None:
    for bench_idx, benchmark_name in enumerate(sorted(df_benchmarks['benchmark_group'].unique()), 1):
        display_name = benchmark_names_map.get(benchmark_name, benchmark_name.replace("_", " ").title())
        row_axes = axes[bench_idx]
        
        # Add benchmark name as subtitle for this row
        row_axes[0].text(0.5, 1.08, f"Benchmark: {display_name}", 
                        transform=row_axes[0].transAxes, ha='center', fontsize=12, fontweight='bold')
        
        # Calculate decisions for this benchmark
        benchmark_diagnostic_data = []
        for _, row in df_benchmarks[df_benchmarks['benchmark_group'] == benchmark_name].iterrows():
            lifetime = row["lifetime_ticks"]
            total_decisions = row.get("decisions", 0)
            decisions_per_tick = total_decisions / lifetime if lifetime > 0 else 0
            
            benchmark_diagnostic_data.append({
                "group": display_name,
                "lifetime": lifetime,
                "total_decisions": total_decisions,
                "decisions_per_tick": decisions_per_tick
            })
        
        if benchmark_diagnostic_data:
            df_benchmark_diag = pd.DataFrame(benchmark_diagnostic_data)
            
            # Use benchmark color
            bench_color = benchmark_colors_list[
                list(sorted(df_benchmarks['benchmark_group'].unique())).index(benchmark_name) % len(benchmark_colors_list)
            ]
            
            # Plot 1: Total decisions vs Lifetime
            row_axes[0].scatter(df_benchmark_diag["total_decisions"], df_benchmark_diag["lifetime"], 
                               alpha=0.6, s=30, color=bench_color, edgecolors='black', linewidth=1)
            row_axes[0].set_xlabel("Total Decisions Made", fontsize=11)
            row_axes[0].set_ylabel("Lifetime (ticks)", fontsize=11)
            row_axes[0].set_title("Decision Count vs Survival Time", fontsize=12, fontweight='bold')
            row_axes[0].grid(alpha=0.3)
            
            # Plot 2: Decisions per tick vs Lifetime
            row_axes[1].scatter(df_benchmark_diag["decisions_per_tick"], df_benchmark_diag["lifetime"],
                               alpha=0.6, s=30, color=bench_color, edgecolors='black', linewidth=1)
            row_axes[1].set_xlabel("Decisions per Tick", fontsize=11)
            row_axes[1].set_ylabel("Lifetime (ticks)", fontsize=11)
            row_axes[1].set_title("Decision Density vs Survival Time", fontsize=12, fontweight='bold')
            row_axes[1].grid(alpha=0.3)
            
            # Plot 3: Box plot of decisions per tick (CHANGED from histogram to boxplot)
            diagnostic_data_for_bench_box = []
            for val in df_benchmark_diag["decisions_per_tick"].values:
                diagnostic_data_for_bench_box.append({"group": display_name, "decisions_per_tick": val})
            
            df_bench_box = pd.DataFrame(diagnostic_data_for_bench_box)
            
            sns.stripplot(
                data=df_bench_box,
                x="group",
                y="decisions_per_tick",
                color=bench_color,
                size=6,
                alpha=0.6,
                jitter=True,
                ax=row_axes[2]
            )
            
            sns.boxplot(
                data=df_bench_box,
                x="group",
                y="decisions_per_tick",
                color=bench_color,
                width=0.3,
                ax=row_axes[2],
                showcaps=True,
                whiskerprops={'linewidth': 1.5},
                boxprops={'linewidth': 1.5},
                medianprops={'color': 'black', 'linewidth': 1.5},
                fliersize=0
            )
            
            for patch in row_axes[2].patches:
                patch.set_alpha(0.4)
            
            row_axes[2].set_xlabel("", fontsize=11)
            row_axes[2].set_ylabel("Decisions per Tick", fontsize=11)
            row_axes[2].set_title("Decision Density Distribution", fontsize=12, fontweight='bold')
            row_axes[2].set_xticklabels([])

plt.tight_layout()

fig_diagnostic_path = results_dir / f"decision_diagnostic_{EXPERIMENT_NAME}.png"
fig.savefig(fig_diagnostic_path, dpi=150, bbox_inches="tight")
print(f"\nSaved diagnostic plots: {fig_diagnostic_path}")
fig_paths.append(("Decision Diagnostic Analysis", fig_diagnostic_path))

# Print diagnostic summary
print("\nDecision Statistics Summary:")
print("\nUnsuccessful variants:")
unsuccessful_diag = df_diagnostic[df_diagnostic["group"] == "Unsuccessful"]
print(f"  Mean decisions/tick: {unsuccessful_diag['decisions_per_tick'].mean():.3f}")
print(f"  Median decisions/tick: {unsuccessful_diag['decisions_per_tick'].median():.3f}")
print(f"  Mean total decisions: {unsuccessful_diag['total_decisions'].mean():.1f}")

print("\nSuccessful variants:")
successful_diag = df_diagnostic[df_diagnostic["group"] == "Successful"]
print(f"  Mean decisions/tick: {successful_diag['decisions_per_tick'].mean():.3f}")
print(f"  Median decisions/tick: {successful_diag['decisions_per_tick'].median():.3f}")
print(f"  Mean total decisions: {successful_diag['total_decisions'].mean():.1f}")

print("\n** INTERPRETATION **")
print("Unsuccessful variants have MORE decisions per tick because they encounter")
print("more situations where they sense food but aren't on it. They're making")
print("more food-directed corrections, but this strategy doesn't lead to long survival.")
print("\nSuccessful variants make FEWER decisions per tick, suggesting they use a")
print("different movement strategy that's less reliant on constant food-guided corrections.")

plt.show()

print("\n" + "="*60)
print("MOVEMENT & FEEDING ANALYSIS")
print("="*60)

# Extract movement and feeding data
movement_feeding_data = []

for variant_name in unsuccessful_variants:
    df_variant = df_all[df_all["variant"] == variant_name]
    for _, row in df_variant.iterrows():
        lifetime = row.get("lifetime_ticks", 0)
        distance = row.get("distance", 0)
        foods = row.get("foods", 0)
        distance_per_tick = distance / lifetime if lifetime > 0 else 0
        foods_per_tick = foods / lifetime if lifetime > 0 else 0
        
        movement_feeding_data.append({
            "group": "Unsuccessful\n(Bottom 10%)",
            "distance": distance,
            "distance_per_tick": distance_per_tick,
            "foods": foods,
            "foods_per_tick": foods_per_tick
        })

for variant_name in successful_variants:
    df_variant = df_all[df_all["variant"] == variant_name]
    for _, row in df_variant.iterrows():
        lifetime = row.get("lifetime_ticks", 0)
        distance = row.get("distance", 0)
        foods = row.get("foods", 0)
        distance_per_tick = distance / lifetime if lifetime > 0 else 0
        foods_per_tick = foods / lifetime if lifetime > 0 else 0
        
        movement_feeding_data.append({
            "group": "Successful\n(Top 10%)",
            "distance": distance,
            "distance_per_tick": distance_per_tick,
            "foods": foods,
            "foods_per_tick": foods_per_tick
        })

# Add benchmark data if available
if df_benchmarks is not None:
    for benchmark_name in sorted(df_benchmarks['benchmark_group'].unique()):
        df_benchmark = df_benchmarks[df_benchmarks["benchmark_group"] == benchmark_name]
        display_name = benchmark_names_map.get(benchmark_name, benchmark_name.replace("_", " ").title())
        for _, row in df_benchmark.iterrows():
            lifetime = row.get("lifetime_ticks", 0)
            distance = row.get("distance", 0)
            foods = row.get("foods", 0)
            distance_per_tick = distance / lifetime if lifetime > 0 else 0
            foods_per_tick = foods / lifetime if lifetime > 0 else 0
            
            movement_feeding_data.append({
                "group": f"{display_name}\n(Benchmark)",
                "distance": distance,
                "distance_per_tick": distance_per_tick,
                "foods": foods,
                "foods_per_tick": foods_per_tick
            })

df_movement_feeding = pd.DataFrame(movement_feeding_data)

# Create 4 figures: distance, distance/tick, foods, foods/tick
metrics = [
    ("distance", "Total Distance Travelled", "Distance"),
    ("distance_per_tick", "Distance Efficiency (Per Tick)", "Distance/Tick"),
    ("foods", "Total Food Eaten", "Food Count"),
    ("foods_per_tick", "Food Consumption Rate (Per Tick)", "Foods/Tick")
]

metric_stats = {}

for metric_col, metric_title, y_label in metrics:
    print(f"\n{metric_title} Analysis:")
    
    unsuccessful_vals = df_movement_feeding[df_movement_feeding["group"] == "Unsuccessful\n(Bottom 10%)"][metric_col].values
    successful_vals = df_movement_feeding[df_movement_feeding["group"] == "Successful\n(Top 10%)"][metric_col].values
    
    # Prepare benchmark data for statistical test
    benchmark_vals_dict = {}
    if df_benchmarks is not None:
        for benchmark_name in sorted(df_benchmarks['benchmark_group'].unique()):
            display_name = benchmark_names_map.get(benchmark_name, benchmark_name.replace("_", " ").title())
            group_label = f"{display_name}\n(Benchmark)"
            group_vals = df_movement_feeding[df_movement_feeding['group'] == group_label][metric_col].values
            if len(group_vals) > 0:
                benchmark_vals_dict[display_name] = group_vals
    
    # Perform statistical test
    stat_metric, p_value_metric, test_name_m, stat_name_m, posthoc_metric = perform_statistical_test(
        unsuccessful_vals, successful_vals, benchmark_vals_dict
    )
    
    # Store stats for document
    metric_stats[metric_col] = {
        "title": metric_title,
        "y_label": y_label,
        "unsuccessful_mean": unsuccessful_vals.mean(),
        "unsuccessful_median": np.median(unsuccessful_vals),
        "unsuccessful_std": unsuccessful_vals.std(),
        "unsuccessful_min": unsuccessful_vals.min(),
        "unsuccessful_max": unsuccessful_vals.max(),
        "unsuccessful_n": len(unsuccessful_vals),
        "successful_mean": successful_vals.mean(),
        "successful_median": np.median(successful_vals),
        "successful_std": successful_vals.std(),
        "successful_min": successful_vals.min(),
        "successful_max": successful_vals.max(),
        "successful_n": len(successful_vals),
        "stat": stat_metric,
        "p_value": p_value_metric,
        "test_name": test_name_m,
        "stat_name": stat_name_m,
        "posthoc": posthoc_metric
    }
    
    print(f"  Unsuccessful - Mean: {unsuccessful_vals.mean():.3f}, Median: {np.median(unsuccessful_vals):.3f}")
    print(f"  Successful - Mean: {successful_vals.mean():.3f}, Median: {np.median(successful_vals):.3f}")
    print(f"  {test_name_m}: {stat_metric:.1f}, p-value: {p_value_metric:.4e}")
    
    # Print post-hoc results if available
    if posthoc_metric is not None:
        print(f"  Dunn's Post-hoc (bonferroni-adjusted p-values):")
        print(posthoc_metric.round(4))
    
    # Create color palette
    metric_palette = {
        "Unsuccessful\n(Bottom 10%)": red_color,
        "Successful\n(Top 10%)": green_color
    }
    
    if df_benchmarks is not None:
        benchmark_colors = ["#2E8B9E", "#9E2E8B", "#8B9E2E", "#2E8B57"]
        for idx, benchmark_name in enumerate(sorted(df_benchmarks['benchmark_group'].unique())):
            display_name = benchmark_names_map.get(benchmark_name, benchmark_name.replace("_", " ").title())
            metric_palette[f"{display_name}\n(Benchmark)"] = benchmark_colors[idx % len(benchmark_colors)]
    
    # Create jitter + box plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create jitter plot
    sns.stripplot(
        data=df_movement_feeding,
        x="group",
        y=metric_col,
        hue="group",
        palette=metric_palette,
        size=4,
        alpha=0.7,
        jitter=True,
        ax=ax,
        dodge=False
    )
    
    # Add box plot
    sns.boxplot(
        data=df_movement_feeding,
        x="group",
        y=metric_col,
        hue="group",
        palette=metric_palette,
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
    title_text = f"{metric_title}: Variants Comparison\n({test_name_m}: {stat_name_m}={stat_metric:.1f}, p={p_value_metric:.4e})"
    ax.set_title(title_text, fontsize=12, fontweight='bold')
    
    # Remove legend if it exists
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    
    plt.tight_layout()
    
    # Save figure
    fig_metric_path = results_dir / f"{metric_col}_{EXPERIMENT_NAME}.png"
    fig.savefig(fig_metric_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {fig_metric_path}")
    fig_paths.append((metric_title, fig_metric_path))
    
    plt.show()

print("\n" + "="*60)
print("="*60)

# =====================================================================
# Generate Report Document - Reorganized Structure
# =====================================================================

doc = Document()

# ==================================================================
# 1. EXPERIMENT INFORMATION
# ==================================================================
doc.add_heading("Analysis Report: Random Wiring Variants", level=0)
doc.add_heading(f"{EXPERIMENT_NAME}", level=1)

doc.add_heading("1. Experiment Information", level=2)

doc.add_paragraph("Random Wiring Variants:", style="Heading 3")
doc.add_paragraph(f"Experiment folder: {EXPERIMENT_DIR.name}")
doc.add_paragraph(f"Total random variants analyzed: {len(variant_dirs)}")
doc.add_paragraph(f"Runs per variant: 100")
doc.add_paragraph(f"Total data points (random): {len(df_all)}")

if df_benchmarks is not None and len(df_benchmarks) > 0:
    doc.add_paragraph("Benchmark Variants:", style="Heading 3")
    for benchmark_name in sorted(df_benchmarks['benchmark_group'].unique()):
        display_name = benchmark_names_map.get(benchmark_name, benchmark_name.replace("_", " ").title())
        df_bench = df_benchmarks[df_benchmarks['benchmark_group'] == benchmark_name]
        num_runs = len(df_bench)
        # Infer number of variants from the data (assuming runs are equally distributed)
        num_bench_variants = len(df_bench['variant'].unique())
        # Get folder path from dataframe or mapping
        if 'benchmark_folder_path' in df_bench.columns:
            folder_path = df_bench['benchmark_folder_path'].iloc[0]
        else:
            folder_path = benchmark_paths_map.get(benchmark_name, "(path not available)")
        doc.add_paragraph(f"{display_name}:")
        doc.add_paragraph(f"  Folder: {folder_path}", style="List Bullet")
        doc.add_paragraph(f"  Number of variants: {num_bench_variants}", style="List Bullet")
        doc.add_paragraph(f"  Runs per variant: {num_runs // num_bench_variants if num_bench_variants > 0 else num_runs}", style="List Bullet")
        doc.add_paragraph(f"  Total data points: {num_runs}", style="List Bullet")

# ==================================================================
# 1.1. SUMMARY DESCRIPTIVE STATISTICS - RANDOM VARIANTS
# ==================================================================
doc.add_heading("1.1. Summary Statistics - Random Wiring Variants", level=2)

doc.add_paragraph(
    "Summary of how long random wiring variants survive in the experimental environment. "
    "We measure survival time in simulation ticks and compare the mean and median lifetimes across all 1000 variants to identify which wiring patterns lead to longer survival. "
    "The percentile analysis identifies the successful variants with median survival in the top 10% and the least successful variants with median survival in the bottom 10%, "
    "which are used for detailed comparisons in subsequent sections."
)

doc.add_paragraph("Overall Descriptive Statistics (Across All Variants):", style="Heading 3")
table = doc.add_table(rows=5, cols=2)
table.style = "Light Grid Accent 1"
cells = table.rows[0].cells
cells[0].text = "Metric"
cells[1].text = "Value"
cells = table.rows[1].cells
cells[0].text = "Mean survival across variants (ticks)"
cells[1].text = f"{df_stats['mean_survival'].mean():.1f}"
cells = table.rows[2].cells
cells[0].text = "Std Dev across variants"
cells[1].text = f"{df_stats['mean_survival'].std():.1f}"
cells = table.rows[3].cells
cells[0].text = "Min variant mean"
cells[1].text = f"{df_stats['mean_survival'].min():.1f}"
cells = table.rows[4].cells
cells[0].text = "Max variant mean"
cells[1].text = f"{df_stats['mean_survival'].max():.1f}"

doc.add_paragraph("Percentile Analysis:", style="Heading 3")
table = doc.add_table(rows=4, cols=2)
table.style = "Light Grid Accent 1"
cells = table.rows[0].cells
cells[0].text = "Metric"
cells[1].text = "Value (ticks)"
cells = table.rows[1].cells
cells[0].text = "90th percentile threshold (Successful)"
cells[1].text = f"{median_90_pct_threshold:.1f}"
cells = table.rows[2].cells
cells[0].text = "10th percentile threshold (Unsuccessful)"
cells[1].text = f"{median_10_pct_threshold:.1f}"
cells = table.rows[3].cells
cells[0].text = "Variants identified in each group"
cells[1].text = f"Successful: {len(successful_variants)}, Unsuccessful: {len(unsuccessful_variants)}"

# ==================================================================
# 1.2. SUMMARY DESCRIPTIVE STATISTICS - BENCHMARK VARIANTS
# ==================================================================
if df_benchmarks is not None:
    doc.add_heading("1.2. Summary Statistics - Benchmark Variants", level=2)
    
    doc.add_paragraph(
        "Benchmark variants are hard-wired control brains that follow predetermined algorithms rather than evolved random wiring. "
        "We compare benchmark survival times to our random wiring variants to evaluate how well random circuits can match or exceed hand-designed solutions. "
        "This provides a reference point for assessing the task solution potential of random wiring approaches."
    )
    
    table = doc.add_table(rows=len(df_benchmarks['benchmark_group'].unique())+1, cols=6)
    table.style = "Light Grid Accent 1"
    cells = table.rows[0].cells
    cells[0].text = "Benchmark"
    cells[1].text = "N runs"
    cells[2].text = "Mean (ticks)"
    cells[3].text = "Median (ticks)"
    cells[4].text = "Std Dev"
    cells[5].text = "Min / Max"
    
    for row_idx, benchmark_name in enumerate(sorted(df_benchmarks['benchmark_group'].unique()), 1):
        df_bench = df_benchmarks[df_benchmarks["benchmark_group"] == benchmark_name]
        benchmark_times = df_bench["lifetime_ticks"].values
        display_name = benchmark_names_map.get(benchmark_name, benchmark_name.replace("_", " ").title())
        
        cells = table.rows[row_idx].cells
        cells[0].text = display_name
        cells[1].text = f"{len(benchmark_times)}"
        cells[2].text = f"{benchmark_times.mean():.1f}"
        cells[3].text = f"{np.median(benchmark_times):.1f}"
        cells[4].text = f"{benchmark_times.std():.1f}"
        cells[5].text = f"{benchmark_times.min():.1f} / {benchmark_times.max():.1f}"
else:
    doc.add_heading("1.2. Summary Statistics - Benchmark Variants", level=2)
    doc.add_paragraph("No benchmark variants provided for this analysis.")

# ==================================================================
# 2. SURVIVAL RACE PLOT & GROUP IDENTIFICATION
# ==================================================================
doc.add_heading("2. Survival Race Selection", level=2)

doc.add_paragraph(
    "The survival race plot tracks how many individuals from each wiring variant survive over time. "
    "Each curve represents a variant's cumulative survival, showing how fitness changes across the simulation. "
    "We use this visualization to identify which variants are robustly long-lived. The comparison table shows key statistics "
    "for successful versus unsuccessful variants, helping quantify the performance gap."
)

doc.add_paragraph(
    f"This plot shows the survival curves for all {len(variant_dirs)} wiring variants. "
    f"Variants are color-coded by their performance percentile:"
)
doc.add_paragraph(f"  • Green: Top 10% (median survival ≥ {median_90_pct_threshold:.1f} ticks) — {len(successful_variants)} variants")
doc.add_paragraph(f"  • Red: Bottom 10% (median survival ≤ {median_10_pct_threshold:.1f} ticks) — {len(unsuccessful_variants)} variants")
doc.add_paragraph(f"  • Gray: Middle 80% (other variants)")
if df_benchmarks is not None:
    doc.add_paragraph("  • Colored lines: Benchmark variants")

fig_path = results_dir / f"survival_race_{EXPERIMENT_NAME}.png"
if fig_path.exists():
    doc.add_picture(str(fig_path), width=Inches(6))
doc.add_paragraph()

doc.add_paragraph("Successful vs Unsuccessful Comparison:")
table = doc.add_table(rows=7, cols=3)
table.style = "Light Grid Accent 1"
cells = table.rows[0].cells
cells[0].text = "Metric"
cells[1].text = "Successful (Top 10%)"
cells[2].text = "Unsuccessful (Bottom 10%)"
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

fig_path = results_dir / f"comp_survival_{EXPERIMENT_NAME}.png"
if fig_path.exists():
    doc.add_picture(str(fig_path), width=Inches(6))
doc.add_paragraph()

# ==================================================================
# MANN-WHITNEY U TEST - RANDOM VARIANTS ONLY (No numbered header)
# ==================================================================
doc.add_paragraph("Mann-Whitney U Test Results (Random Variants Only):", style="Heading 3")
doc.add_paragraph("This statistical test compares whether successful and unsuccessful random variants have significantly different survival distributions. ")
table = doc.add_table(rows=3, cols=2)
table.style = "Light Grid Accent 1"
cells = table.rows[0].cells
cells[0].text = "Test Statistic"
cells[1].text = "Value"
cells = table.rows[1].cells
cells[0].text = "U-statistic"
cells[1].text = f"{stat_mw:.4f}"
cells = table.rows[2].cells
cells[0].text = "p-value"
cells[1].text = f"{p_value_mw:.4e}"
doc.add_paragraph()

# ==================================================================
# KRUSKAL-WALLIS TEST - ALL GROUPS (No numbered header)
# ==================================================================
if df_benchmarks is not None:
    doc.add_paragraph()
    doc.add_paragraph("Kruskal-Wallis Test Results (All Groups):", style="Heading 3")
    doc.add_paragraph(
        "Kruskal-Wallis test compares all three groups simultaneously: successful variants, unsuccessful variants, and benchmark variants. "
        "It tests whether significant differences exist across all groups."
    )
    table = doc.add_table(rows=3, cols=2)
    table.style = "Light Grid Accent 1"
    cells = table.rows[0].cells
    cells[0].text = "Test Statistic"
    cells[1].text = "Value"
    cells = table.rows[1].cells
    cells[0].text = "H-statistic"
    cells[1].text = f"{stat:.4f}"
    cells = table.rows[2].cells
    cells[0].text = "p-value"
    cells[1].text = f"{p_value:.4e}"
    
    doc.add_paragraph()
    
    # Add post-hoc results if available
    if posthoc_survival is not None:
        doc.add_paragraph("Dunn's Post-hoc Test Results (pairwise comparisons, Bonferroni-adjusted):")
        # Create table from post-hoc DataFrame
        posthoc_table = doc.add_table(rows=len(posthoc_survival) + 1, cols=len(posthoc_survival.columns) + 1)
        posthoc_table.style = "Light Grid Accent 1"
        
        # Header row
        header_cells = posthoc_table.rows[0].cells
        header_cells[0].text = "Comparison"
        for col_idx, col_name in enumerate(posthoc_survival.columns):
            header_cells[col_idx + 1].text = str(col_name)
        
        # Data rows
        for row_idx, (idx_name, row_data) in enumerate(posthoc_survival.iterrows(), 1):
            data_cells = posthoc_table.rows[row_idx].cells
            data_cells[0].text = str(idx_name)
            for col_idx, val in enumerate(row_data):
                data_cells[col_idx + 1].text = f"{float(val):.4f}"

# ==================================================================
# 3. COMPARISON BETWEEN GROUPS
# ==================================================================
doc.add_heading("3. Comparison between groups", level=2)

doc.add_paragraph("Break down of successful versus unsuccessful variants across multiple metrics. ")

# Section 3.1: Movement Efficiency
doc.add_heading("3.1. Movement Efficiency", level=2)
doc.add_paragraph(
    "For each cardinal direction (North, East, South, West), we calculate the ratio of movement commands to sensory detections. "
    "A ratio close to 1.0 indicates tight coupling between sensing and movement (reflexory behavior: the worm moves in response to each sensory detection). "
    "Ratios far from 1.0 (either much higher or much lower) indicate decoupling: the worm either moves more without sensing, or senses food but moves selectively. "
    "This metric reveals fundamental differences in movement strategy between successful and unsuccessful variants—whether they rely on reflexory sensorimotor coupling or employ a different navigation strategy."
)
doc.add_paragraph("Movement Efficiency = Movement Count ÷ Sensing Count per direction")

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
    unsucc_row = unsuccessful_eff[unsuccessful_eff['direction'] == direction].iloc[0]
    succ_row = successful_eff[successful_eff['direction'] == direction].iloc[0]
    cells[1].text = f"{unsucc_row['mean']:.3f} ± {unsucc_row['std']:.3f}"
    cells[2].text = f"{succ_row['mean']:.3f} ± {succ_row['std']:.3f}"

fig_path = results_dir / f"movement_efficiency_{EXPERIMENT_NAME}.png"
if fig_path.exists():
    doc.add_picture(str(fig_path), width=Inches(6))
doc.add_paragraph()

# Section 3.2: Decision Accuracy
if not df_accuracy.empty:
    doc.add_heading("3.2. Decision Accuracy", level=2)
    doc.add_paragraph(
        "Decision accuracy quantifies how often choices are aligned with food source locations. "
        "This measures the quality of neural computation, whether the wiring interprets sensory input to direct movement toward food. "
        "Higher accuracy indicates more reliable neural circuits. We compare the distribution of accuracy values between successful and unsuccessful "
        "variants to determine whether fitness correlates with decision-making precision, or if other factors like movement efficiency play a larger role. Note that moving onto sensed food is "
        "a very shortsighted strategy that might not lead to an optimal solution in more complex tasks e.g. with food regrow in previous positions."
    )
    doc.add_paragraph("Decision Accuracy = Correct Decisions ÷ Total Decisions")
    
    table = doc.add_table(rows=7, cols=3)
    table.style = "Light Grid Accent 1"
    cells = table.rows[0].cells
    cells[0].text = "Metric"
    cells[1].text = "Successful (Top 10%)"
    cells[2].text = "Unsuccessful (Bottom 10%)"
    cells = table.rows[1].cells
    cells[0].text = "Number of runs"
    cells[1].text = f"{len(successful_accuracy)}"
    cells[2].text = f"{len(unsuccessful_accuracy)}"
    cells = table.rows[2].cells
    cells[0].text = "Mean accuracy"
    cells[1].text = f"{successful_accuracy.mean():.3f}"
    cells[2].text = f"{unsuccessful_accuracy.mean():.3f}"
    cells = table.rows[3].cells
    cells[0].text = "Median accuracy"
    cells[1].text = f"{np.median(successful_accuracy):.3f}"
    cells[2].text = f"{np.median(unsuccessful_accuracy):.3f}"
    cells = table.rows[4].cells
    cells[0].text = "Std Dev"
    cells[1].text = f"{successful_accuracy.std():.3f}"
    cells[2].text = f"{unsuccessful_accuracy.std():.3f}"
    cells = table.rows[5].cells
    cells[0].text = "Min"
    cells[1].text = f"{successful_accuracy.min():.3f}"
    cells[2].text = f"{unsuccessful_accuracy.min():.3f}"
    cells = table.rows[6].cells
    cells[0].text = "Max"
    cells[1].text = f"{successful_accuracy.max():.3f}"
    cells[2].text = f"{unsuccessful_accuracy.max():.3f}"
    
    doc.add_paragraph(f"{test_name_acc} Test Results:")
    table = doc.add_table(rows=3, cols=2)
    table.style = "Light Grid Accent 1"
    cells = table.rows[0].cells
    cells[0].text = "Test Statistic"
    cells[1].text = "Value"
    cells = table.rows[1].cells
    cells[0].text = stat_name_acc
    cells[1].text = f"{stat_acc:.4f}"
    cells = table.rows[2].cells
    cells[0].text = "p-value"
    cells[1].text = f"{p_value_acc:.4e}"
    
    doc.add_paragraph()
    
    # Add post-hoc results if available
    if posthoc_acc is not None:
        doc.add_paragraph("Dunn's Post-hoc Test Results (pairwise comparisons, Bonferroni-adjusted):")
        # Create table from post-hoc DataFrame
        posthoc_table = doc.add_table(rows=len(posthoc_acc) + 1, cols=len(posthoc_acc.columns) + 1)
        posthoc_table.style = "Light Grid Accent 1"
        
        # Header row
        header_cells = posthoc_table.rows[0].cells
        header_cells[0].text = "Comparison"
        for col_idx, col_name in enumerate(posthoc_acc.columns):
            header_cells[col_idx + 1].text = str(col_name)
        
        # Data rows
        for row_idx, (idx_name, row_data) in enumerate(posthoc_acc.iterrows(), 1):
            data_cells = posthoc_table.rows[row_idx].cells
            data_cells[0].text = str(idx_name)
            for col_idx, val in enumerate(row_data):
                data_cells[col_idx + 1].text = f"{float(val):.4f}"
    
    fig_path = results_dir / f"decision_accuracy_{EXPERIMENT_NAME}.png"
    if fig_path.exists():
        doc.add_picture(str(fig_path), width=Inches(6))
    doc.add_paragraph()

# Section 3.3: Decision Diagnostic Analysis
doc.add_heading("3.3. Decision Details", level=2)

doc.add_paragraph(
    "Decision-making behavior at the level of individual simulation ticks. We track how frequently decisions are made and how this relates to survival. "
    "This reveals whether success comes from making more decisions, fewer decisions, or from the quality rather than quantity of decisions."
)

print_diagnostic_stats = df_diagnostic[df_diagnostic["group"] == "Unsuccessful"]
successful_print_diagnostic = df_diagnostic[df_diagnostic["group"] == "Successful"]

table = doc.add_table(rows=4, cols=3)
table.style = "Light Grid Accent 1"
cells = table.rows[0].cells
cells[0].text = "Metric"
cells[1].text = "Successful"
cells[2].text = "Unsuccessful"
cells = table.rows[1].cells
cells[0].text = "Mean decisions/tick"
cells[1].text = f"{successful_print_diagnostic['decisions_per_tick'].mean():.3f}"
cells[2].text = f"{print_diagnostic_stats['decisions_per_tick'].mean():.3f}"
cells = table.rows[2].cells
cells[0].text = "Median decisions/tick"
cells[1].text = f"{successful_print_diagnostic['decisions_per_tick'].median():.3f}"
cells[2].text = f"{print_diagnostic_stats['decisions_per_tick'].median():.3f}"
cells = table.rows[3].cells
cells[0].text = "Mean total decisions/run"
cells[1].text = f"{successful_print_diagnostic['total_decisions'].mean():.1f}"
cells[2].text = f"{print_diagnostic_stats['total_decisions'].mean():.1f}"

fig_path = results_dir / f"decision_diagnostic_{EXPERIMENT_NAME}.png"
if fig_path.exists():
    doc.add_picture(str(fig_path), width=Inches(6))
doc.add_paragraph()

# Section 3.4: Distance travelled (with subsections 3.4.1 and 3.4.2)
# Section 3.5: Food eaten (with subsections 3.5.1 and 3.5.2)
metric_names_for_doc = [
    ("distance", "3.4", "Distance travelled", [
        ("distance", "3.4.1", "Distance travelled absolute"),
        ("distance_per_tick", "3.4.2", "Distance travelled per tick")
    ]),
    ("foods", "3.5", "Food eaten", [
        ("foods", "3.5.1", "Food consumption absolute"),
        ("foods_per_tick", "3.5.2", "Food consumption per tick")
    ])
]

for main_metric, main_section_num, main_title, subsections in metric_names_for_doc:
    # Add main section heading
    doc.add_heading(f"{main_section_num}. {main_title}", level=2)
    
    if main_section_num == "3.4":
        doc.add_paragraph(
            "Distance travelled total and normalized per tick for lifespan differences. This reveals movement intensity. "
            "We compare movement patterns between successful and unsuccessful variants to understand how spatial exploration versus local exploitation correlates with fitness."
        )
    elif main_section_num == "3.5":
        doc.add_paragraph(
            "Food consumption is the ultimate solution of the task. The more food consumed, the longer the survival. "
            "Normalized per tick food consumption may be the metric most directly linked to survival fitness."
        )
    
    # Process each subsection
    for metric_col, section_num, section_title in subsections:
        if metric_col not in metric_stats:
            continue
        
        stats = metric_stats[metric_col]
        doc.add_heading(f"{section_num}. {section_title}", level=3)
        
        table = doc.add_table(rows=7, cols=3)
        table.style = "Light Grid Accent 1"
        cells = table.rows[0].cells
        cells[0].text = "Metric"
        cells[1].text = "Successful (Top 10%)"
        cells[2].text = "Unsuccessful (Bottom 10%)"
        cells = table.rows[1].cells
        cells[0].text = "Number of runs"
        cells[1].text = f"{stats['successful_n']}"
        cells[2].text = f"{stats['unsuccessful_n']}"
        cells = table.rows[2].cells
        cells[0].text = "Mean"
        cells[1].text = f"{stats['successful_mean']:.3f}"
        cells[2].text = f"{stats['unsuccessful_mean']:.3f}"
        cells = table.rows[3].cells
        cells[0].text = "Median"
        cells[1].text = f"{stats['successful_median']:.3f}"
        cells[2].text = f"{stats['unsuccessful_median']:.3f}"
        cells = table.rows[4].cells
        cells[0].text = "Std Dev"
        cells[1].text = f"{stats['successful_std']:.3f}"
        cells[2].text = f"{stats['unsuccessful_std']:.3f}"
        cells = table.rows[5].cells
        cells[0].text = "Min"
        cells[1].text = f"{stats['successful_min']:.3f}"
        cells[2].text = f"{stats['unsuccessful_min']:.3f}"
        cells = table.rows[6].cells
        cells[0].text = "Max"
        cells[1].text = f"{stats['successful_max']:.3f}"
        cells[2].text = f"{stats['unsuccessful_max']:.3f}"
        
        doc.add_paragraph(f"{stats.get('test_name', 'Kruskal-Wallis')} Test Results:")
        table = doc.add_table(rows=3, cols=2)
        table.style = "Light Grid Accent 1"
        cells = table.rows[0].cells
        cells[0].text = "Test Statistic"
        cells[1].text = "Value"
        cells = table.rows[1].cells
        cells[0].text = stats.get('stat_name', 'H-statistic')
        cells[1].text = f"{stats['stat']:.1f}"
        cells = table.rows[2].cells
        cells[0].text = "p-value"
        cells[1].text = f"{stats['p_value']:.4e}"
        
        doc.add_paragraph()
        
        # Add post-hoc results if available
        if stats.get('posthoc') is not None:
            posthoc_df = stats['posthoc']
            doc.add_paragraph("Dunn's Post-hoc Test Results (pairwise comparisons, Bonferroni-adjusted):")
            # Create table from post-hoc DataFrame
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
        
        fig_path = results_dir / f"{metric_col}_{EXPERIMENT_NAME}.png"
        if fig_path.exists():
            doc.add_picture(str(fig_path), width=Inches(6))
        doc.add_paragraph()

# Save document
report_path = results_dir / f"report_{EXPERIMENT_NAME}.docx"
doc.save(report_path)
print(f"\nReport saved to: {report_path}")

input("Press Enter to close all figures...")
plt.ioff()
