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

plt.ion()  # Enable interactive mode

# =====================================================================
# CONFIGURATION
# =====================================================================


# Point to your variant data
BASE_DIR = Path(__file__).resolve().parents[2] / "rawdata"

# Manually specify the experiment folder (or set to None to auto-detect most recent)
EXPERIMENT_FOLDER = "2026-03-07_17-00-49_random_wiring_test"  # Change this to your experiment folder name

EXPERIMENT_DIR = BASE_DIR / EXPERIMENT_FOLDER
if not EXPERIMENT_DIR.exists():
    raise FileNotFoundError(f"Experiment folder not found: {EXPERIMENT_DIR}")


# =====================================================================
# Load all variants
# =====================================================================

variant_dirs = sorted([d for d in EXPERIMENT_DIR.iterdir() if d.is_dir() and d.name.startswith("variant_")])
print(f"Found {len(variant_dirs)} variants")

all_data = []

for variant_dir in variant_dirs:
    variant_name = variant_dir.name  # e.g., "variant_001"
    summary_file = variant_dir / "summary_random_wiring_test.csv"
    
    if summary_file.exists():
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

# Calculate coefficient of variation (normalized variance)
df_stats["cv_survival"] = df_stats["std_survival"] / df_stats["mean_survival"]

# Identify special variants
top_10_variants = set(df_stats.nlargest(10, "mean_survival")["variant"].values)
bottom_10_variants = set(df_stats.nsmallest(10, "mean_survival")["variant"].values)
high_cv_variants = set(df_stats.nlargest(5, "cv_survival")["variant"].values)
low_cv_variants = set(df_stats.nsmallest(5, "cv_survival")["variant"].values)

# =====================================================================
# Create survival race plot with color/style coding
# =====================================================================

sns.set_theme(style="whitegrid", context="talk")

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
    
    # Determine color, linewidth, linestyle, and alpha
    if variant_name in top_10_variants:
        color = "green"
        alpha = 0.8
    elif variant_name in bottom_10_variants:
        color = "#8B3A3A"  # wine/dark red color
        alpha = 0.8
    else:
        color = "gray"
        alpha = 0.3
    
    if variant_name in high_cv_variants:
        linewidth = 2.5  # bold
        linestyle = "-"
    elif variant_name in low_cv_variants:
        linewidth = 1.0
        linestyle = "--"  # dashed
    else:
        linewidth = 0.8
        linestyle = "-"  # continuous
    
    plt.plot(
        ticks,
        alive_count,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        color=color
    )

plt.xlabel("Tick", fontsize=12)
plt.ylabel("Number of Bytes alive", fontsize=12)
plt.title(f"Survival Race: All {len(variant_dirs)} Wiring Variants\n(100 world seeds per variant)")
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save figure to this same folder
results_dir = Path(__file__).resolve().parent
results_dir.mkdir(exist_ok=True)
fig_path = results_dir / "survival_race_all_variants.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {fig_path}")

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
        "min_variant_mean": float(df_stats['mean_survival'].min()),
        "max_variant_mean": float(df_stats['mean_survival'].max()),
        "range": float(df_stats['mean_survival'].max() - df_stats['mean_survival'].min()),
    },
    "median_survival": {
        "across_variants_mean": float(df_stats['median_survival'].mean()),
        "across_variants_std": float(df_stats['median_survival'].std()),
    },
    "variance_in_survival": {
        "mean_cv_across_variants": float(df_stats['cv_survival'].mean()),
        "std_of_cvs": float(df_stats['cv_survival'].std()),
    },
    "top_10_variants": df_stats.nlargest(10, "mean_survival")[["variant", "mean_survival", "std_survival", "cv_survival"]].to_dict('records'),
    "bottom_10_variants": df_stats.nsmallest(10, "mean_survival")[["variant", "mean_survival", "std_survival", "cv_survival"]].to_dict('records'),
    "high_cv_variants": df_stats.nlargest(5, "cv_survival")[["variant", "mean_survival", "cv_survival"]].to_dict('records'),
    "low_cv_variants": df_stats.nsmallest(5, "cv_survival")[["variant", "mean_survival", "cv_survival"]].to_dict('records'),
}

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
stats_file = results_dir / "statistical_summary_all_variants.json"
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
print(f"  Min variant mean: {df_stats['mean_survival'].min():.1f}")
print(f"  Max variant mean: {df_stats['mean_survival'].max():.1f}")
print(f"  Range: {df_stats['mean_survival'].max() - df_stats['mean_survival'].min():.1f}")

print(f"\nMedian survival time (ticks):")
print(f"  Across variants - Mean: {df_stats['median_survival'].mean():.1f}")
print(f"  Across variants - Std:  {df_stats['median_survival'].std():.1f}")

print(f"\nVariance in survival (normalized, coefficient of variation):")
print(f"  Mean CV across variants: {df_stats['cv_survival'].mean():.3f}")
print(f"  Std of CVs: {df_stats['cv_survival'].std():.3f}")

print(f"\nVariants by mean survival (Top 10):")
print(df_stats.nlargest(10, "mean_survival")[["variant", "mean_survival", "std_survival", "cv_survival"]])

print(f"\nVariants by mean survival (Bottom 10):")
print(df_stats.nsmallest(10, "mean_survival")[["variant", "mean_survival", "std_survival", "cv_survival"]])

# Check for variants with high/low relative variability
print(f"\nVariants with HIGH relative variability (high CV, top 5):")
print(df_stats.nlargest(5, "cv_survival")[["variant", "mean_survival", "cv_survival"]])

print(f"\nVariants with LOW relative variability (low CV, bottom 5):")
print(df_stats.nsmallest(5, "cv_survival")[["variant", "mean_survival", "cv_survival"]])

print("\nDone!")

input("Press Enter to close all figures...")
plt.ioff()
