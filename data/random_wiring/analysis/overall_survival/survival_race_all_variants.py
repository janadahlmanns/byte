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

# =====================================================================
# CONFIGURATION
# =====================================================================

# Point to your variant data - go up two levels from overall_survival to get to rawdata
BASE_DIR = Path(__file__).resolve().parents[2] / "rawdata"

# Find the most recent random_wiring experiment
experiment_dirs = sorted(BASE_DIR.glob("2026-*_random_wiring_test"))
if not experiment_dirs:
    raise FileNotFoundError(f"No random_wiring_test directories found in {BASE_DIR}")

EXPERIMENT_DIR = experiment_dirs[-1]  # Most recent
print(f"Using experiment: {EXPERIMENT_DIR.name}")

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
# Create survival race plot for ALL variants
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
    
    # Plot with low alpha so overlapping lines create a "heat map" effect
    plt.plot(
        ticks,
        alive_count,
        linewidth=0.8,
        alpha=0.3,
        color="steelblue"
    )

plt.xlabel("Tick", fontsize=12)
plt.ylabel("Number of Bytes alive", fontsize=12)
plt.title(f"Survival Race: All {len(variant_dirs)} Wiring Variants\n(100 world seeds per variant, mean trajectory shown)")
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save figure to this same folder
results_dir = Path(__file__).resolve().parent
results_dir.mkdir(exist_ok=True)
fig_path = results_dir / "survival_race_all_variants.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {fig_path}")

plt.close()

# =====================================================================
# Also compute summary statistics across all variants
# =====================================================================

print("\n" + "="*60)
print("STATISTICAL SUMMARY ACROSS ALL VARIANTS")
print("="*60)

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

print(f"\nMean survival time (ticks):")
print(f"  Across variants - Mean: {df_stats['mean_survival'].mean():.1f}")
print(f"  Across variants - Std:  {df_stats['mean_survival'].std():.1f}")
print(f"  Min variant mean: {df_stats['mean_survival'].min():.1f}")
print(f"  Max variant mean: {df_stats['mean_survival'].max():.1f}")
print(f"  Range: {df_stats['mean_survival'].max() - df_stats['mean_survival'].min():.1f}")

print(f"\nMedian survival time (ticks):")
print(f"  Across variants - Mean: {df_stats['median_survival'].mean():.1f}")
print(f"  Across variants - Std:  {df_stats['median_survival'].std():.1f}")

print(f"\nVariance in survival (std within variants):")
print(f"  Mean std across variants: {df_stats['std_survival'].mean():.1f}")
print(f"  Std of stds: {df_stats['std_survival'].std():.1f}")

print(f"\nVariants by mean survival (Top 10):")
print(df_stats.nlargest(10, "mean_survival")[["variant", "mean_survival", "std_survival"]])

print(f"\nVariants by mean survival (Bottom 10):")
print(df_stats.nsmallest(10, "mean_survival")[["variant", "mean_survival", "std_survival"]])

# Check for any weird outliers
print(f"\nVariants with unusually HIGH variance in survival (top 5):")
print(df_stats.nlargest(5, "std_survival")[["variant", "mean_survival", "std_survival"]])

print(f"\nVariants with unusually LOW variance in survival (bottom 5):")
print(df_stats.nsmallest(5, "std_survival")[["variant", "mean_survival", "std_survival"]])

print("\nDone!")
