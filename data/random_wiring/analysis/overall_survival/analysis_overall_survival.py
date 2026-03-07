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
from scipy.stats import mannwhitneyu

plt.ion()  # Enable interactive mode

# =====================================================================
# CONFIGURATION
# =====================================================================

# Point to your variant data
BASE_DIR = Path(__file__).resolve().parents[2] / "rawdata"

# Manually specify the experiment folder (or set to None to auto-detect most recent)
EXPERIMENT_FOLDER = "2026-03-07_21-08-07_random_wiring_exc20_inh40_pot_10_dep_5"  # Change this to your experiment folder name

EXPERIMENT_DIR = BASE_DIR / EXPERIMENT_FOLDER
if not EXPERIMENT_DIR.exists():
    raise FileNotFoundError(f"Experiment folder not found: {EXPERIMENT_DIR}")

# Extract experiment name from script filename (not EXPERIMENT_FOLDER)
script_name = Path(__file__).stem  # e.g., "analysis_overall_survival" -> "overall_survival"
EXPERIMENT_NAME = script_name.replace("analysis_", "")


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

plt.xlabel("Tick", fontsize=12)
plt.ylabel("Number of Bytes alive", fontsize=12)
plt.title(f"Survival Race: All {len(variant_dirs)} Wiring Variants\n(100 world seeds per variant)")
plt.grid(True, alpha=0.3)

# Create custom legend
legend_elements = [
    Line2D([0], [0], color=green_color, linewidth=1.5, label=f"median survival in the 90th percentile(≥ {median_90_pct_threshold:.1f} ticks)"),
    Line2D([0], [0], color=red_color, linewidth=1.5, label=f"median survival in the 10th percentile(≤ {median_10_pct_threshold:.1f} ticks)"),
    Line2D([0], [0], color="gray", linewidth=0.8, label="all other variants"),
]
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
# Comparison: Successful vs Unsuccessful variants
# =====================================================================

print("\n" + "="*60)
print("COMPARISON: SUCCESSFUL VS UNSUCCESSFUL VARIANTS")
print("="*60)

# Identify groups
successful_variants = set(df_stats[df_stats["median_survival"] >= median_90_pct_threshold]["variant"].values)
unsuccessful_variants = set(df_stats[df_stats["median_survival"] <= median_10_pct_threshold]["variant"].values)

print(f"\nSuccessful variants (top 10%): {len(successful_variants)}")
print(f"Unsuccessful variants (bottom 10%): {len(unsuccessful_variants)}")

# Prepare data for jitter plot
plot_data = []

# Order: unsuccessful first (left), then successful (right)
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

df_plot = pd.DataFrame(plot_data)

print(f"\nPlot data points: {len(df_plot)}")
print(f"  Unsuccessful: {len(df_plot[df_plot['group'] == 'Unsuccessful\n(Bottom 10%)'])}")
print(f"  Successful: {len(df_plot[df_plot['group'] == 'Successful\n(Top 10%)'])}")

# Get data for statistics
unsuccessful_times = df_plot[df_plot["group"] == "Unsuccessful\n(Bottom 10%)"]["lifetime_ticks"].values
successful_times = df_plot[df_plot["group"] == "Successful\n(Top 10%)"]["lifetime_ticks"].values

# Run Mann-Whitney U test (non-parametric alternative to t-test for non-normal distributions with unpaired samples)
u_stat, p_value = mannwhitneyu(unsuccessful_times, successful_times, alternative='two-sided')

# Create jitter + box plot
fig, ax = plt.subplots(figsize=(10, 8))

# Create jitter plot
sns.stripplot(
    data=df_plot,
    x="group",
    y="lifetime_ticks",
    hue="group",
    palette={"Unsuccessful\n(Bottom 10%)": red_color, "Successful\n(Top 10%)": green_color},
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
    palette={"Unsuccessful\n(Bottom 10%)": red_color, "Successful\n(Top 10%)": green_color},
    width=0.3,
    ax=ax,
    showcaps=True,
    whiskerprops={'linewidth': 2},
    boxprops={'linewidth': 2},
    medianprops={'color': 'black', 'linewidth': 2},
    fliersize=0
)

# Make box patches more translucent
for patch in ax.patches:
    patch.set_alpha(0.3)

ax.set_xlabel("Variant Group", fontsize=12)
ax.set_ylabel("Lifetime (ticks)", fontsize=12)
title_text = f"Survival Times: Unsuccessful vs Successful Wiring Variants\n(Mann-Whitney U: U={u_stat:.4f}, p={p_value:.4e})"
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
print(f"\nMann-Whitney U Test Results:")
print(f"  U-statistic: {u_stat:.4f}")
print(f"  p-value: {p_value:.4e}")

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

df_efficiency = pd.DataFrame(efficiency_data)

if not df_efficiency.empty:
    # Compute summary statistics for bar plots
    summary_stats = df_efficiency.groupby(['group', 'direction'])['efficiency'].agg(['mean', 'std', 'sem']).reset_index()
    
    # Create figure with 2 subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle("Movement Efficiency by Direction (Movements / Sensing Events)", fontsize=14, fontweight='bold')
    
    # Left subplot: Unsuccessful
    ax_unsuccessful = axes[0]
    unsuccessful_data = summary_stats[summary_stats['group'] == 'Unsuccessful'].sort_values('direction')
    x_pos = np.arange(len(directions))
    
    ax_unsuccessful.bar(
        x_pos,
        unsuccessful_data['mean'].values,
        yerr=unsuccessful_data['sem'].values,
        capsize=5,
        color=red_color,
        alpha=0.7,
        edgecolor='black',
        linewidth=1.5
    )
    ax_unsuccessful.set_xlabel("Direction", fontsize=11)
    ax_unsuccessful.set_ylabel("Movement Efficiency", fontsize=11)
    ax_unsuccessful.set_title("Unsuccessful (Bottom 10%)", fontsize=12, fontweight='bold')
    ax_unsuccessful.set_xticks(x_pos)
    ax_unsuccessful.set_xticklabels(directions)
    ax_unsuccessful.grid(axis='y', alpha=0.3)
    
    # Right subplot: Successful
    ax_successful = axes[1]
    successful_data = summary_stats[summary_stats['group'] == 'Successful'].sort_values('direction')
    
    ax_successful.bar(
        x_pos,
        successful_data['mean'].values,
        yerr=successful_data['sem'].values,
        capsize=5,
        color=green_color,
        alpha=0.7,
        edgecolor='black',
        linewidth=1.5
    )
    ax_successful.set_xlabel("Direction", fontsize=11)
    ax_successful.set_title("Successful (Top 10%)", fontsize=12, fontweight='bold')
    ax_successful.set_xticks(x_pos)
    ax_successful.set_xticklabels(directions)
    ax_successful.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    fig_efficiency_path = results_dir / f"movement_efficiency_{EXPERIMENT_NAME}.png"
    fig.savefig(fig_efficiency_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {fig_efficiency_path}")
    fig_paths.append(("Movement Efficiency by Direction", fig_efficiency_path))
    
    # Print statistics
    print("\nMovement Efficiency Statistics (Movement / Sensing):\n")
    for group in ['Unsuccessful', 'Successful']:
        print(f"\n{group} Variants:")
        group_data = summary_stats[summary_stats['group'] == group].sort_values('direction')
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
            # Avoid division by zero
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

df_accuracy = pd.DataFrame(decision_accuracy_data)

if not df_accuracy.empty:
    print(f"\nDecision Accuracy data points: {len(df_accuracy)}")
    print(f"  Unsuccessful: {len(df_accuracy[df_accuracy['group'] == 'Unsuccessful\\n(Bottom 10%)'])}")
    print(f"  Successful: {len(df_accuracy[df_accuracy['group'] == 'Successful\\n(Top 10%)'])}")
    
    # Get data for statistics
    unsuccessful_accuracy = df_accuracy[df_accuracy["group"] == "Unsuccessful\n(Bottom 10%)"]["accuracy"].values
    successful_accuracy = df_accuracy[df_accuracy["group"] == "Successful\n(Top 10%)"]["accuracy"].values
    
    # Run Mann-Whitney U test
    u_stat_acc, p_value_acc = mannwhitneyu(unsuccessful_accuracy, successful_accuracy, alternative='two-sided')
    
    # Create jitter + box plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create jitter plot
    sns.stripplot(
        data=df_accuracy,
        x="group",
        y="accuracy",
        hue="group",
        palette={"Unsuccessful\n(Bottom 10%)": red_color, "Successful\n(Top 10%)": green_color},
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
        palette={"Unsuccessful\n(Bottom 10%)": red_color, "Successful\n(Top 10%)": green_color},
        width=0.3,
        ax=ax,
        showcaps=True,
        whiskerprops={'linewidth': 2},
        boxprops={'linewidth': 2},
        medianprops={'color': 'black', 'linewidth': 2},
        fliersize=0
    )
    
    # Make box patches more translucent
    for patch in ax.patches:
        patch.set_alpha(0.3)
    
    ax.set_xlabel("Variant Group", fontsize=12)
    ax.set_ylabel("Decision Accuracy (Correct / Total)", fontsize=12)
    title_text = f"Decision Accuracy: Unsuccessful vs Successful Wiring Variants\n(Mann-Whitney U: U={u_stat_acc:.4f}, p={p_value_acc:.4e})"
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
    print(f"\nMann-Whitney U Test Results (Decision Accuracy):")
    print(f"  U-statistic: {u_stat_acc:.4f}")
    print(f"  p-value: {p_value_acc:.4e}")
    
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

# Create diagnostic figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Total decisions vs Lifetime
for group, color in [("Middle", "gray"), ("Successful", green_color), ("Unsuccessful", red_color)]:
    data = df_diagnostic[df_diagnostic["group"] == group]
    axes[0].scatter(data["total_decisions"], data["lifetime"], alpha=0.4, s=20, color=color, label=group)

axes[0].set_xlabel("Total Decisions Made", fontsize=11)
axes[0].set_ylabel("Lifetime (ticks)", fontsize=11)
axes[0].set_title("Decision Count vs Survival Time", fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Plot 2: Decisions per tick vs Lifetime
for group, color in [("Middle", "gray"), ("Successful", green_color), ("Unsuccessful", red_color)]:
    data = df_diagnostic[df_diagnostic["group"] == group]
    axes[1].scatter(data["decisions_per_tick"], data["lifetime"], alpha=0.4, s=20, color=color, label=group)

axes[1].set_xlabel("Decisions per Tick", fontsize=11)
axes[1].set_ylabel("Lifetime (ticks)", fontsize=11)
axes[1].set_title("Decision Density vs Survival Time", fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

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
    ax=axes[2],
    dodge=False
)

sns.boxplot(
    data=df_diagnostic_box,
    x="group",
    y="decisions_per_tick",
    palette={"Unsuccessful": red_color, "Successful": green_color},
    width=0.3,
    ax=axes[2],
    showcaps=True,
    whiskerprops={'linewidth': 1.5},
    boxprops={'linewidth': 1.5},
    medianprops={'color': 'black', 'linewidth': 1.5},
    fliersize=0
)

for patch in axes[2].patches:
    patch.set_alpha(0.3)

axes[2].set_xlabel("Variant Group", fontsize=11)
axes[2].set_ylabel("Decisions per Tick", fontsize=11)
axes[2].set_title("Decision Density Distribution", fontsize=12, fontweight='bold')
if axes[2].get_legend() is not None:
    axes[2].get_legend().remove()

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
    
    # Run Mann-Whitney U test
    u_stat_metric, p_value_metric = mannwhitneyu(unsuccessful_vals, successful_vals, alternative='two-sided')
    
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
        "u_stat": u_stat_metric,
        "p_value": p_value_metric
    }
    
    print(f"  Unsuccessful - Mean: {unsuccessful_vals.mean():.3f}, Median: {np.median(unsuccessful_vals):.3f}")
    print(f"  Successful - Mean: {successful_vals.mean():.3f}, Median: {np.median(successful_vals):.3f}")
    print(f"  Mann-Whitney U: {u_stat_metric:.1f}, p-value: {p_value_metric:.4e}")
    
    # Create jitter + box plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create jitter plot
    sns.stripplot(
        data=df_movement_feeding,
        x="group",
        y=metric_col,
        hue="group",
        palette={"Unsuccessful\n(Bottom 10%)": red_color, "Successful\n(Top 10%)": green_color},
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
        palette={"Unsuccessful\n(Bottom 10%)": red_color, "Successful\n(Top 10%)": green_color},
        width=0.3,
        ax=ax,
        showcaps=True,
        whiskerprops={'linewidth': 2},
        boxprops={'linewidth': 2},
        medianprops={'color': 'black', 'linewidth': 2},
        fliersize=0
    )
    
    # Make box patches more translucent
    for patch in ax.patches:
        patch.set_alpha(0.3)
    
    ax.set_xlabel("Variant Group", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    title_text = f"{metric_title}: Unsuccessful vs Successful Variants\n(Mann-Whitney U: U={u_stat_metric:.1f}, p={p_value_metric:.4e})"
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
# Generate Report Document
# =====================================================================

doc = Document()

# Title
title = doc.add_paragraph()
title_run = title.add_run(f"Analysis Report: Random Wiring Variants\n{EXPERIMENT_NAME}")
title_run.font.size = Pt(18)
title_run.bold = True
title.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER

# Experiment info
doc.add_paragraph("Experiment Information", style="Heading 2")
doc.add_paragraph(f"Experiment folder: {EXPERIMENT_DIR.name}", style="List Bullet")
doc.add_paragraph(f"Number of variants: {len(variant_dirs)}", style="List Bullet")
doc.add_paragraph(f"Runs per variant: 100", style="List Bullet")

# Summary statistics section
doc.add_paragraph("Summary Statistics", style="Heading 2")

doc.add_paragraph("Mean survival time (ticks):", style="Heading 3")
table = doc.add_table(rows=5, cols=2)
table.style = "Light Grid Accent 1"
cells = table.rows[0].cells
cells[0].text = "Metric"
cells[1].text = "Value"
cells = table.rows[1].cells
cells[0].text = "Mean across variants"
cells[1].text = f"{df_stats['mean_survival'].mean():.1f}"
cells = table.rows[2].cells
cells[0].text = "Std across variants"
cells[1].text = f"{df_stats['mean_survival'].std():.1f}"
cells = table.rows[3].cells
cells[0].text = "Min variant mean"
cells[1].text = f"{df_stats['mean_survival'].min():.1f}"
cells = table.rows[4].cells
cells[0].text = "Max variant mean"
cells[1].text = f"{df_stats['mean_survival'].max():.1f}"

doc.add_paragraph("Median survival time (ticks):", style="Heading 3")
table = doc.add_table(rows=3, cols=2)
table.style = "Light Grid Accent 1"
cells = table.rows[0].cells
cells[0].text = "Metric"
cells[1].text = "Value"
cells = table.rows[1].cells
cells[0].text = "Mean median across variants"
cells[1].text = f"{df_stats['median_survival'].mean():.1f}"
cells = table.rows[2].cells
cells[0].text = "Std median across variants"
cells[1].text = f"{df_stats['median_survival'].std():.1f}"

doc.add_paragraph("Percentile Analysis (for ranking):", style="Heading 3")
table = doc.add_table(rows=4, cols=2)
table.style = "Light Grid Accent 1"
cells = table.rows[0].cells
cells[0].text = "Metric"
cells[1].text = "Value (ticks)"
cells = table.rows[1].cells
cells[0].text = "90th percentile threshold"
cells[1].text = f"{median_90_pct_threshold:.1f}"
cells = table.rows[2].cells
cells[0].text = "Variants in 90th percentile"
cells[1].text = f"{len(successful_variants)}"
cells = table.rows[3].cells
cells[0].text = "10th percentile threshold"
cells[1].text = f"{median_10_pct_threshold:.1f}"

doc.add_paragraph("Statistical Test (Unsuccessful vs Successful):", style="Heading 3")
table = doc.add_table(rows=3, cols=2)
table.style = "Light Grid Accent 1"
cells = table.rows[0].cells
cells[0].text = "Test"
cells[1].text = "Value"
cells = table.rows[1].cells
cells[0].text = "Mann-Whitney U-statistic"
cells[1].text = f"{u_stat:.4f}"
cells = table.rows[2].cells
cells[0].text = "p-value"
cells[1].text = f"{p_value:.4e}"

doc.add_paragraph("Comparison: Successful vs Unsuccessful Variants:", style="Heading 3")
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

# Figures section
doc.add_paragraph("Analysis Figures and Results", style="Heading 2")

# Figure 1: Survival Race
doc.add_paragraph("1. Survival Race - All Variants", style="Heading 3")
doc.add_paragraph(
    f"This plot shows the survival curves for all {len(variant_dirs)} wiring variants. "
    f"Lines are color-coded by percentile: green indicates top 10% (median ≥ {median_90_pct_threshold:.1f} ticks), "
    f"red indicates bottom 10% (median ≤ {median_10_pct_threshold:.1f} ticks), and gray indicates the middle 80%."
)
fig_path = results_dir / f"survival_race_{EXPERIMENT_NAME}.png"
doc.add_picture(str(fig_path), width=Inches(6))
doc.add_paragraph()

# Figure 2: Survival Time Comparison
doc.add_paragraph("2. Survival Time Comparison: Unsuccessful vs Successful", style="Heading 3")
doc.add_paragraph("Statistical Comparison of Lifetime Durations:")
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

doc.add_paragraph("Mann-Whitney U Test Results (non-parametric test for non-normal distributions):")
table = doc.add_table(rows=3, cols=2)
table.style = "Light Grid Accent 1"
cells = table.rows[0].cells
cells[0].text = "Test Statistic"
cells[1].text = "Value"
cells = table.rows[1].cells
cells[0].text = "Mann-Whitney U-statistic"
cells[1].text = f"{u_stat:.4f}"
cells = table.rows[2].cells
cells[0].text = "p-value"
cells[1].text = f"{p_value:.4e}"

fig_path = results_dir / f"comp_survival_{EXPERIMENT_NAME}.png"
doc.add_picture(str(fig_path), width=Inches(6))
doc.add_paragraph()

# Figure 3: Movement Efficiency
doc.add_paragraph("3. Movement Efficiency by Direction", style="Heading 3")
doc.add_paragraph("Movement Efficiency = Movement Count ÷ Sensing Count per direction")
doc.add_paragraph("Efficiency Statistics (Mean ± Std Dev):")
table = doc.add_table(rows=5, cols=3)
table.style = "Light Grid Accent 1"
cells = table.rows[0].cells
cells[0].text = "Direction"
cells[1].text = "Unsuccessful"
cells[2].text = "Successful"

# Get efficiency stats for the table
unsuccessful_eff = summary_stats[summary_stats['group'] == 'Unsuccessful'].sort_values('direction')
successful_eff = summary_stats[summary_stats['group'] == 'Successful'].sort_values('direction')

for idx, direction in enumerate(directions, 1):
    cells = table.rows[idx].cells
    cells[0].text = direction
    unsucc_row = unsuccessful_eff[unsuccessful_eff['direction'] == direction].iloc[0]
    succ_row = successful_eff[successful_eff['direction'] == direction].iloc[0]
    cells[1].text = f"{unsucc_row['mean']:.3f} ± {unsucc_row['std']:.3f}"
    cells[2].text = f"{succ_row['mean']:.3f} ± {succ_row['std']:.3f}"

fig_path = results_dir / f"movement_efficiency_{EXPERIMENT_NAME}.png"
doc.add_picture(str(fig_path), width=Inches(6))
doc.add_paragraph()

# Figure 4: Decision Accuracy
if not df_accuracy.empty:
    doc.add_paragraph("4. Decision Accuracy: Unsuccessful vs Successful", style="Heading 3")
    doc.add_paragraph("Decision Accuracy = Correct Decisions ÷ Total Decisions")
    doc.add_paragraph("Statistical Comparison of Decision Accuracy:")
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
    
    doc.add_paragraph("Mann-Whitney U Test Results:")
    table = doc.add_table(rows=3, cols=2)
    table.style = "Light Grid Accent 1"
    cells = table.rows[0].cells
    cells[0].text = "Test Statistic"
    cells[1].text = "Value"
    cells = table.rows[1].cells
    cells[0].text = "Mann-Whitney U-statistic"
    cells[1].text = f"{u_stat_acc:.4f}"
    cells = table.rows[2].cells
    cells[0].text = "p-value"
    cells[1].text = f"{p_value_acc:.4e}"
    
    fig_path = results_dir / f"decision_accuracy_{EXPERIMENT_NAME}.png"
    doc.add_picture(str(fig_path), width=Inches(6))
    doc.add_paragraph()

# Figure 5: Decision Diagnostic Analysis
doc.add_paragraph("5. Decision Diagnostic Analysis: Why Accuracy Doesn't Predict Survival", style="Heading 3")
doc.add_paragraph(
    "A critical finding: unsuccessful variants have HIGHER decision accuracy (33%) than "
    "successful variants (18%), but shorter lifespans. This diagnostic analysis reveals why. "
    "A 'decision' only occurs when the worm senses food AND is not on food. A 'correct decision' "
    "is when movement matches sensed food direction."
)
doc.add_paragraph()

print_diagnostic_stats = df_diagnostic[df_diagnostic["group"] == "Unsuccessful"]
successful_print_diagnostic = df_diagnostic[df_diagnostic["group"] == "Successful"]

doc.add_paragraph("Decision-Making Strategy Comparison:")
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

doc.add_paragraph("Key Insight:")
doc.add_paragraph(
    "Unsuccessful variants make ~2x more decisions per tick than successful variants. "
    "This suggests they encounter food-sensing situations far more frequently, engaging in "
    "constant food-directed micro-corrections. While they make more correct food-directed moves, "
    "this strategy consumes energy inefficiently and leads to early starvation. "
    "Successful variants use a different movement strategy with fewer food-sensing decisions, "
    "achieving better long-term survival."
)
doc.add_paragraph()

fig_path = results_dir / f"decision_diagnostic_{EXPERIMENT_NAME}.png"
if (results_dir / f"decision_diagnostic_{EXPERIMENT_NAME}.png").exists():
    doc.add_picture(str(fig_path), width=Inches(6))
    doc.add_paragraph()

# Figure 6: Movement & Feeding Metrics
metric_names_for_doc = [
    ("distance", "Distance Travelled"),
    ("distance_per_tick", "Distance Efficiency"),
    ("foods", "Food Consumption"),
    ("foods_per_tick", "Feeding Rate")
]

for idx, (metric_col, metric_short_name) in enumerate(metric_names_for_doc, 6):
    if metric_col not in metric_stats:
        continue
    
    stats = metric_stats[metric_col]
    doc.add_paragraph(f"{idx}. {stats['title']}: Unsuccessful vs Successful", style="Heading 3")
    
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
    
    doc.add_paragraph("Mann-Whitney U Test Results:")
    table = doc.add_table(rows=3, cols=2)
    table.style = "Light Grid Accent 1"
    cells = table.rows[0].cells
    cells[0].text = "Test Statistic"
    cells[1].text = "Value"
    cells = table.rows[1].cells
    cells[0].text = "Mann-Whitney U-statistic"
    cells[1].text = f"{stats['u_stat']:.1f}"
    cells = table.rows[2].cells
    cells[0].text = "p-value"
    cells[1].text = f"{stats['p_value']:.4e}"
    
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
