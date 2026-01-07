import pandas as pd
from pathlib import Path

# =====================================================================
# CONFIGURATION: GROUP DEFINITIONS
# =====================================================================
# Leave group name as "" (empty string) to disable that group
# Supports 2, 3, or 4 groups

BASE_DIR = Path(__file__).resolve().parents[1] / "rawdata"

GROUP1_NAME = "No noise"
GROUP1_DIR = "2026-01-07_12-43-42_no_noise"

GROUP2_NAME = "Noise = 0.1"
GROUP2_DIR = "2026-01-07_12-43-09_noise_0_1"

GROUP3_NAME = ""  # Leave empty to disable
GROUP3_DIR = ""

GROUP4_NAME = ""  # Leave empty to disable
GROUP4_DIR = ""

# Color palette: green, gold, cool steel, dark wine
PALETTE_COLORS = {
    GROUP1_NAME: "#0B3D2E",   # dark green
    GROUP2_NAME: "#D4AF37",   # gold
    GROUP3_NAME: "#88A0A8",   # cool steel
    GROUP4_NAME: "#721817",   # dark wine
}

print("BASE_DIR =", BASE_DIR)
print("Exists:", BASE_DIR.exists())
print("Contents:", list(BASE_DIR.iterdir()))


# =====================================================================
# Build active groups list
# =====================================================================
GROUPS = []
for name, dir_name in [
    (GROUP1_NAME, GROUP1_DIR),
    (GROUP2_NAME, GROUP2_DIR),
    (GROUP3_NAME, GROUP3_DIR),
    (GROUP4_NAME, GROUP4_DIR),
]:
    if name.strip():  # Skip empty group names
        GROUPS.append((name, BASE_DIR / dir_name))

print(f"\nActive groups: {[g[0] for g in GROUPS]}")


# =====================================================================
# Helper function: load one condition (summary + runs)
# =====================================================================
def load_condition(condition_dir, condition_label):
    condition_dir = Path(condition_dir)

    # ---- load summary ----
    summary_file = next(condition_dir.glob("summary_*.csv"))
    df_summary = pd.read_csv(summary_file)
    df_summary["condition"] = condition_label

    # ---- load runs ----
    runs_dir = condition_dir / "runs"
    run_dfs = []

    for run_file in sorted(runs_dir.glob("run_*.csv")):
        df = pd.read_csv(run_file)

        # extract run id: run_0010.csv -> 10
        run_id = int(run_file.stem.split("_")[1])

        df["run_id"] = run_id
        df["condition"] = condition_label

        run_dfs.append(df)

    df_runs = pd.concat(run_dfs, ignore_index=True)

    return df_summary, df_runs


# =====================================================================
# Load all active conditions
# =====================================================================
all_summaries = []
all_runs = []

for group_name, group_dir in GROUPS:
    summary, runs = load_condition(group_dir, group_name)
    all_summaries.append(summary)
    all_runs.append(runs)

df_summary = pd.concat(all_summaries, ignore_index=True)
df_runs = pd.concat(all_runs, ignore_index=True)


# =====================================================================
# Sanity prints 
# =====================================================================
print("Summary shape:", df_summary.shape)
print("Runs shape:", df_runs.shape)
print(df_runs.head())


# =====================================================================
# Exploratory data analysis
# =====================================================================
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

sns.set_theme(style="whitegrid", context="talk")

# Build palette for active groups only
palette = {name: PALETTE_COLORS[name] for name, _ in GROUPS}

plt.ion()


# --- energy over time
plt.figure(figsize=(8, 5))

for condition, df_c in df_runs.groupby("condition"):
    for _, df_r in df_c.groupby("run_id"):
        plt.plot(
            df_r["tick"],
            df_r["energy"],
            color=palette[condition],
            alpha=0.4,
            linewidth=1
        )

plt.xlabel("Tick")
plt.ylabel("Energy")
plt.title("Energy dynamics per run")

legend_elements = [
    Line2D([0], [0], color=palette[name], lw=3, label=name)
    for name, _ in GROUPS
]

plt.legend(handles=legend_elements, frameon=False)
plt.tight_layout()
plt.show()


# --- food over time
plt.figure(figsize=(8, 5))

for condition, df_c in df_runs.groupby("condition"):
    for _, df_r in df_c.groupby("run_id"):
        plt.plot(
            df_r["tick"],
            df_r["eats"],
            color=palette[condition],
            alpha=0.4,
            linewidth=1
        )

plt.xlabel("Tick")
plt.ylabel("Food eaten")
plt.title("Food accumulation per run")

plt.legend(handles=legend_elements, frameon=False)
plt.tight_layout()
plt.show()


# --- distance over time
plt.figure(figsize=(8, 5))

for condition, df_c in df_runs.groupby("condition"):
    for _, df_r in df_c.groupby("run_id"):
        plt.plot(
            df_r["tick"],
            df_r["distance"],
            color=palette[condition],
            alpha=0.4,
            linewidth=1
        )

plt.xlabel("Tick")
plt.ylabel("Distance")
plt.title("Distance accumulation per run")

plt.legend(handles=legend_elements, frameon=False)
plt.tight_layout()
plt.show()


# --- survival race plot
plt.figure(figsize=(8, 5))

for condition, df_c in df_summary.groupby("condition"):
    survival_times = df_c["lifetime_ticks"].values

    max_t = survival_times.max()
    ticks = range(max_t + 1)

    alive = [(survival_times >= t).sum() for t in ticks]

    plt.plot(
        ticks,
        alive,
        color=palette[condition],
        linewidth=3,
        label=condition
    )

plt.xlabel("Tick")
plt.ylabel("Bytes alive")
plt.title("Survival race of Byte simulations")

plt.legend(frameon=False)
plt.tight_layout()
plt.show()


# --- KS test (pairwise comparisons for all groups)
from scipy.stats import ks_2samp

print("\n" + "="*60)
print("STATISTICAL TESTS (Kolmogorov-Smirnov)")
print("="*60)

group_names = [name for name, _ in GROUPS]
group_data = {
    name: df_summary.loc[df_summary["condition"] == name, "lifetime_ticks"].values
    for name, _ in GROUPS
}

for i, name1 in enumerate(group_names):
    for name2 in group_names[i+1:]:
        ks_stat, p_value = ks_2samp(group_data[name1], group_data[name2])
        print(f"\n{name1} vs {name2}:")
        print(f"  KS statistic: {ks_stat:.4f}")
        print(f"  p-value: {p_value:.4e}")


# --- log-log CCDF
import numpy as np

plt.figure(figsize=(8, 5))

for condition, df_c in df_summary.groupby("condition"):
    lifetimes = np.sort(df_c["lifetime_ticks"].values)
    unique_t = np.unique(lifetimes)
    ccdf = [(lifetimes >= t).mean() for t in unique_t]

    plt.plot(
        unique_t,
        ccdf,
        marker="o",
        linestyle="none",
        markersize=4,
        alpha=0.7,
        color=palette[condition],
        label=condition
    )

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Lifetime (ticks)")
plt.ylabel("P(T ≥ t)")
plt.title("Survival CCDF (log–log)")

plt.legend(frameon=False)
plt.tight_layout()
plt.show()


input("Press Enter to close all figures...")
plt.ioff()
