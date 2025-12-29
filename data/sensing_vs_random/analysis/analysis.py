import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------
# INPUT PATH
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1] / "rawdata"

print("BASE_DIR =", BASE_DIR)
print("Exists:", BASE_DIR.exists())
print("Contents:", list(BASE_DIR.iterdir()))


# ---------------------------------------------------------------------
# Helper function: load one condition (summary + runs)
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Load both conditions
# ---------------------------------------------------------------------
sensing_dir = BASE_DIR / "2025-12-29_18-04-10_sensing"
non_sensing_dir = BASE_DIR / "2025-12-29_18-08-03_non_sensing"

summary_sensing, runs_sensing = load_condition(sensing_dir, "sensing")
summary_non, runs_non = load_condition(non_sensing_dir, "non_sensing")


# ---------------------------------------------------------------------
# Final concatenation
# ---------------------------------------------------------------------
df_summary = pd.concat([summary_sensing, summary_non], ignore_index=True)
df_runs = pd.concat([runs_sensing, runs_non], ignore_index=True)


# ---------------------------------------------------------------------
# Sanity prints 
# ---------------------------------------------------------------------
print("Summary shape:", df_summary.shape)
print("Runs shape:", df_runs.shape)
print(df_runs.head())

# ---------------------------------------------------------------------
# Exploratory data analysis
# ---------------------------------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

sns.set_theme(style="whitegrid", context="talk")

palette = {
    "sensing": "#0B3D2E",      # dark christmas-tree green
    "non_sensing": "#D4AF37"   # rich gold
}
plt.ion()


# --- energy over time
plt.figure(figsize=(8, 5))

for condition, df_c in df_runs.groupby("condition"):
    for run_id, df_r in df_c.groupby("run_id"):
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

# ---- manual legend (robust & clean) ----
legend_elements = [
    Line2D([0], [0], color=palette["sensing"], lw=3, label="Sensing"),
    Line2D([0], [0], color=palette["non_sensing"], lw=3, label="Non-sensing")
]

plt.legend(handles=legend_elements, frameon=False)

plt.tight_layout()
plt.show()


# --- food over time
plt.figure(figsize=(8, 5))

for condition, df_c in df_runs.groupby("condition"):
    for run_id, df_r in df_c.groupby("run_id"):
        plt.plot(
            df_r["tick"],
            df_r["eats"],
            color=palette[condition],
            alpha=0.4,
            linewidth=1
        )

plt.xlabel("Tick")
plt.ylabel("Food")
plt.title("Food accumulation per run")

# ---- manual legend (robust & clean) ----
legend_elements = [
    Line2D([0], [0], color=palette["sensing"], lw=3, label="Sensing"),
    Line2D([0], [0], color=palette["non_sensing"], lw=3, label="Non-sensing")
]

plt.legend(handles=legend_elements, frameon=False)

plt.tight_layout()
plt.show()

# --- distance over time
plt.figure(figsize=(8, 5))

for condition, df_c in df_runs.groupby("condition"):
    for run_id, df_r in df_c.groupby("run_id"):
        plt.plot(
            df_r["tick"],
            df_r["distance"],
            color=palette[condition],
            alpha=0.4,
            linewidth=1
        )

plt.xlabel("Tick")
plt.ylabel("Dinstance")
plt.title("Distance accumulation per run")

# ---- manual legend (robust & clean) ----
legend_elements = [
    Line2D([0], [0], color=palette["sensing"], lw=3, label="Sensing"),
    Line2D([0], [0], color=palette["non_sensing"], lw=3, label="Non-sensing")
]

plt.legend(handles=legend_elements, frameon=False)

plt.tight_layout()
plt.show()


# --- survival race plot
plt.figure(figsize=(8, 5))

for condition, df_c in df_summary.groupby("condition"):
    # vector of survival times
    survival_times = df_c["lifetime_ticks"].values

    max_t = survival_times.max()
    ticks = range(max_t + 1)

    alive = [
        (survival_times >= t).sum()
        for t in ticks
    ]

    plt.plot(
        ticks,
        alive,
        color=palette[condition],
        linewidth=3,
        label=condition.capitalize()
    )

plt.xlabel("Tick")
plt.ylabel("Bytes alive")
plt.title("Survival race of Byte simulations")

plt.legend(frameon=False)
plt.tight_layout()
plt.show()

# --- KS test
from scipy.stats import ks_2samp

# extract survival times
lifetimes_sensing = df_summary.loc[
    df_summary["condition"] == "sensing", "lifetime_ticks"
].values

lifetimes_non = df_summary.loc[
    df_summary["condition"] == "non_sensing", "lifetime_ticks"
].values

# KS test
ks_stat, p_value = ks_2samp(lifetimes_sensing, lifetimes_non)

print("KS statistic:", ks_stat)
print("p-value:", p_value)

import numpy as np

# --- log-log plot of survival durations
plt.figure(figsize=(8, 5))

for condition, df_c in df_summary.groupby("condition"):
    lifetimes = np.sort(df_c["lifetime_ticks"].values)

    # CCDF: P(T >= t)
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
        label=condition.capitalize()
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