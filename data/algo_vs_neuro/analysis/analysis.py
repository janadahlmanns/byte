import pandas as pd
from pathlib import Path
# ---------------------------------------------------------------------
# INPUT PATH
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1] / "rawdata"

# ---------------------------------------------------------------------
# EXPERIMENT LABELS
# ---------------------------------------------------------------------
GROUP1_NAME = "Algorithmic Byte"
GROUP2_NAME = "Neuronal Byte"

group1_dir = BASE_DIR / "2025-12-30_21-59-39_algo"
group2_dir = BASE_DIR / "2025-12-30_21-59-20_neuro"


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

summary_group1, runs_group1 = load_condition(group1_dir, GROUP1_NAME)
summary_group2, runs_group2 = load_condition(group2_dir, GROUP2_NAME)


# ---------------------------------------------------------------------
# Final concatenation
# ---------------------------------------------------------------------
df_summary = pd.concat([summary_group1, summary_group2], ignore_index=True)
df_runs = pd.concat([runs_group1, runs_group2], ignore_index=True)


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
    GROUP1_NAME: "#0B3D2E",   # dark green
    GROUP2_NAME: "#D4AF37",   # gold
}

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
    Line2D([0], [0], color=palette[GROUP1_NAME], lw=3, label=GROUP1_NAME),
    Line2D([0], [0], color=palette[GROUP2_NAME], lw=3, label=GROUP2_NAME),
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


# --- KS test
from scipy.stats import ks_2samp

lifetimes_1 = df_summary.loc[
    df_summary["condition"] == GROUP1_NAME, "lifetime_ticks"
].values

lifetimes_2 = df_summary.loc[
    df_summary["condition"] == GROUP2_NAME, "lifetime_ticks"
].values

ks_stat, p_value = ks_2samp(lifetimes_1, lifetimes_2)

print("KS statistic:", ks_stat)
print("p-value:", p_value)


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
