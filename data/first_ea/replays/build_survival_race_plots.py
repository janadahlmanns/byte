"""
Build survival race buildup plots for presentation.

Generates 5 progressive figures that can be clicked through as a thumb-flip animation:
  0 – Random only, x range = max lifetime of Random data
  1 – Random + Hard-Coded, x range = max lifetime of both
  2 – Random + Hard-Coded, x range = 400
  3 – Random + Hard-Coded, x range = 600
  4 – Random + Hard-Coded + EA from Random, x range = max of all data, y max = 1000

Output: figures_first_ea/survival_race_buildup_0.png  …  survival_race_buildup_4.png
"""

# ==================================================================================================================================================
# IMPORTS
# ==================================================================================================================================================

import h5py
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ==================================================================================================================================================
# CONFIGURATION
# ==================================================================================================================================================

EXPERIMENT_NAME = "first_ea"

# Group colors (colorblind-friendly)
COLOR_MAP = {
    "Random":         "#56B4E9",
    "Hard-Coded":     "#0072B2",
    "EA from Random": "#E69F00",
}

# HDF5 file names (without .h5 extension) – expected next to this script
HDF5_FILES = {
    "Random":         "2026-05-05_17-29-22_random_genomes_all_runs_all",
    "Hard-Coded":     "2026-05-05_17-28-53_lookup_hard_genomes_all_runs_all",
    "EA from Random": "2026-05-05_17-42-25_ea_from_random_genomes_all_runs_all",
}

# Font sizes: triple the original defaults (~10 / 12 pt)
FONT_LABEL  = 30
FONT_TITLE  = 36
FONT_TICK   = 30
FONT_LEGEND = 30


# ==================================================================================================================================================
# HELPERS
# ==================================================================================================================================================

def _find_hdf5_file(hdf5_name: str) -> Path:
    """Locate HDF5 file by name in the same directory as this script."""
    search_dir = Path(__file__).resolve().parent
    hdf5_path = search_dir / f"{hdf5_name}.h5"
    if hdf5_path.exists():
        return hdf5_path
    # Try without adding extension (in case caller included it)
    hdf5_path = search_dir / hdf5_name
    if hdf5_path.exists():
        return hdf5_path
    raise FileNotFoundError(f"HDF5 file not found: {hdf5_name}  (searched in {search_dir})")


def _load_summary(hdf5_name: str, group_name: str) -> pd.DataFrame:
    """
    Load lifetime_ticks (and variant / group labels) from all variant summaries
    in an HDF5 file.  Returns a DataFrame with columns:
        group, variant (int), lifetime_ticks (float)
    """
    path = _find_hdf5_file(hdf5_name)
    parts = []
    with h5py.File(path, 'r') as f:
        variant_keys = sorted([k for k in f.keys() if k.startswith('variant_')])
        if not variant_keys:
            raise RuntimeError(f"No variant_ groups found in {path}")
        for vk in variant_keys:
            vg = f[vk]
            if 'summary' not in vg:
                continue
            df_var = pd.DataFrame(vg['summary'][:])
            df_var['variant'] = int(vk.split('_')[1])
            df_var['group'] = group_name
            parts.append(df_var)
    if not parts:
        raise ValueError(f"No summary data found in {hdf5_name}")
    df = pd.concat(parts, ignore_index=True)
    return df[['group', 'variant', 'lifetime_ticks']].copy()


# ==================================================================================================================================================
# LOAD DATA
# ==================================================================================================================================================

print("Loading data...")
df_random    = _load_summary(HDF5_FILES["Random"],         "Random")
df_hardcoded = _load_summary(HDF5_FILES["Hard-Coded"],     "Hard-Coded")
df_ea_random = _load_summary(HDF5_FILES["EA from Random"], "EA from Random")
print(f"  Random:         {len(df_random):>6} runs across {df_random['variant'].nunique()} variants")
print(f"  Hard-Coded:     {len(df_hardcoded):>6} runs across {df_hardcoded['variant'].nunique()} variants")
print(f"  EA from Random: {len(df_ea_random):>6} runs across {df_ea_random['variant'].nunique()} variants")


# ==================================================================================================================================================
# X-LIMIT HELPERS  (determined from data so plots align naturally)
# ==================================================================================================================================================

x_max_random    = int(df_random['lifetime_ticks'].max())
x_max_hardcoded = int(max(df_random['lifetime_ticks'].max(),
                          df_hardcoded['lifetime_ticks'].max()))
x_max_ea        = int(max(df_random['lifetime_ticks'].max(),
                          df_hardcoded['lifetime_ticks'].max(),
                          df_ea_random['lifetime_ticks'].max()))


# ==================================================================================================================================================
# PLOT CONFIGURATIONS
# ==================================================================================================================================================

# Each entry defines one output figure.
# 'groups_df' : DataFrame containing the data to draw (subset of loaded DataFrames)
# 'x_max'     : right limit of the x-axis (int)
# 'y_max'     : upper limit of the y-axis (int or None = auto)
# 'filename'  : output file name inside figures_first_ea/

PLOT_CONFIGS = [
    # 0 – Random only
    {
        'groups_df': df_random.copy(),
        'x_max':     x_max_random,
        'y_max':     None,
        'filename':  'survival_race_buildup_0.png',
    },
    # 1 – Random + Hard-Coded, x extends to accommodate Hard-Coded data
    {
        'groups_df': pd.concat([df_random, df_hardcoded], ignore_index=True),
        'x_max':     x_max_hardcoded,
        'y_max':     None,
        'filename':  'survival_race_buildup_1.png',
    },
    # 2 – Random + Hard-Coded, x = 400
    {
        'groups_df': pd.concat([df_random, df_hardcoded], ignore_index=True),
        'x_max':     400,
        'y_max':     None,
        'filename':  'survival_race_buildup_2.png',
    },
    # 3 – Random + Hard-Coded, x = 600
    {
        'groups_df': pd.concat([df_random, df_hardcoded], ignore_index=True),
        'x_max':     600,
        'y_max':     None,
        'filename':  'survival_race_buildup_3.png',
    },
    # 4 – All three groups, x extends to accommodate EA data, y max = 1000
    {
        'groups_df': pd.concat([df_random, df_hardcoded, df_ea_random], ignore_index=True),
        'x_max':     x_max_ea,
        'y_max':     1000,
        'filename':  'survival_race_buildup_4.png',
    },
]


# ==================================================================================================================================================
# GENERATE PLOTS
# ==================================================================================================================================================

figures_dir = Path(__file__).resolve().parent / f'figures_{EXPERIMENT_NAME}'
figures_dir.mkdir(exist_ok=True)

print("\nGenerating plots...")

for config in PLOT_CONFIGS:
    df      = config['groups_df']
    x_max   = config['x_max']
    y_max   = config['y_max']
    outfile = config['filename']

    # Preserve insertion order for consistent legend ordering
    all_groups = list(dict.fromkeys(df['group'].tolist()))

    fig, ax = plt.subplots(figsize=(12, 8))

    for (group_name, variant_id), grp in df.groupby(['group', 'variant'], sort=False):
        survival_times = grp['lifetime_ticks'].to_numpy(dtype=float)
        color = COLOR_MAP.get(group_name, '#808080')
        max_tick = int(survival_times.max())
        ticks = np.arange(0, max_tick + 1)
        alive = np.array([(survival_times >= t).sum() for t in ticks])
        ax.plot(ticks, alive, color=color)

    legend_elements = [
        Line2D([0], [0], color=COLOR_MAP.get(g, '#808080'), linewidth=2, label=g)
        for g in all_groups
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=FONT_LEGEND)

    ax.set_xlabel('Ticks', fontsize=FONT_LABEL)
    ax.set_ylabel('Runs Alive', fontsize=FONT_LABEL)
    ax.set_title('Survival Race', fontsize=FONT_TITLE)
    ax.tick_params(labelsize=FONT_TICK)
    ax.grid(True, alpha=0.3)

    # Origin at (0, 0) as required; x right limit set per config
    ax.set_xlim(0, x_max)
    if y_max is not None:
        ax.set_ylim(0, y_max)
    else:
        ax.set_ylim(bottom=0)

    output_path = figures_dir / outfile
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")

print("\nDone.")
