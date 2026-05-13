"""
Analysis script for generating EA (Evolutionary Algorithm) simulation results report.

This script generates a Word document report with analysis of EA simulation data,
including statistical comparisons, visualizations, and descriptive statistics.
"""

# ==================================================================================================================================================
# SECTION A) IMPORTS AND INPUTS
# ==================================================================================================================================================

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from docx import Document
from scipy.stats import gaussian_kde
from scipy.integrate import trapezoid
from scipy.ndimage import gaussian_filter1d

# Add workspace root to path for imports
_current_path = Path(__file__).resolve()
_workspace_root = None
while _current_path.parent != _current_path:
    if (_current_path / "data").exists() and (_current_path / "simulate").exists():
        _workspace_root = _current_path
        break
    _current_path = _current_path.parent

if _workspace_root:
    sys.path.insert(0, str(_workspace_root))
    from analysis_tools.network_visualization.network_viz import plot_network_visualizations, load_network_viz_config

# =====================================================================
# User Configuration and Data Selection
# =====================================================================

# Experiment name
EXPERIMENT_NAME = "first_ea_from_lookup"  # Used for file naming and report titles

# HDF5 file containing EA variant data (omit .h5 extension)
EXPERIMENT_HDF5 = "2026-05-06_11-31-03_ea_from_lookup_hard"


# Benchmark data (optional): List of tuples (benchmark_display_name, hdf5_filename_without_extension)
# Leave as empty list [] if no benchmarks to compare
BENCHMARK_HDF5_FILES = [
    ("Soft-wired Lookup", "2026-05-05_17-28-28_lookup_soft"),
    ("Hard-wired Lookup", "2026-05-05_17-28-53_lookup_hard"),
    ("Random networks", "2026-05-05_17-29-22_random"),
]



# Color scheme for visualizations
PRIMARY_COLOR = "#0B3D2E"      # Dark green for best performing variants
SECONDARY_COLOR = "#8B3A3A"    # Wine red for worst performing variants
TERTIARY_COLOR = "#4A7C8C"     # Grayish ice blue for benchmarks
HIGHLIGHT_COLOR = "#D4AF37"     # Gold for highlights

# Build source_names dictionary for display in plots (initialized empty, populated after benchmark loading)
source_names = {}

# ==================================================================================================================================================
# SECTION B) HELPER FUNCTIONS
# ==================================================================================================================================================


def _find_hdf5_file(hdf5_name: str, search_dir: Path = None) -> Path:
    """
    Locate HDF5 file by name (with or without .h5 extension).
    
    Args:
        hdf5_name: Filename without .h5 extension
        search_dir: Directory to search in. If None, searches parent directory of script's directory
    
    Returns:
        Path to the HDF5 file
        
    Raises:
        FileNotFoundError: If file not found
    """
    if search_dir is None:
        # Script is in data/first_ea/analysis/, search in data/first_ea/
        search_dir = Path(__file__).resolve().parent.parent
    
    # Try with .h5 extension
    hdf5_path = search_dir / f"{hdf5_name}.h5"
    if hdf5_path.exists():
        return hdf5_path
    
    # Try without modification (in case user included extension)
    hdf5_path = search_dir / hdf5_name
    if hdf5_path.exists():
        return hdf5_path
    
    raise FileNotFoundError(f"HDF5 file not found: {hdf5_name} in {search_dir}")


def _load_ea_attributes(hdf5_path: Path) -> dict:
    """
    Load experiment parameters from HDF5 file top-level attributes.
    
    Args:
        hdf5_path: Path to HDF5 file
    
    Returns:
        Dictionary of attributes
    """
    with h5py.File(hdf5_path, 'r') as f:
        attrs = dict(f.attrs)
    return attrs


def _load_generation_stats(hdf5_path: Path) -> pd.DataFrame:
    """
    Load generation_stats dataset into DataFrame.
    
    Expected columns: generation, mean, median, min, max, std, iqr
    
    Args:
        hdf5_path: Path to HDF5 file
    
    Returns:
        DataFrame with generation statistics
    """
    with h5py.File(hdf5_path, 'r') as f:
        if 'generation_stats' not in f:
            raise ValueError("generation_stats dataset not found in HDF5 file")
        data = f['generation_stats'][:]
        df = pd.DataFrame(data)
    return df


def _load_elite_lifespans(hdf5_path: Path) -> pd.DataFrame:
    """
    Load elite_genomes/lifespans dataset into DataFrame.
    
    Args:
        hdf5_path: Path to HDF5 file
    
    Returns:
        DataFrame with elite lifespan data
    """
    with h5py.File(hdf5_path, 'r') as f:
        if 'elite_genomes/lifespans' not in f:
            raise ValueError("elite_genomes/lifespans dataset not found in HDF5 file")
        data = f['elite_genomes/lifespans'][:]
        df = pd.DataFrame(data)
    return df


def _load_elite_genomes(hdf5_path: Path) -> pd.DataFrame:
    """
    Load elite genome data into DataFrame.
    
    Each row represents one elite genome (elite_0, elite_1, etc.)
    Columns: elite_id (N), eta, and tonic_activation_0, tonic_activation_1, ...
    
    Args:
        hdf5_path: Path to HDF5 file
    
    Returns:
        DataFrame with elite genome data
    """
    all_data = []
    
    with h5py.File(hdf5_path, 'r') as f:
        if 'elite_genomes' not in f:
            raise ValueError("elite_genomes folder not found in HDF5 file")
        
        elite_group = f['elite_genomes']
        # Find all elite_N folders
        elite_folders = sorted([key for key in elite_group.keys() if key.startswith('elite_')])
        
        for elite_folder in elite_folders:
            try:
                elite_id = int(elite_folder.split('_')[1])
                folder = elite_group[elite_folder]
                
                # Load eta (it's stored as a 1D array with one element)
                if 'eta' not in folder:
                    continue
                eta_array = folder['eta'][:]
                eta = eta_array[0] if len(eta_array) > 0 else eta_array[()]
                
                # Load tonic_activations
                if 'tonic_activations' not in folder:
                    continue
                tonic_acts = folder['tonic_activations'][:]
                
                # Create row
                row = {'elite_id': elite_id, 'eta': eta}
                for i, val in enumerate(tonic_acts):
                    row[f'tonic_activation_{i}'] = val
                all_data.append(row)
            except Exception as e:
                continue
    
    if not all_data:
        raise ValueError(f"No elite genome data found in {hdf5_path}")
    
    df = pd.DataFrame(all_data)
    return df 


def _load_connection_weights(hdf5_path: Path) -> dict:
    """
    Load connection weights for all elite genomes from HDF5 file.
    
    Returns a dictionary mapping elite_id -> connection_weights array.
    
    Args:
        hdf5_path: Path to HDF5 file
    
    Returns:
        Dictionary {elite_id: connection_weights_array, ...}
    """
    connection_weights = {}
    
    with h5py.File(hdf5_path, 'r') as f:
        if 'elite_genomes' not in f:
            raise ValueError("elite_genomes folder not found in HDF5 file")
        
        elite_group = f['elite_genomes']
        # Find all elite_N folders
        elite_folders = sorted([key for key in elite_group.keys() if key.startswith('elite_')])
        
        for elite_folder in elite_folders:
            try:
                elite_id = int(elite_folder.split('_')[1])
                folder = elite_group[elite_folder]
                
                # Load connection_weights
                if 'connection_weights' not in folder:
                    continue
                
                weights = folder['connection_weights'][:]
                connection_weights[elite_id] = weights
            except Exception as e:
                continue
    
    return connection_weights


def _load_tonic_activations(hdf5_path: Path) -> dict:
    """
    Load tonic activations for all elite genomes from HDF5 file.
    
    Returns a dictionary mapping elite_id -> tonic_activations array.
    
    Args:
        hdf5_path: Path to HDF5 file
    
    Returns:
        Dictionary {elite_id: tonic_activations_array, ...}
    """
    tonic_activations = {}
    
    with h5py.File(hdf5_path, 'r') as f:
        if 'elite_genomes' not in f:
            return tonic_activations  # Return empty dict if no elite_genomes
        
        elite_group = f['elite_genomes']
        # Find all elite_N folders
        elite_folders = sorted([key for key in elite_group.keys() if key.startswith('elite_')])
        
        for elite_folder in elite_folders:
            try:
                elite_id = int(elite_folder.split('_')[1])
                folder = elite_group[elite_folder]
                
                # Load tonic_activations if available
                if 'tonic_activations' not in folder:
                    continue
                
                tonic_act = folder['tonic_activations'][:]
                tonic_activations[elite_id] = tonic_act
            except Exception as e:
                continue
    
    return tonic_activations


def _load_modulation_specs(hdf5_path: Path, source_label: str) -> pd.DataFrame:
    """
    Load modulation specifications for all elite genomes from HDF5 file.
    
    Each row represents one modulation entry (source neuron → target neuron, modulated by modulating_neuron).
    
    Args:
        hdf5_path: Path to HDF5 file
        source_label: Label to add as 'source' column (e.g., 'experiment', benchmark name)
    
    Returns:
        DataFrame with columns: source, elite, source, target, modulating_neuron, modulation_weight
        (Note: 'source' at front is the experiment/benchmark label)
    """
    all_data = []
    
    with h5py.File(hdf5_path, 'r') as f:
        if 'elite_genomes' not in f:
            raise ValueError("elite_genomes folder not found in HDF5 file")
        
        elite_group = f['elite_genomes']
        # Find all elite_N folders
        elite_folders = sorted([key for key in elite_group.keys() if key.startswith('elite_')])
        
        for elite_folder in elite_folders:
            try:
                elite_id = int(elite_folder.split('_')[1])
                folder = elite_group[elite_folder]
                
                # Load modulation_spec
                if 'modulation_spec' not in folder:
                    continue
                
                mod_data = folder['modulation_spec'][:]
                
                # Convert structured array to DataFrame
                df_mod = pd.DataFrame(mod_data)
                
                # Add elite identifier
                df_mod['elite'] = elite_id
                
                # Add source label
                df_mod['source'] = source_label
                
                all_data.append(df_mod)
            except Exception as e:
                continue
    
    if not all_data:
        # Return empty DataFrame with correct structure
        return pd.DataFrame(columns=['source', 'elite', 'source_neuron', 'target_neuron', 'modulating_neuron', 'modulation_weight'])
    
    df = pd.concat(all_data, ignore_index=True)
    
    # Reorder columns: source at front, then elite, then the modulation columns
    # Get the modulation columns (exclude 'source' and 'elite')
    mod_cols = [col for col in df.columns if col not in ['source', 'elite']]
    df = df[['source', 'elite'] + mod_cols]
    
    return df


def _compute_effective_connection_weights(hdf5_path: Path) -> pd.DataFrame:
    """
    Compute effective connection weights from elite genomes.
    
    For each elite, loads connectivity data with shape (11, 11, 2) where:
    - Layer 0: connection weights
    - Layer 1: reliability/confidence values
    
    Effective weight = connection_weight × reliability for each connection.
    
    Args:
        hdf5_path: Path to HDF5 file with elite_genomes dataset
    
    Returns:
        DataFrame with one row per elite and columns for each effective weight value
    """
    all_elites_data = []
    
    with h5py.File(hdf5_path, 'r') as f:
        if 'elite_genomes' not in f:
            return pd.DataFrame()
        
        elite_genomes = f['elite_genomes']
        
        # Process each elite (skip non-group items like datasets)
        for elite_key in sorted(elite_genomes.keys()):
            elite_group = elite_genomes[elite_key]
            
            # Skip if not a group (could be datasets like 'lifespans', 'run_seeds')
            if not isinstance(elite_group, h5py.Group):
                continue
            
            if 'connection_weights' not in list(elite_group.keys()):
                continue
            
            # Load connection weights: shape (11, 11, 2)
            # Layer 0: weights, Layer 1: reliability
            conn_weights_data = elite_group['connection_weights'][()]
            
            # Extract layers
            weights = conn_weights_data[:, :, 0]  # Shape (11, 11)
            reliability = conn_weights_data[:, :, 1]  # Shape (11, 11)
            
            # Compute effective weights: weight × reliability
            effective_weights = weights * reliability
            
            # Flatten and store as individual columns
            flat_effective_weights = effective_weights.flatten()
            row_data = {f'weight_{i}': w for i, w in enumerate(flat_effective_weights)}
            all_elites_data.append(row_data)
    
    if not all_elites_data:
        return pd.DataFrame()
    
    # Convert to DataFrame with proper column alignment
    df = pd.DataFrame(all_elites_data)
    
    return df


def _compute_raw_connection_weights(hdf5_path: Path) -> pd.DataFrame:
    """
    Extract raw connection weights (without reliability) from elite genomes.
    
    For each elite, loads connectivity data with shape (11, 11, 2) where:
    - Layer 0: connection weights
    - Layer 1: reliability/confidence values (ignored here)
    
    Args:
        hdf5_path: Path to HDF5 file with elite_genomes dataset
    
    Returns:
        DataFrame with one row per elite and columns for each weight value
    """
    all_elites_data = []
    
    with h5py.File(hdf5_path, 'r') as f:
        if 'elite_genomes' not in f:
            return pd.DataFrame()
        
        elite_genomes = f['elite_genomes']
        
        # Process each elite (skip non-group items like datasets)
        for elite_key in sorted(elite_genomes.keys()):
            elite_group = elite_genomes[elite_key]
            
            # Skip if not a group (could be datasets like 'lifespans', 'run_seeds')
            if not isinstance(elite_group, h5py.Group):
                continue
            
            if 'connection_weights' not in list(elite_group.keys()):
                continue
            
            # Load connection weights: shape (11, 11, 2)
            # Layer 0: weights, Layer 1: reliability
            conn_weights_data = elite_group['connection_weights'][()]
            
            # Extract weights layer only
            weights = conn_weights_data[:, :, 0]  # Shape (11, 11)
            
            # Flatten and store as individual columns
            flat_weights = weights.flatten()
            row_data = {f'weight_{i}': w for i, w in enumerate(flat_weights)}
            all_elites_data.append(row_data)
    
    if not all_elites_data:
        return pd.DataFrame()
    
    # Convert to DataFrame with proper column alignment
    df = pd.DataFrame(all_elites_data)
    
    return df


def _compute_raw_reliability_values(hdf5_path: Path) -> pd.DataFrame:
    """
    Extract raw reliability values (confidence/connection strength) from elite genomes.
    
    For each elite, loads connectivity data with shape (11, 11, 2) where:
    - Layer 0: connection weights (ignored here)
    - Layer 1: reliability/confidence values
    
    Args:
        hdf5_path: Path to HDF5 file with elite_genomes dataset
    
    Returns:
        DataFrame with one row per elite and columns for each reliability value
    """
    all_elites_data = []
    
    with h5py.File(hdf5_path, 'r') as f:
        if 'elite_genomes' not in f:
            return pd.DataFrame()
        
        elite_genomes = f['elite_genomes']
        
        # Process each elite (skip non-group items like datasets)
        for elite_key in sorted(elite_genomes.keys()):
            elite_group = elite_genomes[elite_key]
            
            # Skip if not a group (could be datasets like 'lifespans', 'run_seeds')
            if not isinstance(elite_group, h5py.Group):
                continue
            
            if 'connection_weights' not in list(elite_group.keys()):
                continue
            
            # Load connection weights: shape (11, 11, 2)
            # Layer 0: weights, Layer 1: reliability
            conn_weights_data = elite_group['connection_weights'][()]
            
            # Extract reliability layer only
            reliability = conn_weights_data[:, :, 1]  # Shape (11, 11)
            
            # Flatten and store as individual columns
            flat_reliability = reliability.flatten()
            row_data = {f'reliability_{i}': r for i, r in enumerate(flat_reliability)}
            all_elites_data.append(row_data)
    
    if not all_elites_data:
        return pd.DataFrame()
    
    # Convert to DataFrame with proper column alignment
    df = pd.DataFrame(all_elites_data)
    
    return df


def _compute_modulation_weights(hdf5_path: Path) -> pd.DataFrame:
    """
    Extract modulation weights from elite genomes' modulation specifications.
    
    For each elite, loads modulation_spec dataset and extracts the modulation_weight column.
    
    Args:
        hdf5_path: Path to HDF5 file with elite_genomes dataset
    
    Returns:
        DataFrame with one row per elite and columns for each modulation weight value
    """
    all_elites_data = []
    
    with h5py.File(hdf5_path, 'r') as f:
        if 'elite_genomes' not in f:
            return pd.DataFrame()
        
        elite_genomes = f['elite_genomes']
        
        # Process each elite (skip non-group items like datasets)
        for elite_key in sorted(elite_genomes.keys()):
            elite_group = elite_genomes[elite_key]
            
            # Skip if not a group (could be datasets like 'lifespans', 'run_seeds')
            if not isinstance(elite_group, h5py.Group):
                continue
            
            if 'modulation_spec' not in list(elite_group.keys()):
                continue
            
            # Load modulation_spec dataset
            modulation_spec = elite_group['modulation_spec'][()]
            
            # Extract modulation_weight column if it exists
            # modulation_spec is typically a structured array with named fields
            if 'modulation_weight' in modulation_spec.dtype.names:
                mod_weights = modulation_spec['modulation_weight']
                # Flatten and store as individual columns
                row_data = {f'modulation_weight_{i}': w for i, w in enumerate(mod_weights)}
                all_elites_data.append(row_data)
    
    if not all_elites_data:
        return pd.DataFrame()
    
    # Convert to DataFrame with proper column alignment
    df = pd.DataFrame(all_elites_data)
    
    return df


def plot_ea_results(hdf5_path, doc, figures_dir):
    """
    Plot generation statistics from HDF5 file and add to Word document.
    
    Displays a plot with mean±std and median±IQR shading, plus min/max lines.
    Saves the figure to a PNG file and adds it to the report.
    
    Args:
        hdf5_path: Path to HDF5 file with generation_stats dataset
        doc: python-docx Document object to add the figure to
        figures_dir: Path to directory where figure PNG files are saved
    """
    # Load generation stats from HDF5
    with h5py.File(hdf5_path, 'r') as f:
        gen_stats_data = f["generation_stats"][:]
    
    # Extract columns
    generations = gen_stats_data['generation']
    mean_vals = gen_stats_data['mean']
    median_vals = gen_stats_data['median']
    min_vals = gen_stats_data['min']
    max_vals = gen_stats_data['max']
    std_vals = gen_stats_data['std']
    iqr_vals = gen_stats_data['iqr']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot mean with std shading (primary)
    ax.fill_between(generations, mean_vals - std_vals, mean_vals + std_vals, 
                    alpha=0.3, color=PRIMARY_COLOR, label='Mean with std')
    ax.plot(generations, mean_vals, '-', linewidth=2.5, color=PRIMARY_COLOR)
    
    # Plot median with IQR shading (secondary)
    ax.fill_between(generations, median_vals - iqr_vals/2, median_vals + iqr_vals/2, 
                    alpha=0.3, color=SECONDARY_COLOR, label='Median with IQR')
    ax.plot(generations, median_vals, '-', linewidth=2.5, color=SECONDARY_COLOR)
    
    # Plot min and max (tertiary)
    ax.plot(generations, min_vals, '--', linewidth=2, color=TERTIARY_COLOR, label='Min and max')
    ax.plot(generations, max_vals, '--', linewidth=2, color=TERTIARY_COLOR)
    
    ax.set_xlabel('Generations', fontsize=12)
    ax.set_ylabel('Lifespan [ticks]', fontsize=12)
    ax.set_title('Lifespan Across Generations', fontsize=14)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    # Save to file
    figures_dir.mkdir(exist_ok=True)
    figure_path = figures_dir / 'ea_lifespan_generations.png'
    figure_path_abs = figure_path.resolve()
    fig.savefig(str(figure_path_abs), dpi=150, bbox_inches='tight')
    
    fig.tight_layout()
    
    # Add to report
    doc.add_picture(str(figure_path_abs), width=6.5 * 914400)
    doc.add_paragraph()
    
    # Close figure
    plt.close(fig)


def plot_distribution(df_data, metric_name, doc, figures_dir, filename_base=None, bin_count=30, source_names=None):
    """
    Plot a generic distribution as normalized histograms with lines and uncertainty shading.
    
    Plots data similarly to plot_lifespan_distributions but is metric-agnostic.
    Handles both single sources and multiple sources (via 'source' column).
    
    Args:
        df_data: DataFrame with data to plot. Expected columns:
                 - 'source' (optional): experiment or benchmark identifier
                 - numeric columns: data values to plot
        metric_name: Name of the metric for x-axis label and plot title
        doc: python-docx Document object to add figure to
        figures_dir: Path to directory where figure PNG files are saved
        filename_base: Base name for saved PNG (default: metric_name with underscores)
        bin_count: Number of bins for histogram (default 30)
        source_names: Optional dict mapping source key to display name (default: None)
    """
    if filename_base is None:
        filename_base = metric_name.lower().replace(' ', '_')
    
    if source_names is None:
        source_names = {}
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get columns for data (all numeric columns, exclude 'source' if present)
    data_cols = [col for col in df_data.columns if col != 'source']
    
    # Collect all data to determine global bin range
    all_data = df_data[data_cols].values.flatten()
    all_data = np.asarray(pd.to_numeric(all_data, errors='coerce'))
    all_data = all_data[~np.isnan(all_data)]
    
    if len(all_data) == 0:
        print(f"No data to plot for {metric_name}")
        return
    
    # Create bins from min to max (handles negative values)
    data_min = np.min(all_data)
    data_max = np.max(all_data)
    bins_global = np.linspace(data_min, data_max, bin_count)
    bin_centers_global = (bins_global[:-1] + bins_global[1:]) / 2
    
    # Check if 'source' column exists
    has_source = 'source' in df_data.columns
    
    if not has_source:
        # Single source: compute histogram and plot
        hist_counts_per_row = []
        for idx, row in df_data.iterrows():
            data_values = row[data_cols].values
            data_values = np.asarray(pd.to_numeric(data_values, errors='coerce'))
            data_values = data_values[~np.isnan(data_values)]
            if len(data_values) < 2:
                continue
            
            counts, _ = np.histogram(data_values, bins=bins_global, density=False)
            counts = counts / len(data_values)
            hist_counts_per_row.append(counts)
        
        if hist_counts_per_row:
            hist_counts_per_row = np.array(hist_counts_per_row)
            mean_counts = hist_counts_per_row.mean(axis=0)
            std_counts = hist_counts_per_row.std(axis=0)
            
            # Apply subtle smoothing
            mean_counts_smooth = gaussian_filter1d(mean_counts, sigma=1.0)
            std_counts_smooth = gaussian_filter1d(std_counts, sigma=1.0)
            
            # Add origin point (0, 0)
            bin_centers_plot = np.concatenate([[0], bin_centers_global])
            mean_counts_plot = np.concatenate([[0], mean_counts_smooth])
            std_counts_plot = np.concatenate([[0], std_counts_smooth])
            
            ax.plot(bin_centers_plot, mean_counts_plot, label='Result', color=PRIMARY_COLOR, linewidth=2)
            ax.fill_between(bin_centers_plot, mean_counts_plot - std_counts_plot, mean_counts_plot + std_counts_plot, 
                           alpha=0.2, color=PRIMARY_COLOR)
    else:
        # Multiple sources: plot experiment first with primary color, then benchmarks
        benchmark_colors = [SECONDARY_COLOR, TERTIARY_COLOR, HIGHLIGHT_COLOR]
        unique_sources = sorted(df_data['source'].unique())
        
        # Separate experiment from benchmarks and plot experiment first
        if 'experiment' in unique_sources:
            sources_to_plot = ['experiment'] + [s for s in unique_sources if s != 'experiment']
        else:
            sources_to_plot = unique_sources
        
        for plot_idx, source_name in enumerate(sources_to_plot):
            # Determine color: primary for experiment, benchmarks get the other colors
            if source_name == 'experiment':
                color = PRIMARY_COLOR
            else:
                bench_idx = plot_idx - 1 if 'experiment' in unique_sources else plot_idx
                color = benchmark_colors[bench_idx % len(benchmark_colors)]
            
            source_data = df_data[df_data['source'] == source_name]
            
            # Collect data for this source
            all_source_data = source_data[data_cols].values.flatten()
            all_source_data = np.asarray(pd.to_numeric(all_source_data, errors='coerce'))
            all_source_data = all_source_data[~np.isnan(all_source_data)]
            
            if len(all_source_data) == 0:
                continue
            
            # Create bins specific to this source (from min to max to handle negatives)
            source_min = np.min(all_source_data)
            source_max = np.max(all_source_data)
            bins_source = np.linspace(source_min, source_max, bin_count)
            bin_centers_source = (bins_source[:-1] + bins_source[1:]) / 2
            
            # Compute histograms
            hist_counts_per_row = []
            for row_id, row in source_data.iterrows():
                data_values = row[data_cols].values
                data_values = np.asarray(pd.to_numeric(data_values, errors='coerce'))
                data_values = data_values[~np.isnan(data_values)]
                if len(data_values) < 2:
                    continue
                
                counts, _ = np.histogram(data_values, bins=bins_source, density=False)
                counts = counts / len(data_values)
                hist_counts_per_row.append(counts)
            
            if hist_counts_per_row:
                hist_counts_per_row = np.array(hist_counts_per_row)
                mean_counts = hist_counts_per_row.mean(axis=0)
                std_counts = hist_counts_per_row.std(axis=0)
                
                # Apply subtle smoothing
                mean_counts_smooth = gaussian_filter1d(mean_counts, sigma=1.0)
                std_counts_smooth = gaussian_filter1d(std_counts, sigma=1.0)
                
                # Add origin point (0, 0)
                bin_centers_plot = np.concatenate([[0], bin_centers_source])
                mean_counts_plot = np.concatenate([[0], mean_counts_smooth])
                std_counts_plot = np.concatenate([[0], std_counts_smooth])
                
                ax.plot(bin_centers_plot, mean_counts_plot, label=source_names.get(source_name, source_name), 
                       color=color, linewidth=2)
                ax.fill_between(bin_centers_plot, mean_counts_plot - std_counts_plot, mean_counts_plot + std_counts_plot, 
                               alpha=0.2, color=color)
    
    ax.set_xlabel(metric_name, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'{metric_name} Distribution', fontsize=14)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Set x-axis limits to span from min to max (handles negative values)
    ax.set_xlim(data_min, data_max)
    
    fig.tight_layout()
    
    # Save to file
    figures_dir.mkdir(exist_ok=True)
    figure_path = figures_dir / f'{filename_base}_distribution.png'
    fig.savefig(str(figure_path), dpi=150, bbox_inches='tight')
    
    # Add to report
    doc.add_picture(str(figure_path), width=6.5 * 914400)
    doc.add_paragraph()
    
    # Close figure
    plt.close(fig)


def plot_weight_distribution(df_data, metric_name, doc, figures_dir, filename_base=None, bin_count=30, source_names=None):
    """
    Plot weight distribution as normalized histograms with lines and uncertainty shading.
    
    Specialized version of plot_distribution() for connection weight data with additional
    features specific to weight analysis. Generates two plots: one excluding zeros and one including them.
    Handles both single sources and multiple sources (via 'source' column).
    
    Special handling for weights:
    - Zero values are included in frequency calculations
    - Bin range includes negative weights (from min to max)
    - Generates two visualizations: (1) zeros filtered out, (2) zeros included
    
    Args:
        df_data: DataFrame with data to plot. Expected columns:
                 - 'source' (optional): experiment or benchmark identifier
                 - numeric columns: data values to plot
        metric_name: Name of the metric for x-axis label and plot title
        doc: python-docx Document object to add figure to
        figures_dir: Path to directory where figure PNG files are saved
        filename_base: Base name for saved PNG (default: metric_name with underscores)
        bin_count: Number of bins for histogram (default 30)
        source_names: Optional dict mapping source key to display name (default: None)
    """
    if filename_base is None:
        filename_base = metric_name.lower().replace(' ', '_')
    
    if source_names is None:
        source_names = {}
    
    # Get columns for data (all numeric columns, exclude 'source' if present)
    data_cols = [col for col in df_data.columns if col != 'source']
    
    # Collect all data to determine global bin range (including negatives)
    all_data = df_data[data_cols].values.flatten()
    all_data = np.asarray(pd.to_numeric(all_data, errors='coerce'))
    all_data = all_data[~np.isnan(all_data)]
    
    if len(all_data) == 0:
        print(f"No data to plot for {metric_name}")
        return
    
    # Create bins from min to max to include negative weights
    data_min = np.min(all_data)
    data_max = np.max(all_data)
    bins_global = np.linspace(data_min, data_max, bin_count)
    bin_centers_global = (bins_global[:-1] + bins_global[1:]) / 2
    
    # Check if 'source' column exists
    has_source = 'source' in df_data.columns
    
    # Generate two plots: one without zeros, one with zeros
    for include_zeros in [False, True]:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if not has_source:
            # Single source: compute histogram and plot
            hist_counts_per_row = []
            for idx, row in df_data.iterrows():
                data_values = row[data_cols].values
                data_values = np.asarray(pd.to_numeric(data_values, errors='coerce'))
                data_values = data_values[~np.isnan(data_values)]
                if len(data_values) < 2:
                    continue
                
                # Keep original length for normalization (includes zeros)
                orig_len = len(data_values)
                
                if not include_zeros:
                    # Remove zeros from histogram computation
                    data_values_plot = data_values[data_values != 0]
                else:
                    # Keep all values including zeros
                    data_values_plot = data_values
                
                # Histogram from filtered or unfiltered values
                counts, _ = np.histogram(data_values_plot, bins=bins_global, density=False)
                # Normalize by original length (which includes zeros) for proper frequency
                counts = counts / orig_len
                hist_counts_per_row.append(counts)
            
            if hist_counts_per_row:
                hist_counts_per_row = np.array(hist_counts_per_row)
                mean_counts = hist_counts_per_row.mean(axis=0)
                std_counts = hist_counts_per_row.std(axis=0)
                
                # Apply subtle smoothing
                mean_counts_smooth = gaussian_filter1d(mean_counts, sigma=1.0)
                std_counts_smooth = gaussian_filter1d(std_counts, sigma=1.0)
                
                if not include_zeros:
                    # Filter out zero bin centers for plotting
                    non_zero_mask = bin_centers_global != 0
                    bin_centers_plot = bin_centers_global[non_zero_mask]
                    mean_counts_plot = mean_counts_smooth[non_zero_mask]
                    std_counts_plot = std_counts_smooth[non_zero_mask]
                else:
                    # Use all bins including zero
                    bin_centers_plot = bin_centers_global
                    mean_counts_plot = mean_counts_smooth
                    std_counts_plot = std_counts_smooth
                
                ax.plot(bin_centers_plot, mean_counts_plot, label='Result', color=PRIMARY_COLOR, linewidth=2)
                ax.fill_between(bin_centers_plot, mean_counts_plot - std_counts_plot, mean_counts_plot + std_counts_plot, 
                               alpha=0.2, color=PRIMARY_COLOR)
        else:
            # Multiple sources: plot experiment first with primary color, then benchmarks
            benchmark_colors = [SECONDARY_COLOR, TERTIARY_COLOR, HIGHLIGHT_COLOR]
            unique_sources = sorted(df_data['source'].unique())
            
            # Separate experiment from benchmarks and plot experiment first
            if 'experiment' in unique_sources:
                sources_to_plot = ['experiment'] + [s for s in unique_sources if s != 'experiment']
            else:
                sources_to_plot = unique_sources
            
            for plot_idx, source_name in enumerate(sources_to_plot):
                # Determine color: primary for experiment, benchmarks get the other colors
                if source_name == 'experiment':
                    color = PRIMARY_COLOR
                else:
                    bench_idx = plot_idx - 1 if 'experiment' in unique_sources else plot_idx
                    color = benchmark_colors[bench_idx % len(benchmark_colors)]
                
                source_data = df_data[df_data['source'] == source_name]
                
                # Collect data for this source
                all_source_data = source_data[data_cols].values.flatten()
                all_source_data = np.asarray(pd.to_numeric(all_source_data, errors='coerce'))
                all_source_data = all_source_data[~np.isnan(all_source_data)]
                
                if len(all_source_data) == 0:
                    continue
                
                # Create bins specific to this source (from min to max to include negatives)
                source_min = np.min(all_source_data)
                source_max = np.max(all_source_data)
                bins_source = np.linspace(source_min, source_max, bin_count)
                bin_centers_source = (bins_source[:-1] + bins_source[1:]) / 2
                
                # Compute histograms
                hist_counts_per_row = []
                for row_id, row in source_data.iterrows():
                    data_values = row[data_cols].values
                    data_values = np.asarray(pd.to_numeric(data_values, errors='coerce'))
                    data_values = data_values[~np.isnan(data_values)]
                    if len(data_values) < 2:
                        continue
                    
                    # Keep original length for normalization (includes zeros)
                    orig_len = len(data_values)
                    
                    if not include_zeros:
                        # Remove zeros from histogram computation
                        data_values_plot = data_values[data_values != 0]
                    else:
                        # Keep all values including zeros
                        data_values_plot = data_values
                    
                    # Histogram from filtered or unfiltered values
                    counts, _ = np.histogram(data_values_plot, bins=bins_source, density=False)
                    # Normalize by original length (which includes zeros) for proper frequency
                    counts = counts / orig_len
                    hist_counts_per_row.append(counts)
                
                if hist_counts_per_row:
                    hist_counts_per_row = np.array(hist_counts_per_row)
                    mean_counts = hist_counts_per_row.mean(axis=0)
                    std_counts = hist_counts_per_row.std(axis=0)
                    
                    # Apply subtle smoothing
                    mean_counts_smooth = gaussian_filter1d(mean_counts, sigma=1.0)
                    std_counts_smooth = gaussian_filter1d(std_counts, sigma=1.0)
                    
                    if not include_zeros:
                        # Filter out zero bin centers for plotting
                        non_zero_mask = bin_centers_source != 0
                        bin_centers_plot = bin_centers_source[non_zero_mask]
                        mean_counts_plot = mean_counts_smooth[non_zero_mask]
                        std_counts_plot = std_counts_smooth[non_zero_mask]
                    else:
                        # Use all bins including zero
                        bin_centers_plot = bin_centers_source
                        mean_counts_plot = mean_counts_smooth
                        std_counts_plot = std_counts_smooth
                    
                    ax.plot(bin_centers_plot, mean_counts_plot, label=source_names.get(source_name, source_name), 
                           color=color, linewidth=2)
                    ax.fill_between(bin_centers_plot, mean_counts_plot - std_counts_plot, mean_counts_plot + std_counts_plot, 
                                   alpha=0.2, color=color)
        
        ax.set_xlabel(metric_name, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        
        # Adjust title based on whether zeros are included
        zeros_label = "with Zeros" if include_zeros else "Excluding Zeros"
        ax.set_title(f'{metric_name} Distribution ({zeros_label})', fontsize=14)
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Set x-axis limits to span from min to max
        ax.set_xlim(data_min, data_max)
        
        fig.tight_layout()
        
        # Save to file with appropriate suffix
        figures_dir.mkdir(exist_ok=True)
        suffix = "with_zeros" if include_zeros else "without_zeros"
        figure_path = figures_dir / f'{filename_base}_{suffix}_distribution.png'
        fig.savefig(str(figure_path), dpi=150, bbox_inches='tight')
        
        # Add to report
        doc.add_picture(str(figure_path), width=6.5 * 914400)
        doc.add_paragraph()
        
        # Close figure
        plt.close(fig)


def plot_lifespan_distributions(df_elite_lifespans_exp, df_benchmarks_elite_lifespans, doc, figures_dir):
    """
    Plot normalized lifespan distributions in the final elite generation as line plots.
    
    For the experiment, plots separate lines for each elite genome.
    For benchmarks, plots averaged lines with uncertainty shading for each benchmark source.
    
    Args:
        df_elite_lifespans_exp: DataFrame with experiment elite lifespans (one row per elite, one col per run)
        df_benchmarks_elite_lifespans: Optional DataFrame with benchmark lifespans (includes 'source' column)
        doc: python-docx Document object to add the figure to
        figures_dir: Path to directory where figure PNG files are saved
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get columns for experiment (all numeric columns, exclude 'source' if present)
    exp_cols = [col for col in df_elite_lifespans_exp.columns if col != 'source']
    
    # Collect experiment lifespans to determine experiment bin range
    all_lifespans_exp = df_elite_lifespans_exp[exp_cols].values.flatten()
    all_lifespans_exp = np.asarray(pd.to_numeric(all_lifespans_exp, errors='coerce'))
    all_lifespans_exp = all_lifespans_exp[~np.isnan(all_lifespans_exp)]
    
    if len(all_lifespans_exp) == 0:
        print("No lifespan data to plot")
        return
    
    # Plot experiment distributions (average of all elites with uncertainty)
    bins_exp = np.linspace(0, np.nanmax(all_lifespans_exp), 30)
    bin_centers_exp = (bins_exp[:-1] + bins_exp[1:]) / 2
    
    hist_counts_per_elite = []
    for elite_idx, elite_row in df_elite_lifespans_exp.iterrows():
        lifespans = elite_row[exp_cols].values
        lifespans = np.asarray(pd.to_numeric(lifespans, errors='coerce'))
        lifespans = lifespans[~np.isnan(lifespans)]
        if len(lifespans) < 2:
            continue
        
        # Compute histogram and normalize by count
        counts, _ = np.histogram(lifespans, bins=bins_exp, density=False)
        counts = counts / len(lifespans)
        hist_counts_per_elite.append(counts)
    
    if hist_counts_per_elite:
        hist_counts_per_elite = np.array(hist_counts_per_elite)
        mean_counts = hist_counts_per_elite.mean(axis=0)
        std_counts = hist_counts_per_elite.std(axis=0)
        
        # Apply subtle smoothing
        mean_counts_smooth = gaussian_filter1d(mean_counts, sigma=1.0)
        std_counts_smooth = gaussian_filter1d(std_counts, sigma=1.0)
        
        # Add origin point (0, 0)
        bin_centers_plot = np.concatenate([[0], bin_centers_exp])
        mean_counts_plot = np.concatenate([[0], mean_counts_smooth])
        std_counts_plot = np.concatenate([[0], std_counts_smooth])
        
        ax.plot(bin_centers_plot, mean_counts_plot, label='EA result', color=PRIMARY_COLOR, linewidth=2)
        ax.fill_between(bin_centers_plot, mean_counts_plot - std_counts_plot, mean_counts_plot + std_counts_plot, 
                       alpha=0.2, color=PRIMARY_COLOR)
    
    # Plot benchmark distributions
    if df_benchmarks_elite_lifespans is not None and len(df_benchmarks_elite_lifespans) > 0:
        benchmark_colors = [SECONDARY_COLOR, TERTIARY_COLOR, HIGHLIGHT_COLOR]
        bench_cols = [col for col in df_benchmarks_elite_lifespans.columns if col != 'source']
        
        unique_sources = sorted(df_benchmarks_elite_lifespans['source'].unique())
        
        for idx, bench_name in enumerate(unique_sources):
            bench_data = df_benchmarks_elite_lifespans[df_benchmarks_elite_lifespans['source'] == bench_name]
            
            # Collect benchmark-specific lifespans
            all_lifespans_bench = bench_data[bench_cols].values.flatten()
            all_lifespans_bench = np.asarray(pd.to_numeric(all_lifespans_bench, errors='coerce'))
            all_lifespans_bench = all_lifespans_bench[~np.isnan(all_lifespans_bench)]
            
            if len(all_lifespans_bench) == 0:
                continue
            
            # Create bins specific to this benchmark
            bins_bench = np.linspace(0, np.nanmax(all_lifespans_bench), 30)
            bin_centers_bench = (bins_bench[:-1] + bins_bench[1:]) / 2
            
            # Compute histogram for each elite and collect bin counts
            hist_counts_per_elite = []
            for elite_id, elite_row in bench_data.iterrows():
                lifespans = elite_row[bench_cols].values
                lifespans = np.asarray(pd.to_numeric(lifespans, errors='coerce'))
                lifespans = lifespans[~np.isnan(lifespans)]
                if len(lifespans) < 2:
                    continue
                
                # Compute histogram and normalize by count
                counts, _ = np.histogram(lifespans, bins=bins_bench, density=False)
                counts = counts / len(lifespans)
                hist_counts_per_elite.append(counts)
            
            if hist_counts_per_elite:
                hist_counts_per_elite = np.array(hist_counts_per_elite)
                mean_counts = hist_counts_per_elite.mean(axis=0)
                std_counts = hist_counts_per_elite.std(axis=0)
                
                # Apply subtle smoothing
                mean_counts_smooth = gaussian_filter1d(mean_counts, sigma=1.0)
                std_counts_smooth = gaussian_filter1d(std_counts, sigma=1.0)
                
                # Add origin point (0, 0)
                bin_centers_plot = np.concatenate([[0], bin_centers_bench])
                mean_counts_plot = np.concatenate([[0], mean_counts_smooth])
                std_counts_plot = np.concatenate([[0], std_counts_smooth])
                
                color = benchmark_colors[idx % len(benchmark_colors)]
                ax.plot(bin_centers_plot, mean_counts_plot, label=bench_name, color=color, linewidth=2)
                ax.fill_between(bin_centers_plot, mean_counts_plot - std_counts_plot, mean_counts_plot + std_counts_plot, 
                               alpha=0.2, color=color)
            else:
                print(f"No valid histogram data for benchmark: {bench_name}")
    
    ax.set_xlabel('Lifespan [ticks]', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Lifespan Distribution in Final Elite', fontsize=14)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Set x-axis limit to global maximum for consistent comparison
    all_max = np.nanmax(all_lifespans_exp)
    if df_benchmarks_elite_lifespans is not None and len(df_benchmarks_elite_lifespans) > 0:
        bench_cols_global = [col for col in df_benchmarks_elite_lifespans.columns if col != 'source']
        bench_lifespans_global = df_benchmarks_elite_lifespans[bench_cols_global].values.flatten()
        bench_lifespans_global = np.asarray(pd.to_numeric(bench_lifespans_global, errors='coerce'))
        bench_lifespans_global = bench_lifespans_global[~np.isnan(bench_lifespans_global)]
        all_max = max(all_max, np.nanmax(bench_lifespans_global))
    
    ax.set_xlim(0, all_max)
    ax.axvline(x=16, linestyle='--', color='gray', alpha=0.7)
    
    fig.tight_layout()
    
    # Save to file
    figures_dir.mkdir(exist_ok=True)
    figure_path = figures_dir / 'lifespan_distribution_final_elite.png'
    fig.savefig(str(figure_path), dpi=150, bbox_inches='tight')
    
    # Add to report
    doc.add_picture(str(figure_path), width=6.5 * 914400)
    doc.add_paragraph()
    
    # Close figure
    plt.close(fig)


def summarize_parameter(doc, experiment_attrs, experiment_filename, benchmark_attrs=None, benchmark_hdf5_files=None):
    """
    Add experiment and benchmark parameters summary to the Word document.
    
    Creates formatted tables showing selected parameters loaded from HDF5 attributes,
    plus the filenames used.
    
    Args:
        doc: python-docx Document object to add tables to
        experiment_attrs: Dictionary of experiment attributes from HDF5
        experiment_filename: String of experiment HDF5 filename (without .h5)
        benchmark_attrs: Optional dict mapping benchmark name to attributes dict
        benchmark_hdf5_files: Optional list of tuples (benchmark_name, hdf5_filename)
    """
    # Build benchmark filenames mapping
    benchmark_filenames_map = {name: filename for name, filename in benchmark_hdf5_files} if benchmark_hdf5_files else {}
    
    # Define which attributes to include (actual HDF5 attribute names from data)
    attributes_to_include = {
        'brain_max_decision_delay',
        'brain_n_neurons',
        'brain_noise_level',
        'experiment_evolutionary_algorithm_elite_selection_metric',
        'experiment_evolutionary_algorithm_elite_size',
        'experiment_evolutionary_algorithm_mutation_method',
        'experiment_evolutionary_algorithm_mutation_rate',
        'experiment_evolutionary_algorithm_num_generations',
        'experiment_genome_type',
        'experiment_max_ticks',
        'experiment_n_runs',
        'experiment_population_size',
        'food_feeding_paradigm_initial',
        'food_feeding_paradigm_regrow',
        'food_initial_fraction_per_cell',
        'food_regrow_time',
        'worm_decisionmaking_version',
        'worm_energy_capacity',
        'worm_metabolic_rate',
        'worm_movement_cost'
    }
    
    def filter_attributes(attrs_dict):
        """Filter attributes to only include those in the list (case-insensitive)."""
        filtered = {}
        attrs_lower = {k.lower(): (k, v) for k, v in attrs_dict.items()}
        for attr_name in attributes_to_include:
            if attr_name.lower() in attrs_lower:
                original_key, value = attrs_lower[attr_name.lower()]
                filtered[original_key] = value
        return filtered
    
    # Add experiment parameters section
    doc.add_heading('Experiment Parameters', level=2)
    
    # Filter experiment attributes
    filtered_exp_attrs = filter_attributes(experiment_attrs)
    
    # Create table for experiment parameters (filename + filtered attributes)
    num_rows = len(filtered_exp_attrs) + 2  # +2 for header and filename row
    exp_table = doc.add_table(rows=num_rows, cols=2)
    exp_table.style = 'Light Grid Accent 1'
    
    # Header row
    exp_table.rows[0].cells[0].text = 'Parameter'
    exp_table.rows[0].cells[1].text = 'Value'
    
    # Filename row
    exp_table.rows[1].cells[0].text = 'Filename'
    exp_table.rows[1].cells[1].text = experiment_filename
    
    # Parameters from filtered attributes (sorted by key)
    for row_idx, (key, value) in enumerate(sorted(filtered_exp_attrs.items()), 2):
        exp_table.rows[row_idx].cells[0].text = str(key)
        exp_table.rows[row_idx].cells[1].text = str(value)
    
    doc.add_paragraph()
    
    # Add benchmark parameters if available
    if benchmark_attrs:
        for bench_name in sorted(benchmark_attrs.keys()):
            bench_attrs_dict = benchmark_attrs[bench_name]
            bench_filename = benchmark_filenames_map.get(bench_name, 'Unknown')
            
            # Filter benchmark attributes
            filtered_bench_attrs = filter_attributes(bench_attrs_dict)
            
            # Add heading for this benchmark
            doc.add_heading(f'Benchmark: {bench_name} Parameters', level=2)
            
            # Create table for benchmark parameters
            num_rows_bench = len(filtered_bench_attrs) + 2  # +2 for header and filename row
            bench_table = doc.add_table(rows=num_rows_bench, cols=2)
            bench_table.style = 'Light Grid Accent 1'
            
            # Header row
            bench_table.rows[0].cells[0].text = 'Parameter'
            bench_table.rows[0].cells[1].text = 'Value'
            
            # Filename row
            bench_table.rows[1].cells[0].text = 'Filename'
            bench_table.rows[1].cells[1].text = bench_filename
            
            # Parameters from filtered attributes (sorted by key)
            for row_idx, (key, value) in enumerate(sorted(filtered_bench_attrs.items()), 2):
                bench_table.rows[row_idx].cells[0].text = str(key)
                bench_table.rows[row_idx].cells[1].text = str(value)
            
            doc.add_paragraph()


# ==================================================================================================================================================
# SECTION C) DATA LOADING
# ==================================================================================================================================================


experiment_hdf5_path = _find_hdf5_file(EXPERIMENT_HDF5)

# Load experiment attributes
experiment_attrs = _load_ea_attributes(experiment_hdf5_path)


# Load experiment generation stats
df_generation_stats_exp = _load_generation_stats(experiment_hdf5_path)


# Load experiment elite lifespans
df_elite_lifespans_exp = _load_elite_lifespans(experiment_hdf5_path)

# Load experiment elite genomes
df_elite_genomes_exp = _load_elite_genomes(experiment_hdf5_path)


# Load experiment connection weights
connection_weights_experiment = _load_connection_weights(experiment_hdf5_path)

# Load experiment tonic activations
tonic_activations_experiment = _load_tonic_activations(experiment_hdf5_path)

# Compute effective connection weights for experiment
df_effective_cw_exp = _compute_effective_connection_weights(experiment_hdf5_path)

# Compute raw connection weights for experiment
df_raw_weights_exp = _compute_raw_connection_weights(experiment_hdf5_path)

# Compute raw reliability values for experiment
df_raw_reliability_exp = _compute_raw_reliability_values(experiment_hdf5_path)

# Compute modulation weights for experiment
df_modulation_weights_exp = _compute_modulation_weights(experiment_hdf5_path)

print("\nExperiment data loaded.")

# Load benchmark data if specified
df_benchmarks_generations_stats = None
df_benchmarks_elite_lifespans = None
df_benchmarks_elite_genomes = None
df_benchmarks_modulation_specs = None
df_benchmarks_effective_cw = None
df_benchmarks_raw_weights = None
df_benchmarks_raw_reliability = None
df_benchmarks_modulation_weights = None
benchmark_attrs = {}
connection_weights_collection = {}  # Will collect all connection_weights structures
tonic_activations_collection = {}  # Will collect all tonic_activations structures

if BENCHMARK_HDF5_FILES:
    for bench_name, bench_hdf5 in BENCHMARK_HDF5_FILES:
        try:
            bench_path = _find_hdf5_file(bench_hdf5)
            print(f"\nLoading benchmark '{bench_name}'...")
            
            # Load benchmark attributes
            bench_attrs = _load_ea_attributes(bench_path)
            benchmark_attrs[bench_name] = bench_attrs

            
            # Load benchmark generation stats
            df_gen_stats = _load_generation_stats(bench_path)
            df_gen_stats['source'] = bench_name
            if df_benchmarks_generations_stats is None:
                df_benchmarks_generations_stats = df_gen_stats
            else:
                df_benchmarks_generations_stats = pd.concat([df_benchmarks_generations_stats, df_gen_stats], ignore_index=True)

            
            # Load benchmark elite lifespans
            df_lifespans = _load_elite_lifespans(bench_path)
            df_lifespans['source'] = bench_name
            if df_benchmarks_elite_lifespans is None:
                df_benchmarks_elite_lifespans = df_lifespans
            else:
                df_benchmarks_elite_lifespans = pd.concat([df_benchmarks_elite_lifespans, df_lifespans], ignore_index=True)
            
            # Load benchmark elite genomes
            df_genomes = _load_elite_genomes(bench_path)
            df_genomes['source'] = bench_name
            if df_benchmarks_elite_genomes is None:
                df_benchmarks_elite_genomes = df_genomes
            else:
                df_benchmarks_elite_genomes = pd.concat([df_benchmarks_elite_genomes, df_genomes], ignore_index=True)
            
            # Load benchmark connection weights
            # Create a sanitized name for the dictionary key (replace spaces with underscores)
            bench_key = bench_name.replace(" ", "_").replace("-", "_").lower()
            connection_weights_key = f"connection_weights_{bench_key}"
            connection_weights_data = _load_connection_weights(bench_path)
            connection_weights_collection[connection_weights_key] = connection_weights_data
            
            # Load benchmark tonic activations
            tonic_activations_key = f"tonic_activations_{bench_key}"
            tonic_activations_data = _load_tonic_activations(bench_path)
            tonic_activations_collection[tonic_activations_key] = tonic_activations_data
            
            # Compute effective connection weights for this benchmark
            df_eff_cw = _compute_effective_connection_weights(bench_path)
            df_eff_cw['source'] = bench_name
            if df_benchmarks_effective_cw is None:
                df_benchmarks_effective_cw = df_eff_cw
            else:
                df_benchmarks_effective_cw = pd.concat([df_benchmarks_effective_cw, df_eff_cw], ignore_index=True)
            
            # Compute raw connection weights for this benchmark
            df_raw_w = _compute_raw_connection_weights(bench_path)
            df_raw_w['source'] = bench_name
            if df_benchmarks_raw_weights is None:
                df_benchmarks_raw_weights = df_raw_w
            else:
                df_benchmarks_raw_weights = pd.concat([df_benchmarks_raw_weights, df_raw_w], ignore_index=True)
            
            # Compute raw reliability values for this benchmark
            df_raw_r = _compute_raw_reliability_values(bench_path)
            df_raw_r['source'] = bench_name
            if df_benchmarks_raw_reliability is None:
                df_benchmarks_raw_reliability = df_raw_r
            else:
                df_benchmarks_raw_reliability = pd.concat([df_benchmarks_raw_reliability, df_raw_r], ignore_index=True)
            
            # Compute modulation weights for this benchmark
            df_mod_w = _compute_modulation_weights(bench_path)
            df_mod_w['source'] = bench_name
            if df_benchmarks_modulation_weights is None:
                df_benchmarks_modulation_weights = df_mod_w
            else:
                df_benchmarks_modulation_weights = pd.concat([df_benchmarks_modulation_weights, df_mod_w], ignore_index=True)
            
        except FileNotFoundError as e:
            pass
        except Exception as e:
            pass
else:
    print("No benchmarks specified.")

print("\nAll HDF5 data loaded.")

# Create figures directory
figures_dir = Path(__file__).resolve().parent / 'figures'

# Load network visualization configuration for 11-neuron network
try:
    _, neuron_types = load_network_viz_config(11)
except Exception as e:
    print(f"Warning: Could not load network visualization config: {e}")
    neuron_types = {}

# Populate source_names dictionary with experiment and benchmark names
source_names['experiment'] = EXPERIMENT_NAME
for bench_name in benchmark_attrs.keys():
    # Use the benchmark name directly as stored in benchmark_attrs
    source_names[bench_name] = bench_name

# ==================================================================================================================================================
# SECTION D) ANALYSIS
# ==================================================================================================================================================

doc = Document()
doc.add_heading(f"EA Analysis Report: {EXPERIMENT_NAME}", level=0)

# region Overview

# Call parameter summary function
summarize_parameter(doc, experiment_attrs, EXPERIMENT_HDF5, benchmark_attrs, BENCHMARK_HDF5_FILES)

# endregion Overview
# region EA Results / Fitness

# Plot and add first figure
plot_ea_results(experiment_hdf5_path, doc, figures_dir)

# plot lifespan distributions
plot_lifespan_distributions(df_elite_lifespans_exp, df_benchmarks_elite_lifespans, doc, figures_dir)

# endregion EA Results / Fitness
# region connectivity

doc.add_heading('Network Visualizations', level=2)
plot_network_visualizations(
    connection_weights_experiment, 
    tonic_activations_experiment,
    connection_weights_collection, 
    tonic_activations_collection,
    neuron_types, 
    source_names, 
    doc, 
    figures_dir
)

# add histogram of effective connection weights
if df_effective_cw_exp is not None and not df_effective_cw_exp.empty:
    # Combine experiment and benchmark effective connection weights for plotting
    df_plot_data = df_effective_cw_exp.copy()
    df_plot_data['source'] = 'experiment'
    
    if df_benchmarks_effective_cw is not None and not df_benchmarks_effective_cw.empty:
        df_plot_data = pd.concat([df_plot_data, df_benchmarks_effective_cw], ignore_index=True)
    
    plot_weight_distribution(df_plot_data, 'Effective Connection Weights', doc, figures_dir, 
                            filename_base='effective_connection_weights', source_names=source_names, bin_count = 40)

# add histogram of raw connection weights
if df_raw_weights_exp is not None and not df_raw_weights_exp.empty:
    # Combine experiment and benchmark raw weights for plotting
    df_plot_data = df_raw_weights_exp.copy()
    df_plot_data['source'] = 'experiment'
    
    if df_benchmarks_raw_weights is not None and not df_benchmarks_raw_weights.empty:
        df_plot_data = pd.concat([df_plot_data, df_benchmarks_raw_weights], ignore_index=True)
    
    plot_weight_distribution(df_plot_data, 'Raw Connection Weights', doc, figures_dir, 
                            filename_base='raw_connection_weights', source_names=source_names, bin_count = 40)

# add histogram of raw reliability values
if df_raw_reliability_exp is not None and not df_raw_reliability_exp.empty:
    # Combine experiment and benchmark raw reliability for plotting
    df_plot_data = df_raw_reliability_exp.copy()
    df_plot_data['source'] = 'experiment'
    
    if df_benchmarks_raw_reliability is not None and not df_benchmarks_raw_reliability.empty:
        df_plot_data = pd.concat([df_plot_data, df_benchmarks_raw_reliability], ignore_index=True)
    
    plot_weight_distribution(df_plot_data, 'Connection Reliability', doc, figures_dir, 
                            filename_base='connection_reliability', source_names=source_names, bin_count = 40)


# endregion connectivity
# region plasticity

# add histogram of modulation weights
if df_modulation_weights_exp is not None and not df_modulation_weights_exp.empty:
    # Combine experiment and benchmark modulation weights for plotting
    df_plot_data = df_modulation_weights_exp.copy()
    df_plot_data['source'] = 'experiment'
    
    if df_benchmarks_modulation_weights is not None and not df_benchmarks_modulation_weights.empty:
        df_plot_data = pd.concat([df_plot_data, df_benchmarks_modulation_weights], ignore_index=True)
    
    plot_distribution(df_plot_data, 'Modulation Weights', doc, figures_dir, 
                     filename_base='modulation_weights', source_names=source_names, bin_count = 40)

# endregion plasticity





# ==================================================================================================================================================
# SAVE REPORT
# ==================================================================================================================================================



output_dir = Path(__file__).resolve().parent
output_path = output_dir / f'results_{EXPERIMENT_NAME}.docx'

doc.save(str(output_path))
print(f"Report saved to: {output_path}")


