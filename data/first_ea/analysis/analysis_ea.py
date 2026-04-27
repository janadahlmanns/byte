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
import random
import time
from docx import Document
from typing import Dict, List, Tuple, Optional

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
    # from analysis_tools.network_visualization import network_viz

# Import network visualization module
try:
    from analysis_tools.network_visualization.network_viz import draw_network, compute_layout
except ImportError:
    print("Warning: network_viz module not available. Network visualizations will be skipped.")
    draw_network = None
    compute_layout = None

# =====================================================================
# User Configuration and Data Selection
# =====================================================================

# Experiment name
EXPERIMENT_NAME = "slow_mutation"  # Used for file naming and report titles

# HDF5 file containing EA variant data (omit .h5 extension)
EXPERIMENT_HDF5 = "2026-04-22_20-57-56_slow_mutation"

# Benchmark data (optional): List of tuples (benchmark_display_name, hdf5_filename_without_extension)
# Leave as empty list [] if no benchmarks to compare
BENCHMARK_HDF5_FILES = [
    ("Hard-wired Lookup", "2026-04-22_20-24-58_lookup"),
    ("Random networks", "2026-04-22_20-38-37_random"),
]



# Color scheme for visualizations
PRIMARY_COLOR = "#0B3D2E"      # Dark green for best performing variants
SECONDARY_COLOR = "#8B3A3A"    # Wine red for worst performing variants
TERTIARY_COLOR = "#4A7C8C"     # Grayish ice blue for benchmarks
HIGHLIGHT_COLOR = "#D4AF37"     # Gold for highlights

# Color palette for different experiments/benchmarks (uses random if more than 4)
COLOR_PALETTE = [PRIMARY_COLOR, SECONDARY_COLOR, TERTIARY_COLOR, "#F0E2E7"]

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


def extract_neuron_types(hdf5_attrs: dict) -> Dict[int, str]:
    """
    Extract neuron types (input, output, hidden) from HDF5 attributes.
    
    Parses:
    - brain_output_mapping_N attributes: N is the output neuron ID
    - brain_sensory_mapping_* attributes: first element in JSON array is input neuron ID
    
    Args:
        hdf5_attrs: Dictionary of HDF5 top-level attributes
    
    Returns:
        Dict mapping neuron_id -> type ('input', 'output', or 'hidden')
    """
    import json
    neuron_types = {}
    
    # Extract output neurons from brain_output_mapping_N attributes
    for key, value in hdf5_attrs.items():
        if key.startswith('brain_output_mapping_'):
            # Extract the neuron ID from the key name
            try:
                neuron_id = int(key.replace('brain_output_mapping_', ''))
                neuron_types[neuron_id] = 'output'
            except ValueError:
                continue
    
    # Extract input neurons from brain_sensory_mapping_* attributes
    for key, value in hdf5_attrs.items():
        if key.startswith('brain_sensory_mapping_'):
            # Value is stored as a JSON string (e.g., "[2, 0.8, 0.9]")
            try:
                if isinstance(value, str):
                    # Parse JSON string
                    arr = json.loads(value)
                    if isinstance(arr, (list, tuple)) and len(arr) > 0:
                        neuron_id = int(arr[0])
                        neuron_types[neuron_id] = 'input'
                elif isinstance(value, (tuple, list)) and len(value) > 0:
                    neuron_id = int(value[0])
                    neuron_types[neuron_id] = 'input'
                elif isinstance(value, (int, np.integer)):
                    neuron_id = int(value)
                    neuron_types[neuron_id] = 'input'
            except (ValueError, TypeError, IndexError, json.JSONDecodeError):
                continue
    
    return neuron_types


def collect_lifespan_data(experiment_name: str, df_elite_lifespans_exp: pd.DataFrame, df_benchmarks_elite_lifespans: pd.DataFrame = None) -> pd.DataFrame:
    """
    Collect all lifespan data from experiment and benchmarks into a unified dataframe.
    
    Args:
        experiment_name: Name of the experiment
        df_elite_lifespans_exp: DataFrame with experiment elite lifespan data
        df_benchmarks_elite_lifespans: Optional DataFrame with benchmark elite lifespan data
    
    Returns:
        DataFrame with columns: [experiment, elite_ID, run, lifespan]
    """
    lifespan_data = []
    
    # Extract experiment elite lifespans
    exp_cols = [col for col in df_elite_lifespans_exp.columns if col != 'source']
    
    for elite_idx, row in df_elite_lifespans_exp.iterrows():
        for run_idx, col in enumerate(exp_cols):
            lifespan = row[col]
            lifespan_data.append({
                'experiment': experiment_name,
                'elite_ID': elite_idx,
                'run': run_idx,
                'lifespan': lifespan
            })
    
    # Extract benchmark elite lifespans if available
    if df_benchmarks_elite_lifespans is not None:
        benchmark_names_in_data = df_benchmarks_elite_lifespans['source'].unique() if 'source' in df_benchmarks_elite_lifespans.columns else []
        
        for bench_name in benchmark_names_in_data:
            bench_data = df_benchmarks_elite_lifespans[df_benchmarks_elite_lifespans['source'] == bench_name]
            bench_cols = [col for col in bench_data.columns if col not in ['source']]
            
            for elite_idx, row in bench_data.iterrows():
                for run_idx, col in enumerate(bench_cols):
                    lifespan = row[col]
                    lifespan_data.append({
                        'experiment': bench_name,
                        'elite_ID': elite_idx,
                        'run': run_idx,
                        'lifespan': lifespan
                    })
    
    return pd.DataFrame(lifespan_data)


def _close_word_document(filepath: Path) -> None:
    """
    Close a Word document if it's currently open in Microsoft Word.
    
    Prompts the user with a "Save changes?" dialog if the document has been modified.
    Silently closes without prompting if no changes were made.
    Does nothing if Word is not running or the document is not open.
    
    Args:
        filepath: Path to the Word document (.docx file)
    """
    try:
        import win32com.client
        
        try:
            # Get the running Word application
            word_app = win32com.client.GetObject(Class="Word.Application")
            
            # Search through all open documents
            for doc in word_app.Documents:
                # Compare full paths to ensure we match the right document
                if str(filepath.resolve()) in doc.FullName or doc.FullName in str(filepath.resolve()):
                    print(f"Closing Word document: {doc.FullName}")
                    # Close without saving (value 0 = wdDoNotSaveChanges)
                    doc.Close(0)
                    # Wait a moment for Word to release the file
                    time.sleep(1.0)
                    print("Document closed successfully")
                    return
        except Exception as e:
            print(f"Could not close Word document: {e}")
            pass
    except ImportError:
        print("pywin32 not installed - skipping Word close")
        pass


def plot_ea_results(df_generation_stats, doc, figures_dir):
    """
    Plot generation statistics and add to Word document.
    
    Displays a plot with mean±std and median±IQR shading, plus min/max lines.
    Saves the figure to a PNG file and adds it to the report.
    
    Args:
        df_generation_stats: DataFrame with generation statistics (columns: generation, mean, median, min, max, std, iqr)
        doc: python-docx Document object to add the figure to
        figures_dir: Path to directory where figure PNG files are saved
    """
    # Extract columns from dataframe
    generations = df_generation_stats['generation']
    mean_vals = df_generation_stats['mean']
    median_vals = df_generation_stats['median']
    min_vals = df_generation_stats['min']
    max_vals = df_generation_stats['max']
    std_vals = df_generation_stats['std']
    iqr_vals = df_generation_stats['iqr']
    
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


def plot_fitness_distribution(df_lifespan_data, doc, figures_dir):
    """
    Plot distribution of elite lifespans across runs for experiment and benchmarks.
    
    Args:
        df_lifespan_data: DataFrame with lifespan data (columns: experiment, elite_ID, run, lifespan)
        doc: python-docx Document object to add the figure to
        figures_dir: Path to directory where figure PNG files are saved
    """
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Group by experiment and elite_ID, then plot histogram for each group
    grouped = df_lifespan_data.groupby(['experiment', 'elite_ID'])
    
    # Track which experiments we've already added to legend
    legend_experiments_done = set()
    
    for (experiment, elite_id), group_data in grouped:
        lifespans = group_data['lifespan'].values
        
        # Compute histogram with 30 bins
        counts, bin_edges = np.histogram(lifespans, bins=30)
        # Convert bin edges to bin centers for plotting
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Determine color for this experiment
        color = experiment_colors[experiment]
        
        # Only add label to legend once per experiment
        if experiment not in legend_experiments_done:
            ax.plot(bin_centers, counts, '-', linewidth=2.5, color=color, label=experiment, alpha=0.8)
            legend_experiments_done.add(experiment)
        else:
            ax.plot(bin_centers, counts, '-', linewidth=2.5, color=color, alpha=0.8)
    
    ax.set_xlabel('Lifespan [ticks]', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Fitness Across Individual Runs', fontsize=14)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    # Save to file
    figures_dir.mkdir(exist_ok=True)
    figure_path = figures_dir / 'ea_fitness_distributions.png'
    figure_path_abs = figure_path.resolve()
    fig.savefig(str(figure_path_abs), dpi=150, bbox_inches='tight')
    
    # Add to report
    doc.add_picture(str(figure_path_abs), width=6.5 * 914400)
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


def convert_weights_to_network_format(weights_matrix: np.ndarray,
                                      neuron_types: Optional[Dict[int, str]] = None) -> Tuple[Dict, List[Dict]]:
    """
    Convert connection weight matrix to format expected by draw_network.
    
    Handles both 2D (NxN) and 3D (NxNx2) weight matrices.
    For 3D format: [:, :, 0] = connection weights, [:, :, 1] = reliability
    Computes effective_weight = weight * reliability
    
    Args:
        weights_matrix: NxN or NxNx2 connection weight matrix
        neuron_types: Optional dict mapping neuron_id -> type (default: inferred as 'hidden')
    
    Returns:
        (neurons, connections_list) tuple ready for draw_network()
        - neurons: Dict mapping neuron_id -> {'pos': (x, y), 'type': ...}
        - connections_list: List of dicts with 'src', 'tgt', 'weight' (effective weight)
    """
    # Extract weight and reliability matrices
    if weights_matrix.ndim == 3:
        W = weights_matrix[:, :, 0]  # Connection weights
        R = weights_matrix[:, :, 1]  # Reliability
    else:
        W = weights_matrix
        R = np.ones_like(W)  # Default reliability = 1.0 if not provided
    
    n = W.shape[0]
    
    if neuron_types is None:
        neuron_types = {}
    
    # Extract all non-zero connections (based on raw weight)
    connections_list = []
    for i in range(n):
        for j in range(n):
            weight = W[i, j]
            reliability = R[i, j]
            
            # Only include if raw weight is non-zero
            if weight != 0:
                # Calculate effective weight
                effective_weight = weight * reliability
                connections_list.append({
                    'src': i,
                    'tgt': j,
                    'weight': float(effective_weight)
                })
    
    # Auto-compute positions using network structure
    positions = {}
    if connections_list and compute_layout is not None:
        try:
            connections_df = pd.DataFrame(connections_list)
            positions = compute_layout(connections_df, use_graphviz=False)
        except Exception as e:
            print(f"Warning: compute_layout failed: {e}. Using circular layout.")
            # Fallback to circular layout
            for i in range(n):
                angle = 2 * np.pi * i / n
                positions[i] = (np.cos(angle), np.sin(angle))
    else:
        # Fallback to circular if no connections or compute_layout not available
        for i in range(n):
            angle = 2 * np.pi * i / n
            positions[i] = (np.cos(angle), np.sin(angle))
    
    # Build neurons dict
    neurons = {}
    for nid in range(n):
        neurons[nid] = {
            'pos': positions[nid],
            'type': neuron_types.get(nid, 'hidden')
        }
    
    return neurons, connections_list


def compute_topology_metrics(adjacency_matrix: np.ndarray) -> dict:
    """
    Compute network topology metrics from adjacency matrix.
    
    Handles both 2D (NxN) and 3D (NxNx2) weight matrices.
    For 3D format: [:, :, 0] = connection weights, [:, :, 1] = reliability (ignored)
    Assumes weights convention: positive = excitatory, negative = inhibitory
    
    Args:
        adjacency_matrix: NxN or NxNx2 connection weight matrix
    
    Returns:
        Dictionary with topology metrics (both binary and weighted)
    """
    try:
        W = np.asarray(adjacency_matrix, dtype=float)
        
        # Extract weight layer, skip reliability layer
        if W.ndim == 3 and W.shape[2] == 2:
            W = W[:, :, 0]  # Use only weight layer, ignore reliability
        elif W.ndim == 3:
            # Unexpected 3D shape, try reshaping
            W = W.reshape(W.shape[0], -1)
        
        # Now W should be 2D
        if W.ndim != 2:
            raise ValueError(f"Expected 2D weight matrix after layer extraction, got {W.ndim}D")
        
        n = W.shape[0]
        max_possible_connections = n * n
        
        # Separate excitatory (positive) and inhibitory (negative) weights
        W_exc = np.clip(W, 0, None)      # Positive weights only
        W_inh = np.abs(np.clip(W, None, 0))  # Absolute value of negative weights
        
        # Binary metrics: Count non-zero connections
        total_exc_connections = np.sum(W_exc > 0)
        total_inh_connections = np.sum(W_inh > 0)
        
        # Connectivity degree: actual connections / possible connections (n*n)
        exc_out_degree = total_exc_connections / max_possible_connections
        inh_out_degree = total_inh_connections / max_possible_connections
        
        # Weighted metrics: Sum of connection strengths normalized by n²
        exc_sum_weights = np.sum(W_exc)
        inh_sum_weights = np.sum(W_inh)
        exc_weighted_degree = exc_sum_weights / max_possible_connections
        inh_weighted_degree = inh_sum_weights / max_possible_connections
        
        # Mean connection strength (only non-zero connections)
        exc_mean_strength = np.mean(W_exc[W_exc > 0]) if total_exc_connections > 0 else 0.0
        inh_mean_strength = np.mean(W_inh[W_inh > 0]) if total_inh_connections > 0 else 0.0
        
        # Self connections (diagonal)
        diag_exc = W_exc.diagonal()
        diag_inh = W_inh.diagonal()
        self_exc = np.sum(diag_exc > 0)
        self_inh = np.sum(diag_inh > 0)
        
        # Bidirectional connections
        exc_bidirectional = np.sum((W_exc > 0) & (W_exc.T > 0)) / 2
        inh_bidirectional = np.sum((W_inh > 0) & (W_inh.T > 0)) / 2
        
        metrics = {
            # Binary metrics
            'exc_out_degree': float(exc_out_degree),
            'inh_out_degree': float(inh_out_degree),
            # Weighted metrics
            'exc_weighted_degree': float(exc_weighted_degree),
            'inh_weighted_degree': float(inh_weighted_degree),
            'exc_mean_strength': float(exc_mean_strength),
            'inh_mean_strength': float(inh_mean_strength),
            # Other metrics
            'self_connection_ratio': float((self_exc + self_inh) / n),
            'exc_bidirectional_ratio': float(exc_bidirectional / total_exc_connections) if total_exc_connections > 0 else 0.0,
            'inh_bidirectional_ratio': float(inh_bidirectional / total_inh_connections) if total_inh_connections > 0 else 0.0,
            'num_neurons': n,
        }
        return metrics
    except Exception as e:
        print(f"Error computing topology metrics: {e}")
        return {
            'exc_out_degree': 0.0,
            'exc_in_degree': 0.0,
            'inh_out_degree': 0.0,
            'inh_in_degree': 0.0,
            'exc_weighted_degree': 0.0,
            'inh_weighted_degree': 0.0,
            'exc_mean_strength': 0.0,
            'inh_mean_strength': 0.0,
            'self_connection_ratio': 0.0,
            'exc_bidirectional_ratio': 0.0,
            'inh_bidirectional_ratio': 0.0,
            'num_neurons': 0,
        }


def plot_topology_connectivity(df_topology, doc, figures_dir, experiment_colors):
    """
    Plot binary connectivity degree (excitatory and inhibitory) as grouped bar charts.
    
    Groups bars by source (experiment/benchmark), with pairs of bars per elite.
    Shows actual number of connections (not normalized).
    
    Args:
        df_topology: DataFrame with topology metrics
        doc: python-docx Document object to add figure to
        figures_dir: Path to directory for saving figure PNG
        experiment_colors: Dict mapping source name to color
    """
    if df_topology.empty:
        return
    
    # Sort by source and elite_id
    df_sorted = df_topology.sort_values(['source', 'elite_id'])
    
    # Get the number of neurons (same for all networks)
    n_neurons = int(df_sorted.iloc[0]['num_neurons'])
    n_squared = n_neurons * n_neurons
    
    # Create x-axis labels and data
    x_labels = []
    exc_out_degrees = []
    inh_out_degrees = []
    colors = []
    
    for idx, row in df_sorted.iterrows():
        source = row['source']
        elite_id = row['elite_id']
        x_labels.append(f"{source}\nelite_{int(elite_id)}")
        # Un-normalize: convert from degree (connections/n²) back to actual connection count
        exc_out_degrees.append(row['exc_out_degree'] * n_squared)
        inh_out_degrees.append(row['inh_out_degree'] * n_squared)
        colors.append(experiment_colors.get(source, '#808080'))
    
    x = np.arange(len(x_labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Create bars
    bars1 = ax.bar(x - width/2, exc_out_degrees, width, label='Excitatory Out-Degree', 
                   alpha=0.8, color=colors)
    bars2 = ax.bar(x + width/2, inh_out_degrees, width, label='Inhibitory Out-Degree', 
                   alpha=0.6, color=colors)
    
    ax.set_xlabel('Source and Elite', fontsize=12)
    ax.set_ylabel(f'Connectivity Degree [number of connections, {n_squared} possible]', fontsize=12)
    ax.set_title('Binary Network Connectivity: Number of Connections', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.tight_layout()
    
    # Save to file
    figures_dir.mkdir(exist_ok=True)
    figure_path = figures_dir / 'topology_connectivity_degrees.png'
    figure_path_abs = figure_path.resolve()
    fig.savefig(str(figure_path_abs), dpi=150, bbox_inches='tight')
    
    # Add to report
    doc.add_picture(str(figure_path_abs), width=6.5 * 914400)
    doc.add_paragraph()
    
    plt.close(fig)


def plot_topology_weighted_connectivity(df_topology, doc, figures_dir, experiment_colors):
    """
    Plot weighted connectivity degree (excitatory and inhibitory) as grouped bar charts.
    
    Groups bars by source (experiment/benchmark), with pairs of bars per elite.
    Shows total connection strength (weighted measure considering connection weights).
    
    Args:
        df_topology: DataFrame with topology metrics
        doc: python-docx Document object to add figure to
        figures_dir: Path to directory for saving figure PNG
        experiment_colors: Dict mapping source name to color
    """
    if df_topology.empty:
        return
    
    # Sort by source and elite_id
    df_sorted = df_topology.sort_values(['source', 'elite_id'])
    
    # Create x-axis labels and data
    x_labels = []
    exc_weighted_degrees = []
    inh_weighted_degrees = []
    colors = []
    
    for idx, row in df_sorted.iterrows():
        source = row['source']
        elite_id = row['elite_id']
        x_labels.append(f"{source}\nelite_{int(elite_id)}")
        exc_weighted_degrees.append(row['exc_weighted_degree'])
        inh_weighted_degrees.append(row['inh_weighted_degree'])
        colors.append(experiment_colors.get(source, '#808080'))
    
    x = np.arange(len(x_labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Create bars
    bars1 = ax.bar(x - width/2, exc_weighted_degrees, width, label='Excitatory Weighted', 
                   alpha=0.8, color=colors)
    bars2 = ax.bar(x + width/2, inh_weighted_degrees, width, label='Inhibitory Weighted', 
                   alpha=0.6, color=colors)
    
    ax.set_xlabel('Source and Elite', fontsize=12)
    ax.set_ylabel('Weighted Connectivity (sum of strengths / n²)', fontsize=12)
    ax.set_title('Weighted Network Connectivity: Total Connection Strength', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.tight_layout()
    
    # Save to file
    figures_dir.mkdir(exist_ok=True)
    figure_path = figures_dir / 'topology_weighted_connectivity.png'
    figure_path_abs = figure_path.resolve()
    fig.savefig(str(figure_path_abs), dpi=150, bbox_inches='tight')
    
    # Add to report
    doc.add_picture(str(figure_path_abs), width=6.5 * 914400)
    doc.add_paragraph()
    
    plt.close(fig)


def plot_topology_mean_strength(df_topology, doc, figures_dir, experiment_colors):
    """
    Plot mean connection strength (excitatory and inhibitory) as grouped bar charts.
    
    Groups bars by source (experiment/benchmark), with pairs of bars per elite.
    Shows average intensity of connections (only counting non-zero weights).
    
    Args:
        df_topology: DataFrame with topology metrics
        doc: python-docx Document object to add figure to
        figures_dir: Path to directory for saving figure PNG
        experiment_colors: Dict mapping source name to color
    """
    if df_topology.empty:
        return
    
    # Sort by source and elite_id
    df_sorted = df_topology.sort_values(['source', 'elite_id'])
    
    # Create x-axis labels and data
    x_labels = []
    exc_mean_strengths = []
    inh_mean_strengths = []
    colors = []
    
    for idx, row in df_sorted.iterrows():
        source = row['source']
        elite_id = row['elite_id']
        x_labels.append(f"{source}\nelite_{int(elite_id)}")
        exc_mean_strengths.append(row['exc_mean_strength'])
        inh_mean_strengths.append(row['inh_mean_strength'])
        colors.append(experiment_colors.get(source, '#808080'))
    
    x = np.arange(len(x_labels))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Create bars
    bars1 = ax.bar(x - width/2, exc_mean_strengths, width, label='Excitatory Mean Strength', 
                   alpha=0.8, color=colors)
    bars2 = ax.bar(x + width/2, inh_mean_strengths, width, label='Inhibitory Mean Strength', 
                   alpha=0.6, color=colors)
    
    ax.set_xlabel('Source and Elite', fontsize=12)
    ax.set_ylabel('Mean Connection Strength', fontsize=12)
    ax.set_title('Connection Intensity: Average Weight of Existing Connections', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.tight_layout()
    
    # Save to file
    figures_dir.mkdir(exist_ok=True)
    figure_path = figures_dir / 'topology_mean_strength.png'
    figure_path_abs = figure_path.resolve()
    fig.savefig(str(figure_path_abs), dpi=150, bbox_inches='tight')
    
    # Add to report
    doc.add_picture(str(figure_path_abs), width=6.5 * 914400)
    doc.add_paragraph()
    
    plt.close(fig)


# ==================================================================================================================================================
# SECTION C) DATA LOADING
# ==================================================================================================================================================



experiment_hdf5_path = _find_hdf5_file(EXPERIMENT_HDF5)

# Load experiment attributes
experiment_attrs = _load_ea_attributes(experiment_hdf5_path)

# Extract neuron types from attributes (used for all networks, same for all files)
neuron_types = extract_neuron_types(experiment_attrs)

# Load experiment generation stats
df_generation_stats_exp = _load_generation_stats(experiment_hdf5_path)


# Load experiment elite lifespans
df_elite_lifespans_exp = _load_elite_lifespans(experiment_hdf5_path)

# Load experiment elite genomes
df_elite_genomes_exp = _load_elite_genomes(experiment_hdf5_path)


# Load experiment connection weights
connection_weights_experiment = _load_connection_weights(experiment_hdf5_path)


# Load experiment modulation specs

df_modulation_specs_exp = _load_modulation_specs(experiment_hdf5_path, 'experiment')


print("\nExperiment data loaded.")

# Load benchmark data if specified
df_benchmarks_generations_stats = None
df_benchmarks_elite_lifespans = None
df_benchmarks_elite_genomes = None
df_benchmarks_modulation_specs = None
benchmark_attrs = {}
connection_weights_collection = {}  # Will collect all connection_weights structures

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
            
            # Load benchmark modulation specs
            df_mod_specs = _load_modulation_specs(bench_path, bench_name)
            if df_benchmarks_modulation_specs is None:
                df_benchmarks_modulation_specs = df_mod_specs
            else:
                df_benchmarks_modulation_specs = pd.concat([df_benchmarks_modulation_specs, df_mod_specs], ignore_index=True)
            
        except FileNotFoundError as e:
            pass
        except Exception as e:
            pass
else:
    print("No benchmarks specified.")


# Collect all lifespan data into a unified dataframe
df_lifespan_data = collect_lifespan_data(EXPERIMENT_NAME, df_elite_lifespans_exp, df_benchmarks_elite_lifespans)

# Collect topology metrics from all connection weight data
topology_data = []

# Experiment topology
for elite_id, weights in connection_weights_experiment.items():
    metrics = compute_topology_metrics(weights)
    metrics['source'] = EXPERIMENT_NAME
    metrics['elite_id'] = elite_id
    topology_data.append(metrics)

# Benchmark topology
for bench_name, bench_weights_dict in connection_weights_collection.items():
    for elite_id, weights in bench_weights_dict.items():
        metrics = compute_topology_metrics(weights)
        # Map back to benchmark display name (remove prefix and convert underscores)
        bench_display_name = bench_name.replace('connection_weights_', '').replace('_', ' ').title()
        # Try to find the original benchmark name
        for original_name in benchmark_attrs.keys():
            if original_name.lower().replace(' ', '_').replace('-', '_') == bench_name.replace('connection_weights_', ''):
                bench_display_name = original_name
                break
        metrics['source'] = bench_display_name
        metrics['elite_id'] = elite_id
        topology_data.append(metrics)

df_topology = pd.DataFrame(topology_data)

print("\nAll HDF5 data loaded.")

# Assign colors to experiments (used across all figures)
experiments = sorted(df_lifespan_data['experiment'].unique())
experiment_colors = {}
for idx, exp in enumerate(experiments):
    if idx < len(COLOR_PALETTE):
        # Use predefined colors
        experiment_colors[exp] = COLOR_PALETTE[idx]
    else:
        # Generate random color if more than palette size
        experiment_colors[exp] = "#{:06x}".format(random.randint(0, 0xFFFFFF))

# Create figures directory
figures_dir = Path(__file__).resolve().parent / 'figures'


# ==================================================================================================================================================
# SECTION D) ANALYSIS
# ==================================================================================================================================================

doc = Document()
doc.add_heading(f"EA Analysis Report: {EXPERIMENT_NAME}", level=0)

# region Overview
doc.add_heading(f"Summary experiment parameters", level=1)
# Call parameter summary function
summarize_parameter(doc, experiment_attrs, EXPERIMENT_HDF5, benchmark_attrs, BENCHMARK_HDF5_FILES)

# endregion Overview
# region EA Results / Fitness
doc.add_heading(f"EA Results / Fitness", level=1)

# Plot and add first figure
plot_ea_results(df_generation_stats_exp, doc, figures_dir)

# Plot fitness distribution histogram
plot_fitness_distribution(df_lifespan_data, doc, figures_dir)

# endregion EA Results / Fitness
# region connectivity

doc.add_heading(f"Connectivity", level=1)

plot_topology_connectivity(df_topology, doc, figures_dir, experiment_colors)

plot_topology_weighted_connectivity(df_topology, doc, figures_dir, experiment_colors)

plot_topology_mean_strength(df_topology, doc, figures_dir, experiment_colors)

# Plot network visualizations
if draw_network is not None:
    doc.add_heading('Network Visualizations', level=2)
    
    # Plot experiment elites
    for elite_id in sorted(connection_weights_experiment.keys()):
        try:
            weights = connection_weights_experiment[elite_id]
            neurons, connections_list = convert_weights_to_network_format(weights, neuron_types=neuron_types)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Convert connections list to DataFrame, with fallback for empty networks
            if connections_list:
                connections_df = pd.DataFrame(connections_list)
            else:
                # Create minimal valid DataFrame for empty networks
                connections_df = pd.DataFrame({'src': [0], 'tgt': [0], 'weight': [0.0]})
            
            # Draw network
            draw_network(
                neurons=neurons,
                connections=connections_df,
                ax=ax,
                title=f'{EXPERIMENT_NAME} Elite {elite_id}',
                show_weights=True,
                weight_column='weight'
            )
            
            # Save to file
            figures_dir.mkdir(exist_ok=True)
            figure_path = figures_dir / f'network_{EXPERIMENT_NAME}_elite_{elite_id}.png'
            fig.savefig(str(figure_path), dpi=150, bbox_inches='tight')
            
            # Add to report
            doc.add_picture(str(figure_path), width=6.5 * 914400)
            doc.add_paragraph()
            
            plt.close(fig)
        except Exception as e:
            print(f"Error plotting network for experiment elite {elite_id}: {e}")
            plt.close('all')
            continue
    
    # Plot benchmark variants (up to 3 per benchmark)
    for bench_key, bench_weights_dict in sorted(connection_weights_collection.items()):
        # Get the display name from benchmark_attrs
        bench_display_name = None
        for original_name in benchmark_attrs.keys():
            if original_name.lower().replace(' ', '_').replace('-', '_') == bench_key.replace('connection_weights_', ''):
                bench_display_name = original_name
                break
        
        if bench_display_name is None:
            # Fallback name
            bench_display_name = bench_key.replace('connection_weights_', '').replace('_', ' ').title()
        
        # Plot first 3 variants
        for variant_idx, (elite_id, weights) in enumerate(sorted(bench_weights_dict.items())):
            if variant_idx >= 3:  # Only plot first 3
                break
            
            try:
                neurons, connections_list = convert_weights_to_network_format(weights, neuron_types=neuron_types)
                
                # Create figure
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Convert connections list to DataFrame
                if connections_list:
                    connections_df = pd.DataFrame(connections_list)
                else:
                    connections_df = pd.DataFrame({'src': [0], 'tgt': [0], 'weight': [0.0]})
                
                # Draw network
                draw_network(
                    neurons=neurons,
                    connections=connections_df,
                    ax=ax,
                    title=f'{bench_display_name} {elite_id}',
                    show_weights=True,
                    weight_column='weight'
                )
                
                # Save to file
                figures_dir.mkdir(exist_ok=True)
                figure_path = figures_dir / f'network_{bench_key}_{elite_id}.png'
                fig.savefig(str(figure_path), dpi=150, bbox_inches='tight')
                
                # Add to report
                doc.add_picture(str(figure_path), width=6.5 * 914400)
                doc.add_paragraph()
                
                plt.close(fig)
            except Exception as e:
                print(f"Error plotting network for benchmark {bench_display_name} variant {elite_id}: {e}")
                plt.close('all')
                continue
else:
    print("Skipping network visualizations (network_viz module not available)")

# endregion connectivity
# region plasticity

# endregion plasticity





# ==================================================================================================================================================
# SECTION E) SAVE REPORT
# ==================================================================================================================================================


output_dir = Path(__file__).resolve().parent
output_path = output_dir / f'results_{EXPERIMENT_NAME}.docx'

# Close any open Word document with the same name before saving
# This prevents file-locked errors and prompts user to save changes if needed
_close_word_document(output_path)

doc.save(str(output_path))
print(f"Report saved to: {output_path}")


