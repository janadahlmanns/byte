# Batch simulation of Byte with randomized wiring variants
# Usage: python -m simulate.run_batch --config configs/experiments/neurons_random_wiring.yaml

import sys
import argparse
import importlib
from pathlib import Path
from datetime import datetime
import yaml
import numpy as np
import h5py
import matplotlib.pyplot as plt
from mvb.simulation_API import eval_generation
from mvb.genome import generate_genome_mutate_simple
from .pause_manager import init_pause_manager, cleanup_pause_manager


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run batch simulation of Byte',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m simulate.run_batch --config plasticity_batch
  python -m simulate.run_batch --config neurons_lookup_validation
        """
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Name of the experiment config file (without .yaml/.yml extension)')
    return parser.parse_args()

def resolve_config_path(config_name: str, config_dir: str = "configs/experiments") -> str:
    """Resolve a config name to a full path.
    
    Args:
        config_name: Name of config file (with or without .yaml/.yml extension)
        config_dir: Directory to search for YAML files (default: configs/experiments)
    
    Returns:
        Full path to the config file
    
    Raises:
        FileNotFoundError: If config file not found
    """
    # If config_name already has an extension, use it as-is
    if config_name.endswith(('.yaml', '.yml')):
        full_path = Path(config_dir) / config_name
    else:
        # Try .yaml first, then .yml
        yaml_path = Path(config_dir) / f"{config_name}.yaml"
        yml_path = Path(config_dir) / f"{config_name}.yml"
        
        if yaml_path.exists():
            full_path = yaml_path
        elif yml_path.exists():
            full_path = yml_path
        else:
            raise FileNotFoundError(f"Configuration file not found: {config_name}")
    
    return str(full_path)

def find_available_configs(config_dir: str = "configs/experiments") -> list:
    """Find all YAML configuration files in the configs/experiments directory.
    
    Args:
        config_dir: Directory to search for YAML files (default: configs/experiments)
    
    Returns:
        List of config file names (without extension)
    """
    config_path = Path(config_dir)
    if not config_path.exists():
        return []
    
    yaml_files = sorted(config_path.glob("*.yaml")) + sorted(config_path.glob("*.yml"))
    # Return just the names without extensions
    return [f.stem for f in yaml_files]

def load_config(config_name: str):
    """Load YAML configuration file from configs/experiments directory.
    
    Args:
        config_name: Name of the config file (with or without extension)
    
    Returns:
        Parsed YAML configuration as dict
    """
    config_path = resolve_config_path(config_name)
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_genome_generator(genome_type: str):
    """Load genome generator function from mvb.genome module.
    
    Parameters
    ----------
    genome_type : str
        Name of the genome generator (e.g., "random", "lookup_soft", "lookup_hard")
    
    Returns
    -------
    callable
        The genome generator function (generate_random_genome, generate_lookup_soft_genome, generate_lookup_hard_genome, etc.)
    """
    if not genome_type or genome_type.lower() == "none":
        raise ValueError(f"Invalid genome type: '{genome_type}'")
    
    # Map genome type names to actual function names
    function_map = {
        "random": "generate_random_genome",
        "lookup_soft": "generate_lookup_soft_genome",
        "lookup_hard": "generate_lookup_hard_genome",
    }
    
    function_name = function_map.get(genome_type.lower())
    if not function_name:
        raise ValueError(f"Unknown genome type '{genome_type}'. Available types: {list(function_map.keys())}")
    
    try:
        from mvb import genome as genome_module
        if not hasattr(genome_module, function_name):
            raise AttributeError(f"Genome module has no function '{function_name}'.")
        return getattr(genome_module, function_name)
    except ImportError:
        raise ImportError(f"Could not import genome module from mvb.")

def load_mutation_method(mutation_method: str):
    """Load mutation method function from mvb.genome module.
    
    Parameters
    ----------
    mutation_method : str
        Name of the mutation method (e.g., "simple")
    
    Returns
    -------
    callable
        The mutation function (generate_genome_mutate_simple, etc.)
    """
    if not mutation_method or mutation_method.lower() == "none":
        raise ValueError(f"Invalid mutation method: '{mutation_method}'")
    
    # Map mutation method names to actual function names
    function_map = {
        "simple": "generate_genome_mutate_simple",
    }
    
    function_name = function_map.get(mutation_method.lower())
    if not function_name:
        raise ValueError(f"Unknown mutation method '{mutation_method}'. Available methods: {list(function_map.keys())}")
    
    try:
        from mvb import genome as genome_module
        if not hasattr(genome_module, function_name):
            raise AttributeError(f"Genome module has no function '{function_name}'.")
        return getattr(genome_module, function_name)
    except ImportError:
        raise ImportError(f"Could not import genome module from mvb.")

def check_ea_input_parameters(experiment_cfg):
    """
    Validate that all required evolutionary algorithm parameters are present.
    
    This function checks that the 'evolutionary_algorithm' section exists and
    contains all required parameters with no defaults allowed.
    
    Args:
        experiment_cfg: The 'experiment' section of the YAML config
    
    Raises:
        KeyError: If any required EA parameter is missing
    """
    # REQUIRED: evolutionary_algorithm section must be present
    try:
        ea_cfg = experiment_cfg["evolutionary_algorithm"]
    except KeyError:
        raise KeyError("[ERROR] EA parameter not complete. REQUIRED: 'evolutionary_algorithm' section not found in config. Please add it with all required parameters: num_generations, elite_size, elite_selection_metric, mutation_rate, mutation_method, mutation_seed.")
    
    # REQUIRED: All EA parameters must be explicitly specified - NO DEFAULTS
    required_params = ["num_generations", "elite_size", "elite_selection_metric", "mutation_rate", "mutation_method", "mutation_strength"]
    for param in required_params:
        if param not in ea_cfg:
            raise KeyError(f"[ERROR] EA parameter not complete. REQUIRED: {param} missing in 'evolutionary_algorithm' section. Please specify all of: {', '.join(required_params)}")

def calculate_lifespan_statistics(lifespans_array):
    """
    Calculate statistics on an array of lifespans.
    
    Args:
        lifespans_array: 1D numpy array of lifespan values
    
    Returns:
        dict with keys: mean, median, min, max, std, iqr
    """
    lifespans_array = np.asarray(lifespans_array)
    stats = {
        'mean': float(np.mean(lifespans_array)),
        'median': float(np.median(lifespans_array)),
        'min': float(np.min(lifespans_array)),
        'max': float(np.max(lifespans_array)),
        'std': float(np.std(lifespans_array)),
        'iqr': float(np.percentile(lifespans_array, 75) - np.percentile(lifespans_array, 25))
    }
    return stats

def pick_elite_deterministic(all_lifespans, elite_selection_metric, elite_size):
    """
    Select elite genomes based on fitness metric.
    
    Ranks variants according to elite_selection_metric (average, max, or median lifespan)
    and returns the indices of the top elite_size variants.
    
    Args:
        all_lifespans: dict mapping variant_id → lifespan array
        elite_selection_metric: 'average', 'max', or 'median'
        elite_size: number of elite genomes to select
    
    Returns:
        Tuple of (elite_indices, generation_stats_tuple)
        - elite_indices: list of variant_ids of elite genomes
        - generation_stats_tuple: tuple (mean, median, min, max, std, iqr) from ALL variants' fitness metrics
    """
    # Calculate fitness metric for each variant
    variant_fitness = {}
    for variant_id, lifespan_results in all_lifespans.items():
        lifespans_array = np.asarray(lifespan_results['lifespan_vector'])
        if elite_selection_metric == 'average':
            fitness = np.mean(lifespans_array)
        elif elite_selection_metric == 'max':
            fitness = np.max(lifespans_array)
        elif elite_selection_metric == 'median':
            fitness = np.median(lifespans_array)
        else:
            raise ValueError(f"Unknown elite_selection_metric: {elite_selection_metric}")
        variant_fitness[variant_id] = fitness
    
    # Sort variants by fitness (descending), with variant_id as tiebreaker
    sorted_variants = sorted(variant_fitness.items(), key=lambda x: (-x[1], x[0]))
    
    # Pick top elite_size
    elite_indices = [variant_id for variant_id, fitness in sorted_variants[:elite_size]]
    
    # Compute generation-level statistics from ALL variants' fitness metrics (not just elite)
    all_fitness_values = np.array(list(variant_fitness.values()))
    
    generation_stats_tuple = (
        float(np.mean(all_fitness_values)),
        float(np.median(all_fitness_values)),
        float(np.min(all_fitness_values)),
        float(np.max(all_fitness_values)),
        float(np.std(all_fitness_values)),
        float(np.percentile(all_fitness_values, 75) - np.percentile(all_fitness_values, 25))
    )
    
    return elite_indices, generation_stats_tuple


def save_config_recursive(hdf5_group, config_dict, prefix=""):
    """Recursively save config dict to HDF5 attributes."""
    for key, value in config_dict.items():
        attr_name = f"{prefix}{key}" if prefix else key
        if isinstance(value, dict):
            save_config_recursive(hdf5_group, value, f"{attr_name}_")
        elif isinstance(value, list):
            # Convert lists to string representation
            hdf5_group.attrs[attr_name] = str(value)
        elif isinstance(value, (str, int, float, bool, type(None))):
            hdf5_group.attrs[attr_name] = value


def save_elite_to_hdf5(hdf5_path, elite_genomes, elite_lifespan_vectors, elite_seeds_noise, elite_seeds_decision):
    """Save/update elite genomes, lifespans, and seed data to HDF5.
    
    Overwrites the existing elite_genomes group with fresh data from the current generation.
    Position indices are used: the i-th genome corresponds to the i-th lifespan array and seed arrays.
    Raw lifespan data and per-run seed values are preserved for resuming evolution from checkpoints.
    
    Args:
        hdf5_path: Path to HDF5 file
        elite_genomes: List of elite genome objects
        elite_lifespan_vectors: List of lifespan arrays (parallel to elite_genomes)
        elite_seeds_noise: List of noise RNG seed arrays (one N_RUNS-length array per elite, parallel to elite_genomes)
        elite_seeds_decision: List of decision RNG seed arrays (one N_RUNS-length array per elite, parallel to elite_genomes)
    """
    with h5py.File(hdf5_path, 'a') as f:
        # Remove old elite_genomes group and recreate
        if "elite_genomes" in f:
            del f["elite_genomes"]
        elite_group = f.create_group("elite_genomes")
        
        # Save elite genomes with positional indices
        for elite_pos, elite_genome in enumerate(elite_genomes):
            elite_subgroup = elite_group.create_group(f"elite_{elite_pos}")
            elite_subgroup.create_dataset("connection_weights", data=elite_genome.connection_weights)
            elite_subgroup.create_dataset("tonic_activations", data=elite_genome.tonic_activations)
            elite_subgroup.create_dataset("eta", data=np.array([elite_genome.eta]))
            
            # Flatten modulation spec into single dataset: one row per modulation entry
            mod_records = [
                (int(src), int(tgt), int(mod_neuron), float(mod_weight))
                for (src, tgt), modulators in elite_genome.modulation_spec.items()
                for mod_neuron, mod_weight in modulators
            ]
            if mod_records:
                mod_dtype = np.dtype([
                    ('source', np.int32),
                    ('target', np.int32),
                    ('modulating_neuron', np.int32),
                    ('modulation_weight', np.float32)
                ])
                elite_subgroup.create_dataset("modulation_spec", data=np.array(mod_records, dtype=mod_dtype))
        
        # Save elite lifespans as single dataset
        elite_group.create_dataset("lifespans", data=elite_lifespan_vectors, dtype=np.float32)
        
        # Save elite seed data (N_RUNS seeds per elite, stored as list of arrays)
        # elite_seeds_noise is a list of N_RUNS-length arrays, one per elite
        for elite_pos, (noise_seeds, decision_seeds) in enumerate(zip(elite_seeds_noise, elite_seeds_decision)):
            elite_subgroup = elite_group[f"elite_{elite_pos}"]
            elite_subgroup.create_dataset("seeds_noise", data=noise_seeds, dtype=np.uint32)
            elite_subgroup.create_dataset("seeds_decision", data=decision_seeds, dtype=np.uint32)


def write_generation_stats_to_hdf5(hdf5_path, generation, gen_stats, run_seeds):
    """Write a generation's stats and run_seeds to HDF5.
    
    Args:
        hdf5_path: Path to HDF5 file
        generation: Generation index (0, 1, 2, ...)
        gen_stats: Tuple of (mean, median, min, max, std, iqr)
        run_seeds: Array of N_RUNS seeds drawn for world initialization this generation
    """
    gen_stats_tuple = (generation,) + gen_stats
    
    with h5py.File(hdf5_path, 'a') as f:
        stats_dataset = f['generation_stats']
        stats_dataset[generation] = gen_stats_tuple
        
        # Save/update run_seeds dataset
        # If elite_genomes group exists, save there; otherwise at root level
        if 'elite_genomes' in f:
            if 'run_seeds' not in f['elite_genomes']:
                f['elite_genomes'].create_dataset('run_seeds', data=run_seeds)
            else:
                f['elite_genomes']['run_seeds'][:] = run_seeds
        else:
            if 'run_seeds' not in f:
                f.create_dataset('run_seeds', data=run_seeds)
            else:
                f['run_seeds'][:] = run_seeds
        if "generation_stats" not in f:
            raise ValueError("generation_stats dataset not found. Initialize HDF5 first.")
        
        stats_dataset = f["generation_stats"]
        stats_dataset[generation] = gen_stats_tuple


def initialize_hdf5_file(experiment_folder, simulation_name, cfg, num_generations=None):
    """Initialize HDF5 file and save configuration.
    
    Creates the HDF5 file with timestamp naming and saves all configuration
    as attributes. Pre-allocates the stats dataset for incremental writes.
    
    Args:
        experiment_folder: Path to output folder
        simulation_name: Name of the simulation for the filename
        cfg: Full configuration dictionary
        num_generations: Number of generations (for pre-allocating stats dataset)
    
    Returns:
        Path to the created HDF5 file
    """
    # Create HDF5 file path with timestamp
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    hdf5_filename = f"{ts}_{simulation_name}.h5"
    hdf5_path = Path(experiment_folder) / hdf5_filename
    Path(experiment_folder).mkdir(parents=True, exist_ok=True)
    
    # Create file and save configuration
    with h5py.File(hdf5_path, 'w') as f:
        # Save full config as attributes
        save_config_recursive(f, cfg)
        
        # Create placeholder groups for results (will be populated later)
        f.create_group("elite_genomes")
        
        # Pre-allocate stats dataset if num_generations is known
        if num_generations is not None:
            gen_stats_dtype = np.dtype([
                ('generation', np.int32),
                ('mean', np.float32),
                ('median', np.float32),
                ('min', np.float32),
                ('max', np.float32),
                ('std', np.float32),
                ('iqr', np.float32)
            ])
            # Create with size num_generations 
            f.create_dataset("generation_stats",
                           shape=(num_generations,),
                           dtype=gen_stats_dtype)
    
    print(f"\n[HDF5 Init] File created: {hdf5_path}")
    return str(hdf5_path)


def print_results(hdf5_path, num_generations, elite_size):
    """Print evolutionary algorithm completion summary."""
    print("\n" + "="*80)
    print("EVOLUTIONARY ALGORITHM COMPLETED")
    print("="*80)
    print(f"\nResults saved to: {hdf5_path}")
    print(f"Total generations evolved: {num_generations}")
    print(f"Final elite size: {elite_size}")
    print("="*80 + "\n")


def plot_results(hdf5_path, save_path=None, show=False):
    """Plot generation statistics from HDF5 file.
    
    Displays a plot with mean±std and median±IQR shading, plus min/max lines.
    
    Args:
        hdf5_path: Path to HDF5 file with generation_stats dataset
        save_path: Optional path to save the figure as PNG. If None, figure is not saved.
        show: Whether to display the plot interactively (plt.show())
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
    
    # Color scheme
    primary_color = "#D4AF37"      # Light Gold
    secondary_color = "#E69F00"    # Dark Gold
    tertiary_color = "#4A7C8C"     # Grayish ice blue
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot mean with std shading (primary)
    ax.fill_between(generations, mean_vals - std_vals, mean_vals + std_vals, 
                    alpha=0.3, color=primary_color, label='Mean with std')
    ax.plot(generations, mean_vals, '-', linewidth=2.5, color=primary_color)
    
    # Plot median with IQR shading (secondary)
    ax.fill_between(generations, median_vals - iqr_vals/2, median_vals + iqr_vals/2, 
                    alpha=0.3, color=secondary_color, label='Median with IQR')
    ax.plot(generations, median_vals, '-', linewidth=2.5, color=secondary_color)
    
    # Plot min and max (tertiary)
    ax.plot(generations, min_vals, '--', linewidth=2, color=tertiary_color, label='Min and max')
    ax.plot(generations, max_vals, '--', linewidth=2, color=tertiary_color)
    
    ax.set_xlabel('Generations', fontsize=12)
    ax.set_ylabel('Average Lifespan [ticks]', fontsize=12)
    ax.set_title('Average Lifespan Across Generations', fontsize=14)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    # Save to file if requested
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot Saved] {save_path}")
    
    # Show interactively if requested
    if show:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# main
# ============================================================


def main():
    # ============================================================
    # 1. IMPORTS & CONFIGURATION LOADING
    # ============================================================
    args = parse_arguments()
    
    # Try to load config with helpful error message if not found
    try:
        cfg = load_config(args.config)
    except FileNotFoundError as e:
        available_configs = find_available_configs()
        print("\n" + "="*80)
        print(f"[ERROR] {e}")
        print("="*80)
        print("\nAvailable experiment configurations:")
        if available_configs:
            for config_name in available_configs:
                print(f"  • {config_name}")
        else:
            print("  (No YAML configuration files found in 'configs/experiments/' directory)")
        print("\nExample:")
        print("  python -m simulate.run_batch --config plasticity_batch")
        print("="*80 + "\n")
        sys.exit(1)
    
    experiment_cfg = cfg["experiment"]
    brain_module_name = str(cfg["worm"]["decisionmaking"]["version"])
    
    # Check if EA is enabled and validate parameters
    EA_ENABLED = experiment_cfg["evolutionary_algorithm_enabled"]
    if EA_ENABLED:
        check_ea_input_parameters(experiment_cfg)
    
    # Extract experiment parameters from YAML (must all be present)
    EXPERIMENT_FOLDER = experiment_cfg["output_folder"]
    SIMULATION_NAME = experiment_cfg["simulation_name"]
    GENOME_TYPE = experiment_cfg["genome_type"]
    genome_generator = load_genome_generator(GENOME_TYPE)
    
    # Get wiring randomization seed from genome-type-specific config
    if GENOME_TYPE.lower() == "random":
        WIRING_RANDOMIZATION_SEED = experiment_cfg["random_genome"]["wiring_randomization_seed"]
    else:
        # For lookup or other genome types, wiring seed is not used
        WIRING_RANDOMIZATION_SEED = 0
    
    POPULATION_SIZE = experiment_cfg["population_size"]
    MAX_TICKS = experiment_cfg["max_ticks"]
    N_RUNS = experiment_cfg["n_runs"]
    
    VIZ_ENABLED = experiment_cfg["viz_enabled"]
    VIZ_FPS = experiment_cfg["viz_fps"]
    VIZ_BRAIN_ENABLED = experiment_cfg["viz_brain_enabled"]
    VIZ_BRAIN_FPS = experiment_cfg["viz_brain_fps"]
    ENABLE_PER_RUN_TRACKING = experiment_cfg["enable_per_run_tracking"]
    ENABLE_PER_TICK_TRACKING = experiment_cfg["enable_per_tick_tracking"]
    ENABLE_HEAT_MAP_TRACKING = experiment_cfg["enable_heat_map_tracking"]
    
    # Extract EA parameters if enabled
    if EA_ENABLED:
        ea_cfg = experiment_cfg["evolutionary_algorithm"]
        NUM_GENERATIONS = ea_cfg["num_generations"]
        ELITE_SIZE = ea_cfg["elite_size"]
        ELITE_SELECTION_METRIC = ea_cfg["elite_selection_metric"]
        MUTATION_RATE = ea_cfg["mutation_rate"]
        MUTATION_STRENGTH = ea_cfg["mutation_strength"]
        MUTATION_METHOD = ea_cfg["mutation_method"]
        mutation_function = load_mutation_method(MUTATION_METHOD)
        
        print(f"[EA Config] Population: {POPULATION_SIZE}, Generations: {NUM_GENERATIONS}, Elite: {ELITE_SIZE}")
        print(f"[EA Config] Selection metric: {ELITE_SELECTION_METRIC}, Mutation rate: {MUTATION_RATE}, Mutation strength: {MUTATION_STRENGTH}")
        print(f"[EA Config] Mutation method: {MUTATION_METHOD}")
        
        # Each generation produces POPULATION_SIZE - ELITE_SIZE offspring.
        # The elite is re-evaluated alongside the offspring so all individuals
        # are compared on the same run_seeds (fair fitness comparison).
        n_offspring = POPULATION_SIZE - ELITE_SIZE
        if n_offspring <= 0:
            raise ValueError(f"[ERROR] POPULATION_SIZE ({POPULATION_SIZE}) must be greater than ELITE_SIZE ({ELITE_SIZE}).")
        offspring_per_parent_base = n_offspring // ELITE_SIZE
        remainder = n_offspring % ELITE_SIZE
        offspring_counts = [offspring_per_parent_base] * ELITE_SIZE
        if remainder > 0:
            offspring_counts[-1] += remainder  # Last parent gets remainder
        if NUM_GENERATIONS == 1:
            ELITE_SIZE = POPULATION_SIZE
            print(f"[EA Config] Single generation: ELITE_SIZE will be set to POPULATION_SIZE to save all variants.")
    else:
        ELITE_SIZE = POPULATION_SIZE
        NUM_GENERATIONS = 1
        ELITE_SELECTION_METRIC = 'average'  # Not used when EA is disabled, but set to default for consistency  
        print(f"[EA Config] EA Disbaled: First generation will be evaluated and saved directly.")

    # BUILD RNG STREAMS AT BATCH LEVEL (very first thing)
    SIMULATION_SEED = experiment_cfg["simulation_seed"]
    
    # ============================================================
    # 2. VALIDATION & USER CHECKS
    # ============================================================
    # Check for genome_type vs config consistency
    has_brain_config = cfg["worm"]["decisionmaking"]["brain"]
    if GENOME_TYPE.lower() == "none" and has_brain_config:
        raise ValueError(f"Config specifies brain: true but GENOME_TYPE is 'none'. Please set GENOME_TYPE in the 'experiment' section.")
    
    # Check if tracking/viz is enabled for EA - not recommended for evolutionary algorithm
    if ENABLE_PER_RUN_TRACKING or ENABLE_PER_TICK_TRACKING or ENABLE_HEAT_MAP_TRACKING or VIZ_ENABLED or VIZ_BRAIN_ENABLED:
        response = input("[WARNING] Tracking and/or visualization is enabled. Continue without? (y/n): ").strip().lower()
        if response != 'y':
            print("[EXIT] EA execution stopped.")
            return
        
        ENABLE_PER_RUN_TRACKING = False
        ENABLE_PER_TICK_TRACKING = False
        ENABLE_HEAT_MAP_TRACKING = False
        VIZ_ENABLED = False
        VIZ_BRAIN_ENABLED = False
    
    # Extract config components for worker
    grid_width = cfg["world"]["grid_width"]
    grid_height = cfg["world"]["grid_height"]
    start_pos = cfg["world"]["start_pos"]
    worm_speed = cfg["worm"]["speed"]
    worm_energy_capacity = cfg["worm"]["energy_capacity"]
    worm_metabolic_rate = cfg["worm"]["metabolic_rate"]
    worm_movement_cost = cfg["worm"]["movement_cost"]
    sensor_cfg = cfg["worm"]["sensors"]["active"]
    
    # Use config subsections directly (no wrapping)
    feeding_cfg = cfg["food"]
    brain_cfg = cfg["brain"]
    
    # Create HDF5 file early to catch file system errors before experiment runs
    hdf5_path = initialize_hdf5_file(EXPERIMENT_FOLDER, SIMULATION_NAME, cfg, NUM_GENERATIONS)
    
    # ============================================================
    # 3. GENERATE RNG SEEDS FOR VARIANTS
    # ============================================================

    # Spawn RNG streams from single simulation seed
    # Always spawn: rng_noise, rng_decision, rng_world
    # Additionally spawn rng_mutation if EA is enabled
    # DO THIS ONLY ONCE IN THE BEGINNING OF RUNNING ANYTHING, NOT FOR EVERY GENERATION!!!!!
    num_streams = 4 if EA_ENABLED else 3
    seed_seq_sim = np.random.SeedSequence(int(SIMULATION_SEED))
    streams_sim = seed_seq_sim.spawn(num_streams)
    
    rng_noise = np.random.default_rng(streams_sim[0])
    rng_decision = np.random.default_rng(streams_sim[1])
    rng_world = np.random.default_rng(streams_sim[2])
    if EA_ENABLED:
        rng_mutation = np.random.default_rng(streams_sim[3])    

    # ============================================================
    # 4. GENERATE INITIAL POPULATION (GENERATION 0)
    # ============================================================
    # Gen 0: evaluate a fresh population of POPULATION_SIZE genomes and select elite from it.
    # From gen 1 onward the elite is re-evaluated together with the offspring.

    stats = []  # Collect generation statistics as tuples for direct array creation
    genomes = []
    for variant_id in range(POPULATION_SIZE):
        genome = genome_generator(cfg, rng_seed=WIRING_RANDOMIZATION_SEED + variant_id)
        genomes.append(genome)
    
    # ============================================================
    # 5. EVALUATE INITIAL POPULATION (GENERATION 0)
    # ============================================================

    lifespans, run_seeds_gen0 = eval_generation(genomes, cfg, EXPERIMENT_FOLDER, SIMULATION_NAME, ENABLE_PER_RUN_TRACKING, ENABLE_PER_TICK_TRACKING,
                                    ENABLE_HEAT_MAP_TRACKING, VIZ_ENABLED, VIZ_BRAIN_ENABLED, VIZ_FPS, VIZ_BRAIN_FPS, POPULATION_SIZE,
                                    rng_noise, rng_decision, rng_world, brain_module_name, MAX_TICKS, N_RUNS, grid_width, grid_height, start_pos, worm_speed, worm_energy_capacity, worm_metabolic_rate, worm_movement_cost, sensor_cfg, feeding_cfg, brain_cfg)                             


    # ============================================================
    # 6. SELECTION ON INITIAL POPULATION (GENERATION 0)
    # ============================================================
    
    elite_idx, gen_stats = pick_elite_deterministic(lifespans, ELITE_SELECTION_METRIC, ELITE_SIZE)
    
    # Append directly: (generation_idx,) + stats_tuple
    stats.append((0,) + gen_stats)
    

    # ============================================================
    # 7. APPLY SELECTION & SAVE
    # ============================================================
    
    elite_genomes = [genomes[i] for i in elite_idx]
    elite_lifespan_vectors = [lifespans[i]['lifespan_vector'] for i in elite_idx]
    elite_seeds_noise = [lifespans[i]['seeds_noise_all_runs'] for i in elite_idx]
    elite_seeds_decision = [lifespans[i]['seeds_decision_all_runs'] for i in elite_idx]
    save_elite_to_hdf5(hdf5_path, elite_genomes, elite_lifespan_vectors, elite_seeds_noise, elite_seeds_decision)
    write_generation_stats_to_hdf5(hdf5_path, 0, gen_stats, run_seeds_gen0)

    # ============================================================
    # 8. LOOP OVER GENERATIONS
    # ============================================================

    for generation in range(1, NUM_GENERATIONS):

        # ============================================================
        # 9. GENERATE OFFSPRING VIA MUTATION (POPULATION_SIZE - ELITE_SIZE offspring)
        # ============================================================
        genomes_new = []
        for elite_genome, num_offspring in zip(elite_genomes, offspring_counts):
            for offspring_num in range(num_offspring):
                mutated_genome = mutation_function(elite_genome, MUTATION_RATE, MUTATION_STRENGTH, rng_mutation)
                genomes_new.append(mutated_genome)

        # ============================================================
        # 10. COMBINE ELITE + OFFSPRING, THEN EVALUATE TOGETHER
        # Elite is re-evaluated with the same run_seeds as the offspring so
        # the fitness comparison is on equal footing every generation.
        # ============================================================
        genomes_combined = elite_genomes + genomes_new  # ELITE_SIZE + n_offspring = POPULATION_SIZE

        lifespans_combined, run_seeds_gen = eval_generation(genomes_combined, cfg, EXPERIMENT_FOLDER, SIMULATION_NAME, ENABLE_PER_RUN_TRACKING, ENABLE_PER_TICK_TRACKING,
                                    ENABLE_HEAT_MAP_TRACKING, VIZ_ENABLED, VIZ_BRAIN_ENABLED, VIZ_FPS, VIZ_BRAIN_FPS, POPULATION_SIZE,
                                    rng_noise, rng_decision, rng_world, brain_module_name, MAX_TICKS, N_RUNS, grid_width, grid_height, start_pos, worm_speed, worm_energy_capacity, worm_metabolic_rate, worm_movement_cost, sensor_cfg, feeding_cfg, brain_cfg)

        # ============================================================
        # 11. PICK ELITE FROM COMBINED SET (all evaluated on same run_seeds)
        # ============================================================

        elite_idx, gen_stats = pick_elite_deterministic(lifespans_combined, ELITE_SELECTION_METRIC, ELITE_SIZE)
        
        # Append directly: (generation_idx,) + stats_tuple
        stats.append((generation,) + gen_stats)
        
        # ============================================================
        # 12. APPLY SELECTION
        # ============================================================

        elite_genomes = [genomes_combined[i] for i in elite_idx]
        elite_lifespan_vectors = [lifespans_combined[i]['lifespan_vector'] for i in elite_idx]
        elite_seeds_noise = [lifespans_combined[i]['seeds_noise_all_runs'] for i in elite_idx]
        elite_seeds_decision = [lifespans_combined[i]['seeds_decision_all_runs'] for i in elite_idx]
        
        # ============================================================
        # 13. SAVE GENERATION RESULTS 
        # ============================================================
        save_elite_to_hdf5(hdf5_path, elite_genomes, elite_lifespan_vectors, elite_seeds_noise, elite_seeds_decision)
        write_generation_stats_to_hdf5(hdf5_path, generation, gen_stats, run_seeds_gen)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [Gen {generation + 1}/{NUM_GENERATIONS}] Saved. Max: {gen_stats[3]:.2f}, Median: {gen_stats[1]:.2f}, Mean: {gen_stats[0]:.2f}, Std: {gen_stats[4]:.2f}")

    # ============================================================
    # 14. PRINT AND PLOT RESULTS
    # ============================================================

    print_results(hdf5_path, NUM_GENERATIONS, ELITE_SIZE)
    
    # Generate and save results plot
    plot_save_path = str(Path(hdf5_path).with_name(Path(hdf5_path).stem + ".png"))
    plot_results(hdf5_path, save_path=plot_save_path, show=False)
    
    # Ask user if they want to view it interactively
    response = input("Display generation statistics plot? (y/n): ").strip().lower()
    if response == 'y':
        plot_results(hdf5_path, save_path=None, show=True)
    
    print("Done.\n")



if __name__ == "__main__":
    try:
        # Check if --config is missing and show helpful error
        if "--config" not in sys.argv and "--help" not in sys.argv and "-h" not in sys.argv:
            available_configs = find_available_configs()
            print("\n" + "="*80)
            print("[ERROR] Missing required argument: --config")
            print("="*80)
            print("\nUsage: python -m simulate.run_batch --config <name>")
            print("\nAvailable experiment configurations:")
            if available_configs:
                for config_name in available_configs:
                    print(f"  • {config_name}")
            else:
                print("  (No YAML configuration files found in 'configs/experiments/' directory)")
            print("\nExample:")
            print("  python -m simulate.run_batch --config plasticity_batch")
            print("="*80 + "\n")
            sys.exit(1)
        
        main()
    except SystemExit as e:
        if e.code != 0:
            raise

