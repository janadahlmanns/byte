# Batch simulation of Byte with randomized wiring variants
# Usage: python -m simulate.run_batch --config configs/experiments/neurons_random_wiring.yaml

import sys
import argparse
import importlib
from pathlib import Path
import yaml
import numpy as np
from mvb.simulation_API import eval_generation
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
        Name of the genome generator (e.g., "random", "lookup")
    
    Returns
    -------
    callable
        The genome generator function (generate_random_genome, generate_lookup_genome, etc.)
    """
    if not genome_type or genome_type.lower() == "none":
        raise ValueError(f"Invalid genome type: '{genome_type}'")
    
    # Map genome type names to actual function names
    function_map = {
        "random": "generate_random_genome",
        "lookup": "generate_lookup_genome",
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
        raise KeyError("[ERROR] EA parameter not complete. REQUIRED: 'evolutionary_algorithm' section not found in config. Please add it with all required parameters: num_generations, elite_size, elite_selection_metric, mutation_rate, mutation_types, mutation_seed.")
    
    # REQUIRED: All EA parameters must be explicitly specified - NO DEFAULTS
    required_params = ["num_generations", "elite_size", "elite_selection_metric", "mutation_rate", "mutation_types", "mutation_seed"]
    for param in required_params:
        if param not in ea_cfg:
            raise KeyError(f"[ERROR] EA parameter not complete. REQUIRED: {param} missing in 'evolutionary_algorithm' section. Please specify all of: {', '.join(required_params)}")


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
        MUTATION_TYPES = ea_cfg["mutation_types"]
        MUTATION_SEED = ea_cfg["mutation_seed"]
        
        print(f"[EA Config] Population: {POPULATION_SIZE}, Generations: {NUM_GENERATIONS}, Elite: {ELITE_SIZE}")
        print(f"[EA Config] Selection metric: {ELITE_SELECTION_METRIC}, Mutation rate: {MUTATION_RATE}")

    # BUILD RNG STREAMS AT BATCH LEVEL (very first thing)
    SIMULATION_SEED = experiment_cfg["simulation_seed"]
    GENERATION_SEED = cfg["world"]["generation_seed"]
    
    # Build independent RNG streams for decision-making, neuron noise, and mutations
    # DO THIS ONLY ONCE IN THE BEGINNING OF RUNNING ANYTHING, NOT FOR EVERY GENERATION!!!!!
    seed_seq_sim = np.random.SeedSequence(int(SIMULATION_SEED))
    streams_sim = seed_seq_sim.spawn(3)  # 3 streams: decision, neuron_noise, mutation
    rng_decision = np.random.default_rng(streams_sim[0])
    rng_neuron_noise = np.random.default_rng(streams_sim[1])
    rng_mutation = np.random.default_rng(streams_sim[2])
    
    # Build RNG stream for world (food distribution)
    seed_seq_gen = np.random.SeedSequence(int(GENERATION_SEED))
    rng_world = np.random.default_rng(seed_seq_gen.spawn(1)[0])

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
    
    # ============================================================
    # 3. GENERATE RNG SEEDS FOR VARIANTS
    # ============================================================
    
   
    variant_decision_seeds = rng_decision.integers(0, 2**32, size=POPULATION_SIZE, dtype=np.uint32)
    variant_noise_seeds = rng_neuron_noise.integers(0, 2**32, size=POPULATION_SIZE, dtype=np.uint32)
    
    # ============================================================
    # 4. GENERATE INITIAL POPULATION (GENERATION 0)
    # ============================================================
    print(f"\n[Gen 0] Generating {POPULATION_SIZE} initial genomes...")
    genomes = []
    for variant_id in range(POPULATION_SIZE):
        genome = genome_generator(cfg, rng_seed=WIRING_RANDOMIZATION_SEED + variant_id)
        genomes.append(genome)
    
    print(f"[Gen 0] Evaluating {POPULATION_SIZE} genomes...")
    all_lifespans = eval_generation(genomes, cfg, EXPERIMENT_FOLDER, SIMULATION_NAME, ENABLE_PER_RUN_TRACKING, ENABLE_PER_TICK_TRACKING,
                                    ENABLE_HEAT_MAP_TRACKING, rng_world, VIZ_ENABLED, VIZ_BRAIN_ENABLED, VIZ_FPS, VIZ_BRAIN_FPS, POPULATION_SIZE,
                                    brain_module_name, MAX_TICKS, N_RUNS, grid_width, grid_height, start_pos, worm_speed, worm_energy_capacity, worm_metabolic_rate, worm_movement_cost, sensor_cfg, feeding_cfg, brain_cfg, variant_decision_seeds, variant_noise_seeds)                             

    total_lifespans = sum(len(v) for v in all_lifespans.values())
    print(f"\n[Gen 0 Results] {total_lifespans} lifespans collected from {POPULATION_SIZE} genomes")
    print(f"[Gen 0 Results] Average lifespan per genome: {np.mean([np.mean(ls) for ls in all_lifespans.values()]):.2f} ticks")


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

