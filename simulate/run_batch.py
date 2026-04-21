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

def validate_tracking_flags(enable_per_run, enable_per_tick, enable_heat_map):
    """
    Validate tracking flags and adjust if necessary.
    
    Per-run tracking is a prerequisite for detailed tracking (per-tick and heatmap).
    If the user specifies detailed tracking without per-run tracking, ask them to choose:
    1. Exit and reconsider configuration
    2. Enable per-run tracking (keep the detailed flags)
    3. Disable all detailed tracking (keep only lifespan)
    
    Args:
        enable_per_run: Whether per-run tracking is enabled
        enable_per_tick: Whether per-tick tracking is enabled
        enable_heat_map: Whether heatmap tracking is enabled
    
    Returns: Tuple of (should_continue, corrected_enable_per_run, corrected_enable_per_tick, corrected_enable_heat_map)
    """
    # Check for invalid configuration: detailed tracking without per-run tracking
    if not enable_per_run and (enable_per_tick or enable_heat_map):
        print("\n" + "="*80)
        print("[ERROR] Invalid tracking configuration:")
        print("="*80)
        print(f"  ENABLE_PER_RUN_TRACKING = {enable_per_run}")
        print(f"  ENABLE_PER_TICK_TRACKING = {enable_per_tick}")
        print(f"  ENABLE_HEAT_MAP_TRACKING = {enable_heat_map}")
        print("\n[REASON] Per-tick and heatmap tracking can only be enabled if per-run")
        print("tracking is also enabled. They require per-run data structures.")
        print("\n[OPTIONS] Choose one of the following:")
        print("  [1] Exit execution and reconsider your configuration")
        print("  [2] Enable per-run tracking (keep the detailed flags as-is)")
        print("  [3] Disable all detailed tracking (record only lifespan)")
        print("="*80)
        
        while True:
            response = input("\nEnter your choice (1-3): ").strip()
            if response == '1':
                print("[EXIT] Execution stopped. Please fix your configuration.\n")
                return (False, False, False, False)
            elif response == '2':
                print("[CONFIG] Enabling per-run tracking with detailed flags enabled.")
                return (True, True, enable_per_tick, enable_heat_map)
            elif response == '3':
                print("[CONFIG] Disabling all detailed tracking. Only lifespan will be recorded.")
                return (True, False, False, False)
            else:
                print("[ERROR] Invalid choice. Enter 1, 2, or 3.")
    
    # No conflict: return original flags unchanged
    return (True, enable_per_run, enable_per_tick, enable_heat_map)

def validate_viz_flags(n_variants, n_runs, viz_enabled, viz_brain_enabled):
    """
    Validate visualization flags and adjust if necessary.
    
    If visualization is enabled for a large batch (total_simulations > 2),
    warn the user that it will be slow and ask whether to disable it.
    
    Args:
        n_variants: Number of variants
        n_runs: Number of runs per variant
        viz_enabled: Whether world visualization is enabled
        viz_brain_enabled: Whether brain visualization is enabled
    
    Returns: Tuple of (updated_viz_enabled, updated_viz_brain_enabled)
    """
    total_simulations = n_variants * n_runs
    if total_simulations > 2 and viz_enabled:
        print(f"\n[WARNING] Visualization is enabled for {n_variants} variants × {n_runs} runs = {total_simulations} total simulations.")
        print("This will be VERY SLOW. Batch runs typically disable visualization.")
        response = input("Continue with visualization? (y/n): ").strip().lower()
        if response != 'y':
            print("[INFO] Disabling visualization for this batch run.")
            return False, False
    
    return viz_enabled, viz_brain_enabled


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
    
    # BUILD RNG STREAMS AT BATCH LEVEL (very first thing)
    SIMULATION_SEED = experiment_cfg["simulation_seed"]
    GENERATION_SEED = cfg["world"]["generation_seed"]
    
    # Build independent RNG streams for decision-making and neuron noise
    seed_seq_sim = np.random.SeedSequence(int(SIMULATION_SEED))
    streams_sim = seed_seq_sim.spawn(2)
    rng_decision = np.random.default_rng(streams_sim[0])
    rng_neuron_noise = np.random.default_rng(streams_sim[1])
    
    # Build RNG stream for world (food distribution)
    seed_seq_gen = np.random.SeedSequence(int(GENERATION_SEED))
    rng_world = np.random.default_rng(seed_seq_gen.spawn(1)[0])
    
    # Extract experiment parameters from YAML (must all be present)
    EXPERIMENT_FOLDER = experiment_cfg["output_folder"]
    SIMULATION_NAME = experiment_cfg["simulation_name"]
    GENOME_TYPE = experiment_cfg["genome_type"]
    genome_generator = load_genome_generator(GENOME_TYPE)
    WIRING_RANDOMIZATION_SEED = experiment_cfg["wiring_randomization_seed"]
    N_VARIANTS = experiment_cfg["n_variants"]
    MAX_TICKS = experiment_cfg["max_ticks"]
    N_RUNS = experiment_cfg["n_runs"]
    
    VIZ_ENABLED = experiment_cfg["viz_enabled"]
    VIZ_FPS = experiment_cfg["viz_fps"]
    VIZ_BRAIN_ENABLED = experiment_cfg["viz_brain_enabled"]
    VIZ_BRAIN_FPS = experiment_cfg["viz_brain_fps"]
    ENABLE_PER_RUN_TRACKING = experiment_cfg["enable_per_run_tracking"]
    ENABLE_PER_TICK_TRACKING = experiment_cfg["enable_per_tick_tracking"]
    ENABLE_HEAT_MAP_TRACKING = experiment_cfg["enable_heat_map_tracking"]


    # ============================================================
    # 2. VALIDATION & USER CHECKS
    # ============================================================
    # Check for genome_type vs config consistency
    has_brain_config = cfg["worm"]["decisionmaking"]["brain"]
    if GENOME_TYPE.lower() == "none" and has_brain_config:
        raise ValueError(f"Config specifies brain: true but GENOME_TYPE is 'none'. Please set GENOME_TYPE in the 'experiment' section.")
    
    validation_result = validate_tracking_flags(ENABLE_PER_RUN_TRACKING, ENABLE_PER_TICK_TRACKING, ENABLE_HEAT_MAP_TRACKING)
    should_continue, ENABLE_PER_RUN_TRACKING, ENABLE_PER_TICK_TRACKING, ENABLE_HEAT_MAP_TRACKING = validation_result
    if not should_continue:
        return
    
    VIZ_ENABLED, VIZ_BRAIN_ENABLED = validate_viz_flags(N_VARIANTS, N_RUNS, VIZ_ENABLED, VIZ_BRAIN_ENABLED)
    
    # Extract config components for worker
    grid_width = cfg["world"]["grid_width"]
    grid_height = cfg["world"]["grid_height"]
    start_pos = cfg["world"]["start_pos"]
    worm_speed = cfg["worm"]["speed"]
    worm_energy_capacity = cfg["worm"]["energy_capacity"]
    worm_metabolic_rate = cfg["worm"]["metabolic_rate"]
    worm_movement_cost = cfg["worm"]["movement_cost"]
    sensor_cfg = cfg.get("worm", {}).get("sensors", {}).get("active", ["current_field"])
    
    # Use config subsections directly (no wrapping)
    feeding_cfg = cfg["food"]
    brain_cfg = cfg["brain"]
    
    # ============================================================
    # 3. SPLIT OFF CONTINUOIS RNG STREAMS FOR VARIANTS
    # ============================================================
    
    # DO THIS ONLY ONCE IN THE BEGINNING OF RUNNING ANYTHING, NOT FOR EVERY GENERATION!!!!!
    variant_decision_seeds = rng_decision.integers(0, 2**32, size=N_VARIANTS, dtype=np.uint32)
    variant_noise_seeds = rng_neuron_noise.integers(0, 2**32, size=N_VARIANTS, dtype=np.uint32)
    
    # ============================================================
    # 4. GENERATE GENOMES
    # ============================================================
    # Generate all genomes before dispatching workers
    genomes = []
    for variant_id in range(N_VARIANTS):
        genome = genome_generator(cfg, rng_seed=WIRING_RANDOMIZATION_SEED + variant_id)
        genomes.append(genome)


    all_lifespans = eval_generation(genomes, cfg, EXPERIMENT_FOLDER, SIMULATION_NAME, ENABLE_PER_RUN_TRACKING, ENABLE_PER_TICK_TRACKING,
                                    ENABLE_HEAT_MAP_TRACKING, rng_world, VIZ_ENABLED, VIZ_BRAIN_ENABLED, VIZ_FPS, VIZ_BRAIN_FPS, N_VARIANTS,
                                    brain_module_name, MAX_TICKS, N_RUNS, grid_width, grid_height, start_pos, worm_speed, worm_energy_capacity, worm_metabolic_rate, worm_movement_cost, sensor_cfg, feeding_cfg, brain_cfg, variant_decision_seeds, variant_noise_seeds)                             

    total_lifespans = sum(len(v) for v in all_lifespans.values())
    print(f"[results] {total_lifespans} lifespans and {N_VARIANTS} variant RNG seed sets collected")


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

