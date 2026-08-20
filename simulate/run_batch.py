# Batch simulation of Byte with randomized wiring variants
# Usage: python -m simulate.run_batch --config configs/experiments/neurons_random_wiring.yaml

import sys
import argparse
import importlib
from pathlib import Path
import yaml
import json
import numpy as np
import h5py
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
        Name of the genome generator (e.g., "random", "lookup_soft", "lookup_hard", "from_file")
    
    Returns
    -------
    callable
        The genome generator function (generate_random_genome, generate_lookup_soft_genome, generate_lookup_hard_genome, generate_genome_from_file, etc.)
    """
    if not genome_type or genome_type.lower() == "none":
        raise ValueError(f"Invalid genome type: '{genome_type}'")
    
    # Map genome type names to actual function names
    function_map = {
        "random": "generate_random_genome",
        "lookup_soft": "generate_lookup_soft_genome",
        "lookup_hard": "generate_lookup_hard_genome",
        "from_file": "generate_genome_from_file",
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


def unflatten_config_from_hdf5(flat_dict):
    """Reconstruct nested config dict from HDF5's granular flattened attributes."""
    import ast

    def deserialize_value(val):
        if not isinstance(val, str):
            return val
        if val.startswith(('[', '{', '(')):
            try:
                return ast.literal_eval(val)
            except (ValueError, SyntaxError):
                pass
        if val.startswith(('[', '{')):
            try:
                return json.loads(val)
            except Exception:
                pass
        return val

    result = {'world': {}, 'food': {}, 'worm': {}, 'brain': {}}

    # New format: food phases stored as a single JSON list under the bare 'food' key.
    # load_and_reconstruct_hdf5_cfg already parses JSON strings starting with '[',
    # so the value here is either a Python list or a JSON string.
    if 'food' in flat_dict:
        food_raw = flat_dict['food']
        if isinstance(food_raw, list):
            result['food'] = food_raw
        elif isinstance(food_raw, str):
            result['food'] = deserialize_value(food_raw)

    for flat_key, value in flat_dict.items():
        if not flat_key.startswith(('world_', 'worm_', 'brain_')):
            continue
        value = deserialize_value(value)
        parts = flat_key.split('_', 1)
        section = parts[0]
        rest = parts[1] if len(parts) > 1 else ""
        if section not in result:
            continue

        if section == 'worm' and rest.startswith('decisionmaking_'):
            result[section].setdefault('decisionmaking', {})[rest.replace('decisionmaking_', '')] = value
        elif section == 'worm' and rest.startswith('sensors_'):
            result[section].setdefault('sensors', {})[rest.replace('sensors_', '')] = value
        elif section == 'brain' and rest.startswith('sensory_mapping_'):
            result[section].setdefault('sensory_mapping', {})[rest.replace('sensory_mapping_', '')] = value
        elif section == 'brain' and rest.startswith('output_mapping_'):
            idx_str = rest.replace('output_mapping_', '')
            try:
                result[section].setdefault('output_mapping', {})[int(idx_str)] = value
            except ValueError:
                result[section].setdefault('output_mapping', {})[idx_str] = value
        else:
            result[section][rest] = value

    return result


def load_and_reconstruct_hdf5_cfg(hdf5_path):
    """Read flattened config attributes from an HDF5 file and return as nested dict."""
    with h5py.File(hdf5_path, 'r') as f:
        hdf5_cfg_flat = {}
        for attr_name, attr_value in f.attrs.items():
            if isinstance(attr_value, str) and attr_value.startswith('['):
                try:
                    attr_value = json.loads(attr_value)
                except Exception:
                    pass
            hdf5_cfg_flat[attr_name] = attr_value
    return unflatten_config_from_hdf5(hdf5_cfg_flat)


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

    # Optional pre-drawn randomness (test-only; absent => live RNGs, see mvb/predrawn.py)
    RANDOMNESS_CFG = experiment_cfg.get("predrawn_randomness", None)
    if RANDOMNESS_CFG is not None and RANDOMNESS_CFG.get("enabled", False):
        if "max_brain_ticks" not in RANDOMNESS_CFG:
            raise KeyError("[ERROR] predrawn_randomness.enabled is true but "
                           "'max_brain_ticks' is missing from the config.")
        print(f"[predrawn] enabled (max_brain_ticks="
              f"{RANDOMNESS_CFG['max_brain_ticks']})")


    # ============================================================
    # 2. DETECT EXECUTION MODE
    # ============================================================
    # Modes:
    #   REPLAY    – genome_type: "from_file"  AND simulation_seed: "from_file"
    #               Exact reproduction of a past simulation run.
    #   BENCHMARK – genome_type: <de novo>   AND simulation_seed: "from_file"
    #               Fresh genomes run on the same world seeds as a reference experiment.
    #   NORMAL    – genome_type: <de novo>   AND simulation_seed: <integer>
    #               Fully fresh simulation, nothing loaded from file.
    #   TEST      – genome_type: "from_file"  AND simulation_seed: <integer>
    #               Genome(s) loaded from file, run on a fresh simulation defined by YAML.

    GENOME_TYPE = experiment_cfg["genome_type"]
    SIMULATION_SEED_RAW = experiment_cfg.get("simulation_seed", 1)
    GENOME_FROM_FILE = GENOME_TYPE.lower() == "from_file"
    SEEDS_FROM_FILE = str(SIMULATION_SEED_RAW).lower() == "from_file"

    pre_computed_seeds_dict = None
    replay_info = None

    # ============================================================
    # 3. TEST MODE: genome from file, fresh simulation from YAML
    # ============================================================
    if GENOME_FROM_FILE and not SEEDS_FROM_FILE:
        print("\n" + "="*80)
        print("[TEST MODE] Loading genomes from HDF5, running with YAML simulation parameters")
        print("="*80)

        from_file_cfg = experiment_cfg.get("from_file_source", {})
        file_folder = from_file_cfg.get("file_folder")
        filename = from_file_cfg.get("filename")
        if not file_folder or not filename:
            raise ValueError("[ERROR] 'from_file_source' section missing 'file_folder' or 'filename'")

        hdf5_source_path = str(Path(file_folder) / f"{filename}.h5")
        if not Path(hdf5_source_path).exists():
            raise FileNotFoundError(f"[ERROR] HDF5 source file not found: {hdf5_source_path}")

        genome_ID = from_file_cfg.get("genome_ID", "all")
        with h5py.File(hdf5_source_path, 'r') as f:
            if genome_ID == "all":
                available_elites = [key for key in f['elite_genomes'].keys() if key.startswith('elite_')]
                selected_elite_ids = [int(k.split('_')[1]) for k in sorted(available_elites)]
            elif isinstance(genome_ID, int):
                selected_elite_ids = [genome_ID]
            elif isinstance(genome_ID, list):
                selected_elite_ids = genome_ID
            else:
                raise ValueError(f"[ERROR] genome_ID must be 'all', int, or list. Got: {genome_ID}")

        genome_generator = load_genome_generator(GENOME_TYPE)
        genomes = []
        for elite_id in selected_elite_ids:
            genome = genome_generator(cfg, elite_id=elite_id, hdf5_path=hdf5_source_path)
            genomes.append(genome)

        N_VARIANTS = len(selected_elite_ids)
        MAX_TICKS = experiment_cfg["max_ticks"]
        N_RUNS = experiment_cfg["n_runs"]

        VIZ_ENABLED = experiment_cfg["viz_enabled"]
        VIZ_FPS = experiment_cfg["viz_fps"]
        VIZ_BRAIN_ENABLED = experiment_cfg["viz_brain_enabled"]
        VIZ_BRAIN_FPS = experiment_cfg["viz_brain_fps"]
        ENABLE_PER_RUN_TRACKING = experiment_cfg["enable_per_run_tracking"]
        ENABLE_PER_TICK_TRACKING = experiment_cfg["enable_per_tick_tracking"]
        ENABLE_HEAT_MAP_TRACKING = experiment_cfg["enable_heat_map_tracking"]

        EXPERIMENT_FOLDER = experiment_cfg["output_folder"]
        SIMULATION_NAME = experiment_cfg["simulation_name"]
        SIMULATION_SEED = int(SIMULATION_SEED_RAW)
        WIRING_RANDOMIZATION_SEED = 0

        genome_id_str = (
            "all" if genome_ID == "all"
            else "-".join(str(x) for x in selected_elite_ids) if isinstance(genome_ID, list)
            else str(genome_ID)
        )
        replay_info = {
            'mode': 'test',
            'source_path': hdf5_source_path,
            'source_filename': filename,
            'genome_ids': genome_id_str,
            'runs_to_load': 'n/a',
        }

    # ============================================================
    # 4. LOAD FROM HDF5 (replay mode or benchmark mode)
    # ============================================================
    if SEEDS_FROM_FILE:
        print("\n" + "="*80)
        if GENOME_FROM_FILE:
            print("[REPLAY MODE] Loading genomes, config, and seeds from HDF5 file")
        else:
            print("[BENCHMARK MODE] Loading simulation parameters and run_seeds from HDF5 file")
        print("="*80)

        from_file_cfg = experiment_cfg.get("from_file_source", {})
        file_folder = from_file_cfg.get("file_folder")
        filename = from_file_cfg.get("filename")
        if not file_folder or not filename:
            raise ValueError("[ERROR] 'from_file_source' section missing 'file_folder' or 'filename'")

        hdf5_source_path = str(Path(file_folder) / f"{filename}.h5")
        if not Path(hdf5_source_path).exists():
            raise FileNotFoundError(f"[ERROR] HDF5 source file not found: {hdf5_source_path}")

        # Reconstruct world/food/worm/brain config from HDF5 attributes
        reconstructed_cfg = load_and_reconstruct_hdf5_cfg(hdf5_source_path)
        for section in ['world', 'food', 'worm', 'brain']:
            if section in reconstructed_cfg and reconstructed_cfg[section]:
                cfg[section] = reconstructed_cfg[section]

        required_sections = ['world', 'food', 'worm', 'brain']
        missing_sections = [s for s in required_sections if s not in cfg or not cfg[s]]
        if missing_sections:
            print(f"[ERROR] Could not reconstruct config sections from HDF5: {missing_sections}")
            sys.exit(1)

        # Load run-level seeds and metadata from HDF5
        with h5py.File(hdf5_source_path, 'r') as f:
            max_ticks_hdf5 = int(f.attrs.get('experiment_max_ticks', experiment_cfg.get("max_ticks", 2000)))
            n_runs_hdf5 = int(f.attrs.get('experiment_n_runs', experiment_cfg.get("n_runs", 1)))
            run_seeds_full = f['elite_genomes/run_seeds'][:]

            if GENOME_FROM_FILE:
                # Also load per-elite seeds for replay mode
                genome_ID = from_file_cfg.get("genome_ID", "all")
                runs_to_load = from_file_cfg.get("runs_to_load", "all")

                if genome_ID == "all":
                    available_elites = [key for key in f['elite_genomes'].keys() if key.startswith('elite_')]
                    selected_elite_ids = [int(k.split('_')[1]) for k in sorted(available_elites)]
                elif isinstance(genome_ID, int):
                    selected_elite_ids = [genome_ID]
                elif isinstance(genome_ID, list):
                    selected_elite_ids = genome_ID
                else:
                    raise ValueError(f"[ERROR] genome_ID must be 'all', int, or list. Got: {genome_ID}")

                if runs_to_load == "all":
                    runs_indices = list(range(n_runs_hdf5))
                elif isinstance(runs_to_load, list):
                    runs_indices = runs_to_load
                else:
                    raise ValueError(f"[ERROR] runs_to_load must be 'all' or list. Got: {runs_to_load}")

                elite_seeds_noise_full = {}
                elite_seeds_decision_full = {}
                for elite_id in selected_elite_ids:
                    elite_group_name = f"elite_genomes/elite_{elite_id}"
                    if elite_group_name not in f:
                        raise ValueError(f"[ERROR] Elite {elite_id} not found in {hdf5_source_path}")
                    elite_seeds_noise_full[elite_id] = f[f'{elite_group_name}/seeds_noise'][:]
                    elite_seeds_decision_full[elite_id] = f[f'{elite_group_name}/seeds_decision'][:]

        # ============================================================
        # 4a. REPLAY MODE: genomes and seeds both from file
        # ============================================================
        if GENOME_FROM_FILE:
            genome_id_is_all = (genome_ID == "all")
            runs_to_load_is_all = (runs_to_load == "all")

            pre_computed_seeds_dict = {}
            for new_variant_id, elite_id in enumerate(selected_elite_ids):
                pre_computed_seeds_dict[new_variant_id] = {
                    'run_seeds': run_seeds_full[runs_indices],
                    'noise_seeds': elite_seeds_noise_full[elite_id][runs_indices],
                    'decision_seeds': elite_seeds_decision_full[elite_id][runs_indices],
                }

            genome_generator = load_genome_generator(GENOME_TYPE)
            genomes = []
            for new_variant_id, elite_id in enumerate(selected_elite_ids):
                genome = genome_generator(cfg, elite_id=elite_id, hdf5_path=hdf5_source_path)
                genomes.append(genome)

            N_VARIANTS = len(selected_elite_ids)
            N_RUNS = len(runs_indices)
            MAX_TICKS = max_ticks_hdf5

            VIZ_ENABLED = experiment_cfg["viz_enabled"]
            VIZ_FPS = experiment_cfg["viz_fps"]
            VIZ_BRAIN_ENABLED = experiment_cfg["viz_brain_enabled"]
            VIZ_BRAIN_FPS = experiment_cfg["viz_brain_fps"]
            ENABLE_PER_RUN_TRACKING = experiment_cfg["enable_per_run_tracking"]
            ENABLE_PER_TICK_TRACKING = experiment_cfg["enable_per_tick_tracking"]
            ENABLE_HEAT_MAP_TRACKING = experiment_cfg["enable_heat_map_tracking"]

            genome_id_str = "all" if genome_id_is_all else (
                "-".join(str(x) for x in selected_elite_ids) if len(selected_elite_ids) > 1
                else str(selected_elite_ids[0])
            )
            runs_str = "all" if runs_to_load_is_all else (
                "-".join(str(x) for x in runs_indices) if len(runs_indices) > 1
                else str(runs_indices[0])
            )
            EXPERIMENT_FOLDER = str(Path(file_folder))
            SIMULATION_NAME = f"{filename}_genomes_{genome_id_str}_runs_{runs_str}"
            WIRING_RANDOMIZATION_SEED = 0
            SIMULATION_SEED = 0

            replay_info = {
                'source_path': hdf5_source_path,
                'source_filename': filename,
                'genome_ids': genome_id_str,
                'runs_to_load': runs_str,
            }

        # ============================================================
        # 4b. BENCHMARK MODE: seeds from file, genomes de novo
        # ============================================================
        else:
            genome_generator = load_genome_generator(GENOME_TYPE)
            if GENOME_TYPE.lower() == "random":
                WIRING_RANDOMIZATION_SEED = experiment_cfg["random_genome"]["wiring_randomization_seed"]
            else:
                WIRING_RANDOMIZATION_SEED = 0

            N_VARIANTS = experiment_cfg["population_size"]
            N_RUNS = n_runs_hdf5
            MAX_TICKS = max_ticks_hdf5

            VIZ_ENABLED = experiment_cfg["viz_enabled"]
            VIZ_FPS = experiment_cfg["viz_fps"]
            VIZ_BRAIN_ENABLED = experiment_cfg["viz_brain_enabled"]
            VIZ_BRAIN_FPS = experiment_cfg["viz_brain_fps"]
            ENABLE_PER_RUN_TRACKING = experiment_cfg["enable_per_run_tracking"]
            ENABLE_PER_TICK_TRACKING = experiment_cfg["enable_per_tick_tracking"]
            ENABLE_HEAT_MAP_TRACKING = experiment_cfg["enable_heat_map_tracking"]

            # Output folder = source file folder; name encodes the origin + BM label
            EXPERIMENT_FOLDER = str(Path(file_folder))
            SIMULATION_NAME = f"{filename}_BM_{experiment_cfg['simulation_name']}"
            # Seed noise/decision RNGs from the wiring seed so benchmarks are reproducible
            SIMULATION_SEED = WIRING_RANDOMIZATION_SEED

            # Stash the file's run_seeds; pre_computed_seeds_dict is built after the
            # RNG streams are initialised (step 7) so that noise/decision seeds are real arrays.
            bm_run_seeds = run_seeds_full

            # replay_info being non-None tells eval_generation to skip the timestamp prefix.
            # It also records the source file in the HDF5 output.
            replay_info = {
                'source_path': hdf5_source_path,
                'source_filename': filename,
                'genome_ids': 'benchmark',
                'runs_to_load': 'all',
            }

    # ============================================================
    # 5. NORMAL MODE: nothing from file
    # ============================================================
    elif not GENOME_FROM_FILE:
        genome_generator = load_genome_generator(GENOME_TYPE)

        if GENOME_TYPE.lower() == "random":
            WIRING_RANDOMIZATION_SEED = experiment_cfg["random_genome"]["wiring_randomization_seed"]
        else:
            WIRING_RANDOMIZATION_SEED = 0

        N_VARIANTS = experiment_cfg["population_size"]
        MAX_TICKS = experiment_cfg["max_ticks"]
        N_RUNS = experiment_cfg["n_runs"]

        VIZ_ENABLED = experiment_cfg["viz_enabled"]
        VIZ_FPS = experiment_cfg["viz_fps"]
        VIZ_BRAIN_ENABLED = experiment_cfg["viz_brain_enabled"]
        VIZ_BRAIN_FPS = experiment_cfg["viz_brain_fps"]
        ENABLE_PER_RUN_TRACKING = experiment_cfg["enable_per_run_tracking"]
        ENABLE_PER_TICK_TRACKING = experiment_cfg["enable_per_tick_tracking"]
        ENABLE_HEAT_MAP_TRACKING = experiment_cfg["enable_heat_map_tracking"]

        EXPERIMENT_FOLDER = experiment_cfg["output_folder"]
        SIMULATION_NAME = experiment_cfg["simulation_name"]
        SIMULATION_SEED = experiment_cfg["simulation_seed"]

    # Extract brain_module_name from config (used in all modes)
    brain_module_name = str(cfg["worm"]["decisionmaking"]["version"])

    # ============================================================
    # 6. VALIDATION & USER CHECKS
    # ============================================================
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

    sorted_phases = sorted(cfg["food"], key=lambda p: p["phase_from"])
    initial_feeding_cfg = sorted_phases[0]
    switch_phases = sorted_phases[1:]
    brain_cfg = cfg["brain"]

    # ============================================================
    # 7. SPLIT OFF CONTINUOUS RNG STREAMS
    # ============================================================
    # Replay mode:    SIMULATION_SEED = 0 (RNGs unused; all seeds from pre_computed_seeds_dict).
    # Benchmark mode: SIMULATION_SEED = WIRING_RANDOMIZATION_SEED; rng_noise/rng_decision used
    #                 for per-variant seeds; rng_world unused (run_seeds from file).
    # Normal/Test mode: SIMULATION_SEED from YAML; all three RNGs used.
    seed_seq_sim = np.random.SeedSequence(int(SIMULATION_SEED))
    streams_sim = seed_seq_sim.spawn(3)
    rng_noise = np.random.default_rng(streams_sim[0])
    rng_decision = np.random.default_rng(streams_sim[1])
    rng_world = np.random.default_rng(streams_sim[2])

    # Benchmark mode: now that RNGs are ready, build pre_computed_seeds_dict.
    # run_seeds come from the source HDF5; noise/decision seeds are generated fresh
    # so each benchmark variant gets its own independent stochastic stream.
    if SEEDS_FROM_FILE and not GENOME_FROM_FILE:
        pre_computed_seeds_dict = {}
        for variant_id in range(N_VARIANTS):
            pre_computed_seeds_dict[variant_id] = {
                'run_seeds': bm_run_seeds,
                'noise_seeds': rng_noise.integers(0, 2**32, size=N_RUNS, dtype=np.uint32),
                'decision_seeds': rng_decision.integers(0, 2**32, size=N_RUNS, dtype=np.uint32),
            }

    # ============================================================
    # 8. GENERATE GENOMES (de novo modes; replay loads from HDF5 above)
    # ============================================================
    if not GENOME_FROM_FILE:
        genomes = []
        for variant_id in range(N_VARIANTS):
            genome = genome_generator(cfg, rng_seed=WIRING_RANDOMIZATION_SEED + variant_id)
            genomes.append(genome)
    # else: genomes already loaded from HDF5 in replay mode block above

    all_lifespans, run_seeds_generated = eval_generation(genomes, cfg, EXPERIMENT_FOLDER, SIMULATION_NAME, ENABLE_PER_RUN_TRACKING, ENABLE_PER_TICK_TRACKING,
                                    ENABLE_HEAT_MAP_TRACKING, VIZ_ENABLED, VIZ_BRAIN_ENABLED, VIZ_FPS, VIZ_BRAIN_FPS, N_VARIANTS,
                                    rng_noise, rng_decision, rng_world, brain_module_name, MAX_TICKS, N_RUNS, grid_width, grid_height, start_pos, worm_speed, worm_energy_capacity, worm_metabolic_rate, worm_movement_cost, sensor_cfg, initial_feeding_cfg, brain_cfg,
                                    pre_computed_seeds_dict=pre_computed_seeds_dict, replay_info=replay_info, switch_phases=switch_phases,
                                    randomness_cfg=RANDOMNESS_CFG)


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

