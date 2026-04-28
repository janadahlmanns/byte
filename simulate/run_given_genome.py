# Run a single genome loaded from an HDF5 file with optional visualization and tracking
# Usage: Configure parameters at the top of this script, then run:
#   python -m simulate.run_given_genome

import sys
import json
import ast
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import h5py
from datetime import datetime

from mvb.simulation_API import eval_generation
from simulate.pause_manager import init_pause_manager, cleanup_pause_manager


# ============================================================
# USER CONFIGURATION - EDIT THESE PARAMETERS
# ============================================================

# HDF5 source file (relative to workspace root, e.g., 'first_ea' or 'algo_vs_neuro')
HDF5_EXPERIMENT_PATH = "first_ea"

# HDF5 filename without extension (e.g., '2026-04-28_11-29-54_random_100')
HDF5_FILENAME = "2026-04-28_11-12-39_random_100"

# Elite genome ID to run (as integer index, e.g., 0, 1, 2, ...)
ELITE_ID = 0

# Number of runs to perform:
#   - Use 'all' to run all runs from the loaded file's config
#   - Use an integer like 16 to run only that specific run (0-indexed)
# Note: If the file has fewer runs than requested, the script will error
RUN = "all"  # or e.g., 16

# Visualization and tracking flags
VIZ_ENABLED = False
VIZ_FPS = 60
VIZ_BRAIN_ENABLED = False
VIZ_BRAIN_FPS = 10

ENABLE_PER_RUN_TRACKING = False
ENABLE_PER_TICK_TRACKING = False
ENABLE_HEAT_MAP_TRACKING = False

# ============================================================
# END USER CONFIGURATION
# ============================================================


@dataclass
class GenomeContainer:
    """Container for genome data loaded from HDF5."""
    connection_weights: np.ndarray
    modulation_spec: dict
    tonic_activations: np.ndarray
    eta: float
    
    def __getitem__(self, key: str):
        """Support dict-like subscripting for compatibility with genome processing code."""
        if key == "connection_weights":
            return self.connection_weights
        elif key == "modulation_spec":
            return self.modulation_spec
        elif key == "tonic_activations":
            return self.tonic_activations
        elif key == "eta":
            return self.eta
        else:
            raise KeyError(f"GenomeContainer has no key '{key}'")


def load_config_from_hdf5(hdf5_path: str) -> dict:
    """
    Load essential config parameters directly from HDF5 attributes.
    
    Rather than trying to unflatten the entire config (which loses info about
    which underscores are literal vs hierarchy), we extract the specific
    parameters needed for simulation and reconstruct a minimal config dict.
    
    Args:
        hdf5_path: Path to the HDF5 file
    
    Returns:
        Config dict with necessary sections for simulation
    """
    with h5py.File(hdf5_path, 'r') as f:
        attributes = dict(f.attrs)
    
    # Helper to extract attributes with a given prefix
    def get_prefixed(prefix: str) -> dict:
        """Extract all attributes starting with prefix_"""
        result = {}
        search_prefix = f"{prefix}_"
        for key, value in attributes.items():
            if key.startswith(search_prefix):
                sub_key = key[len(search_prefix):]
                # Parse string lists
                if isinstance(value, str) and value.startswith('['):
                    try:
                        value = ast.literal_eval(value)
                    except (ValueError, SyntaxError):
                        pass
                result[sub_key] = value
        return result
    
    # Reconstruct nested dicts for known config sections
    config = {
        "experiment": get_prefixed("experiment"),
        "worm": get_prefixed("worm"),
        "brain": get_prefixed("brain"),
        "world": get_prefixed("world"),
        "food": get_prefixed("food"),
    }
    
    # Post-process brain config to handle nested structures
    # The HDF5 keys like "brain_output_mapping_5" should become output_mapping: {5: value}
    brain_cfg = config["brain"]
    processed_brain = {}
    
    # Handle known nested structures in brain config
    output_mapping = {}
    sensory_mapping = {}  # flat dict: sense_key -> (neuron_id, weight, reliability)
    
    for key, value in brain_cfg.items():
        if key.startswith("output_mapping_"):
            idx = key.split("_")[-1]
            output_mapping[int(idx)] = value
        elif key.startswith("sensory_mapping_"):
            # Flatten sensory_mapping: "sensory_mapping_food_east" -> "food_east" -> tuple
            sense_key = key.replace("sensory_mapping_", "")
            # value should be [neuron_id, weight, reliability]
            if isinstance(value, list) and len(value) == 3:
                sensory_mapping[sense_key] = tuple(value)
            else:
                sensory_mapping[sense_key] = value
        else:
            processed_brain[key] = value
    
    # Reconstruct output_mapping as nested structure
    if output_mapping:
        processed_brain["output_mapping"] = output_mapping
    
    # Keep sensory_mapping flat
    if sensory_mapping:
        processed_brain["sensory_mapping"] = sensory_mapping
    
    # Convert string values to proper types where needed
    if "n_neurons" in processed_brain:
        processed_brain["n_neurons"] = int(processed_brain["n_neurons"])
    
    config["brain"] = processed_brain
    
    # Post-process worm config to handle nested structures
    worm_cfg = config["worm"]
    processed_worm = {}
    decisionmaking_cfg = {}
    sensors_cfg = {}
    
    for key, value in worm_cfg.items():
        if key.startswith("decisionmaking_"):
            sub_key = key.replace("decisionmaking_", "")
            decisionmaking_cfg[sub_key] = value
        elif key.startswith("sensors_"):
            sub_key = key.replace("sensors_", "")
            sensors_cfg[sub_key] = value
        else:
            processed_worm[key] = value
    
    if decisionmaking_cfg:
        processed_worm["decisionmaking"] = decisionmaking_cfg
    if sensors_cfg:
        processed_worm["sensors"] = sensors_cfg
    
    # Convert string booleans
    if "brain" in decisionmaking_cfg:
        decisionmaking_cfg["brain"] = decisionmaking_cfg["brain"] in ("True", "true", "1", True)
    
    config["worm"] = processed_worm
    
    # Similar type conversions for other sections
    if "n_runs" in config["experiment"]:
        config["experiment"]["n_runs"] = int(config["experiment"]["n_runs"])
    if "max_ticks" in config["experiment"]:
        config["experiment"]["max_ticks"] = int(config["experiment"]["max_ticks"])
    if "simulation_seed" in config["experiment"]:
        config["experiment"]["simulation_seed"] = int(config["experiment"]["simulation_seed"])
    
    if "generation_seed" in config["world"]:
        config["world"]["generation_seed"] = int(config["world"]["generation_seed"])
    
    # Post-process food config to handle nested feeding_paradigm structure
    food_cfg = config["food"]
    processed_food = {}
    feeding_paradigm_cfg = {}
    
    for key, value in food_cfg.items():
        if key.startswith("feeding_paradigm_"):
            sub_key = key.replace("feeding_paradigm_", "")
            # Convert string bools
            if isinstance(value, str):
                value = value in ("True", "true", "1", True)
            feeding_paradigm_cfg[sub_key] = value
        else:
            processed_food[key] = value
    
    if feeding_paradigm_cfg:
        processed_food["feeding_paradigm"] = feeding_paradigm_cfg
    
    config["food"] = processed_food
    
    return config


def load_genome_from_hdf5(hdf5_path: str, elite_id: int) -> GenomeContainer:
    """
    Load a single genome from HDF5 elite_genomes group.
    
    Args:
        hdf5_path: Path to the HDF5 file
        elite_id: Index of the elite genome (0-indexed, e.g., elite_0, elite_1, ...)
    
    Returns:
        GenomeContainer with the loaded genome
    
    Raises:
        KeyError: If the requested elite_id doesn't exist
    """
    with h5py.File(hdf5_path, 'r') as f:
        elite_group_name = f"elite_genomes/elite_{elite_id}"
        
        if elite_group_name not in f:
            available = [key for key in f["elite_genomes"].keys()]
            raise KeyError(
                f"Elite genome '{elite_group_name}' not found in HDF5 file. "
                f"Available: {available}"
            )
        
        elite_group = f[elite_group_name]
        
        # Load genome components
        connection_weights = np.array(elite_group["connection_weights"])
        tonic_activations = np.array(elite_group["tonic_activations"])
        
        # eta can be stored as attribute or dataset
        if 'eta' in elite_group.attrs:
            eta = float(elite_group.attrs['eta'])
        else:
            eta_data = np.array(elite_group["eta"])
            eta = float(eta_data.flat[0]) if eta_data.size > 0 else 0.0
        
        # Load modulation_spec if it exists
        modulation_spec = {}
        if "modulation_spec" in elite_group:
            mod_data = elite_group["modulation_spec"][()]
            for record in mod_data:
                src = int(record['source'])
                tgt = int(record['target'])
                mod_neuron = int(record['modulating_neuron'])
                mod_weight = float(record['modulation_weight'])
                
                key = (src, tgt)
                if key not in modulation_spec:
                    modulation_spec[key] = []
                modulation_spec[key].append((mod_neuron, mod_weight))
        
        return GenomeContainer(
            connection_weights=connection_weights,
            modulation_spec=modulation_spec,
            tonic_activations=tonic_activations,
            eta=eta
        )


def resolve_hdf5_path(experiment_path: str, filename: str) -> str:
    """
    Resolve HDF5 file path from experiment path and filename.
    
    Args:
        experiment_path: Relative path like 'first_ea' or 'algo_vs_neuro'
        filename: Base filename like '2026-04-28_11-29-54_random_100'
    
    Returns:
        Full path to the HDF5 file (with .h5 extension)
    
    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    full_path = Path("data") / experiment_path / f"{filename}.h5"
    if not full_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {full_path}")
    return str(full_path)


def resolve_output_path(experiment_path: str, filename: str, elite_id: int) -> str:
    """
    Resolve output HDF5 file path.
    
    Args:
        experiment_path: Relative path like 'first_ea'
        filename: Base filename like '2026-04-28_11-29-54_random_100'
        elite_id: Elite genome ID
    
    Returns:
        Full path to the output HDF5 file (with genome_X.h5 naming)
    """
    output_filename = f"{filename}_genome_{elite_id}.h5"
    output_path = Path("data") / experiment_path / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return str(output_path)


def reconstruct_full_config(config_dict: dict) -> dict:
    """
    Reconstruct the full config dict with proper nesting for special keys.
    
    This handles cases where numeric indices were stored as part of keys,
    converting them back to proper nested structures (e.g., lists or dicts).
    
    Args:
        config_dict: Partially reconstructed config from HDF5
    
    Returns:
        Full reconstructed config with proper nested structures
    """
    # This is a simplified version; for more complex configs you may need
    # additional logic to convert numeric keys back to arrays or dicts
    return config_dict


def validate_run_parameter(run_param, n_runs_in_file: int) -> int:
    """
    Validate and resolve the RUN parameter.
    
    Args:
        run_param: Either 'all' string or an integer
        n_runs_in_file: Number of runs stored in the HDF5 file
    
    Returns:
        Number of runs to execute
    
    Raises:
        ValueError: If run_param is invalid or out of range
    """
    if isinstance(run_param, str) and run_param.lower() == 'all':
        return n_runs_in_file
    
    try:
        run_id = int(run_param)
        if run_id < 0 or run_id >= n_runs_in_file:
            raise ValueError(
                f"RUN parameter {run_id} is out of range. "
                f"File has {n_runs_in_file} runs (0-{n_runs_in_file - 1})."
            )
        return 1  # Run only that specific run
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"RUN parameter must be either 'all' or an integer (0-{n_runs_in_file - 1}). "
            f"Got: {run_param}"
        )


def save_single_genome_run_to_hdf5(output_path: str, genome: GenomeContainer, 
                                   config: dict, lifespan_data: dict, elite_id: int):
    """
    Save the results of running a single genome to HDF5.
    
    Args:
        output_path: Path to output HDF5 file
        genome: GenomeContainer with the genome data
        config: Full configuration dict
        lifespan_data: Dict with lifespan results from eval_generation
        elite_id: The elite ID that was run
    """
    with h5py.File(output_path, 'w') as f:
        # Save configuration as attributes
        f.attrs['elite_id'] = elite_id
        f.attrs['source_hdf5_path'] = str(resolve_hdf5_path(HDF5_EXPERIMENT_PATH, HDF5_FILENAME))
        f.attrs['timestamp'] = datetime.now().isoformat()
        
        # Save genome data
        genome_group = f.create_group("genome")
        genome_group.create_dataset("connection_weights", data=genome.connection_weights)
        genome_group.create_dataset("tonic_activations", data=genome.tonic_activations)
        genome_group.create_dataset("eta", data=np.array([genome.eta]))
        
        # Save modulation spec if it exists
        if genome.modulation_spec:
            mod_records = [
                (int(src), int(tgt), int(mod_neuron), float(mod_weight))
                for (src, tgt), modulators in genome.modulation_spec.items()
                for mod_neuron, mod_weight in modulators
            ]
            if mod_records:
                mod_dtype = np.dtype([
                    ('source', np.int32),
                    ('target', np.int32),
                    ('modulating_neuron', np.int32),
                    ('modulation_weight', np.float32)
                ])
                genome_group.create_dataset("modulation_spec", data=np.array(mod_records, dtype=mod_dtype))
        
        # Save lifespan results
        results_group = f.create_group("results")
        for variant_id, lifespans in lifespan_data.items():
            results_group.create_dataset(f"variant_{variant_id}_lifespans", 
                                        data=np.array(lifespans, dtype=np.float32))


def main():
    # ============================================================
    # 1. RESOLVE AND LOAD SOURCE HDF5 FILE
    # ============================================================
    try:
        hdf5_source_path = resolve_hdf5_path(HDF5_EXPERIMENT_PATH, HDF5_FILENAME)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}\n")
        sys.exit(1)
    
    print(f"[INFO] Loading source HDF5: {hdf5_source_path}")
    
    # ============================================================
    # 2. LOAD CONFIGURATION FROM HDF5
    # ============================================================
    config = load_config_from_hdf5(hdf5_source_path)
    
    experiment_cfg = config.get("experiment", {})
    brain_module_name = str(config.get("worm", {}).get("decisionmaking", {}).get("version", "default"))
    
    # ============================================================
    # 3. LOAD GENOME FROM HDF5
    # ============================================================
    try:
        genome = load_genome_from_hdf5(hdf5_source_path, ELITE_ID)
        print(f"[INFO] Loaded genome from elite_{ELITE_ID}")
    except KeyError as e:
        print(f"\n[ERROR] {e}\n")
        sys.exit(1)
    
    # ============================================================
    # 4. DETERMINE N_RUNS AND VALIDATE RUN PARAMETER
    # ============================================================
    with h5py.File(hdf5_source_path, 'r') as f:
        n_runs_in_file = int(f.attrs.get("experiment_n_runs", 1))
    
    try:
        num_runs = validate_run_parameter(RUN, n_runs_in_file)
    except ValueError as e:
        print(f"\n[ERROR] {e}\n")
        sys.exit(1)
    
    if isinstance(RUN, str) and RUN.lower() == 'all':
        print(f"[INFO] Running all {num_runs} runs")
    else:
        print(f"[INFO] Running run #{RUN} only")
    
    # ============================================================
    # 5. BUILD RNG STREAMS (from config seeds)
    # ============================================================
    simulation_seed = int(experiment_cfg.get("simulation_seed", 42))
    generation_seed = int(config.get("world", {}).get("generation_seed", 42))
    
    seed_seq_sim = np.random.SeedSequence(simulation_seed)
    streams_sim = seed_seq_sim.spawn(2)
    rng_decision = np.random.default_rng(streams_sim[0])
    rng_neuron_noise = np.random.default_rng(streams_sim[1])
    
    seed_seq_gen = np.random.SeedSequence(generation_seed)
    rng_world = np.random.default_rng(seed_seq_gen.spawn(1)[0])
    
    # ============================================================
    # 6. GENERATE VARIANT SEEDS FOR THIS SINGLE GENOME
    # ============================================================
    variant_decision_seed = rng_decision.integers(0, 2**32, dtype=np.uint32)
    variant_noise_seed = rng_neuron_noise.integers(0, 2**32, dtype=np.uint32)
    
    variant_decision_seeds = np.array([variant_decision_seed], dtype=np.uint32)
    variant_noise_seeds = np.array([variant_noise_seed], dtype=np.uint32)
    
    # ============================================================
    # 7. EXTRACT SIMULATION PARAMETERS FROM CONFIG
    # ============================================================
    max_ticks = int(experiment_cfg.get("max_ticks", 1000))
    
    grid_width = int(config.get("world", {}).get("grid_width", 100))
    grid_height = int(config.get("world", {}).get("grid_height", 100))
    start_pos = config.get("world", {}).get("start_pos", [50, 50])
    
    worm_speed = float(config.get("worm", {}).get("speed", 1.0))
    worm_energy_capacity = float(config.get("worm", {}).get("energy_capacity", 100.0))
    worm_metabolic_rate = float(config.get("worm", {}).get("metabolic_rate", 1.0))
    worm_movement_cost = float(config.get("worm", {}).get("movement_cost", 1.0))
    
    sensor_cfg = config.get("worm", {}).get("sensors", {}).get("active", ["current_field"])
    feeding_cfg = config.get("food", {})
    brain_cfg = config.get("brain", {})
    
    # ============================================================
    # 8. CREATE GENOME LIST (single genome wrapped in list for eval_generation)
    # ============================================================
    # eval_generation expects a list of genome objects; we need to adapt our loaded genome
    # to match the expected interface
    genomes = [genome]
    
    # ============================================================
    # 9. RUN EVALUATION
    # ============================================================
    print(f"[INFO] Running genome {ELITE_ID} with {num_runs} run(s)...")
    
    all_lifespans = eval_generation(
        genomes, config, 
        str(Path(hdf5_source_path).parent),  # experiment folder
        f"genome_{ELITE_ID}",  # simulation name
        ENABLE_PER_RUN_TRACKING, ENABLE_PER_TICK_TRACKING, ENABLE_HEAT_MAP_TRACKING,
        rng_world, 
        VIZ_ENABLED, VIZ_BRAIN_ENABLED, VIZ_FPS, VIZ_BRAIN_FPS, 
        1,  # n_variants (just 1)
        brain_module_name, 
        max_ticks, 
        num_runs,
        grid_width, grid_height, start_pos, 
        worm_speed, worm_energy_capacity, worm_metabolic_rate, worm_movement_cost,
        sensor_cfg, feeding_cfg, brain_cfg, 
        variant_decision_seeds, variant_noise_seeds
    )
    
    # ============================================================
    # 10. SAVE RESULTS TO OUTPUT HDF5
    # ============================================================
    output_path = resolve_output_path(HDF5_EXPERIMENT_PATH, HDF5_FILENAME, ELITE_ID)
    print(f"[INFO] Saving results to: {output_path}")
    
    save_single_genome_run_to_hdf5(output_path, genome, config, all_lifespans, ELITE_ID)
    
    print(f"[SUCCESS] Completed running genome {ELITE_ID}")
    print(f"Total lifespans collected: {sum(len(v) for v in all_lifespans.values())}")
    print("Done.\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[FATAL ERROR] {type(e).__name__}: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
