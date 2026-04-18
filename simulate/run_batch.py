# Batch simulation of Byte with randomized wiring variants
# Usage: python -m simulate.run_batch --config configs/experiments/neurons_random_wiring.yaml

import sys
import argparse
import importlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager

import yaml
import numpy as np

from mvb.world import World, WorldConfig
from mvb.feeding import FeedingConfig, seed_food
from mvb.worm import Worm, WormConfig
from mvb.world_renderer_qt import QtRenderer
from mvb.brain_renderer_qt import BrainQtRenderer
from .pause_manager import init_pause_manager, cleanup_pause_manager, get_pause_manager, PauseManagerExit
from .hdf5_utils import (
    create_hdf5_file,
    save_variant_summary_to_hdf5,
    save_wiring_to_hdf5,
    save_modulation_to_hdf5,
    save_heatmaps_to_hdf5,
    save_per_tick_to_hdf5,
    save_genome_properties_to_hdf5,
)


# --- COMMAND-LINE ARGUMENT PARSING ---

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
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# --- HELPER FUNCTIONS ---

def get_num_workers(viz_enabled, viz_brain_enabled):
    """Determine number of worker processes and handle viz/serial execution.
    
    If visualization is enabled, forces serial execution and initializes pause manager.
    Otherwise, reserves 2 cores for system tasks and returns worker count.
    Returns None if system has ≤2 cores or if visualization is enabled (force serial).
    
    Args:
        viz_enabled: Whether world visualization is enabled
        viz_brain_enabled: Whether brain visualization is enabled
    
    Returns:
        Number of workers to use, or None for serial execution
    """
    # Force serial execution if any visualization is enabled
    if viz_enabled or viz_brain_enabled:
        print("[INFO] Visualization enabled. Running serially.")
        init_pause_manager()
        return None
    
    # Determine parallel worker count
    try:
        available_cores = os.cpu_count()
        if available_cores is None or available_cores <= 2:
            print("[INFO] Insufficient CPU cores. Running serially.")
            return None
        
        num_workers = max(1, available_cores - 2)
        print(f"[INFO] Parallel execution on {num_workers} cores ({available_cores} total).")
        return num_workers
    except Exception:
        print("[INFO] Insufficient CPU cores. Running serially.")
        return None

def build_rng_streams(seed: int, has_brain: bool):
    """Build separate RNG streams for simulation aspects."""
    seed = int(seed)
    rng_food = np.random.default_rng(seed)
    rng_decision = np.random.default_rng(seed)
    rng_neuron_noise = np.random.default_rng(seed) if has_brain else None
    return rng_food, rng_decision, rng_neuron_noise

# --- helpers ---

def load_brain_module(version: str):
    module_name = f"mvb.brains.decisionmaking_{version}"
    module = importlib.import_module(module_name)
    if not hasattr(module, "decide"):
        raise AttributeError(f"{module_name} has no decide()")
    return module

def load_brain_init(brain_init_name: str, wiring_seed: int = None, **wiring_params):
    """Load brain initialization config. Returns brain spec or None."""
    if brain_init_name.lower() == "none" or not brain_init_name:
        return None
    module_name = f"configs.brain_init_{brain_init_name}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise ImportError(f"Could not find brain init module '{module_name}'.")
    if not hasattr(module, "build_brain_spec"):
        raise AttributeError(f"Brain init module '{module_name}' has no 'build_brain_spec' function.")
    return module.build_brain_spec(wiring_seed=wiring_seed, **wiring_params)

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

def make_world(grid_width, grid_height, start_pos, rng_seed):
    return World(
        WorldConfig(
            grid_width=int(grid_width),
            grid_height=int(grid_height),
            start_pos=tuple(start_pos),
            rng_seed=int(rng_seed),
        ),
    )

def make_feeding_cfg(feeding_paradigm, initial_fraction_per_cell, regrow_time):
    return FeedingConfig(
        feeding_paradigm=feeding_paradigm,
        initial_fraction_per_cell=initial_fraction_per_cell,
        regrow_time=regrow_time,
    )

def make_worm(world, speed, energy_capacity, metabolic_rate, movement_cost):
    return Worm(
        WormConfig(
            speed=int(speed),
            energy_capacity=int(energy_capacity),
            metabolic_rate=int(metabolic_rate),
            movement_cost=int(movement_cost),
        ),
        world,
    )

def make_sensor_cfg(cfg_yaml):
    return cfg_yaml.get("worm", {}).get("sensors", {}).get("active", ["current_field"])

def make_decision_cfg(cfg_yaml):
    return str(cfg_yaml["worm"]["decisionmaking"]["version"])

def reset_sim(world, feeding_cfg, rng_food, worm):
    world.reset_food()
    seed_food(world, feeding_cfg, rng_food)
    worm.reset()


def _rename_world_config_keys(cfg: dict) -> dict:
    """
    Rename world config keys to include prefixes for HDF5 attribute clarity.
    
    Transforms keys like:
      world.grid_width → world_grid_width
      food.feeding_paradigm.initial → feeding_initial
      worm.speed → worm_speed
      sensors.active → worm_sensors_active
      decisionmaking.version → decisionmaking_version
    
    Skips the 'viz' section entirely.
    """
    renamed = {}
    
    # Process world section
    if "world" in cfg:
        for key, val in cfg["world"].items():
            renamed[f"world_{key}"] = val
    
    # Process food section - flatten feeding_paradigm keys
    if "food" in cfg:
        food_cfg = cfg["food"]
        if "feeding_paradigm" in food_cfg:
            for key, val in food_cfg["feeding_paradigm"].items():
                renamed[f"feeding_{key}"] = val
    
    # Process worm section
    if "worm" in cfg:
        for key, val in cfg["worm"].items():
            renamed[f"worm_{key}"] = val
    
    # Process sensors section
    if "sensors" in cfg:
        for key, val in cfg["sensors"].items():
            renamed[f"worm_sensors_{key}"] = val
    
    # Process decisionmaking section
    if "decisionmaking" in cfg:
        for key, val in cfg["decisionmaking"].items():
            renamed[f"decisionmaking_{key}"] = val
    
    return renamed


def get_connection_weight(brain_module, src_neuron_id: int, tgt_neuron_id: int) -> float:
    """Get weight of specific neuron-to-neuron connection from brain state.
    
    Args:
        brain_module: The brain module (e.g., decisionmaking_plasticity)
        src_neuron_id: Source neuron ID
        tgt_neuron_id: Target neuron ID
    
    Returns:
        Current weight of the connection, or 0.0 if not found
    """
    if not hasattr(brain_module, '_brain_state'):
        return 0.0
    
    brain_state = brain_module._brain_state
    if brain_state is None:
        return 0.0
    
    # Search through connections for the one from src to tgt
    for conn in brain_state.connections:
        # Check if this is a neuron-to-neuron connection (not input source)
        if hasattr(conn.source, 'id'):
            if conn.source.id == src_neuron_id:
                # Check target by finding which neuron has this in its incoming list
                for neuron in brain_state.neurons:
                    if neuron.id == tgt_neuron_id and conn in neuron.incoming:
                        return conn.weight
    
    return 0.0


# --- output + metrics ---

def make_experiment_dir(experiment_folder: str, simulation_name: str) -> Path:
    """Create HDF5 file path for experiment.
    
    Args:
        experiment_folder: Base folder for experiment output
        simulation_name: Name of the simulation
    
    Returns:
        Path to HDF5 file for saving all results.
    """
    base = Path(experiment_folder)
    base.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    hdf5_path = base / f"{ts}_{simulation_name}.h5"
    
    return hdf5_path





@dataclass
class MetricsRecorder:
    per_tick_data: np.ndarray = None
    per_tick_count: int = 0
    connections_to_track: list[tuple] = None
    start_y: int = 0  # Starting Y position for manhattan distance calculation
    start_x: int = 0  # Starting X position for manhattan distance calculation
    prev_y: int = 0
    prev_x: int = 0
    prev_eats: int = 0  # Track food consumption this tick
    prev_action: tuple = None  # Track which movement happened
    grid_height: int = 0  # World grid height for heatmap indexing
    grid_width: int = 0  # World grid width for heatmap indexing
    staying_heatmap: np.ndarray = None  # 2D array (height, width) - field ticks spent
    moves_north: int = 0
    moves_south: int = 0
    moves_east: int = 0
    moves_west: int = 0
    food_sensed_north: int = 0
    food_sensed_east: int = 0
    food_sensed_south: int = 0
    food_sensed_west: int = 0
    prev_on_food: bool = False  # Track if worm was on food before the action
    prev_action_was_decision: bool = False  # Track if previous action was a decision
    decisions: int = 0
    correct_decisions: int = 0
    enable_per_tick_tracking: bool = True  # Whether to track per-tick metrics
    enable_heat_map_tracking: bool = True  # Whether to track heatmaps

    @classmethod
    def empty(cls, worm: Worm, genome, enable_per_run_tracking=True, enable_per_tick_tracking=True, enable_heat_map_tracking=True, max_ticks=2000):
        """Initialize recorder with genome and worm state.
        
        If enable_per_run_tracking is False, only lifetime metrics are tracked.
        
        Args:
            worm: Worm instance
            genome: Genome dict with connection_weights
            enable_per_run_tracking: Whether to track per-run metrics
            enable_per_tick_tracking: Whether to track per-tick data
            enable_heat_map_tracking: Whether to track heatmaps
            max_ticks: Maximum simulation ticks
        """
        connection_weights = genome["connection_weights"]
        
        connections_to_track = []
        for src in range(connection_weights.shape[0]):
            for tgt in range(connection_weights.shape[1]):
                if connection_weights[src, tgt, 0] != 0.0:
                    connections_to_track.append((src, tgt))
        
        grid_height = worm.world.cfg.grid_height
        grid_width = worm.world.cfg.grid_width
                
        if enable_per_run_tracking:
            kwargs = {
            'per_tick_count': 0,
            'connections_to_track': connections_to_track,
            'start_y': worm.y,
            'start_x': worm.x,
            'prev_y': worm.y,
            'prev_x': worm.x,
            'prev_eats': worm.eats,
            'prev_action': None,
            'grid_height': grid_height,
            'grid_width': grid_width,
            'moves_north': 0,
            'moves_south': 0,
            'moves_east': 0,
            'moves_west': 0,
            'food_sensed_north': 0,
            'food_sensed_east': 0,
            'food_sensed_south': 0,
            'food_sensed_west': 0,
            'prev_on_food': False,
            'prev_action_was_decision': False,
            'decisions': 0,
            'correct_decisions': 0,
            'enable_per_tick_tracking': enable_per_tick_tracking,
            'enable_heat_map_tracking': enable_heat_map_tracking,
            }
            if enable_per_tick_tracking:
                dtype_fields = [
                    ('tick', 'i4'),
                    ('food_sensed_N', 'u1'),
                    ('food_sensed_E', 'u1'),
                    ('food_sensed_S', 'u1'),
                    ('food_sensed_W', 'u1'),
                    ('movement', 'S4'),
                    ('food_consumed', 'u1'),
                    ('energy', 'f4'),
                    ('manhattan_dist', 'u2'),
                    ('decision_made', 'u1'),
                ]
                for src, tgt in connections_to_track:
                    dtype_fields.append((f'{src}_{tgt}', 'f4'))
                kwargs['per_tick_data'] = np.zeros(max_ticks, dtype=dtype_fields)
            
            if enable_heat_map_tracking:
                staying_heatmap = np.zeros((grid_height, grid_width), dtype=np.int32)
                kwargs['staying_heatmap'] = staying_heatmap
            return cls(**kwargs)
        else:
            return cls()        
        
    def reset(self):
        """Reset all counters and state for a new run while keeping arrays allocated."""
        self.per_tick_count = 0
        self.prev_y = self.start_y
        self.prev_x = self.start_x
        self.prev_eats = 0
        self.prev_action = None
        self.moves_north = 0
        self.moves_south = 0
        self.moves_east = 0
        self.moves_west = 0
        self.food_sensed_north = 0
        self.food_sensed_east = 0
        self.food_sensed_south = 0
        self.food_sensed_west = 0
        self.prev_on_food = False
        self.prev_action_was_decision = False
        self.decisions = 0
        self.correct_decisions = 0
        # Clear array data if allocated
        if self.per_tick_data is not None:
            self.per_tick_data.fill(0)
        if self.staying_heatmap is not None:
            self.staying_heatmap.fill(0)

    def record(self, worm: Worm):
        """Record metrics for this tick."""
        dy = worm.y - self.prev_y
        dx = worm.x - self.prev_x
        
        if dy > 0:
            self.moves_south += 1
        elif dy < 0:
            self.moves_north += 1
        
        if dx > 0:
            self.moves_east += 1
        elif dx < 0:
            self.moves_west += 1
        
        # Check if food was sensed in any direction (regardless of other directions)
        sense = getattr(worm, "sensory_information", {})
        food_north = sense.get("food_north", 0.0) > 0.0
        food_east = sense.get("food_east", 0.0) > 0.0
        food_south = sense.get("food_south", 0.0) > 0.0
        food_west = sense.get("food_west", 0.0) > 0.0
        
        # Count food sensing occurrences
        if food_north:
            self.food_sensed_north += 1
        if food_east:
            self.food_sensed_east += 1
        if food_south:
            self.food_sensed_south += 1
        if food_west:
            self.food_sensed_west += 1
        
        # Track decision-making accuracy using new rules:
        # Rule 1: Stay is a decision if worm is NOT on food CURRENTLY (before the stay action)
        # Rule 2: Movement is a decision if ANY food is sensed on the 5 current sensing fields
        # Rule 3: Decision is correct if it was deemed a decision AND worm is on food in NEXT tick
        on_food = sense.get("on_food", 0) > 0
        any_food_sensed = food_north or food_east or food_south or food_west
        
        # Check if the previous action was correct (in this tick after previous action)
        if self.prev_action_was_decision and on_food:
            self.correct_decisions += 1
        
        # Determine if current action (about to happen) is a decision
        # We use prev_on_food because the decision is made BEFORE the action
        stayed = (dy == 0 and dx == 0)
        
        is_decision = False
        if stayed:
            # Stay is a decision only if worm is NOT on food CURRENTLY (before the stay)
            is_decision = not self.prev_on_food
        else:
            # Movement is a decision if ANY food is sensed on the 5 current sensing fields
            is_decision = any_food_sensed
        
        if is_decision:
            self.decisions += 1
        
        # Update state for next tick
        self.prev_on_food = on_food
        self.prev_action_was_decision = is_decision
        
        # Track comprehensive per-tick data (if enabled)
        if self.enable_per_tick_tracking and self.per_tick_data is not None:
            # Determine movement direction from previous action
            movement_str = "stay"
            if self.prev_action is not None:
                if self.prev_action[0] == "move":
                    move_y, move_x = self.prev_action[1]
                    if move_y < self.prev_y:
                        movement_str = "N"
                    elif move_y > self.prev_y:
                        movement_str = "S"
                    elif move_x > self.prev_x:
                        movement_str = "E"
                    elif move_x < self.prev_x:
                        movement_str = "W"
            
            # Check if food was consumed this tick
            food_consumed = 1 if worm.eats > self.prev_eats else 0
            
            # Calculate manhattan distance from start position
            manhattan_dist = abs(worm.y - self.start_y) + abs(worm.x - self.start_x)
            
            # Check if decision was made (action is not None)
            decision_made = 1 if worm.action is not None else 0
            
            # Populate array at current tick index
            tick_idx = worm.ticks
            self.per_tick_data[tick_idx] = (
                worm.ticks,
                int(food_north),
                int(food_east),
                int(food_south),
                int(food_west),
                movement_str,
                food_consumed,
                worm.energy,
                manhattan_dist,
                decision_made,
            ) + tuple(get_connection_weight(worm.brain, src, tgt) for src, tgt in self.connections_to_track)
            
            self.per_tick_count = tick_idx + 1
        
        if self.enable_heat_map_tracking and self.staying_heatmap is not None:
            self.staying_heatmap[worm.y, worm.x] += 1
        
        self.prev_y = worm.y
        self.prev_x = worm.x
        self.prev_eats = worm.eats
        self.prev_action = worm.action



# Implementation of section 5b (Simulate runs)
def run_variant_worker(
    variant_id,
    brain_module_name,
    genome,
    viz_enabled,
    enable_per_run_tracking,
    enable_per_tick_tracking,
    enable_heat_map_tracking,
    max_ticks,
    n_runs,
    viz_fps,
    viz_brain_enabled,
    viz_brain_fps,
    grid_width,
    grid_height,
    start_pos,
    world_rng_seed,
    worm_speed,
    worm_energy_capacity,
    worm_metabolic_rate,
    worm_movement_cost,
    sensor_cfg,
    feeding_paradigm,
    feeding_initial_fraction_per_cell,
    feeding_regrow_time,
    brain_n_neurons,
    brain_threshold,
    brain_noise_level,
    brain_sensory_mapping,
    brain_output_mapping,
    brain_max_decision_delay,
    rng_seed,
    has_brain_config,
    hdf5_path=None,
    hdf5_lock=None,
):
    """Execute a single variant's simulation runs and write data directly to HDF5.
    
    Args:
        variant_id: Index of this variant
        brain_module_name: Name of brain module to import
        genome: Pre-generated genome dict with connection_weights, modulation_spec, etc.
        viz_enabled: Whether to create and display renderer visualization
        enable_per_run_tracking: Whether to track per-run metrics
        enable_per_tick_tracking: Whether to track per-tick data
        enable_heat_map_tracking: Whether to track heatmaps
        max_ticks: Maximum simulation ticks
        n_runs: Number of runs per variant
        viz_fps: World visualization FPS
        viz_brain_enabled: Whether to enable brain visualization
        viz_brain_fps: Brain visualization FPS
        grid_width: World grid width
        grid_height: World grid height
        start_pos: Worm starting position (tuple)
        world_rng_seed: RNG seed for world
        worm_speed: Worm speed parameter
        worm_energy_capacity: Worm energy capacity
        worm_metabolic_rate: Worm metabolic rate
        worm_movement_cost: Worm movement cost
        sensor_cfg: Sensor configuration list
        feeding_paradigm: Feeding paradigm dict
        feeding_initial_fraction_per_cell: Initial food fraction per cell
        feeding_regrow_time: Food regrow time
        brain_n_neurons: Number of neurons in brain
        brain_threshold: Neuron threshold (applied to all neurons)
        brain_noise_level: Neuron noise level (applied to all neurons)
        brain_sensory_mapping: Sensory input mapping dict
        brain_output_mapping: Output neuron mapping dict
        brain_max_decision_delay: Maximum decision delay
        rng_seed: RNG seed for this variant
        has_brain_config: Whether brain is enabled
        hdf5_path: Path to HDF5 file to write to
        hdf5_lock: multiprocessing.Lock() for synchronized writes
    """
    
    # ============================================================
    # ============================================================
    # 5a: Create World instance
    # ============================================================
    world = make_world(grid_width, grid_height, start_pos, world_rng_seed)
    feeding_cfg = make_feeding_cfg(feeding_paradigm, feeding_initial_fraction_per_cell, feeding_regrow_time)
    world.feeding_cfg = feeding_cfg

    # ============================================================
    # 5b: Load Brain Module
    # ============================================================
    brain_module = load_brain_module(brain_module_name)

    # ============================================================
    # 5c: Create Worm instance
    # ============================================================
    worm = make_worm(world, worm_speed, worm_energy_capacity, worm_metabolic_rate, worm_movement_cost)
    worm.active_sensors = sensor_cfg
    worm.brain = brain_module

    # ============================================================
    # 5d: Prepare simulation & Build RNG streams
    # ============================================================
    # Extract genome components for tracking
    connection_weights = genome["connection_weights"]
    connections_to_track = []
    for src in range(connection_weights.shape[0]):
        for tgt in range(connection_weights.shape[1]):
            if connection_weights[src, tgt, 0] != 0.0:
                connections_to_track.append((src, tgt))

    # Always allocate lightweight lifespan array (only lifetime_ticks tracking)
    dtype_lifespan = [('lifetime_ticks', 'i4')]
    lifespan_array = np.zeros(n_runs, dtype=dtype_lifespan)

    # Conditionally allocate full summary array (only if per-run tracking enabled)
    if enable_per_run_tracking:
        summary_array = None
        dtype_summary = [('run_id', 'i2'), ('lifetime_ticks', 'i4'), ('foods', 'i4'),
                         ('distance', 'i4'), ('final_energy', 'f4'),
                         ('moves_north', 'i4'), ('moves_south', 'i4'), ('moves_east', 'i4'), ('moves_west', 'i4'),
                         ('food_sensed_north', 'i4'), ('food_sensed_east', 'i4'), ('food_sensed_south', 'i4'), ('food_sensed_west', 'i4'),
                         ('decisions', 'i4'), ('correct_decisions', 'i4')]
        summary_array = np.zeros(n_runs, dtype=dtype_summary)

        # Only allocate tracking structures if per-run tracking is enabled
        # Pre-allocate wiring array with columns for all run final weights
        dtype_wiring = [('src', 'i2'), ('tgt', 'i2'), ('weight_initial', 'f4')]
        for run_id in range(1, n_runs + 1):
            dtype_wiring.append((f'weight_final_run_{run_id:04d}', 'f4'))
        wiring_array = np.zeros(len(connections_to_track), dtype=dtype_wiring)

        for idx, (src, tgt) in enumerate(connections_to_track):
            wiring_array[idx]['src'] = src
            wiring_array[idx]['tgt'] = tgt
            wiring_array[idx]['weight_initial'] = connection_weights[src, tgt, 0]

        # Pre-allocate modulation array
        dtype_modulation = [('target_src', 'i2'), ('target_tgt', 'i2'), ('modulator_src', 'i2'), ('modulation_weight', 'f4')]
        modulation_list = []
        modulator_spec = genome["modulation_spec"]
        for (target_src, target_tgt), modulators in modulator_spec.items():
            for mod_src, mod_weight in modulators:
                modulation_list.append((target_src, target_tgt, mod_src, mod_weight))
        modulation_array = np.array(modulation_list, dtype=dtype_modulation) if modulation_list else np.array([], dtype=dtype_modulation)

        # Accumulate per-tick and heatmap data for batch write after all runs
        per_tick_all_runs = {}
        heatmaps_all_runs = {}

    # ============================================================
    # 5e: Create MetricsRecorder to track per-tick data
    # ============================================================
    rec = MetricsRecorder.empty(worm, genome, enable_per_run_tracking, enable_per_tick_tracking, enable_heat_map_tracking, max_ticks)
    rec.record(worm)
    
    # ============================================================
    # 5f: Simulate runs. For each run do:
    # ============================================================
    for run_id in range(n_runs):

        # ============================================================
        # 5f1: Prepare simulation & Build RNG streams
        # ============================================================
        rng_food, rng_decision, rng_neuron_noise = build_rng_streams(
            rng_seed + run_id, has_brain_config
        )

        # ============================================================
        # 5f2: Call brain_module.init_brain(genome, brain_cfg, rng_noise)
        # ============================================================
        brain_cfg_dict = {
            'n_neurons': brain_n_neurons,
            'threshold': brain_threshold,
            'noise_level': brain_noise_level,
            'sensory_mapping': brain_sensory_mapping,
            'output_mapping': brain_output_mapping,
            'max_decision_delay': brain_max_decision_delay,
        }
        brain_module.init_brain(genome, brain_cfg_dict, rng_neuron_noise)

        # ============================================================
        # 5f3: Reset world & worm & simulation
        # ============================================================
        world.reset_food()
        seed_food(world, feeding_cfg, rng_food)
        worm.reset()
        reset_sim(world, feeding_cfg, rng_food, worm)
        rec.reset()

        # Create world renderer independently if world visualization is enabled
        if viz_enabled:
            try:
                if viz_fps > 0:
                    worm.renderer = QtRenderer(world, worm, viz_fps)
                    print(f"[viz] Created world renderer at {viz_fps} FPS")
                else:
                    worm.renderer = None
            except Exception as e:
                print(f"[WARNING] Failed to create world renderer: {e}.")
                worm.renderer = None
        else:
            worm.renderer = None
        
        # Create brain renderer independently if brain visualization is enabled
        if viz_brain_enabled:
            try:
                if viz_brain_fps > 0:
                    brain_renderer = BrainQtRenderer(fps=viz_brain_fps)
                    brain_module._brain_renderer = brain_renderer
                    print(f"[viz] Created brain renderer at {viz_brain_fps} FPS")
                else:
                    brain_module._brain_renderer = None
            except Exception as e:
                print(f"[WARNING] Failed to create brain renderer: {e}.")
                brain_module._brain_renderer = None
        else:
            brain_module._brain_renderer = None

        # Get pause manager for checkpoints (if any visualization enabled)
        pause_mgr = None
        if viz_enabled or viz_brain_enabled:
            try:
                pause_mgr = get_pause_manager()
            except RuntimeError:
                pass

        # ============================================================
        # 5f4: Simulate
        # ============================================================
        try:
            while worm.alive and worm.ticks < max_ticks:
                # Check pause/exit at start of each tick
                if pause_mgr is not None:
                    pause_mgr.check_pause()
                
                world.step()
                worm.step_day(rng_decision)
                worm.ticks += 1
                rec.record(worm)
                
                # Double-check exit flag after each step
                if pause_mgr is not None and pause_mgr.should_exit():
                    raise PauseManagerExit("Exit requested during simulation")
                
                # Wait to maintain FPS if visualization is enabled
                if worm.renderer is not None:
                    worm.renderer.wait_frame()
        except PauseManagerExit:
            pass  # Exit simulation gracefully
        
        if enable_per_tick_tracking and rec.per_tick_data is not None:
            per_tick_data = rec.per_tick_data[:rec.per_tick_count]
            per_tick_all_runs[run_id+1] = per_tick_data
        
        if enable_heat_map_tracking and rec.staying_heatmap is not None:
            heatmaps_all_runs[run_id+1] = rec.staying_heatmap.copy()
        
        # Always record lifespan
        lifespan_array[run_id]['lifetime_ticks'] = worm.ticks
        
        # Conditionally record full summary metrics
        if enable_per_run_tracking:
            for idx, (src, tgt) in enumerate(connections_to_track):
                w = get_connection_weight(worm.brain, src, tgt)
                wiring_array[idx][f'weight_final_run_{run_id+1:04d}'] = w
            
            summary_array[run_id]['run_id'] = run_id + 1
            summary_array[run_id]['lifetime_ticks'] = worm.ticks
            summary_array[run_id]['foods'] = worm.eats
            summary_array[run_id]['distance'] = worm.distance
            summary_array[run_id]['final_energy'] = worm.energy
            summary_array[run_id]['moves_north'] = rec.moves_north
            summary_array[run_id]['moves_south'] = rec.moves_south
            summary_array[run_id]['moves_east'] = rec.moves_east
            summary_array[run_id]['moves_west'] = rec.moves_west
            summary_array[run_id]['food_sensed_north'] = rec.food_sensed_north
            summary_array[run_id]['food_sensed_east'] = rec.food_sensed_east
            summary_array[run_id]['food_sensed_south'] = rec.food_sensed_south
            summary_array[run_id]['food_sensed_west'] = rec.food_sensed_west
            summary_array[run_id]['decisions'] = rec.decisions
            summary_array[run_id]['correct_decisions'] = rec.correct_decisions

    # Extract lifespan vector (always available)
    lifespan_vector = lifespan_array['lifetime_ticks']

    # ============================================================
    # 5g: Batch write to HDF5
    # ============================================================
    # Batch write all variant data after all runs complete (only if per-run tracking enabled)
    if enable_per_run_tracking:
        with hdf5_lock:
            # Write summary array only if per-run tracking is enabled
            if summary_array is not None:
                save_variant_summary_to_hdf5(hdf5_path, variant_id + 1, summary_array)
                save_wiring_to_hdf5(hdf5_path, variant_id + 1, wiring_array)
                save_modulation_to_hdf5(hdf5_path, variant_id + 1, modulation_array)
                
                # Write accumulated per-tick data if any
                if per_tick_all_runs:
                    for run_id, per_tick_data in per_tick_all_runs.items():
                        save_per_tick_to_hdf5(hdf5_path, variant_id + 1, run_id, per_tick_data)
                
                # Write accumulated heatmap data if any
                if heatmaps_all_runs:
                    for run_id, staying_heatmap in heatmaps_all_runs.items():
                        save_heatmaps_to_hdf5(hdf5_path, variant_id + 1, run_id, staying_heatmap)
    
    # Keep visualization window open if it was created (but not if exit was requested)
    should_show_event_loop = viz_enabled or viz_brain_enabled
    if should_show_event_loop:
        if pause_mgr is not None and pause_mgr.should_exit():
            should_show_event_loop = False
        
        if should_show_event_loop:
            try:
                from PySide6.QtWidgets import QApplication
                app = QApplication.instance()
                if app is not None:
                    print("[INFO] Visualization complete. Close the window to continue.")
                    app.exec()
            except Exception as e:
                pass  # Silently fail if no Qt window exists
    
    return (variant_id, lifespan_vector)


# --- TRACKING VALIDATION ---

def validate_tracking_flags(enable_per_run, enable_per_tick, enable_heat_map):
    """
    Validate tracking flags and adjust if necessary.
    
    If enable_per_run is False but other tracking flags are True,
    warn the user and ask whether to disable them or exit.
    
    Args:
        enable_per_run: Whether per-run tracking is enabled
        enable_per_tick: Whether per-tick tracking is enabled
        enable_heat_map: Whether heatmap tracking is enabled
    
    Returns: True to continue, False to exit
    """
    if not enable_per_run and (enable_per_tick or enable_heat_map):
        print("\n[WARNING] Conflicting tracking configuration:")
        print(f"  ENABLE_PER_RUN_TRACKING = {enable_per_run}")
        print(f"  ENABLE_PER_TICK_TRACKING = {enable_per_tick}")
        print(f"  ENABLE_HEAT_MAP_TRACKING = {enable_heat_map}")
        print("\nWhen per-run tracking is disabled, detailed per-tick and heatmap")
        print("tracking are useless. Only lifespan metrics will be recorded.")
        response = input("\nDisable all detailed tracking and continue? (y/n): ").strip().lower()
        if response == 'y':
            return True
        else:
            print("[EXIT] User cancelled due to tracking configuration conflict.")
            return False
    return True


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


def print_lifespan_summary(all_lifespans: dict):
    """Print lifespan (lifetime ticks) from all variants to terminal.
    
    Args:
        all_lifespans: Dict mapping variant_id to lifespan_vector (1D array of lifetime_ticks)
    """
    print("\n" + "="*80)
    print("[lifespan] SIMULATION SUMMARY")
    print("="*80)
    
    for variant_id in sorted(all_lifespans.keys()):
        lifespan_vector = all_lifespans[variant_id]
        print(f"\nVariant {variant_id:02d}:")
        print("-" * 80)
        print(f"{'Run':>4} {'Ticks':>10}")
        print("-" * 80)
        
        for run_idx, ticks in enumerate(lifespan_vector, start=1):
            print(f"{run_idx:>4} {int(ticks):>10}")
        
        # Print variant average
        avg_ticks = np.mean(lifespan_vector)
        print("-" * 80)
        print(f"{'AVG':>4} {avg_ticks:>10.1f}")
        print("=" * 80)


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
        CONFIG_PATH = resolve_config_path(args.config)
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
    brain_module_name = make_decision_cfg(cfg)
    
    
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
    
    if not validate_tracking_flags(ENABLE_PER_RUN_TRACKING, ENABLE_PER_TICK_TRACKING, ENABLE_HEAT_MAP_TRACKING):
        return
    
    VIZ_ENABLED, VIZ_BRAIN_ENABLED = validate_viz_flags(N_VARIANTS, N_RUNS, VIZ_ENABLED, VIZ_BRAIN_ENABLED)
    
    # ============================================================
    # Extract config components for worker
    # ============================================================
    grid_width = cfg["world"]["grid_width"]
    grid_height = cfg["world"]["grid_height"]
    start_pos = cfg["world"]["start_pos"]
    world_rng_seed = cfg["world"]["rng_seed"]
    worm_speed = cfg["worm"]["speed"]
    worm_energy_capacity = cfg["worm"]["energy_capacity"]
    worm_metabolic_rate = cfg["worm"]["metabolic_rate"]
    worm_movement_cost = cfg["worm"].get("movement_cost", 1)
    sensor_cfg = make_sensor_cfg(cfg)
    feeding_paradigm = cfg["food"].get("feeding_paradigm", {"initial": True, "regrow": True})
    feeding_initial_fraction_per_cell = cfg["food"].get("initial_fraction_per_cell", 0.25)
    feeding_regrow_time = cfg["food"].get("regrow_time", 3000)
    brain_n_neurons = cfg["brain"]["n_neurons"]
    brain_threshold = cfg["brain"]["threshold"]
    brain_noise_level = cfg["brain"]["noise_level"]
    brain_sensory_mapping = cfg["brain"]["sensory_mapping"]
    brain_output_mapping = cfg["brain"]["output_mapping"]
    brain_max_decision_delay = cfg["brain"]["max_decision_delay"]
    rng_seed = cfg["world"]["rng_seed"]
    
    # ============================================================
    # 3. GENERATE GENOMES
    # ============================================================
    # Generate all genomes before dispatching workers
    genomes = []
    for variant_id in range(N_VARIANTS):
        genome = genome_generator(cfg, rng_seed=WIRING_RANDOMIZATION_SEED + variant_id)
        genomes.append(genome)
    
    # ============================================================
    # 4. PREPARE DATA TRACKING IF ENABLED
    # ============================================================
        
    # Only create HDF5 file and manager if per-run tracking is enabled
    if ENABLE_PER_RUN_TRACKING:
        hdf5_path = make_experiment_dir(EXPERIMENT_FOLDER, SIMULATION_NAME)
        print(f"[batch] writing to {hdf5_path}\n")
        create_hdf5_file(hdf5_path, cfg)
        # Save genome generation parameters to HDF5
        save_genome_properties_to_hdf5(hdf5_path, genomes)
        
        # Create manager and lock for parallel HDF5 writing
        manager = Manager()
        hdf5_lock = manager.Lock()

    # ============================================================
    # 5. PREPARE WORKERS, THEN EITHER PARALLEL OR SERIAL
    # ============================================================
    
    try:
        # Run simulation
        all_lifespans = {}
        num_workers = get_num_workers(VIZ_ENABLED, VIZ_BRAIN_ENABLED)
        if num_workers is None:
            for variant_id in range(N_VARIANTS):
                print(f"[variant {variant_id+1:02d}/{N_VARIANTS:02d}] Simulating...", end='', flush=True)
                
                kwargs = {
                    'variant_id': variant_id,
                    'brain_module_name': brain_module_name,
                    'genome': genomes[variant_id],
                    'viz_enabled': VIZ_ENABLED,
                    'enable_per_run_tracking': ENABLE_PER_RUN_TRACKING,
                    'enable_per_tick_tracking': ENABLE_PER_TICK_TRACKING,
                    'enable_heat_map_tracking': ENABLE_HEAT_MAP_TRACKING,
                    'max_ticks': MAX_TICKS,
                    'n_runs': N_RUNS,
                    'viz_fps': VIZ_FPS,
                    'viz_brain_enabled': VIZ_BRAIN_ENABLED,
                    'viz_brain_fps': VIZ_BRAIN_FPS,
                    'grid_width': grid_width,
                    'grid_height': grid_height,
                    'start_pos': start_pos,
                    'world_rng_seed': world_rng_seed,
                    'worm_speed': worm_speed,
                    'worm_energy_capacity': worm_energy_capacity,
                    'worm_metabolic_rate': worm_metabolic_rate,
                    'worm_movement_cost': worm_movement_cost,
                    'sensor_cfg': sensor_cfg,
                    'feeding_paradigm': feeding_paradigm,
                    'feeding_initial_fraction_per_cell': feeding_initial_fraction_per_cell,
                    'feeding_regrow_time': feeding_regrow_time,
                    'brain_n_neurons': brain_n_neurons,
                    'brain_threshold': brain_threshold,
                    'brain_noise_level': brain_noise_level,
                    'brain_sensory_mapping': brain_sensory_mapping,
                    'brain_output_mapping': brain_output_mapping,
                    'brain_max_decision_delay': brain_max_decision_delay,
                    'rng_seed': rng_seed,
                    'has_brain_config': has_brain_config,
                }
                if ENABLE_PER_RUN_TRACKING:
                    kwargs['hdf5_path'] = hdf5_path
                    kwargs['hdf5_lock'] = hdf5_lock
                
                returned_variant_id, lifespan_vector = run_variant_worker(**kwargs)

                all_lifespans[variant_id] = lifespan_vector
                print(" done")
        
        else:
            completed = 0
            
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {}
                for variant_id in range(N_VARIANTS):
                    kwargs = {
                        'variant_id': variant_id,
                        'brain_module_name': brain_module_name,
                        'genome': genomes[variant_id],
                        'viz_enabled': False,  # Never viz in parallel (serial only)
                        'enable_per_run_tracking': ENABLE_PER_RUN_TRACKING,
                        'enable_per_tick_tracking': ENABLE_PER_TICK_TRACKING,
                        'enable_heat_map_tracking': ENABLE_HEAT_MAP_TRACKING,
                        'max_ticks': MAX_TICKS,
                        'n_runs': N_RUNS,
                        'viz_fps': VIZ_FPS,
                        'viz_brain_enabled': VIZ_BRAIN_ENABLED,
                        'viz_brain_fps': VIZ_BRAIN_FPS,
                        'grid_width': grid_width,
                        'grid_height': grid_height,
                        'start_pos': start_pos,
                        'world_rng_seed': world_rng_seed,
                        'worm_speed': worm_speed,
                        'worm_energy_capacity': worm_energy_capacity,
                        'worm_metabolic_rate': worm_metabolic_rate,
                        'worm_movement_cost': worm_movement_cost,
                        'sensor_cfg': sensor_cfg,
                        'feeding_paradigm': feeding_paradigm,
                        'feeding_initial_fraction_per_cell': feeding_initial_fraction_per_cell,
                        'feeding_regrow_time': feeding_regrow_time,
                        'brain_n_neurons': brain_n_neurons,
                        'brain_threshold': brain_threshold,
                        'brain_noise_level': brain_noise_level,
                        'brain_sensory_mapping': brain_sensory_mapping,
                        'brain_output_mapping': brain_output_mapping,
                        'brain_max_decision_delay': brain_max_decision_delay,
                        'rng_seed': rng_seed,
                        'has_brain_config': has_brain_config,
                    }
                    if ENABLE_PER_RUN_TRACKING:
                        kwargs['hdf5_path'] = hdf5_path
                        kwargs['hdf5_lock'] = hdf5_lock
                    
                    future = executor.submit(run_variant_worker, **kwargs)
                    futures[future] = variant_id
                
                for future in as_completed(futures):
                    variant_id = futures[future]
                    completed += 1
                    returned_variant_id, lifespan_vector = future.result()
                    all_lifespans[returned_variant_id] = lifespan_vector
                    print(f"\rProcessing variants... ({completed}/{N_VARIANTS} completed)", end='', flush=True)
            
            print()
        
        
        # ============================================================
        # 6. WRAP-UP
        # ============================================================
        if ENABLE_PER_RUN_TRACKING:
            print(f"[batch] Simulation completed. Saved to {hdf5_path.name}")
        else:
            print(f"[batch] Simulation completed. (No data recording)")
            if all_lifespans:
                print_lifespan_summary(all_lifespans)
    
    except PauseManagerExit:
        print("[EXIT] Batch simulation stopped by user.")
    finally:
        if VIZ_ENABLED or VIZ_BRAIN_ENABLED:
            cleanup_pause_manager()


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

