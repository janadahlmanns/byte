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
import h5py

from mvb.world import World
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

def build_rng_streams(seed: int):
    """Build properly independent RNG streams using SeedSequence.
    
    Creates three independent random streams from a single seed, suitable for
    continuous use throughout the batch (not reset between runs).
    
    Args:
        seed: Master seed for the batch
    
    Returns:
        Tuple of (rng_world, rng_decision, rng_neuron_noise) as independent streams
    """
    seed = int(seed)
    seed_seq = np.random.SeedSequence(seed)
    # Spawn 3 truly independent streams
    streams = seed_seq.spawn(3)
    return (
        np.random.default_rng(streams[0]),
        np.random.default_rng(streams[1]),
        np.random.default_rng(streams[2]),
    )

# --- helpers ---

def load_brain_module(version: str):
    module_name = f"mvb.brains.decisionmaking_{version}"
    module = importlib.import_module(module_name)
    if not hasattr(module, "decide"):
        raise AttributeError(f"{module_name} has no decide()")
    return module



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

def make_feeding_cfg(feeding_paradigm, initial_fraction_per_cell, regrow_time):
    return FeedingConfig(
        feeding_paradigm=feeding_paradigm,
        initial_fraction_per_cell=initial_fraction_per_cell,
        regrow_time=regrow_time,
    )

def make_brain_cfg(n_neurons, threshold, noise_level, sensory_mapping, output_mapping, max_decision_delay):
    return {
        'n_neurons': n_neurons,
        'threshold': threshold,
        'noise_level': noise_level,
        'sensory_mapping': sensory_mapping,
        'output_mapping': output_mapping,
        'max_decision_delay': max_decision_delay,
    }

def make_sensor_cfg(cfg_yaml):
    return cfg_yaml.get("worm", {}).get("sensors", {}).get("active", ["current_field"])

def make_decision_cfg(cfg_yaml):
    return str(cfg_yaml["worm"]["decisionmaking"]["version"])



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


def simulate_run(world, worm, rec, rng_worker_decision, rng_worker_neuron_noise, max_ticks, pause_mgr):
    """Execute a single simulation run with all ticks until worm dies or max_ticks reached.
    
    Args:
        world: World instance
        worm: Worm instance
        rec: MetricsRecorder instance for tracking metrics, or None if no tracking details
        rng_worker_decision: RNG for decision-making
        rng_worker_neuron_noise: RNG for neuron noise during brain computation
        max_ticks: Maximum simulation ticks
        pause_mgr: Optional PauseManager for pause/exit handling
    
    Returns:
        Tuple of (world, worm, rec, pause_mgr) - all objects modified in place during simulation
    """
    try:
        while worm.alive and worm.ticks < max_ticks:
            # Check pause/exit at start of each tick
            if pause_mgr is not None:
                pause_mgr.check_pause()
            
            world.step()
            worm.step_day(rng_worker_decision, rng_worker_neuron_noise)
            worm.ticks += 1
            if rec is not None:
                rec.record(worm)
            
            # Double-check exit flag after each step
            if pause_mgr is not None and pause_mgr.should_exit():
                raise PauseManagerExit("Exit requested during simulation")
            
            # Wait to maintain FPS if visualization is enabled
            if worm.renderer is not None:
                worm.renderer.wait_frame()
    except PauseManagerExit:
        pass  # Exit simulation gracefully
    
    return world, worm, rec, pause_mgr


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
    def empty(cls, genome, start_y, start_x, grid_height, grid_width, enable_per_run_tracking=True, enable_per_tick_tracking=True, enable_heat_map_tracking=True, max_ticks=2000):
        """Initialize recorder with genome and explicit configuration parameters.
        
        If enable_per_run_tracking is False, only lifetime metrics are tracked.
        
        Args:
            genome: Genome dict with connection_weights
            start_y: Starting Y position of worm
            start_x: Starting X position of worm
            grid_height: Height of world grid
            grid_width: Width of world grid
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
                
        if enable_per_run_tracking:
            kwargs = {
            'per_tick_count': 0,
            'connections_to_track': connections_to_track,
            'start_y': start_y,
            'start_x': start_x,
            'prev_y': start_y,
            'prev_x': start_x,
            'prev_eats': 0,
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


def eval_variant(
    brain_module_name,
    grid_width,
    grid_height,
    start_pos,
    enable_per_run_tracking,
    enable_per_tick_tracking,
    enable_heat_map_tracking,
    feeding_cfg,
    worm_speed,
    worm_energy_capacity,
    worm_metabolic_rate,
    worm_movement_cost,
    n_runs,
    run_seeds,
    genome,
    brain_cfg,
    viz_enabled,
    viz_fps,
    viz_brain_enabled,
    viz_brain_fps,
    rng_worker_decision,
    rng_worker_neuron_noise,
    max_ticks,
    sensor_cfg,
):
    """Execute all runs for a single variant and return tracking results.
    
    Args:
        brain_module_name: Name of brain module to import
        grid_width: World grid width
        grid_height: World grid height
        start_pos: Worm starting position (tuple)
        enable_per_run_tracking: Whether to track per-run metrics
        enable_per_tick_tracking: Whether to track per-tick data
        enable_heat_map_tracking: Whether to track heatmaps
        feeding_cfg: FeedingConfig object
        worm_speed: Worm speed parameter
        worm_energy_capacity: Worm energy capacity
        worm_metabolic_rate: Worm metabolic rate
        worm_movement_cost: Worm movement cost
        n_runs: Number of runs
        run_seeds: Array of seeds for world initialization
        genome: Pre-generated genome dict
        brain_cfg: Brain configuration dict
        viz_enabled: Whether to enable world visualization
        viz_fps: World visualization FPS
        viz_brain_enabled: Whether to enable brain visualization
        viz_brain_fps: Brain visualization FPS
        rng_worker_decision: RNG for decision-making
        rng_worker_neuron_noise: RNG for neuron noise
        max_ticks: Maximum simulation ticks
        sensor_cfg: Sensor configuration list
    
    Returns:
        Dict with tracking results containing:
        - lifespan_vector: Always included
        - summary_array, wiring_array, modulation_array: If enable_per_run_tracking
        - per_tick_all_runs: If enable_per_tick_tracking
        - heatmaps_all_runs: If enable_heat_map_tracking
    """
    rec = None
    # Always allocate lifespan vector (always available, no tracking flags needed)
    lifespan_vector = np.zeros(n_runs, dtype='i4')

    # Conditionally allocate full summary array (only if per-run tracking enabled)
    if enable_per_run_tracking:
        # Extract genome components for tracking
        connection_weights = genome["connection_weights"]
        connections_to_track = []
        for src in range(connection_weights.shape[0]):
            for tgt in range(connection_weights.shape[1]):
                if connection_weights[src, tgt, 0] != 0.0:
                    connections_to_track.append((src, tgt))
        summary_array = None
        dtype_summary = [('run_id', 'i2'), ('lifetime_ticks', 'i4'), ('foods', 'i4'),
                         ('distance', 'i4'), ('final_energy', 'f4')]
        summary_array = np.zeros(n_runs, dtype=dtype_summary)

        # Pre-allocate wiring array with columns for all run final weights
        dtype_wiring = [('src', 'i2'), ('tgt', 'i2'), ('weight_initial', 'f4')]
        for run_id in range(n_runs):
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

    # ============================================================
    # 6c: Create MetricsRecorder to track per-tick data (only if tracking enabled)
    # ============================================================
  
        if enable_per_tick_tracking or enable_heat_map_tracking:
            rec = MetricsRecorder.empty(genome, start_pos[0], start_pos[1], grid_height, grid_width, enable_per_run_tracking, enable_per_tick_tracking, enable_heat_map_tracking, max_ticks)     
        if enable_per_tick_tracking:
            per_tick_all_runs = {}
        if enable_heat_map_tracking:
            heatmaps_all_runs = {}
    else:
        summary_array = None
        wiring_array = None
        modulation_array = None
    


    # ============================================================
    # 6d: Load Brain Module
    # ============================================================
    brain_module = load_brain_module(brain_module_name)

    # ============================================================
    # 6e: Create temporary world and worm (will be reset per run so seed doesnt matter)
    # ============================================================
    world = World(grid_width, grid_height, start_pos, 0)
    world.feeding_cfg = feeding_cfg
    
    worm = Worm(worm_speed, worm_energy_capacity, worm_metabolic_rate, worm_movement_cost, world)
    worm.active_sensors = sensor_cfg
    worm.brain = brain_module

    # ============================================================
    # 6f: Simulate runs. For each run do:
    # ============================================================
    

    for run_id in range(n_runs):

        # ============================================================
        # 6f1: Set world seed for this run
        # ============================================================
        rng_world_run = np.random.default_rng(run_seeds[run_id])

        # ============================================================
        # 6f2: Call brain_module.init_brain(genome, brain_cfg, rng_noise) for a clean reset
        # ============================================================
        brain_module.init_brain(genome, brain_cfg)

        # ============================================================
        # 6f3: Reset worm & simulation, rest world with the according run rng 
        # ============================================================
        world.reset_food()
        seed_food(world, feeding_cfg, rng_world_run)
        worm.reset()
        if rec is not None:
            rec.reset()
            rec.record(worm)

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
        # 6f4: Simulate
        # ============================================================
        world, worm, rec, pause_mgr = simulate_run(world, worm, rec, rng_worker_decision, rng_worker_neuron_noise, max_ticks, pause_mgr)
         
        # Always record lifespan
        lifespan_vector[run_id] = worm.ticks 
        if enable_per_run_tracking:
            # Conditionally record full summary metrics
            for idx, (src, tgt) in enumerate(connections_to_track):
                w = get_connection_weight(worm.brain, src, tgt)
                wiring_array[idx][f'weight_final_run_{run_id:04d}'] = w
            
            summary_array[run_id]['run_id'] = run_id
            summary_array[run_id]['lifetime_ticks'] = worm.ticks
            summary_array[run_id]['foods'] = worm.eats
            summary_array[run_id]['distance'] = worm.distance
            summary_array[run_id]['final_energy'] = worm.energy
            if enable_per_tick_tracking:
                per_tick_data = rec.per_tick_data[:rec.per_tick_count]
                per_tick_all_runs[run_id] = per_tick_data
            
            if enable_heat_map_tracking:
                heatmaps_all_runs[run_id] = rec.staying_heatmap.copy()
    
    # Build tracking results dict
    tracking_results = {'lifespan_vector': lifespan_vector}
    
    if enable_per_run_tracking:
        tracking_results['summary_array'] = summary_array
        tracking_results['wiring_array'] = wiring_array
        tracking_results['modulation_array'] = modulation_array
        
        if enable_per_tick_tracking:
            tracking_results['per_tick_all_runs'] = per_tick_all_runs
        
        if enable_heat_map_tracking:
            tracking_results['heatmaps_all_runs'] = heatmaps_all_runs
    
    return tracking_results


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
    worm_speed,
    worm_energy_capacity,
    worm_metabolic_rate,
    worm_movement_cost,
    sensor_cfg,
    feeding_cfg,
    brain_cfg,
    variant_decision_seed,
    variant_noise_seed,
    run_seeds,
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
        worm_speed: Worm speed parameter
        worm_energy_capacity: Worm energy capacity
        worm_metabolic_rate: Worm metabolic rate
        worm_movement_cost: Worm movement cost
        sensor_cfg: Sensor configuration list
        feeding_cfg: FeedingConfig object with feeding_paradigm, initial_fraction_per_cell, regrow_time
        brain_cfg: Dict with brain configuration (n_neurons, threshold, noise_level, sensory_mapping, output_mapping, max_decision_delay)
        variant_decision_seed: RNG seed for decision-making in this variant
        variant_noise_seed: RNG seed for neuron noise in this variant
        run_seeds: Array of N_RUNS seeds for world initialization (same across all variants)
        hdf5_path: Path to HDF5 file to write to
        hdf5_lock: multiprocessing.Lock() for synchronized writes
    """
    
    # ============================================================
    # 6a: Build RNG streams for this variant (persistent across all runs)
    # ============================================================
    rng_worker_decision = np.random.default_rng(variant_decision_seed)
    rng_worker_neuron_noise = np.random.default_rng(variant_noise_seed)
    
    # ============================================================
    # 6b: Prepare simulation arrays
    # ============================================================
    tracking_results = eval_variant(
        brain_module_name,
        grid_width,
        grid_height,
        start_pos,
        enable_per_run_tracking,
        enable_per_tick_tracking,
        enable_heat_map_tracking,
        feeding_cfg,
        worm_speed,
        worm_energy_capacity,
        worm_metabolic_rate,
        worm_movement_cost,
        n_runs,
        run_seeds,
        genome,
        brain_cfg,
        viz_enabled,
        viz_fps,
        viz_brain_enabled,
        viz_brain_fps,
        rng_worker_decision,
        rng_worker_neuron_noise,
        max_ticks,
        sensor_cfg,
    )

    # ============================================================
    # 6g: Batch write to HDF5
    # ============================================================
    # Extract tracking results
    lifespan_vector = tracking_results['lifespan_vector']
    
    # Batch write all variant data after all runs complete (only if per-run tracking enabled)
    if enable_per_run_tracking:
        summary_array = tracking_results['summary_array']
        wiring_array = tracking_results['wiring_array']
        modulation_array = tracking_results['modulation_array']
        
        with hdf5_lock:
            # Write summary array only if per-run tracking is enabled
            if summary_array is not None:
                save_variant_summary_to_hdf5(hdf5_path, variant_id, summary_array)
                save_wiring_to_hdf5(hdf5_path, variant_id, wiring_array)
                save_modulation_to_hdf5(hdf5_path, variant_id, modulation_array)
                
                # Write accumulated per-tick data if any
                if enable_per_tick_tracking:
                    per_tick_all_runs = tracking_results.get('per_tick_all_runs', {})
                    for run_id, per_tick_data in per_tick_all_runs.items():
                        save_per_tick_to_hdf5(hdf5_path, variant_id, run_id, per_tick_data)
                
                # Write accumulated heatmap data if any
                if enable_heat_map_tracking:
                    heatmaps_all_runs = tracking_results.get('heatmaps_all_runs', {})
                    for run_id, staying_heatmap in heatmaps_all_runs.items():
                        save_heatmaps_to_hdf5(hdf5_path, variant_id, run_id, staying_heatmap)
                
                # Write RNG seed information to HDF5 variant attributes (to existing variant group)
                with h5py.File(hdf5_path, 'a') as f:
                    variant_group_name = f'variant_{variant_id}'
                    if variant_group_name in f:
                        variant_group = f[variant_group_name]
                        variant_group.attrs['rng_seed_decision'] = int(variant_decision_seed)
                        variant_group.attrs['rng_seed_noise'] = int(variant_noise_seed)
    
    return (variant_id, lifespan_vector)

def eval_generation(genomes, cfg, EXPERIMENT_FOLDER, SIMULATION_NAME, ENABLE_PER_RUN_TRACKING, ENABLE_PER_TICK_TRACKING,
                                ENABLE_HEAT_MAP_TRACKING, rng_world, VIZ_ENABLED, VIZ_BRAIN_ENABLED, VIZ_FPS, VIZ_BRAIN_FPS, N_VARIANTS,
                                brain_module_name, MAX_TICKS, N_RUNS, grid_width, grid_height, start_pos, worm_speed, worm_energy_capacity, worm_metabolic_rate, worm_movement_cost, sensor_cfg, feeding_cfg, brain_cfg, variant_decision_seeds, variant_noise_seeds):
    
# starting here instead of in place code execution we want to call a new function called eval_generation.
# input arguments will be: EXPeriment folder, simulation name, enable per run tracking, enable per tick tracking, enable heatmap tracking
# cfg, N_RUNS,  viz enabled viz brain enabled, genome, all of these that i have not already mentioned:                    'variant_id': variant_id,
                    # 'brain_module_name': brain_module_name,
                    # 'genome': genomes[variant_id],
                    # 'viz_enabled': VIZ_ENABLED,
                    # 'enable_per_run_tracking': ENABLE_PER_RUN_TRACKING,
                    # 'enable_per_tick_tracking': ENABLE_PER_TICK_TRACKING,
                    # 'enable_heat_map_tracking': ENABLE_HEAT_MAP_TRACKING,
                    # 'max_ticks': MAX_TICKS,
                    # 'n_runs': N_RUNS,
                    # 'viz_fps': VIZ_FPS,
                    # 'viz_brain_enabled': VIZ_BRAIN_ENABLED,
                    # 'viz_brain_fps': VIZ_BRAIN_FPS,
                    # 'grid_width': grid_width,
                    # 'grid_height': grid_height,
                    # 'start_pos': start_pos,
                    # 'worm_speed': worm_speed,
                    # 'worm_energy_capacity': worm_energy_capacity,
                    # 'worm_metabolic_rate': worm_metabolic_rate,
                    # 'worm_movement_cost': worm_movement_cost,
                    # 'sensor_cfg': sensor_cfg,
                    # 'feeding_cfg': feeding_cfg,
                    # 'brain_cfg': brain_cfg,
                    # 'variant_decision_seed': variant_decision_seeds[variant_id],
                    # 'variant_noise_seed': variant_noise_seeds[variant_id],
                    # 'run_seeds': run_seeds.copy(),
        



    # ============================================================
    # 5. ONE-TIME HDF5 INITIALIZATION (if tracking enabled)
    # ============================================================
    
    if ENABLE_PER_RUN_TRACKING:
        hdf5_path = make_experiment_dir(EXPERIMENT_FOLDER, SIMULATION_NAME)
        print(f"[batch] writing to {hdf5_path}\n")
        create_hdf5_file(hdf5_path, cfg)
        
        # Save the actual tracking flags used (after validation/user input corrections)
        with h5py.File(hdf5_path, 'a') as f:
            f.attrs['tracking_per_run_enabled'] = int(ENABLE_PER_RUN_TRACKING)
            f.attrs['tracking_per_tick_enabled'] = int(ENABLE_PER_TICK_TRACKING)
            f.attrs['tracking_heatmap_enabled'] = int(ENABLE_HEAT_MAP_TRACKING)
        
        # Create manager and lock for parallel HDF5 writing
        manager = Manager()
        hdf5_lock = manager.Lock()


    # ============================================================
    # 6. PREPARE WORKERS and , THEN EITHER PARALLEL OR SERIAL
    # ============================================================

    # Draw RNG seeds for the N_RUNS to be handed to workers
    run_seeds = rng_world.integers(0, 2**32, size=N_RUNS, dtype=np.uint32)
    # Save per-generation data to HDF5 (only if per-run tracking is enabled)
    if ENABLE_PER_RUN_TRACKING:
        # Save genome generation parameters to HDF5
        save_genome_properties_to_hdf5(hdf5_path, genomes)
        # Save run_seeds for reproducibility
        with h5py.File(hdf5_path, 'a') as f:
            f.create_dataset('run_seeds', data=run_seeds)

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
                    'worm_speed': worm_speed,
                    'worm_energy_capacity': worm_energy_capacity,
                    'worm_metabolic_rate': worm_metabolic_rate,
                    'worm_movement_cost': worm_movement_cost,
                    'sensor_cfg': sensor_cfg,
                    'feeding_cfg': feeding_cfg,
                    'brain_cfg': brain_cfg,
                    'variant_decision_seed': variant_decision_seeds[variant_id],
                    'variant_noise_seed': variant_noise_seeds[variant_id],
                    'run_seeds': run_seeds.copy(),
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
                futures = set()
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
                        'worm_speed': worm_speed,
                        'worm_energy_capacity': worm_energy_capacity,
                        'worm_metabolic_rate': worm_metabolic_rate,
                        'worm_movement_cost': worm_movement_cost,
                        'sensor_cfg': sensor_cfg,
                        'feeding_cfg': feeding_cfg,
                        'brain_cfg': brain_cfg,
                        'variant_decision_seed': variant_decision_seeds[variant_id],
                        'variant_noise_seed': variant_noise_seeds[variant_id],
                        'run_seeds': run_seeds.copy(),
                    }
                    if ENABLE_PER_RUN_TRACKING:
                        kwargs['hdf5_path'] = hdf5_path
                        kwargs['hdf5_lock'] = hdf5_lock
                    
                    future = executor.submit(run_variant_worker, **kwargs)
                    futures.add(future)
                
                for future in as_completed(futures):
                    completed += 1
                    returned_variant_id, lifespan_vector = future.result()
                    all_lifespans[returned_variant_id] = lifespan_vector
                    print(f"\rProcessing variants... ({completed}/{N_VARIANTS} completed)", end='', flush=True)
            
            print()
        # ============================================================
        # 7. WRAP-UP
        # ============================================================
        if ENABLE_PER_RUN_TRACKING:
            print(f"[batch] Simulation completed. Saved to {hdf5_path.name}")
        else:
            print(f"[batch] Simulation completed. (No data recording)")
            if all_lifespans:
                total_lifespans = sum(len(v) for v in all_lifespans.values())
                print(f"[results] {total_lifespans} lifespans and {N_VARIANTS} variant RNG seed sets collected")
    
    except PauseManagerExit:
        print("[EXIT] Batch simulation stopped by user.")
    finally:
        if VIZ_ENABLED or VIZ_BRAIN_ENABLED:
            cleanup_pause_manager()
        
    # the new eval_generation will end here. return of the function will be: all lifespans     
    return all_lifespans

# --- TRACKING VALIDATION ---

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
    
    # BUILD RNG STREAMS AT BATCH LEVEL (very first thing)
    SIMULATION_SEED = experiment_cfg["simulation_seed"]
    GENERATION_SEED = cfg["world"]["generation_seed"]
    
    # Build independent RNG streams for decision-making and neuron noise
    rng_decision, rng_neuron_noise, _ = build_rng_streams(SIMULATION_SEED)
    # Build RNG stream for world (food distribution)
    rng_world = build_rng_streams(GENERATION_SEED)[0]
    
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
    sensor_cfg = make_sensor_cfg(cfg)
    feeding_paradigm = cfg["food"]["feeding_paradigm"]
    feeding_initial_fraction_per_cell = cfg["food"]["initial_fraction_per_cell"]
    feeding_regrow_time = cfg["food"]["regrow_time"]
    brain_n_neurons = cfg["brain"]["n_neurons"]
    brain_threshold = cfg["brain"]["threshold"]
    brain_noise_level = cfg["brain"]["noise_level"]
    brain_sensory_mapping = cfg["brain"]["sensory_mapping"]
    brain_output_mapping = cfg["brain"]["output_mapping"]
    brain_max_decision_delay = cfg["brain"]["max_decision_delay"]
    
    # Wrap configs once to pass to workers (avoid 300k redundant wrappings)
    feeding_cfg = make_feeding_cfg(feeding_paradigm, feeding_initial_fraction_per_cell, feeding_regrow_time)
    brain_cfg = make_brain_cfg(brain_n_neurons, brain_threshold, brain_noise_level, brain_sensory_mapping, brain_output_mapping, brain_max_decision_delay)
    
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

