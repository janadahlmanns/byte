# Batch simulation of Byte with randomized wiring variants
# Each variant's wiring seed is incremented by 1
# Results are saved to HDF5

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
from .pause_manager import init_pause_manager, cleanup_pause_manager, PauseManagerExit
from mvb.feeding import FeedingConfig, seed_food
from mvb.worm import Worm, WormConfig
from mvb.world_renderer_qt import QtRenderer
from .hdf5_utils import (
    create_hdf5_file,
    write_variant_to_hdf5,
)


# ============================================================
# EXPERIMENT DEFINITION
# ============================================================

EXPERIMENT_FOLDER = "data/random_vs_lookup/"
SIMULATION_NAME   = "lookup"  # descriptive name for this batch of runs, used in output folder and file names

CONFIG_PATH = "configs/neurons_random_wiring.yaml"
BRAIN_INIT  = "random_lookup"  # Set to "random" for randomized wiring

# ============================================================
# WIRING RANDOMIZATION PARAMETERS
# ============================================================

CONNECTIVITY_DEGREE_EXCITATORY = 0.2       # Fraction of excitatory connections
CONNECTIVITY_DEGREE_INHIBITORY = 0.4       # Fraction of inhibitory connections
MODULATION_DEGREE_POTENTIATION = 0.5       # Fraction for potentiation modulation
MODULATION_DEGREE_DEPRESSION = 0.5        # Fraction for depression modulation
WIRING_RANDOMIZATION_SEED = 1              # Base seed for wiring randomization
N_VARIANTS = 1                            # Number of randomized wiring variants to generate

# ============================================================
# SIMULATION PARAMETERS 
# ============================================================

MAX_TICKS   = 2000
N_RUNS      = 300                           # 300 runs per variant as determined by convergence analysis
INITIAL_FRACTION_PER_CELL = 0.25           # Initial fraction of food per cell
REGROW_TIME = 3000                           # Time for food to regrow

# ============================================================
# VISUALIZATION PARAMETERS
# ============================================================

VIZ_ENABLED = False                        # Enable visualization
VIZ_FPS = 4                                # Frames per second for world visualization
VIZ_BRAIN_ENABLED = False                  # Enable brain visualization
VIZ_BRAIN_FPS = 4                          # Frames per second for brain visualization

# ============================================================
# DATA TRACKING PARAMETERS
# ============================================================

ENABLE_PER_TICK_TRACKING = True              # Enable per-tick tracking and CSV export (tracks weights, sensory, movement, energy, distance, and decisions)
ENABLE_HEAT_MAP_TRACKING = True             # Enable tracking of Byte position heat map

# ============================================================
# helpers
# ============================================================

def get_num_workers():
    """Determine number of worker processes. Reserves 2 cores for system tasks.
    Returns None if system has ≤2 cores (force serial execution)."""
    try:
        available_cores = os.cpu_count()
        if available_cores is None or available_cores <= 2:
            return None
        return max(1, available_cores - 2)
    except Exception:
        return None

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_rng_streams(seed: int, has_brain: bool):
    """Build separate RNG streams for simulation aspects."""
    seed = int(seed)
    rng_food = np.random.default_rng(seed)
    rng_decision = np.random.default_rng(seed)
    rng_neuron_noise = np.random.default_rng(seed) if has_brain else None
    return rng_food, rng_decision, rng_neuron_noise

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

def make_world(cfg_yaml):
    return World(
        WorldConfig(
            grid_width=int(cfg_yaml["world"]["grid_width"]),
            grid_height=int(cfg_yaml["world"]["grid_height"]),
            start_pos=tuple(cfg_yaml["world"]["start_pos"]),
            rng_seed=int(cfg_yaml["world"]["rng_seed"]),
        ),
    )

def make_feeding_cfg(cfg_yaml):
    f = cfg_yaml["food"]
    return FeedingConfig(
        feeding_paradigm=f.get("feeding_paradigm", {"initial": True, "regrow": True}),
        initial_fraction_per_cell=INITIAL_FRACTION_PER_CELL,
        regrow_time=REGROW_TIME,
    )

def make_worm(world, cfg_yaml):
    w = cfg_yaml["worm"]
    return Worm(
        WormConfig(
            speed=int(w["speed"]),
            energy_capacity=int(w["energy_capacity"]),
            metabolic_rate=int(w["metabolic_rate"]),
        ),
        world,
    )

def make_sensor_cfg(cfg_yaml):
    return cfg_yaml.get("sensors", {}).get("active", ["current_field"])

def make_decision_cfg(cfg_yaml):
    return str(cfg_yaml["decisionmaking"]["version"])

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


# ============================================================
# output + metrics
# ============================================================

def make_experiment_dir() -> Path:
    """Create HDF5 file path for experiment.
    
    Returns:
        Path to HDF5 file for saving all results.
    """
    base = Path(EXPERIMENT_FOLDER)
    base.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    hdf5_path = base / f"{ts}_{SIMULATION_NAME}.h5"
    
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
    entering_heatmap: np.ndarray = None  # 2D array (height, width) - field entry counts
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

    @classmethod
    def empty(cls, worm: Worm, brain_init_spec):
        """Initialize recorder with brain init spec and worm state."""
        neuron_params, connections, sensory_mapping, max_decision_delay, eta, modulator_spec = brain_init_spec
        
        connections_to_track = []
        for src in range(connections.shape[0]):
            for tgt in range(connections.shape[1]):
                if connections[src, tgt, 0] != 0.0:
                    connections_to_track.append((src, tgt))
        
        grid_height = worm.world.cfg.grid_height
        grid_width = worm.world.cfg.grid_width
        entering_heatmap = np.zeros((grid_height, grid_width), dtype=np.int32)
        staying_heatmap = np.zeros((grid_height, grid_width), dtype=np.int32)
        entering_heatmap[worm.y, worm.x] = 1
        
        per_tick_data = None
        if ENABLE_PER_TICK_TRACKING:
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
            per_tick_data = np.zeros(MAX_TICKS, dtype=dtype_fields)
        
        return cls(
            per_tick_data=per_tick_data,
            per_tick_count=0,
            connections_to_track=connections_to_track,
            start_y=worm.y,
            start_x=worm.x,
            prev_y=worm.y,
            prev_x=worm.x,
            prev_eats=worm.eats,
            prev_action=None,
            grid_height=grid_height,
            grid_width=grid_width,
            entering_heatmap=entering_heatmap,
            staying_heatmap=staying_heatmap,
            moves_north=0,
            moves_south=0,
            moves_east=0,
            moves_west=0,
            food_sensed_north=0,
            food_sensed_east=0,
            food_sensed_south=0,
            food_sensed_west=0,
            prev_on_food=False,
            prev_action_was_decision=False,
            decisions=0,
            correct_decisions=0,
        )

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
        if ENABLE_PER_TICK_TRACKING and self.per_tick_data is not None:
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
        
        if ENABLE_HEAT_MAP_TRACKING:
            self.staying_heatmap[worm.y, worm.x] += 1
            position_changed = (worm.y != self.prev_y) or (worm.x != self.prev_x)
            food_consumed = worm.eats > self.prev_eats
            stayed_to_eat = (self.prev_action is not None and 
                            self.prev_action[0] == "stay" and 
                            food_consumed)
            if position_changed and not stayed_to_eat:
                self.entering_heatmap[worm.y, worm.x] += 1
        
        self.prev_y = worm.y
        self.prev_x = worm.x
        self.prev_eats = worm.eats
        self.prev_action = worm.action





# ============================================================
# worker function for parallel execution
# ============================================================

def run_variant_worker(
    variant_id,
    brain_module_name,
    cfg,
    hdf5_path,
    write_lock,
    enable_per_tick_tracking,
    enable_heat_map_tracking,
):
    """Execute a single variant's simulation runs and return data for HDF5 write."""
    
    brain_module = load_brain_module(brain_module_name)
    
    # Calculate wiring seed for this variant
    wiring_seed = WIRING_RANDOMIZATION_SEED + variant_id
    
    brain_init_spec = load_brain_init(
        BRAIN_INIT,
        wiring_seed=wiring_seed,
        connectivity_degree_excitatory=CONNECTIVITY_DEGREE_EXCITATORY,
        connectivity_degree_inhibitory=CONNECTIVITY_DEGREE_INHIBITORY,
        modulation_degree_potentiation=MODULATION_DEGREE_POTENTIATION,
        modulation_degree_depression=MODULATION_DEGREE_DEPRESSION,
    )
    
    # Extract brain spec components
    neuron_params, connections, sensory_mapping, max_decision_delay, eta, modulator_spec = brain_init_spec
    
    # Identify non-zero connections at initialization
    connections_to_track = []
    for src in range(connections.shape[0]):
        for tgt in range(connections.shape[1]):
            if connections[src, tgt, 0] != 0.0:
                connections_to_track.append((src, tgt))
    
    # Pre-allocate wiring array with columns for all run final weights
    dtype_wiring = [('src', 'i2'), ('tgt', 'i2'), ('weight_initial', 'f4')]
    for run_id in range(1, N_RUNS + 1):
        dtype_wiring.append((f'weight_final_run_{run_id:04d}', 'f4'))
    wiring_array = np.zeros(len(connections_to_track), dtype=dtype_wiring)
    
    for idx, (src, tgt) in enumerate(connections_to_track):
        wiring_array[idx]['src'] = src
        wiring_array[idx]['tgt'] = tgt
        wiring_array[idx]['weight_initial'] = connections[src, tgt, 0]
    
    # Pre-allocate modulation array
    dtype_modulation = [('target_src', 'i2'), ('target_tgt', 'i2'), ('modulator_src', 'i2'), ('modulation_weight', 'f4')]
    modulation_list = []
    for (target_src, target_tgt), modulators in modulator_spec.items():
        for mod_src, mod_weight in modulators:
            modulation_list.append((target_src, target_tgt, mod_src, mod_weight))
    modulation_array = np.array(modulation_list, dtype=dtype_modulation) if modulation_list else np.array([], dtype=dtype_modulation)
    
    # Pre-allocate summary array
    dtype_summary = [('run_id', 'i2'), ('lifetime_ticks', 'i4'), ('foods', 'i4'), 
                     ('distance', 'i4'), ('final_energy', 'f4'),
                     ('moves_north', 'i4'), ('moves_south', 'i4'), ('moves_east', 'i4'), ('moves_west', 'i4'),
                     ('food_sensed_north', 'i4'), ('food_sensed_east', 'i4'), ('food_sensed_south', 'i4'), ('food_sensed_west', 'i4'),
                     ('decisions', 'i4'), ('correct_decisions', 'i4')]
    summary_array = np.zeros(N_RUNS, dtype=dtype_summary)
    
    heatmaps_all_runs = {}
    per_tick_all_runs = {}
    
    has_brain_config = cfg.get("decisionmaking", {}).get("brain", False)
    
    for run_id in range(N_RUNS):
        rng_food, rng_decision, rng_neuron_noise = build_rng_streams(
            cfg["world"]["rng_seed"] + run_id, has_brain_config
        )
        
        world = make_world(cfg)
        feeding_cfg = make_feeding_cfg(cfg)
        world.feeding_cfg = feeding_cfg
        
        worm = make_worm(world, cfg)
        worm.active_sensors = make_sensor_cfg(cfg)
        worm.brain = brain_module
        if hasattr(worm.brain, "init"):
            if brain_init_spec is not None:
                worm.brain.init(worm, cfg, rng_neuron_noise, brain_init_spec=brain_init_spec)
            else:
                worm.brain.init(worm, cfg, rng_neuron_noise)
        
        reset_sim(world, feeding_cfg, rng_food, worm)
        worm.renderer = None
        
        rec = MetricsRecorder.empty(worm, brain_init_spec)
        rec.record(worm)
        
        while worm.alive and worm.ticks < MAX_TICKS:
            world.step()
            worm.step_day(rng_decision)
            worm.ticks += 1
            rec.record(worm)
        
        if ENABLE_PER_TICK_TRACKING and rec.per_tick_data is not None:
            per_tick_all_runs[run_id+1] = rec.per_tick_data[:rec.per_tick_count]
        
        if ENABLE_HEAT_MAP_TRACKING:
            heatmaps_all_runs[run_id+1] = (rec.entering_heatmap.copy(), rec.staying_heatmap.copy())
        
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
    
    # WRITE DIRECTLY TO HDF5 BEFORE RETURNING (with lock serialization)
    write_variant_to_hdf5(
        hdf5_path, 
        variant_id + 1, 
        write_lock,
        wiring_array,
        modulation_array,
        summary_array,
        per_tick_all_runs=per_tick_all_runs,
        heatmaps_all_runs=heatmaps_all_runs,
        enable_per_tick_tracking=enable_per_tick_tracking,
        enable_heat_map_tracking=enable_heat_map_tracking,
    )
    
    return variant_id


# ============================================================
# main
# ============================================================



def main():
    # ============================================================
    # SETUP & CONFIGURATION
    # ============================================================
    cfg = load_config(CONFIG_PATH)
    
    # Apply visualization parameters from top of file to the config
    if "viz" not in cfg:
        cfg["viz"] = {}
    cfg["viz"]["enabled"] = VIZ_ENABLED
    cfg["viz"]["fps"] = VIZ_FPS
    cfg["viz"]["brain_enabled"] = VIZ_BRAIN_ENABLED
    cfg["viz"]["brain_fps"] = VIZ_BRAIN_FPS
    
    # Check for brain_init vs config consistency
    has_brain_config = cfg.get("decisionmaking", {}).get("brain", False)
    
    if BRAIN_INIT.lower() == "none" and has_brain_config:
        raise ValueError(f"Config specifies brain: true but BRAIN_INIT is 'none'. Please set BRAIN_INIT parameter.")
    
    viz_enabled = VIZ_ENABLED
    
    if N_RUNS > 2 and viz_enabled:
        print(f"\n[WARNING] Visualization is enabled for {N_RUNS} runs.")
        print("This will be VERY SLOW. Batch runs typically disable visualization.")
        response = input("Continue with visualization? (y/n): ").strip().lower()
        if response != 'y':
            print("[INFO] Disabling visualization for this batch run.")
            viz_enabled = False
    
    num_workers = None
    if viz_enabled and VIZ_BRAIN_ENABLED:
        print("[INFO] Brain visualization enabled. Running serially.")
    else:
        num_workers = get_num_workers()
        if num_workers is None:
            print("[INFO] Insufficient CPU cores. Running serially.")
        else:
            available_cores = os.cpu_count()
            print(f"[INFO] Parallel execution on {num_workers} cores ({available_cores} total).")
    
    brain = load_brain_module(make_decision_cfg(cfg))
    hdf5_path = make_experiment_dir()
    print(f"[batch] writing to {hdf5_path}\n")
    pause_mgr = init_pause_manager() if viz_enabled else None

    # ============================================================
    # INITIALIZE HDF5 FILE
    # ============================================================
    try:
        brain_module_name = make_decision_cfg(cfg)
        brain_init_spec_test = load_brain_init(
            BRAIN_INIT,
            wiring_seed=WIRING_RANDOMIZATION_SEED,
            connectivity_degree_excitatory=CONNECTIVITY_DEGREE_EXCITATORY,
            connectivity_degree_inhibitory=CONNECTIVITY_DEGREE_INHIBITORY,
            modulation_degree_potentiation=MODULATION_DEGREE_POTENTIATION,
            modulation_degree_depression=MODULATION_DEGREE_DEPRESSION,
        )
        
        neuron_params, connections, sensory_mapping, max_decision_delay, eta, modulator_spec = brain_init_spec_test
        n_neurons = neuron_params.shape[0]
        excitatory_weight = None
        inhibitory_weight = None
        for src in range(n_neurons):
            for tgt in range(n_neurons):
                w = connections[src, tgt, 0]
                if w > 0 and excitatory_weight is None:
                    excitatory_weight = float(w)
                elif w < 0 and inhibitory_weight is None:
                    inhibitory_weight = float(w)
        
        comprehensive_config = {
            "experiment_metadata": {
                "experiment_folder": EXPERIMENT_FOLDER,
                "simulation_name": SIMULATION_NAME,
                "simulation_config_path": CONFIG_PATH,
                "brain_init_type": BRAIN_INIT,
            },
            "wiring_randomization": {
                "connectivity_degree_excitatory": CONNECTIVITY_DEGREE_EXCITATORY,
                "connectivity_degree_inhibitory": CONNECTIVITY_DEGREE_INHIBITORY,
                "modulation_degree_potentiation": MODULATION_DEGREE_POTENTIATION,
                "modulation_degree_depression": MODULATION_DEGREE_DEPRESSION,
                "wiring_randomization_seed_base": WIRING_RANDOMIZATION_SEED,
                "n_variants": N_VARIANTS,
            },
            "simulation_parameters": {
                "max_ticks": MAX_TICKS,
                "n_runs": N_RUNS,
                "feeding_initial_fraction_per_cell": INITIAL_FRACTION_PER_CELL,
                "feeding_regrow_time": REGROW_TIME,
            },
            "data_tracking": {
                "enable_per_tick_tracking": ENABLE_PER_TICK_TRACKING,
                "enable_heat_map_tracking": ENABLE_HEAT_MAP_TRACKING,
            },
            "brain_architecture": {
                "n_neurons": int(n_neurons),
                "max_decision_delay": float(max_decision_delay),
                "eta": float(eta),
                "excitatory_weight": excitatory_weight,
                "inhibitory_weight": inhibitory_weight,
                "n_input_neurons": 5,
                "n_output_neurons": 5,
                "n_always_on_neurons": 1,
            },
        }
        
        comprehensive_config.update(_rename_world_config_keys(cfg))
        create_hdf5_file(hdf5_path, comprehensive_config)
        print(f"[config] Created HDF5 file: {hdf5_path.name}\n")
        
        # Run simulation
        
        if num_workers is None:
            with Manager() as manager:
                write_lock = manager.Lock()
                
                for variant_id in range(N_VARIANTS):
                    print(f"[variant {variant_id+1:02d}/{N_VARIANTS:02d}] Simulating...", end='', flush=True)
                    
                    run_variant_worker(
                        variant_id,
                        brain_module_name,
                        cfg,
                        hdf5_path,
                        write_lock,
                        ENABLE_PER_TICK_TRACKING,
                        ENABLE_HEAT_MAP_TRACKING,
                    )
                    
                    print(" done")
        
        else:
            completed = 0
            
            with Manager() as manager:
                write_lock = manager.Lock()
                
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = {}
                    for variant_id in range(N_VARIANTS):
                        future = executor.submit(
                            run_variant_worker,
                            variant_id,
                            brain_module_name,
                            cfg,
                            hdf5_path,
                            write_lock,
                            ENABLE_PER_TICK_TRACKING,
                            ENABLE_HEAT_MAP_TRACKING,
                        )
                        futures[future] = variant_id
                    
                    for future in as_completed(futures):
                        completed += 1
                        variant_id = future.result()  # Only returns variant_id after HDF5 write complete
                        print(f"\rProcessing variants... ({completed}/{N_VARIANTS} completed)", end='', flush=True)
                
                print()
        
        print(f"[batch] Simulation completed. Saved to {hdf5_path.name}")
    
    except PauseManagerExit:
        print("[EXIT] Batch simulation stopped by user.")
    finally:
        if pause_mgr:
            cleanup_pause_manager()


if __name__ == "__main__":
    main()
