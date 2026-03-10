# ------------------------------------------------------------
# run batch of simulations of Byte with randomized wiring
# randomizer seed is incremented by 1 with each variant!
# visualization is optional and specified in input parameters
# data are recorded into specified folder, plus summary data of the whole batch
# ------------------------------------------------------------

import importlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml
import numpy as np
import pandas as pd

from mvb.world import World, WorldConfig
from .pause_manager import init_pause_manager, cleanup_pause_manager, PauseManagerExit
from mvb.feeding import FeedingConfig, seed_food
from mvb.worm import Worm, WormConfig
from mvb.world_renderer_qt import QtRenderer


# ============================================================
# EXPERIMENT DEFINITION
# ============================================================

EXPERIMENT_FOLDER = "data/random_no_regrow/rawdata/"
SIMULATION_NAME   = "lookup_no_regrow_all_tracked"  # descriptive name for this batch of runs, used in output folder and file names

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
N_RUNS      = 1
INITIAL_FRACTION_PER_CELL = 0.25           # Initial fraction of food per cell
REGROW_TIME = 3000                           # Time for food to regrow

# ============================================================
# VISUALIZATION PARAMETERS
# ============================================================

VIZ_ENABLED = True                        # Enable visualization
VIZ_FPS = 4                                # Frames per second for world visualization
VIZ_BRAIN_ENABLED = True                  # Enable brain visualization
VIZ_BRAIN_FPS = 4                          # Frames per second for brain visualization

# ============================================================
# DATA TRACKING PARAMETERS
# ============================================================

ENABLE_PER_TICK_TRACKING = True              # Enable per-tick tracking and CSV export (tracks weights, sensory, movement, energy, distance, and decisions)
ENABLE_HEAT_MAP_TRACKING = True             # Enable tracking of Byte position heat map
# ============================================================
# helpers
# ============================================================

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_rng_streams(seed: int, has_brain: bool):
    """Build separate RNG streams for different aspects of simulation."""
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
    base = Path(EXPERIMENT_FOLDER)
    base.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = base / f"{ts}_{SIMULATION_NAME}"
    run_dir.mkdir()
    return run_dir


def append_wiring_column_to_csv(wiring_file: Path, run_id: int, final_weights: dict):
    """
    DEPRECATED: This function is no longer used because it causes dataframe fragmentation.
    Final weights are now collected in memory and written all at once per variant.
    """
    pass


@dataclass
class MetricsRecorder:
    rows: list[tuple]  # Main tracked data per tick
    per_tick_rows: list[tuple]  # Comprehensive per-tick tracking (if ENABLE_PER_TICK_TRACKING is True)
    connections_to_track: list[tuple]  # List of (src, tgt) pairs to track over time
    start_y: int = 0  # Starting Y position for manhattan distance calculation
    start_x: int = 0  # Starting X position for manhattan distance calculation
    prev_y: int = 0
    prev_x: int = 0
    prev_eats: int = 0  # Track food consumption this tick
    prev_action: tuple = None  # Track which movement happened
    grid_height: int = 0  # World grid height for heatmap indexing
    grid_width: int = 0  # World grid width for heatmap indexing
    entering_heatmap: dict = None  # {(y, x): count} - field entry counts
    staying_heatmap: dict = None  # {(y, x): count} - field ticks spent
    moves_north: int = 0
    moves_south: int = 0
    moves_east: int = 0
    moves_west: int = 0
    food_sensed_north: int = 0
    food_sensed_east: int = 0
    food_sensed_south: int = 0
    food_sensed_west: int = 0
    decisions: int = 0
    correct_decisions: int = 0

    @classmethod
    def empty(cls, worm: Worm, brain_init_spec):
        """Initialize recorder with brain init spec to extract connection tracking."""
        neuron_params, connections, sensory_mapping, max_decision_delay, eta, modulator_spec = brain_init_spec
        
        # Identify all non-zero connections at initialization
        connections_to_track = []
        for src in range(connections.shape[0]):
            for tgt in range(connections.shape[1]):
                if connections[src, tgt, 0] != 0.0:
                    connections_to_track.append((src, tgt))
        
        # Initialize heatmaps
        grid_height = worm.world.cfg.grid_height
        grid_width = worm.world.cfg.grid_width
        entering_heatmap = {}
        staying_heatmap = {}
        
        # Initialize all grid positions with 0, then set start position to 1 for entering
        for y in range(grid_height):
            for x in range(grid_width):
                entering_heatmap[(y, x)] = 0
                staying_heatmap[(y, x)] = 0
        
        # Starting position gets 1 enter count
        entering_heatmap[(worm.y, worm.x)] = 1
        
        return cls(
            rows=[],
            per_tick_rows=[],
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
            decisions=0,
            correct_decisions=0,
        )

    def record(self, worm: Worm):
        """
        Record metrics for this tick.
        - Always tracks: movement and food sensing (for summary statistics)
        - Conditionally tracks: comprehensive per-tick data (controlled by ENABLE_PER_TICK_TRACKING)
        """
        # Track directional movement
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
        
        # Track decision-making accuracy
        # A decision only happens if food was sensed AND worm is not on food
        on_food = sense.get("on_food", 0) > 0
        any_food_sensed = food_north or food_east or food_south or food_west
        
        if any_food_sensed and not on_food:
            # A decision opportunity exists (food was sensed)
            self.decisions += 1
            
            # Check if movement direction matches a sensed direction
            moved_north = dy < 0
            moved_south = dy > 0
            moved_east = dx > 0
            moved_west = dx < 0
            
            # Correct decision: movement is in one of the sensed directions
            if (moved_north and food_north) or \
               (moved_south and food_south) or \
               (moved_east and food_east) or \
               (moved_west and food_west):
                self.correct_decisions += 1
        
        # Track comprehensive per-tick data (if enabled)
        if ENABLE_PER_TICK_TRACKING:
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
            
            # Collect per-tick data
            tick_data = [
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
            ]
            
            # Add all connection weights
            for src, tgt in self.connections_to_track:
                w = get_connection_weight(worm.brain, src, tgt)
                tick_data.append(w)
            
            self.per_tick_rows.append(tuple(tick_data))
        
        # Track heatmaps (if enabled)
        if ENABLE_HEAT_MAP_TRACKING:
            # STAYING heatmap: increment for every tick on current field
            self.staying_heatmap[(worm.y, worm.x)] += 1
            
            # ENTERING heatmap: increment only when entering a new field (not when staying to eat)
            position_changed = (worm.y != self.prev_y) or (worm.x != self.prev_x)
            food_consumed = worm.eats > self.prev_eats
            stayed_to_eat = (self.prev_action is not None and 
                            self.prev_action[0] == "stay" and 
                            food_consumed)
            
            # Enter a new field incrementing (but NOT if we stayed in place to eat)
            if position_changed and not stayed_to_eat:
                self.entering_heatmap[(worm.y, worm.x)] += 1
        
        # Update previous position, eats count, and action for next call
        self.prev_y = worm.y
        self.prev_x = worm.x
        self.prev_eats = worm.eats
        self.prev_action = worm.action

    def save_csv(self, path: Path):
        lines = ["tick,energy,eats,distance,conn_1_6_weight,conn_2_7_weight,conn_3_8_weight,conn_4_9_weight,food_north_sensed,food_east_sensed,food_south_sensed,food_west_sensed"]
        lines += [f"{t},{e},{k},{d},{w16:.6f},{w27:.6f},{w38:.6f},{w49:.6f},{int(fn)},{int(fe)},{int(fs)},{int(fw)}" 
                  for t, e, k, d, w16, w27, w38, w49, fn, fe, fs, fw in self.rows]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    
    def save_per_tick_csv(self, path: Path):
        """Save comprehensive per-tick tracking data."""
        header = ["tick", "food_sensed_N", "food_sensed_E", "food_sensed_S", "food_sensed_W",
                  "movement", "food_consumed", "energy", "manhattan_dist", "decision_made"] + \
                 [f"conn_{src}_{tgt}" for src, tgt in self.connections_to_track]
        lines = [",".join(header)]
        for row in self.per_tick_rows:
            # First 10 columns: tick, food_sensed (4x), movement, food_consumed, energy, manhattan_dist, decision_made
            formatted_row = [str(row[0])] + [str(int(row[i])) for i in range(1, 5)] + [str(row[5]), str(row[6]), str(row[7]), str(row[8]), str(row[9])]
            # Remaining columns: connection weights (floats)
            formatted_row += [f"{val:.6f}" if isinstance(val, float) else str(val) for val in row[10:]]
            lines.append(",".join(formatted_row))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    
    def save_heatmaps_csv(self, path: Path):
        """Save entering and staying heatmaps for all grid positions."""
        header = ["field_y", "field_x", "entering_count", "staying_count"]
        lines = [",".join(header)]
        
        # Sort by y, then x for consistent output
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                entering_count = self.entering_heatmap.get((y, x), 0)
                staying_count = self.staying_heatmap.get((y, x), 0)
                lines.append(f"{y},{x},{entering_count},{staying_count}")
        
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    
    def save_wiring_snapshot(self, path: Path, brain_init_spec, label: str):
        """Save a snapshot of the wiring (initial or final)."""
        neuron_params, connections, sensory_mapping, max_decision_delay, eta, modulator_spec = brain_init_spec
        lines = ["src,tgt,weight"]
        for src, tgt in self.connections_to_track:
            weight = connections[src, tgt, 0]
            lines.append(f"{src},{tgt},{weight:.6f}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_wiring_summary(brain_init_spec):
    """Print a quick wiring diagram of connections and modulation."""
    neuron_params, connections, sensory_mapping, max_decision_delay, eta, modulator_spec = brain_init_spec
    
    # Count connections by type
    excitatory_conns = np.count_nonzero(connections[:, :, 0] > 0)
    inhibitory_conns = np.count_nonzero(connections[:, :, 0] < 0)
    total_conns = excitatory_conns + inhibitory_conns
    
    # Count modulations
    total_modulators = sum(len(mods) for mods in modulator_spec.values())
    potentiation_mods = sum(
        sum(1 for _, w in mods if w > 0)
        for mods in modulator_spec.values()
    )
    depression_mods = sum(
        sum(1 for _, w in mods if w < 0)
        for mods in modulator_spec.values()
    )
    
    print(f"\n  [WIRING] Connections: {total_conns} total ({excitatory_conns} exc, {inhibitory_conns} inh)")
    print(f"  [WIRING] Modulation: {total_modulators} total ({potentiation_mods} potentiation, {depression_mods} depression)")
    
    # List all non-zero connections
    print(f"  [CONNECTIONS]")
    for src in range(connections.shape[0]):
        for tgt in range(connections.shape[1]):
            weight = connections[src, tgt, 0]
            if weight != 0.0:
                conn_type = "exc" if weight > 0 else "inh"
                print(f"    {src:2d} -> {tgt:2d}  weight={weight:6.2f} ({conn_type})")
    
    # List modulations if present
    if modulator_spec:
        print(f"  [MODULATION]")
        for (conn_src, conn_tgt), mods in modulator_spec.items():
            if mods:
                mod_str = "; ".join([f"{m_src}({m_w:+.1f})" for m_src, m_w in mods])
                print(f"    {conn_src} -> {conn_tgt}:  {mod_str}")


# ============================================================
# main
# ============================================================



def main():
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
    
    # Check visualization settings for batch runs
    viz_enabled = VIZ_ENABLED
    
    if N_RUNS > 2 and viz_enabled:
        print(f"\n[WARNING] Visualization is enabled for {N_RUNS} runs.")
        print("This will be VERY SLOW. Batch runs typically disable visualization.")
        response = input("Continue with visualization? (y/n): ").strip().lower()
        if response != 'y':
            print("[INFO] Disabling visualization for this batch run.")
            viz_enabled = False
    
    brain = load_brain_module(make_decision_cfg(cfg))

    run_dir = make_experiment_dir()
    print(f"[batch] writing to {run_dir}")

    # Initialize pause manager only if visualization is enabled
    pause_mgr = init_pause_manager() if viz_enabled else None

    # ============================================================
    # OUTER LOOP: Iterate over wiring variants
    # ============================================================
    try:
        # Create one test variant to extract brain initialization parameters
        wiring_seed_test = WIRING_RANDOMIZATION_SEED
        brain_init_spec_test = load_brain_init(
            BRAIN_INIT,
            wiring_seed=wiring_seed_test,
            connectivity_degree_excitatory=CONNECTIVITY_DEGREE_EXCITATORY,
            connectivity_degree_inhibitory=CONNECTIVITY_DEGREE_INHIBITORY,
            modulation_degree_potentiation=MODULATION_DEGREE_POTENTIATION,
            modulation_degree_depression=MODULATION_DEGREE_DEPRESSION,
        )
        
        # Extract brain parameters from the spec
        neuron_params, connections, sensory_mapping, max_decision_delay, eta, modulator_spec = brain_init_spec_test
        n_neurons = neuron_params.shape[0]
        
        # Count total connections
        num_total_connections = int(np.count_nonzero(connections[:, :, 0]))
        
        # Infer weights from actual connections (take first one found)
        excitatory_weight = None
        inhibitory_weight = None
        for src in range(n_neurons):
            for tgt in range(n_neurons):
                w = connections[src, tgt, 0]
                if w > 0 and excitatory_weight is None:
                    excitatory_weight = float(w)
                elif w < 0 and inhibitory_weight is None:
                    inhibitory_weight = float(w)
        
        # Build comprehensive config dict
        comprehensive_config = {
            "experiment_metadata": {
                "experiment_folder": EXPERIMENT_FOLDER,
                "simulation_name": SIMULATION_NAME,
                "config_path": CONFIG_PATH,
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
                "initial_fraction_per_cell": INITIAL_FRACTION_PER_CELL,
                "regrow_time": REGROW_TIME,
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
            "world_config": cfg,
        }
        
        # Save comprehensive config as JSON
        import json
        config_json_path = run_dir / f"config_used_{SIMULATION_NAME}.json"
        with open(config_json_path, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_config, f, indent=2)
        print(f"[config] Saved comprehensive config to {config_json_path.name}")
        
        for variant_id in range(N_VARIANTS):
            print(f"[variant {variant_id+1:02d}/{N_VARIANTS:02d}] Simulating...")
            
            # Create randomized brain initialization for this variant
            wiring_seed = WIRING_RANDOMIZATION_SEED + variant_id
            brain_init_spec = load_brain_init(
                BRAIN_INIT,
                wiring_seed=wiring_seed,
                connectivity_degree_excitatory=CONNECTIVITY_DEGREE_EXCITATORY,
                connectivity_degree_inhibitory=CONNECTIVITY_DEGREE_INHIBITORY,
                modulation_degree_potentiation=MODULATION_DEGREE_POTENTIATION,
                modulation_degree_depression=MODULATION_DEGREE_DEPRESSION,
            )
            
            # Print wiring diagram
            # print_wiring_summary(brain_init_spec)  # Disabled for cleaner output
            
            # Create variant-specific subdirectory
            variant_dir = run_dir / f"variant_{variant_id+1:02d}"
            variant_dir.mkdir()
            (variant_dir / "runs").mkdir()
            
            # Save initial wiring for this variant (same for all runs)
            wiring_file = variant_dir / "wiring.csv"
            neuron_params, connections, sensory_mapping, max_decision_delay, eta, modulator_spec = brain_init_spec
            wiring_lines = ["src,tgt,weight_initial"]
            for src in range(connections.shape[0]):
                for tgt in range(connections.shape[1]):
                    if connections[src, tgt, 0] != 0.0:
                        weight = connections[src, tgt, 0]
                        wiring_lines.append(f"{src},{tgt},{weight:.6f}")
            wiring_file.write_text("\n".join(wiring_lines) + "\n", encoding="utf-8")
            
            # Save modulation spec for this variant
            modulation_file = variant_dir / "modulation.csv"
            modulation_lines = ["target_src,target_tgt,modulator_src,modulation_weight"]
            for (target_src, target_tgt), modulators in modulator_spec.items():
                for mod_src, mod_weight in modulators:
                    modulation_lines.append(f"{target_src},{target_tgt},{mod_src},{mod_weight:.6f}")
            modulation_file.write_text("\n".join(modulation_lines) + "\n", encoding="utf-8")
            
            summary_lines = ["run_id,lifetime_ticks,foods,distance,final_energy,moves_north,moves_south,moves_east,moves_west,food_sensed_north,food_sensed_east,food_sensed_south,food_sensed_west,decisions,correct_decisions"]
            
            # Collect final weights for all runs (to write all at once at the end)
            final_weights_all_runs = {}  # {run_id: {(src, tgt): weight}}
            
            # Collect heatmaps for all runs (to write all at once at the end)
            heatmaps_all_runs = {}  # {run_id: (entering_heatmap, staying_heatmap)}

            # ============================================================
            # INNER LOOP: Iterate over world seeds for this variant
            # ============================================================
            for run_id in range(N_RUNS):
                # Build RNG streams for this run
                rng_food, rng_decision, rng_neuron_noise = build_rng_streams(cfg["world"]["rng_seed"] + run_id, has_brain_config)

                world = make_world(cfg)
                feeding_cfg = make_feeding_cfg(cfg)
                world.feeding_cfg = feeding_cfg

                worm = make_worm(world, cfg)
                worm.active_sensors = make_sensor_cfg(cfg)
                worm.brain = brain
                if hasattr(worm.brain, "init"):
                    if brain_init_spec is not None:
                        worm.brain.init(worm, cfg, rng_neuron_noise, brain_init_spec=brain_init_spec)
                    else:
                        worm.brain.init(worm, cfg, rng_neuron_noise)

                reset_sim(world, feeding_cfg, rng_food, worm)

                # Setup world visualization (if enabled)
                renderer = None
                if viz_enabled:
                    renderer = QtRenderer(world, worm, fps=VIZ_FPS)
                worm.renderer = renderer

                rec = MetricsRecorder.empty(worm, brain_init_spec)
                rec.record(worm)

                while worm.alive and worm.ticks < MAX_TICKS:
                    # CHECKPOINT: Check for pause/exit
                    if pause_mgr:
                        pause_mgr.check_pause()

                    world.step()
                    worm.step_day(rng_decision)
                    worm.ticks += 1
                    rec.record(worm)
                    
                    # Frame pacing for visualization
                    if renderer:
                        renderer.wait_frame()

                # Clean up renderer for this run
                if renderer:
                    renderer.close()

                # ============================================================
                # Per-tick tracking: conditional per-tick CSV saves
                # ============================================================
                # Save comprehensive per-tick data (if enabled)
                if ENABLE_PER_TICK_TRACKING:
                    per_tick_file = variant_dir / "runs" / f"run_{run_id+1:04d}_per_tick.csv"
                    rec.save_per_tick_csv(per_tick_file)
                
                # Collect heatmaps for this run (to write all at once per variant)
                if ENABLE_HEAT_MAP_TRACKING:
                    heatmaps_all_runs[run_id+1] = (rec.entering_heatmap.copy(), rec.staying_heatmap.copy())
                
                # Collect final wiring for this run (to be written all at once per variant)
                final_weights_dict = {}
                for src, tgt in rec.connections_to_track:
                    w = get_connection_weight(worm.brain, src, tgt)
                    final_weights_dict[(src, tgt)] = w
                final_weights_all_runs[run_id+1] = final_weights_dict

                summary_lines.append(
                    f"{run_id+1},{worm.ticks},{worm.eats},{worm.distance},{worm.energy},{rec.moves_north},{rec.moves_south},{rec.moves_east},{rec.moves_west},{rec.food_sensed_north},{rec.food_sensed_east},{rec.food_sensed_south},{rec.food_sensed_west},{rec.decisions},{rec.correct_decisions}"
                )

            # ============================================================
            # Write all final weights to wiring file at once (avoid fragmentation)
            # ============================================================
            df_wiring = pd.read_csv(wiring_file)
            
            # Build all final weight columns at once before assigning
            final_weights = {}
            for run_id, weights_dict in final_weights_all_runs.items():
                column_name = f"weight_final_run_{run_id:04d}"
                weights_list = []
                for _, row in df_wiring.iterrows():
                    src = int(row['src'])
                    tgt = int(row['tgt'])
                    weight = weights_dict.get((src, tgt), 0.0)
                    weights_list.append(weight)
                final_weights[column_name] = weights_list
            
            # Create new DataFrame with all weight columns and concatenate
            df_final_weights = pd.DataFrame(final_weights)
            df_wiring = pd.concat([df_wiring, df_final_weights], axis=1)
            df_wiring.to_csv(wiring_file, index=False)

            # Save summary for this variant
            summary_name = f"summary_{SIMULATION_NAME}.csv"
            (variant_dir / summary_name).write_text(
                "\n".join(summary_lines) + "\n", encoding="utf-8"
            )
            
            # Save consolidated heatmaps for all runs in this variant
            if ENABLE_HEAT_MAP_TRACKING and heatmaps_all_runs:
                # Get grid dimensions from first run's heatmap
                first_entering = list(heatmaps_all_runs.values())[0][0]
                grid_height = max(y for y, x in first_entering.keys()) + 1
                grid_width = max(x for y, x in first_entering.keys()) + 1
                
                # Build header: field_y, field_x, then alternating entering/staying for each run
                header = ["field_y", "field_x"]
                for run_id in sorted(heatmaps_all_runs.keys()):
                    header.append(f"entering_run_{run_id:04d}")
                    header.append(f"staying_run_{run_id:04d}")
                
                # Build rows: linearize grid and collect data from all runs
                lines = [",".join(header)]
                for y in range(grid_height):
                    for x in range(grid_width):
                        row = [str(y), str(x)]
                        for run_id in sorted(heatmaps_all_runs.keys()):
                            entering_heatmap, staying_heatmap = heatmaps_all_runs[run_id]
                            entering_count = entering_heatmap.get((y, x), 0)
                            staying_count = staying_heatmap.get((y, x), 0)
                            row.append(str(entering_count))
                            row.append(str(staying_count))
                        lines.append(",".join(row))
                
                # Write consolidated heatmaps file for this variant
                heatmaps_name = f"heatmaps_{SIMULATION_NAME}.csv"
                (variant_dir / heatmaps_name).write_text(
                    "\n".join(lines) + "\n", encoding="utf-8"
                )

    except PauseManagerExit:
        print("[EXIT] Batch simulation stopped by user.")
    finally:
        if pause_mgr:
            cleanup_pause_manager()

    print("[batch] all variants done.")


if __name__ == "__main__":
    main()
