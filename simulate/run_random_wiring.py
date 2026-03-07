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

from mvb.world import World, WorldConfig
from .pause_manager import init_pause_manager, cleanup_pause_manager, PauseManagerExit
from mvb.feeding import FeedingConfig, seed_food
from mvb.worm import Worm, WormConfig
from mvb.world_renderer_qt import QtRenderer


# ============================================================
# EXPERIMENT DEFINITION
# ============================================================

EXPERIMENT_FOLDER = "data/random_wiring/rawdata/"
SIMULATION_NAME   = "random_wiring_test"  # descriptive name for this batch of runs, used in output folder and file names

CONFIG_PATH = "configs/neurons_random_wiring.yaml"
BRAIN_INIT  = "random"  # Set to "random" for randomized wiring

# ============================================================
# WIRING RANDOMIZATION PARAMETERS
# ============================================================

CONNECTIVITY_DEGREE_EXCITATORY = 0.2       # Fraction of excitatory connections
CONNECTIVITY_DEGREE_INHIBITORY = 0.4       # Fraction of inhibitory connections
MODULATION_DEGREE_POTENTIATION = 0.1       # Fraction for potentiation modulation
MODULATION_DEGREE_DEPRESSION = 0.05        # Fraction for depression modulation
WIRING_RANDOMIZATION_SEED = 1              # Base seed for wiring randomization
N_VARIANTS = 500                             # Number of randomized wiring variants to generate

# ============================================================
# SIMULATION PARAMETERS
# ============================================================

MAX_TICKS   = 1000
N_RUNS      = 100
INITIAL_FRACTION_PER_CELL = 0.25           # Initial fraction of food per cell
REGROW_TIME = 15                           # Time for food to regrow

# ============================================================
# VISUALIZATION PARAMETERS
# ============================================================

VIZ_ENABLED = False                        # Enable visualization
VIZ_FPS = 4                                # Frames per second for world visualization
VIZ_BRAIN_ENABLED = False                  # Enable brain visualization
VIZ_BRAIN_FPS = 4                          # Frames per second for brain visualization

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
    (run_dir / "runs").mkdir()
    return run_dir


@dataclass
class MetricsRecorder:
    rows: list[tuple]  # Main tracked data per tick
    weight_rows: list[tuple]  # All connection weights per tick
    connections_to_track: list[tuple]  # List of (src, tgt) pairs to track over time
    prev_y: int = 0
    prev_x: int = 0
    moves_north: int = 0
    moves_south: int = 0
    moves_east: int = 0
    moves_west: int = 0

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
        
        return cls(
            rows=[],
            weight_rows=[],
            connections_to_track=connections_to_track,
            prev_y=worm.y,
            prev_x=worm.x,
            moves_north=0,
            moves_south=0,
            moves_east=0,
            moves_west=0,
        )

    def record(self, worm: Worm):
        # Get weights of all 4 plastic direction connections
        weight_1_6 = get_connection_weight(worm.brain, 1, 6)  # north
        weight_2_7 = get_connection_weight(worm.brain, 2, 7)  # east
        weight_3_8 = get_connection_weight(worm.brain, 3, 8)  # south
        weight_4_9 = get_connection_weight(worm.brain, 4, 9)  # west
        
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
        
        self.rows.append(
            (worm.ticks, worm.energy, worm.eats, worm.distance, 
             weight_1_6, weight_2_7, weight_3_8, weight_4_9,
             food_north, food_east, food_south, food_west)
        )
        
        # Track all connection weights for this tick
        tick_weights = [worm.ticks]
        for src, tgt in self.connections_to_track:
            w = get_connection_weight(worm.brain, src, tgt)
            tick_weights.append(w)
        self.weight_rows.append(tuple(tick_weights))
        
        # Update previous position for next call
        self.prev_y = worm.y
        self.prev_x = worm.x

    def save_csv(self, path: Path):
        lines = ["tick,energy,eats,distance,conn_1_6_weight,conn_2_7_weight,conn_3_8_weight,conn_4_9_weight,food_north_sensed,food_east_sensed,food_south_sensed,food_west_sensed"]
        lines += [f"{t},{e},{k},{d},{w16:.6f},{w27:.6f},{w38:.6f},{w49:.6f},{int(fn)},{int(fe)},{int(fs)},{int(fw)}" 
                  for t, e, k, d, w16, w27, w38, w49, fn, fe, fs, fw in self.rows]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    
    def save_weights_csv(self, path: Path):
        """Save all tracked connection weights over time."""
        header = ["tick"] + [f"conn_{src}_{tgt}" for src, tgt in self.connections_to_track]
        lines = [",".join(header)]
        lines += [",".join([f"{val:.6f}" if isinstance(val, float) else str(val) for val in row]) 
                  for row in self.weight_rows]
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

    # snapshot config
    (run_dir / f"config_used_{SIMULATION_NAME}.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False),
        encoding="utf-8",
    )

    # Initialize pause manager only if visualization is enabled
    pause_mgr = init_pause_manager() if viz_enabled else None

    # ============================================================
    # OUTER LOOP: Iterate over wiring variants
    # ============================================================
    try:
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
            initial_wiring_file = variant_dir / "wiring_initial.csv"
            neuron_params, connections, sensory_mapping, max_decision_delay, eta, modulator_spec = brain_init_spec
            initial_lines = ["src,tgt,weight"]
            for src in range(connections.shape[0]):
                for tgt in range(connections.shape[1]):
                    if connections[src, tgt, 0] != 0.0:
                        weight = connections[src, tgt, 0]
                        initial_lines.append(f"{src},{tgt},{weight:.6f}")
            initial_wiring_file.write_text("\n".join(initial_lines) + "\n", encoding="utf-8")
            
            # Save modulation spec for this variant
            modulation_file = variant_dir / "modulation.csv"
            modulation_lines = ["target_src,target_tgt,modulator_src,modulation_weight"]
            for (target_src, target_tgt), modulators in modulator_spec.items():
                for mod_src, mod_weight in modulators:
                    modulation_lines.append(f"{target_src},{target_tgt},{mod_src},{mod_weight:.6f}")
            modulation_file.write_text("\n".join(modulation_lines) + "\n", encoding="utf-8")
            
            summary_lines = ["run_id,seed,lifetime_ticks,foods,distance,final_energy,moves_north,moves_south,moves_east,moves_west"]

            # ============================================================
            # INNER LOOP: Iterate over world seeds for this variant
            # ============================================================
            for run_id in range(N_RUNS):
                seed = cfg["world"]["rng_seed"] + run_id
                
                # Build RNG streams for this run
                rng_food, rng_decision, rng_neuron_noise = build_rng_streams(seed, has_brain_config)

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

                # Save metrics and tracking data for this run
                run_file = variant_dir / "runs" / f"run_{run_id+1:04d}.csv"
                rec.save_csv(run_file)
                
                weights_file = variant_dir / "runs" / f"run_{run_id+1:04d}_weights.csv"
                rec.save_weights_csv(weights_file)
                
                # Save final wiring for this run
                final_wiring_file = variant_dir / "runs" / f"run_{run_id+1:04d}_wiring_final.csv"
                final_lines = ["src,tgt,weight"]
                for src, tgt in rec.connections_to_track:
                    w = get_connection_weight(worm.brain, src, tgt)
                    final_lines.append(f"{src},{tgt},{w:.6f}")
                final_wiring_file.write_text("\n".join(final_lines) + "\n", encoding="utf-8")

                summary_lines.append(
                    f"{run_id+1},{seed},{worm.ticks},{worm.eats},{worm.distance},{worm.energy},{rec.moves_north},{rec.moves_south},{rec.moves_east},{rec.moves_west}"
                )

            # Save summary for this variant
            summary_name = f"summary_{SIMULATION_NAME}.csv"
            (variant_dir / summary_name).write_text(
                "\n".join(summary_lines) + "\n", encoding="utf-8"
            )

    except PauseManagerExit:
        print("[EXIT] Batch simulation stopped by user.")
    finally:
        if pause_mgr:
            cleanup_pause_manager()

    print("[batch] all variants done.")


if __name__ == "__main__":
    main()
