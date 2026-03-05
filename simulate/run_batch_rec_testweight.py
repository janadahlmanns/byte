# ------------------------------------------------------------
# run batch of simulations of Byte with the specified parameters
# randomizer seed is incremented by 1 with each iteration!
# visualization is optional and specified in pamaeter YAML
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

EXPERIMENT_FOLDER = "data/plasticity_test/rawdata/"
SIMULATION_NAME   = "plastic_w_0_1_eta_0_01_mod_neg_0_05"  # descriptive name for this batch of runs, used in output folder and file names

CONFIG_PATH = "configs/neurons_noise_plasticity.yaml"
BRAIN_INIT  = "plasticity"  # Set to brain init name (e.g., "prio_food") or "none" to disable
MAX_TICKS   = 2000
N_RUNS      = 1000  


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

def load_brain_init(brain_init_name: str):
    """Load brain initialization config. Returns (neuron_params, connections, sensory_mapping, max_decision_delay) or None."""
    if brain_init_name.lower() == "none" or not brain_init_name:
        return None
    module_name = f"configs.brain_init_{brain_init_name}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise ImportError(f"Could not find brain init module '{module_name}'.")
    if not hasattr(module, "build_brain_spec"):
        raise AttributeError(f"Brain init module '{module_name}' has no 'build_brain_spec' function.")
    return module.build_brain_spec()

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
        feeding_paradigm=f.get("feeding_paradigm", {"initial": True}),
        initial_fraction_per_cell=float(f["initial_fraction_per_cell"]),
        regrow_time=int(f["regrow_time"]),
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
    rows: list[tuple[int, int, int, int, float, bool, bool]]  # Added sensing and movement tracking
    prev_y: int = 0  # Track previous position to detect movement direction
    prev_x: int = 0

    @classmethod
    def empty(cls, worm: Worm):
        return cls(rows=[], prev_y=worm.y, prev_x=worm.x)

    def record(self, worm: Worm):
        # Get weight of connection 1→6 from brain state
        weight_1_6 = get_connection_weight(worm.brain, 1, 6)
        
        # Check if food was sensed ONLY to the north (not in other directions)
        sense = getattr(worm, "sensory_information", {})
        food_north = sense.get("food_north", 0.0) > 0.0
        food_east = sense.get("food_east", 0.0) > 0.0
        food_south = sense.get("food_south", 0.0) > 0.0
        food_west = sense.get("food_west", 0.0) > 0.0
        
        # Only flag if food is EXCLUSIVELY north
        food_north_only = food_north and not (food_east or food_south or food_west)
        
        # Check if worm moved north (y decreased, with wrapping)
        moved_north = False
        if worm.y != self.prev_y or worm.x != self.prev_x:
            # Movement occurred - check if it was north
            world = worm.world
            dy = (worm.y - self.prev_y) % world.height
            # North is dy == -1 (or world.height - 1 when wrapped)
            if dy == world.height - 1:  # Moved north
                moved_north = True
        
        self.rows.append(
            (worm.ticks, worm.energy, worm.eats, worm.distance, weight_1_6, food_north_only, moved_north)
        )
        
        # Update previous position for next call
        self.prev_y = worm.y
        self.prev_x = worm.x

    def save_csv(self, path: Path):
        lines = ["tick,energy,eats,distance,conn_1_6_weight,food_north_sensed,moved_north"]
        lines += [f"{t},{e},{k},{d},{w:.6f},{int(fn)},{int(mn)}" for t, e, k, d, w, fn, mn in self.rows]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================
# main
# ============================================================



def main():
    cfg = load_config(CONFIG_PATH)
    
    # Check for brain_init vs config consistency
    has_brain_config = cfg.get("decisionmaking", {}).get("brain", False)
    brain_init_spec = load_brain_init(BRAIN_INIT)
    
    if brain_init_spec is not None and not has_brain_config:
        print(f"[WARNING] BRAIN_INIT='{BRAIN_INIT}' specified but config has brain: false. Ignoring brain_init.")
    
    if brain_init_spec is None and has_brain_config:
        raise ValueError(f"Config specifies brain: true but BRAIN_INIT is 'none'. Please set BRAIN_INIT parameter.")
    
    # Check visualization settings for batch runs
    viz_cfg = cfg.get("viz", {})
    viz_enabled = bool(viz_cfg.get("enabled", False))
    
    if N_RUNS > 2 and viz_enabled:
        print(f"\n[WARNING] Visualization is enabled in config for {N_RUNS} runs.")
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

    # Save brain init file (if brain init is specified)
    if brain_init_spec is not None:
        brain_init_path = Path(f"configs/brain_init_{BRAIN_INIT}.py")
        if brain_init_path.exists():
            brain_init_content = brain_init_path.read_text(encoding="utf-8")
            (run_dir / f"brain_used_{SIMULATION_NAME}.py").write_text(
                brain_init_content,
                encoding="utf-8",
            )

    # Initialize pause manager only if visualization is enabled
    pause_mgr = init_pause_manager() if viz_enabled else None

    summary_lines = ["run_id,seed,lifetime_ticks,foods,distance,final_energy"]

    try:
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
            
            # DEBUG: Check initial connection weight on first run
            if run_id == 0:
                initial_weight = get_connection_weight(worm.brain, 1, 6)
                print(f"[DEBUG run 0] Initial connection weight (1 to 6): {initial_weight:.6f}")

            # Setup world visualization (if enabled)
            renderer = None
            if viz_enabled:
                renderer = QtRenderer(world, worm, fps=int(viz_cfg.get("fps", 10)))
            worm.renderer = renderer

            rec = MetricsRecorder.empty(worm)  # Pass worm to initialize position tracking
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

            run_file = run_dir / "runs" / f"run_{run_id:04d}.csv"
            rec.save_csv(run_file)

            # Count how many times food was sensed exclusively north
            food_north_only_count = sum(1 for row in rec.rows if row[5])  # row[5] is food_north_only

            summary_lines.append(
                f"{run_id},{seed},{worm.ticks},{worm.eats},{worm.distance},{worm.energy}"
            )

            print(f"[run {run_id:02d}] ticks={worm.ticks} eats={worm.eats} food_north_only={food_north_only_count}")

    except PauseManagerExit:
        print("[EXIT] Batch simulation stopped by user.")
    finally:
        if pause_mgr:
            cleanup_pause_manager()

    summary_name = f"summary_{SIMULATION_NAME}.csv"
    (run_dir / summary_name).write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )

    print("[batch] done.")


if __name__ == "__main__":
    main()
