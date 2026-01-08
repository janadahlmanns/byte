# ------------------------------------------------------------
# run one single simulation of Byte with the specified parameters
# visualization is optional and specified in parameter YAML
# data are recorded into specified folder
# ------------------------------------------------------------

import importlib
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

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

EXPERIMENT_FOLDER = "data/temp/rawdata/"
SIMULATION_NAME   = "neurons_w_noise"

CONFIG_PATH = "configs/sensing_neurons.yaml"
BRAIN_INIT  = "prio_food"  # Set to brain init name (e.g., "prio_food") or "none" to disable
MAX_TICKS   = 1000

   
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
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise ImportError(f"Could not find brain module '{module_name}'.")
    if not hasattr(module, "decide"):
        raise AttributeError(f"Brain module '{module_name}' has no 'decide' function.")
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
    wcfg = WorldConfig(
        grid_width=int(cfg_yaml["world"]["grid_width"]),
        grid_height=int(cfg_yaml["world"]["grid_height"]),
        start_pos=tuple(cfg_yaml["world"]["start_pos"]),
        rng_seed=int(cfg_yaml["world"]["rng_seed"]),
    )
    return World(wcfg)

def make_feeding_cfg(cfg_yaml):
    f = cfg_yaml["food"]
    return FeedingConfig(
        feeding_paradigm=f.get("feeding_paradigm", {"initial": True}),
        initial_fraction_per_cell=float(f["initial_fraction_per_cell"]),
        regrow_time=int(f["regrow_time"]),
    )

def make_worm(world, cfg_yaml):
    w = cfg_yaml["worm"]
    wcfg = WormConfig(
        speed=int(w["speed"]),
        energy_capacity=int(w["energy_capacity"]),
        metabolic_rate=int(w["metabolic_rate"]),
    )
    return Worm(wcfg, world)

def make_sensor_cfg(cfg_yaml):
    sensors = cfg_yaml.get("sensors", {})
    return sensors.get("active", ["current_field"])

def make_decision_cfg(cfg_yaml):
    return str(cfg_yaml["decisionmaking"]["version"])

def reset_sim(world, feeding_cfg, rng_food, worm):
    world.reset_food()
    seed_food(world, feeding_cfg, rng_food)
    worm.reset()


# ============================================================
# output + metrics
# ============================================================

def make_run_dir(experiment_folder: str, simulation_name: str) -> Path:
    base = Path(experiment_folder)
    base.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_name = "".join(
        c if c.isalnum() or c in "-_." else "_" for c in simulation_name.strip()
    )
    run_dir = base / f"{ts}_{safe_name}"
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir


@dataclass
class MetricsRecorder:
    tick: list[int]
    energy: list[int]
    eats: list[int]
    distance: list[int]

    @classmethod
    def empty(cls) -> "MetricsRecorder":
        return cls(tick=[], energy=[], eats=[], distance=[])

    def record(self, worm: Worm):
        self.tick.append(int(worm.ticks))
        self.energy.append(int(worm.energy))
        self.eats.append(int(worm.eats))
        self.distance.append(int(worm.distance))

    def save_csv(self, path: Path):
        lines = ["tick,energy,eats,distance"]
        for t, e, k, d in zip(self.tick, self.energy, self.eats, self.distance):
            lines.append(f"{t},{e},{k},{d}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================
# main
# ============================================================

def main():
    # Initialize pause manager
    pause_mgr = init_pause_manager()
    print("[PAUSE MANAGER] Initialized. Press 'p' to pause, 'c' to cancel, 'n' to step.")
    
    cfg = load_config(CONFIG_PATH)
    
    # Check for brain_init vs config consistency
    has_brain_config = cfg.get("decisionmaking", {}).get("brain", False)
    brain_init_spec = load_brain_init(BRAIN_INIT)
    
    if brain_init_spec is not None and not has_brain_config:
        print(f"[WARNING] BRAIN_INIT='{BRAIN_INIT}' specified but config has brain: false. Ignoring brain_init.")
    
    if brain_init_spec is None and has_brain_config:
        raise ValueError(f"Config specifies brain: true but BRAIN_INIT is 'none'. Please set BRAIN_INIT parameter.")
    
    # Build RNG streams (before creating world/worm)
    rng_food, rng_decision, rng_neuron_noise = build_rng_streams(cfg["world"]["rng_seed"], has_brain_config)

    # build simulation objects
    world = make_world(cfg)
    feeding_cfg = make_feeding_cfg(cfg)
    world.feeding_cfg = feeding_cfg

    worm = make_worm(world, cfg)
    worm.active_sensors = make_sensor_cfg(cfg)
    worm.brain = load_brain_module(make_decision_cfg(cfg))
    if hasattr(worm.brain, "init"):
        if brain_init_spec is not None:
            worm.brain.init(worm, cfg, rng_neuron_noise, brain_init_spec=brain_init_spec)
        else:
            worm.brain.init(worm, cfg, rng_neuron_noise)

    reset_sim(world, feeding_cfg, rng_food, worm)

    # output folder
    run_dir = make_run_dir(EXPERIMENT_FOLDER, SIMULATION_NAME)
    print(f"[run_single_rec] Writing data to: {run_dir}")

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

    # visualization (optional)
    viz_cfg = cfg.get("viz", {})
    viz_enabled = bool(viz_cfg.get("enabled", True))

    renderer: Optional[QtRenderer] = None
    if viz_enabled:
        renderer = QtRenderer(world, worm, fps=int(viz_cfg.get("fps", 10)))

    # Pass renderer to worm so it can draw at the right moment
    worm.renderer = renderer

    # metrics
    rec = MetricsRecorder.empty()
    rec.record(worm)  # initial state (day 0)

    # simulation loop
    try:
        while worm.ticks < MAX_TICKS:
            # CHECKPOINT 1: Before incrementing worm ticks
            pause_mgr.check_pause()

            # ----------------------------------------------------
            # ONE day advancement
            # ----------------------------------------------------
            world.step()            # 1) World dynamics
            worm.step_day(rng_decision)      # 2) Byte lives one day (includes world draw after sense)
            worm.ticks += 1         # 3) Advance simulation time

            rec.record(worm)        # 4) Record metrics

            # Stop experiment if Byte is dead
            if not worm.alive:
                break

            # Frame pacing
            if renderer:
                renderer.wait_frame()
            else: 
                time.sleep(0.001) # avoid busy loop if headless
    except PauseManagerExit:
        print("[EXIT] Simulation stopped by user.")
        pass


    # save outputs
    rec.save_csv(run_dir / "metrics.csv")

    summary = [
        f"simulation_name: {SIMULATION_NAME}",
        f"config: {CONFIG_PATH}",
        f"lifetime_ticks: {worm.ticks}",
        f"foods_eaten: {worm.eats}",
        f"distance: {worm.distance}",
        f"final_energy: {worm.energy}",
        f"alive_at_end: {worm.alive}",
        f"max_ticks_cap: {MAX_TICKS}",
    ]
    (run_dir / f"summary_{SIMULATION_NAME}.txt").write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )

    # keep window open at end
    if renderer:
        renderer.draw()
        print("Simulation complete. Close the window to exit.")
        while renderer.isVisible():
            renderer.app.processEvents()
            time.sleep(0.1)
        renderer.close()
    
    # Clean up pause manager
    cleanup_pause_manager()


if __name__ == "__main__":
    main()
