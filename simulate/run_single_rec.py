# ------------------------------------------------------------
# run one single simulation of Byte with the specified parameters
# visualization is optional and specified in pamaeter YAML
# data are recorded into specified folder
# ------------------------------------------------------------

import importlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml
import numpy as np

from mvb.world import World, WorldConfig
from mvb.feeding import FeedingConfig, seed_food
from mvb.worm import Worm, WormConfig
from mvb.world_renderer_qt import QtRenderer


# ============================================================
# EXPERIMENT DEFINITION
# ============================================================

EXPERIMENT_FOLDER = "data/sensing_vs_random/rawdata/"
SIMULATION_NAME   = "sensing_neurons"

CONFIG_PATH = "configs/sensing_neurons.yaml"
MAX_TICKS   = 1000


# ============================================================
# helpers
# ============================================================

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_rng(seed: int):
    return np.random.default_rng(int(seed))

def load_brain_module(version: str):
    module_name = f"mvb.brains.decisionmaking_{version}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise ImportError(f"Could not find brain module '{module_name}'.")
    if not hasattr(module, "decide"):
        raise AttributeError(f"Brain module '{module_name}' has no 'decide' function.")
    return module

def make_world(cfg_yaml, rng):
    wcfg = WorldConfig(
        grid_width=int(cfg_yaml["world"]["grid_width"]),
        grid_height=int(cfg_yaml["world"]["grid_height"]),
        start_pos=tuple(cfg_yaml["world"]["start_pos"]),
        rng_seed=int(cfg_yaml["world"]["rng_seed"]),
    )
    return World(wcfg, rng)

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

def reset_sim(world, feeding_cfg, rng, worm):
    world.reset_food()
    seed_food(world, feeding_cfg, rng)
    worm.reset()


# ============================================================
# output + metrics
# ============================================================

def make_run_dir(experiment_folder: str, simulation_name: str) -> Path:
    """
    Creates:
      <experiment_folder>/<YYYY-MM-DD_HH-MM-SS>_<simulation_name>/
    """
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
    # load config
    cfg = load_config(CONFIG_PATH)
    rng = build_rng(cfg["world"]["rng_seed"])

    # build simulation objects
    world = make_world(cfg, rng)
    feeding_cfg = make_feeding_cfg(cfg)
    world.feeding_cfg = feeding_cfg

    worm = make_worm(world, cfg)
    worm.active_sensors = make_sensor_cfg(cfg)
    worm.brain = load_brain_module(make_decision_cfg(cfg))
    if hasattr(worm.brain, "init"):
        worm.brain.init(worm, rng, cfg)

    reset_sim(world, feeding_cfg, rng, worm)

    # output folder
    run_dir = make_run_dir(EXPERIMENT_FOLDER, SIMULATION_NAME)
    print(f"[run_single_rec] Writing data to: {run_dir}")

    # save config snapshot
    (run_dir / f"config_used_{SIMULATION_NAME}.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False),
        encoding="utf-8",
    )

    # visualization (optional)
    viz_cfg = cfg.get("viz", {})
    viz_enabled = bool(viz_cfg.get("enabled", True))

    renderer = None
    if viz_enabled:
        renderer = QtRenderer(world, worm, fps=int(viz_cfg.get("fps", 10)))

    paused = False
    running = True
    reset_requested = False

    # metrics
    rec = MetricsRecorder.empty()
    rec.record(worm)  # record initial state (tick 0)

    def check_controls():
        nonlocal running, paused, reset_requested
        if renderer is None:
            return

        if renderer.stop_flag:
            if renderer.paused:
                reset_requested = True
                renderer.stop_flag = False
            else:
                running = False
                renderer.running = False

        paused = renderer.paused

    # -------------------------
    # simulation loop
    # -------------------------
    while running:
            # FIRST: Draw current state (so user sees it before we check controls)
            if renderer:
                renderer.draw()
            
            # THEN: Check controls and update flags
            if renderer:
                check_controls()

            # Handle reset request
            if reset_requested:
                reset_sim(world, feeding_cfg, rng, worm)
                rec = MetricsRecorder.empty()
                rec.record(worm)
                if renderer:
                    renderer.paused = False
                    renderer.single_step = False
                    renderer.stop_flag = False
                    renderer.running = True
                reset_requested = False
                continue  # Skip to next iteration to show reset state

            # Run simulation step if not paused (or if single-stepping)
            if (not paused) or (renderer and renderer.single_step):
                world.step()
                worm.step(rng)

                rec.record(worm)

                if renderer:
                    renderer.single_step = False

                # Check end conditions
                if (not worm.alive) or (worm.ticks >= MAX_TICKS):
                    running = False
                    if renderer:
                        renderer.running = False

            # Wait for next frame
            if renderer:
                renderer.wait_frame()

    # save metrics
    rec.save_csv(run_dir / "metrics.csv")

    # summary
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

    # Final draw to show end state
    if renderer:
        renderer.draw()
        # Keep window open until user closes it
        print("Simulation complete. Close the window to exit.")
        import time
        while renderer.isVisible():
            renderer.app.processEvents()
            time.sleep(0.1)
        renderer.close()


if __name__ == "__main__":
    main()
