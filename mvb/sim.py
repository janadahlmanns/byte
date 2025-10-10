import argparse
import sys
import time
import yaml
import numpy as np

from .world import World, WorldConfig
from .feeding import FeedingConfig, apply_feeding
from .worm import Worm, WormConfig
from .render_mpl import MPLRenderer

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_rng(seed: int):
    return np.random.default_rng(int(seed))

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
        feeding_paradigm=str(f["feeding_paradigm"]),
        initial_fraction_per_cell=float(f["initial_fraction_per_cell"]),
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
    apply_feeding(world, feeding_cfg, rng)
    worm.reset()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rng = build_rng(cfg["world"]["rng_seed"])

    world = make_world(cfg, rng)
    feeding_cfg = make_feeding_cfg(cfg)
    worm = make_worm(world, cfg)
    worm.active_sensors = make_sensor_cfg(cfg)
    worm.decision_version = make_decision_cfg(cfg)

    reset_sim(world, feeding_cfg, rng, worm)

    fps = int(cfg["viz"]["fps"])
    renderer = MPLRenderer(world, worm, fps=fps)

    paused = False
    running = True
    reset_requested = False

    # Small helper: interpret renderer.stop_flag as stop vs reset (if paused)
    def check_controls():
        nonlocal running, paused, reset_requested
        # 'r' sets stop_flag; we treat it as reset if paused or always as reset to be simple:
        if renderer.stop_flag:
            if renderer.paused:
                reset_requested = True
                renderer.stop_flag = False
            else:
                # stop
                running = False

        paused = renderer.paused

    while running:
        check_controls()

        if reset_requested:
            reset_sim(world, feeding_cfg, rng, worm)
            renderer.paused = False
            renderer.single_step = False
            renderer.stop_flag = False
            reset_requested = False

        if not paused or renderer.single_step:
            worm.step(rng)
            renderer.single_step = False
            # If worm died, stop the loop
            if not worm.alive:
                running = False

        renderer.draw()
        renderer.wait_frame()

    # Final frame draw to show death state
    renderer.draw()
    # Keep window open until closed
    import matplotlib.pyplot as plt
    plt.show()

if __name__ == "__main__":
    main()
