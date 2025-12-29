# run single
# run one single simulation of Byte with the specified parameters
# visualization is optional and specified in pamaeter YAML
# no data is recorded

import argparse
import yaml
import numpy as np
import importlib

from mvb.world import World, WorldConfig
from mvb.feeding import FeedingConfig, seed_food
from mvb.worm import Worm, WormConfig
from mvb.render_mpl import MPLRenderer


# -------------------------
# helpers
# -------------------------

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_rng(seed: int):
    return np.random.default_rng(int(seed))

def load_brain_module(version: str):
    """
    Dynamically import a decision-making module.
    Example: version='prio_food' -> mvb.brains.decisionmaking_prio_food
    """
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


# -------------------------
# main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    rng = build_rng(cfg["world"]["rng_seed"])

    # build simulation objects
    world = make_world(cfg, rng)
    feeding_cfg = make_feeding_cfg(cfg)
    world.feeding_cfg = feeding_cfg

    worm = make_worm(world, cfg)
    worm.active_sensors = make_sensor_cfg(cfg)
    worm.brain = load_brain_module(make_decision_cfg(cfg))

    reset_sim(world, feeding_cfg, rng, worm)

    # visualization (optional)
    viz_cfg = cfg.get("viz", {})
    viz_enabled = viz_cfg.get("enabled", True)

    renderer = None
    if viz_enabled:
        renderer = MPLRenderer(
            world, worm, fps=int(viz_cfg.get("fps", 10))
        )

    paused = False
    running = True
    reset_requested = False

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
        if renderer:
            check_controls()

        if reset_requested:
            reset_sim(world, feeding_cfg, rng, worm)
            if renderer:
                renderer.paused = False
                renderer.single_step = False
                renderer.stop_flag = False
            reset_requested = False

        if (not paused) or (renderer and renderer.single_step):
            world.step()
            worm.step(rng)

            if renderer:
                renderer.single_step = False

            if not worm.alive:
                running = False
                if renderer:
                    renderer.running = False

        if renderer:
            renderer.draw()
            renderer.wait_frame()

    # final draw if visualized
    if renderer:
        renderer.draw()
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
