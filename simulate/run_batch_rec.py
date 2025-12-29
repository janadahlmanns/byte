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
from mvb.feeding import FeedingConfig, seed_food
from mvb.worm import Worm, WormConfig


# ============================================================
# EXPERIMENT DEFINITION
# ============================================================

EXPERIMENT_FOLDER = "data/sensing_vs_random/rawdata/"
SIMULATION_NAME   = "test"

CONFIG_PATH = "configs/non_sensing.yaml"
MAX_TICKS   = 1000
N_RUNS      = 2


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
    module = importlib.import_module(module_name)
    if not hasattr(module, "decide"):
        raise AttributeError(f"{module_name} has no decide()")
    return module

def make_world(cfg_yaml, rng):
    return World(
        WorldConfig(
            grid_width=int(cfg_yaml["world"]["grid_width"]),
            grid_height=int(cfg_yaml["world"]["grid_height"]),
            start_pos=tuple(cfg_yaml["world"]["start_pos"]),
            rng_seed=int(cfg_yaml["world"]["rng_seed"]),
        ),
        rng,
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

def reset_sim(world, feeding_cfg, rng, worm):
    world.reset_food()
    seed_food(world, feeding_cfg, rng)
    worm.reset()


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
    rows: list[tuple[int, int, int, int]]

    @classmethod
    def empty(cls):
        return cls(rows=[])

    def record(self, worm: Worm):
        self.rows.append(
            (worm.ticks, worm.energy, worm.eats, worm.distance)
        )

    def save_csv(self, path: Path):
        lines = ["tick,energy,eats,distance"]
        lines += [f"{t},{e},{k},{d}" for t, e, k, d in self.rows]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================
# main
# ============================================================



def main():
    cfg = load_config(CONFIG_PATH)
    brain = load_brain_module(make_decision_cfg(cfg))

    run_dir = make_experiment_dir()
    print(f"[batch] writing to {run_dir}")

    # snapshot config
    (run_dir / "config_used.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False),
        encoding="utf-8",
    )

    summary_lines = ["run_id,seed,lifetime_ticks,foods,distance,final_energy"]

    for run_id in range(N_RUNS):
        seed = cfg["world"]["rng_seed"] + run_id
        rng = build_rng(seed)

        world = make_world(cfg, rng)
        feeding_cfg = make_feeding_cfg(cfg)
        world.feeding_cfg = feeding_cfg

        worm = make_worm(world, cfg)
        worm.active_sensors = make_sensor_cfg(cfg)
        worm.brain = brain

        reset_sim(world, feeding_cfg, rng, worm)

        rec = MetricsRecorder.empty()
        rec.record(worm)

        while worm.alive and worm.ticks < MAX_TICKS:
            world.step()
            worm.step(rng)
            rec.record(worm)

        run_file = run_dir / "runs" / f"run_{run_id:04d}.csv"
        rec.save_csv(run_file)

        summary_lines.append(
            f"{run_id},{seed},{worm.ticks},{worm.eats},{worm.distance},{worm.energy}"
        )

        print(f"[run {run_id:02d}] ticks={worm.ticks} eats={worm.eats}")

    summary_name = f"summary_{SIMULATION_NAME}.csv"
    (run_dir / summary_name).write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )

    print("[batch] done.")


if __name__ == "__main__":
    main()
