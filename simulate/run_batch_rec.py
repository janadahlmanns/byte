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

EXPERIMENT_FOLDER = "data/noise_vs_nonoise/rawdata/"
SIMULATION_NAME   = "no_noise"

CONFIG_PATH = "configs/sensing_neurons.yaml"
BRAIN_INIT  = "none"  # Set to brain init name (e.g., "prio_food") or "none" to disable
MAX_TICKS   = 1000
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
    
    # Check for brain_init vs config consistency
    has_brain_config = "brain" in cfg.get("decisionmaking", {})
    brain_init_spec = load_brain_init(BRAIN_INIT)
    
    if brain_init_spec is not None and not has_brain_config:
        print(f"[WARNING] BRAIN_INIT='{BRAIN_INIT}' specified but config has no neuronal decision making. Ignoring brain_init.")
    
    if brain_init_spec is None and has_brain_config:
        raise ValueError(f"Config requires a neuronal brain but BRAIN_INIT is 'none'. Please set BRAIN_INIT parameter.")
    
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

    # Initialize pause manager
    pause_mgr = init_pause_manager()

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

            rec = MetricsRecorder.empty()
            rec.record(worm)

            while worm.alive and worm.ticks < MAX_TICKS:
                # CHECKPOINT: Check for pause/exit
                pause_mgr.check_pause()

                world.step()
                worm.step_day(rng_decision)
                worm.ticks += 1
                rec.record(worm)

            run_file = run_dir / "runs" / f"run_{run_id:04d}.csv"
            rec.save_csv(run_file)

            summary_lines.append(
                f"{run_id},{seed},{worm.ticks},{worm.eats},{worm.distance},{worm.energy}"
            )

            print(f"[run {run_id:02d}] ticks={worm.ticks} eats={worm.eats}")

    except PauseManagerExit:
        print("[EXIT] Batch simulation stopped by user.")
    finally:
        cleanup_pause_manager()

    summary_name = f"summary_{SIMULATION_NAME}.csv"
    (run_dir / summary_name).write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )

    print("[batch] done.")


if __name__ == "__main__":
    main()
