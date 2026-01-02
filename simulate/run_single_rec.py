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
from typing import Optional, List

import yaml
import numpy as np

from mvb.world import World, WorldConfig
from mvb.feeding import FeedingConfig, seed_food
from mvb.worm import Worm, WormConfig
from mvb.world_renderer_qt import QtRenderer


# ============================================================
# EXPERIMENT DEFINITION
# ============================================================

EXPERIMENT_FOLDER = "data/temp/rawdata/"
SIMULATION_NAME   = "sensing_neurons"

CONFIG_PATH = "configs/sensing_neurons.yaml"
MAX_TICKS   = 1000


# ============================================================
# Universal simulation status / gate
# ============================================================

class SimulationClock:
    """
    Single source of truth for RUNNING/PAUSED/STEP/STOP/RESET.

    Renderers may set their own local flags internally, but we treat them as
    *input devices* only: we read & translate their flags into this clock.
    """

    def __init__(self):
        self.paused: bool = False
        self._step_once: bool = False
        self.stopped: bool = False
        self.reset_requested: bool = False

        self._renderers: List[object] = []

    def attach_renderer(self, renderer: object):
        if renderer is not None and renderer not in self._renderers:
            self._renderers.append(renderer)

    def _process_renderer_flags(self):
        """
        Pull flags from any attached renderer(s) and translate to clock requests.
        This is the *only* place where renderer flags are interpreted.
        """
        for r in list(self._renderers):
            # Stop/cancel
            if getattr(r, "stop_flag", False):
                # Reset = stop_flag while paused
                if getattr(r, "paused", False):
                    self.reset_requested = True
                    # consume the renderer stop flag so it doesn't retrigger
                    r.stop_flag = False
                else:
                    self.stopped = True
                    # let renderer show stopped state
                    if hasattr(r, "running"):
                        r.running = False
                    return

            # Pause state is authoritative from UI perspective:
            # if ANY renderer is paused -> paused.
            if getattr(r, "paused", False):
                self.paused = True

            # Single step: consume request and convert to ONE global step token
            if getattr(r, "single_step", False):
                self.paused = True
                self._step_once = True
                r.single_step = False  # consume renderer request

    def pump_ui(self):
        """
        Call often. Keeps Qt responsive and imports user intent into the clock.
        """
        # Let every renderer process Qt events (so key presses get registered)
        for r in self._renderers:
            app = getattr(r, "app", None)
            if app is not None:
                app.processEvents()

        # Then read & translate their flags
        self._process_renderer_flags()

    def request_resume(self):
        """Optional helper if you later add a direct 'resume' action."""
        self.paused = False

    def may_advance_once(self) -> bool:
        """
        Non-blocking permission query:
        - RUNNING -> True
        - PAUSED  -> True only if a step token exists (then consumes it)
        """
        # always pump UI before answering
        self.pump_ui()

        if self.stopped:
            return False

        if not self.paused:
            return True

        if self._step_once:
            self._step_once = False
            return True

        return False


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

    (run_dir / f"config_used_{SIMULATION_NAME}.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False),
        encoding="utf-8",
    )

    # visualization (optional)
    viz_cfg = cfg.get("viz", {})
    viz_enabled = bool(viz_cfg.get("enabled", True))

    renderer: Optional[QtRenderer] = None
    if viz_enabled:
        renderer = QtRenderer(world, worm, fps=int(viz_cfg.get("fps", 10)))

    # --------------------------------------------------------
    # Universal simulation clock
    # --------------------------------------------------------
    clock = SimulationClock()
    if renderer:
        clock.attach_renderer(renderer)

    # Inject the universal gate into the worm/brain
    # (brain currently expects a callable returning True once per allowed step)
    worm.set_simulation_gate(clock.may_advance_once)

    # metrics
    rec = MetricsRecorder.empty()
    rec.record(worm)  # initial state (day 0)

    # simulation loop
    while not clock.stopped:

        # keep UI responsive + pull key intents into clock
        clock.pump_ui()

        # Draw current world state
        if renderer:
            renderer.draw()

        # Handle reset request
        if clock.reset_requested:
            reset_sim(world, feeding_cfg, rng, worm)
            rec = MetricsRecorder.empty()
            rec.record(worm)

            # clear clock + renderer UI state
            clock.reset_requested = False
            clock.paused = False
            clock._step_once = False  # internal token reset
            clock.stopped = False

            if renderer:
                renderer.paused = False
                renderer.single_step = False
                renderer.stop_flag = False
                renderer.running = True

            continue

        # ----------------------------------------------------
        # ONE day advancement (if permitted)
        # ----------------------------------------------------
        if clock.may_advance_once():

            # 1) Advance world by one day
            world.step()

            # 2) Let Byte live through one full day (may internally do beats)
            worm.step_day(rng)

            # 3) Advance simulation time by one day
            worm.ticks += 1

            # 4) Record metrics
            rec.record(worm)

            # 5) End conditions
            if (not worm.alive) or (worm.ticks >= MAX_TICKS):
                clock.stopped = True
                if renderer:
                    renderer.running = False

        # Frame pacing
        if renderer:
            renderer.wait_frame()
        else:
            # avoid busy loop if headless
            time.sleep(0.001)

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


if __name__ == "__main__":
    main()
