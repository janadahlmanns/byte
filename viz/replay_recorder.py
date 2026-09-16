"""Frame recorders for replay export.

These record simulation state instead of painting it. They deliberately expose
the same interface the Qt renderers do, so they can be attached to an ordinary
simulation run without any change to the simulation core:

    worm.renderer = WorldFrameRecorder(world, worm)

`mvb.simulation_API.simulate_run` then drives them exactly as it drives a
renderer, which means a recorded replay cannot drift from the simulation it is
meant to reproduce.

Nothing here mutates world, worm or rng state.
"""

import numpy as np


# ============================================================
# World recording
# ============================================================

class WorldFrameRecorder:
    """Captures one frame per simulation tick.

    Implements the renderer interface (`draw`, `wait_frame`) so it can be
    assigned to `worm.renderer`.

    **Where the frame is sampled.** Capture happens in `wait_frame()`, not in
    `draw()`. In `simulate_run` the order per tick is::

        world.step()
        worm.step_day(...)   ->  calls renderer.draw() mid-tick
        worm.ticks += 1
        rec.record(worm)     ->  MetricsRecorder samples here (HDF5 per-tick)
        renderer.wait_frame()

    `draw()` is called inside `step_day` *before* metabolism is applied and
    before the tick counter advances, so sampling there would record energy one
    step out of date. `wait_frame()` fires at the same settled point as
    `MetricsRecorder`, which is what makes a tick-for-tick comparison against
    the HDF5 per-tick datasets meaningful.

    Args:
        world: World instance (read-only)
        worm: Worm instance (read-only)
    """

    def __init__(self, world, worm):
        self.world = world
        self.worm = worm

        self.static = {
            "grid_width": int(world.width),
            "grid_height": int(world.height),
            "start_pos": [int(world.start_pos[0]), int(world.start_pos[1])],
            "energy_capacity": int(worm.energy_capacity),
            "metabolic_rate": int(worm.metabolic_rate),
            "movement_cost": int(worm.movement_cost),
            "speed": int(worm.speed),
            "sensors": list(getattr(worm, "active_sensors", [])),
        }

        self.initial_food = None   # full grid at tick 0; deltas are relative to it
        self.frames = []
        self._prev_food = None
        self._started = False

    # --- lifecycle ------------------------------------------------

    def start(self):
        """Capture the tick-0 baseline.

        Call after the world has been seeded and the worm reset, but before
        `simulate_run`. The renderer interface has no hook for the initial
        state — `draw()` only fires once the agent has already acted — so the
        starting frame has to be taken explicitly or it would be missing.
        """
        self.initial_food = (self.world.food > 0).astype(np.uint8).copy()
        self._prev_food = self.initial_food.copy()
        self._started = True
        self._append_frame(food_added=[], food_removed=[])

    # --- renderer interface ---------------------------------------

    def draw(self):
        """Part of the renderer interface. Intentionally does nothing.

        See the class docstring: the frame is taken in `wait_frame()` instead,
        because `draw()` fires before metabolism is applied.
        """
        return

    def wait_frame(self):
        """Capture this tick's frame. No pacing — recording runs at full speed."""
        if not self._started:
            # Defensive: start() should have been called, but never lose a run
            # over it. Treat the first frame we see as the baseline.
            self.start()
            return

        current = (self.world.food > 0).astype(np.uint8)
        added = np.flatnonzero((current == 1) & (self._prev_food == 0))
        removed = np.flatnonzero((current == 0) & (self._prev_food == 1))
        self._prev_food = current

        self._append_frame(
            food_added=[int(i) for i in added],
            food_removed=[int(i) for i in removed],
        )

    # --- capture --------------------------------------------------

    def _append_frame(self, food_added, food_removed):
        worm = self.worm
        sense = getattr(worm, "sensory_information", {}) or {}

        self.frames.append({
            "t": int(worm.ticks),
            "y": int(worm.y),
            "x": int(worm.x),
            "energy": int(worm.energy),
            "eats": int(worm.eats),
            "distance": int(worm.distance),
            "alive": bool(worm.alive),
            "sense": {k: int(v) for k, v in sense.items()},
            # The action decided *during* this tick, to be executed on the next
            # one (step_day decides after acting). Movement that already
            # happened is derivable from consecutive y/x instead.
            "next_action": _action_label(worm.action),
            "food_added": food_added,
            "food_removed": food_removed,
        })

    # --- output ---------------------------------------------------

    def to_dict(self):
        """Return the recording as plain JSON-serialisable data."""
        if self.initial_food is None:
            raise RuntimeError("Nothing recorded — was start() called?")
        return {
            "static": dict(self.static),
            "initial_food": [int(i) for i in np.flatnonzero(self.initial_food)],
            "frames": self.frames,
        }

    def summary(self):
        """Short human-readable description, for verification output."""
        if not self.frames:
            return "WorldFrameRecorder: no frames"
        last = self.frames[-1]
        food0 = int(self.initial_food.sum()) if self.initial_food is not None else 0
        return (
            f"frames={len(self.frames)} "
            f"final_tick={last['t']} "
            f"eats={last['eats']} "
            f"distance={last['distance']} "
            f"alive={last['alive']} "
            f"initial_food_cells={food0}"
        )


# ============================================================
# Helpers
# ============================================================

def _action_label(action):
    """Compact label for a worm action tuple.

    ("stay",)                    -> "stay"
    ("move", (y, x), "north")    -> "N"
    None                         -> None
    """
    if not action:
        return None
    verb = action[0]
    if verb == "move" and len(action) > 2:
        return {"north": "N", "south": "S", "east": "E", "west": "W"}.get(action[2], "?")
    return str(verb)
