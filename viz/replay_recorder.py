"""Frame recorders for replay export.

Record simulation state instead of painting it, exposing the same interface as
the Qt renderers so they can be attached to an ordinary run:

    worm.renderer = WorldFrameRecorder(world, worm)

`simulate_run` then drives them as it would a renderer, so a replay cannot
drift from the simulation it reproduces.

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

    Frames are taken in `wait_frame()` rather than `draw()`. Within a tick,
    `simulate_run` calls `draw()` from inside `step_day`, before metabolism is
    applied and before the tick counter advances; `wait_frame()` runs afterwards,
    at the same point `MetricsRecorder.record()` samples. Capturing there keeps
    the series aligned with the per_tick datasets written to HDF5.

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
        self._prev_eats = 0
        self._started = False

    # --- lifecycle ------------------------------------------------

    def start(self):
        """Capture the tick-0 baseline.

        Call after the world is seeded and the worm reset, but before
        `simulate_run`. The renderer interface has no hook for the initial state,
        so it has to be taken explicitly.
        """
        self.initial_food = (self.world.food > 0).astype(np.uint8).copy()
        self._prev_food = self.initial_food.copy()
        self._prev_eats = int(self.worm.eats)
        self._started = True
        self._append_frame(food_added=[], food_removed=[])

    # --- renderer interface ---------------------------------------

    def draw(self):
        """No-op. Frames are taken in `wait_frame()`; see the class docstring."""
        return

    def wait_frame(self):
        """Capture this tick's frame. No pacing; recording runs at full speed."""
        if not self._started:
            # start() was skipped; treat the first frame seen as the baseline.
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

        # Which cell was eaten, as a flat index, or None.
        #
        # This cannot be derived from the food deltas above. Within one tick the
        # order is world.step() (regrow) -> step_day() (eat) -> wait_frame()
        # (snapshot), so with a short regrow_time a cell regrows and is eaten
        # again between two snapshots and the grid looks unchanged. A camping
        # agent can take hundreds of meals without producing a single delta.
        #
        # Only do_stay/do_eat consume food, and both act on the worm's current
        # cell, which is still its position when this runs.
        ate = worm.eats > self._prev_eats
        self._prev_eats = worm.eats
        eaten = int(worm.y) * self.static["grid_width"] + int(worm.x) if ate else None

        self.frames.append({
            "t": int(worm.ticks),
            "y": int(worm.y),
            "x": int(worm.x),
            "energy": int(worm.energy),
            "eats": int(worm.eats),
            "distance": int(worm.distance),
            "alive": bool(worm.alive),
            "sense": {k: int(v) for k, v in sense.items()},
            # Decided during this tick, executed on the next one. Movement that
            # already happened is derivable from consecutive y/x.
            "next_action": _action_label(worm.action),
            "ate": eaten,
            "food_added": food_added,
            "food_removed": food_removed,
        })

    # --- output ---------------------------------------------------

    def to_dict(self):
        """Return the recording as plain JSON-serialisable data."""
        if self.initial_food is None:
            raise RuntimeError("Nothing recorded; start() was never called")
        return {
            "static": dict(self.static),
            "initial_food": [int(i) for i in np.flatnonzero(self.initial_food)],
            "frames": self.frames,
        }

    def summary(self):
        """One-line description of the recording."""
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
