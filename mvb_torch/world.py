"""Batched food world: seeding, regrowth, phase switches and sensing.

Shapes follow the §5.1 layout: grids are `(P, R, H, W)` and are flattened to
`(P*R, H*W)` only for the gather/scatter that reads or writes a single cell per
agent. Flattening a contiguous tensor is a view, so it is free.

Behaviour reproduced exactly (plan_evotorch.md Step 4)
-----------------------------------------------------
* **F4.6** `regrow_time` is off by one from its name: `on_eat` writes the timer
  *after* that tick's regrow pass, and the pass decrements before testing `== 1`, so
  food returns `regrow_time - 1` ticks after being eaten. `regrow_time <= 1` means it
  never returns at all (the timer goes 1 -> 0 and `(0 == 1)` is false).
* **F4.7** A phase switch re-seeds the grid and returns early, skipping that tick's
  regrow pass entirely.
* Seeding only happens when the phase has `initial: true`; otherwise food and timers
  survive the switch untouched.

Two representations
-------------------
* `WorldState` -- the faithful two-grid form (`food` + `regrow_timer`), equivalent to
  the scalar sim by construction. Step 4's `run.py` uses it, and it is the reference the
  clock form is proven against.
* `ClockWorldState` -- the regrow clock (plan Step 8, R4), used by the generation loop.
  Each world counts its regrow passes (`clock`); each cell stores the clock value from
  which it has food (`avail`, INF = never until reseeded). Food present <=> avail <=
  clock. A regrow pass is `clock += 1` instead of a pass over every cell. A STORAGE
  change only: tests/test_world_clock.py drives both forms through identical random
  histories and requires `food == (avail <= clock)` for every cell after every tick.

`has_food_at`, `consume_at` and `sense_batch` accept either form.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

# Offsets used by mvb/sensory.py. `perceive` wraps with `% h` / `% w`, so the world
# is a torus and no bounds check is needed.
SENSOR_OFFSETS: Dict[str, Tuple[int, int]] = {
    "on_food": (0, 0),
    "food_north": (-1, 0),
    "food_south": (1, 0),
    "food_west": (0, -1),
    "food_east": (0, 1),
}


# Which input keys each sensor module in mvb/sensory.py produces. A sensor that is not
# active produces nothing, and `InputSource.update` then reads `inputs.get(key, 0.0)`,
# so its keys are fed as 0.0 -- not sensed.
SENSOR_PROVIDES: Dict[str, Tuple[str, ...]] = {
    "current_field": ("on_food",),
    "adjacent_binary": ("food_north", "food_south", "food_west", "food_east"),
}


def active_key_mask(sensor_keys: Sequence[str], active_sensors: Sequence[str]) -> List[bool]:
    """Per sensor key: is it produced by an active sensor?

    `perceive()` only warns and skips on an unknown sensor name. Here that fails
    loudly instead: a misspelt sensor silently zeroing inputs is exactly the kind of
    fallback the repository rules out, and it changes nothing for a valid config.
    """
    unknown = [a for a in active_sensors if a not in SENSOR_PROVIDES]
    if unknown:
        raise ValueError(
            f"unknown active sensor(s) {unknown}; mvb/sensory.py defines "
            f"{sorted(SENSOR_PROVIDES)}"
        )
    provided = {k for a in active_sensors for k in SENSOR_PROVIDES[a]}
    return [k in provided for k in sensor_keys]


@dataclass
class WorldState:
    """Food grids for P x R independent worlds."""

    food: torch.Tensor          # (P, R, H, W) int8, 0 or 1
    regrow_timer: torch.Tensor  # (P, R, H, W) int16, ticks remaining

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        return tuple(self.food.shape)  # type: ignore[return-value]


# "No food until reseeded". Clock values never get near it: max_ticks < AVAIL_INF is
# checked, and avail is saturated here, which is exact (such a cell can never be seen).
AVAIL_INF = 32767


@dataclass
class ClockWorldState:
    """Food grids as regrow-clock timestamps (plan Step 8, R4)."""

    avail: torch.Tensor   # (P, R, H, W) int16: clock value from which the cell has food
    clock: torch.Tensor   # (P, R) int64: regrow passes this world has run

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        return tuple(self.avail.shape)  # type: ignore[return-value]

    def derived_food(self) -> torch.Tensor:
        """(P, R, H, W) int8 food grid, computed -- for tests and inspection only. Not a
        property on purpose: code that reads `.food` on a clock world fails loudly
        instead of silently paying for a full-grid pass every tick."""
        return (self.avail.to(torch.int64) <= self.clock[..., None, None]).to(torch.int8)


def avail_from_seed(seeded: torch.Tensor) -> torch.Tensor:
    """A seeded food mask (0/1) -> avail values: 0 (food now) or AVAIL_INF (none)."""
    return torch.where(seeded > 0, torch.zeros_like(seeded, dtype=torch.int16),
                       torch.full_like(seeded, AVAIL_INF, dtype=torch.int16))


def init_world(P: int, R: int, height: int, width: int, device) -> WorldState:
    return WorldState(
        food=torch.zeros((P, R, height, width), dtype=torch.int8, device=device),
        regrow_timer=torch.zeros((P, R, height, width), dtype=torch.int16, device=device),
    )


def _flat_index(y: torch.Tensor, x: torch.Tensor, width: int) -> torch.Tensor:
    """(P,R) coordinates -> (P*R, 1) index into a flattened H*W grid."""
    return (y * width + x).reshape(-1, 1)


def seed_food_batch(
    world: WorldState,
    fraction: float,
    uniform: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> None:
    """`food = uniform < fraction`, timers zeroed -- `setup_food_initially`.

    `uniform` is `(P, R, H, W)`, one pre-drawn grid per agent, so the tensor side can
    consume exactly the values the scalar side drew from `rng_world_run.random(...)`.
    `mask` is `(P, R)`: agents outside it keep their world untouched (a dead agent's
    world stops evolving, because the scalar loop has already exited).
    """
    new_food = (uniform < fraction).to(torch.int8)
    new_timer = torch.zeros_like(world.regrow_timer)
    if mask is None:
        world.food.copy_(new_food)
        world.regrow_timer.copy_(new_timer)
    else:
        m = mask.unsqueeze(-1).unsqueeze(-1)
        world.food.copy_(torch.where(m, new_food, world.food))
        world.regrow_timer.copy_(torch.where(m, new_timer, world.regrow_timer))


def regrow_batch(world: WorldState, mask: Optional[torch.Tensor] = None) -> None:
    """One `tick_regrow`: decrement, then regrow where the timer has reached 1.

    `timer[timer > 0] -= 1` is exactly `(timer - 1).clamp(min=0)`, with no boolean
    mask needed. Food returns when the *post*-decrement value is 1, i.e. when the
    pre-decrement value was 2 -- which is where F4.6's off-by-one comes from.
    """
    new_timer = (world.regrow_timer.to(torch.int32) - 1).clamp(min=0).to(torch.int16)
    new_food = torch.where(
        new_timer == 1, torch.ones_like(world.food), world.food
    )
    if mask is None:
        world.regrow_timer.copy_(new_timer)
        world.food.copy_(new_food)
    else:
        m = mask.unsqueeze(-1).unsqueeze(-1)
        world.regrow_timer.copy_(torch.where(m, new_timer, world.regrow_timer))
        world.food.copy_(torch.where(m, new_food, world.food))


def feeding_tick_batch(
    world: WorldState,
    world_tick: int,
    current_cfg: Dict[str, Any],
    pending_phases: List[Dict[str, Any]],
    seed_uniform: Optional[torch.Tensor],
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """One `feeding_tick`. Returns the (possibly new) current phase config.

    A phase switch re-seeds and **returns early**, so the regrow pass is skipped for
    that tick (F4.7). `pending_phases` is mutated, mirroring `switch_phases.pop(0)`.
    """
    if pending_phases and world_tick == pending_phases[0]["phase_from"]:
        new_cfg = pending_phases.pop(0)
        if new_cfg["feeding_paradigm"].get("initial", False):
            if seed_uniform is None:
                raise ValueError(
                    f"phase switch at tick {world_tick} needs a seeding uniform but "
                    f"none was supplied"
                )
            seed_food_batch(
                world, new_cfg["initial_fraction_per_cell"], seed_uniform, mask
            )
        return new_cfg

    if current_cfg["feeding_paradigm"].get("regrow", False):
        regrow_batch(world, mask)
    return current_cfg


def sense_batch(
    world: WorldState,
    y: torch.Tensor,
    x: torch.Tensor,
    sensor_keys: Sequence[str],
    dtype: torch.dtype,
    active_sensors: Optional[Sequence[str]] = None,
) -> torch.Tensor:
    """`(P, R, len(sensor_keys))` of exact 0.0/1.0, in the given key order.

    Matches `mvb/sensory.py`: `int(world.has_food(...))` on the cell itself and the
    four toroidal neighbours. Columns follow `sensor_keys` (the YAML
    `sensory_mapping` order) rather than any fixed order, so the codec and the world
    cannot silently disagree about which column is which.

    `active_sensors` (the YAML `worm.sensors.active` list) zeroes the keys of any
    sensor that is not active, as the scalar sim does. None means every sensor is
    active, which is what Step 4's callers assume.
    """
    clock_form = isinstance(world, ClockWorldState)
    grid = world.avail if clock_form else world.food
    P, R, H, W = grid.shape
    flat = grid.reshape(P * R, H * W)
    clock = world.clock.reshape(-1, 1) if clock_form else None
    out = torch.zeros((P, R, len(sensor_keys)), dtype=dtype, device=grid.device)
    live = (
        [True] * len(sensor_keys)
        if active_sensors is None
        else active_key_mask(sensor_keys, active_sensors)
    )
    for i, key in enumerate(sensor_keys):
        if key not in SENSOR_OFFSETS:
            raise ValueError(
                f"unknown sensor key {key!r}; mvb/sensory.py defines "
                f"{sorted(SENSOR_OFFSETS)}"
            )
        if not live[i]:
            continue                       # inactive sensor -> stays 0.0
        dy, dx = SENSOR_OFFSETS[key]
        idx = _flat_index((y + dy) % H, (x + dx) % W, W)
        cell = flat.gather(1, idx)
        present = (cell.to(torch.int64) <= clock) if clock_form else (cell > 0)
        out[..., i] = present.reshape(P, R).to(dtype)
    return out


def has_food_at(world: WorldState, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """(P, R) bool: is there food under each agent right now?"""
    if isinstance(world, ClockWorldState):
        P, R, H, W = world.avail.shape
        cell = world.avail.reshape(P * R, H * W).gather(1, _flat_index(y, x, W))
        return cell.reshape(P, R).to(torch.int64) <= world.clock
    P, R, H, W = world.food.shape
    flat = world.food.reshape(P * R, H * W)
    return flat.gather(1, _flat_index(y, x, W)).reshape(P, R) > 0


def consume_at(
    world: WorldState,
    y: torch.Tensor,
    x: torch.Tensor,
    eat: torch.Tensor,
    regrow_enabled: bool,
    regrow_time: int,
) -> None:
    """`on_eat` for the agents in `eat`: remove the food, maybe start the timer.

    The timer is written only when the current phase has `regrow: true`, matching
    `on_eat`'s inner guard -- not whenever `regrow_time` happens to be set.

    `regrow_enabled` / `regrow_time` are either scalars (one phase for the whole
    batch, Step 4) or `(P, R)` tensors (a phase per slot, Step 5).

    Clock form (R4): the eaten cell's food returns after R - 1 more regrow passes --
    `avail = clock + R - 1` -- exactly when the timer, set to R after this tick's pass,
    would reach 1 (F4.6). R <= 1 or no regrow: never (AVAIL_INF).
    """
    if isinstance(world, ClockWorldState):
        P, R, H, W = world.avail.shape
        idx = _flat_index(y, x, W)
        flat = world.avail.reshape(P * R, H * W)
        cur = flat.gather(1, idx)
        rt = torch.as_tensor(regrow_time, dtype=torch.int64, device=cur.device)
        rt = rt.expand(P, R).reshape(-1, 1)
        en = torch.as_tensor(regrow_enabled, dtype=torch.bool, device=cur.device)
        en = en.expand(P, R).reshape(-1, 1)
        back = (world.clock.reshape(-1, 1) + rt - 1).clamp(max=AVAIL_INF)
        new = torch.where(en & (rt >= 2), back, torch.full_like(back, AVAIL_INF))
        flat.scatter_(1, idx, torch.where(eat.reshape(-1, 1), new.to(cur.dtype), cur))
        return
    P, R, H, W = world.food.shape
    idx = _flat_index(y, x, W)

    flat_food = world.food.reshape(P * R, H * W)
    cur = flat_food.gather(1, idx)
    eat_flat = eat.reshape(-1, 1)
    flat_food.scatter_(1, idx, torch.where(eat_flat, cur - 1, cur))

    if isinstance(regrow_enabled, torch.Tensor):
        # Per-slot phase settings (Step 5): slots in different feeding phases can
        # have different regrow rules on the same tick.
        flat_timer = world.regrow_timer.reshape(P * R, H * W)
        cur_t = flat_timer.gather(1, idx)
        write = (eat & regrow_enabled).reshape(-1, 1)
        rt = torch.as_tensor(regrow_time, device=cur_t.device)
        val = rt.expand(P, R).reshape(-1, 1).to(cur_t.dtype)
        flat_timer.scatter_(1, idx, torch.where(write, val, cur_t))
    elif regrow_enabled:
        flat_timer = world.regrow_timer.reshape(P * R, H * W)
        cur_t = flat_timer.gather(1, idx)
        val = torch.full_like(cur_t, int(regrow_time))
        flat_timer.scatter_(1, idx, torch.where(eat_flat, val, cur_t))
