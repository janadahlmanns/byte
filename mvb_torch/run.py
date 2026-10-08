"""Batched world-tick loop: the tensor equivalent of `simulate_run`.

One run per slot. Run-chaining (refilling a finished slot with the next run) is
Step 5; here every agent starts at tick 0 and the loop ends when all are dead or
`max_ticks` is reached.

Tick order (plan_evotorch.md Step 4, F4.3/F4.5)
----------------------------------------------
The scalar loop is::

    while worm.alive and worm.ticks < max_ticks:
        world.step()        # world.ticks += 1, then feeding_tick
        worm.step_day()     # act -> sense -> decide -> metabolise -> death
        worm.ticks += 1

so during iteration `k` the world is at tick `k + 1` while the worm is still at
tick `k`. That matters twice: pre-drawn noise and decision uniforms are indexed by
**`worm.ticks`** (`k`), while a phase switch tests **`world.ticks`** (`k + 1`). An
off-by-one here silently shifts either the noise stream or the phase boundary.

`act` applies the action decided on the *previous* iteration, so the first tick acts
on nothing at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch

from .brain import BrainTensorState, init_state
from .decision import OutputSpec, decide_batch
from .genome_codec import GenomeBatch
from .world import WorldState, feeding_tick_batch, init_world, seed_food_batch, sense_batch
from .worm import WormState, act_batch, init_worm, metabolise


@dataclass
class RunHistory:
    """Per-world-tick trajectory, for equivalence testing. `(T+1, P, R)` each.

    Row 0 is the pre-run state, matching `eval_variant`'s `rec.record(worm)` before
    the loop (F4.9), so a history has `lifetime + 1` rows exactly like the reference
    HDF5 files.
    """

    y: List[torch.Tensor]
    x: List[torch.Tensor]
    energy: List[torch.Tensor]
    eats: List[torch.Tensor]
    distance: List[torch.Tensor]
    alive: List[torch.Tensor]
    action: List[torch.Tensor]
    food_sum: List[torch.Tensor]

    @staticmethod
    def empty() -> "RunHistory":
        return RunHistory([], [], [], [], [], [], [], [])

    def append(self, worm: WormState, world: WorldState) -> None:
        self.y.append(worm.y.clone())
        self.x.append(worm.x.clone())
        self.energy.append(worm.energy.clone())
        self.eats.append(worm.eats.clone())
        self.distance.append(worm.distance.clone())
        self.alive.append(worm.alive.clone())
        self.action.append(worm.action.clone())
        self.food_sum.append(world.food.sum(dim=(-2, -1)).clone())


def simulate_batch(
    batch: GenomeBatch,
    spec: OutputSpec,
    *,
    n_runs: int,
    height: int,
    width: int,
    start_pos: Sequence[int],
    energy_capacity: int,
    metabolic_rate: int,
    movement_cost: int,
    feeding_cfg: Dict[str, Any],
    switch_phases: Sequence[Dict[str, Any]],
    max_ticks: int,
    food_uniform: torch.Tensor,
    noise: Optional[torch.Tensor],
    decision_uniform: torch.Tensor,
    contraction: str = "sequential",
    record: bool = False,
):
    """Run P x R worms to death or `max_ticks`.

    Parameters
    ----------
    food_uniform
        `(E, P, R, H, W)` -- one grid per seeding event: the initial seed plus one
        per `initial: true` phase switch, in order.
    noise
        `(max_ticks, K, P, R, n)` pre-scaled neuron noise, K = brain ticks per world
        tick. None means noiseless.
    decision_uniform
        `(max_ticks, P, R)` in [0, 1), one per agent per world tick (Step 3, F3.7).
    """
    P, R = batch.n_pop, n_runs
    dev = batch.device

    world = init_world(P, R, height, width, dev)
    worm = init_worm(P, R, start_pos, energy_capacity, dev)
    brain = init_state(batch, R)

    current_cfg = feeding_cfg
    pending: List[Dict[str, Any]] = [dict(p) for p in switch_phases]
    event = 0

    # `world.reset_food()` then `seed_food(world, feeding_cfg)` before the loop.
    if current_cfg["feeding_paradigm"].get("initial", False):
        seed_food_batch(world, current_cfg["initial_fraction_per_cell"],
                        food_uniform[event])
        event += 1

    history = RunHistory.empty() if record else None
    if history is not None:
        history.append(worm, world)   # row 0: pre-run state (F4.9)

    for k in range(max_ticks):
        running = worm.alive.clone()
        if not bool(running.any()):
            break
        world_tick = k + 1

        # 1. world.step(): phase switch (which skips regrow, F4.7) or regrow.
        switching = bool(pending) and world_tick == pending[0]["phase_from"]
        needs_seed = switching and pending[0]["feeding_paradigm"].get("initial", False)
        seed_u = food_uniform[event] if needs_seed else None
        current_cfg = feeding_tick_batch(
            world, world_tick, current_cfg, pending, seed_u, running
        )
        if needs_seed:
            event += 1

        regrow_enabled = current_cfg["feeding_paradigm"].get("regrow", False)
        regrow_time = int(current_cfg.get("regrow_time", 0))

        # 2. act, using the action decided last tick (F4.3).
        act_batch(world, worm, running, movement_cost, energy_capacity,
                  regrow_enabled, regrow_time)

        # 3. sense from the NEW position.
        sens = sense_batch(world, worm.y, worm.x, batch.sensor_keys, batch.dtype)

        # 4. decide, storing for next tick. Dead agents' brains must not advance.
        new_brain, dec = decide_batch(
            batch, brain, spec, sens,
            None if noise is None else noise[k],
            decision_uniform[k],
            contraction=contraction,
        )
        brain = BrainTensorState(
            act=torch.where(running.unsqueeze(-1), new_brain.act, brain.act),
            Wabs=torch.where(
                running.unsqueeze(-1).unsqueeze(-1), new_brain.Wabs, brain.Wabs
            ),
        )
        worm.action = torch.where(running, dec.action, worm.action)

        # 5. metabolism, then 6. the death gate.
        metabolise(worm, running, metabolic_rate)
        worm.ticks = worm.ticks + running.to(torch.int64)

        if history is not None:
            history.append(worm, world)

    return world, worm, brain, history
