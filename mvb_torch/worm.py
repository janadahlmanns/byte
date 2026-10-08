"""Batched worm: acting, eating, metabolism and death.

Behaviour reproduced exactly (plan_evotorch.md Step 4)
-----------------------------------------------------
* **F4.1** Eating happens **only on `stay`**. `do_stay` calls `on_eat`; `do_move`
  does not. A worm that moves onto food does not eat it -- it must then decide
  `stay` on that cell. This is the most behaviour-defining rule in the model and the
  intuitive reading (eat on arrival) is wrong.
* **F4.2** Eating **resets** energy to `energy_capacity`, it does not add to it. With
  integer capacity and unit costs, energy is therefore an exact integer in
  `[0, capacity]` for the whole run -- there is no floating point in this layer.
* **F4.3** One-tick action lag: `act` applies the action decided on the *previous*
  world tick. The first tick has no action at all.
* **F4.8** Death is `energy <= 0` after metabolism. `Worm.death_gate()` is dead code
  and is not reproduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from .decision import FALLBACK_ACTIONS
from .world import WorldState, consume_at, has_food_at

# Action codes are the indices of decision.FALLBACK_ACTIONS; -1 means "nothing
# decided yet", which is the `action is None` of the first tick (F4.3).
NO_ACTION = -1

# dy/dx per action code, from `_stable_outputs_to_decision`'s direction table.
_DY = (0, -1, 0, 1, 0)   # stay, north, east, south, west
_DX = (0, 0, 1, 0, -1)


@dataclass
class WormState:
    """Per-agent worm state, all `(P, R)`."""

    y: torch.Tensor        # int64
    x: torch.Tensor        # int64
    energy: torch.Tensor   # int64 -- exact, see F4.2
    eats: torch.Tensor     # int64
    distance: torch.Tensor  # int64
    alive: torch.Tensor    # bool
    ticks: torch.Tensor    # int64
    action: torch.Tensor   # int64, NO_ACTION until the first decision


def init_worm(
    P: int, R: int, start_pos: Tuple[int, int], energy_capacity: int, device
) -> WormState:
    """`Worm.reset()`: start position, full energy, no action yet.

    `Worm.reset` does `self.x, self.y = self.world.start_pos`, i.e. it unpacks
    start_pos as (x, y) -- the REVERSE of the (y, x) convention used everywhere else.
    Reproduced here rather than corrected.
    """
    start_x, start_y = int(start_pos[0]), int(start_pos[1])
    z = lambda v, dt: torch.full((P, R), v, dtype=dt, device=device)  # noqa: E731
    return WormState(
        y=z(start_y, torch.int64),
        x=z(start_x, torch.int64),
        energy=z(int(energy_capacity), torch.int64),
        eats=z(0, torch.int64),
        distance=z(0, torch.int64),
        alive=torch.ones((P, R), dtype=torch.bool, device=device),
        ticks=z(0, torch.int64),
        action=z(NO_ACTION, torch.int64),
    )


def act_batch(
    world: WorldState,
    worm: WormState,
    running: torch.Tensor,
    movement_cost: int,
    energy_capacity: int,
    regrow_enabled: bool,
    regrow_time: int,
) -> None:
    """Apply each agent's stored action, in place. `mvb/acting.py`.

    Move and stay are disjoint, so both are computed and selected with `where`
    rather than branched on.
    """
    P, R, H, W = world.shape        # both world forms (R4); never touches the cells
    act = worm.action
    is_move = running & (act > 0)                       # codes 1..4
    is_stay = running & (act == 0)

    # --- move: toroidal step, energy cost, distance ---------------------
    dy = torch.tensor(_DY, dtype=torch.int64, device=act.device)
    dx = torch.tensor(_DX, dtype=torch.int64, device=act.device)
    safe = act.clamp(min=0)                             # NO_ACTION would index -1
    ny = (worm.y + dy[safe]) % H
    nx = (worm.x + dx[safe]) % W
    worm.y = torch.where(is_move, ny, worm.y)
    worm.x = torch.where(is_move, nx, worm.x)
    worm.energy = torch.where(
        is_move, (worm.energy - movement_cost).clamp(min=0), worm.energy
    )
    worm.distance = torch.where(is_move, worm.distance + 1, worm.distance)

    # --- stay: eat if and only if there is food here (F4.1) -------------
    # Unconditional, not `if eat.any()`: that test would be a host sync every world
    # tick (plan Step 8, item 2), and consume_at is masked by `eat` anyway.
    eat = is_stay & has_food_at(world, worm.y, worm.x)
    consume_at(world, worm.y, worm.x, eat, regrow_enabled, regrow_time)
    # A full reset, not an increment (F4.2).
    worm.energy = torch.where(
        eat, torch.full_like(worm.energy, int(energy_capacity)), worm.energy
    )
    worm.eats = torch.where(eat, worm.eats + 1, worm.eats)


def metabolise(worm: WormState, running: torch.Tensor, metabolic_rate: int) -> None:
    """`energy = max(0, energy - rate)`, then die on `energy <= 0` (F4.8)."""
    worm.energy = torch.where(
        running, (worm.energy - metabolic_rate).clamp(min=0), worm.energy
    )
    worm.alive = worm.alive & ~(running & (worm.energy <= 0))


def action_names(action: torch.Tensor):
    """Debug helper: action codes -> readable names."""
    return [
        [FALLBACK_ACTIONS[int(a)] if int(a) >= 0 else "none" for a in row]
        for row in action
    ]
