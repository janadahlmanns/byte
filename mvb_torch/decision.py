"""Batched decision machine: the tensor equivalent of one whole `decide()` call.

`decide()` runs a variable number of brain ticks and `return`s the moment the agent
settles on an action. A batch cannot return, so this runs a fixed loop to
`max_ticks.max()` and carries masks. Two consequences drive the whole design:

* a decided agent's brain must be **frozen**, not merely ignored. `_brain_state` is
  module-level and `decide()` never resets it, so `act` and `Wabs` persist into the
  next world tick; letting a decided agent keep ticking would drift its weights
  (plan_evotorch.md pitfall 1.3, F3.8).
* the propagation and stability phases are per-genome, because `warmup_ticks` and
  `max_ticks` are derived per genome (footnote F2).

Tiebreak order (F3.6) -- the subtle one
---------------------------------------
`_stable_outputs_to_decision` builds its candidate movement list by iterating a
Python ``set``, then indexes it with ``rng.integers(len(...))``. A set iterates in
hash-slot order, which for ints in a size-8 table is ``id & 7`` -- so neurons 8 and
9 (slots 0 and 1) come **before** 6 and 7. ``{6, 8}`` iterates as ``[8, 6]``.

**Sorting the ids would therefore change which direction the worm moves**, on 9 of
the 15 possible movement subsets. This module reproduces slot order instead. The
movement set always has <= 4 members (a stable "stay" neuron returns early), so the
table is always size 8 and the rule is exactly ``sorted(ids, key=lambda i: i & 7)``.

RNG (F3.7)
----------
At most one decision draw per world tick, and a "stay" decision draws nothing.
`PredrawnRandomness.integers()` indexes by world tick rather than by a running
cursor, so a skipped draw cannot desynchronise anything -- which is what makes this
step bit-exact. `decision_uniform` is one value per agent per world tick, consumed as
``int(u * k)`` exactly as the pre-drawn path does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from .brain import BrainTensorState, brain_tick
from .genome_codec import GenomeBatch

# First neuron of the output window; matches brain.OUTPUT_SLICE and
# `_get_output_state`, which hardcodes range(5, 10).
OUTPUT_START = 5

# Action codes, in the order of `_get_random_decision`'s `choices` list. The
# fallback draws `int(u * 5)` and indexes straight into it, so the order is load
# bearing, not cosmetic.
FALLBACK_ACTIONS = ("stay", "north", "east", "south", "west")
ACTION_CODE = {name: i for i, name in enumerate(FALLBACK_ACTIONS)}

# output_mapping action name -> direction name used by `direction_map`.
_MOVE_TO_DIRECTION = {
    "move_north": "north",
    "move_south": "south",
    "move_east": "east",
    "move_west": "west",
}

# Candidacy and stability constants, from _get_candidate_neurons /
# _check_candidate_stability.
PROP_WINDOW = 5      # last 5 propagation snapshots
CANDIDATE_MIN = 3    # active in >= 3 of them
STABILITY_MIN = 3    # need >= 3 stability snapshots before any decision


@dataclass(frozen=True)
class OutputSpec:
    """How the output window maps onto actions, derived from YAML `output_mapping`."""

    n_outputs: int
    stay_slot: Optional[int]          # window index of the "stay" neuron, if any
    move_slots: Tuple[int, ...]       # window indices in CPython set-slot order
    move_actions: Tuple[int, ...]     # action code per entry of `move_slots`


def build_output_spec(brain_cfg: Dict[str, Any], n_outputs: int = 5) -> OutputSpec:
    """Derive the action mapping, reproducing CPython set-slot iteration order.

    `output_mapping` keys may be ints or strings, matching the scalar sim's
    ``output_mapping.get(str(id)) or output_mapping.get(id)`` lookup.
    """
    raw = brain_cfg["output_mapping"]
    mapping = {}
    for key, action in raw.items():
        mapping[int(key)] = action

    ids = [OUTPUT_START + i for i in range(n_outputs)]
    missing = [i for i in ids if i not in mapping]
    if missing:
        raise ValueError(
            f"output_mapping does not cover output neurons {missing}; the decision "
            f"machine would silently never act on them"
        )

    stay_slot = None
    move_ids = []
    for nid in ids:
        action = mapping[nid]
        if action == "stay":
            if stay_slot is not None:
                raise ValueError("more than one output neuron maps to 'stay'")
            stay_slot = nid - OUTPUT_START
        elif action in _MOVE_TO_DIRECTION:
            move_ids.append(nid)
        else:
            raise ValueError(
                f"output_mapping[{nid}] = {action!r} is neither 'stay' nor a "
                f"move_* action; refusing to guess what it means"
            )

    # The set iterated in `_stable_outputs_to_decision` never contains the stay
    # neuron (it returns early), so it holds at most len(move_ids) members. The
    # `id & 7` rule is only valid while CPython keeps the table at size 8, which
    # holds up to 4 members; beyond that the table grows and the order changes.
    if len(move_ids) > 4:
        raise ValueError(
            f"{len(move_ids)} movement neurons: with more than 4, CPython resizes "
            f"the set's hash table and iteration order changes from `id & 7` to "
            f"value order. The slot-order rule here would silently be wrong -- see "
            f"plan_evotorch.md Step 3, F3.6."
        )

    slot_order = sorted(move_ids, key=lambda i: i & 7)
    return OutputSpec(
        n_outputs=n_outputs,
        stay_slot=stay_slot,
        move_slots=tuple(i - OUTPUT_START for i in slot_order),
        move_actions=tuple(
            ACTION_CODE[_MOVE_TO_DIRECTION[mapping[i]]] for i in slot_order
        ),
    )


@dataclass
class Decisions:
    """One decision per (genome, run) for a single world tick."""

    action: torch.Tensor        # (P, R) int64, indexes FALLBACK_ACTIONS
    decided_at: torch.Tensor    # (P, R) int64, brain tick, or -1 for the fallback
    via_fallback: torch.Tensor  # (P, R) bool
    n_brain_ticks: int          # loop length actually run


def decide_batch(
    batch: GenomeBatch,
    state: BrainTensorState,
    spec: OutputSpec,
    sens: Optional[torch.Tensor],
    noise: Optional[torch.Tensor],
    decision_uniform: torch.Tensor,
    *,
    contraction: str = "sequential",
    n_brain_ticks: Optional[int] = None,
) -> Tuple[BrainTensorState, Decisions]:
    """Run one full `decide()` for every (genome, run) in the batch.

    Parameters
    ----------
    sens
        `(P, R, n_sensors)`, latched once for the whole call (Step 2, S6).
    noise
        `(K, P, R, n)` pre-scaled noise, K >= max_ticks.max(). None means noiseless.
    decision_uniform
        `(P, R)` in [0, 1), one value per agent for this world tick (F3.7).
    n_brain_ticks
        Loop length K. Must be >= `max_ticks.max()`; extra ticks are exact no-ops for
        every agent, because each one runs only while `t < its own max_ticks`. Pass it
        when calling every world tick: computing it here costs a host sync (plan Step 8,
        item 2). None computes it from the batch.

    The loop has no data-dependent branches, so it never waits on the device: an agent
    that is decided, or past its own max_ticks, is masked out of every update instead.
    """
    P, R, n = state.act.shape
    dev, dt = batch.device, batch.dtype
    warmup = batch.warmup_ticks.reshape(P, 1)           # (P,1), broadcast over runs
    max_ticks = batch.max_ticks.reshape(P, 1)
    K = int(batch.max_ticks.max().item()) if n_brain_ticks is None else int(n_brain_ticks)

    if noise is not None and noise.shape[0] < K:
        raise ValueError(
            f"noise provides {noise.shape[0]} brain ticks but max_ticks.max() is {K}"
        )
    if decision_uniform.shape != (P, R):
        raise ValueError(
            f"decision_uniform has shape {tuple(decision_uniform.shape)}, "
            f"expected {(P, R)}"
        )

    n_out = spec.n_outputs
    ilong = torch.int64
    # Last PROP_WINDOW propagation snapshots, as a circular buffer. Unwritten slots
    # stay 0.0, which is exactly "inactive", so summing all of them equals counting
    # only the valid ones -- no separate validity mask is needed for the count.
    prop_win = torch.zeros((P, R, PROP_WINDOW, n_out), dtype=dt, device=dev)
    # `_get_candidate_neurons` returns empty when fewer than 3 snapshots exist.
    prop_filled = torch.minimum(
        batch.warmup_ticks, torch.tensor(PROP_WINDOW, device=dev)
    ).reshape(P, 1)

    candidates = torch.zeros((P, R, n_out), dtype=torch.bool, device=dev)
    cand_locked = torch.zeros((P, R), dtype=torch.bool, device=dev)
    stab_count = torch.zeros((P, R, n_out), dtype=dt, device=dev)
    stab_ticks = torch.zeros((P, R), dtype=dt, device=dev)

    decided = torch.zeros((P, R), dtype=torch.bool, device=dev)
    action = torch.full((P, R), -1, dtype=ilong, device=dev)
    decided_at = torch.full((P, R), -1, dtype=ilong, device=dev)

    move_slots = torch.tensor(spec.move_slots, dtype=ilong, device=dev)
    move_actions = torch.tensor(spec.move_actions, dtype=ilong, device=dev)

    for t in range(K):
        # An agent runs only while undecided and within ITS OWN max_ticks.
        active = (~decided) & (t < max_ticks)

        new_state, snap = brain_tick(
            batch,
            state,
            sens,
            None if noise is None else noise[t],
            contraction=contraction,
        )

        in_prop = t < warmup                      # (P,1) -> broadcasts to (P,R)

        # Candidates are frozen at the first stability tick, from the propagation
        # window as it stands BEFORE this tick's snapshot is filed (F3.1). An empty
        # result is kept, which locks the agent into the fallback -- the common
        # case, not an edge case.
        #
        # `cand_locked` here is defensive rather than load-bearing: prop_win writes
        # are masked by `active & in_prop`, so an agent's window is frozen the
        # moment it leaves propagation and recomputing would be idempotent. What
        # actually enforces F3.1 is that empty `candidates` can never yield a
        # non-empty `stable`, so such an agent never decides. The guard is kept so
        # that stays true if the window ever becomes mutable post-warmup.
        at_boundary = active & (~in_prop) & (~cand_locked)
        counts = prop_win.sum(dim=2)                              # (P,R,n_out)
        fresh = (counts >= CANDIDATE_MIN) & (prop_filled >= CANDIDATE_MIN).unsqueeze(-1)
        candidates = torch.where(at_boundary.unsqueeze(-1), fresh, candidates)
        cand_locked = cand_locked | at_boundary

        # Commit the brain, but only for agents still running (F3.8).
        state = BrainTensorState(
            act=torch.where(active.unsqueeze(-1), new_state.act, state.act),
            Wabs=torch.where(
                active.unsqueeze(-1).unsqueeze(-1), new_state.Wabs, state.Wabs
            ),
        )

        # File this tick's snapshot into the phase it belongs to, after the commit,
        # exactly as decide() does.
        write_prop = (active & in_prop).unsqueeze(-1)
        slot = t % PROP_WINDOW
        prop_win[:, :, slot, :] = torch.where(
            write_prop, snap, prop_win[:, :, slot, :]
        )
        in_stab = active & (~in_prop)
        stab_count = stab_count + snap * in_stab.unsqueeze(-1).to(dt)
        stab_ticks = stab_ticks + in_stab.to(dt)

        # Strict majority, with a denominator that grows every tick (F3.3).
        stable = candidates & (stab_count > (stab_ticks * 0.5).unsqueeze(-1))
        eligible = (
            active
            & cand_locked
            & (stab_ticks >= STABILITY_MIN)
            & candidates.any(dim=-1)
        )
        newly = eligible & stable.any(dim=-1)
        act_now = _resolve(stable, spec, move_slots, move_actions, decision_uniform)
        action = torch.where(newly, act_now, action)
        decided_at = torch.where(newly, torch.full_like(decided_at, t), decided_at)
        decided = decided | newly

    # Whatever never decided takes `_get_random_decision`: int(u * 5) over
    # FALLBACK_ACTIONS, whose index order is the action encoding.
    via_fallback = ~decided
    fallback = (decision_uniform * len(FALLBACK_ACTIONS)).to(ilong)
    fallback = fallback.clamp(max=len(FALLBACK_ACTIONS) - 1)
    action = torch.where(via_fallback, fallback, action)

    return state, Decisions(
        action=action,
        decided_at=decided_at,
        via_fallback=via_fallback,
        n_brain_ticks=K,
    )


def _resolve(
    stable: torch.Tensor,
    spec: OutputSpec,
    move_slots: torch.Tensor,
    move_actions: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """Stable outputs -> action code, matching `_stable_outputs_to_decision`.

    Stay wins outright, regardless of any stable movement neuron. Otherwise the
    stable movements are laid out in CPython set-slot order and one is picked with
    ``int(u * count)`` (F3.6, F3.7).
    """
    ilong = torch.int64
    # Movements in slot order -> (P, R, n_moves)
    m = stable.index_select(-1, move_slots)
    count = m.sum(dim=-1)
    # int(u * k), as PredrawnRandomness.integers does. u < 1 already implies
    # pick <= count-1; the clamp only guards u rounding to exactly 1.0.
    pick = (u * count.to(u.dtype)).to(ilong)
    pick = pick.clamp(min=0).minimum((count - 1).clamp(min=0))

    # Position of each present movement within the ordered list of present ones.
    pos = m.to(ilong).cumsum(dim=-1) - 1
    chosen = m & (pos == pick.unsqueeze(-1))
    move_action = (chosen.to(ilong) * move_actions).sum(dim=-1)

    out = move_action
    if spec.stay_slot is not None:
        stay = stable[..., spec.stay_slot]
        out = torch.where(stay, torch.full_like(out, ACTION_CODE["stay"]), out)
    return out
