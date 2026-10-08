"""Per-run, per-tick and heat-map tracking for the tensor evaluator (plan Step 7).

Produces exactly the arrays `mvb.simulation_API.eval_variant` hands to
`simulate/hdf5_utils.py`: the same datasets, dtypes, field names, row counts and
values. The scalar builds them inside `eval_variant` and `MetricsRecorder.record`; that
code is inline there, so its rules are restated here (plan F7.3 / F7.4) and pinned by
tests/test_io_tensor.py.

How the per-tick record is collected (plan D2 / D3 / D5)
--------------------------------------------------------
Every loop iteration writes one entry per slot into a device buffer of `F` iterations:
the slot's genome and run index, tick, position, cumulative eats, energy, raw food
senses and (for per-tick tracking) the signed weights of the genome's tracked
connections. Every `F` iterations the buffer goes to the host in one copy. Each entry
carries its own genome and run, so a slot that finishes and takes another run --
of any genome, through the shared queue (Step 8) -- needs no special case. On the
host, the entries are grouped by (genome, run) and sorted by tick.

Row 0 (the state right after `Worm.reset`) is never logged: the config and the genome
determine it completely. Movement, food consumed and the heat map are derived on the
host from consecutive rows, which is how `MetricsRecorder.record` derives them too.

Two recorder quirks are reproduced on purpose (plan F7.5), because the tensor file
must match the scalar file and `mvb/` is not to be touched:

* Q1: `movement` compares the (already wrapped) target cell with the previous
  position without wrapping, so a move north from y = 0 is recorded as "S".
* Q2: `manhattan_dist` is measured from (y, x) = (start_pos[0], start_pos[1]), while
  the worm itself starts at (x, y) = (start_pos[0], start_pos[1]).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .world import sense_batch

# ------------------------------------------------------------------
# The scalar's dtypes (mvb/simulation_API.py, mvb/simulation_helper_functions.py)
# ------------------------------------------------------------------
SUMMARY_DTYPE = [('run_id', 'i2'), ('lifetime_ticks', 'i4'), ('foods', 'i4'),
                 ('distance', 'i4'), ('final_energy', 'f4'), ('seed_noise', 'u4'),
                 ('seed_decision', 'u4')]
WIRING_BASE_DTYPE = [('src', 'i2'), ('tgt', 'i2'), ('weight_initial', 'f4'),
                     ('reliability', 'f4')]
MODULATION_DTYPE = [('target_src', 'i2'), ('target_tgt', 'i2'), ('modulator_src', 'i2'),
                    ('modulation_weight', 'f4')]
PER_TICK_BASE_DTYPE = [
    ('tick', 'i4'),
    ('food_sensed_N', 'u1'),
    ('food_sensed_E', 'u1'),
    ('food_sensed_S', 'u1'),
    ('food_sensed_W', 'u1'),
    ('movement', 'S4'),
    ('food_consumed', 'u1'),
    ('energy', 'f4'),
    ('manhattan_dist', 'u2'),
    ('decision_made', 'u1'),
]

# The recorder reads `worm.sensory_information`, which holds every ACTIVE sensor's
# output whether or not `sensory_mapping` feeds it to the brain (plan D4). So these are
# sensed separately from the brain input, in the recorder's N, E, S, W column order.
RAW_SENSE_KEYS = ("food_north", "food_east", "food_south", "food_west")

# Columns of the integer log buffer.
_RUN, _TICK, _Y, _X, _EATS, _ENERGY, _SN, _SE, _SS, _SW, _GEN = range(11)
_N_INT = 11


# ------------------------------------------------------------------
# Genome-level arrays (built once per genome, like eval_variant does)
# ------------------------------------------------------------------

def tracked_connections(genome) -> List[Tuple[int, int]]:
    """`eval_variant`'s `connections_to_track`: every (src, tgt) whose initial weight
    is non-zero, src-major, tgt ascending."""
    cw = genome["connection_weights"]
    return [(src, tgt) for src in range(cw.shape[0]) for tgt in range(cw.shape[1])
            if cw[src, tgt, 0] != 0.0]


def wiring_template(genome, n_runs: int) -> np.ndarray:
    """The `wiring` array with `src, tgt, weight_initial, reliability` filled and every
    `weight_final_run_*` column still 0."""
    cw = genome["connection_weights"]
    conns = tracked_connections(genome)
    dtype = list(WIRING_BASE_DTYPE) + [(f'weight_final_run_{r:04d}', 'f4')
                                       for r in range(n_runs)]
    arr = np.zeros(len(conns), dtype=dtype)
    for i, (src, tgt) in enumerate(conns):
        arr[i]['src'] = src
        arr[i]['tgt'] = tgt
        arr[i]['weight_initial'] = cw[src, tgt, 0]
        arr[i]['reliability'] = cw[src, tgt, 1]
    return arr


def modulation_array(genome) -> np.ndarray:
    rows = [(ts, tt, ms, mw)
            for (ts, tt), mods in genome["modulation_spec"].items()
            for ms, mw in mods]
    return (np.array(rows, dtype=MODULATION_DTYPE) if rows
            else np.array([], dtype=MODULATION_DTYPE))


def genome_eta(genome) -> float:
    return float(genome["eta"])


def genome_tonic(genome) -> np.ndarray:
    return np.array(genome["tonic_activations"], dtype=np.float32)


# ------------------------------------------------------------------
# Device-side log
# ------------------------------------------------------------------

@dataclass(frozen=True)
class TrackingFlags:
    """The EFFECTIVE flags. In `eval_variant`, per-tick and heat map are nested under
    per-run: with per-run off, nothing at all is tracked."""

    per_run: bool
    per_tick: bool
    heat_map: bool

    @staticmethod
    def effective(per_run, per_tick, heat_map) -> "TrackingFlags":
        per_run = bool(per_run)
        return TrackingFlags(per_run, per_run and bool(per_tick), per_run and bool(heat_map))

    @property
    def any(self) -> bool:
        return self.per_run


class Tracker:
    """Collects everything the scalar tracks, during `eval_generation_batch`.

    The evaluator calls `sense` (after the act), `log` (after `ticks += 1`, before the
    refill) and `record_final` (with the same sync-free index as the lifespans). Nothing
    here draws randomness or writes simulation state, so tracking cannot change a
    result (test T3).
    """

    def __init__(self, genomes: Sequence, flags: TrackingFlags, n_runs: int,
                 n_slots: int, active_sensors: Sequence[str], device, dtype,
                 flush_every: int = 64):
        if not flags.any:
            raise ValueError("Tracker needs per-run tracking enabled")
        if int(flush_every) < 1:
            raise ValueError(f"flush_every must be >= 1, got {flush_every}")
        self.flags = flags
        self.P, self.R = len(genomes), int(n_runs)
        self.B = int(n_slots)               # the starting width (Step 8, R3); see resize()
        self.F = int(flush_every)
        self.active_sensors = tuple(active_sensors)
        self.device, self.dtype = device, dtype
        n = int(genomes[0]["connection_weights"].shape[0])

        # Tracked connections per genome, padded to C_max, as flat src * n + tgt indices.
        self.conns = [tracked_connections(g) for g in genomes]
        self.C = max((len(c) for c in self.conns), default=0)
        idx = np.zeros((self.P, self.C), dtype=np.int64)
        for p, cs in enumerate(self.conns):
            for c, (src, tgt) in enumerate(cs):
                idx[p, c] = src * n + tgt
        self._flat_idx = torch.as_tensor(idx, device=device)
        self._n = n

        # Final weights, one row per (genome, run) plus the dummy row P * R.
        self.final_w = torch.zeros((self.P * self.R + 1, self.C), dtype=dtype, device=device)

        # The per-iteration log exists only when per-tick or heat-map tracking needs it.
        self.logging = flags.per_tick or flags.heat_map
        self.log_weights = flags.per_tick
        if self.logging:
            B, F = self.B, self.F
            self._ints = torch.zeros((F, B, _N_INT), dtype=torch.int32, device=device)
            self._valid = torch.zeros((F, B), dtype=torch.bool, device=device)
            self._w = (torch.zeros((F, B, self.C), dtype=dtype, device=device)
                       if self.log_weights else None)
        self._k = 0
        self._chunks: List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = []

    # --- called by the evaluator -----------------------------------------

    def signed_weights(self, slot_batch, brain, slot_genome) -> torch.Tensor:
        """(B, C) `conn.weight` of every tracked connection of each slot's genome:
        sign(w) * |w|. `slot_batch` holds the slots' genome tensors (`slot_view`)."""
        B, n = self.B, self._n
        W = (slot_batch.Wsign * brain.Wabs).reshape(B, n * n)
        return W.gather(1, self._flat_idx[slot_genome.reshape(-1)])

    def sense(self, world, worm) -> Optional[torch.Tensor]:
        """Raw N/E/S/W food senses right after the act, as the recorder sees them."""
        if not self.logging:
            return None
        return sense_batch(world, worm.y, worm.x, RAW_SENSE_KEYS, torch.int32,
                           active_sensors=self.active_sensors)

    def log(self, active, run_idx, slot_genome, worm, raw_sense, slot_batch,
            brain) -> None:
        """One entry per slot. All slot tensors are `(B, 1)` (Step 8 layout)."""
        if not self.logging:
            return
        k = self._k
        row = self._ints[k]
        flat = lambda t: t.reshape(-1).to(torch.int32)  # noqa: E731
        row[:, _RUN] = flat(run_idx)
        row[:, _GEN] = flat(slot_genome)
        row[:, _TICK] = flat(worm.ticks)
        row[:, _Y] = flat(worm.y)
        row[:, _X] = flat(worm.x)
        row[:, _EATS] = flat(worm.eats)
        row[:, _ENERGY] = flat(worm.energy)
        row[:, _SN:_SW + 1] = raw_sense.reshape(self.B, 4)
        self._valid[k] = active.reshape(-1)
        if self.log_weights:
            self._w[k] = self.signed_weights(slot_batch, brain, slot_genome)
        self._k += 1
        if self._k == self.F:
            self.flush()

    def record_final(self, idx: torch.Tensor, slot_batch, brain, slot_genome) -> None:
        """Scatter the final weights of the slots that finished (idx as for lifespans;
        slots that did not finish write the dummy row)."""
        if self.C == 0:
            return
        w = self.signed_weights(slot_batch, brain, slot_genome)
        self.final_w.scatter_(0, idx.view(-1, 1).expand(-1, self.C), w)

    def resize(self, new_B: int) -> None:
        """Follow a compaction of the batch (plan Step 8, R2). Call after `flush()`: the
        buffers are reallocated at the new width, so no logged row may be pending."""
        if self._k != 0:
            raise RuntimeError("Tracker.resize with unflushed rows; flush() first")
        self.B = int(new_B)
        if self.logging:
            dev = self._ints.device
            self._ints = torch.zeros((self.F, self.B, _N_INT), dtype=torch.int32, device=dev)
            self._valid = torch.zeros((self.F, self.B), dtype=torch.bool, device=dev)
            self._w = (torch.zeros((self.F, self.B, self.C), dtype=self.dtype, device=dev)
                       if self.log_weights else None)

    def flush(self) -> None:
        """One device -> host copy of the filled part of the buffer."""
        if not self.logging or self._k == 0:
            return
        k = self._k
        valid = self._valid[:k].cpu().numpy()
        ints = self._ints[:k].cpu().numpy()
        w = self._w[:k].cpu().numpy() if self.log_weights else None
        entries = ints[valid]
        self._chunks.append((
            entries[:, _GEN].astype(np.int64),      # genome of each entry
            entries,
            w[valid] if w is not None else None,
        ))
        self._k = 0

    # --- host-side assembly -----------------------------------------------

    def per_run_rows(self):
        """Yield `(p, r, ints, w)` for every (genome, run) with its logged entries sorted
        by tick. Loud failure if any run's ticks are not exactly 1..lifetime."""
        self.flush()
        if not self._chunks:
            return
        p = np.concatenate([c[0] for c in self._chunks])
        ints = np.concatenate([c[1] for c in self._chunks])
        w = (np.concatenate([c[2] for c in self._chunks]) if self.log_weights else None)
        order = np.lexsort((ints[:, _TICK], ints[:, _RUN], p))
        p, ints = p[order], ints[order]
        if w is not None:
            w = w[order]
        key = p * self.R + ints[:, _RUN]
        starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
        ends = np.r_[starts[1:], len(key)]
        for a, b in zip(starts, ends):
            ticks = ints[a:b, _TICK]
            if not np.array_equal(ticks, np.arange(1, b - a + 1)):
                raise RuntimeError(
                    f"per-tick log of genome {p[a]}, run {ints[a, _RUN]} is not the "
                    f"contiguous ticks 1..{b - a}: the log lost or duplicated entries")
            yield int(p[a]), int(ints[a, _RUN]), ints[a:b], (w[a:b] if w is not None else None)


def build_per_tick(ints: np.ndarray, w: Optional[np.ndarray], genome, conns,
                   start_pos, energy_capacity: int) -> np.ndarray:
    """One run's `per_tick` array, rows 0..lifetime (plan F7.4)."""
    L = ints.shape[0]
    dtype = list(PER_TICK_BASE_DTYPE) + [(f'{s}_{t}', 'f4') for s, t in conns]
    out = np.zeros(L + 1, dtype=dtype)

    # Worm.reset: x, y = start_pos
    y = np.r_[int(start_pos[1]), ints[:, _Y]].astype(np.int64)
    x = np.r_[int(start_pos[0]), ints[:, _X]].astype(np.int64)
    eats = np.r_[0, ints[:, _EATS]].astype(np.int64)

    out['tick'] = np.arange(L + 1)
    out['food_sensed_N'][1:] = ints[:, _SN] > 0
    out['food_sensed_E'][1:] = ints[:, _SE] > 0
    out['food_sensed_S'][1:] = ints[:, _SS] > 0
    out['food_sensed_W'][1:] = ints[:, _SW] > 0

    # Q1: compare the new position with the previous one, without wrapping. For a stay
    # (or no action) the position is unchanged, which gives "stay" as the recorder does.
    py, px, cy, cx = y[:-1], x[:-1], y[1:], x[1:]
    movement = np.full(L + 1, b"stay", dtype='S4')
    movement[1:] = np.select(
        [cy < py, cy > py, cx > px, cx < px],
        [b"N", b"S", b"E", b"W"],
        default=b"stay",
    )
    out['movement'] = movement

    out['food_consumed'] = np.r_[0, np.diff(eats) > 0]
    out['energy'] = np.r_[int(energy_capacity), ints[:, _ENERGY]]
    # Q2: the recorder's start is (y, x) = (start_pos[0], start_pos[1]).
    out['manhattan_dist'] = np.abs(y - int(start_pos[0])) + np.abs(x - int(start_pos[1]))
    out['decision_made'][1:] = 1

    cw = genome["connection_weights"]
    for c, (s, t) in enumerate(conns):
        col = out[f'{s}_{t}']
        col[0] = cw[s, t, 0]
        col[1:] = w[:, c]
    return out


def build_heat_map(ints: np.ndarray, start_pos, height: int, width: int) -> np.ndarray:
    """`staying`: +1 at the worm's cell on every row 0..lifetime (an occupancy count)."""
    hm = np.zeros((height, width), dtype=np.int32)
    y = np.r_[int(start_pos[1]), ints[:, _Y]]
    x = np.r_[int(start_pos[0]), ints[:, _X]]
    np.add.at(hm, (y, x), 1)
    return hm
