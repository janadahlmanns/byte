"""Batched generation evaluation: P genomes x R runs -> a (P, R) lifespan array.

This is the tensor equivalent of `eval_generation` / `eval_variant`. It adds the two
things no earlier step needed (plan_evotorch.md Step 5):

A run queue shared by all genomes (plan Step 8, item 1)
-------------------------------------------------------
There are `B` slots (the width) in one flat batch, and one queue of all `P * R` runs in
genome-major order (`q = g * R + r`). The moment a slot's worm dies (or hits
`max_ticks`) the slot takes the next run in the queue -- of ANY genome -- and is reset.
Each slot therefore carries its genome (`slot_genome`) as well as its run (`run_idx`),
and the genome tensors are gathered per slot (`slot_view`). Because slots finish runs
at different iterations, every slot keeps its own clock: `worm.ticks`, `phase_idx` and
`run_idx` are per slot.

Why one queue: with runs confined to their own genome's slots, the longest-lived genome
kept 83% of the batch idle (Step 8, first measurement). Genome-major order also starts
the elites' runs first, because `run_ea` lists genomes as elites in descending fitness,
then their offspring -- so the runs most likely to be long are not left for the tail.

Slots are laid out as `(B, 1)`: every slot is presented to the brain, decision, world
and worm code as a "genome" with one run, so those modules are unchanged.

The width is a pure scheduling parameter and is chosen automatically (`choose_width`,
plan Step 8, R3): every run at once, unless device memory caps it. It cannot change a
result: pre-drawn bundles and live (Philox) numbers are keyed by (genome, run), and
compaction (R2) only drops finished slots. The tests enforce identical results for
every width.

Seeding exactly as the scalar seeds
-----------------------------------
* `run_seeds` are drawn once per generation and shared by all genomes (common random
  numbers, F5.2). So a generation has only R distinct worlds; they are built once as a
  `(R, E, H, W)` table and gathered by `run_idx`.
* The world's only randomness is food seeding (F5.4): event `e` of run `r` is block
  `e` of `default_rng(run_seeds[r]).random((E_bundle, H, W))`. That is the same draw in
  both randomness modes, so worlds match production exactly even in live mode.
* Noise and decision seeds are per genome, drawn row by row in genome order (F5.3).

Two randomness modes
--------------------
* `PredrawnSource`: noise and decision uniforms from `mvb.predrawn.make_bundle`, built
  per run as it enters a slot. Bit-exact against the scalar with
  `predrawn_randomness` enabled. Test scale only (F5.6).
* `KeyedSource` (live mode): noise and decision uniforms computed on the device by
  `mvb_torch.philox` from each run's own `noise_seed` / `decision_seed` and its tick.
  A run's numbers do not depend on its slot, the width or the schedule, so a single
  run can be replayed from the per-run seeds stored in the HDF5 (plan Step 8, R1).
  Statistically equivalent to the scalar's numpy streams, not bit-exact with them.

Every run starts from a fully reset brain (`Wabs0`, zero activity) -- F5.1. This is the
opposite of evotorch_example/maze_task.py, which never resets between runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from mvb.predrawn import make_bundle

from . import philox
from .brain import BrainTensorState
from .decision import OutputSpec, decide_batch
from .genome_codec import GenomeBatch
from .world import (AVAIL_INF, ClockWorldState, active_key_mask, avail_from_seed,
                    sense_batch)
from .worm import NO_ACTION, WormState, act_batch, metabolise

_SEED_SPAN = 2**32


# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True)
class SimConfig:
    """Everything eval_generation reads from the YAML, extracted the same way."""

    max_ticks: int
    n_runs: int
    height: int
    width: int
    start_pos: Tuple[int, int]
    energy_capacity: int
    metabolic_rate: int
    movement_cost: int
    active_sensors: Tuple[str, ...]
    feeding_cfg: Dict[str, Any]
    switch_phases: Tuple[Dict[str, Any], ...]
    noise_level: float


def sim_config_from_yaml(cfg: Dict[str, Any]) -> SimConfig:
    """Mirror simulate/run_batch.py's extraction: phases sorted by `phase_from`, the
    first is the initial feeding config and the rest are switches."""
    exp, world, worm = cfg["experiment"], cfg["world"], cfg["worm"]
    phases = sorted(cfg["food"], key=lambda p: p["phase_from"])
    return SimConfig(
        max_ticks=int(exp["max_ticks"]),
        n_runs=int(exp["n_runs"]),
        height=int(world["grid_height"]),
        width=int(world["grid_width"]),
        start_pos=(int(world["start_pos"][0]), int(world["start_pos"][1])),
        energy_capacity=int(worm["energy_capacity"]),
        metabolic_rate=int(worm["metabolic_rate"]),
        movement_cost=int(worm["movement_cost"]),
        active_sensors=tuple(worm["sensors"]["active"]),
        feeding_cfg=phases[0],
        switch_phases=tuple(phases[1:]),
        noise_level=float(cfg["brain"]["noise_level"]),
    )


# ============================================================
# Seeds (F5.2, F5.3)
# ============================================================

@dataclass(frozen=True)
class SeedSet:
    run_seeds: np.ndarray       # (R,)   uint32, shared by every genome
    noise_seeds: np.ndarray     # (P, R) uint32
    decision_seeds: np.ndarray  # (P, R) uint32


def make_simulation_rngs(simulation_seed: int):
    """(rng_noise, rng_decision, rng_world), exactly as run_batch / run_ea build them.

    run_ea spawns 4 streams and run_batch 3, but SeedSequence children depend only on
    their index, so streams 0-2 are identical either way.
    """
    streams = np.random.SeedSequence(int(simulation_seed)).spawn(3)
    return tuple(np.random.default_rng(s) for s in streams)


def draw_seeds(rng_noise, rng_decision, rng_world, P: int, R: int) -> SeedSet:
    """Draw one generation's seeds in eval_generation's order.

    `run_seeds` first, once, from the world stream; then each genome's noise and
    decision rows in genome order. The parallel scalar path draws these in the main
    process in the same order, so this is reproducible in both execution modes.
    """
    run_seeds = rng_world.integers(0, _SEED_SPAN, size=R, dtype=np.uint32)
    noise = np.empty((P, R), dtype=np.uint32)
    decision = np.empty((P, R), dtype=np.uint32)
    for p in range(P):
        noise[p] = rng_noise.integers(0, _SEED_SPAN, size=R, dtype=np.uint32)
        decision[p] = rng_decision.integers(0, _SEED_SPAN, size=R, dtype=np.uint32)
    return SeedSet(run_seeds=run_seeds, noise_seeds=noise, decision_seeds=decision)


# ============================================================
# Phase table and world table
# ============================================================

@dataclass(frozen=True)
class PhaseTable:
    """The feeding schedule as flat per-phase arrays, indexed per slot by phase_idx.

    Phase 0 is the initial feeding config; phases 1.. are the switches in order.
    `seed_event[q]` is which seeding event entering phase q consumes (-1 if it does
    not seed). It is static: phases are entered strictly in order (only the NEXT
    pending phase is ever tested, as in `feeding_tick`), so the number of seedings
    before phase q never depends on the trajectory.
    """

    phase_from: Tuple[int, ...]
    seeds_on_entry: Tuple[bool, ...]
    regrow: Tuple[bool, ...]
    regrow_time: Tuple[int, ...]
    seed_event: Tuple[int, ...]
    event_fraction: Tuple[float, ...]   # food fraction of each seeding event
    n_bundle_events: int                # 1 + len(switches): what eval_variant allocates


def build_phase_table(feeding_cfg: Dict[str, Any],
                      switch_phases: Sequence[Dict[str, Any]]) -> PhaseTable:
    phases = [feeding_cfg, *switch_phases]
    seeds, regrow, rtime, events, fracs = [], [], [], [], []
    for q in phases:
        para = q["feeding_paradigm"]
        does_seed = bool(para.get("initial", False))
        seeds.append(does_seed)
        regrow.append(bool(para.get("regrow", False)))
        rtime.append(int(q.get("regrow_time", 0)))
        if does_seed:
            events.append(len(fracs))
            fracs.append(float(q["initial_fraction_per_cell"]))
        else:
            events.append(-1)
    return PhaseTable(
        phase_from=tuple(int(q.get("phase_from", 0)) for q in phases),
        seeds_on_entry=tuple(seeds),
        regrow=tuple(regrow),
        regrow_time=tuple(rtime),
        seed_event=tuple(events),
        event_fraction=tuple(fracs),
        n_bundle_events=1 + len(switch_phases),
    )


def build_world_table(run_seeds: np.ndarray, phases: PhaseTable, height: int,
                      width: int, device) -> torch.Tensor:
    """(R, E, H, W) int8: the seeded food mask of every seeding event of every run.

    Block `e` of one `random((E_bundle, H, W))` call is the same value stream as the
    e-th sequential `random((H, W))` call that `setup_food_initially` makes, and it is
    exactly what `make_bundle` stores as `food_uniform` -- so this table serves the
    pre-drawn and the live mode alike.
    """
    R = len(run_seeds)
    E = max(len(phases.event_fraction), 1)
    table = np.zeros((R, E, height, width), dtype=np.int8)
    for r, seed in enumerate(run_seeds):
        u = np.random.default_rng(int(seed)).random(
            (phases.n_bundle_events, height, width)
        )
        for e, frac in enumerate(phases.event_fraction):
            table[r, e] = (u[e] < frac).astype(np.int8)
    return torch.as_tensor(table, device=device)


# ============================================================
# Randomness sources
# ============================================================

class KeyedSource:
    """Live-mode randomness keyed by (run seed, tick): `mvb_torch.philox` (plan Step 8, R1).

    Holds the generation's per-run seeds on the device and computes each slot's numbers
    from the seeds of the run it currently holds. Nothing is drawn from a stateful
    generator, so nothing depends on how many slots there are or which slot a run is in.
    """

    def __init__(self, seeds: SeedSet, noise_level: float, K: int, n: int, device, dtype,
                 rounds: int):
        as_dev = lambda a: torch.as_tensor(np.asarray(a, dtype=np.int64), device=device)  # noqa: E731
        self.noise_seeds = as_dev(seeds.noise_seeds)          # (P, R)
        self.decision_seeds = as_dev(seeds.decision_seeds)    # (P, R)
        self.level, self.K, self.n = noise_level, K, n
        self.dtype = dtype
        self.rounds = philox.check_rounds(rounds)
        self.R = self.noise_seeds.shape[1]

    def load(self, refill: torch.Tensor, run_idx: torch.Tensor,
             slot_genome: torch.Tensor) -> None:
        pass   # nothing to load: every number is computed from its address

    def compact(self, keep: torch.Tensor) -> None:
        pass   # no per-slot state: a run's numbers follow its seeds, not its slot

    def _seed(self, table: torch.Tensor, slot_genome: torch.Tensor,
              run_idx: torch.Tensor) -> torch.Tensor:
        # Retired slots (run_idx == R) read any valid seed; they are masked anyway.
        return table[slot_genome, run_idx.clamp(0, self.R - 1)]          # (B, 1)

    def noise(self, tick: torch.Tensor, slot_genome: torch.Tensor,
              run_idx: torch.Tensor) -> torch.Tensor:
        seed = self._seed(self.noise_seeds, slot_genome, run_idx)
        z = philox.standard_normals(seed, tick, self.K, self.n, self.dtype,
                                    rounds=self.rounds)
        return z * self.level                                            # (K, B, 1, n)

    def decision(self, tick: torch.Tensor, slot_genome: torch.Tensor,
                 run_idx: torch.Tensor) -> torch.Tensor:
        seed = self._seed(self.decision_seeds, slot_genome, run_idx)
        return philox.decision_uniforms(seed, tick, self.dtype,
                                        rounds=self.rounds)                  # (B, 1)


class PredrawnSource:
    """Per-run pre-drawn bundles, loaded as each run enters its slot.

    Memory is bounded by the slot count `B`, not by `P x R`. Loading is a host loop over
    the refilled slots, which is fine because this mode is test scale only (F5.6).
    Bundles are keyed by (genome, run), so a run gets the same numbers whichever slot
    it lands in -- which is why pre-drawn results do not depend on the schedule.
    """

    def __init__(self, seeds: SeedSet, cfg: SimConfig, phases: PhaseTable,
                 max_brain_ticks: int, K: int, B: int, n: int, device, dtype):
        if max_brain_ticks < K:
            raise ValueError(
                f"max_brain_ticks={max_brain_ticks} is smaller than the longest decision "
                f"({K} brain ticks); the bundle would run out"
            )
        self.seeds, self.cfg, self.phases = seeds, cfg, phases
        self.max_brain_ticks, self.K, self.n = max_brain_ticks, K, n
        T = cfg.max_ticks
        # Standard normals; scaled by noise_level at use, like PredrawnRandomness.
        self.z = torch.zeros((B, 1, T, K, n), dtype=dtype, device=device)
        self.u = torch.zeros((B, 1, T), dtype=dtype, device=device)
        self.level = cfg.noise_level
        self.B = B

    def load(self, refill: torch.Tensor, run_idx: torch.Tensor,
             slot_genome: torch.Tensor) -> None:
        for b, _ in refill.nonzero().tolist():
            g, r = int(slot_genome[b, 0]), int(run_idx[b, 0])
            bnd = make_bundle(
                run_seed=self.seeds.run_seeds[r],
                noise_seed=self.seeds.noise_seeds[g, r],
                decision_seed=self.seeds.decision_seeds[g, r],
                max_ticks=self.cfg.max_ticks,
                max_brain_ticks=self.max_brain_ticks,
                n_neurons=self.n,
                grid_shape=(self.cfg.height, self.cfg.width),
                n_seed_events=self.phases.n_bundle_events,
            )
            self.z[b, 0] = torch.as_tensor(bnd.neuron_noise[:, : self.K, :],
                                           dtype=self.z.dtype)
            self.u[b, 0] = torch.as_tensor(bnd.decision_uniform, dtype=self.u.dtype)

    def compact(self, keep: torch.Tensor) -> None:
        """Keep only the slots in `keep` (plan Step 8, R2). Bundles move with their run."""
        self.z = self.z.index_select(0, keep)
        self.u = self.u.index_select(0, keep)
        self.B = int(keep.shape[0])

    def _gather(self, buf: torch.Tensor, tick: torch.Tensor) -> torch.Tensor:
        T = buf.shape[2]
        t = tick.clamp(0, T - 1)                         # finished slots: any valid row
        bi = torch.arange(self.B, device=buf.device).view(-1, 1)
        si = torch.zeros((self.B, 1), dtype=torch.int64, device=buf.device)
        return buf[bi, si, t]

    def noise(self, tick: torch.Tensor, slot_genome=None, run_idx=None) -> torch.Tensor:
        z = self._gather(self.z, tick)                   # (B, 1, K, n)
        return z.permute(2, 0, 1, 3) * self.level        # (K, B, 1, n)

    def decision(self, tick: torch.Tensor, slot_genome=None, run_idx=None) -> torch.Tensor:
        return self._gather(self.u, tick)


# ============================================================
# Slot state, per-slot genomes, and reset
# ============================================================

@dataclass
class SlotState:
    """Everything that belongs to the run currently occupying each slot. All `(B, 1)`
    (or `(B, 1, ...)`): one row per slot, presented to the kernels as one run."""

    world: ClockWorldState      # regrow clock (R4): avail (B, 1, H, W), clock (B, 1)
    worm: WormState
    brain: BrainTensorState
    phase_idx: torch.Tensor     # (B, 1) int64
    run_idx: torch.Tensor       # (B, 1) int64; == R means the slot is retired
    slot_genome: torch.Tensor   # (B, 1) int64; meaningful only while run_idx < R


@dataclass(frozen=True)
class _Context:
    batch: GenomeBatch              # genome level, (P, ...)
    cfg: SimConfig
    world_table: torch.Tensor       # (R, E, H, W) int16 avail: 0 (seeded) / AVAIL_INF
    seeds_on_entry: torch.Tensor    # (Q,) bool
    seed_event: torch.Tensor        # (Q,) int64, -1 -> clamped, guarded by seeds_on_entry
    phase_from: torch.Tensor        # (Q,) int64
    regrow: torch.Tensor            # (Q,) bool
    regrow_time: torch.Tensor       # (Q,) int64
    n_phases: int


def slot_view(batch: GenomeBatch, slot_genome: torch.Tensor) -> GenomeBatch:
    """The genome tensors of each slot's current genome, as a batch of B "genomes".

    A real copy per slot (`Mod`: B x n^3, ~42 MB at B = 7,800 in float32), which plan
    5.2's correction allows: the binding rule exists to forbid P x R copies, and this
    is per slot, not per run. Gathering copies values exactly, so every downstream
    product and sum is the same as with the (P, ...) tensors.
    """
    g = slot_genome.reshape(-1)
    return replace(
        batch,
        Wabs0=batch.Wabs0[g],          # (B, 1, n, n)
        Wsign=batch.Wsign[g],
        Rel=batch.Rel[g],
        Mod=batch.Mod[g],              # (B, n, n, n)
        Tonic=batch.Tonic[g],          # (B, 1, n)
        Eta=batch.Eta[g],              # (B, 1, 1, 1)
        warmup_ticks=batch.warmup_ticks[g],
        max_ticks=batch.max_ticks[g],
        n_pop=int(g.shape[0]),
    )


def reset_slots(state: SlotState, mask: torch.Tensor, ctx: _Context, source,
                slot_batch: GenomeBatch) -> None:
    """Return every masked slot to the start of the run named by its run_idx, with the
    brain of its slot_genome (`slot_batch` must already reflect the new genomes).

    This is the reset checklist from the plan, and every item is load-bearing --
    anything missed leaks from the previous run into the next:

      world  avail (from the world table), clock -> 0, phase_idx
      worm   position, energy, eats, distance, ticks, alive, action
      brain  activities -> 0, Wabs -> Wabs0 of the slot's genome (F5.1)

    `action` is the easiest to miss: left set, the new run's first tick would act on
    the previous run's last decision (breaking F4.3).
    """
    cfg = ctx.cfg
    m2 = mask                                   # (B, 1)
    m4 = mask.unsqueeze(-1).unsqueeze(-1)       # (B, 1, 1, 1)
    R = ctx.world_table.shape[0]

    # --- world: initial phase, seeded (or empty) food, no pending regrowth -----
    r_safe = state.run_idx.clamp(0, R - 1)
    ev0 = ctx.seed_event[0].clamp(min=0)
    seeded = ctx.world_table[r_safe, ev0]                           # (B, 1, H, W)
    initial = torch.where(ctx.seeds_on_entry[0], seeded, torch.full_like(seeded, AVAIL_INF))
    state.world.avail.copy_(torch.where(m4, initial, state.world.avail))
    state.world.clock = torch.where(m2, torch.zeros_like(state.world.clock),
                                    state.world.clock)
    state.phase_idx = torch.where(m2, torch.zeros_like(state.phase_idx), state.phase_idx)

    # --- worm: Worm.reset() -- note start_pos unpacks as (x, y) ------------------
    w = state.worm
    sx, sy = cfg.start_pos
    w.x = torch.where(m2, torch.full_like(w.x, sx), w.x)
    w.y = torch.where(m2, torch.full_like(w.y, sy), w.y)
    w.energy = torch.where(m2, torch.full_like(w.energy, cfg.energy_capacity), w.energy)
    w.eats = torch.where(m2, torch.zeros_like(w.eats), w.eats)
    w.distance = torch.where(m2, torch.zeros_like(w.distance), w.distance)
    w.ticks = torch.where(m2, torch.zeros_like(w.ticks), w.ticks)
    w.alive = w.alive | m2
    w.action = torch.where(m2, torch.full_like(w.action, NO_ACTION), w.action)

    # --- brain: a fresh init_brain for every run, of the slot's genome (F5.1) ----
    state.brain = BrainTensorState(
        act=torch.where(m2.unsqueeze(-1), torch.zeros_like(state.brain.act),
                        state.brain.act),
        Wabs=torch.where(m4, slot_batch.Wabs0, state.brain.Wabs),
    )

    source.load(mask, state.run_idx, state.slot_genome)


# ============================================================
# Width: how many runs are in flight at once (plan Step 8, R3)
# ============================================================

# Share of the device's TOTAL memory the batch may plan for. Total, never free: free
# memory changes between runs and would make the width non-deterministic.
MEMORY_FRACTION = 0.5
# Peak / counted bytes per slot, measured on MPS (51x51, n = 11, float32, live), peak
# driver memory sampled every iteration:
#   R3 (two-grid world): 81.5 KB per slot vs 33.3 KB counted -> 2.45 (factor was 3.0);
#   R4 (regrow clock):   54.3 KB per slot all-inclusive at width 78,000 (slope 40.7 KB)
#                        vs 30.8 KB counted -> 1.3-1.8.  2.0 keeps a >= 40% margin on the
#                        slope and covers the all-inclusive figure.
# It covers the temporaries: `where` copies, the per-iteration `slot_view` re-gather,
# Philox intermediates.
PEAK_FACTOR = 2.0
# Fixed device memory that does not scale with the width (allocator pools, world table,
# genome tensors, kernels): measured ~0.4 GB; reserved out of the budget.
BASE_BYTES = 512 * 2**20


def physical_ram() -> int:
    """Total physical RAM in bytes, on Linux, macOS and Windows."""
    import os
    import sys
    if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    if sys.platform == "win32":                 # no os.sysconf on Windows
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return int(status.ullTotalPhys)
    raise RuntimeError(f"no physical-memory query for platform {sys.platform!r}")


def device_memory(device) -> int:
    """Total memory the device offers, in bytes (not the currently free amount)."""
    device = torch.device(device)
    if device.type == "mps":
        return int(torch.mps.recommended_max_memory())
    if device.type == "cuda":
        return int(torch.cuda.get_device_properties(device).total_memory)
    if device.type == "cpu":
        return physical_ram()
    raise ValueError(f"no memory query for device type {device.type!r}")


def configure_device(device) -> None:
    """Pin numeric settings that could otherwise differ between machines (CUDA).

    * TF32 off: on Ampere and newer NVIDIA GPUs, float32 matrix products may run at
      reduced (TF32) precision depending on global flags. The brain's einsums are
      matrix products, so a flag set elsewhere would silently change results.
    * `CUBLAS_WORKSPACE_CONFIG=:4096:8`: cuBLAS's documented setting for reproducible
      results. It only takes effect if set before cuBLAS starts, so this must run before
      the first CUDA matrix product -- the evaluator calls it at setup.
    Idempotent; does nothing on CPU and MPS.
    """
    import os
    device = torch.device(device)
    if device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")


def release_device_cache(device) -> None:
    """Return PyTorch's cached-but-unused device memory to the system.

    Freed tensors stay in PyTorch's cache for reuse. Compaction (R2) creates tensors at
    ~20-30 different widths per generation, different every generation, so blocks of
    ever more sizes piled up: measured 4.87 -> 5.42 -> 6.54 GB over three production
    generations with 0.00 GB of live tensors (the user saw ~10 GB by generation 3).
    Released once per generation, it stays flat at 4.87 GB, at no measurable cost.
    """
    device = torch.device(device)
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def estimate_slot_bytes(*, height: int, width_cells: int, n: int, K: int,
                        dtype: torch.dtype, mode: str, max_ticks: int,
                        tracked_connections: Optional[int] = None,
                        flush_every: int = 64) -> int:
    """Peak bytes one slot needs, from the config alone (deterministic).

    The slot's state and per-iteration buffers are counted exactly from their shapes;
    PEAK_FACTOR (measured) covers the temporaries. Pre-drawn bundles and tracker
    buffers are added when used -- they are large and do not scale with the factor.
    """
    ds = torch.tensor([], dtype=dtype).element_size()
    cells = height * width_cells
    state = (
        cells * 2                                # avail int16 (regrow clock, R4)
        + (n ** 3 + 3 * n * n + n + 1) * ds      # per-slot genome copy (slot_view)
        + (n * n + n) * ds                       # brain: Wabs, act
        + 16 * 8                                 # worm, phase, run, genome indices
    )
    per_iteration = (
        K * n * ds                               # noise (K, B, 1, n)
        + K * 3 * 4 * 8 * 6                      # Philox int64 words, ~6 alive at once
        + 2 * cells                              # the reseed gather (int16)
    )
    total = PEAK_FACTOR * (state + per_iteration)
    if mode == "predrawn":
        total += (max_ticks * K * n + max_ticks) * ds
    if tracked_connections is not None:
        total += flush_every * (11 * 4 + 1 + tracked_connections * ds)
    return int(math.ceil(total))


def choose_width(P: int, R: int, slot_bytes: int, device, *,
                 memory_fraction: float = MEMORY_FRACTION,
                 total_memory: Optional[int] = None) -> Tuple[int, str]:
    """`min(P*R, (MEMORY_FRACTION x total - BASE_BYTES) // slot_bytes)` and a one-line
    reason for the log. `total_memory` overrides the device query (tests)."""
    total = device_memory(device) if total_memory is None else int(total_memory)
    budget = int(memory_fraction * total) - BASE_BYTES
    cap = budget // int(slot_bytes)
    if cap < 1:
        raise RuntimeError(
            f"one slot needs ~{slot_bytes / 2**20:.1f} MB but the budget is "
            f"{budget / 2**30:.2f} GB ({memory_fraction:.0%} of {total / 2**30:.2f} GB)")
    Q = P * R
    est = lambda b: b * slot_bytes / 2**30   # noqa: E731
    if Q <= cap:
        return Q, (f"all runs; est. {est(Q):.1f} GB of a {budget / 2**30:.1f} GB "
                   f"budget")
    return cap, (f"memory cap: all {Q} runs would need ~{est(Q):.1f} GB of a "
                 f"{budget / 2**30:.1f} GB budget; the queue feeds the rest")


# Compact once the queue is empty and at most this fraction of the width is still
# running (plan Step 8, R2). A speed knob, not a result-affecting one: compaction only
# drops finished slots, and results are identical for every value (tests enforce it).
# Measured (user, 2026-10-08, production, width 56,846): 0.75 left utilisation at 89%
# and gen 1 at 78.2 s; 0.9 -> 95% and 73.2 s (= R2's 72.5 s at width 7,800); 0.95 -> no
# better (95%, 73.7 s, more compactions). The rest of the idle time comes from checking
# only every `sync_every` iterations.
COMPACT_BELOW = 0.9


def compact_slots(state: SlotState, keep: torch.Tensor, source, tracker=None) -> None:
    """Keep only the slots listed in `keep` (1-D int64), in order -- every per-slot
    tensor, so no state is left behind at the old width. Results are indexed by
    (genome, run), not by slot, so they are untouched."""
    sel = lambda t: t.index_select(0, keep)   # noqa: E731
    state.world = ClockWorldState(avail=sel(state.world.avail), clock=sel(state.world.clock))
    w = state.worm
    state.worm = WormState(y=sel(w.y), x=sel(w.x), energy=sel(w.energy), eats=sel(w.eats),
                           distance=sel(w.distance), alive=sel(w.alive),
                           ticks=sel(w.ticks), action=sel(w.action))
    state.brain = BrainTensorState(act=sel(state.brain.act), Wabs=sel(state.brain.Wabs))
    state.phase_idx = sel(state.phase_idx)
    state.run_idx = sel(state.run_idx)
    state.slot_genome = sel(state.slot_genome)
    source.compact(keep)
    if tracker is not None:
        tracker.flush()                         # rows logged at the old width first
        tracker.resize(int(keep.shape[0]))


# ============================================================
# The generation loop
# ============================================================

@dataclass
class GenerationResult:
    lifespans: torch.Tensor     # (P, R) int64 -- worm.ticks at the end of each run
    eats: torch.Tensor          # (P, R) int64
    distance: torch.Tensor      # (P, R) int64
    final_energy: torch.Tensor  # (P, R) int64
    iterations: int
    # Reporting only (plan Step 8, items 0 and R2). slot_iterations: slot-iterations
    # spent on a live run; slot_capacity: slot-iterations actually computed (the width,
    # summed over iterations -- it shrinks when the batch is compacted); n_slots: the
    # starting width; final_width: the width at the end; compactions: how many.
    slot_iterations: int = 0
    n_slots: int = 0
    slot_capacity: int = 0
    final_width: int = 0
    compactions: int = 0

    @property
    def utilisation(self) -> float:
        """Live slots per slot computed."""
        return self.slot_iterations / self.slot_capacity if self.slot_capacity else 0.0


def setup_generation(
    batch: GenomeBatch,
    cfg: SimConfig,
    seeds: SeedSet,
    *,
    width: int,
    mode: str,
    max_brain_ticks: Optional[int] = None,
    philox_rounds: Optional[int] = None,
):
    """Build the slot state, the shared context and the randomness source, with slot b
    reset onto queue item b (genome b // R, run b % R; slots past the queue retire).
    Returns `(state, ctx, source)`; `slot_view(ctx.batch, state.slot_genome)` gives the
    slots' genome tensors."""
    if cfg.max_ticks < 1:
        raise ValueError(f"max_ticks must be >= 1, got {cfg.max_ticks}")
    configure_device(batch.device)
    P, R, n = batch.n_pop, cfg.n_runs, batch.n_neurons
    B = int(width)
    if not 1 <= B <= P * R:
        raise ValueError(f"width must be between 1 and P*R = {P * R}, got {width}")
    if seeds.noise_seeds.shape != (P, R) or seeds.run_seeds.shape != (R,):
        raise ValueError(
            f"seed shapes {seeds.run_seeds.shape}/{seeds.noise_seeds.shape} do not "
            f"match P={P}, R={R}"
        )
    dev, dt = batch.device, batch.dtype
    K = int(batch.max_ticks.max().item())
    Q = P * R
    if cfg.max_ticks >= AVAIL_INF:
        raise ValueError(f"max_ticks must be < {AVAIL_INF} (regrow clock, R4), got "
                         f"{cfg.max_ticks}")

    phases = build_phase_table(cfg.feeding_cfg, cfg.switch_phases)
    ctx = _Context(
        batch=batch,
        cfg=cfg,
        world_table=avail_from_seed(
            build_world_table(seeds.run_seeds, phases, cfg.height, cfg.width, dev)),
        seeds_on_entry=torch.tensor(phases.seeds_on_entry, device=dev),
        seed_event=torch.tensor(phases.seed_event, dtype=torch.int64, device=dev),
        phase_from=torch.tensor(phases.phase_from, dtype=torch.int64, device=dev),
        regrow=torch.tensor(phases.regrow, device=dev),
        regrow_time=torch.tensor(phases.regrow_time, dtype=torch.int64, device=dev),
        n_phases=len(phases.phase_from),
    )

    if mode == "predrawn":
        if max_brain_ticks is None:
            raise ValueError("predrawn mode needs max_brain_ticks")
        if philox_rounds is not None:
            raise ValueError("philox_rounds has no effect in predrawn mode; do not pass it")
        source = PredrawnSource(seeds, cfg, phases, int(max_brain_ticks), K, B, n,
                                dev, dt)
    elif mode == "live":
        if philox_rounds is None:
            raise ValueError("live mode needs philox_rounds (7-10; plan Step 8, R1.1)")
        source = KeyedSource(seeds, cfg.noise_level, K, n, dev, dt, philox_rounds)
    else:
        raise ValueError(f"mode must be 'predrawn' or 'live', got {mode!r}")

    i64 = torch.int64
    q0 = torch.arange(B, dtype=i64, device=dev).view(B, 1)
    has_item = q0 < Q
    zeros = lambda: torch.zeros((B, 1), dtype=i64, device=dev)  # noqa: E731
    state = SlotState(
        world=ClockWorldState(
            avail=torch.full((B, 1, cfg.height, cfg.width), AVAIL_INF, dtype=torch.int16,
                             device=dev),
            clock=torch.zeros((B, 1), dtype=torch.int64, device=dev),
        ),
        worm=WormState(y=zeros(), x=zeros(), energy=zeros(), eats=zeros(),
                       distance=zeros(), alive=torch.zeros((B, 1), dtype=torch.bool,
                                                           device=dev),
                       ticks=zeros(), action=zeros()),
        brain=BrainTensorState(
            act=torch.zeros((B, 1, n), dtype=dt, device=dev),
            Wabs=torch.zeros((B, 1, n, n), dtype=dt, device=dev),
        ),
        phase_idx=zeros(),
        run_idx=torch.where(has_item, q0 % R, torch.full_like(q0, R)),
        slot_genome=torch.where(has_item, q0 // R, torch.zeros_like(q0)),
    )
    reset_slots(state, has_item, ctx, source, slot_view(batch, state.slot_genome))
    return state, ctx, source


def eval_generation_batch(
    batch: GenomeBatch,
    spec: OutputSpec,
    cfg: SimConfig,
    seeds: SeedSet,
    *,
    width: int,
    mode: str,
    max_brain_ticks: Optional[int] = None,
    philox_rounds: Optional[int] = None,
    contraction: str = "sequential",
    sync_every: int = 8,
    compact_below: float = COMPACT_BELOW,
    tracker=None,
    _state_hook=None,
) -> GenerationResult:
    """Evaluate every genome on every run; the tensor `eval_generation`.

    Parameters
    ----------
    width
        Slots in flight at the start, 1..P*R, shared by all genomes through one run
        queue. Required: callers take it from `choose_width` (the adapter) or set it
        explicitly (tests). Results are independent of it; it only trades iterations
        against per-iteration cost (plan Step 8, R3).
    mode
        "predrawn" (needs `max_brain_ticks`, which must equal the scalar's
        `predrawn_randomness.max_brain_ticks`, because the bundle's noise layout
        depends on it) or "live" (randomness keyed by each run's seeds, `KeyedSource`;
        needs `philox_rounds`, 7-10, which pre-drawn mode refuses).
    sync_every
        How often to ask the device whether every run has finished (and whether to
        compact). Extra iterations after the last run are exact no-ops, so a coarse
        check is safe.
    compact_below
        R2: once the queue is empty, drop finished slots from the batch when at most
        this fraction of the current width is still running. 0 never compacts, 1
        compacts at every check that finds a finished slot. Speed only.
    tracker
        Optional `mvb_torch.tracking.Tracker` (plan Step 7). It only reads state, so
        results are identical with and without it. None: no tracking work at all.
    """
    if not 0.0 <= float(compact_below) <= 1.0:
        raise ValueError(f"compact_below must be in [0, 1], got {compact_below}")
    state, ctx, source = setup_generation(
        batch, cfg, seeds, width=width, mode=mode,
        max_brain_ticks=max_brain_ticks, philox_rounds=philox_rounds,
    )
    P, R = batch.n_pop, cfg.n_runs
    B, Q = int(width), P * R
    dev, dt = batch.device, batch.dtype
    i64 = torch.int64
    K = int(batch.max_ticks.max().item())   # once per generation, not per world tick
    key_mask = torch.tensor(
        active_key_mask(batch.sensor_keys, cfg.active_sensors), dtype=dt, device=dev
    )
    slot_batch = slot_view(batch, state.slot_genome)

    # Results, with one extra dummy cell so that recording is a sync-free scatter:
    # slots that did not finish this iteration write into index Q = P*R.
    out = {k: torch.full((Q + 1,), -1, dtype=i64, device=dev)
           for k in ("lifespans", "eats", "distance", "final_energy")}
    dummy = torch.full((B, 1), Q, dtype=i64, device=dev)
    # The queue pointer stays on the device: claiming runs needs no host sync.
    next_q = torch.tensor(B, dtype=i64, device=dev)

    # While the queue is not empty every slot is busy, and the runs need at most
    # Q * max_ticks slot-iterations in total; once it is empty, every remaining run
    # ends within max_ticks. So this bounds the loop.
    max_iter = math.ceil(Q * cfg.max_ticks / B) + cfg.max_ticks
    busy = torch.zeros((), dtype=i64, device=dev)   # read once, after the loop
    executed = 0          # iterations that did work; the one that finds nobody running
    capacity = 0          # slot-iterations computed: the width, summed (host-side)
    compactions = 0
    for it in range(max_iter):
        active = state.run_idx < R
        if it % sync_every == 0:
            # One read per check: live slots, and whether the queue is empty.
            n_live, queue_empty = torch.stack(
                (active.sum(), (next_q >= Q).to(i64))).tolist()
            if n_live == 0:
                break
            # R2: once no run is waiting, drop finished slots -- they would otherwise
            # cost as much as live ones until the generation ends.
            if queue_empty and n_live < B and n_live <= compact_below * B:
                keep = active.reshape(-1).nonzero().reshape(-1)
                compact_slots(state, keep, source, tracker)
                B = n_live
                dummy = torch.full((B, 1), Q, dtype=i64, device=dev)
                slot_batch = slot_view(batch, state.slot_genome)
                active = state.run_idx < R
                compactions += 1
        executed += 1
        capacity += B
        busy += active.sum()
        w, world = state.worm, state.world

        # 1. world.step(): during it the world is at worm.ticks + 1 (F4.5). Only the
        #    NEXT pending phase is ever tested, exactly as feeding_tick does.
        world_tick = w.ticks + 1
        has_next = (state.phase_idx + 1) < ctx.n_phases
        nxt = (state.phase_idx + 1).clamp(max=ctx.n_phases - 1)
        switching = active & has_next & (world_tick == ctx.phase_from[nxt])
        state.phase_idx = state.phase_idx + switching.to(i64)
        q = state.phase_idx

        reseed = switching & ctx.seeds_on_entry[q]
        ev = ctx.seed_event[q].clamp(min=0)
        new_avail = ctx.world_table[state.run_idx.clamp(0, R - 1), ev]
        r4 = reseed.unsqueeze(-1).unsqueeze(-1)
        world.avail.copy_(torch.where(r4, new_avail, world.avail))
        # The regrow pass (R4): one clock tick per slot instead of a pass over every cell.
        # A switch returns early -- no regrow pass that tick (F4.7) -- and a phase without
        # regrow runs none: in both cases the clock simply does not advance.
        world.clock = world.clock + (active & ~switching & ctx.regrow[q]).to(i64)

        # 2-6. step_day, with this slot's own phase settings and genome.
        act_batch(world, w, active, cfg.movement_cost, cfg.energy_capacity,
                  ctx.regrow[q], ctx.regrow_time[q])
        sens = sense_batch(world, w.y, w.x, batch.sensor_keys, dt) * key_mask
        raw_sense = tracker.sense(world, w) if tracker is not None else None
        new_brain, dec = decide_batch(
            slot_batch, state.brain, spec, sens,
            source.noise(w.ticks, state.slot_genome, state.run_idx),
            source.decision(w.ticks, state.slot_genome, state.run_idx),
            contraction=contraction, n_brain_ticks=K,
        )
        state.brain = BrainTensorState(
            act=torch.where(active.unsqueeze(-1), new_brain.act, state.brain.act),
            Wabs=torch.where(active.unsqueeze(-1).unsqueeze(-1), new_brain.Wabs,
                             state.brain.Wabs),
        )
        w.action = torch.where(active, dec.action, w.action)
        metabolise(w, active, cfg.metabolic_rate)
        w.ticks = w.ticks + active.to(i64)
        if tracker is not None:
            # After ticks += 1 and before the refill: MetricsRecorder.record's moment.
            tracker.log(active, state.run_idx, state.slot_genome, w, raw_sense,
                        slot_batch, state.brain)

        # The scalar loop's condition: `while worm.alive and worm.ticks < max_ticks`.
        done = active & (~w.alive | (w.ticks >= cfg.max_ticks))
        idx = torch.where(done, state.slot_genome * R + state.run_idx.clamp(0, R - 1),
                          dummy).reshape(-1)
        out["lifespans"].scatter_(0, idx, w.ticks.reshape(-1))
        out["eats"].scatter_(0, idx, w.eats.reshape(-1))
        out["distance"].scatter_(0, idx, w.distance.reshape(-1))
        out["final_energy"].scatter_(0, idx, w.energy.reshape(-1))
        if tracker is not None:
            tracker.record_final(idx, slot_batch, state.brain, state.slot_genome)

        # Refill from the shared queue: the k-th finished slot (in slot order) takes
        # queue item next_q + k. Finished slots with no item left retire (run_idx = R).
        d = done.reshape(-1)
        claim = next_q + torch.cumsum(d.to(i64), 0) - 1
        take = d & (claim < Q)
        g_old, r_old = state.slot_genome.reshape(-1), state.run_idx.reshape(-1)
        state.slot_genome = torch.where(take, claim // R, g_old).view(B, 1)
        state.run_idx = torch.where(
            take, claim % R, torch.where(d, torch.full_like(r_old, R), r_old)).view(B, 1)
        next_q = next_q + d.sum()
        refill = take.view(B, 1)

        slot_batch = slot_view(batch, state.slot_genome)
        if _state_hook is not None:
            _state_hook(it, state, refill)
        reset_slots(state, refill, ctx, source, slot_batch)

    res = {k: v[:Q].view(P, R) for k, v in out.items()}
    # Drop this generation's per-slot state, then hand the cache back (R3 memory fix):
    # otherwise each generation's compaction widths add new cached block sizes.
    del state, source, slot_batch, dummy, out
    release_device_cache(dev)
    if bool((res["lifespans"] < 0).any()):
        missing = int((res["lifespans"] < 0).sum())
        raise RuntimeError(
            f"{missing} runs never finished within {max_iter} iterations; the loop bound "
            f"or the run assignment is wrong"
        )
    return GenerationResult(iterations=executed, slot_iterations=int(busy.item()),
                            n_slots=int(width), slot_capacity=capacity, final_width=B,
                            compactions=compactions, **res)
