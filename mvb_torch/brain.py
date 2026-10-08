"""Batched brain tick: the tensor equivalent of one iteration of `decide()`'s loop.

Scope
-----
ONE brain tick. Sensory input and noise are arguments, not responsibilities. The
loop that calls this, the propagation/stability phases, candidate selection and the
decision itself are Step 3; the world and worm are Step 4.

Two contraction paths (plan_evotorch.md §5.4)
---------------------------------------------
``contraction="sequential"`` accumulates the input in the scalar simulation's exact
order and is **bit-exact** against it. ``contraction="einsum"`` uses a batched
contraction that is 1.05x-1.30x faster per tick but sums in a kernel-chosen order.

The scalar sim accumulates left to right starting from the tonic drive
(``Neuron.compute_input``), and float addition is not associative, so the grouping
is part of the specification rather than an implementation detail. Every *product*
is exact -- activities are exactly 0.0/1.0 and sensory inputs are exactly 0/1 -- so
summation order is the only thing that can differ, which is precisely why fixing
the order recovers bit-exactness.

Use ``sequential`` on CPU/float64 as the equivalence oracle and ``einsum`` on
GPU/float32 for production. Correctness of the fast path is established by testing
it against the slow one, which is in turn proven against the scalar sim.

Semantics reproduced deliberately (see plan_evotorch.md Step 2, S1-S7)
---------------------------------------------------------------------
* **S1** Synchronous update: everything reads state ``t`` and writes state ``t+1``.
  Plasticity is driven by the activities at the START of the tick.
* **S2** The returned output snapshot is the **pre-tick** activity. In the scalar
  sim ``_get_output_state()`` is called after ``update()`` but before ``commit()``,
  so it reports the previous tick's activity. Getting this wrong shifts every
  decision by one tick.
* **S3** Accumulation order: tonic -> sensory (YAML key order) -> neuron-to-neuron
  (source ascending) -> noise last.
* **S4** Plasticity groups as ``abs_w + (((eta * abs_w) * (1 - abs_w)) * modsum)``.
* **S5** Output neurons are the hardcoded slice ``5:10``, matching
  ``_get_output_state``; it does NOT consult ``output_mapping``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .genome_codec import GenomeBatch

# Matches `_get_output_state`, which hardcodes `range(5, 10)` rather than deriving
# the range from output_mapping (S5). Do not "improve" this independently of the
# scalar sim -- they must agree.
OUTPUT_SLICE = slice(5, 10)

_CONTRACTIONS = ("sequential", "einsum")


@dataclass
class BrainTensorState:
    """Per-run brain state. Everything here has BOTH a P and an R dimension.

    Genome-level tensors (`Wsign`, `Rel`, `Mod`, `Tonic`, `Eta`) live on the
    `GenomeBatch` and never acquire a run dimension -- see the binding rule in
    `mvb_torch/__init__.py`.
    """

    act: torch.Tensor   # (P, R, n)     activities, exactly 0.0 or 1.0
    Wabs: torch.Tensor  # (P, R, n, n)  plastic magnitude -- the only mutable brain state

    @property
    def n_runs(self) -> int:
        return self.act.shape[1]


def init_state(batch: GenomeBatch, n_runs: int) -> BrainTensorState:
    """Fresh brain state for `n_runs` runs of each of the batch's P genomes.

    Activities start at zero, matching `Neuron.__init__` -- which is why the first
    output snapshot of a run is all-zero (S2).
    """
    if n_runs < 1:
        raise ValueError(f"n_runs must be >= 1, got {n_runs}")
    P, n = batch.n_pop, batch.n_neurons
    act = torch.zeros((P, n_runs, n), dtype=batch.dtype, device=batch.device)
    # Wabs0 is (P, 1, n, n); expand is a view, so clone to get writable per-run state.
    Wabs = batch.Wabs0.expand(P, n_runs, n, n).clone()
    return BrainTensorState(act=act, Wabs=Wabs)


def _accumulate_input(
    batch: GenomeBatch,
    state: BrainTensorState,
    W: torch.Tensor,
    sens: Optional[torch.Tensor],
    noise: Optional[torch.Tensor],
    contraction: str,
) -> torch.Tensor:
    """Net input per neuron, in the scalar sim's accumulation order (S3)."""
    P, R, n = state.act.shape

    # Tonic seeds the accumulator -- NOT added at the end. This alone accounts for
    # 92/1000 of the einsum/scalar mismatches measured during planning, because
    # `einsum(...) + Tonic` groups as `(sum of terms) + tonic` instead of
    # `((tonic + t1) + t2) + ...`.
    net = batch.Tonic.expand(P, R, n).clone()

    # Sensory next, in YAML key order, because init_brain wires sensory connections
    # before neuron-to-neuron ones and `Neuron.incoming` is traversed in order.
    # `(w * activity) * r` matches Connection.propagate; weight and reliability are
    # deliberately not pre-multiplied (see genome_codec).
    if sens is not None:
        for s in range(len(batch.sensor_keys)):
            net = net + (batch.S_w[s] * sens[..., s : s + 1]) * batch.S_r[s]

    # Then neuron-to-neuron, source ascending.
    if contraction == "sequential":
        for i in range(n):
            # addcmul_ is in-place to avoid reallocating the accumulator; verified
            # bit-identical to `net = net + act_i * W_i`.
            net.addcmul_(state.act[..., i : i + 1], W[:, :, i, :])
    else:
        net = net + torch.einsum("pri,prij->prj", state.act, W)

    # Noise last: `compute_input` returns `total + noise`.
    if noise is not None:
        net = net + noise
    return net


def _modulation_sum(
    batch: GenomeBatch, state: BrainTensorState, contraction: str
) -> torch.Tensor:
    """tanh of the modulator-weighted activity sum, per edge -> (P, R, n, n).

    Bit-exactness caveat: the scalar sim accumulates in `modulation_spec` LIST order
    while any dense form must sum over ascending modulator index. With <= 2
    modulators on an edge the orders agree exactly (float addition is commutative,
    just not associative), which covers the `random`, `lookup_hard` and
    `lookup_soft` generators. Mutated genomes can exceed that on a few edges; see
    plan_evotorch.md Step 2, "Where bit-exactness stops".
    """
    if contraction == "sequential":
        P, R, n = state.act.shape
        total = torch.zeros((P, R, n, n), dtype=batch.dtype, device=batch.device)
        for k in range(n):
            # act[:, :, k] -> (P, R, 1, 1);  Mod[:, k] -> (P, 1, n, n)
            total.addcmul_(
                state.act[..., k].unsqueeze(-1).unsqueeze(-1),
                batch.Mod[:, k].unsqueeze(1),
            )
    else:
        total = torch.einsum("pkij,prk->prij", batch.Mod, state.act)
    return torch.tanh(total)


def brain_tick(
    batch: GenomeBatch,
    state: BrainTensorState,
    sens: Optional[torch.Tensor] = None,
    noise: Optional[torch.Tensor] = None,
    *,
    contraction: str = "sequential",
) -> Tuple[BrainTensorState, torch.Tensor]:
    """Advance every (genome, run) brain by one tick.

    Parameters
    ----------
    batch
        Encoded genomes. Supplies `Wsign`, `Rel`, `Mod`, `Tonic`, `Eta`, all at
        `(P, ...)` and broadcast into the run dimension.
    state
        Current activities and plastic magnitudes.
    sens
        `(P, R, n_sensors)` sensory activations, exactly 0.0/1.0. Constant across
        the brain ticks of one `decide()` call (S6). None means no sensory drive.
    noise
        `(P, R, n)` pre-scaled additive noise, i.e. already `sd * z`. None means
        noiseless, matching `draw_noise(None)`.
    contraction
        "sequential" (bit-exact, reference) or "einsum" (fast, production).

    Returns
    -------
    (next_state, output_snapshot)
        `output_snapshot` is `(P, R, 5)` and holds the **pre-tick** activity of
        neurons 5-9, reproducing where `_get_output_state()` is called relative to
        `commit()` (S2).
    """
    if contraction not in _CONTRACTIONS:
        raise ValueError(
            f"contraction must be one of {_CONTRACTIONS}, got {contraction!r}"
        )
    P, R, n = state.act.shape
    if (P, n) != (batch.n_pop, batch.n_neurons):
        raise ValueError(
            f"state shape (P={P}, n={n}) does not match batch "
            f"(P={batch.n_pop}, n={batch.n_neurons})"
        )
    if sens is not None and sens.shape != (P, R, len(batch.sensor_keys)):
        raise ValueError(
            f"sens has shape {tuple(sens.shape)}, expected "
            f"{(P, R, len(batch.sensor_keys))}"
        )
    if noise is not None and noise.shape != (P, R, n):
        raise ValueError(
            f"noise has shape {tuple(noise.shape)}, expected {(P, R, n)}"
        )

    # Snapshot BEFORE anything is committed: this is the value the scalar sim
    # records for this tick (S2).
    output_snapshot = state.act[..., OUTPUT_SLICE].clone()

    # Signed, reliability-scaled weights. (Wabs * Wsign) reproduces the genome's
    # signed weight exactly (|w| * +/-1), then * Rel matches Connection.propagate.
    W = (state.Wabs * batch.Wsign) * batch.Rel

    net = _accumulate_input(batch, state, W, sens, noise, contraction)

    # Hard threshold, compared pre-tanh against atanh(threshold) -- see Step 0b.
    next_act = (net >= batch.threshold_raw).to(batch.dtype)

    # Plasticity rule K on |w|, grouped exactly as the scalar sim groups it (S4).
    # Reads the PRE-tick activities (S1), which is why this uses `state.act`.
    modsum = _modulation_sum(batch, state, contraction)
    next_Wabs = (
        state.Wabs + ((batch.Eta * state.Wabs) * (1.0 - state.Wabs)) * modsum
    ).clamp(min=0.0)

    return BrainTensorState(act=next_act, Wabs=next_Wabs), output_snapshot
