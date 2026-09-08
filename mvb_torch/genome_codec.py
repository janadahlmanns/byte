"""Encode a list of scalar genomes into the batched tensors the tensor brain uses.

This module is the boundary between the scalar world (``mvb.genome.*`` producing
numpy arrays and Python dicts) and the tensor world.  It is deliberately the only
place that knows the genome's dict layout, so the simulation kernel in later steps
never has to touch a dict.

The encoding must reproduce ``mvb.brains.decisionmaking_plasticity.init_brain``
*exactly*, including its edge cases:

  * an edge exists iff ``connection_weights[src, tgt, 0] != 0.0``;
  * modulation entries on non-existent (zero-weight) edges are silently ignored,
    because ``init_brain`` hits its ``continue`` before it ever consults
    ``modulation_spec``;
  * sensory inputs are wired in ``sensory_mapping`` key order and are not plastic;
  * ``warmup_ticks`` is the number of neurons that have at least one incoming
    connection (sensory counts) or at least one outgoing connection to a neuron.

See ``plan_evotorch.md`` §5.1 for the shape contract and ``mvb_torch/__init__.py``
for the binding rule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch


# ============================================================
# Result structure
# ============================================================

@dataclass
class GenomeBatch:
    """Batched genome tensors for P genomes over n neurons.

    Every field with a leading ``P`` dimension is genome-derived and is *never*
    expanded to the run dimension R -- see the binding rule in the package
    docstring.  The singleton axis in the ``(P, 1, n, n)`` fields is the run axis,
    present so these broadcast against ``(P, R, n, n)`` state without a reshape.
    """

    # --- plastic connection state, initial values -------------------------
    Wabs0: torch.Tensor   # (P, 1, n, n)  |w| at t=0; the only quantity plasticity changes
    Wsign: torch.Tensor   # (P, 1, n, n)  sign(w) in {-1, 0, +1}; constant for life
    Rel: torch.Tensor     # (P, 1, n, n)  reliability; NEVER folded into the weight,
                          #               because plasticity acts on |w| alone

    # --- fixed genome parameters -----------------------------------------
    Mod: torch.Tensor     # (P, n, n, n)  Mod[p, k, i, j] = weight of modulator k on edge i->j
    Tonic: torch.Tensor   # (P, 1, n)     per-neuron tonic drive
    Eta: torch.Tensor     # (P, 1, 1, 1)  global plasticity factor, broadcast over edges

    # --- derived per-genome integers --------------------------------------
    warmup_ticks: torch.Tensor  # (P,) int64
    max_ticks: torch.Tensor     # (P,) int64

    # --- shared across the whole batch (from YAML, not from the genome) ---
    S_w: torch.Tensor           # (n_sensors, n)  sensory weight, one nonzero per row
    S_r: torch.Tensor           # (n_sensors, n)  sensory reliability
    sensor_keys: Tuple[str, ...]
    sensor_targets: torch.Tensor  # (n_sensors,) int64
    threshold_raw: float          # atanh(threshold); compare pre-tanh, see 0b
    max_decision_delay: float

    # --- metadata ----------------------------------------------------------
    n_pop: int
    n_neurons: int
    device: torch.device
    dtype: torch.dtype

    # Number of (genome, edge, modulator) collisions folded together during
    # encoding; see `_fill_modulation`.  Non-zero is legal but means the tensor
    # brain cannot be bit-exact against the scalar sim for those edges.
    n_folded_modulators: int = 0
    folded_modulator_edges: List[Tuple[int, int, int, int]] = field(default_factory=list)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"GenomeBatch(P={self.n_pop}, n={self.n_neurons}, "
            f"device={self.device}, dtype={self.dtype}, "
            f"n_sensors={len(self.sensor_keys)}, "
            f"folded_modulators={self.n_folded_modulators})"
        )


# ============================================================
# Guards
# ============================================================

def _resolve_device_dtype(device, dtype) -> Tuple[torch.device, torch.dtype]:
    """Normalise device/dtype and reject combinations the backend cannot honour."""
    device = torch.device(device)
    if dtype not in (torch.float32, torch.float64):
        raise ValueError(
            f"unsupported dtype {dtype!r}: the codec supports float32 (production) "
            f"and float64 (reference/equivalence tests) only"
        )
    if device.type == "mps" and dtype == torch.float64:
        raise ValueError(
            "float64 is not available on the MPS backend -- Metal Shading Language "
            "has no double type, so this would have to be silently downcast to "
            "float32 and the resulting run could not be compared bit-exactly "
            "against the scalar reference. Use device='cpu' for float64 reference "
            "runs, or dtype=torch.float32 on MPS for benchmarks. "
            "See reproducibility.md §3."
        )
    return device, dtype


def _require(condition: bool, message: str) -> None:
    """Assertion that survives ``python -O`` -- these guard result correctness."""
    if not condition:
        raise ValueError(message)


# ============================================================
# Per-genome encoding helpers
# ============================================================

def _fill_modulation(
    mod_out: np.ndarray,
    modulation_spec: Dict[Tuple[int, int], Sequence[Tuple[int, float]]],
    weights: np.ndarray,
    n_neurons: int,
    p: int,
    folded: List[Tuple[int, int, int, int]],
) -> None:
    """Write one genome's modulation spec into ``mod_out[k, i, j]``.

    Two behaviours here are not obvious and both are required to match
    ``init_brain``:

    1.  Entries whose edge has zero weight are dropped.  ``init_brain`` reaches
        its ``continue`` before it looks at ``modulation_spec``, so such entries
        never become a ``Connection`` and never affect the simulation.  Mutation
        can leave them behind (e.g. a weight clipped to exactly 0.0), so this is
        a real case, not a defensive one.

    2.  The same modulator may appear more than once on the same edge.  The
        mutation operators make this reachable: ``add_modulation`` does not check
        for an existing entry, and ``connection_new_source`` / ``connection_new_target``
        ``extend`` one edge's modulator list onto another's.  The scalar sim keeps
        them as separate list entries and accumulates
        ``w1*a_k + w2*a_k``; a dense ``(k, i, j)`` tensor can only hold one number
        per modulator, so the duplicates are summed into ``(w1 + w2)*a_k``.

        That is exact in real arithmetic and, because activities are exactly 0.0
        or 1.0, differs from the scalar sim only in the grouping of a
        floating-point sum -- so it can differ in the last bit.  Every collision
        is recorded so an equivalence failure can be attributed instead of hunted.
    """
    for (src, tgt), modulators in modulation_spec.items():
        _require(
            0 <= src < n_neurons and 0 <= tgt < n_neurons,
            f"genome {p}: modulation_spec key ({src}, {tgt}) out of range for "
            f"n_neurons={n_neurons}",
        )
        if weights[src, tgt] == 0.0:
            # No Connection object exists for this edge; init_brain ignores it.
            continue
        for mod_id, mod_weight in modulators:
            mod_id = int(mod_id)
            _require(
                0 <= mod_id < n_neurons,
                f"genome {p}: modulator id {mod_id} on edge ({src}, {tgt}) out of "
                f"range for n_neurons={n_neurons}",
            )
            if mod_out[mod_id, src, tgt] != 0.0:
                folded.append((p, mod_id, src, tgt))
            mod_out[mod_id, src, tgt] += float(mod_weight)


def _warmup_and_max_ticks(
    weights: np.ndarray,
    sensor_targets: Sequence[int],
    max_decision_delay: float,
) -> Tuple[int, int]:
    """Reproduce ``_calculate_warmup_and_max_ticks`` from the genome alone.

    The scalar version counts neurons that have any incoming connection, unioned
    with the *sources* of all connections whose source is a Neuron.  Sensory
    connections contribute their target (they give it an ``incoming``) but not a
    source (an ``InputSource`` is not a ``Neuron``).
    """
    has_edge = weights != 0.0
    active = set(np.flatnonzero(has_edge.any(axis=0)).tolist())   # any incoming from a neuron
    active |= set(np.flatnonzero(has_edge.any(axis=1)).tolist())  # any outgoing to a neuron
    active |= {int(t) for t in sensor_targets}                    # incoming from a sensor
    warmup_ticks = len(active)
    return warmup_ticks, int(warmup_ticks * max_decision_delay)


# ============================================================
# Public entry point
# ============================================================

def encode_genomes(
    genomes: Sequence[Any],
    brain_cfg: Dict[str, Any],
    device: Any = "cpu",
    dtype: torch.dtype = torch.float64,
) -> GenomeBatch:
    """Encode P scalar genomes into a :class:`GenomeBatch`.

    Parameters
    ----------
    genomes
        Sequence of genome objects supporting ``genome["connection_weights"]``,
        ``["modulation_spec"]``, ``["tonic_activations"]``, ``["eta"]`` -- i.e.
        ``GenomeRandomResult``, ``GenomeLookuResult``, or a plain dict.
    brain_cfg
        The YAML ``brain`` section: ``n_neurons``, ``threshold``,
        ``sensory_mapping``, ``max_decision_delay``.  ``noise_level`` and
        ``output_mapping`` are not genome data and are consumed elsewhere
        (the decision machine, Step 3).
    device, dtype
        See the device/dtype policy in the package docstring.  Defaults are the
        reference configuration (cpu/float64), so a test that forgets to pass
        them still gets the exact one.

    Raises
    ------
    ValueError
        On any structural inconsistency.  There are no defaults and no silent
        repairs: a genome that would not have produced this brain under
        ``init_brain`` must not encode.
    """
    device, dtype = _resolve_device_dtype(device, dtype)

    _require(len(genomes) > 0, "encode_genomes: empty genome list")
    P = len(genomes)

    for key in ("n_neurons", "threshold", "sensory_mapping", "max_decision_delay"):
        _require(key in brain_cfg, f"brain config is missing required key '{key}'")
    n = int(brain_cfg["n_neurons"])
    threshold = float(brain_cfg["threshold"])
    max_decision_delay = float(brain_cfg["max_decision_delay"])
    sensory_mapping = brain_cfg["sensory_mapping"]

    # --- sensory wiring (shared across the batch, comes from YAML) --------
    sensor_keys: List[str] = list(sensory_mapping.keys())
    n_sensors = len(sensor_keys)
    S_w_np = np.zeros((n_sensors, n), dtype=np.float64)
    S_r_np = np.zeros((n_sensors, n), dtype=np.float64)
    sensor_targets: List[int] = []
    for s_idx, key in enumerate(sensor_keys):
        target, s_weight, s_rel = sensory_mapping[key]
        target = int(target)
        _require(
            0 <= target < n,
            f"sensory_mapping['{key}'] targets neuron {target}, outside [0, {n})",
        )
        S_w_np[s_idx, target] = float(s_weight)
        S_r_np[s_idx, target] = float(s_rel)
        sensor_targets.append(target)
    # Distinct targets are not required by init_brain, but a shared target would
    # make the dense (n_sensors, n) contraction sum sensory inputs in a different
    # order than the scalar sim's per-neuron `incoming` list. Refuse rather than
    # silently lose bit-exactness.
    _require(
        len(set(sensor_targets)) == n_sensors,
        f"two sensors share a target neuron ({sensor_targets}); the dense sensory "
        f"contraction would reorder their summation relative to the scalar sim",
    )

    # --- per-genome buffers, filled in numpy then moved to torch once -----
    Wabs0_np = np.zeros((P, n, n), dtype=np.float64)
    Wsign_np = np.zeros((P, n, n), dtype=np.float64)
    Rel_np = np.zeros((P, n, n), dtype=np.float64)
    Mod_np = np.zeros((P, n, n, n), dtype=np.float64)
    Tonic_np = np.zeros((P, n), dtype=np.float64)
    Eta_np = np.zeros((P,), dtype=np.float64)
    warmup_np = np.zeros((P,), dtype=np.int64)
    maxticks_np = np.zeros((P,), dtype=np.int64)
    folded: List[Tuple[int, int, int, int]] = []

    for p, genome in enumerate(genomes):
        cw = np.asarray(genome["connection_weights"], dtype=np.float64)
        _require(
            cw.shape == (n, n, 2),
            f"genome {p}: connection_weights has shape {cw.shape}, expected "
            f"({n}, {n}, 2) for n_neurons={n}",
        )
        weights = cw[:, :, 0]
        reliabilities = cw[:, :, 1]

        _require(
            bool(np.all(np.abs(weights) <= 1.0)),
            f"genome {p}: |connection weight| exceeds 1.0 "
            f"(max {np.abs(weights).max()}); the plasticity rule's invariant "
            f"|w| in (0, 1] assumes it does not",
        )
        # Reliability is only read on edges that exist; zeroed slots keep the
        # [0.0, 0.0] pair that the mutation operators write on removal.
        live = weights != 0.0
        if live.any():
            live_rel = reliabilities[live]
            _require(
                bool(np.all((live_rel >= 0.0) & (live_rel <= 1.0))),
                f"genome {p}: reliability outside [0, 1] on a live edge "
                f"(min {live_rel.min()}, max {live_rel.max()})",
            )

        Wabs0_np[p] = np.abs(weights)
        Wsign_np[p] = np.sign(weights)
        # Reliability is stored only where an edge exists, so a stale nonzero
        # reliability on a removed edge cannot leak into the contraction.
        Rel_np[p] = np.where(live, reliabilities, 0.0)

        _require(
            bool(np.all((Wsign_np[p] != 0.0) == (Wabs0_np[p] > 0.0))),
            f"genome {p}: sign/magnitude disagree about which edges exist",
        )

        _fill_modulation(Mod_np[p], genome["modulation_spec"], weights, n, p, folded)

        tonic = np.asarray(genome["tonic_activations"], dtype=np.float64)
        _require(
            tonic.shape == (n,),
            f"genome {p}: tonic_activations has shape {tonic.shape}, expected ({n},)",
        )
        Tonic_np[p] = tonic

        eta = float(genome["eta"])
        _require(
            0.0 <= eta <= 1.0,
            f"genome {p}: eta={eta} outside [0, 1]; the plasticity invariant "
            f"|w| in (0, 1] assumes eta <= 1",
        )
        Eta_np[p] = eta

        warmup_np[p], maxticks_np[p] = _warmup_and_max_ticks(
            weights, sensor_targets, max_decision_delay
        )

    def _t(array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, dtype=dtype, device=device)

    return GenomeBatch(
        Wabs0=_t(Wabs0_np).unsqueeze(1),   # (P, 1, n, n)
        Wsign=_t(Wsign_np).unsqueeze(1),
        Rel=_t(Rel_np).unsqueeze(1),
        Mod=_t(Mod_np),                    # (P, n, n, n) -- no run axis, ever
        Tonic=_t(Tonic_np).unsqueeze(1),   # (P, 1, n)
        Eta=_t(Eta_np).reshape(P, 1, 1, 1),
        warmup_ticks=torch.as_tensor(warmup_np, device=device),
        max_ticks=torch.as_tensor(maxticks_np, device=device),
        S_w=_t(S_w_np),
        S_r=_t(S_r_np),
        sensor_keys=tuple(sensor_keys),
        sensor_targets=torch.as_tensor(
            np.asarray(sensor_targets, dtype=np.int64), device=device
        ),
        threshold_raw=_raw_threshold(threshold),
        max_decision_delay=max_decision_delay,
        n_pop=P,
        n_neurons=n,
        device=device,
        dtype=dtype,
        n_folded_modulators=len(folded),
        folded_modulator_edges=folded,
    )


def _raw_threshold(threshold: float) -> float:
    """Pre-image of ``threshold`` under tanh.

    Mirrors ``mvb.brains.decisionmaking_plasticity._raw_threshold``; duplicated
    rather than imported so that ``mvb_torch`` does not reach into the scalar
    package at import time.  The equivalence test asserts the two agree bit-for-bit.
    """
    if threshold >= 1.0:
        return math.inf
    if threshold <= -1.0:
        return -math.inf
    return math.atanh(threshold)
