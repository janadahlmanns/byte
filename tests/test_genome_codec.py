#!/usr/bin/env python3
"""Tests for mvb_torch/genome_codec.py (plan_evotorch.md Step 1).

The codec's whole job is to reproduce `init_brain`'s wiring decisions in tensor
form. So the tests do not check the codec against a hand-written expectation --
they check it against `init_brain` itself, on genomes produced by every generator
the project uses.

The important pair is forward + reverse edge consistency:

    forward  every Connection object init_brain built has a matching tensor entry
    reverse  every nonzero tensor entry has a matching Connection object

Forward alone would miss spurious edges; reverse alone would miss dropped ones.
Both together pin the topology exactly.

Usage
-----
    python -m tests.test_genome_codec

Exits 0 on success, 1 if any check failed. No pytest dependency.
"""

import copy
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mvb.brains.decisionmaking_plasticity as dp  # noqa: E402
from mvb.brains.decisionmaking_plasticity import Neuron  # noqa: E402
from mvb.genome.generate_genome_lookup_hard import generate_lookup_hard_genome  # noqa: E402
from mvb.genome.generate_genome_lookup_soft import generate_lookup_soft_genome  # noqa: E402
from mvb.genome.generate_genome_mutate_simple import generate_genome_mutate_simple  # noqa: E402
from mvb.genome.generate_genome_random import generate_random_genome  # noqa: E402
from mvb_torch.genome_codec import encode_genomes  # noqa: E402

CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs", "experiments", "test_predrawn.yaml",
)


# ============================================================
# Harness
# ============================================================

_RESULTS = []


def check(name, condition):
    _RESULTS.append((name, bool(condition)))
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}")
    return bool(condition)


def raises(name, exc, fn):
    try:
        fn()
    except exc:
        return check(name, True)
    except Exception as e:  # noqa: BLE001
        return check(f"{name} (wrong exception: {type(e).__name__}: {e})", False)
    return check(f"{name} (no exception raised)", False)


# ============================================================
# Fixtures
# ============================================================

def load_brain_cfg():
    with open(CONFIG, "r") as f:
        return yaml.safe_load(f)["brain"]


def make_genomes(brain_cfg):
    """One genome of every kind the project can produce, including a mutated one.

    The mutated genome matters most: mutation is the only path that produces the
    awkward states (stale modulation, duplicated modulators, removed edges), and
    those are exactly what the codec has to get right.
    """
    cfg = {"brain": brain_cfg}
    random_g = generate_random_genome(cfg, rng_seed=1)
    hard_g = generate_lookup_hard_genome(cfg, rng_seed=2)
    soft_g = generate_lookup_soft_genome(cfg, rng_seed=3)

    # 15 generations of mutation, as an EA run would produce by generation 15.
    rng = np.random.default_rng(42)
    mutated = generate_random_genome(cfg, rng_seed=7)
    for _ in range(15):
        mutated = generate_genome_mutate_simple(mutated, 0.3, 0.2, rng)

    return {
        "random": random_g,
        "lookup_hard": hard_g,
        "lookup_soft": soft_g,
        "mutated_gen15": mutated,
    }


def build_reference(genome, brain_cfg):
    """Run the scalar init_brain and hand back its module-level BrainState."""
    dp.init_brain(genome, brain_cfg)
    return dp._brain_state


def split_connections(state):
    """Partition a BrainState's connections into neuron->neuron and sensory."""
    neuron_conns, sensory_conns = {}, []
    for tgt_neuron in state.neurons:
        for conn in tgt_neuron.incoming:
            if isinstance(conn.source, Neuron):
                neuron_conns[(conn.source.id, tgt_neuron.id)] = conn
            else:
                sensory_conns.append((conn.source.key, tgt_neuron.id, conn))
    return neuron_conns, sensory_conns


# ============================================================
# 1. Edge consistency, forward and reverse
# ============================================================

def test_edges(brain_cfg, genomes):
    print("\n[1] Edge consistency (forward + reverse) vs init_brain")
    for label, genome in genomes.items():
        state = build_reference(genome, brain_cfg)
        batch = encode_genomes([genome], brain_cfg)
        neuron_conns, _ = split_connections(state)

        W = (batch.Wabs0[0, 0] * batch.Wsign[0, 0]).numpy()
        Rel = batch.Rel[0, 0].numpy()
        tensor_edges = {(int(i), int(j)) for i, j in zip(*np.nonzero(batch.Wabs0[0, 0].numpy()))}

        # forward: every scalar Connection is present in the tensor, with the
        # same weight and reliability, at the SAME [src, tgt] index (a transposed
        # codec would still pass a count-only check, so compare per index).
        fwd = all(
            W[i, j] == conn.weight and Rel[i, j] == conn.reliability
            for (i, j), conn in neuron_conns.items()
        )
        check(f"{label}: forward -- every Connection has a matching tensor entry", fwd)

        # reverse: no tensor entry without a Connection behind it
        check(f"{label}: reverse -- no spurious/missing tensor edges",
              tensor_edges == set(neuron_conns.keys()))

        check(f"{label}: sign is exact on live edges and zero elsewhere",
              np.array_equal(batch.Wsign[0, 0].numpy() != 0.0,
                             batch.Wabs0[0, 0].numpy() > 0.0))

        # reliability must NOT be folded into the weight -- plasticity acts on
        # |w| alone, so a folded product would drift differently under the rule.
        unfolded = all(W[i, j] == conn.weight for (i, j), conn in neuron_conns.items()
                       if conn.reliability != 1.0)
        check(f"{label}: reliability is not folded into the weight", unfolded)


# ============================================================
# 2. Modulation
# ============================================================

def test_modulation(brain_cfg, genomes):
    print("\n[2] Modulation tensor vs Connection.modulating_inputs")
    for label, genome in genomes.items():
        state = build_reference(genome, brain_cfg)
        batch = encode_genomes([genome], brain_cfg)
        neuron_conns, _ = split_connections(state)
        Mod = batch.Mod[0].numpy()  # (k, i, j)

        # Expected tensor rebuilt from the live Connection objects, summing
        # duplicate modulators the same way the codec does.
        expected = np.zeros_like(Mod)
        n_scalar_entries = 0
        for (i, j), conn in neuron_conns.items():
            for mod_neuron, mod_weight in conn.modulating_inputs:
                expected[mod_neuron.id, i, j] += float(mod_weight)
                n_scalar_entries += 1

        check(f"{label}: Mod[k,i,j] matches modulating_inputs (identity + weight)",
              np.array_equal(Mod, expected))

        # Modulation on edges that init_brain never built must not appear.
        live = batch.Wabs0[0, 0].numpy() > 0.0
        check(f"{label}: no modulation on dead edges",
              not Mod[:, ~live].any())

        # Modulator count per edge, ignoring folded duplicates.
        folded_here = {(k, i, j) for (_p, k, i, j) in batch.folded_modulator_edges}
        tensor_count = int((Mod != 0.0).sum()) + len(folded_here)
        check(f"{label}: modulator count matches ({n_scalar_entries} scalar entries)",
              tensor_count == n_scalar_entries or _explain_zero_weight_mods(
                  neuron_conns, Mod, n_scalar_entries, tensor_count))


def _explain_zero_weight_mods(neuron_conns, Mod, n_scalar, n_tensor):
    """A modulator whose weight is exactly 0.0 is invisible in a dense tensor.

    That is harmless (it contributes 0.0 to the modulation sum either way) but it
    makes a raw nonzero-count comparison off by one. Accept the mismatch only if
    it is fully explained by such entries.
    """
    zero_weight = sum(
        1
        for conn in neuron_conns.values()
        for _mod, w in conn.modulating_inputs
        if float(w) == 0.0
    )
    return n_tensor + zero_weight == n_scalar


# ============================================================
# 3. Sensory, tonic, eta, tick counts, threshold
# ============================================================

def test_scalars(brain_cfg, genomes):
    print("\n[3] Sensory wiring, tonic, eta, tick counts, threshold")
    for label, genome in genomes.items():
        state = build_reference(genome, brain_cfg)
        batch = encode_genomes([genome], brain_cfg)
        _, sensory_conns = split_connections(state)

        S_w = batch.S_w.numpy()
        S_r = batch.S_r.numpy()
        ok = len(sensory_conns) == len(batch.sensor_keys)
        for key, tgt, conn in sensory_conns:
            s = batch.sensor_keys.index(key)
            ok &= (S_w[s, tgt] == conn.weight and S_r[s, tgt] == conn.reliability)
            ok &= int(batch.sensor_targets[s]) == tgt
            ok &= int((S_w[s] != 0.0).sum()) == 1
        check(f"{label}: the five sensory connections match sensory_mapping", bool(ok))

        check(f"{label}: tonic matches neuron.tonic_level",
              np.array_equal(batch.Tonic[0, 0].numpy(),
                             np.array([nrn.tonic_level for nrn in state.neurons])))
        check(f"{label}: eta matches", float(batch.Eta[0, 0, 0, 0]) == state.eta)

        w_ref, m_ref = dp._calculate_warmup_and_max_ticks(state)
        check(f"{label}: warmup_ticks/max_ticks match "
              f"_calculate_warmup_and_max_ticks ({w_ref}, {m_ref})",
              int(batch.warmup_ticks[0]) == w_ref and int(batch.max_ticks[0]) == m_ref)

        check(f"{label}: threshold_raw matches the scalar sim bit-for-bit",
              batch.threshold_raw == dp._raw_threshold(brain_cfg["threshold"]))


# ============================================================
# 3b. Sparse genomes: warmup < n_neurons
# ============================================================

def test_warmup_sparse(brain_cfg, genomes):
    """The dense genomes above all give warmup == n_neurons, which hides bugs.

    Production does not: footnote F2 in plan_evotorch.md found 38% of
    `ea_from_lookup` genomes running with warmup < 11. The interesting part is
    that a neuron with no neuron-to-neuron edges is still "active" if a sensor
    targets it, because the sensory Connection lands in its `incoming` list --
    so the sensor targets have to be unioned in, and a codec that only looked at
    the weight matrix would undercount.
    """
    print("\n[3b] Sparse genomes (warmup < n_neurons)")
    n = brain_cfg["n_neurons"]
    n_sensors = len(brain_cfg["sensory_mapping"])

    def sparsify(edges):
        g = copy.deepcopy(genomes["random"])
        g.connection_weights[:] = 0.0
        g.modulation_spec = {}
        for (i, j) in edges:
            g.connection_weights[i, j] = [0.5, 0.9]
        return g

    cases = [
        ("no neuron edges at all -> only sensor targets are active", [], n_sensors),
        ("one edge between two non-sensor neurons", [(5, 6)], n_sensors + 2),
        ("one edge into an already-active sensor target", [(5, 0)], n_sensors + 1),
        ("a self-loop counts its neuron once", [(7, 7)], n_sensors + 1),
    ]
    for label, edges, expected in cases:
        g = sparsify(edges)
        state = build_reference(g, brain_cfg)
        w_ref, m_ref = dp._calculate_warmup_and_max_ticks(state)
        batch = encode_genomes([g], brain_cfg)
        check(f"{label}: warmup == {expected}",
              int(batch.warmup_ticks[0]) == expected == w_ref)
        check(f"{label}: max_ticks matches the scalar sim ({m_ref})",
              int(batch.max_ticks[0]) == m_ref)

    # max_ticks truncates rather than rounds: int(7 * 1.5) == 10, not 11.
    frac = copy.deepcopy(brain_cfg)
    frac["max_decision_delay"] = 1.5
    g = sparsify([(5, 6)])
    state = build_reference(g, frac)
    w_ref, m_ref = dp._calculate_warmup_and_max_ticks(state)
    batch = encode_genomes([g], frac)
    check(f"fractional max_decision_delay truncates ({w_ref} * 1.5 -> {m_ref})",
          int(batch.max_ticks[0]) == m_ref == 10)


# ============================================================
# 4. Batching: P genomes at once must equal P encodings of one
# ============================================================

def test_batching(brain_cfg, genomes):
    print("\n[4] Batching")
    gs = list(genomes.values())
    batch = encode_genomes(gs, brain_cfg)
    check("shapes follow the (P, ...) contract",
          batch.Wabs0.shape == (len(gs), 1, batch.n_neurons, batch.n_neurons)
          and batch.Mod.shape == (len(gs), batch.n_neurons, batch.n_neurons, batch.n_neurons)
          and batch.Tonic.shape == (len(gs), 1, batch.n_neurons)
          and batch.Eta.shape == (len(gs), 1, 1, 1)
          and batch.warmup_ticks.shape == (len(gs),))
    check("Mod carries no run dimension (binding rule §5.1)", batch.Mod.dim() == 4)

    per_genome_ok = True
    for p, g in enumerate(gs):
        one = encode_genomes([g], brain_cfg)
        per_genome_ok &= bool(
            torch.equal(batch.Wabs0[p], one.Wabs0[0])
            and torch.equal(batch.Wsign[p], one.Wsign[0])
            and torch.equal(batch.Rel[p], one.Rel[0])
            and torch.equal(batch.Mod[p], one.Mod[0])
            and torch.equal(batch.Tonic[p], one.Tonic[0])
            and torch.equal(batch.Eta[p], one.Eta[0])
            and int(batch.warmup_ticks[p]) == int(one.warmup_ticks[0])
            and int(batch.max_ticks[p]) == int(one.max_ticks[0])
        )
    check("slice p of a batch == a solo encoding of genome p", per_genome_ok)

    # Genomes differ, so the batch must not have collapsed onto one of them.
    check("batched genomes stay distinct",
          not torch.equal(batch.Wabs0[0], batch.Wabs0[1]))


# ============================================================
# 5. Device / dtype
# ============================================================

def test_device_dtype(brain_cfg, genomes):
    print("\n[5] Device and dtype")
    g = [genomes["random"]]

    f64 = encode_genomes(g, brain_cfg, device="cpu", dtype=torch.float64)
    f32 = encode_genomes(g, brain_cfg, device="cpu", dtype=torch.float32)
    check("float64 on cpu", f64.Wabs0.dtype == torch.float64 and f64.Mod.dtype == torch.float64)
    check("float32 on cpu", f32.Wabs0.dtype == torch.float32 and f32.Mod.dtype == torch.float32)
    check("default is the reference config (cpu/float64)",
          encode_genomes(g, brain_cfg).dtype == torch.float64)
    check("index tensors stay int64 regardless of dtype",
          f32.warmup_ticks.dtype == torch.int64 and f32.sensor_targets.dtype == torch.int64)
    check("float32 round-trips to float64 values",
          torch.equal(f32.Wabs0.double(), f64.Wabs0.double().float().double()))

    raises("unsupported dtype fails loudly", ValueError,
           lambda: encode_genomes(g, brain_cfg, dtype=torch.float16))
    raises("float64 on mps fails loudly instead of downcasting", ValueError,
           lambda: encode_genomes(g, brain_cfg, device="mps", dtype=torch.float64))

    if torch.backends.mps.is_available():
        mps = encode_genomes(g, brain_cfg, device="mps", dtype=torch.float32)
        check("float32 on mps works",
              mps.Wabs0.device.type == "mps"
              and torch.equal(mps.Wabs0.cpu(), f32.Wabs0))
    else:
        print("  [SKIP] MPS not available on this machine")


# ============================================================
# 6. Every guard fires on a deliberately corrupted genome
# ============================================================

def test_guards(brain_cfg, genomes):
    print("\n[6] Guards fire on corrupted input")
    base = genomes["random"]
    n = brain_cfg["n_neurons"]

    def corrupt(fn):
        g = copy.deepcopy(base)
        fn(g)
        return lambda: encode_genomes([g], brain_cfg)

    def first_live_edge(g):
        w = g.connection_weights[:, :, 0]
        i, j = np.argwhere(w != 0.0)[0]
        return int(i), int(j)

    def set_weight(g, v):
        i, j = first_live_edge(g)
        g.connection_weights[i, j, 0] = v

    def set_reliability(g, v):
        i, j = first_live_edge(g)
        g.connection_weights[i, j, 1] = v

    raises("|weight| > 1 rejected", ValueError, corrupt(lambda g: set_weight(g, 2.0)))
    raises("reliability outside [0,1] rejected", ValueError,
           corrupt(lambda g: set_reliability(g, 1.5)))
    raises("eta > 1 rejected", ValueError,
           corrupt(lambda g: setattr(g, "eta", 1.5)))
    raises("eta < 0 rejected", ValueError,
           corrupt(lambda g: setattr(g, "eta", -0.1)))
    raises("wrong connection_weights shape rejected", ValueError,
           corrupt(lambda g: setattr(g, "connection_weights",
                                     np.zeros((n + 1, n + 1, 2), dtype=np.float32))))
    raises("wrong tonic_activations shape rejected", ValueError,
           corrupt(lambda g: setattr(g, "tonic_activations",
                                     np.zeros(n + 1, dtype=np.float32))))
    raises("out-of-range modulator id rejected", ValueError,
           corrupt(lambda g: g.modulation_spec.__setitem__(first_live_edge(g),
                                                           [(n + 5, 0.5)])))
    raises("out-of-range modulation_spec key rejected", ValueError,
           corrupt(lambda g: g.modulation_spec.__setitem__((n + 3, 0), [(0, 0.5)])))
    raises("empty genome list rejected", ValueError,
           lambda: encode_genomes([], brain_cfg))

    missing = {k: v for k, v in brain_cfg.items() if k != "sensory_mapping"}
    raises("missing brain config key fails loudly", ValueError,
           lambda: encode_genomes([base], missing))

    shared = copy.deepcopy(brain_cfg)
    shared["sensory_mapping"]["food_north"] = [0, 0.8, 0.9]  # collide with on_food
    raises("two sensors on one neuron rejected", ValueError,
           lambda: encode_genomes([base], shared))

    oob = copy.deepcopy(brain_cfg)
    oob["sensory_mapping"]["on_food"] = [n + 2, 1.0, 0.9]
    raises("sensor targeting a nonexistent neuron rejected", ValueError,
           lambda: encode_genomes([base], oob))

    # A guard that must NOT fire: modulation on a removed edge is legal genome
    # state (mutation leaves it behind) and init_brain silently ignores it.
    stale = copy.deepcopy(base)
    i, j = first_live_edge(stale)
    stale.modulation_spec[(i, j)] = [(0, 0.5)]
    stale.connection_weights[i, j] = [0.0, 0.0]
    enc = encode_genomes([stale], brain_cfg)
    check("stale modulation on a dead edge is ignored, not rejected",
          float(enc.Mod[0, 0, i, j]) == 0.0)


# ============================================================
# 7. Duplicate modulators are folded and reported
# ============================================================

def test_folding(brain_cfg, genomes):
    print("\n[7] Duplicate-modulator folding")
    g = copy.deepcopy(genomes["random"])
    w = g.connection_weights[:, :, 0]
    i, j = (int(x) for x in np.argwhere(w != 0.0)[0])
    g.modulation_spec[(i, j)] = [(3, 0.25), (3, 0.5), (4, -0.125)]

    batch = encode_genomes([g], brain_cfg)
    check("duplicate modulators are summed into one tensor entry",
          float(batch.Mod[0, 3, i, j]) == 0.75 and float(batch.Mod[0, 4, i, j]) == -0.125)
    check("the collision is reported, not swallowed",
          batch.n_folded_modulators == 1
          and batch.folded_modulator_edges == [(0, 3, i, j)])
    check("a clean genome reports no folding",
          encode_genomes([genomes["random"]], brain_cfg).n_folded_modulators == 0)


# ============================================================

def main():
    print("=" * 66)
    print("mvb_torch/genome_codec.py test suite")
    print("=" * 66)
    brain_cfg = load_brain_cfg()
    genomes = make_genomes(brain_cfg)

    test_edges(brain_cfg, genomes)
    test_modulation(brain_cfg, genomes)
    test_scalars(brain_cfg, genomes)
    test_warmup_sparse(brain_cfg, genomes)
    test_batching(brain_cfg, genomes)
    test_device_dtype(brain_cfg, genomes)
    test_guards(brain_cfg, genomes)
    test_folding(brain_cfg, genomes)

    failed = [name for name, ok in _RESULTS if not ok]
    print("\n" + "=" * 66)
    if failed:
        print(f"FAILED {len(failed)}/{len(_RESULTS)}")
        for name in failed:
            print(f"  - {name}")
        return 1
    print(f"OK -- {len(_RESULTS)}/{len(_RESULTS)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
