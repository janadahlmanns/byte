#!/usr/bin/env python3
"""Tests for mvb_torch/brain.py (plan_evotorch.md Step 2).

The oracle is the scalar simulation itself, driven tick by tick. `decide()` cannot
be used directly -- it runs a whole decision and draws its own noise -- so the
harness below calls the same primitives in the same order `decide()` does:

    neuron.update(noise_i)   for all neurons     <- writes next_activity
    conn.update(eta)         for all connections <- reads PRE-commit activities
    _get_output_state(state)                     <- PRE-commit snapshot (S2)
    neuron.commit() / conn.commit()

That ordering is the specification, not an implementation detail: plasticity is
driven by the activities at the start of the tick, and the recorded output lags one
tick behind the activity just computed.

Usage
-----
    python -m tests.test_brain_tick

Exits 0 on success, 1 if any check failed. No pytest dependency.
"""

import copy
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.devices import accelerators, all_devices  # noqa: E402

import mvb.brains.decisionmaking_plasticity as dp  # noqa: E402
from mvb_torch.brain import (  # noqa: E402
    OUTPUT_SLICE,
    _accumulate_input,
    brain_tick,
    init_state,
)
from mvb_torch.genome_codec import encode_genomes  # noqa: E402
from tests.test_genome_codec import load_brain_cfg, make_genomes  # noqa: E402

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
# Scalar oracle
# ============================================================

def new_scalar_brain(genome, brain_cfg):
    """Fresh, independent BrainState. init_brain builds new objects each call."""
    dp.init_brain(genome, brain_cfg)
    return dp._brain_state


def scalar_set_inputs(state, sens_by_key):
    """Latch sensory inputs, as `decide()` does ONCE before its tick loop (S6).

    Kept separate from scalar_tick on purpose: doing it per brain tick would still
    give the same values, but it would let a net-input comparison run against
    un-latched sources and silently compare the wrong thing.
    """
    for src in state.input_sources:
        src.update(sens_by_key)


def scalar_tick(state, noise_vec):
    """One brain tick in `decide()`'s exact order. Returns the pre-commit snapshot."""
    for i, nrn in enumerate(state.neurons):
        nrn.update(noise_vec[i])
    for conn in state.connections:
        conn.update(state.eta)
    snapshot = dp._get_output_state(state)       # AFTER update, BEFORE commit
    for nrn in state.neurons:
        nrn.commit()
    for conn in state.connections:
        conn.commit()
    return snapshot


def scalar_act(state):
    return np.array([nrn.activity for nrn in state.neurons], dtype=np.float64)


def scalar_wabs(state, n):
    """|weight| per (src, tgt). conn_index excludes sensory edges by construction."""
    W = np.zeros((n, n), dtype=np.float64)
    for (src, tgt), conn in state.conn_index.items():
        W[src, tgt] = abs(conn.weight)
    return W


def max_mods_per_edge(genome):
    counts = [len(v) for v in genome.modulation_spec.values()]
    return max(counts) if counts else 0


# ============================================================
# 1. Lockstep equivalence against the scalar sim
# ============================================================

def test_lockstep(brain_cfg, genomes, K=64, R=3):
    print(f"\n[1] Lockstep vs scalar sim ({K} brain ticks, R={R})")
    labels = list(genomes)
    gs = [genomes[k] for k in labels]
    n = brain_cfg["n_neurons"]
    batch = encode_genomes(gs, brain_cfg)            # cpu / float64
    P = len(gs)
    keys = list(batch.sensor_keys)

    rng = np.random.default_rng(11)
    sens_np = rng.integers(0, 2, size=(P, R, len(keys))).astype(np.float64)
    noise_np = rng.normal(0.0, brain_cfg["noise_level"], size=(K, P, R, n))

    brains = {(p, r): new_scalar_brain(gs[p], brain_cfg)
              for p in range(P) for r in range(R)}
    # Latch sensory inputs once, before any tick -- exactly where decide() does it.
    for p in range(P):
        for r in range(R):
            scalar_set_inputs(brains[(p, r)],
                              {k: sens_np[p, r, i] for i, k in enumerate(keys)})
    state = init_state(batch, R)
    sens = torch.tensor(sens_np)

    # The net input is the ONLY place accumulation order is observable. Activities
    # cannot reveal it (the threshold margin is ~3.8e-06 against a ~1e-16
    # perturbation) and weights depend on activities alone, never on net. Without
    # this check, moving tonic out of the accumulator -- exactly the reordering that
    # costs einsum its bit-exactness -- passes the whole suite silently.
    net_bad = 0
    act_bad = {lbl: 0 for lbl in labels}
    w_bad = {lbl: 0 for lbl in labels}
    w_maxdiff = {lbl: 0.0 for lbl in labels}
    snap_bad = 0

    for t in range(K):
        noise = torch.tensor(noise_np[t])
        # Net input, compared before either side advances. compute_input() is pure,
        # so calling it here does not disturb the scalar brain.
        W = (state.Wabs * batch.Wsign) * batch.Rel
        t_net = _accumulate_input(batch, state, W, sens, noise, "sequential")
        for p in range(P):
            for r in range(R):
                s_net = np.array([
                    nrn.compute_input(noise_np[t, p, r, i])
                    for i, nrn in enumerate(brains[(p, r)].neurons)
                ], dtype=np.float64)
                if not np.array_equal(s_net, t_net[p, r].numpy()):
                    net_bad += 1

        state, snap = brain_tick(batch, state, sens, noise, contraction="sequential")
        for p in range(P):
            for r in range(R):
                st = brains[(p, r)]
                pre_act = scalar_act(st)                     # pre-tick, for snapshot
                s_snap = scalar_tick(st, list(noise_np[t, p, r]))
                if not np.array_equal(np.array(s_snap), snap[p, r].numpy()):
                    snap_bad += 1
                if not np.array_equal(np.array(s_snap), pre_act[OUTPUT_SLICE]):
                    snap_bad += 1     # oracle self-check: snapshot IS the pre-tick act
                lbl = labels[p]
                if not np.array_equal(scalar_act(st), state.act[p, r].numpy()):
                    act_bad[lbl] += 1
                sw = scalar_wabs(st, n)
                tw = state.Wabs[p, r].numpy()
                if not np.array_equal(sw, tw):
                    w_bad[lbl] += 1
                    w_maxdiff[lbl] = max(w_maxdiff[lbl], float(np.abs(sw - tw).max()))

    check(f"net input bit-exact vs compute_input, all {K*P*R} evaluations "
          f"({net_bad} bad) -- pins the accumulation ORDER", net_bad == 0)
    check(f"output snapshot matches the pre-commit read, every tick ({snap_bad} bad)",
          snap_bad == 0)
    for lbl in labels:
        check(f"{lbl}: activities bit-exact for all {K} ticks", act_bad[lbl] == 0)

    # Weights: bit-exact where <= 2 modulators/edge; the modulation sum's order is
    # otherwise unreproducible from a dense tensor (see brain._modulation_sum).
    for lbl in labels:
        mm = max_mods_per_edge(genomes[lbl])
        if mm <= 2:
            check(f"{lbl}: |w| bit-exact (max {mm} mods/edge)", w_bad[lbl] == 0)
        else:
            ok = w_maxdiff[lbl] < 1e-12
            check(f"{lbl}: |w| within 1e-12 (max {mm} mods/edge -> order differs, "
                  f"max diff {w_maxdiff[lbl]:.2e})", ok)


# ============================================================
# 2. Transpose detector
# ============================================================

def _one_edge_genome(base, brain_cfg, src, tgt):
    g = copy.deepcopy(base)
    g.connection_weights[:] = 0.0
    g.modulation_spec = {}
    g.connection_weights[src, tgt] = [1.0, 1.0]
    g.tonic_activations[:] = 0.0
    g.tonic_activations[src] = 1.0       # > atanh(0.5) = 0.549 -> src fires every tick
    g.eta = 0.0
    return g


def test_transpose(brain_cfg, genomes):
    print("\n[2] Transpose detector (a single 3->7 edge must not flow 7->3)")
    g = _one_edge_genome(genomes["random"], brain_cfg, 3, 7)
    batch = encode_genomes([g], brain_cfg)
    state = init_state(batch, 1)
    hist = []
    for _ in range(4):
        state, _ = brain_tick(batch, state, None, None, contraction="sequential")
        hist.append(state.act[0, 0].numpy().copy())

    check("tick 1: source neuron 3 fires from tonic", hist[0][3] == 1.0)
    check("tick 2: target neuron 7 fires (signal flowed 3 -> 7)", hist[1][7] == 1.0)
    check("neuron 3 is never driven BY 7 (no reverse flow)",
          all(h[3] == 1.0 for h in hist))
    quiet = [i for i in range(batch.n_neurons) if i not in (3, 7)]
    check("no other neuron ever fires", all(h[quiet].sum() == 0.0 for h in hist))

    # And the mirror image: the transposed genome must behave differently.
    gt = _one_edge_genome(genomes["random"], brain_cfg, 7, 3)
    gt.tonic_activations[:] = 0.0
    gt.tonic_activations[3] = 1.0       # drive 3, but the edge now points 7->3
    bt = encode_genomes([gt], brain_cfg)
    st = init_state(bt, 1)
    for _ in range(4):
        st, _ = brain_tick(bt, st, None, None, contraction="sequential")
    check("with the edge reversed, neuron 7 stays silent",
          st.act[0, 0, 7].item() == 0.0)


# ============================================================
# 3. einsum vs sequential
# ============================================================

def test_einsum_vs_sequential(brain_cfg, genomes, K=48, R=4):
    print(f"\n[3] einsum vs sequential ({K} ticks, R={R})")
    gs = list(genomes.values())
    batch = encode_genomes(gs, brain_cfg)
    P, n = len(gs), brain_cfg["n_neurons"]
    rng = np.random.default_rng(5)
    sens = torch.tensor(rng.integers(0, 2, size=(P, R, len(batch.sensor_keys))).astype(np.float64))
    s_seq, s_ein = init_state(batch, R), init_state(batch, R)
    act_div = 0
    w_max = 0.0
    for t in range(K):
        noise = torch.tensor(rng.normal(0.0, brain_cfg["noise_level"], size=(P, R, n)))
        s_seq, _ = brain_tick(batch, s_seq, sens, noise, contraction="sequential")
        s_ein, _ = brain_tick(batch, s_ein, sens, noise, contraction="einsum")
        if not torch.equal(s_seq.act, s_ein.act):
            act_div += 1
        w_max = max(w_max, (s_seq.Wabs - s_ein.Wabs).abs().max().item())
    check(f"activities identical across all {K} ticks ({act_div} divergent)",
          act_div == 0)
    check(f"|w| agrees to 1e-12 (max diff {w_max:.2e}) -- reduction order only",
          w_max < 1e-12)


# ============================================================
# 4. Run-dimension independence
# ============================================================

def test_run_independence(brain_cfg, genomes, K=24):
    print(f"\n[4] Run-dimension independence ({K} ticks)")
    gs = list(genomes.values())
    batch = encode_genomes(gs, brain_cfg)
    P, n, R = len(gs), brain_cfg["n_neurons"], 5
    rng = np.random.default_rng(9)
    sens_np = rng.integers(0, 2, size=(P, R, len(batch.sensor_keys))).astype(np.float64)
    noise_np = rng.normal(0.0, brain_cfg["noise_level"], size=(K, P, R, n))

    batched = init_state(batch, R)
    for t in range(K):
        batched, _ = brain_tick(batch, batched, torch.tensor(sens_np),
                               torch.tensor(noise_np[t]), contraction="sequential")

    ok = True
    for r in range(R):
        solo = init_state(batch, 1)
        for t in range(K):
            solo, _ = brain_tick(
                batch, solo,
                torch.tensor(sens_np[:, r : r + 1, :]),
                torch.tensor(noise_np[t][:, r : r + 1, :]),
                contraction="sequential",
            )
        ok &= bool(torch.equal(solo.act[:, 0], batched.act[:, r]))
        ok &= bool(torch.equal(solo.Wabs[:, 0], batched.Wabs[:, r]))
    check(f"each of {R} runs stepped alone == its slice of the batch", ok)
    check("runs actually differ (the test is not vacuous)",
          not torch.equal(batched.act[:, 0], batched.act[:, 1]))


# ============================================================
# 5. Binding rule + shapes
# ============================================================

def test_shapes(brain_cfg, genomes):
    print("\n[5] Shapes and the §5.1 binding rule")
    gs = list(genomes.values())
    batch = encode_genomes(gs, brain_cfg)
    P, n, R = len(gs), brain_cfg["n_neurons"], 6
    st = init_state(batch, R)
    check("init_state: act (P,R,n), Wabs (P,R,n,n)",
          st.act.shape == (P, R, n) and st.Wabs.shape == (P, R, n, n))
    check("initial activities are all zero (matches Neuron.__init__)",
          bool((st.act == 0).all()))
    check("initial Wabs == Wabs0 broadcast over runs",
          bool(torch.equal(st.Wabs, batch.Wabs0.expand(P, R, n, n))))
    st2, snap = brain_tick(batch, st, None, None)
    check("snapshot shape (P,R,5)", snap.shape == (P, R, 5))
    check("Mod never gains a run dimension", batch.Mod.dim() == 4)
    check("Wsign/Rel/Tonic/Eta keep their singleton run axis",
          batch.Wsign.shape[1] == 1 and batch.Rel.shape[1] == 1
          and batch.Tonic.shape[1] == 1 and batch.Eta.shape == (P, 1, 1, 1))
    raises("bad contraction name fails loudly", ValueError,
           lambda: brain_tick(batch, st, None, None, contraction="fast"))
    raises("wrong sens shape fails loudly", ValueError,
           lambda: brain_tick(batch, st, torch.zeros(P, R, 99), None))
    raises("wrong noise shape fails loudly", ValueError,
           lambda: brain_tick(batch, st, None, torch.zeros(P, R, 99)))
    raises("n_runs < 1 fails loudly", ValueError, lambda: init_state(batch, 0))


# ============================================================
# 6. Plasticity invariants
# ============================================================

def test_plasticity(brain_cfg, genomes, K=200):
    print(f"\n[6] Plasticity invariants ({K} ticks)")
    gs = list(genomes.values())
    batch = encode_genomes(gs, brain_cfg)
    P, n, R = len(gs), brain_cfg["n_neurons"], 3
    rng = np.random.default_rng(2)
    live = batch.Wabs0.expand(P, R, n, n) > 0
    st = init_state(batch, R)
    sens = torch.tensor(rng.integers(0, 2, size=(P, R, len(batch.sensor_keys))).astype(np.float64))
    ok_range = ok_nonzero = True
    for t in range(K):
        noise = torch.tensor(rng.normal(0.0, brain_cfg["noise_level"], size=(P, R, n)))
        st, _ = brain_tick(batch, st, sens, noise, contraction="sequential")
        ok_range &= bool((st.Wabs <= 1.0).all()) and bool((st.Wabs >= 0.0).all())
        ok_nonzero &= bool((st.Wabs[live] > 0.0).all())
        ok_range &= bool((st.Wabs[~live] == 0.0).all())
    check("|w| stays in [0, 1] and dead edges stay dead", ok_range)
    check("zero is unreachable for live edges -> sparsity is constant for life",
          ok_nonzero)
    check("activities are exactly 0.0 or 1.0",
          bool(((st.act == 0.0) | (st.act == 1.0)).all()))

    # eta = 0 must freeze the weights exactly.
    g0 = copy.deepcopy(genomes["random"])
    g0.eta = 0.0
    b0 = encode_genomes([g0], brain_cfg)
    s0 = init_state(b0, 2)
    W_before = s0.Wabs.clone()
    for t in range(20):
        s0, _ = brain_tick(b0, s0, None,
                           torch.tensor(rng.normal(0, 0.05, size=(1, 2, n))),
                           contraction="sequential")
    check("eta = 0 freezes |w| bit-exactly", bool(torch.equal(s0.Wabs, W_before)))


# ============================================================
# 7. The one-tick output lag (S2)
# ============================================================

def test_output_lag(brain_cfg, genomes):
    print("\n[7] Output snapshot lags one brain tick (S2)")
    batch = encode_genomes([genomes["random"]], brain_cfg)
    st = init_state(batch, 1)
    st1, snap0 = brain_tick(batch, st, None, None)
    check("snapshot at tick 0 is all-zero (fresh brain)",
          bool((snap0 == 0).all()))
    st2, snap1 = brain_tick(batch, st1, None, None)
    check("snapshot at tick 1 == the activity produced by tick 0",
          bool(torch.equal(snap1, st1.act[..., OUTPUT_SLICE])))
    check("snapshot is NOT the activity produced by the same tick",
          not torch.equal(snap1, st2.act[..., OUTPUT_SLICE])
          or bool(torch.equal(st1.act[..., OUTPUT_SLICE], st2.act[..., OUTPUT_SLICE])))


# ============================================================
# 8. dtype / device
# ============================================================

def test_dtype_device(brain_cfg, genomes, K=16):
    print(f"\n[8] dtype and device ({K} ticks)")
    gs = list(genomes.values())
    P, n, R = len(gs), brain_cfg["n_neurons"], 4
    rng = np.random.default_rng(4)
    sens_np = rng.integers(0, 2, size=(P, R, 5)).astype(np.float64)
    noise_np = rng.normal(0.0, brain_cfg["noise_level"], size=(K, P, R, n))

    def roll(dev, dt, contraction):
        b = encode_genomes(gs, brain_cfg, device=dev, dtype=dt)
        s = init_state(b, R)
        sens = torch.tensor(sens_np, device=dev, dtype=dt)
        for t in range(K):
            s, _ = brain_tick(b, s, sens,
                              torch.tensor(noise_np[t], device=dev, dtype=dt),
                              contraction=contraction)
        return s

    ref = roll("cpu", torch.float64, "sequential")
    f32 = roll("cpu", torch.float32, "sequential")
    check("float32 CPU reproduces float64 activities",
          bool(torch.equal(f32.act.double(), ref.act)))
    check(f"float32 CPU |w| within 1e-6 (max {(f32.Wabs.double()-ref.Wabs).abs().max():.2e})",
          (f32.Wabs.double() - ref.Wabs).abs().max().item() < 1e-6)

    for dev in accelerators():                      # MPS and/or CUDA (Step 8)
        gpu = roll(dev, torch.float32, "einsum")
        check(f"{dev} float32 + einsum reproduces float64 activities",
              bool(torch.equal(gpu.act.cpu().double(), ref.act)))
        d = (gpu.Wabs.cpu().double() - ref.Wabs).abs().max().item()
        check(f"{dev} float32 |w| within 1e-6 (max {d:.2e})", d < 1e-6)
    if not accelerators():
        print("  [SKIP] no GPU available")


def test_exact_tanh():
    print("\n[9] tanh on CPU float64 equals the scalar's math.tanh, bit for bit")
    # torch.tanh differs from math.tanh in the last bit for ~0.35% of float64 values on
    # Apple Silicon and ~30% on Linux x86 (tests/platform_probe.py); that made plastic
    # weights drift by 1 ulp on Linux. brain._tanh uses math.tanh on CPU float64.
    import math
    from mvb_torch.brain import _tanh
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.uniform(-3, 3, 400_000), rng.normal(0, 0.5, 400_000),
                        rng.uniform(-1e-3, 1e-3, 200_000)])
    ref = np.array([math.tanh(v) for v in x])
    got = _tanh(torch.as_tensor(x, dtype=torch.float64)).numpy()
    n_torch = int((torch.tanh(torch.as_tensor(x, dtype=torch.float64)).numpy() != ref).sum())
    check(f"brain._tanh == math.tanh on all {x.size} values (plain torch.tanh differs on "
          f"{n_torch} here)", np.array_equal(got, ref) and n_torch > 0)
    shaped = torch.as_tensor(x[:1210], dtype=torch.float64).view(10, 1, 11, 11)
    check("shape is preserved", _tanh(shaped).shape == shaped.shape)


def main():
    print("=" * 68)
    print("mvb_torch/brain.py test suite")
    print("=" * 68)
    brain_cfg = load_brain_cfg()
    genomes = make_genomes(brain_cfg)

    test_lockstep(brain_cfg, genomes)
    test_transpose(brain_cfg, genomes)
    test_einsum_vs_sequential(brain_cfg, genomes)
    test_run_independence(brain_cfg, genomes)
    test_shapes(brain_cfg, genomes)
    test_plasticity(brain_cfg, genomes)
    test_output_lag(brain_cfg, genomes)
    test_dtype_device(brain_cfg, genomes)
    test_exact_tanh()

    failed = [n for n, ok in _RESULTS if not ok]
    print("\n" + "=" * 68)
    if failed:
        print(f"FAILED {len(failed)}/{len(_RESULTS)}")
        for n in failed:
            print(f"  - {n}")
        return 1
    print(f"OK -- {len(_RESULTS)}/{len(_RESULTS)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
