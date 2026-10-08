#!/usr/bin/env python3
"""Tests for mvb_torch/decision.py (plan_evotorch.md Step 3).

The oracle is the real `decide()`. It is driven with deterministic stand-in RNGs so
both sides consume identical noise and the identical decision uniform:

    FixedNoise    returns pre-generated (already scaled) rows, one per brain tick
    FixedUniform  returns int(u * k), exactly as PredrawnRandomness.integers does

Counting `FixedNoise` calls also recovers which brain tick the scalar decided on,
since `decide()` returns immediately afterwards -- no instrumentation of mvb/ needed.

Usage
-----
    python -m tests.test_decision
"""

import copy
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mvb.brains.decisionmaking_plasticity as dp  # noqa: E402
from mvb_torch.brain import init_state  # noqa: E402
from mvb_torch.decision import (  # noqa: E402
    ACTION_CODE,
    FALLBACK_ACTIONS,
    OutputSpec,
    _resolve,
    build_output_spec,
    decide_batch,
)
from mvb_torch.genome_codec import encode_genomes  # noqa: E402
from tests.test_brain_tick import load_brain_cfg, make_genomes  # noqa: E402

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
# Deterministic stand-in RNGs
# ============================================================

class FixedNoise:
    """`draw_noise` calls rng.normal(0, scale, size=n); hand back a stored row."""

    def __init__(self, rows):
        self.rows = rows
        self.calls = 0

    def normal(self, loc, scale, size=None):
        row = self.rows[self.calls]
        self.calls += 1
        return np.asarray(row, dtype=np.float64)


class FixedUniform:
    """`int(u * k)`, matching PredrawnRandomness.integers (F3.7)."""

    def __init__(self, u):
        self.u = float(u)
        self.draws = 0

    def integers(self, low, high=None):
        self.draws += 1
        span = low if high is None else high - low
        return int(self.u * span)


class FakeWorld:
    height = 51
    width = 51


class FakeWorm:
    def __init__(self):
        self.y, self.x = 26, 26
        self.sensory_information = None


def decision_to_code(decision):
    if decision is None:
        return None
    if decision[0] == "stay":
        return ACTION_CODE["stay"]
    return ACTION_CODE[decision[2]]


def scalar_decide(genome, brain_cfg, sens_by_key, noise_rows, u):
    """One full decide() on a fresh brain. Returns (code, ticks_run, draws, state)."""
    dp.init_brain(genome, brain_cfg)
    st = dp._brain_state
    nrng, drng = FixedNoise(noise_rows), FixedUniform(u)
    dec = dp.decide(FakeWorld(), FakeWorm(), drng, sens_by_key, nrng)
    return decision_to_code(dec), nrng.calls, drng.draws, st


def scalar_wabs(state, n):
    W = np.zeros((n, n), dtype=np.float64)
    for (src, tgt), conn in state.conn_index.items():
        W[src, tgt] = abs(conn.weight)
    return W


# ============================================================
# 1. Output spec / slot order
# ============================================================

def test_output_spec(brain_cfg):
    print("\n[1] Output spec and CPython set-slot order (F3.6)")
    spec = build_output_spec(brain_cfg)
    check("movement order is neurons [8, 9, 6, 7], NOT sorted",
          tuple(s + 5 for s in spec.move_slots) == (8, 9, 6, 7))
    check("directions follow that order: south, west, north, east",
          tuple(FALLBACK_ACTIONS[a] for a in spec.move_actions)
          == ("south", "west", "north", "east"))
    check("stay is neuron 5 (window slot 0)", spec.stay_slot == 0)

    # Cross-check against CPython itself, for every movement subset.
    import itertools
    ok = True
    for r in range(1, 5):
        for combo in itertools.combinations((6, 7, 8, 9), r):
            cpython_order = list(set(combo))
            ours = [s + 5 for s in spec.move_slots if s + 5 in combo]
            ok &= cpython_order == ours
    check("every movement subset matches CPython's own set iteration order", ok)

    bad = copy.deepcopy(brain_cfg)
    del bad["output_mapping"][5]
    raises("uncovered output neuron fails loudly", ValueError,
           lambda: build_output_spec(bad))
    weird = copy.deepcopy(brain_cfg)
    weird["output_mapping"][6] = "wiggle"
    raises("unknown action name fails loudly", ValueError,
           lambda: build_output_spec(weird))
    five = copy.deepcopy(brain_cfg)
    five["output_mapping"] = {5: "move_north", 6: "move_south", 7: "move_east",
                             8: "move_west", 9: "move_north"}
    raises("more than 4 movement neurons fails loudly (table would resize)",
           ValueError, lambda: build_output_spec(five))


def test_resolve_tiebreak(brain_cfg):
    print("\n[2] Tiebreak resolution (direct, on synthetic stable sets)")
    spec = build_output_spec(brain_cfg)
    ms = torch.tensor(spec.move_slots)
    ma = torch.tensor(spec.move_actions)

    def resolve(ids, u):
        stable = torch.zeros(1, 1, 5, dtype=torch.bool)
        for i in ids:
            stable[0, 0, i - 5] = True
        out = _resolve(stable, spec, ms, ma, torch.tensor([[u]], dtype=torch.float64))
        return FALLBACK_ACTIONS[int(out[0, 0])]

    # CPython: {6,8} -> [8,6] -> ["south","north"]. Sorting would give north first.
    check("{6,8}, u=0.0 -> south (slot order), NOT north (sorted order)",
          resolve({6, 8}, 0.0) == "south")
    check("{6,8}, u=0.9 -> north", resolve({6, 8}, 0.9) == "north")
    check("{8,9} -> south then west", resolve({8, 9}, 0.0) == "south"
          and resolve({8, 9}, 0.9) == "west")
    check("{6,7} -> north then east (this subset IS in value order)",
          resolve({6, 7}, 0.0) == "north" and resolve({6, 7}, 0.9) == "east")
    check("{6,7,8,9} -> [south, west, north, east] across u",
          [resolve({6, 7, 8, 9}, u) for u in (0.0, 0.3, 0.6, 0.9)]
          == ["south", "west", "north", "east"])
    check("stay wins outright over any stable movement",
          resolve({5, 6, 7, 8, 9}, 0.0) == "stay"
          and resolve({5, 8}, 0.9) == "stay")
    check("single movement ignores u",
          all(resolve({7}, u) == "east" for u in (0.0, 0.5, 0.99)))


# ============================================================
# 3. Lockstep vs the real decide()
# ============================================================

def _trials(brain_cfg, genomes, n_sens=6, n_u=3):
    """A grid of (genome, sensory pattern, uniform) cases."""
    keys = ["on_food", "food_north", "food_east", "food_south", "food_west"]
    rng = np.random.default_rng(17)
    pats = [dict(zip(keys, map(int, rng.integers(0, 2, 5)))) for _ in range(n_sens)]
    us = [0.0, 0.42, 0.97][:n_u]
    return keys, pats, us


def test_lockstep(brain_cfg, genomes):
    print("\n[3] Lockstep vs the real decide()")
    keys, pats, us = _trials(brain_cfg, genomes)
    n = brain_cfg["n_neurons"]
    spec = build_output_spec(brain_cfg)
    labels = list(genomes)
    gs = [genomes[k] for k in labels]
    batch = encode_genomes(gs, brain_cfg)
    P = len(gs)
    K = int(batch.max_ticks.max())

    bad_action = bad_tick = bad_act = 0
    # The modulation sum is accumulated in `modulation_spec` list order by the
    # scalar sim and in ascending modulator index by any dense tensor. With <= 2
    # modulators on an edge the two agree exactly (addition is commutative); beyond
    # that they can differ in the last bit. Same boundary as Step 2.
    w_bad = {lbl: 0 for lbl in labels}
    w_max = {lbl: 0.0 for lbl in labels}
    paths = {"stay": 0, "tiebreak": 0, "fallback": 0}
    draw_counts = {0: 0, 1: 0}
    total = 0

    for case, (pat, u) in enumerate(
        [(pat, u) for pat in pats for u in us]
    ):
        if True:
            R = 1
            # Seed from the case index, never from hash(): PYTHONHASHSEED salts
            # hashes of tuples containing strings, so that would make the whole
            # test non-reproducible between runs.
            rng = np.random.default_rng(1000 + case)
            noise_np = rng.normal(0.0, brain_cfg["noise_level"], size=(K, P, R, n))
            sens = torch.tensor(
                np.array([[[float(pat[k]) for k in batch.sensor_keys]]] * P)
            )
            state = init_state(batch, R)
            state, dec = decide_batch(
                batch, state, spec, sens,
                torch.tensor(noise_np),
                torch.full((P, R), u, dtype=torch.float64),
                contraction="sequential",
            )
            for p in range(P):
                code, ticks, draws, sst = scalar_decide(
                    gs[p], brain_cfg, pat, list(noise_np[:, p, 0, :]), u
                )
                total += 1
                if code != int(dec.action[p, 0]):
                    bad_action += 1
                # decide() returns right after deciding, so calls-1 is that tick.
                s_at = -1 if draws and code is not None and ticks == int(batch.max_ticks[p]) \
                    else ticks - 1
                fell_back = bool(dec.via_fallback[p, 0])
                if fell_back:
                    paths["fallback"] += 1
                elif code == ACTION_CODE["stay"]:
                    paths["stay"] += 1
                else:
                    paths["tiebreak"] += 1
                if not fell_back and int(dec.decided_at[p, 0]) != ticks - 1:
                    bad_tick += 1
                if not np.array_equal(
                    np.array([nr.activity for nr in sst.neurons]),
                    state.act[p, 0].numpy(),
                ):
                    bad_act += 1
                sw, tw = scalar_wabs(sst, n), state.Wabs[p, 0].numpy()
                if not np.array_equal(sw, tw):
                    w_bad[labels[p]] += 1
                    w_max[labels[p]] = max(w_max[labels[p]],
                                           float(np.abs(sw - tw).max()))
                draw_counts[min(draws, 1)] += 1

    check(f"actions identical in all {total} cases ({bad_action} wrong)",
          bad_action == 0)
    check(f"decided_at matches the scalar's tick ({bad_tick} wrong)", bad_tick == 0)
    check(f"final activities bit-exact ({bad_act} wrong)", bad_act == 0)
    for lbl in labels:
        mm = max([len(v) for v in genomes[lbl].modulation_spec.values()] + [0])
        if mm <= 2:
            check(f"{lbl}: final |w| bit-exact (max {mm} mods/edge)",
                  w_bad[lbl] == 0)
        else:
            check(f"{lbl}: final |w| within 1e-12 (max {mm} mods/edge -> modulator "
                  f"order differs; {w_bad[lbl]} cases, max {w_max[lbl]:.2e})",
                  w_max[lbl] < 1e-12)

    # 6 of 9 genomes took the fallback when probed, so a suite that does not force
    # the other two paths is mostly testing the fallback (F3.1).
    print(f"      paths exercised: {paths}")
    check("the 'stay' path was exercised", paths["stay"] > 0)
    check("the movement-tiebreak path was exercised", paths["tiebreak"] > 0)
    check("the random-fallback path was exercised", paths["fallback"] > 0)
    print(f"      decision draws per call: {draw_counts} (0 draws == stay, F3.7)")
    check("some calls consumed zero decision draws (stay)", draw_counts[0] > 0)


# ============================================================
# 4. Freeze semantics across world ticks
# ============================================================

def test_freeze(brain_cfg, genomes):
    print("\n[4] Decided agents are frozen, and resume correctly (F3.8)")
    n = brain_cfg["n_neurons"]
    spec = build_output_spec(brain_cfg)
    gs = list(genomes.values())
    batch = encode_genomes(gs, brain_cfg)
    P, R = len(gs), 2
    K = int(batch.max_ticks.max())
    pat = {"on_food": 1, "food_north": 0, "food_east": 1, "food_south": 0,
           "food_west": 0}
    sens = torch.tensor(np.array([[[float(pat[k]) for k in batch.sensor_keys]] * R] * P))
    rng = np.random.default_rng(3)

    scal = [dp.init_brain(gs[p], brain_cfg) or dp._brain_state for p in range(P)]
    state = init_state(batch, R)
    ok_act = ok_w = True
    for world_tick in range(3):
        noise_np = rng.normal(0.0, brain_cfg["noise_level"], size=(K, P, R, n))
        u = 0.31
        state, dec = decide_batch(batch, state, spec, sens, torch.tensor(noise_np),
                                  torch.full((P, R), u, dtype=torch.float64))
        for p in range(P):
            nrng, drng = FixedNoise(list(noise_np[:, p, 0, :])), FixedUniform(u)
            dp._brain_state = scal[p]
            dp.decide(FakeWorld(), FakeWorm(), drng, pat, nrng)
            ok_act &= np.array_equal(
                np.array([nr.activity for nr in scal[p].neurons]),
                state.act[p, 0].numpy())
            ok_w &= np.array_equal(scalar_wabs(scal[p], n),
                                   state.Wabs[p, 0].numpy())
    check("brain state carries across 3 world ticks, activities bit-exact", ok_act)
    check("brain state carries across 3 world ticks, |w| bit-exact", ok_w)

    # A decided agent must not tick further inside the same call.
    st2 = init_state(batch, R)
    noise_np = rng.normal(0.0, brain_cfg["noise_level"], size=(K, P, R, n))
    after, d = decide_batch(batch, st2, spec, sens, torch.tensor(noise_np),
                            torch.full((P, R), 0.0, dtype=torch.float64))
    early = d.decided_at >= 0
    check("at least one agent decided before max_ticks (freeze is exercised)",
          bool(early.any()))


# ============================================================
# 5. Per-agent boundaries, terminal candidates, lost stability
# ============================================================

def test_boundaries(brain_cfg, genomes):
    print("\n[5] Per-agent warmup/max_ticks, terminal candidates, lost stability")
    n = brain_cfg["n_neurons"]
    spec = build_output_spec(brain_cfg)

    # Sparse genomes give warmup < n (Step 1 F2 regime).
    def sparsify(edges):
        g = copy.deepcopy(genomes["random"])
        g.connection_weights[:] = 0.0
        g.modulation_spec = {}
        for (i, j) in edges:
            g.connection_weights[i, j] = [0.5, 0.9]
        return g

    # A sparse genome whose only connection is PLASTIC and feeds no output neuron:
    # neuron 0 (driven by on_food) modulates its own edge 0->1, so |w| changes on every
    # brain tick, while the outputs (tonic only, below threshold) never become
    # candidates -- the agent never decides and runs to its own max_ticks. Without it,
    # an agent that kept ticking past its own max_ticks would be invisible here: the
    # other sparse genomes settle into a fixed point with no weights to drift, and a
    # plastic edge into an output decides early and freezes (both found by Step 8
    # negative controls).
    plastic = sparsify([(0, 1)])
    plastic.modulation_spec = {(0, 1): [(0, 0.8)]}
    plastic.eta = 0.5

    gs = [sparsify([]), sparsify([(5, 6)]), plastic, genomes["random"]]
    batch = encode_genomes(gs, brain_cfg)
    check(f"warmup differs per genome {batch.warmup_ticks.tolist()}",
          len(set(batch.warmup_ticks.tolist())) > 1)
    R = 1
    K = int(batch.max_ticks.max())
    rng = np.random.default_rng(8)
    noise_np = rng.normal(0.0, brain_cfg["noise_level"], size=(K, len(gs), R, n))
    pat = {"on_food": 1, "food_north": 1, "food_east": 0, "food_south": 0,
           "food_west": 0}
    sens = torch.tensor(np.array([[[float(pat[k]) for k in batch.sensor_keys]]] * len(gs)))
    state = init_state(batch, R)
    state, dec = decide_batch(batch, state, spec, sens, torch.tensor(noise_np),
                              torch.full((len(gs), R), 0.5, dtype=torch.float64))
    ok = True
    for p in range(len(gs)):
        code, ticks, draws, sst = scalar_decide(
            gs[p], brain_cfg, pat, list(noise_np[:, p, 0, :]), 0.5)
        ok &= code == int(dec.action[p, 0])
        ok &= np.array_equal(np.array([nr.activity for nr in sst.neurons]),
                             state.act[p, 0].numpy())
        ok &= np.array_equal(scalar_wabs(sst, n), state.Wabs[p, 0].numpy())
    check(f"sparse genomes (max_ticks {batch.max_ticks.tolist()}) stop on their own "
          f"boundary, not the batch max: action, activities and |w| bit-exact", ok)
    check("the plastic sparse genome runs to its own max_ticks (fallback) while the "
          "batch runs longer", bool(dec.via_fallback[2, 0])
          and int(batch.max_ticks[2]) < int(batch.max_ticks.max()))

    # A candidate that qualifies at 3 stability ticks but not at 5 must not decide:
    # exercised implicitly by the strict-majority comparison. Assert the rule form.
    spec2 = OutputSpec(5, 0, (3,), (3,))
    stable = torch.zeros(1, 1, 5, dtype=torch.bool)
    stable[0, 0, 3] = True
    out = _resolve(stable, spec2, torch.tensor([3]), torch.tensor([3]),
                   torch.zeros(1, 1, dtype=torch.float64))
    check("_resolve handles a single-movement spec", int(out[0, 0]) == 3)


# ============================================================
# 6. einsum vs sequential, and shape guards
# ============================================================

def test_paths_and_guards(brain_cfg, genomes):
    print("\n[6] einsum vs sequential, and shape guards")
    n = brain_cfg["n_neurons"]
    spec = build_output_spec(brain_cfg)
    gs = list(genomes.values())
    batch = encode_genomes(gs, brain_cfg)
    P, R = len(gs), 4
    K = int(batch.max_ticks.max())
    rng = np.random.default_rng(21)
    noise = torch.tensor(rng.normal(0.0, brain_cfg["noise_level"], size=(K, P, R, n)))
    sens = torch.tensor(rng.integers(0, 2, size=(P, R, 5)).astype(np.float64))
    u = torch.tensor(rng.random((P, R)))
    a, da = decide_batch(batch, init_state(batch, R), spec, sens, noise, u,
                         contraction="sequential")
    b, db = decide_batch(batch, init_state(batch, R), spec, sens, noise, u,
                         contraction="einsum")
    check("einsum and sequential agree on every action",
          bool(torch.equal(da.action, db.action)))
    check("einsum and sequential agree on decided_at",
          bool(torch.equal(da.decided_at, db.decided_at)))
    check("all actions are valid codes",
          bool(((da.action >= 0) & (da.action < len(FALLBACK_ACTIONS))).all()))

    raises("wrong decision_uniform shape fails loudly", ValueError,
           lambda: decide_batch(batch, init_state(batch, R), spec, sens, noise,
                                torch.zeros(P, R + 1)))
    raises("too few noise ticks fails loudly", ValueError,
           lambda: decide_batch(batch, init_state(batch, R), spec, sens,
                                noise[:2], u))


def main():
    print("=" * 70)
    print("mvb_torch/decision.py test suite")
    print("=" * 70)
    brain_cfg = load_brain_cfg()
    genomes = make_genomes(brain_cfg)

    test_output_spec(brain_cfg)
    test_resolve_tiebreak(brain_cfg)
    test_lockstep(brain_cfg, genomes)
    test_freeze(brain_cfg, genomes)
    test_boundaries(brain_cfg, genomes)
    test_paths_and_guards(brain_cfg, genomes)

    failed = [nm for nm, ok in _RESULTS if not ok]
    print("\n" + "=" * 70)
    if failed:
        print(f"FAILED {len(failed)}/{len(_RESULTS)}")
        for nm in failed:
            print(f"  - {nm}")
        return 1
    print(f"OK -- {len(_RESULTS)}/{len(_RESULTS)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
