#!/usr/bin/env python3
"""Unit + integration tests for mvb/predrawn.py (plan_evotorch.md Step 0).

Covers what ea_drift.py cannot: the internals of PredrawnRandomness, and that the
production path is untouched when the feature is off.

Usage
-----
    python -m tests.test_predrawn

Exits 0 on success, 1 on the first failure. No pytest dependency.

Note on scope: this does NOT check that predrawn reproduces the live RNG stream.
It deliberately does not -- integers() returns int(u*k) rather than numpy's integer
algorithm so the tensor port can reproduce it with (u*k).long(). Use
tests/ea_drift.py against a pre-change reference to prove the live path is unchanged.
"""

import os
import sys
import tempfile

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mvb.predrawn import PredrawnBundle, PredrawnRandomness, make_bundle  # noqa: E402


# ============================================================
# Fakes
# ============================================================

class _Neuron:
    def __init__(self, noise_level):
        self.noise_level = noise_level


class _State:
    def __init__(self, levels):
        self.neurons = [_Neuron(x) for x in levels]


class _Worm:
    def __init__(self, ticks=0):
        self.ticks = ticks


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
        return check(f"{name} (wrong exception: {type(e).__name__})", False)
    return check(f"{name} (no exception raised)", False)


def _bundle(n=5, T=4, K=3, grid=(6, 7), events=2):
    return make_bundle(run_seed=1, noise_seed=2, decision_seed=3, max_ticks=T,
                       max_brain_ticks=K, n_neurons=n, grid_shape=grid,
                       n_seed_events=events)


# ============================================================
# 1. Unit tests -- PredrawnRandomness indexing
# ============================================================

def test_unit():
    print("\n[1] PredrawnRandomness indexing")
    n, T, K = 5, 4, 3
    b = _bundle(n, T, K)

    check("bundle shapes",
          b.neuron_noise.shape == (T, K, n)
          and b.decision_uniform.shape == (T,)
          and b.food_uniform.shape == (2, 6, 7))

    # cursor maps 1:1 to neuron id and rolls into the next brain tick
    w = _Worm()
    src = PredrawnRandomness(b).bind(w, _State([0.05] * n))
    got = [src.normal(0.0, 1.0) for _ in range(n * 2 + 2)]
    want = ([b.neuron_noise[0, 0, i] for i in range(n)]
            + [b.neuron_noise[0, 1, i] for i in range(n)]
            + [b.neuron_noise[0, 2, 0], b.neuron_noise[0, 2, 1]])
    check("brain-tick rollover", np.array_equal(got, want))

    # a new world tick restarts decide()'s inner loop at brain tick 0
    w.ticks = 2
    check("world-tick reset", src.normal(0.0, 1.0) == b.neuron_noise[2, 0, 0])

    # scale applied at the use site: loc + scale*v  ==  rng.normal(loc, scale)
    src2 = PredrawnRandomness(b).bind(_Worm(), _State([0.05] * n))
    check("loc + scale*v", src2.normal(0.0, 0.05) == 0.05 * b.neuron_noise[0, 0, 0])

    # mixed noise levels must index by REAL neuron id, not by call order
    src4 = PredrawnRandomness(b).bind(_Worm(), _State([0.0, 0.05, 0.0, 0.05, 0.0]))
    got4 = [src4.normal(0.0, 1.0) for _ in range(3)]
    want4 = [b.neuron_noise[0, 0, 1], b.neuron_noise[0, 0, 3], b.neuron_noise[0, 1, 1]]
    check("mixed noise levels -> real neuron id", np.array_equal(got4, want4))

    # decisions: int(u*k), at most once per world tick
    src5 = PredrawnRandomness(b).bind(_Worm(), _State([0.05] * n))
    check("integers() == int(u*k)", src5.integers(4) == int(b.decision_uniform[0] * 4))
    raises("integers() rejects a second draw in the same tick", RuntimeError,
           lambda: src5.integers(4))

    # food seeding: sequential grids, then a loud failure
    src6 = PredrawnRandomness(b).bind(_Worm(), _State([0.05] * n))
    ok = (np.array_equal(src6.random((6, 7)), b.food_uniform[0])
          and np.array_equal(src6.random((6, 7)), b.food_uniform[1]))
    check("food grids served in order", ok)
    raises("random() fails loudly when events are exhausted", IndexError,
           lambda: src6.random((6, 7)))

    # bounds and misuse
    src7 = PredrawnRandomness(b).bind(_Worm(ticks=99), _State([0.05] * n))
    raises("world tick beyond max_ticks", IndexError, lambda: src7.normal(0.0, 1.0))
    raises("use before bind()", RuntimeError,
           lambda: PredrawnRandomness(b).normal(0.0, 1.0))
    raises("bind() rejects a neuron-count mismatch", ValueError,
           lambda: PredrawnRandomness(b).bind(_Worm(), _State([0.05] * (n + 1))))

    # first brain tick must equal a plain numpy draw
    b2 = _bundle(n=11, T=5, K=32, grid=(51, 51))
    s2 = PredrawnRandomness(b2).bind(_Worm(), _State([0.05] * 11))
    check("first brain tick == default_rng(seed).normal(0, s)",
          np.array_equal([s2.normal(0.0, 0.05) for _ in range(11)],
                         np.random.default_rng(2).normal(0.0, 0.05, size=11)))

    # round-trip
    path = os.path.join(tempfile.mkdtemp(), "bundle.npz")
    b.save(path)
    b3 = PredrawnBundle.load(path)
    check("save/load round-trip",
          np.array_equal(b.neuron_noise, b3.neuron_noise)
          and np.array_equal(b.decision_uniform, b3.decision_uniform)
          and np.array_equal(b.food_uniform, b3.food_uniform))


# ============================================================
# 2. Integration -- null path and determinism
# ============================================================

def test_integration():
    print("\n[2] eval_variant integration")
    from mvb.simulation_API import eval_variant
    from mvb.genome.generate_genome_random import generate_random_genome

    cfg = yaml.safe_load(open("configs/experiments/test_ea.yaml", encoding="utf-8"))
    world_cfg, worm_cfg, brain_cfg = cfg["world"], cfg["worm"], cfg["brain"]
    phases = sorted(cfg["food"], key=lambda p: p["phase_from"])

    gr = generate_random_genome(cfg, 7)
    genome = dict(connection_weights=gr.connection_weights,
                  modulation_spec=gr.modulation_spec,
                  tonic_activations=gr.tonic_activations, eta=gr.eta)

    n_runs, max_ticks = 4, 200
    seeds = [np.array([11, 22, 33, 44], dtype=np.uint32)] * 3

    def run(randomness_cfg):
        return eval_variant(
            0, "plasticity", world_cfg["grid_width"], world_cfg["grid_height"],
            tuple(world_cfg["start_pos"]), True, True, False, phases[0],
            worm_cfg["speed"], worm_cfg["energy_capacity"], worm_cfg["metabolic_rate"],
            worm_cfg["movement_cost"], n_runs, genome, brain_cfg, False, 0, False, 0,
            *seeds, max_ticks, worm_cfg["sensors"]["active"],
            switch_phases=phases[1:], randomness_cfg=randomness_cfg)

    rc = {"enabled": True, "max_brain_ticks": 32}
    live_none = run(None)
    live_off = run({"enabled": False})
    pre_a = run(rc)
    pre_b = run(rc)

    check("randomness_cfg=None == enabled:false",
          np.array_equal(live_none["lifespan_vector"], live_off["lifespan_vector"]))
    check("predrawn lifespans are deterministic",
          np.array_equal(pre_a["lifespan_vector"], pre_b["lifespan_vector"]))
    pt_a, pt_b = pre_a["per_tick_all_runs"], pre_b["per_tick_all_runs"]
    check("predrawn per-tick traces are identical",
          all(np.array_equal(pt_a[k], pt_b[k]) for k in pt_a))
    check("predrawn differs from live (BY DESIGN: integers() path)",
          not np.array_equal(live_none["lifespan_vector"], pre_a["lifespan_vector"]))

    raises("missing max_brain_ticks fails loudly", KeyError,
           lambda: run({"enabled": True}))


def main():
    print("=" * 62)
    print("mvb/predrawn.py test suite")
    print("=" * 62)
    test_unit()
    test_integration()

    failed = [name for name, ok in _RESULTS if not ok]
    print("\n" + "=" * 62)
    if failed:
        print(f"FAILED {len(failed)}/{len(_RESULTS)}")
        for name in failed:
            print(f"  - {name}")
        return 1
    print(f"OK -- {len(_RESULTS)}/{len(_RESULTS)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
