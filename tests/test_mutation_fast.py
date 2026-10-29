#!/usr/bin/env python3
"""The faster mutation produces exactly what the old one produced (plan Step 21).

mvb/genome/generate_genome_mutate_simple.py now rebuilds its two helper lists only when
an action may have changed them, instead of before every action. tests/ref_mutate_simple.py
is the frozen old version. For every child, everything must be identical, including the
RNG's full state afterwards (it feeds every later child):

  [1] chained generations (children of children, structure drifting far from
      generation 0) for n = 11, 20, 30, 50, random and lookup genomes, several mutation
      rates and strengths: connection_weights bytes, modulation_spec (key order, list
      order, values, value types), tonic bytes, eta, RNG state
  [2] edge cases forced: no connections, no modulations, every connection present, a
      weight driven to exactly 0.0, a modulated connection moved onto an existing
      modulated key (the `extend` branch)
  [3] speed: old vs new per child at n = 50 (reported, and new must be faster)

Usage
-----
    python -m tests.test_mutation_fast
"""

import copy
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mvb.genome.generate_genome_lookup_hard import generate_lookup_hard_genome  # noqa: E402
from mvb.genome.generate_genome_mutate_simple import generate_genome_mutate_simple as new  # noqa: E402
from mvb.genome.generate_genome_random import GenomeRandomResult, generate_random_genome  # noqa: E402
from tests.ref_mutate_simple import generate_genome_mutate_simple as old  # noqa: E402

_RESULTS = []


def check(name, condition):
    _RESULTS.append((name, bool(condition)))
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}")
    return bool(condition)


def cfg_with(n):
    cfg = yaml.safe_load((ROOT / "configs/experiments/ea_from_random_large_tensor_cuda.yaml").read_text())
    cfg["brain"]["n_neurons"] = n
    return cfg


def same_genome(a, b):
    """Exact equality, including dict and list order and the types inside."""
    if a.connection_weights.dtype != b.connection_weights.dtype \
            or a.connection_weights.tobytes() != b.connection_weights.tobytes():
        return "connection_weights"
    if list(a.modulation_spec) != list(b.modulation_spec):
        return "modulation key order"
    for k in a.modulation_spec:
        la, lb = a.modulation_spec[k], b.modulation_spec[k]
        if la != lb or [tuple(map(type, x)) for x in la] != [tuple(map(type, x)) for x in lb]:
            return f"modulation list at {k}"
        if tuple(map(type, k)) != tuple(map(type, next(kk for kk in b.modulation_spec if kk == k))):
            return f"key types at {k}"
    if a.tonic_activations.tobytes() != b.tonic_activations.tobytes():
        return "tonic_activations"
    if a.eta != b.eta or type(a.eta) is not type(b.eta):
        return "eta"
    return None


def both(genome, rate, strength, seed_state):
    """Mutate with old and new from the same RNG state -> (child_old, child_new, problem)."""
    r_old, r_new = np.random.default_rng(), np.random.default_rng()
    r_old.bit_generator.state = copy.deepcopy(seed_state)
    r_new.bit_generator.state = copy.deepcopy(seed_state)
    a = old(genome, rate, strength, r_old)
    try:
        b = new(genome, rate, strength, r_new)
    except Exception as e:  # noqa: BLE001 -- the old one did not raise here: a difference
        return a, None, f"new raised {type(e).__name__}: {e}", r_old.bit_generator.state
    problem = same_genome(a, b)
    if problem is None and r_old.bit_generator.state != r_new.bit_generator.state:
        problem = "RNG state afterwards"
    return a, b, problem, r_old.bit_generator.state


def test_chains():
    print("\n[1] Chained generations: new child == old child, RNG state included")
    for n in (11, 20, 30, 50):
        cfg = cfg_with(n)
        for kind in ("random", "lookup"):
            if kind == "lookup" and n != 11:
                continue                          # the lookup genome is defined for 11 neurons
            make = generate_random_genome if kind == "random" else generate_lookup_hard_genome
            children, bad = 0, []
            for rate, strength, chain, width in ((0.1, 0.2, 12, 8), (0.3, 0.5, 8, 6), (0.02, 0.05, 6, 6)):
                if n == 50 and rate > 0.1:
                    chain, width = 4, 3           # keep the old (slow) reference affordable
                state = np.random.default_rng(1000 + n).bit_generator.state
                pop = [make(cfg, rng_seed=i) for i in range(width)]
                for _ in range(chain):
                    nxt = []
                    for g in pop:
                        a, b, problem, state = both(g, rate, strength, state)
                        children += 1
                        if problem:
                            bad.append((rate, problem))
                        nxt.append(a)
                    pop = nxt
            check(f"n = {n}, {kind}: {children} children identical ({bad[:2] or 'no difference'})", not bad)


def blank(n, conn):
    return GenomeRandomResult(params=None, connection_weights=conn,
                              modulation_spec={}, tonic_activations=np.full(n, 0.5, np.float32),
                              eta=0.5)


def test_edges():
    print("\n[2] Edge cases")
    n = 6
    rng_state = np.random.default_rng(5).bit_generator.state
    empty = blank(n, np.zeros((n, n, 2), np.float32))
    full = blank(n, np.stack([np.full((n, n), 0.5, np.float32), np.full((n, n), 0.9, np.float32)], -1))
    nomod = blank(n, np.where(np.random.default_rng(1).random((n, n, 1)) < 0.4,
                              np.array([0.3, 0.8], np.float32), np.float32(0)).astype(np.float32))
    for label, g in (("no connections", empty), ("every connection present", full),
                     ("connections, no modulations", nomod)):
        problems = []
        st = rng_state
        for _ in range(40):
            a, _, p, st = both(g, 0.8, 0.3, st)
            problems += [p] if p else []
            g = a
        check(f"{label}: 40 chained children identical ({problems[:1] or 'no difference'})", not problems)

    # A modulated connection moved onto another modulated key (the `extend` branch), and a
    # weight that reaches exactly 0.0 (a weight of -0.0 + noise clipped can't be forced, so
    # use one whose clip boundary is 0: a weight change on a value pushed to the clip is
    # not 0; instead hit the exact-zero branch by monkeypatching the normal draw).
    w = np.zeros((n, n, 2), np.float32)
    w[0, 1] = [0.4, 0.9]
    w[0, 2] = [-0.4, 0.9]
    g = GenomeRandomResult(params=None, connection_weights=w,
                           modulation_spec={(0, 1): [(3, 0.5)], (0, 2): [(4, -0.5)]},
                           tonic_activations=np.full(n, 0.5, np.float32), eta=0.5)
    problems, st, merged = [], np.random.default_rng(9).bit_generator.state, False
    for _ in range(300):
        a, _, p, st = both(g, 0.1, 0.3, st)
        problems += [p] if p else []
        merged |= any(len(v) > 1 for v in a.modulation_spec.values())
    check(f"two modulated connections, 300 children: identical, and the extend branch was "
          f"reached ({merged})", not problems and merged)

    class ZeroNormal:
        """An RNG whose normal() returns exactly minus the weight: forces weight == 0.0."""

        def __init__(self, rng, target):
            self.rng, self.target = rng, target

        def integers(self, *a, **k):
            return self.rng.integers(*a, **k)

        def uniform(self, *a, **k):
            return self.rng.uniform(*a, **k)

        def normal(self, *a, **k):
            self.rng.normal(*a, **k)
            return -self.target

    w = np.zeros((n, n, 2), np.float32)
    w[1, 2] = [0.25, 0.9]
    w[2, 3] = [0.25, 0.9]
    g = GenomeRandomResult(params=None, connection_weights=w, modulation_spec={(1, 2): [(0, 0.3)]},
                           tonic_activations=np.full(n, 0.5, np.float32), eta=0.5)
    problems, zero_seen = [], False
    for seed in range(200):
        ra, rb = np.random.default_rng(seed), np.random.default_rng(seed)
        a = old(g, 0.4, 0.3, ZeroNormal(ra, 0.25))
        try:
            b = new(g, 0.4, 0.3, ZeroNormal(rb, 0.25))
            p = same_genome(a, b) or (None if ra.bit_generator.state == rb.bit_generator.state else "RNG")
        except Exception as e:  # noqa: BLE001
            p = f"new raised {type(e).__name__}"
        problems += [p] if p else []
        zero_seen |= int(np.count_nonzero(a.connection_weights[..., 0])) < 2
    check(f"a weight driven to exactly 0.0 (200 seeds, reached: {zero_seen}): identical "
          f"({problems[:1] or 'no difference'})", not problems and zero_seen)


def test_speed():
    print("\n[3] Speed at n = 50, mutation_rate 0.1 (250 actions per child)")
    cfg = cfg_with(50)
    g = new(generate_random_genome(cfg, rng_seed=3), 0.1, 0.2, np.random.default_rng(0))
    t = {}
    for name, fn in (("old", old), ("new", new)):
        r = np.random.default_rng(4)
        t0 = time.perf_counter()
        for _ in range(20):
            fn(g, 0.1, 0.2, r)
        t[name] = (time.perf_counter() - t0) / 20
    print(f"      old {1e3 * t['old']:.1f} ms per child, new {1e3 * t['new']:.1f} ms "
          f"({t['old'] / t['new']:.1f}x); 950 children: {950 * t['old']:.1f} s -> {950 * t['new']:.1f} s")
    check("the new mutation is faster", t["new"] < t["old"])


def main():
    test_chains()
    test_edges()
    test_speed()
    n_fail = sum(1 for _, ok in _RESULTS if not ok)
    if n_fail:
        print(f"\nFAILED -- {n_fail} of {len(_RESULTS)} checks failed")
        return 1
    print(f"\nOK -- {len(_RESULTS)}/{len(_RESULTS)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
