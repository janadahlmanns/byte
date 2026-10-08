#!/usr/bin/env python3
"""torch.compile for the live path (plan Step 8, R5), on every device present.

Compiled code may differ from eager in the last bits (on CUDA it does: fused
multiply-add, a different tanh), which is acceptable because the setting is recorded and
replay must match it (run_batch.check_replay_rounds). What must hold for compiled runs
is everything the user's requirements rest on:

  [1] Philox compiled == eager, bit for bit (integer arithmetic)
  [2] back-to-back identical; identical for every width; identical with compaction on
      and off (so a run does not depend on the batch around it -> single-run replay)
  [3] the number of compiled graphs stops growing after the first generation, although
      the width changes ~every few iterations (no recompile per width)
  [4] reported, not asserted: compiled vs eager
  [5] compile=True really runs the compiled functions (and compile=False does not): on
      CPU and MPS compiled == eager, so no result above would notice a silent eager run

Usage
-----
    python -m tests.test_compile
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch._dynamo.utils as dynamo_utils  # noqa: E402

import mvb_torch.decision as D  # noqa: E402
import mvb_torch.generation as G  # noqa: E402
from mvb_torch import philox  # noqa: E402
from mvb_torch.decision import build_output_spec  # noqa: E402
from mvb_torch.genome_codec import encode_genomes  # noqa: E402
from tests.devices import all_devices  # noqa: E402
from tests.test_generation import genome_pool, load_cfg, seeds_for  # noqa: E402

_RESULTS = []
FIELDS = ("lifespans", "eats", "distance", "final_energy")


def check(name, condition):
    _RESULTS.append((name, bool(condition)))
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}")
    return bool(condition)


def graphs():
    return int(dynamo_utils.counters["stats"]["unique_graphs"])


def test_philox(dev):
    print(f"\n[1] [{dev}] Philox: compiled == eager")
    B = 3000
    seed = torch.randint(0, 2**32, (B, 1), dtype=torch.int64).to(dev)
    tick = torch.randint(0, 500, (B, 1)).to(dev)
    for rounds in (7, 10):
        ze = philox.standard_normals(seed, tick, 22, 11, torch.float32, rounds=rounds)
        zc = philox.compiled_normals(22, 11, torch.float32, rounds)(seed, tick)
        ue = philox.decision_uniforms(seed, tick, torch.float32, rounds=rounds)
        uc = philox.compiled_uniforms(torch.float32, rounds)(seed, tick)
        # The integer part (Philox words -> uniforms) must be exact everywhere. The
        # normals go through log/cos/sin, which a compiler may implement differently
        # (on CPU it does); that is allowed because `compile` is recorded and replay
        # must match it -- so it is reported, not required.
        check(f"[{dev}] {rounds} rounds: decision uniforms bit-identical (integer path)",
              torch.equal(ue, uc))
        d = (ze - zc).abs().max().item()
        print(f"      [{dev}] {rounds} rounds: normals compiled vs eager "
              f"{'identical' if d == 0 else f'max |diff| {d:.1e}'} (reported)")


def test_generations(dev):
    print(f"\n[2-4] [{dev}] Whole generations with compile on")
    G.configure_device(dev)
    cfg = load_cfg(n_runs=20)
    genomes = genome_pool(cfg["brain"])
    seeds = seeds_for(cfg, len(genomes))
    sim = G.sim_config_from_yaml(cfg)
    spec = build_output_spec(cfg["brain"])
    batch = encode_genomes(genomes, cfg["brain"], device=dev, dtype=torch.float32)
    P = len(genomes)

    def run(width, compile=True, compact_below=G.COMPACT_BELOW):
        return G.eval_generation_batch(
            batch, spec, sim, seeds, width=width, mode="live", philox_rounds=10,
            contraction=G.live_contraction(dev), compile=compile,
            compact_below=compact_below)
    try:
        a = run(P * 4)
    except RuntimeError as e:
        check(f"[{dev}] compile works here ({str(e)[:200]})", False)
        return
    g_after_first = graphs()
    b = run(P * 4)
    same = all(torch.equal(getattr(a, k), getattr(b, k)) for k in FIELDS)
    check(f"[{dev}] compiled: {a.lifespans.numel()} runs identical back to back", same)
    widths = {w: run(w) for w in (P, P * 7, P * 20)}
    check(f"[{dev}] compiled: identical for widths {P}, {P * 4}, {P * 7}, {P * 20}",
          all(torch.equal(getattr(r, k), getattr(a, k)) for r in widths.values()
              for k in FIELDS))
    nocomp = run(P * 4, compact_below=0.0)
    check(f"[{dev}] compiled: identical with compaction off ({a.compactions} compactions "
          f"when on)", a.compactions > 0
          and all(torch.equal(getattr(nocomp, k), getattr(a, k)) for k in FIELDS))
    g_end = graphs()
    check(f"[{dev}] no recompile after the first generation: {g_after_first} graphs after "
          f"it, {g_end} after 6 more generations at 5 widths with compaction",
          g_end == g_after_first)
    eager = run(P * 4, compile=False)
    same_e = all(torch.equal(getattr(eager, k), getattr(a, k)) for k in FIELDS)
    diff = int((eager.lifespans != a.lifespans).sum())
    print(f"      [{dev}] compiled vs eager (reported, not required): "
          f"{'identical' if same_e else f'{diff} of {a.lifespans.numel()} lifespans differ'}")


def test_routing(dev):
    print(f"\n[5] [{dev}] compile=True calls the compiled functions; compile=False does not")
    cfg = load_cfg(n_runs=4)
    genomes = genome_pool(cfg["brain"])
    seeds = seeds_for(cfg, len(genomes))
    sim = G.sim_config_from_yaml(cfg)
    spec = build_output_spec(cfg["brain"])
    batch = encode_genomes(genomes, cfg["brain"], device=dev, dtype=torch.float32)
    calls = {"decide step": 0, "Philox normals": 0, "Philox uniforms": 0}

    def counting(factory, name):        # wraps what the factory returns in a call counter
        def make(*a, **k):
            inner = factory(*a, **k)

            def counted(*x, **y):
                calls[name] += 1
                return inner(*x, **y)
            return counted
        return make
    orig = (D.compiled_decide_step, philox.compiled_normals, philox.compiled_uniforms)
    D.compiled_decide_step = counting(orig[0], "decide step")
    philox.compiled_normals = counting(orig[1], "Philox normals")
    philox.compiled_uniforms = counting(orig[2], "Philox uniforms")
    try:
        counts = {}
        for compile in (True, False):
            for k in calls:
                calls[k] = 0
            G.eval_generation_batch(batch, spec, sim, seeds, width=len(genomes) * 4,
                                    mode="live", philox_rounds=10,
                                    contraction=G.live_contraction(dev), compile=compile)
            counts[compile] = dict(calls)
    finally:
        D.compiled_decide_step, philox.compiled_normals, philox.compiled_uniforms = orig
    check(f"[{dev}] compile=True: every compiled function was called {counts[True]}",
          all(v > 0 for v in counts[True].values()))
    check(f"[{dev}] compile=False: none was called {counts[False]}",
          all(v == 0 for v in counts[False].values()))


def main():
    print("=" * 70)
    print("torch.compile on the live path (plan Step 8, R5)")
    print("=" * 70)
    for dev in all_devices():
        test_philox(dev)
        test_generations(dev)
        test_routing(dev)
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
