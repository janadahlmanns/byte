#!/usr/bin/env python3
"""torch.compile for the live path (plan Step 8, R5), on every device present.

Compiled code may differ from eager in the last bits (on CUDA it does: fused
multiply-add, a different tanh), which is acceptable because the setting is recorded and
replay must match it (run_batch.check_replay_rounds). What must hold for compiled runs
is everything the user's requirements rest on:

  [1] Philox compiled == eager: the words and decision uniforms bit for bit (integer
      arithmetic), the normals within 1e-5 (Box-Muller's log/cos/sin may round
      differently). Until 2026-10-10 the normals were only reported, which let int64
      overflow in the compiled x86 CPU code go unnoticed (plan Step 9).
  [2] back-to-back identical; identical for every width; identical with compaction on
      and off (so a run does not depend on the batch around it -> single-run replay)
  [3] the number of compiled graphs stops growing after the first generation, although
      the width changes ~every few iterations (no recompile per width)
  [4] reported, not asserted: compiled vs eager
  [5] compile=True really runs the compiled functions (and compile=False does not): on
      CPU and MPS compiled == eager, so no result above would notice a silent eager run
  [6] a long evolving run (plan Step 13c): 12 generations whose modulation lists grow
      until L = 8. No width 1, a new graph only for a new L, and torch.compile set to
      FAIL at its recompile limit instead of silently running eager
  [7] the compile warm-up (plan Step 18a), over an evolving population: every new graph
      appears in the one-iteration warm-up at COMPILE_WARMUP_WIDTH and none in the real
      loop, and results with the warm-up equal those without it
  [8] the production compile-cache folder (plan Step 18b), in subprocesses with a
      temporary base folder: a compiled run_ea writes only into <base>/byte-w512-v1, a
      second run reuses it with identical results, and switching after something was
      compiled raises

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


def _words(rounds, K=22):
    """The Philox words exactly as `standard_normals` lays them out (same counters, same
    expand/stack). With the old direct product, x86 CPU code compiled from THIS layout
    came out wrong, while a simpler layout happened to compile correctly (laptop
    negative control, 2026-10-10): undefined behaviour depends on the generated code."""
    nb = philox.blocks_per_brain_tick(11)

    def f(seed, tick):
        dev, B = seed.device, seed.shape[0]
        k = torch.arange(K, dtype=torch.int64, device=dev).view(K, 1, 1)
        blk = torch.arange(nb, dtype=torch.int64, device=dev).view(1, 1, nb)
        shape = (K, B, nb)
        c0 = tick.view(1, B, 1).expand(shape)
        c1 = (k * nb + blk).expand(shape)
        zero = torch.zeros(shape, dtype=torch.int64, device=dev)
        c3 = torch.full(shape, philox.STREAM_NOISE, dtype=torch.int64, device=dev)
        k0 = seed.view(1, B, 1).expand(shape)
        return torch.stack(philox.philox4x32(c0, c1, zero, c3, k0, zero, rounds=rounds),
                           dim=-1)
    return f


def test_philox(dev):
    print(f"\n[1] [{dev}] Philox: compiled == eager")
    B = 3000
    seed = torch.randint(0, 2**32, (B, 1), dtype=torch.int64).to(dev)
    tick = torch.randint(0, 500, (B, 1)).to(dev)
    for rounds in (7, 10):
        we = _words(rounds)(seed, tick)
        wc = torch.compile(_words(rounds))(seed, tick)
        check(f"[{dev}] {rounds} rounds: Philox words bit-identical "
              f"({int((we != wc).sum())} of {we.numel()} differ)", torch.equal(we, wc))
        ze = philox.standard_normals(seed, tick, 22, 11, torch.float32, rounds=rounds)
        zc = philox.compiled_normals(22, 11, torch.float32, rounds)(seed, tick)
        ue = philox.decision_uniforms(seed, tick, torch.float32, rounds=rounds)
        uc = philox.compiled_uniforms(torch.float32, rounds)(seed, tick)
        # The integer part (Philox words -> uniforms) must be exact everywhere. The
        # normals go through log/cos/sin, which a compiler may round differently (on
        # the arm CPU by up to 4.8e-7) -- allowed, because `compile` is recorded and
        # replay must match it; anything larger means different numbers.
        check(f"[{dev}] {rounds} rounds: decision uniforms bit-identical (integer path)",
              torch.equal(ue, uc))
        d = (ze - zc).abs().max().item()
        check(f"[{dev}] {rounds} rounds: normals compiled vs eager within 1e-5 "
              f"({'identical' if d == 0 else f'max |diff| {d:.1e}'})", d <= 1e-5)


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


def test_long_run(dev):
    print(f"\n[6] [{dev}] 12 evolving generations: bounded recompiles, no eager fallback")
    import numpy as np
    import yaml
    from mvb.genome.generate_genome_lookup_hard import generate_lookup_hard_genome
    from mvb.genome.generate_genome_mutate_simple import generate_genome_mutate_simple
    G.configure_device(dev)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = yaml.safe_load(open(os.path.join(root, "configs/experiments/ea_from_lookup_tensor_cuda.yaml")))
    R, P = 8, 60
    cfg["experiment"]["n_runs"] = R
    rng = np.random.default_rng(0)
    pop = [generate_lookup_hard_genome(cfg, rng_seed=i) for i in range(P)]
    sim, spec = G.sim_config_from_yaml(cfg), build_output_spec(cfg["brain"])
    added, Ls, finals = [], [], []
    try:
        for gen in range(12):
            b = encode_genomes(pop, cfg["brain"], device=dev, dtype=torch.float32,
                               dense_modulation=False)
            seeds = G.draw_seeds(*G.make_simulation_rngs(gen + 1), P, R)
            g0 = graphs()
            r = G.eval_generation_batch(b, spec, sim, seeds, width=P * R, mode="live",
                                        philox_rounds=10, contraction="sequential",
                                        compile=True)
            added.append(graphs() - g0)
            Ls.append(len(b.mod_k))
            finals.append(r.final_width)
            order = np.argsort(-r.lifespans.float().mean(1).cpu().numpy())
            elites = [pop[i] for i in order[:6]]
            pop = elites + [generate_genome_mutate_simple(elites[i % 6], 0.4, 0.2, rng)
                            for i in range(P - 6)]
    except RuntimeError as e:
        check(f"[{dev}] the run completes without hitting the recompile limit "
              f"({str(e)[:160]})", False)
        return
    print(f"      L per generation {Ls}; new graphs {added}; final widths {finals}")
    new_L = [i for i in range(1, len(Ls)) if Ls[i] not in Ls[:i]]
    check(f"[{dev}] compaction never leaves width 1 (final widths {sorted(set(finals))})",
          min(finals) >= 2)
    check(f"[{dev}] after generation 0, a graph is added only in a generation with a new "
          f"L (generations {new_L}), at most one each: {added[1:]}",
          all(a == 0 for i, a in enumerate(added) if i and i not in new_L)
          and all(added[i] <= 1 for i in new_L) and max(Ls) == 8)
    check(f"[{dev}] {sum(added)} graphs in all, below dynamo's recompile limit of "
          f"{torch._dynamo.config.recompile_limit}, and hitting it would raise "
          f"(fail_on_recompile_limit_hit = {torch._dynamo.config.fail_on_recompile_limit_hit})",
          sum(added) < torch._dynamo.config.recompile_limit
          and torch._dynamo.config.fail_on_recompile_limit_hit is True)


def test_warmup(dev):
    print(f"\n[7] [{dev}] Compile warm-up: graphs appear in the warm-up, results unchanged")
    import numpy as np
    import yaml
    from mvb.genome.generate_genome_lookup_hard import generate_lookup_hard_genome
    from mvb.genome.generate_genome_mutate_simple import generate_genome_mutate_simple
    G.configure_device(dev)
    torch._dynamo.reset()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = yaml.safe_load(open(os.path.join(root, "configs/experiments/ea_from_lookup_tensor_cuda.yaml")))
    R, P = 12, 60                     # P*R = 720 > COMPILE_WARMUP_WIDTH: a real warm-up width
    cfg["experiment"]["n_runs"] = R
    rng = np.random.default_rng(1)
    pop = [generate_lookup_hard_genome(cfg, rng_seed=i) for i in range(P)]
    sim, spec = G.sim_config_from_yaml(cfg), build_output_spec(cfg["brain"])
    # `sequential` with the sparse lists, as on CUDA: only there does a new list layout
    # need a new graph (einsum uses the dense Mod), which is what the warm-up must catch.
    kw = dict(mode="live", philox_rounds=10, contraction="sequential", compile=True)
    in_warmup, in_loop, Ls, same, auto = [], [], [], [], []
    real = G.eval_generation_batch

    def spy(*a, **k):                      # records the warm-up calls a normal call makes
        if k.get("_warmup_run"):
            auto.append((k["width"], k["compile"]))
        return real(*a, **k)
    for gen in range(6):
        b = encode_genomes(pop, cfg["brain"], device=dev, dtype=torch.float32,
                           dense_modulation=False)
        seeds = G.draw_seeds(*G.make_simulation_rngs(gen + 1), P, R)
        g0 = graphs()
        G.eval_generation_batch(b, spec, sim, seeds, width=P * R, _warmup_run=True, **kw)
        g1 = graphs()
        without = G.eval_generation_batch(b, spec, sim, seeds, width=P * R,
                                          _compile_warmup=False, **kw)
        g2 = graphs()
        n_auto = len(auto)
        G.eval_generation_batch = spy
        try:
            with_ = G.eval_generation_batch(b, spec, sim, seeds, width=P * R, **kw)
        finally:
            G.eval_generation_batch = real
        auto_this = auto[n_auto:]
        if auto_this != [(min(G.COMPILE_WARMUP_WIDTH, P * R), True)]:
            auto.append(("bad", gen, auto_this))
        in_warmup.append(g1 - g0)
        in_loop.append(g2 - g1)
        Ls.append(len(b.mod_k))
        same.append(all(torch.equal(getattr(with_, k), getattr(without, k)) for k in FIELDS))
        order = np.argsort(-with_.lifespans.float().mean(1).cpu().numpy())
        elites = [pop[i] for i in order[:6]]
        pop = elites + [generate_genome_mutate_simple(elites[i % 6], 0.4, 0.2, rng)
                        for i in range(P - 6)]
    print(f"      L per generation {Ls}; new graphs in the warm-up {in_warmup}, in the loop "
          f"{in_loop}")
    check(f"[{dev}] the list layout changes during the test (L {sorted(set(Ls))}), so new "
          f"graphs are needed after generation 0", len(set(Ls)) > 1)
    new_L = [i for i in range(1, len(Ls)) if Ls[i] not in Ls[:i]]
    check(f"[{dev}] every new graph appears in the warm-up (generation 0 and the new-L "
          f"generations {new_L}), none in the real loop",
          all(x == 0 for x in in_loop) and in_warmup[0] > 0
          and all(in_warmup[i] > 0 for i in new_L)
          and all(in_warmup[i] == 0 for i in range(1, len(Ls)) if i not in new_L))
    check(f"[{dev}] results with the warm-up equal results without it, every generation",
          all(same))
    check(f"[{dev}] every normal compiled generation ran its own warm-up at width "
          f"{min(G.COMPILE_WARMUP_WIDTH, P * R)} ({len(auto)} warm-ups for 6 generations)",
          len(auto) == 6 and all(x[0] != "bad" for x in auto))
    check(f"[{dev}] warm-up width {G.COMPILE_WARMUP_WIDTH} is below this test's P*R = {P * R}",
          G.COMPILE_WARMUP_WIDTH < P * R)


def test_cache_folder():
    print("\n[8] The production compile-cache folder (subprocesses, cpu)")
    import glob
    import subprocess
    import tempfile
    import yaml
    from mvb_torch.compile_cache import COMPILE_CACHE_VERSION
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with tempfile.TemporaryDirectory(prefix="test_cache_") as tmp:
        base = os.path.join(tmp, "inductor_base")
        cfg = yaml.safe_load(open(os.path.join(root, "configs/experiments/test_ea_predrawn.yaml")))
        del cfg["experiment"]["predrawn_randomness"]
        cfg["experiment"]["evolutionary_algorithm"]["num_generations"] = 2
        cfg["experiment"]["evaluator"] = {"backend": "tensor", "device": "cpu",
                                          "dtype": "float32", "philox_rounds": 10,
                                          "compile": True}
        env = dict(os.environ, TORCHINDUCTOR_CACHE_DIR=base)
        env.pop("TRITON_CACHE_DIR", None)
        outs, logs = [], []
        for i in range(2):
            out = os.path.join(tmp, f"out{i}")
            os.makedirs(out)
            cfg["experiment"]["output_folder"] = out + "/"
            path = os.path.join(tmp, f"cfg{i}.yaml")
            with open(path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            p = subprocess.run([sys.executable, "-m", "simulate.run_ea", "--config", path],
                               input="n\n", text=True, capture_output=True, cwd=root,
                               env=env, timeout=3600)
            logs.append(p.stdout + p.stderr)
            h5 = glob.glob(os.path.join(out, "*.h5"))
            outs.append(h5[0] if p.returncode == 0 and len(h5) == 1 else None)
        if not check("two compiled run_ea runs completed", all(outs)):
            print(logs[-1][-2000:])
            return
        target = os.path.join(base, COMPILE_CACHE_VERSION)
        check(f"the runs report the versioned folder (compile_cache={target})",
              all(f"compile_cache={target}" in lg for lg in logs))
        check(f"the base folder holds only {COMPILE_CACHE_VERSION}/ ({sorted(os.listdir(base))})",
              sorted(os.listdir(base)) == [COMPILE_CACHE_VERSION])
        check("compiled modules were written into it",
              sum(len(fs) for _, _, fs in os.walk(target)) > 0)
        from tests.test_ea_tensor import diff_h5
        problems, n = diff_h5(outs[0], outs[1])
        check(f"the second run (cache reused) is identical to the first ({n} datasets, "
              f"{len(problems)} differences)", not problems)
        import re
        t = [float(re.findall(r"genomes x \d+ runs in ([\d.]+)s", lg)[0]) for lg in logs]
        check(f"generation 0 is faster with the cache reused ({t[0]:.1f} s -> {t[1]:.1f} s)",
              t[1] < t[0])
        code = ("import torch; torch.compile(lambda x: x + 1)(torch.ones(3)); "
                "from mvb_torch.compile_cache import use_versioned_compile_cache as u; u()")
        p = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True,
                           cwd=root, env=env, timeout=600)
        check("switching after something was compiled raises (no mixed folders)",
              p.returncode != 0 and "needs its own folder" in p.stderr)


def main():
    print("=" * 70)
    print("torch.compile on the live path (plan Step 8, R5)")
    print("=" * 70)
    for dev in all_devices():
        test_philox(dev)
        test_generations(dev)
        test_routing(dev)
        test_long_run(dev)
        test_warmup(dev)
    test_cache_folder()
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
