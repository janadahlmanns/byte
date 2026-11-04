#!/usr/bin/env python3
"""Parallel genome creation and per-child mutation (plan Step 27).

  [1] per_child: the same children for any number of workers (serial, 2, 7, all
      allowed cores), n = 11 and 50, random and lookup parents; the seeds are drawn in
      child order, once
  [2] generation 0 in parallel == the serial loop (both mutation_rng values use it)
  [3] whole EA runs (tensor CPU, small):
        per_child twice -> identical; per_child serial vs parallel -> identical;
        shared == the committed code (HEAD run_ea, no Step 27)
  [4] the config key: required, explicit values only, every shipped EA config has it
  [5] a mutation that raises in a worker fails loudly, naming the child
  [6] no torch in the workers, even when the parent imported it
  [7] old references: a reference without the key matches 'shared' and only 'shared'
  [8] speed at n = 50, 950 children: serial vs parallel (reported)

Usage
-----
    python -m tests.test_mutation_parallel
"""

import copy
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import simulate.parallel_genomes as PG  # noqa: E402
import simulate.run_ea as EA  # noqa: E402
import simulate.run_selected as RS  # noqa: E402
from mvb.genome.generate_genome_lookup_hard import generate_lookup_hard_genome  # noqa: E402
from mvb.genome.generate_genome_mutate_simple import generate_genome_mutate_simple as mutate  # noqa: E402
from mvb.genome.generate_genome_random import generate_random_genome  # noqa: E402
from mvb.simulation_helper_functions import allowed_cpu_count  # noqa: E402
from tests.test_ea_tensor import diff_h5  # noqa: E402
from tests.test_mutation_fast import same_genome  # noqa: E402

_RESULTS = []


def check(name, condition):
    _RESULTS.append((name, bool(condition)))
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}")
    return bool(condition)


def cfg_with(n):
    cfg = yaml.safe_load((ROOT / "configs/experiments/ea_from_random_large_tensor_cuda.yaml").read_text())
    cfg["brain"]["n_neurons"] = n
    return cfg


def all_same(a, b):
    return len(a) == len(b) and all(same_genome(x, y) is None for x, y in zip(a, b))


def broken_mutation(genome, rate, strength, rng):
    """A mutation that fails for one particular draw (test [5])."""
    if rng.integers(0, 1000) == 7:
        raise ValueError("boom")
    return genome


# ============================================================

def test_per_child():
    print("\n[1] per_child: the same children for any number of workers")
    counts = [9] * 9 + [14]                      # 95 children from 10 elites (>= MIN_PARALLEL)
    for n, kind in ((11, "random"), (11, "lookup"), (50, "random")):
        cfg = cfg_with(n)
        make = generate_random_genome if kind == "random" else generate_lookup_hard_genome
        elites = [mutate(make(cfg, rng_seed=i), 0.1, 0.2, np.random.default_rng(i)) for i in range(10)]
        rng = np.random.default_rng(42)
        pidx, seeds = PG.per_child_seeds(rng, counts)
        results = {}
        for w in (1, 2, 7, allowed_cpu_count()):
            with PG.GenomePool(w) as pool:
                results[w] = pool.mutate(mutate, elites, pidx, seeds, 0.1, 0.2)
        ref = results[1]
        check(f"n = {n}, {kind}: workers {sorted(results)} give identical children ({len(ref)})",
              all(all_same(ref, v) for v in results.values()) and len(ref) == 95)
        direct = [mutate(elites[p], 0.1, 0.2, np.random.default_rng(int(s))) for p, s in zip(pidx, seeds)]
        check(f"n = {n}, {kind}: child i == mutate(parent i, default_rng(seed i))", all_same(ref, direct))
    r1, r2 = np.random.default_rng(5), np.random.default_rng(5)
    p, s = PG.per_child_seeds(r1, [2, 3])
    want = r2.integers(0, 2 ** 63, size=5, dtype=np.uint64)
    check(f"seeds drawn once, in child order (elite 0's children first): parents {p}",
          p == [0, 0, 1, 1, 1] and np.array_equal(s, want)
          and r1.bit_generator.state == r2.bit_generator.state)


def test_gen0():
    print("\n[2] Generation 0 in parallel == the serial loop")
    for n, kind, make in ((50, "random", generate_random_genome), (11, "lookup", generate_lookup_hard_genome)):
        cfg = cfg_with(n)
        seeds = [1 + v for v in range(120)]
        serial = [make(cfg, rng_seed=s) for s in seeds]
        with PG.GenomePool(allowed_cpu_count()) as pool:
            par = pool.generate(make, cfg, seeds)
        check(f"n = {n}, {kind}: 120 genomes identical, in order", all_same(serial, par))


# ============================================================

def tiny(rng_mode, P=80):
    cfg = yaml.safe_load((ROOT / "configs/experiments/ea_from_random_tensor_cuda.yaml").read_text())
    x = cfg["experiment"]
    x["population_size"], x["n_runs"], x["max_ticks"] = P, 4, 300
    x["evolutionary_algorithm"].update(num_generations=3, elite_size=10, mutation_rng=rng_mode)
    x["evaluator"].update(device="cpu", compile=False)
    return cfg


RUN = """
import sys
import simulate.parallel_genomes as PG
PG.MIN_PARALLEL = {min_parallel}
sys.argv = ["run_ea", "--config", {cfg!r}]
from simulate import run_ea
run_ea.main()
"""


def run(cfg, tmp, tag, cwd=ROOT, min_parallel=None):
    out = Path(tmp) / tag
    out.mkdir()
    cfg = copy.deepcopy(cfg)
    cfg["experiment"]["output_folder"] = str(out) + "/"
    path = Path(tmp) / f"{tag}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    if min_parallel is None:
        cmd = [sys.executable, "-m", "simulate.run_ea", "--config", str(path)]
    else:
        cmd = [sys.executable, "-c", RUN.format(min_parallel=min_parallel, cfg=str(path))]
    p = subprocess.run(cmd, input="n\n", text=True, capture_output=True, cwd=cwd, timeout=1800)
    h5 = list(out.glob("*.h5"))
    return (h5[0] if len(h5) == 1 else None), p.stdout + p.stderr, p.returncode


def test_runs(tmp):
    print("\n[3] Whole EA runs (tensor CPU, P = 80, 3 generations)")
    pc = tiny("per_child")
    a, out_a, rc_a = run(pc, tmp, "pc_a")
    b, _o, rc_b = run(pc, tmp, "pc_b")
    if not check(f"per_child runs exit 0 ({rc_a}, {rc_b})", rc_a == 0 and rc_b == 0 and a and b):
        print(out_a[-2000:])
        return
    check("the workers were used (the config line names them)", "Genome workers:" in out_a)
    probs, n = diff_h5(str(a), str(b))
    check(f"per_child twice: all {n} datasets identical ({len(probs)} differences)", not probs)
    s, _o, rc_s = run(pc, tmp, "pc_serial", min_parallel=10 ** 9)
    probs, n = diff_h5(str(a), str(s))
    check(f"per_child serial vs parallel: all {n} datasets identical ({len(probs)})", rc_s == 0 and not probs)
    sh = tiny("shared")
    c, _o, rc_c = run(sh, tmp, "shared_new")
    import tests.test_scaling as TS
    head = TS.head_checkout(tmp)
    d, out_d, rc_d = run(sh, tmp, "shared_head", cwd=head)
    if "mutation_rng" in (head / "simulate" / "run_ea.py").read_text():
        print("      (HEAD already has Step 27: the comparison is HEAD vs itself)")
    probs, n = diff_h5(str(c), str(d)) if c and d else (["no output"], 0)
    check(f"shared == the committed run_ea: all {n} datasets identical ({len(probs)} differences)",
          rc_c == 0 and rc_d == 0 and not probs)
    probs, n = diff_h5(str(a), str(c))
    check(f"... and per_child differs from shared (other children): {len(probs)} datasets differ",
          len(probs) > 0)


# ============================================================

def test_key():
    print("\n[4] The config key")
    base = tiny("shared")["experiment"]
    x = copy.deepcopy(base)
    del x["evolutionary_algorithm"]["mutation_rng"]
    try:
        EA.check_ea_input_parameters(x)
        msg = None
    except KeyError as e:
        msg = str(e)
    check(f"missing: KeyError naming both values ({(msg or '')[:60]}...)",
          msg is not None and "'shared'" in msg and "'per_child'" in msg)
    for bad in ("per-child", "Shared", True, None):
        x = copy.deepcopy(base)
        x["evolutionary_algorithm"]["mutation_rng"] = bad
        try:
            EA.check_ea_input_parameters(x)
            ok = False
        except ValueError:
            ok = True
        check(f"value {bad!r}: refused", ok)
    shipped = [p for p in (ROOT / "configs" / "experiments").glob("*.yaml")
               if yaml.safe_load(p.read_text())["experiment"].get("evolutionary_algorithm_enabled")]
    vals = {p.stem: yaml.safe_load(p.read_text())["experiment"]["evolutionary_algorithm"].get("mutation_rng")
            for p in shipped}
    check(f"all {len(vals)} shipped EA configs state it, all 'shared' ({set(vals.values())})",
          len(vals) >= 10 and set(vals.values()) == {"shared"})


def test_failure():
    print("\n[5] A failing mutation in a worker")
    cfg = cfg_with(11)
    elites = [generate_random_genome(cfg, rng_seed=i) for i in range(5)]
    pidx, seeds = PG.per_child_seeds(np.random.default_rng(0), [40] * 5)
    bad_child = next(i for i, s in enumerate(seeds) if np.random.default_rng(int(s)).integers(0, 1000) == 7) \
        if any(np.random.default_rng(int(s)).integers(0, 1000) == 7 for s in seeds) else None
    if bad_child is None:                         # make sure one child fails
        seeds[150] = next(s for s in range(10 ** 6) if np.random.default_rng(s).integers(0, 1000) == 7)
        bad_child = 150
    try:
        with PG.GenomePool(4) as pool:
            pool.mutate(broken_mutation, elites, pidx, seeds, 0.1, 0.2)
        msg = None
    except RuntimeError as e:
        msg = str(e)
    check(f"RuntimeError naming the child ({(msg or 'no error')[:70]})",
          msg is not None and f"child {bad_child} " in msg and "boom" in msg)


def test_no_torch():
    print("\n[6] No torch in the workers")
    # spawn re-imports the parent's main module in each worker. In a real run that is
    # simulate.run_ea (python -m), which must not import torch at the top; and the pool
    # itself must not need it even when the parent has it loaded (a tensor run).
    probe = ("import sys, torch; import simulate.parallel_genomes as PG\n"
             "pool = PG.GenomePool(2); print('WORKER', pool.worker_modules()); pool.close()")
    p = subprocess.run([sys.executable, "-c", probe], cwd=ROOT, capture_output=True, text=True, timeout=600)
    line = next((l for l in p.stdout.splitlines() if l.startswith("WORKER")), p.stderr[-200:])
    check(f"parent with torch loaded: the workers have neither torch nor mvb_torch ({line})",
          line == "WORKER []")
    p = subprocess.run([sys.executable, "-c", "import sys, simulate.run_ea; "
                        "print('MAIN', sorted(m for m in ('torch', 'mvb_torch') if m in sys.modules))"],
                       cwd=ROOT, capture_output=True, text=True, timeout=600)
    line = next((l for l in p.stdout.splitlines() if l.startswith("MAIN")), p.stderr[-200:])
    check(f"simulate.run_ea (the main module workers re-import) loads no torch ({line})", line == "MAIN []")


def test_legacy(tmp):
    print("\n[7] Old references without the key")
    def make(path, rng_value):
        with h5py.File(path, "w") as f:
            f.attrs["experiment_simulation_seed"] = 1
            if rng_value is not None:
                f.attrs["experiment_evolutionary_algorithm_mutation_rng"] = rng_value
            f.create_dataset("x", data=np.arange(3))
    ref, shared, pc = (Path(tmp) / f"{n}.h5" for n in ("ref", "shared", "pc"))
    make(ref, None)
    make(shared, "shared")
    make(pc, "per_child")
    r1 = RS.compare_to_reference(str(shared), str(ref))
    r2 = RS.compare_to_reference(str(pc), str(ref))
    check(f"run_selected: 'shared' vs an old reference: {r1!r}", r1.startswith("identical"))
    check(f"run_selected: 'per_child' vs an old reference: not comparable ({r2[:50]})",
          r2.startswith("not comparable") and "mutation_rng" in r2)
    check("legacy rule only for that key and that value",
          RS.legacy_attr_ok("experiment_evolutionary_algorithm_mutation_rng", b"shared")
          and not RS.legacy_attr_ok("experiment_evolutionary_algorithm_mutation_rng", "per_child")
          and not RS.legacy_attr_ok("experiment_other_key", "shared"))


def test_speed():
    print("\n[8] Speed at n = 50, 950 children (reported)")
    cfg = cfg_with(50)
    elites = [mutate(generate_random_genome(cfg, rng_seed=i), 0.1, 0.2, np.random.default_rng(i)) for i in range(50)]
    pidx, seeds = PG.per_child_seeds(np.random.default_rng(1), [19] * 50)
    t0 = time.perf_counter()
    serial = PG.GenomePool(1).mutate(mutate, elites, pidx, seeds, 0.1, 0.2)
    ts = time.perf_counter() - t0
    with PG.GenomePool(allowed_cpu_count()) as pool:
        pool.worker_modules()                     # start the workers outside the timing
        t0 = time.perf_counter()
        par = pool.mutate(mutate, elites, pidx, seeds, 0.1, 0.2)
        tp = time.perf_counter() - t0
    print(f"      serial {ts:.2f} s, {allowed_cpu_count()} workers {tp:.2f} s ({ts / tp:.1f}x)")
    check("parallel gives the same 950 children, and is faster", all_same(serial, par) and tp < ts)


def main():
    with tempfile.TemporaryDirectory(prefix="test_mutation_parallel_") as tmp:
        test_per_child()
        test_gen0()
        test_key()
        test_failure()
        test_no_torch()
        test_legacy(tmp)
        test_speed()
        test_runs(tmp)
    n_fail = sum(1 for _, ok in _RESULTS if not ok)
    if n_fail:
        print(f"\nFAILED -- {n_fail} of {len(_RESULTS)} checks failed")
        return 1
    print(f"\nOK -- {len(_RESULTS)}/{len(_RESULTS)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
