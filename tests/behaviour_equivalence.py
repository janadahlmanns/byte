#!/usr/bin/env python3
"""Behavioural equivalence at production size (plan_evotorch.md Step 12, rev. 2).

Does the tensor code in production mode (CUDA, float32, Philox, compiled) behave like
the scalar code, and the scalar code like the original (8e17e694c7)? Measured at four
levels, each counting how many runs (and decisions) differ for one cause:

  L1   model      scalar vs tensor, both PRE-DRAWN (the same random numbers; tensor on
                  CPU/float64, the bit-exact reference path): must be 0 runs, 0 ticks
  L1p  precision  tensor live CPU/float64 vs live CUDA/float32, both eager, the same
                  Philox numbers: counted
  L1c  compile    tensor live eager vs compiled, the same numbers: counted
  L2   live       tensor (production) vs scalar, different random streams: judged
                  statistically against the scalar against itself (the "floor"), with
                  criteria fixed in the plan before any run
  L3   original   original code vs current scalar, the same seeds: counted

Every arm runs in its own subprocess and writes arms/<name>.npz. The analysis reads
only those files (plus the HDF5 of the per-tick arms), so `--analyse DIR` redoes the
report and figures without running anything.

Usage (university PC, see plan Step 12)
-----
    python -m tests.behaviour_equivalence --part a \\
        --orig-worktree ../byte_original --orig-commit 8e17e694c7 \\
        --current-worktree ../byte_current
    python -m tests.behaviour_equivalence --part b  (same worktree arguments)
    python -m tests.behaviour_equivalence --resume data/behaviour_equivalence/<dir>
    python -m tests.behaviour_equivalence --analyse data/behaviour_equivalence/<dir>

Exit code: 0 PASS, 1 FAIL, 2 INCONCLUSIVE (the floor itself misses the margin).
"""

import argparse
import copy
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.be_common import (MOVES, PER_TICK_BASE, RESULT_KEYS, EquivalenceError,  # noqa: E402
                             call_eval, check_seeds_recorded, find_h5, flatten_genomes,
                             genome_hash, read_arm, read_per_tick, read_summaries,
                             write_arm)

ROOT = Path(__file__).resolve().parent.parent
OUT_BASE = ROOT / "data" / "behaviour_equivalence"
ORIG_RUNNER = ROOT / "tests" / "orig_arm.py"

# ---------------------------------------------------------------------------
# The two parts (plan 12a / 12b). Sizes fixed in the plan; changed only by revision.
# ---------------------------------------------------------------------------

PARTS = {
    "a": dict(config="configs/experiments/ea_from_lookup_tensor_cuda.yaml",
              elites="tests/refs/behaviour/elites_lookup_n11.h5",
              n_runs=300, decision_runs=20, predrawn_runs=10, topk=10,
              controls=["C1", "C2a", "C2b"],
              blocks=dict(kind="lookup", copies=10, chain_lengths=list(range(1, 10)),
                          chains_per_length=10, elites=10, mutants_per_elite=15)),
    "b": dict(config="configs/experiments/ea_from_random_large_tensor_cuda.yaml",
              elites="tests/refs/behaviour/elites_large_n50.h5",
              n_runs=100, decision_runs=10, predrawn_runs=3, topk=5,
              controls=["C1"],
              blocks=dict(kind="random", random=20, elites=50, mutants=30)),
}

# Pass criteria (plan Step 12, fixed 2026-11-12 before any run).
RHO_SLACK = 0.02            # (1) rho(T,S1) >= min floor rho - 0.02
MARGIN = 0.02               # (2) 90% CI of the mean bias inside +-2% of S1's mean
KS_SLACK = 0.05             # (3) KS share <= max floor share + 5 points
D_FACTOR, D_SLACK = 1.5, 0.005   # (4) D <= 1.5 x max floor D + 0.005
CAP_SLACK = 0.01            # (4) capped share: |diff| <= max floor |diff| + 1 point
TOPK_SLACK = 2              # (5) top-k overlap >= min floor overlap - 2
KS_ALPHA = 0.05
SEED_SETS = (1, 2, 3)       # simulation seeds of S1/T, S2/T2, S3

CONTROL_PATCHES = {
    "C1": ("brain.noise_level", lambda v: v * 2.0, "noise_level x2 (must FAIL)"),
    "C2a": ("brain.noise_level", lambda v: v * 1.2, "noise_level x1.2 (sensitivity)"),
    "C2b": ("brain.threshold", lambda v: 0.52, "threshold 0.5 -> 0.52 (sensitivity)"),
}

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def json_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def git(*args, cwd=ROOT):
    p = subprocess.run(["git", "-C", str(cwd), *args], text=True, capture_output=True)
    if p.returncode != 0:
        raise EquivalenceError(f"git {' '.join(args)} failed: {p.stderr.strip()}")
    return p.stdout.strip()


def code_state():
    """(HEAD, "" or the SHA-256 of the uncommitted diff of tracked files)."""
    head = git("rev-parse", "HEAD")
    diff = subprocess.run(["git", "-C", str(ROOT), "diff", "HEAD", "--binary"],
                          capture_output=True, check=True).stdout
    return head, (hashlib.sha256(diff).hexdigest() if diff else "")


def load_cfg(path):
    with open(ROOT / path) as f:
        return yaml.safe_load(f)


def set_dotted(cfg, dotted, fn):
    d = cfg
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d[k]
    if keys[-1] not in d:
        raise EquivalenceError(f"control key {dotted} not in the config")
    d[keys[-1]] = fn(d[keys[-1]])


def max_brain_ticks(cfg):
    """decide()'s inner-loop bound: warm-up (<= n active neurons) x max_decision_delay."""
    b = cfg["brain"]
    return int(int(b["n_neurons"]) * float(b["max_decision_delay"]))


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------


def build_population(cfg, elites_path, blocks):
    """-> (genomes, block label per genome). Deterministic: every random draw comes from
    a SeedSequence keyed by (12, block, ...), never from a shared stream."""
    from mvb.genome.generate_genome_from_file import generate_genome_from_file
    from mvb.genome.generate_genome_lookup_hard import generate_lookup_hard_genome
    from mvb.genome.generate_genome_mutate_simple import generate_genome_mutate_simple
    from mvb.genome.generate_genome_random import generate_random_genome

    ea = cfg["experiment"]["evolutionary_algorithm"]
    if ea["mutation_method"] != "simple":
        raise EquivalenceError(f"mutation_method {ea['mutation_method']!r}: only 'simple'")
    rate, strength = ea["mutation_rate"], ea["mutation_strength"]

    def mutate(g, *key):
        return generate_genome_mutate_simple(
            g, rate, strength, np.random.default_rng(np.random.SeedSequence([12, *key])))

    n = int(cfg["brain"]["n_neurons"])
    elites = [generate_genome_from_file(cfg, elite_id=e, hdf5_path=str(ROOT / elites_path))
              for e in range(blocks["elites"])]
    for e, g in enumerate(elites):
        if np.asarray(g.connection_weights).shape[0] != n:
            raise EquivalenceError(f"elite {e} of {elites_path} has "
                                   f"{np.asarray(g.connection_weights).shape[0]} neurons, "
                                   f"the config {n}")

    genomes, labels = [], []
    gtype = cfg["experiment"]["genome_type"]
    if blocks["kind"] == "lookup":
        if gtype != "lookup_hard":
            raise EquivalenceError(f"lookup population needs genome_type lookup_hard, got {gtype}")
        # A: generation 0 as run_ea builds it (lookup ignores the seed: identical copies)
        for v in range(blocks["copies"]):
            genomes.append(generate_lookup_hard_genome(cfg, rng_seed=v))
            labels.append("A gen-0")
        # B: chains of 1..9 mutations (early EA)
        for length in blocks["chain_lengths"]:
            for c in range(blocks["chains_per_length"]):
                g = generate_lookup_hard_genome(cfg, rng_seed=0)
                rng = np.random.default_rng(np.random.SeedSequence([12, 1, length, c]))
                for _ in range(length):
                    g = generate_genome_mutate_simple(g, rate, strength, rng)
                genomes.append(g)
                labels.append(f"B chain-{length}")
        # C: each final elite + single mutants (late EA)
        for e, g in enumerate(elites):
            genomes.append(g)
            labels.append("C elite")
            for j in range(blocks["mutants_per_elite"]):
                genomes.append(mutate(g, 2, e, j))
                labels.append("C mutant")
    elif blocks["kind"] == "random":
        if gtype != "random":
            raise EquivalenceError(f"random population needs genome_type random, got {gtype}")
        seed0 = cfg["experiment"]["random_genome"]["wiring_randomization_seed"]
        for v in range(blocks["random"]):
            genomes.append(generate_random_genome(cfg, rng_seed=seed0 + v))
            labels.append("A gen-0")
        genomes += elites
        labels += ["C elite"] * len(elites)
        for e in range(blocks["mutants"]):
            genomes.append(mutate(elites[e], 3, e))
            labels.append("C mutant")
    else:
        raise EquivalenceError(f"unknown population kind {blocks['kind']!r}")
    return genomes, labels


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


def arm_specs(part, spec, ev_cfg, device_override):
    """name -> spec. kind: scalar | tensor | orig. runs: full | decision | predrawn."""
    dev = device_override or ev_cfg["device"]
    prod = dict(device=dev, dtype=ev_cfg["dtype"], philox_rounds=ev_cfg["philox_rounds"],
                compile=ev_cfg["compile"])
    eager = dict(prod, compile=False)
    f64 = dict(device="cpu", dtype="float64", philox_rounds=ev_cfg["philox_rounds"], compile=False)
    A = {}

    def add(name, kind, seed, runs, per_tick=False, tensor=None, patch=None, worktree=None,
            predrawn=False, role=""):
        A[name] = dict(kind=kind, seed=seed, runs=runs, per_tick=per_tick, tensor=tensor,
                       patch=patch, worktree=worktree, predrawn=predrawn, role=role)

    # Tensor first: a CUDA / compile problem shows in minutes, not after the scalar hour.
    add("T", "tensor", 1, "full", tensor=prod, role="production tensor (L2 under test)")
    add("Tr", "tensor", 1, "full", tensor=prod, role="T again: must equal T bit for bit")
    add("T2", "tensor", 2, "full", tensor=prod, role="tensor with S2's seeds")
    add("Te", "tensor", 1, "full", tensor=eager, role="eager (L1c)")
    add("Tf64", "tensor", 1, "full", tensor=f64, role="CPU float64 eager (L1p)")
    for c in spec["controls"]:
        add(c, "tensor", 1, "full", tensor=prod, patch=c, role=CONTROL_PATCHES[c][2])
    add("Td", "tensor", 1, "decision", per_tick=True, tensor=prod, role="T, per-tick")
    add("Ted", "tensor", 1, "decision", per_tick=True, tensor=eager, role="Te, per-tick")
    add("Tf64d", "tensor", 1, "decision", per_tick=True, tensor=f64, role="Tf64, per-tick")
    add("Tp", "tensor", 1, "predrawn", per_tick=True, predrawn=True,
        tensor=dict(device="cpu", dtype="float64"), role="tensor pre-drawn (L1)")
    add("S1", "scalar", 1, "full", role="scalar reference")
    add("S2", "scalar", 2, "full", role="floor")
    add("S3", "scalar", 3, "full", role="floor")
    add("S1d", "scalar", 1, "decision", per_tick=True, role="S1, per-tick")
    add("S2d", "scalar", 2, "decision", per_tick=True, role="S2, per-tick (floor)")
    add("Sp", "scalar", 1, "predrawn", per_tick=True, predrawn=True, role="scalar pre-drawn (L1)")
    add("O0", "orig", 1, "decision", per_tick=True, worktree="current",
        role="runner on the current commit: must equal S1d (positive control)")
    add("Od", "orig", 1, "decision", per_tick=True, worktree="orig", role="original, per-tick (L3)")
    add("O", "orig", 1, "full", worktree="orig", role="original code (L3)")
    return A


def arm_cfg(base_cfg, spec, arm):
    cfg = copy.deepcopy(base_cfg)
    R = {"full": spec["n_runs"], "decision": spec["decision_runs"],
         "predrawn": spec["predrawn_runs"]}[arm["runs"]]
    cfg["experiment"]["n_runs"] = R
    cfg["experiment"]["max_ticks"] = base_cfg["experiment"]["max_ticks"]
    if arm["patch"]:
        key, fn, _ = CONTROL_PATCHES[arm["patch"]]
        set_dotted(cfg, key, fn)
    return cfg


def arm_seeds(all_seeds, arm, R):
    s = all_seeds[arm["seed"]]
    return {"run_seeds": s["run_seeds"][:R], "noise_seeds": s["noise_seeds"][:, :R],
            "decision_seeds": s["decision_seeds"][:, :R]}


def draw_all_seeds(P, R):
    from mvb_torch.generation import draw_seeds, make_simulation_rngs
    """Seed set s: noise / decision seeds as run_ea's generation 0 draws them with
    simulation seed s. The WORLDS (run_seeds) are always seed 1's, so every arm is paired
    world by world; only the noise / decision streams differ (the floor's definition)."""
    out = {}
    for s in SEED_SETS:
        ss = draw_seeds(*make_simulation_rngs(s), P, R)
        out[s] = {"run_seeds": np.asarray(ss.run_seeds), "noise_seeds": np.asarray(ss.noise_seeds),
                  "decision_seeds": np.asarray(ss.decision_seeds)}
    for s in SEED_SETS[1:]:
        out[s]["run_seeds"] = out[SEED_SETS[0]]["run_seeds"].copy()
    return out


def load_seeds(path):
    with np.load(path) as z:
        out = {}
        for name in z.files:
            s, k = name[1:].split("_", 1)
            out.setdefault(int(s), {})[k] = z[name]
    if sorted(out) != list(SEED_SETS):
        raise EquivalenceError(f"{path}: seed sets {sorted(out)}")
    return out


def _results_from_tensor(captured, returned, P, R):
    if len(captured) != 1:
        raise EquivalenceError(f"the adapter ran eval_generation_batch {len(captured)} times, "
                               f"expected once")
    res = captured[0]
    out = {k: getattr(res, k).cpu().numpy() for k in RESULT_KEYS}
    lif = np.stack([returned[v]["lifespan_vector"] for v in range(P)]).astype(np.int64)
    if out["lifespans"].shape != (P, R) or not np.array_equal(lif, out["lifespans"]):
        raise EquivalenceError("the adapter's returned lifespans differ from its batch result")
    return out, res


def run_arm(outdir, name):
    """Run one arm (in its own process) and write arms/<name>.npz."""
    run = json.loads((Path(outdir) / "run.json").read_text())
    arm = run["arms"][name]
    spec = run["spec"]
    base_cfg = load_cfg(run["config"])
    cfg = arm_cfg(base_cfg, spec, arm)
    if json_hash(cfg) != arm["config_hash"]:
        raise EquivalenceError(f"arm {name}: config hash changed since the run was planned")
    seeds_all = load_seeds(Path(outdir) / "seeds.npz")
    R = cfg["experiment"]["n_runs"]
    seeds = arm_seeds(seeds_all, arm, R)
    folder = Path(outdir) / "arms" / name
    if folder.exists():
        import shutil
        shutil.rmtree(folder)            # a crashed earlier attempt of this arm
    folder.mkdir(parents=True)
    head, dirty = code_state()
    meta = dict(arm=name, spec=arm, population_hash=run["population_hash"],
                config_hash=arm["config_hash"], n_runs=R, per_tick=arm["per_tick"],
                code_commit=head, code_dirty=dirty)

    if arm["kind"] == "orig":
        return run_orig_arm(outdir, run, name, arm, cfg, seeds, folder, meta)

    genomes, _labels = build_population(base_cfg, run["elites"], spec["blocks"])
    if genome_hash(genomes) != run["population_hash"]:
        raise EquivalenceError(f"arm {name}: the population rebuilt here differs from the plan")
    P = len(genomes)
    randomness = ({"enabled": True, "max_brain_ticks": max_brain_ticks(cfg)}
                  if arm["predrawn"] else None)
    t0 = time.perf_counter()
    if arm["kind"] == "scalar":
        import mvb.simulation_API as api
        call_eval(api.eval_generation, genomes, cfg, str(folder), seeds=seeds,
                  per_tick=arm["per_tick"], randomness_cfg=randomness)
        wall = time.perf_counter() - t0
        h5 = find_h5(str(folder))
        res = read_summaries(h5, P, R)
        check_seeds_recorded(res, seeds, name)
        meta.update(wall_s=wall, h5=h5)
    else:
        import torch
        import mvb_torch.adapter as adapter
        captured = []
        real = adapter.eval_generation_batch

        def spy(*a, **k):
            r = real(*a, **k)
            captured.append(r)
            return r

        t = arm["tensor"]
        ev = {"backend": "tensor", "device": t["device"], "dtype": t["dtype"]}
        if not arm["predrawn"]:
            ev.update(philox_rounds=t["philox_rounds"], compile=t["compile"])
        adapter.eval_generation_batch = spy
        try:
            evaluate = adapter.make_tensor_evaluator(ev, cfg, randomness)
            tracked = arm["per_tick"]
            if tracked:
                returned, run_seeds = call_eval(evaluate, genomes, cfg, str(folder), seeds=seeds,
                                                per_tick=True, randomness_cfg=randomness)
            else:
                returned, run_seeds = call_eval(evaluate, genomes, cfg, str(folder), seeds=seeds,
                                                per_tick=False, randomness_cfg=randomness,
                                                per_run=False)
        finally:
            adapter.eval_generation_batch = real
        wall = time.perf_counter() - t0
        res, gres = _results_from_tensor(captured, returned, P, R)
        if not np.array_equal(np.asarray(run_seeds).astype(np.int64), seeds["run_seeds"].astype(np.int64)):
            raise EquivalenceError(f"arm {name}: the evaluator returned other run_seeds")
        res.update({k: seeds[k] for k in ("run_seeds", "noise_seeds", "decision_seeds")})
        if tracked:
            h5 = find_h5(str(folder))
            rec = read_summaries(h5, P, R)
            check_seeds_recorded(rec, seeds, name)
            for k in RESULT_KEYS:
                if not np.array_equal(rec[k].astype(np.float64), res[k].astype(np.float64)):
                    raise EquivalenceError(f"arm {name}: tracked {k} differ from the batch result")
            meta["h5"] = h5
        meta.update(wall_s=wall, warmup_s=float(gres.warmup_s), new_graphs=int(gres.new_graphs),
                    torch=torch.__version__,
                    gpu=(torch.cuda.get_device_name(0) if t["device"] == "cuda" else None))
    meta.update(python=sys.version.split()[0], numpy=np.__version__)
    write_arm(Path(outdir) / "arms" / f"{name}.npz", results=res, meta=meta)
    print(f"[arm {name}] done in {meta['wall_s']:.1f} s", flush=True)
    return 0


def run_orig_arm(outdir, run, name, arm, cfg, seeds, folder, meta):
    wt_key = {"orig": "orig_worktree", "current": "current_worktree"}[arm["worktree"]]
    worktree = Path(run[wt_key]).resolve()
    commit = run["orig_commit"] if arm["worktree"] == "orig" else run["code_commit"]
    base_cfg = load_cfg(run["config"])
    genomes, _ = build_population(base_cfg, run["elites"], run["spec"]["blocks"])
    flat = flatten_genomes(genomes)
    inp = folder / "arm_input.npz"
    meta_in = dict(meta, population_hash=run["population_hash"])
    np.savez(inp, **flat, **{k: np.asarray(v) for k, v in seeds.items()},
             cfg_json=np.frombuffer(json.dumps(cfg).encode(), dtype=np.uint8),
             meta_json=np.frombuffer(json.dumps(meta_in).encode(), dtype=np.uint8))
    env = dict(os.environ, PYTHONPATH=str(worktree))
    out = Path(outdir) / "arms" / f"{name}.npz"
    cmd = [sys.executable, str(ORIG_RUNNER), "--input", str(inp), "--out", str(out),
           "--folder", str(folder / "h5"), "--commit", commit, "--arm", name]
    print(f"[arm {name}] {worktree} @ {commit[:10]}", flush=True)
    p = subprocess.run(cmd, cwd=str(worktree), env=env)
    if p.returncode != 0:
        raise EquivalenceError(f"arm {name}: the runner on {worktree} failed (exit {p.returncode})")
    return 0


# ---------------------------------------------------------------------------
# Planning a run
# ---------------------------------------------------------------------------


def check_worktree_arg(path, commit, what):
    wt = Path(path).resolve()
    if wt == ROOT:
        raise EquivalenceError(f"{what}: must be a separate worktree, not this checkout")
    if not (wt / "mvb").is_dir():
        raise EquivalenceError(f"{what}: {wt} has no mvb/ (create it with git worktree add, "
                               f"see the plan)")
    head = git("rev-parse", "HEAD", cwd=wt)
    want = git("rev-parse", f"{commit}^{{commit}}")
    if head != want:
        raise EquivalenceError(f"{what}: {wt} is at {head[:10]}, expected {want[:10]}")
    if git("status", "--porcelain", "--untracked-files=no", cwd=wt):
        raise EquivalenceError(f"{what}: {wt} has local changes")
    return str(wt)


def plan_run(part, spec, outdir, orig_worktree, orig_commit, current_worktree,
             device_override=None):
    base_cfg = load_cfg(spec["config"])
    ev = base_cfg["experiment"]["evaluator"]
    head, dirty = code_state()
    run = dict(part=part, spec=spec, config=spec["config"],
               config_sha256=sha256_file(ROOT / spec["config"]),
               elites=spec["elites"], elites_sha256=sha256_file(ROOT / spec["elites"]),
               code_commit=head, code_dirty=dirty, device_override=device_override,
               orig_commit=git("rev-parse", f"{orig_commit}^{{commit}}"),
               orig_worktree=check_worktree_arg(orig_worktree, orig_commit, "--orig-worktree"),
               current_worktree=check_worktree_arg(current_worktree, head, "--current-worktree"),
               created=datetime.datetime.now().isoformat(timespec="seconds"))
    genomes, labels = build_population(base_cfg, spec["elites"], spec["blocks"])
    run["population_hash"] = genome_hash(genomes)
    run["labels"] = labels
    run["max_ticks"] = int(base_cfg["experiment"]["max_ticks"])
    arms = arm_specs(part, spec, ev, device_override)
    for name, a in arms.items():
        a["config_hash"] = json_hash(arm_cfg(base_cfg, spec, a))
    run["arms"] = arms
    outdir.mkdir(parents=True)
    (outdir / "arms").mkdir()
    seeds = draw_all_seeds(len(genomes), spec["n_runs"])
    np.savez(outdir / "seeds.npz", **{f"s{s}_{k}": v for s, d in seeds.items() for k, v in d.items()})
    (outdir / "run.json").write_text(json.dumps(run, indent=1))
    return run


def system_snapshot(outdir):
    from simulate import machine_load
    from simulate.run_selected import system_info, torch_probe
    info = system_info(torch_probe())
    try:
        info["idle_check"] = machine_load.idle_check()
    except Exception as e:  # noqa: BLE001 -- recorded, not hidden
        info["idle_check"] = f"unavailable: {type(e).__name__}: {e}"
    (Path(outdir) / "system.json").write_text(json.dumps(info, indent=1, default=str))
    ic = info["idle_check"]
    print(f"[system] {info.get('cpu_model')}, gpus {info.get('gpus')}; idle check: "
          f"{ic.get('verdict') if isinstance(ic, dict) else ic}", flush=True)


def execute(outdir, resume=False):
    run = json.loads((outdir / "run.json").read_text())
    head, dirty = code_state()
    if head != run["code_commit"] or dirty != run["code_dirty"]:
        raise EquivalenceError(f"the code changed since this run was planned "
                               f"({run['code_commit'][:10]}{'+diff ' + run['code_dirty'][:8] if run['code_dirty'] else ''} "
                               f"-> {head[:10]}{'+diff ' + dirty[:8] if dirty else ''})")
    for name, arm in run["arms"].items():
        path = outdir / "arms" / f"{name}.npz"
        if path.exists():
            if not resume:
                raise EquivalenceError(f"{path} exists; use --resume")
            m = read_arm(path)["meta"]
            # An original-code arm records the commit it ran (its worktree's).
            commit = run["orig_commit"] if arm.get("worktree") == "orig" else run["code_commit"]
            want = {"population_hash": run["population_hash"], "config_hash": arm["config_hash"],
                    "code_commit": commit}
            if arm["kind"] != "orig":
                want["code_dirty"] = run["code_dirty"]
            bad = [k for k, v in want.items() if m.get(k) != v]
            if bad:
                raise EquivalenceError(f"{path}: {bad} do not match this run; refusing to reuse")
            print(f"[arm {name}] reused", flush=True)
            continue
        print(f"\n[arm {name}] {arm['role']}", flush=True)
        p = subprocess.run([sys.executable, "-m", "tests.behaviour_equivalence",
                            "--run-arm", name, "--dir", str(outdir)], cwd=str(ROOT))
        if p.returncode != 0:
            raise EquivalenceError(f"arm {name} failed (exit {p.returncode}); finished arms "
                                   f"are kept, continue with --resume {outdir}")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def spearman(a, b):
    """Spearman's rho; NaN (no warning) when either side is constant (e.g. a feature
    that is 0 for every genome), which the report prints as nan."""
    from scipy import stats
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(a) < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    return float(stats.spearmanr(a, b).statistic)


def topk(means, k):
    order = np.lexsort((np.arange(len(means)), -means))     # by mean, ties by index
    return set(order[:k].tolist())


def per_run_diff(A, B, max_ticks):
    """How the runs of B differ from the paired runs of A (same worlds)."""
    out = {}
    for k in ("lifespans", "eats", "distance"):
        out[f"share_runs_{k}_differ"] = float(np.mean(A[k] != B[k]))
    d = (B["lifespans"] - A["lifespans"]).ravel()
    q = np.percentile(d, [25, 50, 75])
    out.update(lifespan_diff_median=float(q[1]), lifespan_diff_iqr=[float(q[0]), float(q[2])],
               lifespan_diff_mean=float(d.mean()),
               share_abs_diff_le_1=float(np.mean(np.abs(d) <= 1)),
               share_abs_diff_le_10=float(np.mean(np.abs(d) <= 10)),
               share_abs_diff_le_100=float(np.mean(np.abs(d) <= 100)),
               runs=int(d.size))
    return out


def measures(A, X, key, k, max_ticks):
    """Measures (1)-(5) of X against the reference A, for one variable."""
    from scipy import stats
    a, x = A[key].astype(np.float64), X[key].astype(np.float64)
    P = a.shape[0]
    ma, mx = a.mean(1), x.mean(1)
    d = mx - ma
    se = d.std(ddof=1) / np.sqrt(P)
    t = stats.t.ppf(0.95, P - 1)
    import warnings
    with warnings.catch_warnings():
        # scipy falls back from the exact to the asymptotic p-value for some tied
        # samples and says so per genome; the fallback is the intended behaviour.
        warnings.filterwarnings("ignore", message="ks_2samp: Exact calculation unsuccessful")
        ks_p = np.array([stats.ks_2samp(x[g], a[g]).pvalue for g in range(P)])
    m = dict(rho=spearman(mx, ma), bias=float(d.mean()), bias_ci90=[float(d.mean() - t * se),
             float(d.mean() + t * se)], margin=float(MARGIN * ma.mean()), ref_mean=float(ma.mean()),
             ks_share=float(np.mean(ks_p < KS_ALPHA)),
             pooled_D=float(stats.ks_2samp(x.ravel(), a.ravel()).statistic))
    m["equivalent"] = bool(-m["margin"] < m["bias_ci90"][0] and m["bias_ci90"][1] < m["margin"])
    if key == "lifespans":
        m["capped_share"] = float(np.mean(X[key] >= max_ticks))
        m["capped_share_ref"] = float(np.mean(A[key] >= max_ticks))
        m["topk_overlap"] = len(topk(mx, k) & topk(ma, k))
    return m


def judge(m, floors, key):
    """Criteria (1)-(5) of measures `m` against the floor measures: True / False, or None
    for (2) when the floor itself misses the margin (then the margin is too tight for
    this noise and (2) can decide nothing: INCONCLUSIVE, plan Step 12)."""
    c = {}
    c["1_rank"] = m["rho"] >= min(f["rho"] for f in floors) - RHO_SLACK
    c["2_bias"] = m["equivalent"] if all(f["equivalent"] for f in floors) else None
    c["3_ks_share"] = m["ks_share"] <= max(f["ks_share"] for f in floors) + KS_SLACK
    c["4_pooled_D"] = m["pooled_D"] <= D_FACTOR * max(f["pooled_D"] for f in floors) + D_SLACK
    if key == "lifespans":
        cap = abs(m["capped_share"] - m["capped_share_ref"])
        cap_floor = max(abs(f["capped_share"] - f["capped_share_ref"]) for f in floors)
        c["4_capped"] = cap <= cap_floor + CAP_SLACK
        c["5_topk"] = m["topk_overlap"] >= min(f["topk_overlap"] for f in floors) - TOPK_SLACK
    return c


def decision_features(ticks):
    """Per run: move shares, no-decision share, first food tick, eats per tick alive."""
    rows = []
    for (v, r), d in sorted(ticks.items()):
        d = d[d["tick"] > 0]
        n = max(len(d), 1)
        mv = d["movement"]
        food = np.nonzero(d["food_consumed"] == 1)[0]
        rows.append(dict(v=v, r=r, **{f"move_{m.decode()}": float(np.mean(mv == m)) for m in MOVES},
                         no_decision=float(np.mean(d["decision_made"] == 0)),
                         first_food=(float(d["tick"][food[0]]) if len(food) else np.nan),
                         eats_per_tick=float(d["food_consumed"].sum() / n)))
    return rows


FEATURES = [f"move_{m.decode()}" for m in MOVES] + ["no_decision", "first_food", "eats_per_tick"]


def per_genome_feature(rows, P, feat):
    vals = [[] for _ in range(P)]
    for row in rows:
        if not np.isnan(row[feat]):
            vals[row["v"]].append(row[feat])
    return np.array([np.mean(v) if v else np.nan for v in vals])


def decision_stats(ref, x, floor, P):
    out = {}
    for feat in FEATURES:
        a, b, f = (per_genome_feature(z, P, feat) for z in (ref, x, floor))
        ok = ~(np.isnan(a) | np.isnan(b) | np.isnan(f))
        out[feat] = dict(ref_mean=float(np.nanmean(a)), x_mean=float(np.nanmean(b)),
                         floor_mean=float(np.nanmean(f)),
                         rho=spearman(b[ok], a[ok]), rho_floor=spearman(f[ok], a[ok]),
                         bias=float(np.mean(b[ok] - a[ok])), bias_floor=float(np.mean(f[ok] - a[ok])))
    return out


def weight_columns(d):
    return [c for c in d.dtype.names if c not in PER_TICK_BASE]


def decision_divergence(ta, tb, exact=False):
    """Same random numbers: compare two arms' per-tick records run by run."""
    runs = diverged = ticks = ticks_diff = 0
    firsts, field_mismatch = [], 0
    for key in sorted(ta):
        a, b = ta[key], tb[key]
        runs += 1
        seq_a = np.stack([a["movement"].astype("S4").view(np.uint32), a["decision_made"].astype(np.uint32)], 1)
        seq_b = np.stack([b["movement"].astype("S4").view(np.uint32), b["decision_made"].astype(np.uint32)], 1)
        n = min(len(a), len(b))
        neq = np.any(seq_a[:n] != seq_b[:n], axis=1)
        ticks += max(len(a), len(b))
        ticks_diff += int(neq.sum()) + abs(len(a) - len(b))
        if exact:
            same = (a.dtype.names == b.dtype.names and len(a) == len(b)
                    and all(np.array_equal(a[f], b[f]) for f in a.dtype.names))
            field_mismatch += 0 if same else 1
        if neq.any() or len(a) != len(b):
            diverged += 1
            i = int(np.argmax(neq)) if neq.any() else n
            cols = sorted(set(weight_columns(a)) & set(weight_columns(b)))
            dw = (float(max(abs(float(a[c][i - 1]) - float(b[c][i - 1])) for c in cols))
                  if i > 0 and cols else None)
            firsts.append(dict(run=list(key), tick=int(a["tick"][i]) if i < len(a) else int(len(a)),
                               a=(a["movement"][i].decode() if i < len(a) else "dead"),
                               b=(b["movement"][i].decode() if i < len(b) else "dead"),
                               to_death_a=int(len(a) - i), to_death_b=int(len(b) - i),
                               max_abs_weight_diff_before=dw))
    out = dict(runs=runs, runs_diverged=diverged, share_runs_diverged=diverged / max(runs, 1),
               ticks=ticks, ticks_differ=ticks_diff, share_ticks_differ=ticks_diff / max(ticks, 1),
               first_divergences=firsts[:200])
    if firsts:
        out["first_tick_median"] = float(np.median([f["tick"] for f in firsts]))
    if exact:
        out["runs_with_any_field_different"] = field_mismatch
    return out


def results_equal(A, B, sub=None):
    for k in RESULT_KEYS:
        a = A[k] if sub is None else A[k][:, :sub]
        if not np.array_equal(np.asarray(a, dtype=np.float64), np.asarray(B[k], dtype=np.float64)):
            return False
    return True


def analyse(outdir):
    outdir = Path(outdir)
    run = json.loads((outdir / "run.json").read_text())
    spec, mt, k = run["spec"], run["max_ticks"], run["spec"]["topk"]
    arms = {n: read_arm(outdir / "arms" / f"{n}.npz") for n in run["arms"]
            if (outdir / "arms" / f"{n}.npz").exists()}
    missing = [n for n in run["arms"] if n not in arms]
    if missing:
        raise EquivalenceError(f"arms missing: {missing} (run with --resume {outdir})")
    for n, a in arms.items():
        if a["meta"]["population_hash"] != run["population_hash"]:
            raise EquivalenceError(f"arm {n} has another population")
    P = len(run["labels"])
    kd, kp = spec["decision_runs"], spec["predrawn_runs"]
    S = {"summary_version": 1, "part": run["part"], "run": {k_: run[k_] for k_ in (
        "config", "config_sha256", "elites", "elites_sha256", "code_commit", "code_dirty",
        "orig_commit", "population_hash", "device_override")}, "genomes": P,
         "runs": spec["n_runs"], "checks": {}, "levels": {}}
    checks = S["checks"]

    # --- hard checks ---------------------------------------------------------------
    checks["determinism_T_equals_Tr"] = results_equal(arms["T"], arms["Tr"])
    for d, f in (("S1d", "S1"), ("S2d", "S2"), ("Td", "T"), ("Ted", "Te"), ("Tf64d", "Tf64"),
                 ("Od", "O")):
        checks[f"consistency_{d}_equals_{f}_first_{kd}_runs"] = results_equal(arms[f], arms[d], kd)
    checks["O0_equals_S1d"] = results_equal(arms["S1d"], arms["O0"])
    mods = arms["O0"]["meta"].get("modules", {})
    stale = [m for m, h in mods.items() if not (ROOT / m).exists() or sha256_file(ROOT / m) != h]
    checks["O0_ran_the_same_files_as_this_checkout"] = bool(mods) and not stale
    S["O0_files_differing"] = stale

    def ticks_of(name, R):
        return read_per_tick(arms[name]["meta"]["h5"], P, R)

    # --- L1: pre-drawn, model ------------------------------------------------------
    l1 = dict(per_run=per_run_diff(arms["Sp"], arms["Tp"], mt),
              results_identical=results_equal(arms["Sp"], arms["Tp"]))
    l1["decisions"] = decision_divergence(ticks_of("Sp", kp), ticks_of("Tp", kp), exact=True)
    checks["L1_results_identical"] = l1["results_identical"]
    checks["L1_per_tick_identical"] = (l1["decisions"]["runs_diverged"] == 0
                                       and l1["decisions"]["runs_with_any_field_different"] == 0)
    S["levels"]["L1_model"] = l1

    # --- O0 per tick: positive control of the original-code runner -----------------
    s1d_ticks = ticks_of("S1d", kd)
    o0 = decision_divergence(s1d_ticks, ticks_of("O0", kd), exact=True)
    checks["O0_per_tick_identical"] = o0["runs_diverged"] == 0 and o0["runs_with_any_field_different"] == 0
    S["O0_decisions"] = o0

    # --- L1p precision, L1c compile, L3 original: counted ----------------------------
    td_ticks = ticks_of("Td", kd)
    ted_ticks = ticks_of("Ted", kd)
    S["levels"]["L1p_precision"] = dict(per_run=per_run_diff(arms["Tf64"], arms["Te"], mt),
                                        decisions=decision_divergence(ticks_of("Tf64d", kd), ted_ticks))
    S["levels"]["L1c_compile"] = dict(per_run=per_run_diff(arms["Te"], arms["T"], mt),
                                      decisions=decision_divergence(ted_ticks, td_ticks))
    S["levels"]["L3_original"] = dict(per_run=per_run_diff(arms["S1"], arms["O"], mt),
                                      decisions=decision_divergence(s1d_ticks, ticks_of("Od", kd)),
                                      measures={key: measures(arms["S1"], arms["O"], key, k, mt)
                                                for key in ("lifespans", "eats", "distance")})

    # --- L2: live, statistical -----------------------------------------------------
    floors_pairs = {"S2": arms["S2"], "S3": arms["S3"]}
    l2 = {"per_run": {"T_vs_S1": per_run_diff(arms["S1"], arms["T"], mt),
                      "S2_vs_S1": per_run_diff(arms["S1"], arms["S2"], mt),
                      "S3_vs_S1": per_run_diff(arms["S1"], arms["S3"], mt),
                      "T2_vs_T": per_run_diff(arms["T"], arms["T2"], mt)},
          "variables": {}}
    verdict_parts = []
    for key in ("lifespans", "eats", "distance"):
        mT = measures(arms["S1"], arms["T"], key, k, mt)
        floors = {n: measures(arms["S1"], a, key, k, mt) for n, a in floors_pairs.items()}
        floors["S2_vs_S3"] = measures(arms["S3"], arms["S2"], key, k, mt)
        fl = [floors["S2"], floors["S3"]]
        crit = judge(mT, fl, key)
        floor_eq = all(f["equivalent"] for f in fl)
        entry = dict(T=mT, floors=floors, criteria=crit, floor_meets_margin=floor_eq,
                     T2_vs_T=measures(arms["T"], arms["T2"], key, k, mt),
                     primary=(key == "lifespans"))
        entry["pass"] = (False if any(v is False for v in crit.values())
                         else None if any(v is None for v in crit.values()) else True)
        l2["variables"][key] = entry
        if key == "lifespans":
            verdict_parts.append(entry["pass"])
    s2d_ticks = ticks_of("S2d", kd)
    l2["decision_statistics"] = decision_stats(decision_features(s1d_ticks), decision_features(td_ticks),
                                               decision_features(s2d_ticks), P)
    S["levels"]["L2_live"] = l2

    # --- controls ----------------------------------------------------------------
    controls = {}
    for c in spec["controls"]:
        m = measures(arms["S1"], arms[c], "lifespans", k, mt)
        fl = [measures(arms["S1"], arms[n], "lifespans", k, mt) for n in ("S2", "S3")]
        crit = judge(m, fl, "lifespans")
        controls[c] = dict(what=CONTROL_PATCHES[c][2], measures=m, criteria=crit,
                           caught=any(v is False for v in crit.values()),
                           caught_by_1_2_or_4=any(crit[x] is False
                                                  for x in ("1_rank", "2_bias", "4_pooled_D")))
    if "C1" in controls:
        checks["C1_negative_control_caught"] = controls["C1"]["caught_by_1_2_or_4"]
    S["controls"] = controls

    # --- blocks, timing ------------------------------------------------------------
    labels = np.array([lab.split()[0] for lab in run["labels"]])
    S["blocks"] = {b: dict(genomes=int((labels == b).sum()),
                           S1_mean_lifespan=float(arms["S1"]["lifespans"][labels == b].mean()),
                           T_mean_lifespan=float(arms["T"]["lifespans"][labels == b].mean()))
                   for b in sorted(set(labels))}
    S["timing_not_a_benchmark"] = {n: dict(wall_s=a["meta"].get("wall_s"),
                                           warmup_s=a["meta"].get("warmup_s"))
                                   for n, a in arms.items()}

    hard_ok = all(checks.values())
    if not hard_ok or any(v is False for v in verdict_parts):
        verdict = "FAIL"
    elif any(v is None for v in verdict_parts):
        verdict = "INCONCLUSIVE"
    else:
        verdict = "PASS"
    S["verdict"] = verdict
    (outdir / "summary.json").write_text(json.dumps(S, indent=1, sort_keys=True, default=_jsonable))
    (outdir / "report.txt").write_text(report_text(S))
    draw_figures(outdir, arms, run)
    print((outdir / "report.txt").read_text())
    return verdict


def _jsonable(o):
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    raise TypeError(type(o))


def _crit_word(v):
    return "ok" if v is True else "FAIL" if v is False else "n/a (floor misses the margin)"


def _verdict_word(v):
    return "PASS" if v is True else "FAIL" if v is False else "INCONCLUSIVE"


def _pc(x):
    return f"{100 * x:.2f}%"


def report_text(S):
    L = [f"Behavioural equivalence, part {S['part']} (plan Step 12): VERDICT {S['verdict']}",
         f"  {S['genomes']} genomes x {S['runs']} runs; config {S['run']['config']}; "
         f"code {S['run']['code_commit'][:10]}{'+dirty' if S['run']['code_dirty'] else ''}; "
         f"original {S['run']['orig_commit'][:10]}",
         "", "Hard checks (all must hold):"]
    for k, v in S["checks"].items():
        L.append(f"  [{'OK ' if v else 'BAD'}] {k}")
    if S.get("O0_files_differing"):
        L.append(f"        files the O0 runner used that differ here: {S['O0_files_differing']}")

    def per_run_lines(name, pr):
        return [f"  {name}: runs with a different lifespan {_pc(pr['share_runs_lifespans_differ'])}, "
                f"eats {_pc(pr['share_runs_eats_differ'])}, distance {_pc(pr['share_runs_distance_differ'])}; "
                f"lifespan difference median {pr['lifespan_diff_median']:+.0f} "
                f"(IQR {pr['lifespan_diff_iqr'][0]:+.0f}..{pr['lifespan_diff_iqr'][1]:+.0f}), "
                f"|diff| <= 1: {_pc(pr['share_abs_diff_le_1'])}, <= 10: {_pc(pr['share_abs_diff_le_10'])}, "
                f"<= 100: {_pc(pr['share_abs_diff_le_100'])} ({pr['runs']} runs)"]

    def dec_lines(name, d):
        s = (f"  {name} decisions: {d['runs_diverged']} of {d['runs']} runs diverge "
             f"({_pc(d['share_runs_diverged'])}); {d['ticks_differ']} of {d['ticks']} ticks differ "
             f"({_pc(d['share_ticks_differ'])})")
        if d.get("first_tick_median") is not None:
            s += f"; first divergence median tick {d['first_tick_median']:.0f}"
        return [s]

    lv = S["levels"]
    L += ["", "L1 model (pre-drawn, the same random numbers): expected 0"]
    L += per_run_lines("Sp vs Tp", lv["L1_model"]["per_run"]) + dec_lines("Sp vs Tp", lv["L1_model"]["decisions"])
    L += ["", "L1p precision (live CPU float64 vs CUDA float32, eager, the same Philox numbers)"]
    L += per_run_lines("Tf64 vs Te", lv["L1p_precision"]["per_run"]) + dec_lines("Tf64d vs Ted", lv["L1p_precision"]["decisions"])
    L += ["", "L1c compile (live eager vs compiled, the same numbers)"]
    L += per_run_lines("Te vs T", lv["L1c_compile"]["per_run"]) + dec_lines("Ted vs Td", lv["L1c_compile"]["decisions"])
    L += ["", "L3 original code vs current scalar (the same seeds)"]
    L += per_run_lines("S1 vs O", lv["L3_original"]["per_run"]) + dec_lines("S1d vs Od", lv["L3_original"]["decisions"])
    m3 = lv["L3_original"]["measures"]["lifespans"]
    L.append(f"  O vs S1 lifespan: rho {m3['rho']:.4f}, bias {m3['bias']:+.3f} "
             f"(90% CI {m3['bias_ci90'][0]:+.3f}..{m3['bias_ci90'][1]:+.3f}, margin +-{m3['margin']:.3f}), "
             f"KS share {_pc(m3['ks_share'])}, pooled D {m3['pooled_D']:.4f}")
    l2 = lv["L2_live"]
    L += ["", "L2 live (production tensor vs scalar; different random streams): per run"]
    for n, pr in l2["per_run"].items():
        L += per_run_lines(n, pr)
    for key, e in l2["variables"].items():
        t = e["T"]
        L += ["", f"L2 {key} ({'PRIMARY: decides the verdict' if e['primary'] else 'secondary, reported'}): "
                  f"{_verdict_word(e['pass'])}"
              + ("" if e["floor_meets_margin"] else "  [floor misses the margin: INCONCLUSIVE]")]
        rows = [("T vs S1", t)] + [(f"{n} vs S1" if n != "S2_vs_S3" else "S2 vs S3", f)
                                   for n, f in e["floors"].items()] + [("T2 vs T", e["T2_vs_T"])]
        for n, m in rows:
            s = (f"  {n:9s} rho {m['rho']:.4f}  bias {m['bias']:+8.3f} (90% CI {m['bias_ci90'][0]:+.3f}.."
                 f"{m['bias_ci90'][1]:+.3f}, margin +-{m['margin']:.3f})  KS share {_pc(m['ks_share'])}  "
                 f"D {m['pooled_D']:.4f}")
            if key == "lifespans":
                s += f"  capped {_pc(m['capped_share'])} (ref {_pc(m['capped_share_ref'])})  top-k overlap {m['topk_overlap']}"
            L.append(s)
        L.append("  criteria: " + ", ".join(f"{c} {_crit_word(v)}" for c, v in e["criteria"].items()))
    L += ["", "L2 decision statistics per genome (Td vs S1d, floor S2d vs S1d; reported, not asserted)"]
    for feat, d in l2["decision_statistics"].items():
        L.append(f"  {feat:14s} mean S1 {d['ref_mean']:.4f} T {d['x_mean']:.4f} S2 {d['floor_mean']:.4f}; "
                 f"rho T {d['rho']:.3f} (floor {d['rho_floor']:.3f}); bias T {d['bias']:+.4f} (floor {d['bias_floor']:+.4f})")
    L += ["", "Controls"]
    for c, d in S["controls"].items():
        L.append(f"  {c} {d['what']}: {'caught' if d['caught'] else 'NOT caught'} "
                 f"(" + ", ".join(f"{k} {_crit_word(v)}" for k, v in d["criteria"].items()) + ")")
    L += ["", "Blocks (mean lifespan S1 / T)"]
    for b, d in S["blocks"].items():
        L.append(f"  {b}: {d['genomes']} genomes, {d['S1_mean_lifespan']:.1f} / {d['T_mean_lifespan']:.1f}")
    L += ["", "Wall time per arm (by-product, NOT a benchmark: tracking on in the scalar arms)"]
    for n, d in S["timing_not_a_benchmark"].items():
        w = d["wall_s"]
        L.append(f"  {n:6s} {w:9.1f} s" + (f" (compile warm-up {d['warmup_s']:.1f} s)" if d.get("warmup_s") else ""))
    return "\n".join(L) + "\n"


def draw_figures(outdir, arms, run):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from analysis.plot_neuron_scaling import INK, INK2, SERIES, SURFACE, _style
    colors = list(SERIES) + ["#8a5cd1"]
    mt = run["max_ticks"]
    S1, S2, S3, T, O = (arms[n]["lifespans"].astype(np.float64) for n in ("S1", "S2", "S3", "T", "O"))
    written = []

    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(outdir / f"{name}.{ext}", dpi=200, facecolor=SURFACE)
            written.append(f"{name}.{ext}")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 5), facecolor=SURFACE)
    m1 = S1.mean(1)
    ax.scatter(m1, S2.mean(1), s=10, color=colors[1], label="S2 (scalar, other seeds)", alpha=0.8)
    ax.scatter(m1, T.mean(1), s=10, color=colors[0], label="T (tensor, production)", alpha=0.8)
    lo, hi = m1.min() * 0.9, m1.max() * 1.1
    ax.plot([lo, hi], [lo, hi], color=INK2, linewidth=0.8)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK)
    _style(ax, "Per-genome mean lifespan", "S1 (scalar) mean lifespan", "mean lifespan")
    save(fig, "genome_means")

    fig, ax = plt.subplots(figsize=(6, 4.2), facecolor=SURFACE)
    for X, lab, c in ((S2, "S2 - S1 (floor)", colors[1]), (T, "T - S1", colors[0])):
        mx = X.mean(1)
        ax.scatter((mx + m1) / 2, mx - m1, s=10, color=c, label=lab, alpha=0.8)
        ax.axhline((mx - m1).mean(), color=c, linewidth=1.2, label=f"mean bias {lab.split(' (')[0]}")
    marg = MARGIN * m1.mean()
    for y in (-marg, marg):
        ax.axhline(y, color=INK2, linestyle="--", linewidth=0.8)
    ax.axhline(0, color=INK2, linewidth=0.6)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK)
    _style(ax, f"Bland-Altman of per-genome means (dashed: the +-{MARGIN:.0%} margin for the mean bias)",
           "mean of the two (ticks)", "difference (ticks)")
    save(fig, "bland_altman")

    fig, ax = plt.subplots(figsize=(6, 4.2), facecolor=SURFACE)
    for X, lab, c in ((S1, "S1", colors[2]), (S2, "S2", colors[1]), (S3, "S3", colors[3]), (T, "T", colors[0])):
        x = np.sort(X.ravel())
        ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=c, label=lab, linewidth=1.2)
    ax.axvline(mt, color=INK2, linestyle=":", linewidth=0.8)
    ax.set_xscale("log")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK)
    _style(ax, f"Pooled lifespan ECDF (dotted: cap {mt})", "lifespan (ticks, log)", "share of runs")
    save(fig, "lifespan_ecdf")

    fig, ax = plt.subplots(figsize=(6, 4.2), facecolor=SURFACE)
    for X, lab, c in ((S2, "S2 - S1 (floor)", colors[1]), (T, "T - S1", colors[0]), (O, "O - S1 (original)", colors[3])):
        d = (X - S1).ravel()
        lim = max(1, np.abs(d).max())
        bins = np.unique(np.concatenate([-np.logspace(0, np.log10(lim + 1), 30)[::-1] + 0.5, [-0.5, 0.5],
                                         np.logspace(0, np.log10(lim + 1), 30) - 0.5]))
        ax.hist(d, bins=bins, histtype="step", color=c, label=lab, linewidth=1.2)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("log")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK)
    _style(ax, "Per-run lifespan difference (paired worlds)", "difference (ticks, symlog)", "runs (log)")
    save(fig, "per_run_difference")
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def default_outdir(part):
    gpu = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0).replace(" ", "-")
    except Exception:  # noqa: BLE001 -- only names the folder
        pass
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return OUT_BASE / f"{ts}_{gpu}_part{part}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--part", choices=sorted(PARTS))
    ap.add_argument("--orig-worktree")
    ap.add_argument("--orig-commit")
    ap.add_argument("--current-worktree")
    ap.add_argument("--out")
    ap.add_argument("--resume", metavar="DIR")
    ap.add_argument("--analyse", metavar="DIR")
    ap.add_argument("--run-arm", help=argparse.SUPPRESS)
    ap.add_argument("--dir", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)

    if a.run_arm:
        return run_arm(a.dir, a.run_arm)
    if a.analyse:
        return {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}[analyse(a.analyse)]
    if a.resume:
        outdir = Path(a.resume)
    else:
        if not (a.part and a.orig_worktree and a.orig_commit and a.current_worktree):
            ap.error("a new run needs --part, --orig-worktree, --orig-commit and --current-worktree")
        outdir = Path(a.out) if a.out else default_outdir(a.part)
        if outdir.exists():
            raise EquivalenceError(f"{outdir} exists; use --resume {outdir}")
        run = plan_run(a.part, PARTS[a.part], outdir, a.orig_worktree, a.orig_commit,
                       a.current_worktree)
        print(f"[plan] part {a.part}: {len(run['labels'])} genomes, {len(run['arms'])} arms, "
              f"population {run['population_hash'][:12]}, output {outdir}", flush=True)
        system_snapshot(outdir)
    execute(outdir, resume=bool(a.resume))
    return {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}[analyse(outdir)]


if __name__ == "__main__":
    try:
        sys.exit(main())
    except EquivalenceError as e:
        sys.exit(f"ERROR: {e}")
