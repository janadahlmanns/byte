#!/usr/bin/env python3
"""Real-loop timing of one code version (plan Step 13 follow-up).

Isolated decide-step timings (tests/modulation_variants.py) said Step 13 was faster than
Step 11, yet real runs got slower: a fixed width lets torch.compile specialise, while a
real run's width changes with every compaction. This times what production does --
`eval_generation_batch` with compaction and compile, over an evolving population
(elites + mutants, as the EA makes them) -- against the code in `--code`, so the same
measurement can be repeated on several commits (git worktrees).

Usage (one line per version; see the plan for the commit list)
-----
    git worktree add /tmp/bv_dense 0fc6608785
    python tests/version_bench.py --code /tmp/bv_dense --config ea_from_lookup_tensor_cuda
"""

import argparse
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--code", required=True, help="repository checkout whose code is timed")
ap.add_argument("--config", required=True)
ap.add_argument("--genomes", type=int, default=260)
ap.add_argument("--runs", type=int, default=None, help="default: 300 at n <= 20, else 30")
ap.add_argument("--gens", type=int, default=4)
args = ap.parse_args()
CODE = os.path.abspath(args.code)
sys.path.insert(0, CODE)
os.chdir(CODE)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch._dynamo.utils as du  # noqa: E402
import yaml  # noqa: E402

import mvb_torch.generation as G  # noqa: E402
from mvb.genome.generate_genome_lookup_hard import generate_lookup_hard_genome  # noqa: E402
from mvb.genome.generate_genome_mutate_simple import generate_genome_mutate_simple  # noqa: E402
from mvb.genome.generate_genome_random import generate_random_genome  # noqa: E402
from mvb_torch.decision import build_output_spec  # noqa: E402
from mvb_torch.genome_codec import encode_genomes  # noqa: E402


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()
    elif DEV == "mps":
        torch.mps.synchronize()


DEV = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
G.configure_device(DEV)
cfg = yaml.safe_load(open(os.path.join(CODE, "configs/experiments", f"{args.config}.yaml")))
n = int(cfg["brain"]["n_neurons"])
R = args.runs or (300 if n <= 20 else 30)
cfg["experiment"]["n_runs"] = R
P = args.genomes
make = {"lookup_hard": generate_lookup_hard_genome,
        "random": generate_random_genome}[cfg["experiment"]["genome_type"]]
contraction = G.live_contraction(DEV)
sim, spec = G.sim_config_from_yaml(cfg), build_output_spec(cfg["brain"])
width = min(P * R, 78000 if n <= 20 else 3000)

rng = np.random.default_rng(5)
pop = [make(cfg, rng_seed=i) for i in range(P)]
commit = os.popen(f"git -C {CODE} log -1 --format=%h").read().strip()
print(f"version {commit}: {DEV}, contraction {contraction}, config {args.config}, n = {n}, "
      f"{P} genomes x {R} runs, width {width}, compile on")
total = 0.0
for gen in range(args.gens):
    try:
        b = encode_genomes(pop, cfg["brain"], device=DEV, dtype=torch.float32,
                           dense_modulation=(contraction == "einsum"))
    except TypeError:                       # versions before sparse modulation
        b = encode_genomes(pop, cfg["brain"], device=DEV, dtype=torch.float32)
    seeds = G.draw_seeds(*G.make_simulation_rngs(gen + 1), P, R)
    g0 = int(du.counters["stats"]["unique_graphs"])
    sync()
    t0 = time.perf_counter()
    res = G.eval_generation_batch(b, spec, sim, seeds, width=width, mode="live",
                                  philox_rounds=10, contraction=contraction, compile=True)
    sync()
    dt = time.perf_counter() - t0
    if gen:
        total += dt
    print(f"  gen {gen}: {dt:7.2f} s, {res.iterations} iterations, "
          f"{1e3 * dt / res.iterations:6.1f} ms/iteration, "
          f"{1e6 * dt / int(res.lifespans.sum()):6.2f} µs/run-iteration, "
          f"new graphs {int(du.counters['stats']['unique_graphs']) - g0}", flush=True)
    elites = pop[:10]
    pop = elites + [generate_genome_mutate_simple(elites[i % 10], 0.1, 0.2, rng)
                    for i in range(P - 10)]
print(f"  generations 1-{args.gens - 1}: {total:.2f} s")
