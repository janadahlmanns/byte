#!/usr/bin/env python3
"""Where does the EA's time outside the evaluator go? (EvoTorch question, 2026-10-29)

In the large config (1000 genomes, 50 neurons) a generation took ~472 s between saves,
of which the evaluator's own timer covered ~377 s: ~95 s per generation is EA. This
times, with the real code on real genomes, every step run_ea does per generation
outside that timer, plus the genome encoding (inside the evaluator's timer, but it is
the "Python genomes -> tensors" cost an on-device EA would also remove):

  mutation        run_ea step 9: P - elite children via mvb's generate_genome_mutate_simple
  seeds           the adapter's draw_seeds (before its timer starts)
  encode          genome_codec.encode_genomes (inside the evaluator's timer)
  selection       pick_elite_deterministic on P x R lifespans
  hdf5            save_elite_to_hdf5 + write_generation_stats_to_hdf5
  lifespans dict  the per-genome result dict the adapter returns

Then a cProfile of the mutation: which lines inside it take the time. CPU only, no GPU,
no results written outside a temporary folder. Read-only on mvb/.

Usage
-----
    python -m tests.ea_profile                                   # large config
    python -m tests.ea_profile --config ea_from_lookup_tensor_cuda   # n = 11, for contrast
"""

import argparse
import cProfile
import io
import os
import pstats
import sys
import tempfile
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mvb_torch.generation as G  # noqa: E402
import simulate.run_ea as EA  # noqa: E402
from mvb.genome.generate_genome_mutate_simple import generate_genome_mutate_simple  # noqa: E402
from mvb_torch.genome_codec import encode_genomes  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def timed(label, fn, results):
    t0 = time.perf_counter()
    out = fn()
    results.append((label, time.perf_counter() - t0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ea_from_random_large_tensor_cuda")
    ap.add_argument("--children", type=int, default=None,
                    help="mutate only this many children (default: all P - elite) and scale")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs/experiments", f"{a.config}.yaml")))
    x, ea = cfg["experiment"], cfg["experiment"]["evolutionary_algorithm"]
    P, R, E = int(x["population_size"]), int(x["n_runs"]), int(ea["elite_size"])
    n = int(cfg["brain"]["n_neurons"])
    rate, strength = float(ea["mutation_rate"]), float(ea["mutation_strength"])
    gen = EA.load_genome_generator(x["genome_type"])
    seed = x.get("random_genome", {}).get("wiring_randomization_seed", 0)
    print(f"EA profile: {a.config}, P = {P}, elite = {E}, R = {R}, n = {n}, mutation_rate "
          f"{rate} -> {max(1, int(n * n * rate))} actions per child; CPU {os.cpu_count()} cores")

    rng = np.random.default_rng(7)
    genomes0 = [gen(cfg, rng_seed=seed + i) for i in range(P)]
    # Realistic elites: one round of mutation from generation 0, as from generation 1 on.
    elites = [generate_genome_mutate_simple(genomes0[i], rate, strength, rng) for i in range(E)]
    n_children = P - E
    counts = [n_children // E] * E
    counts[-1] += n_children % E
    todo = a.children or n_children

    res = []

    def mutate():
        out, k = [], 0
        for g, c in zip(elites, counts):
            for _ in range(c):
                if k == todo:
                    return out
                out.append(generate_genome_mutate_simple(g, rate, strength, rng))
                k += 1
        return out
    children = timed("mutation", mutate, res)
    scale = n_children / len(children)
    if scale != 1:
        res[-1] = ("mutation", res[-1][1] * scale)
        print(f"  (mutated {len(children)} of {n_children} children, time scaled x{scale:.1f})")
    pop = elites + (children * int(np.ceil(n_children / len(children))))[:n_children]

    rn, rd, rw = G.make_simulation_rngs(1)
    timed("seeds", lambda: G.draw_seeds(rn, rd, rw, P, R), res)
    timed("encode", lambda: encode_genomes(pop, cfg["brain"], device="cpu", dtype=torch.float32,
                                           dense_modulation=False), res)
    life = np.random.default_rng(3).integers(1, 200, size=(P, R)).astype("i4")
    seeds = G.draw_seeds(rn, rd, rw, P, R)
    lifed = timed("lifespans dict", lambda: {
        v: {"lifespan_vector": life[v], "seeds_noise_all_runs": seeds.noise_seeds[v],
            "seeds_decision_all_runs": seeds.decision_seeds[v]} for v in range(P)}, res)
    elite_idx, stats = timed("selection", lambda: EA.pick_elite_deterministic(
        lifed, ea["elite_selection_metric"], E), res)
    with tempfile.TemporaryDirectory() as tmp:
        h5 = EA.initialize_hdf5_file(tmp + "/", "profile", cfg, int(ea["num_generations"]))

        def save():
            EA.save_elite_to_hdf5(h5, [pop[i] for i in elite_idx],
                                  [lifed[i]["lifespan_vector"] for i in elite_idx],
                                  [lifed[i]["seeds_noise_all_runs"] for i in elite_idx],
                                  [lifed[i]["seeds_decision_all_runs"] for i in elite_idx])
            EA.write_generation_stats_to_hdf5(h5, 1, stats, seeds.run_seeds)
        timed("hdf5", save, res)

    outside = sum(t for k, t in res if k != "encode")
    print(f"\n  step               seconds per generation")
    for k, t in res:
        note = "  (inside the evaluator's timer)" if k == "encode" else ""
        print(f"  {k:16s} {t:10.2f}{note}")
    print(f"  {'-' * 16} {'-' * 10}")
    print(f"  {'outside total':16s} {outside:10.2f}   (compare: ~95 s measured between saves, "
          f"university PC, large config)")

    print("\ncProfile of the mutation (one child):")
    pr = cProfile.Profile()
    pr.enable()
    generate_genome_mutate_simple(elites[0], rate, strength, np.random.default_rng(11))
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(8)
    print("\n".join(line for line in s.getvalue().splitlines() if line.strip())[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
