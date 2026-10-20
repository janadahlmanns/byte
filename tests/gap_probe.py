#!/usr/bin/env python3
"""Why is the production path slower than the width probe? (plan Step 16 follow-up)

At n = 50 on the RTX 3070, tests/width_probe.py ran a generation at ~37 us per
run-tick, the sweep's run_ea generations at ~63 -- same width (~8,900), same
iterations (232 in generation 0), similar run-ticks. This isolates the differences,
one generation each, in one process:

  A  probe path, the probe's population (random + 0-3 mutations), warm-up first
  B  probe path, run_ea's generation-0 population (random genomes, seeds 1..P)
  C  as B, but with the Step 18a compile warm-up switched off: the first compile happens
     at full width, as before Step 18 (the slow kernels)
  D  the production path: the adapter's evaluator (as run_ea calls it), B's population
     and seeds
  E  D while nvidia-smi is polled once per second, exactly as simulate/run_selected.py
     does during a config (GpuSampler) -- the sweep ran under it, the width probe not

Each variant starts from a fresh torch.compile state (dynamo reset) and runs its
generation twice; the second run is the measured one. B and D use the same genomes
and seeds, so their lifespans must be identical -- the only assertion.

Usage
-----
    python -m tests.gap_probe                          # cuda, n = 50, variants A,B,C
    python -m tests.gap_probe --variants D,E           # the production path, own process
    python -m tests.gap_probe --device mps --runs 20   # smoke
    python -m tests.gap_probe --n 40 --variants A,B --profile   # top GPU kernels per variant

--profile records 10 iterations (20-29) of the measured generation with torch.profiler
and prints the top kernels by GPU time (probe-path variants A, B, C only).
"""

import argparse
import contextlib
import io
import os
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mvb_torch.adapter as A  # noqa: E402
import mvb_torch.generation as G  # noqa: E402
import simulate.run_selected as RS  # noqa: E402
from mvb.genome.generate_genome_mutate_simple import generate_genome_mutate_simple  # noqa: E402
from mvb.genome.generate_genome_random import generate_random_genome  # noqa: E402
from mvb_torch.decision import build_output_spec  # noqa: E402
from mvb_torch.genome_codec import encode_genomes, sparse_slot_bytes  # noqa: E402
from tests.devices import sync  # noqa: E402

DT = torch.float32


def populations(cfg, P):
    rng = np.random.default_rng(5)
    probe = []
    for i in range(P):
        g = generate_random_genome(cfg, rng_seed=i)
        for _ in range(i % 4):
            g = generate_genome_mutate_simple(g, 0.1, 0.2, rng)
        probe.append(g)
    seed = cfg["experiment"]["random_genome"]["wiring_randomization_seed"]
    ea = [generate_random_genome(cfg, rng_seed=seed + i) for i in range(P)]
    return probe, ea


def profiled(dev, fn, first=20, active=10):
    """Run fn(hook) with torch.profiler on iterations [first, first + active)."""
    from torch.profiler import ProfilerActivity, profile, schedule
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev == "cuda" else [])
    with profile(activities=acts, schedule=schedule(wait=first - 1, warmup=1, active=active,
                                                     repeat=1)) as prof:
        out = fn(lambda it, state, refill: prof.step())
    key = "self_cuda_time_total" if dev == "cuda" else "self_cpu_time_total"
    print(f"     top kernels over iterations {first}-{first + active - 1} (by {key}):")
    for line in prof.key_averages().table(sort_by=key, row_limit=12, max_name_column_width=70
                                          ).splitlines():
        print("       " + line)
    return out


def layout(batch):
    return (f"L = {len(batch.mod_k)} list positions, widths "
            f"{[int(k.shape[1]) for k in batch.mod_k]}, E = {int(batch.mod_edge.shape[1])}")


def probe_path(dev, cfg, genomes, warm_up, compile_warmup=True):
    """As tests/width_probe.py: encode, width rule, eval_generation_batch."""
    x = cfg["experiment"]
    P, R = len(genomes), int(x["n_runs"])
    contraction = G.live_contraction(dev)
    batch = encode_genomes(genomes, cfg["brain"], device=dev, dtype=DT,
                           dense_modulation=(contraction == "einsum"))
    sim, spec = G.sim_config_from_yaml(cfg), build_output_spec(cfg["brain"])
    n = int(cfg["brain"]["n_neurons"])
    slot = G.estimate_slot_bytes(
        height=sim.height, width_cells=sim.width, n=n,
        K=int(n * float(cfg["brain"]["max_decision_delay"])), dtype=DT, mode="live",
        max_ticks=sim.max_ticks,
        sparse_slot_bytes=sparse_slot_bytes(batch) if contraction == "sequential" else None)
    W, _ = G.choose_width(P, R, slot, dev, batch_bytes=G.genome_batch_bytes(batch))
    rounds = int(x["evaluator"]["philox_rounds"])
    if warm_up:
        warm_sim = G.sim_config_from_yaml({**cfg, "experiment": {**x, "n_runs": 2}})
        G.eval_generation_batch(batch, spec, warm_sim, G.draw_seeds(*G.make_simulation_rngs(2), P, 2),
                                width=min(W, 2 * P), mode="live", philox_rounds=rounds,
                                contraction=contraction, compile=True,
                                _compile_warmup=compile_warmup)
    seeds = G.draw_seeds(*G.make_simulation_rngs(int(x["simulation_seed"])), P, R)

    print(f"     population: {layout(batch)}", flush=True)

    def run(hook=None):
        sync(dev)
        t0 = time.perf_counter()
        res = G.eval_generation_batch(batch, spec, sim, seeds, width=W, mode="live",
                                      philox_rounds=rounds, contraction=contraction,
                                      compile=True, _state_hook=hook,
                                      _compile_warmup=compile_warmup)
        sync(dev)
        return time.perf_counter() - t0, res.lifespans.cpu().numpy().reshape(P, R), res.iterations
    return W, run


def adapter_path(dev, cfg, genomes):
    """As run_ea calls it: make_tensor_evaluator, then the evaluator's signature."""
    x, w, wm = cfg["experiment"], cfg["world"], cfg["worm"]
    P = len(genomes)
    ev_cfg = dict(x["evaluator"], device=dev)
    phases = sorted(cfg["food"], key=lambda p: p["phase_from"])
    with contextlib.redirect_stdout(io.StringIO()):
        ev = A.make_tensor_evaluator(ev_cfg, cfg, None)

    def run():
        rn, rd, rw = G.make_simulation_rngs(int(x["simulation_seed"]))
        args = (genomes, cfg, "unused", "unused", False, False, False, False, False, 4, 4,
                P, rn, rd, rw, wm["decisionmaking"]["version"], x["max_ticks"], x["n_runs"],
                w["grid_width"], w["grid_height"], w["start_pos"], wm["speed"],
                wm["energy_capacity"], wm["metabolic_rate"], wm["movement_cost"],
                wm["sensors"]["active"], phases[0], cfg["brain"])
        buf = io.StringIO()
        sync(dev)
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(buf):
            out, _ = ev(*args, switch_phases=phases[1:], randomness_cfg=None)
        sync(dev)
        dt = time.perf_counter() - t0
        it = int(re.search(r"\((\d+) iterations", buf.getvalue()).group(1))
        width = int(re.search(r"width (\d+) of", buf.getvalue()).group(1))
        return dt, np.stack([out[v]["lifespan_vector"] for v in range(P)]), it, width
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--runs", type=int, default=None, help="default: the config's n_runs")
    ap.add_argument("--device", default=None, help="default: the config's evaluator.device")
    ap.add_argument("--variants", default="A,B,C", help="A,B,C in one process; D,E in another")
    ap.add_argument("--profile", action="store_true",
                    help="profile 10 iterations of the measured run (variants A, B, C)")
    a = ap.parse_args()
    points = {c["brain"]["n_neurons"]: c for _, c in
              RS.expand_sweep(RS.ROOT / "configs" / "sweeps" / "neuron_scaling.yaml")}
    cfg = points[a.n]
    if a.runs:
        cfg["experiment"]["n_runs"] = a.runs
    dev = a.device or cfg["experiment"]["evaluator"]["device"]
    G.configure_device(dev)
    P = int(cfg["experiment"]["population_size"])
    variants = a.variants.split(",")
    if set(variants) & {"D", "E"} and not set(variants) <= {"D", "E"}:
        # D/E build the production evaluator, which keeps its compiled modules in their own
        # folder from the process's first compile (plan Step 18b).
        raise SystemExit("D and E use the production compile-cache folder; run them in their "
                         "own process: --variants D,E")
    probe_pop, ea_pop = populations(cfg, P)
    print(f"gap probe on {dev}"
          + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else "")
          + f": n = {a.n}, {P} genomes x {cfg['experiment']['n_runs']} runs; code "
          f"{RS.code_version()}", flush=True)
    results = {}
    for v in variants:
        torch._dynamo.reset()
        if dev == "cuda":
            torch.cuda.empty_cache()
        t0 = time.perf_counter()
        if v == "A":
            W, run = probe_path(dev, cfg, probe_pop, warm_up=True)
        elif v == "B":
            W, run = probe_path(dev, cfg, ea_pop, warm_up=True)
        elif v == "C":
            W, run = probe_path(dev, cfg, ea_pop, warm_up=False, compile_warmup=False)
        elif v in ("D", "E"):
            W, run = None, adapter_path(dev, cfg, ea_pop)
        else:
            raise SystemExit(f"unknown variant {v!r}")
        sampler = RS.GpuSampler() if v == "E" else contextlib.nullcontext()
        with sampler:
            first = run()
            setup = time.perf_counter() - t0
            if a.profile and v in ("A", "B", "C"):
                profiled(dev, run)               # a third run, profiled; not timed
            second = run()
        dt, L, it = second[:3]
        W = W if W is not None else second[3]
        ticks = int(L.sum())
        results[v] = L
        print(f"  {v}: width {W}, {it} iterations, run-ticks {ticks}: 1st run {first[0]:7.2f} s "
              f"(setup + 1st {setup:6.1f} s), 2nd run {dt:7.2f} s = "
              f"{1e3 * dt / it:6.1f} ms/iteration, {1e6 * dt / ticks:6.2f} us/run-tick", flush=True)
        if v == "E":
            print(f"     (nvidia-smi sampled once per second during E; peak memory.used "
                  f"{sampler.peak} MiB)", flush=True)
    if "B" in results and "D" in results:
        same = np.array_equal(results["B"], results["D"]) and (
            "E" not in results or np.array_equal(results["D"], results["E"]))
        print(f"\n  B, D (and E) lifespans identical (same genomes and seeds): {same}")
        if not same:
            print("FAILED")
            return 1
    print("\nDone. Please send this whole output.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
