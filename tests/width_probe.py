#!/usr/bin/env python3
"""Is a wider batch faster? (plan Step 16a)

From n = 20 up, the width rule caps the batch by memory, yet torch's peak stayed at
~2.3 GB of the 5.6 GB budget in the first sweep report: the per-slot estimate is ~3x
too high at n = 50. Before changing the estimate, this measures whether a wider batch
would be faster at all.

For each n (the `neuron_scaling` sweep points), one fixed population and seed set are
evaluated as a real generation (`eval_generation_batch`: queue, compaction, compile) at
the width rule's own width W and at multiples of it. Per width: wall time, us per
run-tick, iterations, torch peak memory and new compiled graphs. Out of memory is
reported and the probe continues. The lifespans must be identical at every width
(width is scheduling only) -- the one thing this script asserts.

At the end: the measured memory per slot (slope of peak allocated over width) next to
the estimate's terms, and the plan's decision rule 16b, fixed before measuring: fix the
estimate if, at n = 40 or 50, the widest width that fits is >= 10% faster per run-tick
than W.

Usage
-----
    python -m tests.width_probe                       # cuda, n = 30, 40, 50 (~20-30 min)
    python -m tests.width_probe --device mps --n 30 --runs 20 --factors 1,2   # smoke
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch._dynamo.utils as du

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mvb_torch.generation as G  # noqa: E402
import mvb_torch.philox as philox  # noqa: E402
import simulate.run_selected as RS  # noqa: E402
from mvb.genome.generate_genome_mutate_simple import generate_genome_mutate_simple  # noqa: E402
from mvb.genome.generate_genome_random import generate_random_genome  # noqa: E402
from mvb_torch.decision import build_output_spec  # noqa: E402
from mvb_torch.genome_codec import encode_genomes, sparse_slot_bytes  # noqa: E402
from tests.devices import sync  # noqa: E402

DT = torch.float32
SWEEP = RS.ROOT / "configs" / "sweeps" / "neuron_scaling.yaml"
DECISION_MIN_GAIN = 0.10          # plan 16b, fixed before measuring
DECISION_N = (40, 50)


def is_oom(e):
    return isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower()


def memory(dev):
    """(peak allocated, peak reserved) MiB since the last reset; None where unknown."""
    if dev == "cuda":
        return (torch.cuda.max_memory_allocated() / 2**20,
                torch.cuda.max_memory_reserved() / 2**20)
    if dev == "mps":
        return None, torch.mps.driver_allocated_memory() / 2**20   # no peak counter
    return None, None


def reset_memory(dev):
    if dev == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif dev == "mps":
        torch.mps.empty_cache()


def estimate_terms(cfg, n, slot_bytes, sparse_bytes):
    """The estimate's per-slot terms in KB (estimate_slot_bytes, before PEAK_FACTOR)."""
    K = int(n * float(cfg["brain"]["max_decision_delay"]))
    ds = torch.tensor([], dtype=DT).element_size()
    cells = cfg["world"]["grid_width"] * cfg["world"]["grid_height"]
    t = {"philox words": K * philox.blocks_per_brain_tick(n) * 4 * 8 * 6,
         "genome copy": (3 * n * n + n + 1) * ds,
         "modulation copy": sparse_bytes if sparse_bytes is not None else n ** 3 * ds,
         "noise": K * n * ds,
         "grid + reseed": 4 * cells,
         "brain + indices": (n * n + n) * ds + 128}
    return {k: v / 1024 for k, v in t.items()}, slot_bytes / 1024


def probe(dev, n, cfg, factors, runs, population):
    x = cfg["experiment"]
    P = population or int(x["population_size"])
    R = runs or int(x["n_runs"])
    x["population_size"], x["n_runs"] = P, R
    contraction = G.live_contraction(dev)
    rng = np.random.default_rng(5)
    genomes = []
    for i in range(P):                     # generation-1-like: random + some mutations
        g = generate_random_genome(cfg, rng_seed=i)
        for _ in range(i % 4):
            g = generate_genome_mutate_simple(g, 0.1, 0.2, rng)
        genomes.append(g)
    batch = encode_genomes(genomes, cfg["brain"], device=dev, dtype=DT,
                           dense_modulation=(contraction == "einsum"))
    sim, spec = G.sim_config_from_yaml(cfg), build_output_spec(cfg["brain"])
    sparse = sparse_slot_bytes(batch) if contraction == "sequential" else None
    slot = G.estimate_slot_bytes(
        height=sim.height, width_cells=sim.width, n=n,
        K=int(n * float(cfg["brain"]["max_decision_delay"])), dtype=DT, mode="live",
        max_ticks=sim.max_ticks, sparse_slot_bytes=sparse)
    W, why = G.choose_width(P, R, slot, dev, batch_bytes=G.genome_batch_bytes(batch))
    rounds = int(x["evaluator"]["philox_rounds"])
    seeds = G.draw_seeds(*G.make_simulation_rngs(1), P, R)
    print(f"\n{'=' * 78}\nn = {n}: {P} genomes x {R} runs on {dev}, contraction {contraction}, "
          f"compile on\n  width rule: W = {W} ({why})\n{'=' * 78}", flush=True)

    # Compile once, on a few runs of the same genomes (the width dimension is dynamic).
    warm = G.draw_seeds(*G.make_simulation_rngs(2), P, 2)
    t0 = time.perf_counter()
    warm_sim = G.sim_config_from_yaml({**cfg, "experiment": {**x, "n_runs": 2}})
    G.eval_generation_batch(batch, spec, warm_sim, warm, width=min(W, 2 * P), mode="live",
                            philox_rounds=rounds, contraction=contraction, compile=True)
    sync(dev)
    print(f"  warm-up (compile) {time.perf_counter() - t0:.1f} s", flush=True)

    widths = sorted({min(P * R, max(2, int(round(f * W)))) for f in factors})
    rows, ref = [], None
    for w in widths:
        snap = RS.gpu_snapshot()
        busy = ""
        if isinstance(snap.get("gpus"), list):
            busy = ", ".join(f"GPU{g['index']} {g['util_pct']}% {g['mem_used_mib']} MiB"
                             for g in snap["gpus"])
        reset_memory(dev)
        g0 = int(du.counters["stats"]["unique_graphs"])
        try:
            sync(dev)
            t0 = time.perf_counter()
            res = G.eval_generation_batch(batch, spec, sim, seeds, width=w, mode="live",
                                          philox_rounds=rounds, contraction=contraction,
                                          compile=True)
            sync(dev)
            dt = time.perf_counter() - t0
        except Exception as e:  # noqa: BLE001 -- OOM is a result here, anything else is not
            if not is_oom(e):
                raise
            print(f"  width {w:>6}: OOM ({type(e).__name__})", flush=True)
            rows.append({"width": w, "oom": True})
            reset_memory(dev)
            continue
        alloc, reserved = memory(dev)
        ticks = int(res.lifespans.sum())
        if ref is None:
            ref = res.lifespans.clone()
        same = torch.equal(res.lifespans, ref)
        row = {"width": w, "oom": False, "s": dt, "us": 1e6 * dt / ticks, "it": res.iterations,
               "alloc": alloc, "reserved": reserved, "same": same,
               "graphs": int(du.counters["stats"]["unique_graphs"]) - g0}
        rows.append(row)
        print(f"  width {w:>6} ({w / W:4.2f} W): {dt:8.2f} s, {row['us']:7.2f} us/run-tick, "
              f"{res.iterations} iterations, peak alloc "
              f"{'-' if alloc is None else f'{alloc:.0f}'} MiB, reserved "
              f"{'-' if reserved is None else f'{reserved:.0f}'} MiB, new graphs "
              f"{row['graphs']}, lifespans identical to width {widths[0]}: {same}"
              + (f"   [before: {busy}]" if busy else ""), flush=True)

    ok = [r for r in rows if not r["oom"]]
    terms, est_kb = estimate_terms(cfg, n, slot, sparse)
    print(f"\n  estimate: {est_kb:.0f} KB per slot = PEAK_FACTOR {G.PEAK_FACTOR} x ("
          + ", ".join(f"{k} {v:.0f}" for k, v in terms.items()) + ") KB")
    fit = [r for r in ok if r["alloc"] is not None]
    slope = None
    if len(fit) >= 2:
        slope, icpt = np.polyfit([r["width"] for r in fit], [r["alloc"] for r in fit], 1)
        print(f"  measured: {slope * 1024:.0f} KB per slot (slope of peak allocated), "
              f"fixed part {icpt:.0f} MiB -> estimate / measured = "
              f"{est_kb / (slope * 1024):.2f}")
    elif dev != "cuda":
        print(f"  measured per slot: unavailable on {dev} (no peak counter)")
    gain = None
    at_w = [r for r in ok if r["width"] == min(P * R, W)]
    wider = [r for r in ok if r["width"] > W]
    if at_w and wider:
        best = min(ok, key=lambda r: r["us"])
        gain = 1 - wider[-1]["us"] / at_w[0]["us"]
        print(f"  widest width that fits ({wider[-1]['width']}) vs W ({W}): {100 * gain:+.1f}% "
              f"per run-tick (positive = faster); fastest: width {best['width']}")
    elif not wider:
        print("  no width above W ran (all runs already fit at W, or OOM)")
    if at_w and len(ok) > 1:
        best = min(ok, key=lambda r: r["us"])
        print(f"  fastest width {best['width']} ({best['width'] / W:.2f} W): "
              f"{100 * (1 - best['us'] / at_w[0]['us']):+.1f}% per run-tick vs W")
    return {"n": n, "rows": rows, "gain": gain, "slope": slope, "est_kb": est_kb,
            "all_same": all(r["same"] for r in ok)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", default="30,40,50")
    ap.add_argument("--factors", default="1,1.5,2,2.5,3")
    ap.add_argument("--runs", type=int, default=None, help="default: the config's n_runs")
    ap.add_argument("--population", type=int, default=None, help="default: the config's")
    ap.add_argument("--device", default=None, help="default: the config's evaluator.device")
    a = ap.parse_args()
    ns = [int(v) for v in a.n.split(",")]
    factors = [float(v) for v in a.factors.split(",")]
    points = {cfg["brain"]["n_neurons"]: cfg for _, cfg in RS.expand_sweep(SWEEP)}
    missing = [n for n in ns if n not in points]
    if missing:
        raise SystemExit(f"n = {missing} not in {SWEEP.name} (points: {sorted(points)})")
    dev = a.device or points[ns[0]]["experiment"]["evaluator"]["device"]
    G.configure_device(dev)
    print(f"width probe on {dev}: torch {torch.__version__}"
          + (f", {torch.cuda.get_device_name(0)}" if dev == "cuda" else "")
          + f"; code {RS.code_version()}")
    results = [probe(dev, n, points[n], factors, a.runs, a.population) for n in ns]

    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    for r in results:
        line = f"  n = {r['n']}: "
        line += ("gain " + (f"{100 * r['gain']:+.1f}%" if r["gain"] is not None else "-"))
        if r["slope"] is not None:
            line += f", estimate / measured per slot {r['est_kb'] / (r['slope'] * 1024):.2f}"
        line += f", OOM at {[x['width'] for x in r['rows'] if x['oom']] or 'none'}"
        line += f", lifespans identical at every width: {r['all_same']}"
        print(line)
    decided = [r for r in results if r["n"] in DECISION_N and r["gain"] is not None]
    if decided and any(r["gain"] >= DECISION_MIN_GAIN for r in decided):
        print(f"\n16b: a wider batch is >= {100 * DECISION_MIN_GAIN:.0f}% faster at n = "
              f"{[r['n'] for r in decided if r['gain'] >= DECISION_MIN_GAIN]} -> fixing the "
              f"estimate (16c) is worth it.")
    elif decided:
        print(f"\n16b: no gain >= {100 * DECISION_MIN_GAIN:.0f}% at n = "
              f"{[r['n'] for r in decided]} -> keep the estimate as it is.")
    else:
        print(f"\n16b: not decided (needs n = 40 or 50 measured at W and above).")
    bad = [r["n"] for r in results if not r["all_same"]]
    if bad:
        print(f"\nFAILED: lifespans differ between widths at n = {bad} -- width must never "
              f"change results.")
        return 1
    print("\nDone. Please send this whole output.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
