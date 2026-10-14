#!/usr/bin/env python3
"""Where does a large-brain generation spend its time? (plan Step 11 follow-up, option A)

gpu_profile times a loop body that never refills a slot; production at n = 50 ran at
~90 µs per run-iteration where gpu_profile's fit said 61 µs. This probe measures the
REAL loop instead, plus what is inside the decision step:

  [1] a real generation (eval_generation_batch, refills, compaction, tail), with each
      function the loop calls timed in place (synchronized before and after); "rest"
      is the loop's own code. Gives µs per run-iteration as production sees it.
  [2] one brain tick taken apart, eager, at the production width: signed weights,
      net input, threshold, modulation sum, plasticity -- and the decision
      bookkeeping around them (decide_batch minus K ticks). Compiled code fuses these
      parts, so their shares are eager shares; [3] gives the compiled total.
  [3] the compiled decide step under torch.profiler: the kernels that take its time
      (CUDA only), and the memory traffic it implies (is it bandwidth-bound?).

Diagnostic only: prints numbers, asserts nothing.

Usage
-----
    python -m tests.large_profile                              # large config, ~5-10 min
    python -m tests.large_profile --runs 30 --config ea_from_random_large_tensor_cuda
"""

import argparse
import os
import sys
import time
from collections import Counter, defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mvb_torch.brain as B  # noqa: E402
import mvb_torch.generation as G  # noqa: E402
from mvb_torch.brain import BrainTensorState, StepGenome  # noqa: E402
from mvb_torch.decision import build_output_spec, decide_batch  # noqa: E402
from tests.devices import accelerators, sync  # noqa: E402
from mvb_torch.genome_codec import sparse_slot_bytes  # noqa: E402
from tests.gpu_profile import DT, P, layout, production_setup, width_rule  # noqa: E402

TIMES = defaultdict(float)
CALLS = Counter()


def timed(dev, name, fn):
    def wrapper(*a, **k):
        sync(dev)
        t0 = time.perf_counter()
        r = fn(*a, **k)
        sync(dev)
        TIMES[name] += time.perf_counter() - t0
        CALLS[name] += 1
        return r
    return wrapper


def real_generation(dev, cfg, batch, W, runs, compile):
    print(f"\n[1] A real generation: {P} genomes x {runs} runs, width {W}, compile "
          f"{'on' if compile else 'off'} (each called function timed in place)")
    c = dict(cfg)
    c["experiment"] = dict(cfg["experiment"], n_runs=runs)
    sim = G.sim_config_from_yaml(c)
    seeds = G.draw_seeds(*G.make_simulation_rngs(1), P, runs)
    spec = build_output_spec(cfg["brain"])
    kw = dict(width=min(W, P * runs), mode="live", philox_rounds=10,
              contraction=G.live_contraction(dev), compile=compile)
    G.eval_generation_batch(batch, spec, sim, seeds, **kw)          # warm-up / compile
    names = ("decide_batch", "reset_slots", "slot_view", "compact_slots", "act_batch",
             "sense_batch", "metabolise")
    orig = {nm: getattr(G, nm) for nm in names}
    orig_noise, orig_dec = G.KeyedSource.noise, G.KeyedSource.decision
    TIMES.clear()
    CALLS.clear()
    for nm in names:
        setattr(G, nm, timed(dev, nm, orig[nm]))
    G.KeyedSource.noise = timed(dev, "philox noise", orig_noise)
    G.KeyedSource.decision = timed(dev, "philox decision", orig_dec)
    try:
        sync(dev)
        t0 = time.perf_counter()
        res = G.eval_generation_batch(batch, spec, sim, seeds, **kw)
        sync(dev)
        total = time.perf_counter() - t0
    finally:
        for nm in names:
            setattr(G, nm, orig[nm])
        G.KeyedSource.noise, G.KeyedSource.decision = orig_noise, orig_dec
    run_iters = int(res.lifespans.sum().item())
    print(f"  total {total:.1f} s, {res.iterations} iterations, {run_iters:,} run-iterations "
          f"(mean lifespan {res.lifespans.float().mean().item():.1f}), utilisation "
          f"{res.utilisation:.0%}, {res.compactions} compactions")
    print(f"  => {1e6 * total / run_iters:.1f} µs per run-iteration "
          f"({1e3 * total / res.iterations:.1f} ms per iteration)")
    rest = total - sum(TIMES.values())
    for nm, t in sorted(TIMES.items(), key=lambda x: -x[1]) + [("rest of the loop", rest)]:
        print(f"    {nm:18s} {t:8.2f} s  {100 * t / total:5.1f}%  "
              f"{1e6 * t / run_iters:6.1f} µs/run-iter" + (f"  ({CALLS[nm]} calls)" if nm in CALLS else ""))
    return res


def tick_parts(dev, cfg, batch, width):
    print(f"\n[2] One decide step taken apart (eager, width {width}, timed per part)")
    spec = build_output_spec(cfg["brain"])
    K = int(batch.max_ticks.max().item())
    g = (torch.arange(width, device=dev) % P).view(width, 1)
    sb = G.slot_view(batch, g)
    sg = StepGenome.of(sb)
    n = batch.n_neurons
    gen = torch.Generator(device="cpu").manual_seed(0)
    act = (torch.rand(width, 1, n, generator=gen) < 0.5).to(DT).to(dev)
    st = BrainTensorState(act=act, Wabs=sb.Wabs0.clone())
    sens = (torch.rand(width, 1, len(batch.sensor_keys), generator=gen) < 0.3).to(DT).to(dev)
    noise = torch.randn(K, width, 1, n, generator=gen).to(DT).to(dev) * 0.05
    u = torch.rand(width, 1, generator=gen).to(DT).to(dev)
    parts = defaultdict(float)

    def clock(name, fn):
        sync(dev)
        t0 = time.perf_counter()
        r = fn()
        sync(dev)
        parts[name] += time.perf_counter() - t0
        return r
    for _ in range(2):                                   # warm caches / allocator
        decide_batch(sb, st, spec, sens, noise, u, contraction="sequential", n_brain_ticks=K)
    parts.clear()
    s = st
    for k in range(K):
        W = clock("signed weights W", lambda: (s.Wabs * sg.Wsign) * sg.Rel)
        net = clock("net input (n sequential sums)",
                    lambda: B._accumulate_input(sg, s, W, sens, noise[k], "sequential"))
        nxt = clock("threshold", lambda: (net >= sg.threshold_raw).to(s.act.dtype))
        mod = clock("modulation sum (sparse)", lambda: B._modulation_sum(sg, s, "sequential"))
        Wn = clock("plasticity update", lambda: (
            s.Wabs + ((sg.Eta * s.Wabs) * (1.0 - s.Wabs)) * mod).clamp(min=0.0))
        s = BrainTensorState(act=nxt, Wabs=Wn)
    ticks = sum(parts.values())
    sync(dev)
    t0 = time.perf_counter()
    decide_batch(sb, st, spec, sens, noise, u, contraction="sequential", n_brain_ticks=K)
    sync(dev)
    whole = time.perf_counter() - t0
    parts["decision bookkeeping (rest of decide)"] = max(0.0, whole - ticks)
    print(f"  K = {K} brain ticks; eager decide {1e3 * whole:.0f} ms "
          f"({1e6 * whole / width:.1f} µs per slot); parts summed {1e3 * ticks:.0f} ms")
    for nm, t in sorted(parts.items(), key=lambda x: -x[1]):
        print(f"    {nm:40s} {1e3 * t:8.1f} ms  {100 * t / max(whole, ticks):5.1f}%")
    return sb, st, sens, noise, u, K


def compiled_kernels(dev, cfg, sb, st, sens, noise, u, K, width):
    print(f"\n[3] The compiled decide step (width {width})")
    spec = build_output_spec(cfg["brain"])

    def step():
        return decide_batch(sb, st, spec, sens, noise, u, contraction="sequential",
                            n_brain_ticks=K, compile=True)
    for _ in range(2):
        step()
    sync(dev)
    t0 = time.perf_counter()
    for _ in range(3):
        step()
    sync(dev)
    ms = 1e3 * (time.perf_counter() - t0) / 3
    n = sb.n_neurons
    # Bytes one brain tick must move per slot at minimum (float32): read Wabs, Wsign,
    # Rel; write Wabs (4 x n^2), the n x n modulation result written and read (2 x n^2),
    # and the slot's sparse lists.
    lists = sparse_slot_bytes(sb)
    per_tick = 4 * (6 * n * n) + lists
    gbs = per_tick * K * width / (ms / 1e3) / 1e9
    print(f"  compiled decide {ms:.1f} ms ({1e6 * ms / 1e3 / width:.1f} µs per slot, "
          f"{1e6 * ms / 1e3 / width / K * 1e3:.0f} ns per slot-tick)")
    print(f"  sparse lists {lists / 1024:.0f} KB per slot")
    print(f"  minimum traffic ~{per_tick / 1024:.0f} KB per slot-tick -> effective "
          f"{gbs:.0f} GB/s (RTX 3070 peak ~448 GB/s; near it = bandwidth-bound)")
    if dev != "cuda":
        print("  (kernel table: CUDA only)")
        return
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        step()
        sync(dev)
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=12,
                                    max_name_column_width=60))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="cuda or mps (default: cuda if present)")
    ap.add_argument("--config", default="ea_from_random_large_tensor_cuda")
    ap.add_argument("--runs", type=int, default=30,
                    help="runs per genome for [1] (260 genomes x runs)")
    args = ap.parse_args()
    gpus = accelerators()
    dev = args.device or (gpus[0] if gpus else None)
    if dev is None:
        print("No GPU available (neither CUDA nor MPS).")
        return 1
    G.configure_device(dev)
    print("=" * 74)
    print(f"Large-brain profile on {dev}, config {args.config}")
    print("=" * 74)
    cfg, _, batch = production_setup(dev, args.config,
                                     dense_modulation=G.live_contraction(dev) == "einsum")
    W, why, _ = width_rule(dev, cfg, batch)
    print(f"  n = {batch.n_neurons}, modulation {layout(batch)}; width rule {W}")
    real_generation(dev, cfg, batch, W, args.runs, compile=True)
    # [2]/[3] use the sequential (sparse) contraction, production's on CUDA; on MPS
    # they still run it, which is useful only to check this script.
    width = min(W, P * args.runs)
    parts = tick_parts(dev, cfg, batch, width)
    compiled_kernels(dev, cfg, *parts, width)
    print("\nDone. Please send this whole output.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
