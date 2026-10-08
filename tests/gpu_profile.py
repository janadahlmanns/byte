#!/usr/bin/env python3
"""GPU diagnostics for the tensor evaluator (plan_evotorch.md Step 8, CUDA validation).

Not a pass/fail suite (that is `python -m tests.run_all`); this prints the numbers the
next decisions need, on whichever GPU is present (CUDA preferred, else MPS):

  [1] environment: GPU, versions, memory, the pinned numeric settings
  [2] per-phase time of one generation iteration at three widths, with the production
      brain and world (ea_from_lookup), and peak memory per slot at each width
      -- eager and with compile on (R5), with the production contraction
  [3] torch.compile probe for R5: Philox and the decision step exactly as the evaluator
      compiles them, eager vs compiled -- speed, bit-identity, compile time, and
      whether a width change triggers a recompile
  [4] deterministic-algorithms probe: does any operation we use refuse
      torch.use_deterministic_algorithms(True)?

Usage
-----
    python -m tests.gpu_profile              # ~5-10 minutes
    python -m tests.gpu_profile --device mps
"""

import argparse
import os
import platform
import sys
import time

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import mvb_torch.generation as G  # noqa: E402
from mvb.genome.generate_genome_lookup_hard import generate_lookup_hard_genome  # noqa: E402
from mvb.genome.generate_genome_mutate_simple import generate_genome_mutate_simple  # noqa: E402
from mvb_torch import philox  # noqa: E402
from mvb_torch.brain import BrainTensorState  # noqa: E402
from mvb_torch.decision import build_output_spec, decide_batch  # noqa: E402
from mvb_torch.genome_codec import encode_genomes  # noqa: E402
from mvb_torch.world import active_key_mask, sense_batch  # noqa: E402
from mvb_torch.worm import act_batch, metabolise  # noqa: E402
from tests.devices import (accelerators, allocator_bytes, peak_allocator_bytes,  # noqa: E402
                           reset_peak, sync)

DT = torch.float32
P = 260


def production_setup(dev):
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs/experiments/ea_from_lookup.yaml")))
    rng = np.random.default_rng(5)
    genomes = []
    for i in range(P):
        g = generate_lookup_hard_genome(cfg, rng_seed=i)
        for _ in range(i % 4):                      # generation-1-like: elites + mutants
            g = generate_genome_mutate_simple(g, 0.1, 0.2, rng)
        genomes.append(g)
    return cfg, genomes, encode_genomes(genomes, cfg["brain"], device=dev, dtype=DT)


def env(dev):
    print("\n[1] Environment")
    print(f"  python {platform.python_version()}, torch {torch.__version__}, "
          f"platform {platform.platform()}")
    G.configure_device(dev)
    if dev == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"  GPU {p.name}, {p.total_memory / 2**30:.1f} GB, compute capability "
              f"{p.major}.{p.minor}, CUDA {torch.version.cuda}, cuDNN "
              f"{torch.backends.cudnn.version()}")
        print(f"  pinned: TF32 matmul {torch.backends.cuda.matmul.allow_tf32} (must be "
              f"False), float32 matmul precision '{torch.get_float32_matmul_precision()}', "
              f"CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}")
    else:
        print(f"  MPS, {torch.mps.recommended_max_memory() / 2**30:.1f} GB usable")
    print(f"  device memory used for the width rule: {G.device_memory(dev) / 2**30:.1f} GB")


def phase_profile(dev, cfg, batch, width, compile, n_iter=20, warm=5):
    """The real loop body, synchronized between phases. Returns (ms per phase, KB/slot).
    The warm-up iterations absorb compilation when `compile` is on."""
    c = dict(cfg)
    c["experiment"] = dict(cfg["experiment"], n_runs=300)
    sim = G.sim_config_from_yaml(c)
    seeds = G.draw_seeds(*G.make_simulation_rngs(1), P, 300)
    spec = build_output_spec(cfg["brain"])
    sync(dev)
    base = allocator_bytes(dev)
    reset_peak(dev)
    state, ctx, source = G.setup_generation(batch, sim, seeds, width=width, mode="live",
                                            philox_rounds=10, compile=compile)
    R, Q, i64 = 300, P * 300, torch.int64
    K = int(batch.max_ticks.max().item())
    key_mask = torch.tensor(active_key_mask(batch.sensor_keys, sim.active_sensors),
                            dtype=DT, device=dev)
    sb = G.slot_view(batch, state.slot_genome)
    dummy = torch.full((width, 1), Q, dtype=i64, device=dev)
    out = torch.full((Q + 1,), -1, dtype=i64, device=dev)
    T = {k: 0.0 for k in ("world", "act", "sense", "random numbers", "decide (brain)",
                          "commit+done+scatter", "slot_view", "reset_slots")}
    peak = [0]
    for it in range(warm + n_iter):
        rec = it >= warm

        def lap(name, t0):
            sync(dev)
            t1 = time.perf_counter()
            if rec:
                T[name] += t1 - t0
            peak[0] = max(peak[0], allocator_bytes(dev))
            return t1
        t = time.perf_counter()
        active = state.run_idx < R
        w, world = state.worm, state.world
        world_tick = w.ticks + 1
        has_next = (state.phase_idx + 1) < ctx.n_phases
        nxt = (state.phase_idx + 1).clamp(max=ctx.n_phases - 1)
        switching = active & has_next & (world_tick == ctx.phase_from[nxt])
        state.phase_idx = state.phase_idx + switching.to(i64)
        q = state.phase_idx
        reseed = switching & ctx.seeds_on_entry[q]
        ev = ctx.seed_event[q].clamp(min=0)
        new_avail = ctx.world_table[state.run_idx.clamp(0, R - 1), ev]
        world.avail.copy_(torch.where(reseed.unsqueeze(-1).unsqueeze(-1), new_avail, world.avail))
        world.clock = world.clock + (active & ~switching & ctx.regrow[q]).to(i64)
        t = lap("world", t)
        act_batch(world, w, active, sim.movement_cost, sim.energy_capacity, ctx.regrow[q],
                  ctx.regrow_time[q])
        t = lap("act", t)
        sens = sense_batch(world, w.y, w.x, batch.sensor_keys, DT) * key_mask
        t = lap("sense", t)
        nz = source.noise(w.ticks, state.slot_genome, state.run_idx)
        du = source.decision(w.ticks, state.slot_genome, state.run_idx)
        t = lap("random numbers", t)
        new_brain, dec = decide_batch(sb, state.brain, spec, sens, nz, du,
                                      contraction=G.live_contraction(dev), n_brain_ticks=K,
                                      compile=compile)
        t = lap("decide (brain)", t)
        state.brain = BrainTensorState(
            act=torch.where(active.unsqueeze(-1), new_brain.act, state.brain.act),
            Wabs=torch.where(active.unsqueeze(-1).unsqueeze(-1), new_brain.Wabs,
                             state.brain.Wabs))
        w.action = torch.where(active, dec.action, w.action)
        metabolise(w, active, sim.metabolic_rate)
        w.ticks = w.ticks + active.to(i64)
        done = active & (~w.alive | (w.ticks >= sim.max_ticks))
        idx = torch.where(done, state.slot_genome * R + state.run_idx.clamp(0, R - 1),
                          dummy).reshape(-1)
        out.scatter_(0, idx, w.ticks.reshape(-1))
        t = lap("commit+done+scatter", t)
        sb = G.slot_view(batch, state.slot_genome)
        t = lap("slot_view", t)
        G.reset_slots(state, done & False, ctx, source, sb)     # same work, nothing refilled
        t = lap("reset_slots", t)
    sync(dev)
    used = max(peak[0], peak_allocator_bytes(dev)) - base
    del state, ctx, source, sb
    G.release_device_cache(dev)
    return {k: 1000 * v / n_iter for k, v in T.items()}, used / width / 1024


def compile_probe(dev, batch):
    print("\n[3] torch.compile probe (R5): the evaluator's compiled functions vs eager")
    B = 7800

    def bench(fn, *a, reps=20):
        for _ in range(3):
            fn(*a)
        sync(dev)
        t0 = time.perf_counter()
        for _ in range(reps):
            r = fn(*a)
        sync(dev)
        return 1000 * (time.perf_counter() - t0) / reps, r

    def first_call(fn, *a):
        t0 = time.perf_counter()
        r = fn(*a)
        sync(dev)
        return time.perf_counter() - t0, r

    K = int(batch.max_ticks.max().item())
    seed = torch.randint(0, 2**32, (B, 1), dtype=torch.int64, device=dev)
    tick = torch.randint(0, 300, (B, 1), device=dev)
    try:
        cn = philox.compiled_normals(K, 11, DT, 10)
        cu = philox.compiled_uniforms(DT, 10)
        ct, _ = first_call(cn, seed, tick)
        te, ze = bench(lambda s, t: philox.standard_normals(s, t, K, 11, DT, rounds=10),
                       seed, tick)
        tc, zc = bench(cn, seed, tick)
        ue = philox.decision_uniforms(seed, tick, DT, rounds=10)
        uc = cu(seed, tick)
        d = (ze - zc).abs().max().item()
        print(f"  Philox normals, width {B}: eager {te:.2f} ms, compiled {tc:.2f} ms "
              f"({te / tc:.1f}x); normals bit-identical: {d == 0} (max |diff| {d:.1e}); "
              f"uniforms bit-identical: {torch.equal(ue, uc)}; first call {ct:.1f} s "
              f"(~0 = already compiled during [2])")
        s2, t2 = seed[:3000], tick[:3000]
        cn(s2, t2)                                   # 2nd shape: marks the width dynamic
        t3, _ = first_call(cn, seed[:5000], tick[:5000])
        print(f"  Philox at a third width (5000): first call {1000 * t3:.0f} ms "
              f"(a few ms = no recompile)")
    except Exception as e:  # noqa: BLE001 -- a diagnostic: report and continue
        print(f"  Philox compile FAILED: {type(e).__name__}: {str(e)[:300]}")

    spec = build_output_spec(yaml.safe_load(open(os.path.join(
        ROOT, "configs/experiments/ea_from_lookup.yaml")))["brain"])
    contraction = G.live_contraction(dev)

    def inputs(width):
        g = (torch.arange(width, device=dev) % P).view(width, 1)
        sb = G.slot_view(batch, g)
        st = BrainTensorState(act=(torch.rand(width, 1, 11, device=dev) < 0.5).to(DT),
                              Wabs=sb.Wabs0.clone())
        sens = (torch.rand(width, 1, len(batch.sensor_keys), device=dev) < 0.3).to(DT)
        nz = torch.randn(K, width, 1, 11, device=dev) * 0.05
        u = torch.rand(width, 1, device=dev)
        return sb, st, sens, nz, u

    def decide(compile, sb, st, sens, nz, u):
        s2, d = decide_batch(sb, st, spec, sens, nz, u, contraction=contraction,
                             n_brain_ticks=K, compile=compile)
        return s2.act, s2.Wabs, d.action
    try:
        args = inputs(B)
        ct, _ = first_call(decide, True, *args)
        te, oe = bench(decide, False, *args, reps=5)
        tc, oc = bench(decide, True, *args, reps=5)
        same = all(torch.equal(x, y) for x, y in zip(oe, oc))
        d = (oe[1] - oc[1]).abs().max().item()
        n_act = int((oe[2] != oc[2]).sum())
        print(f"  decide ({contraction}, {K} brain ticks), width {B}: eager {te:.1f} ms, "
              f"compiled {tc:.1f} ms ({te / tc:.1f}x); bit-identical: {same} "
              f"(max |dW| {d:.1e}, {n_act} actions differ); first call {ct:.1f} s "
              f"(~0 = already compiled during [2])")
        decide(True, *inputs(3000))                  # 2nd shape: marks the width dynamic
        t3, _ = first_call(decide, True, *inputs(5000))
        print(f"  decide at a third width (5000): first call {1000 * t3:.0f} ms "
              f"(well under the first call above = no recompile)")
    except Exception as e:  # noqa: BLE001
        print(f"  decide compile FAILED: {type(e).__name__}: {str(e)[:300]}")


def deterministic_probe(dev, cfg, batch):
    print("\n[4] Deterministic-algorithms probe")
    c = dict(cfg)
    c["experiment"] = dict(cfg["experiment"], n_runs=20, max_ticks=60)
    sim = G.sim_config_from_yaml(c)
    seeds = G.draw_seeds(*G.make_simulation_rngs(1), P, 20)
    torch.use_deterministic_algorithms(True)
    try:
        G.eval_generation_batch(batch, build_output_spec(cfg["brain"]), sim, seeds,
                                width=P * 20, mode="live", philox_rounds=10,
                                contraction=G.live_contraction(dev))
        print("  a whole generation runs under torch.use_deterministic_algorithms(True): "
              "no operation lacks a deterministic implementation")
    except Exception as e:  # noqa: BLE001
        print(f"  REFUSED: {type(e).__name__}: {str(e)[:400]}")
    finally:
        torch.use_deterministic_algorithms(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="cuda or mps (default: cuda if present)")
    args = ap.parse_args()
    gpus = accelerators()
    dev = args.device or (gpus[0] if gpus else None)
    if dev is None:
        print("No GPU available (neither CUDA nor MPS).")
        return 1
    print("=" * 74)
    print(f"GPU diagnostics on {dev}")
    print("=" * 74)
    env(dev)
    cfg, genomes, batch = production_setup(dev)
    print("\n[2] One iteration by phase (ms, synchronized between phases), production "
          "brain and world, 260 gen-1-like genomes")
    print(f"  contraction: {G.live_contraction(dev)}")
    for width in (7_800, 26_000, 78_000):
        for compile in (False, True):
            try:
                ms, kb = phase_profile(dev, cfg, batch, width, compile)
            except RuntimeError as e:
                print(f"  width {width:6d} compile={compile}: FAILED {str(e)[:300]}")
                continue
            total = sum(ms.values())
            print(f"  width {width:6d} {'compiled' if compile else 'eager   '}: "
                  f"total {total:7.1f} ms | "
                  + " | ".join(f"{k} {v:.1f}" for k, v in ms.items())
                  + f" | peak memory {kb:.1f} KB/slot")
    compile_probe(dev, batch)
    deterministic_probe(dev, cfg, batch)
    print("\nDone. Please send this whole output.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
