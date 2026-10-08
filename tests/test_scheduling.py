#!/usr/bin/env python3
"""Tests for the Step 8 scheduling changes (plan_evotorch.md 8.A, items 0-2).

* Item 0: slot utilisation is reported correctly; back-to-back live runs with the same
  code, config, seed and device are identical (the user's reproducibility requirement,
  2026-10-07, rerun after every Step 8 item).
* Item 1: the run queue shared by all genomes claims every (genome, run) exactly once,
  in genome-major order, and slots really do move between genomes.
* Item 2 is pure control flow; its proof is that the other suites stay bit-exact.

The scalar oracle comparisons live in tests/test_generation.py (bit-exact for every S,
which is the schedule-independence claim). This file checks the schedule itself.

Usage
-----
    python -m tests.test_scheduling
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mvb_torch.decision import build_output_spec  # noqa: E402
from mvb_torch.generation import (eval_generation_batch, live_contraction,  # noqa: E402
                                  sim_config_from_yaml)
from mvb_torch.genome_codec import encode_genomes  # noqa: E402
from tests.devices import all_devices  # noqa: E402
from tests.test_generation import MAX_BRAIN_TICKS, genome_pool, load_cfg, seeds_for  # noqa: E402

_RESULTS = []
FIELDS = ("lifespans", "eats", "distance", "final_energy")


def check(name, condition):
    _RESULTS.append((name, bool(condition)))
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}")
    return bool(condition)


def run(cfg, genomes, seeds, *, S, mode="predrawn", device="cpu", dtype=torch.float64,
        hook=None, compact_below=None, tracker=None):
    batch = encode_genomes(genomes, cfg["brain"], device=device, dtype=dtype)
    return eval_generation_batch(
        batch, build_output_spec(cfg["brain"]), sim_config_from_yaml(cfg), seeds,
        width=min(len(genomes) * S, len(genomes) * cfg["experiment"]["n_runs"]),
        mode=mode, max_brain_ticks=MAX_BRAIN_TICKS,
        philox_rounds=10 if mode == "live" else None,
        contraction="sequential" if mode == "predrawn" else live_contraction(device),
        _state_hook=hook,
        tracker=tracker,
        **({} if compact_below is None else {"compact_below": compact_below}),
    )


# ============================================================
# Item 0
# ============================================================

def test_utilisation():
    print("\n[1] Utilisation is exact when every slot is busy every iteration")
    cfg = load_cfg(n_runs=6)
    cfg["experiment"]["max_ticks"] = 20
    cfg["worm"]["energy_capacity"] = 10_000      # nobody starves: every run lasts 20
    genomes = genome_pool(cfg["brain"], n_random=3, n_mutant=0)
    res = run(cfg, genomes, seeds_for(cfg, len(genomes)), S=3)
    check(f"every run lasts max_ticks (lifespans {sorted(set(res.lifespans.flatten().tolist()))})",
          bool((res.lifespans == 20).all()))
    check(f"utilisation == 1.0 with S dividing R (got {res.utilisation:.4f}, "
          f"{res.iterations} iterations, {res.n_slots} slots)", res.utilisation == 1.0)
    check("iterations == runs per slot x max_ticks (2 x 20)", res.iterations == 40)


def test_back_to_back():
    print("\n[2] Back-to-back live runs are identical (same code, config, seed, device)")
    cfg = load_cfg(n_runs=20)
    genomes = genome_pool(cfg["brain"])
    seeds = seeds_for(cfg, len(genomes))
    devices = all_devices()
    other = seeds_for(cfg, len(genomes), sim_seed=999)
    for dev in devices:
        live = dict(mode="live", device=dev, dtype=torch.float32)
        a = run(cfg, genomes, seeds, S=4, **live)
        b = run(cfg, genomes, seeds, S=4, **live)
        same = all(torch.equal(getattr(a, k), getattr(b, k)) for k in FIELDS)
        check(f"[{dev}] {a.lifespans.numel()} runs identical back to back", same)
        c = run(cfg, genomes, other, S=4, **live)
        check(f"[{dev}] different per-run seeds give different results (the check is not "
              f"vacuous)", not torch.equal(a.lifespans, c.lifespans))
        # R1: live numbers are keyed by each run's seeds, so the width cannot matter.
        widths = {S: run(cfg, genomes, seeds, S=S, **live) for S in (1, 3, 20)}
        same_w = all(torch.equal(getattr(widths[S], k), getattr(a, k))
                     for S in widths for k in FIELDS)
        check(f"[{dev}] live results identical for S = 1, 3, 4, 20 (keyed randomness, R1)",
              same_w)


# ============================================================
# Item 1
# ============================================================

def test_queue():
    print("\n[3] The shared run queue")
    cfg = load_cfg(n_runs=5)
    genomes = genome_pool(cfg["brain"], n_random=6, n_mutant=0)
    P, R = len(genomes), cfg["experiment"]["n_runs"]
    seeds = seeds_for(cfg, P)
    for S in (1, 2, 7):
        B = min(P * S, P * R)          # width (R3): P*S, clamped to P*R
        claims = []                    # (iteration, slot, genome, run), in claim order
        hosted = {b: set() for b in range(B)}

        def hook(it, state, refill):
            g = state.slot_genome.reshape(-1).tolist()
            r = state.run_idx.reshape(-1).tolist()
            for b in refill.reshape(-1).nonzero().flatten().tolist():
                claims.append((it, b, g[b], r[b]))

        # The initial fill (slot b = queue item b) happens before the first hook call.
        initial = [(-1, b, b // R, b % R) for b in range(min(B, P * R))]
        res = run(cfg, genomes, seeds, S=S, hook=hook)
        all_claims = initial + claims
        for _, b, g, _r in all_claims:
            hosted[b].add(g)
        items = [(g, r) for _, _, g, r in all_claims]
        check(f"S={S}: every (genome, run) claimed exactly once "
              f"({len(items)} claims for {P * R} runs)",
              len(items) == P * R and len(set(items)) == P * R)
        order = sorted(all_claims, key=lambda c: (c[0], c[1]))
        check(f"S={S}: claimed in genome-major queue order (g*R + r increasing)",
              [g * R + r for _, _, g, r in order] == list(range(P * R)))
        if B < P * R:
            n_multi = sum(len(v) > 1 for v in hosted.values())
            check(f"S={S}: slots move between genomes ({n_multi} of {B} slots hosted "
                  f"more than one genome)", n_multi > 0)
        check(f"S={S}: every run produced a result", bool((res.lifespans > 0).all()))


# ============================================================
# R2: compaction
# ============================================================

def test_compaction():
    print("\n[4] Compaction (R2): finished slots leave the batch, results do not change")
    cfg = load_cfg(n_runs=8)
    genomes = genome_pool(cfg["brain"], n_random=8, n_mutant=8)
    seeds = seeds_for(cfg, len(genomes))
    settings = (0.0, 0.75, 1.0)
    pre = {c: run(cfg, genomes, seeds, S=4, compact_below=c) for c in settings}
    check(f"pre-drawn: no compaction at 0.0, {pre[0.75].compactions} at 0.75, "
          f"{pre[1.0].compactions} at 1.0; width {pre[1.0].n_slots} -> "
          f"{pre[1.0].final_width}",
          pre[0.0].compactions == 0 and pre[0.75].compactions > 0
          and pre[1.0].compactions >= pre[0.75].compactions
          and pre[1.0].final_width < pre[1.0].n_slots)
    check("pre-drawn: results identical with compaction off, at 0.75 and at every check",
          all(torch.equal(getattr(pre[c], k), getattr(pre[0.0], k))
              for c in settings for k in FIELDS))
    check(f"utilisation is measured against slots computed: "
          f"{pre[0.0].utilisation:.0%} without compaction -> {pre[1.0].utilisation:.0%} "
          f"with, same iterations ({pre[0.0].iterations} / {pre[1.0].iterations})",
          pre[1.0].utilisation > pre[0.0].utilisation
          and pre[1.0].iterations == pre[0.0].iterations
          and pre[1.0].slot_iterations == pre[0.0].slot_iterations)
    devices = all_devices()
    for dev in devices:
        live = {c: run(cfg, genomes, seeds, S=4, mode="live", device=dev,
                       dtype=torch.float32, compact_below=c) for c in settings}
        check(f"[{dev}] live: results identical with compaction off, at 0.75 and at every "
              f"check ({live[1.0].compactions} compactions)",
              live[1.0].compactions > 0
              and all(torch.equal(getattr(live[c], k), getattr(live[0.0], k))
                      for c in settings for k in FIELDS))

    # Tracking across compactions: per-tick, heat-map and final-weight arrays identical.
    from mvb_torch import tracking as trk
    from mvb_torch.genome_codec import encode_genomes as enc
    flags = trk.TrackingFlags.effective(True, True, True)
    sim = sim_config_from_yaml(cfg)

    def tracked(c):
        b = enc(genomes, cfg["brain"])
        tr = trk.Tracker(genomes, flags, sim.n_runs, len(genomes) * 4, sim.active_sensors,
                         b.device,
                         b.dtype, flush_every=7)
        res = run(cfg, genomes, seeds, S=4, compact_below=c, tracker=tr)
        out = {"final_w": tr.final_w[: len(genomes) * sim.n_runs].clone()}
        for v, r, ints, w in tr.per_run_rows():
            out[f"pt_{v}_{r}"] = trk.build_per_tick(ints, w, genomes[v], tr.conns[v],
                                                    sim.start_pos, sim.energy_capacity)
            out[f"hm_{v}_{r}"] = trk.build_heat_map(ints, sim.start_pos, sim.height,
                                                    sim.width)
        return res, out
    r0, t0 = tracked(0.0)
    r1, t1 = tracked(1.0)
    same = set(t0) == set(t1) and all(
        (torch.equal(t0[k], t1[k]) if isinstance(t0[k], torch.Tensor)
         else np.array_equal(t0[k], t1[k])) for k in t0)
    check(f"tracking: {len(t0)} per-tick/heat-map/final-weight arrays identical with "
          f"{r1.compactions} compactions (flush every 7 iterations)",
          same and r1.compactions > 0)


# ============================================================
# R3: automatic width
# ============================================================

_CALIB = """
import sys, torch, yaml
sys.path.insert(0, {root!r})
from tests.devices import allocator_bytes, peak_allocator_bytes, reset_peak, sync
from tests.test_generation import genome_pool, load_cfg, seeds_for
from mvb_torch.genome_codec import encode_genomes
from mvb_torch.decision import build_output_spec
import mvb_torch.generation as G
dev, W = sys.argv[1], int(sys.argv[2])
cfg = load_cfg(n_runs=800); cfg["experiment"]["max_ticks"] = 20
gs = genome_pool(cfg["brain"], n_random=20, n_mutant=20)
b = encode_genomes(gs, cfg["brain"], device=dev, dtype=torch.float32)
sync(dev); base = allocator_bytes(dev); reset_peak(dev)
peak = [0]
def hook(it, state, refill):   # MPS has no peak counter: sample every iteration
    peak[0] = max(peak[0], allocator_bytes(dev))
G.eval_generation_batch(b, build_output_spec(cfg["brain"]), G.sim_config_from_yaml(cfg),
                        seeds_for(cfg, len(gs)), width=W, mode="live", philox_rounds=10,
                        contraction=G.live_contraction(dev), compact_below=0.0,
                        _state_hook=hook)
sync(dev)
print(max(peak[0], peak_allocator_bytes(dev)) - base)
"""


def test_width():
    print("\n[5] Automatic width (R3)")
    import subprocess
    import mvb_torch.generation as G
    from tests.devices import accelerators, allocator_bytes, live_bytes, sync
    GB = 2**30
    slot = 100 * 1024
    w_all, why_all = G.choose_width(260, 300, slot, "cpu", total_memory=64 * GB)
    w_cap, why_cap = G.choose_width(260, 300, slot, "cpu", total_memory=8 * GB)
    exp_cap = (int(0.5 * 8 * GB) - G.BASE_BYTES) // slot
    check(f"all P*R runs when they fit ({w_all}, '{why_all[:9]}'); the memory cap when "
          f"not ({w_cap} = (0.5 x 8 GB - base) / 100 KB, '{why_cap[:10]}')",
          w_all == 78000 and why_all.startswith("all runs")
          and w_cap == exp_cap and why_cap.startswith("memory cap"))
    check("the width is deterministic for a given machine (same inputs, same width)",
          G.choose_width(260, 300, slot, "cpu", total_memory=8 * GB)[0] == w_cap)
    w_cuda, _ = G.choose_width(260, 300, slot, "cuda", total_memory=8 * GB)
    check(f"dedicated GPU memory (CUDA) gets the larger share: {w_cuda} slots from 8 GB "
          f"vs {w_cap} on shared memory (0.8 vs 0.5)",
          w_cuda == (int(0.8 * 8 * GB) - G.BASE_BYTES) // slot and w_cuda > w_cap)
    from tests.devices import physical_ram
    expected_total = {"cpu": physical_ram()}
    if torch.backends.mps.is_available():
        expected_total["mps"] = int(torch.mps.recommended_max_memory())
    if torch.cuda.is_available():
        expected_total["cuda"] = int(torch.cuda.get_device_properties(0).total_memory)
    check(f"device memory is the device's TOTAL memory, never the free amount "
          f"({', '.join(f'{d} {v / GB:.1f} GB' for d, v in expected_total.items())})",
          all(G.device_memory(d) == v for d, v in expected_total.items()))
    kw = dict(n=11, K=22, dtype=torch.float32, max_ticks=200)
    live = G.estimate_slot_bytes(height=51, width_cells=51, mode="live", **kw)
    small = G.estimate_slot_bytes(height=21, width_cells=21, mode="live", **kw)
    pre = G.estimate_slot_bytes(height=51, width_cells=51, mode="predrawn", **kw)
    trk_ = G.estimate_slot_bytes(height=51, width_cells=51, mode="live",
                                 tracked_connections=121, **kw)
    check(f"the estimate grows with the grid ({small // 1024} -> {live // 1024} KB), "
          f"pre-drawn bundles (+{(pre - live) // 1024} KB) and tracking "
          f"(+{(trk_ - live) // 1024} KB)", small < live < trk_ and pre > live + 100_000)
    cfg = load_cfg(n_runs=4)
    genomes = genome_pool(cfg["brain"], n_random=3, n_mutant=0)
    try:
        batch = encode_genomes(genomes, cfg["brain"])
        eval_generation_batch(batch, build_output_spec(cfg["brain"]),
                              sim_config_from_yaml(cfg), seeds_for(cfg, 3), width=13,
                              mode="predrawn", max_brain_ticks=MAX_BRAIN_TICKS)
        check("a width above P*R is refused", False)
    except ValueError:
        check("a width above P*R is refused (no slot may start without a run)", True)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    est = G.estimate_slot_bytes(height=51, width_cells=51, mode="live", **kw)
    for dev in accelerators():                       # MPS and/or CUDA
        # The cache is handed back after every generation (R3 memory fix): without it,
        # compaction's ever-changing widths made the cache grow generation by generation
        # (4.87 -> 5.42 -> 6.54 GB on MPS; ~10 GB by generation 3 in the user's run).
        cfg2 = load_cfg(n_runs=200)
        gs2 = genome_pool(cfg2["brain"], n_random=20, n_mutant=20)
        b2 = encode_genomes(gs2, cfg2["brain"], device=dev, dtype=torch.float32)
        eval_generation_batch(b2, build_output_spec(cfg2["brain"]), sim_config_from_yaml(cfg2),
                              seeds_for(cfg2, len(gs2)), width=len(gs2) * 200, mode="live",
                              philox_rounds=10, contraction=live_contraction(dev))
        sync(dev)
        cached = (allocator_bytes(dev) - live_bytes(dev)) / 2**20
        check(f"[{dev}] after a generation the device cache is released ({cached:.0f} MB "
              f"cached, < 64 MB), so memory cannot climb from generation to generation",
              cached < 64)
        # Memory calibration: widths large enough that the per-slot memory dominates the
        # allocator's pools (at a few thousand slots the reading is noise, even negative).
        used = {}
        for W in (8000, 32000):
            p = subprocess.run([sys.executable, "-c", _CALIB.format(root=root), dev, str(W)],
                               text=True, capture_output=True, timeout=900)
            used[W] = int(p.stdout.strip().splitlines()[-1])
        per_slot = (used[32000] - used[8000]) / 24000
        check(f"[{dev}] memory calibration: measured {per_slot / 1024:.1f} KB per slot <= "
              f"estimate {est / 1024:.1f} KB, so the cap is safe (and the measurement is "
              f"real: > 20 KB)", 20 * 1024 < per_slot <= est)


def test_batch_independence():
    print("\n[6] A run's arithmetic does not depend on the batch around it (replay)")
    # Single-run replay (R1) needs a run computed alone to give the same bits as inside
    # a batch of 78,000. Elementwise operations are independent by construction; the
    # contractions are what can differ: cuBLAS picks its matrix-product algorithm by
    # batch size (measured on CUDA). So each device must use a contraction that is
    # independent there -- `live_contraction(device)`.
    from mvb_torch.brain import BrainTensorState, _accumulate_input, _modulation_sum
    from mvb_torch import generation as G
    cfg = load_cfg()
    gs = genome_pool(cfg["brain"], n_random=20, n_mutant=20)
    gen = torch.Generator().manual_seed(0)
    for dev in all_devices():
        G.configure_device(dev)
        b = encode_genomes(gs, cfg["brain"], device=dev, dtype=torch.float32)
        contraction = live_contraction(dev)
        ok = True
        for B in (7_800, 78_000):
            g = (torch.arange(B) % len(gs)).view(B, 1).to(dev)
            sb = G.slot_view(b, g)
            act = (torch.rand(B, 1, 11, generator=gen) < 0.5).float().to(dev)
            st = BrainTensorState(act=act, Wabs=sb.Wabs0 * (0.5 + torch.rand(
                B, 1, 11, 11, generator=gen).to(dev)))
            W = (st.Wabs * sb.Wsign) * sb.Rel
            full_net = _accumulate_input(sb, st, W, None, None, contraction)
            full_mod = _modulation_sum(sb, st, contraction)
            subsets = [torch.tensor([i], device=dev) for i in (0, 1, 17, B // 2, B - 1)]
            subsets += [torch.arange(0, B, 7, device=dev), torch.arange(0, 1000, device=dev)]
            for sub in subsets:
                sbs = G.slot_view(b, g[sub])
                sts = BrainTensorState(act=act[sub], Wabs=st.Wabs[sub])
                Ws = (sts.Wabs * sbs.Wsign) * sbs.Rel
                ok &= torch.equal(_accumulate_input(sbs, sts, Ws, None, None, contraction),
                                  full_net[sub])
                ok &= torch.equal(_modulation_sum(sbs, sts, contraction), full_mod[sub])
        check(f"[{dev}] live contraction '{contraction}': net input and modulation of rows "
              f"computed alone / in subsets / in batches of 7,800 and 78,000 are "
              f"bit-identical", ok)
    # cuBLAS's einsum is batch-size-dependent (measured on a GTX 1660 Ti), so CUDA must
    # get `sequential` -- checked here too, so a machine without CUDA still guards it.
    check("CUDA live mode uses the batch-independent 'sequential' contraction",
          live_contraction("cuda") == "sequential")


def main():
    print("=" * 70)
    print("Step 8 scheduling tests (items 0-2)")
    print("=" * 70)
    test_utilisation()
    test_back_to_back()
    test_queue()
    test_compaction()
    test_width()
    test_batch_independence()
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
