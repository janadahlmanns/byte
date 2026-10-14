#!/usr/bin/env python3
"""Brains larger than 12 neurons (plan Step 10: 50 neurons), on every device present.

  [1] n = 11 is unchanged by making the noise general: the Philox words at the
      documented 11-neuron addresses (t, 3k + block) have their frozen SHA-256
      (integers: identical on every platform), and standard_normals(n = 11) is exactly
      Box-Muller of those words on this platform
  [2] n = 50 noise: shape, distribution, independent of the batch, compiled == eager
  [3] scalar vs tensor EA at n = 50, pre-drawn: identical HDF5s, every dataset -- the
      brain, decision machine and plasticity are exact at the new size
  [4] live mode at n = 50: back to back, across widths, compaction on/off, compile
  [5] memory: the per-slot estimate (n = 11 unchanged, n = 50 with 13 Philox blocks)
      and the genome batch subtracted from the width budget

Usage
-----
    python -m tests.test_large_brain
"""

import hashlib
import math
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mvb_torch.generation as G  # noqa: E402
from mvb.genome.generate_genome_mutate_simple import generate_genome_mutate_simple  # noqa: E402
from mvb.genome.generate_genome_random import generate_random_genome  # noqa: E402
from mvb_torch import philox  # noqa: E402
from mvb_torch.decision import build_output_spec  # noqa: E402
from mvb_torch.genome_codec import encode_genomes  # noqa: E402
from tests.devices import all_devices  # noqa: E402
from tests.test_ea_tensor import base_cfg, diff_h5, run_ea, tensor_block  # noqa: E402
from tests.test_generation import load_cfg, seeds_for  # noqa: E402

_RESULTS = []
FIELDS = ("lifespans", "eats", "distance", "final_energy")
N = 50

# Frozen 2026-10-13: SHA-256 of the int64 Philox words at the 11-neuron addresses
# (counter (t, 3k + block, 0, STREAM_NOISE), K = 22, 3 blocks) for the seeds/ticks of
# _addresses(). Integers, so the same on every platform. (A first version hashed the
# float64 normals; their log/cos/sin differ in the last bit between arm64 and x86 --
# university PC, 2026-10-13 -- so that hash only held on the Mac.)
FROZEN_N11_WORDS = {
    7: "0f8934ef72e8ac849579453c565111af724670b42d97cb0ef6561c1c598577f4",
    10: "8f47c2f0afcbab682865eff72373bc266b4d630d572b2d396b1e92848a409055",
}
# estimate_slot_bytes before Step 10 (live float32; pre-drawn float64 with tracking).
FROZEN_SLOT_BYTES = {"live": 63048, "predrawn": 486728}


def check(name, condition):
    _RESULTS.append((name, bool(condition)))
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}")
    return bool(condition)


def _addresses(B=500):
    g = torch.Generator().manual_seed(2026)
    seed = torch.randint(0, 2**32, (B, 1), dtype=torch.int64, generator=g)
    tick = torch.randint(0, 2000, (B, 1), dtype=torch.int64, generator=g)
    return seed, tick


def brain50():
    bcfg = load_cfg()["brain"]
    bcfg["n_neurons"] = N
    return bcfg


def pool50(bcfg, n_random=6, n_mutant=6):
    cfg = {"brain": bcfg}
    pool = [generate_random_genome(cfg, rng_seed=300 + i) for i in range(n_random)]
    rng = np.random.default_rng(9)
    for i in range(n_mutant):
        g = pool[i % n_random]
        for _ in range(1 + i % 3):
            g = generate_genome_mutate_simple(g, 0.1, 0.2, rng)
        pool.append(g)
    return pool


# ============================================================

def _reference_words(seed, tick, rounds, K=22, nb=3):
    """The 11-neuron addresses as documented before Step 10, written out independently
    of standard_normals: counter (t, nb*k + block, 0, STREAM_NOISE), key (seed, 0)."""
    B = seed.shape[0]
    k = torch.arange(K, dtype=torch.int64).view(K, 1, 1)
    blk = torch.arange(nb, dtype=torch.int64).view(1, 1, nb)
    shape = (K, B, nb)
    zero = torch.zeros(shape, dtype=torch.int64)
    return torch.stack(philox.philox4x32(
        tick.view(1, B, 1).expand(shape), (k * nb + blk).expand(shape), zero,
        torch.full(shape, philox.STREAM_NOISE, dtype=torch.int64),
        seed.view(1, B, 1).expand(shape), zero, rounds=rounds), dim=-1)


def _box_muller(words, K, B, n, dtype):
    """Box-Muller exactly as standard_normals applies it (same operations, same order)."""
    nb = words.shape[2]
    u = philox.open_uniform(words.reshape(K, B, 2 * nb, 2), dtype)
    r = torch.sqrt(-2.0 * torch.log(u[..., 0]))
    theta = philox._TWO_PI * u[..., 1]
    z = torch.stack((r * torch.cos(theta), r * torch.sin(theta)), dim=-1)
    return z.reshape(K, B, 4 * nb)[..., :n].unsqueeze(2)


def test_n11_unchanged():
    print("\n[1] n = 11: the noise is what it was before n became general")
    seed, tick = _addresses()
    check("11 neurons still use 3 Philox blocks per brain tick",
          philox.blocks_per_brain_tick(11) == 3)
    for rounds, want in FROZEN_N11_WORDS.items():
        w = _reference_words(seed, tick, rounds).contiguous()
        got = hashlib.sha256(w.numpy().tobytes()).hexdigest()
        check(f"{rounds} rounds: SHA-256 of the 22 x 500 x 3 x 4 Philox words at the "
              f"11-neuron addresses unchanged ({got[:12]}..)", got == want)
        z = philox.standard_normals(seed, tick, 22, 11, torch.float64, rounds=rounds)
        check(f"{rounds} rounds: standard_normals(n = 11) == Box-Muller of exactly those "
              f"words (this platform)",
              torch.equal(z, _box_muller(w, 22, seed.shape[0], 11, torch.float64)))


def test_n50_noise(dev):
    print(f"\n[2] [{dev}] n = 50 noise")
    K, B = 100, 400
    seed, tick = (t.to(dev) for t in _addresses(B))
    z = philox.standard_normals(seed, tick, K, N, torch.float32, rounds=10)
    zc = z.cpu().double()
    check(f"[{dev}] shape {tuple(z.shape)} == ({K}, {B}, 1, {N}); {philox.blocks_per_brain_tick(N)} "
          f"blocks per brain tick", tuple(z.shape) == (K, B, 1, N)
          and philox.blocks_per_brain_tick(N) == 13)
    check(f"[{dev}] finite, mean {zc.mean().item():+.4f}, std {zc.std().item():.4f} "
          f"(standard normal)", bool(torch.isfinite(zc).all())
          and abs(zc.mean().item()) < 0.01 and abs(zc.std().item() - 1) < 0.01)
    check(f"[{dev}] every neuron 0..{N - 1} gets its own numbers (no column repeats "
          f"another)", all(not torch.equal(zc[..., i], zc[..., j])
                           for i in range(N) for j in range(i + 1, N)
                           if j == i + 1 or j == i + 12))
    sub = torch.tensor([0, 7, 123, B - 1], device=dev)
    zs = philox.standard_normals(seed[sub], tick[sub], K, N, torch.float32, rounds=10)
    check(f"[{dev}] runs computed alone == the same runs inside a batch of {B}",
          torch.equal(zs, z[:, sub]))
    try:
        zcmp = philox.compiled_normals(K, N, torch.float32, 10)(seed, tick)
        d = (zcmp - z).abs().max().item()
        check(f"[{dev}] compiled vs eager within 1e-5 "
              f"({'identical' if d == 0 else f'max |diff| {d:.1e}'})", d <= 1e-5)
    except RuntimeError as e:
        check(f"[{dev}] compile works here ({str(e)[:150]})", False)


def test_scalar_vs_tensor(tmp):
    print(f"\n[3] Scalar vs tensor EA at n = {N}, pre-drawn: identical HDF5s")
    cfg = base_cfg()
    x = cfg["experiment"]
    cfg["brain"]["n_neurons"] = N
    x["population_size"], x["n_runs"] = 10, 2
    x["evolutionary_algorithm"]["num_generations"] = 3
    x["evolutionary_algorithm"]["elite_size"] = 4
    x["predrawn_randomness"]["max_brain_ticks"] = 2 * N
    ref, out_s, rc = run_ea(cfg, tmp, "scalar50")
    if not check(f"scalar EA at n = {N} completed", rc == 0 and ref is not None):
        print(out_s[-2000:])
        return
    import copy
    c = copy.deepcopy(cfg)
    c["experiment"]["evaluator"] = tensor_block()
    h5, out_t, rc = run_ea(c, tmp, "tensor50")
    if not check(f"tensor EA at n = {N} completed", rc == 0 and h5 is not None):
        print(out_t[-2000:])
        return
    problems, n = diff_h5(ref, h5)
    check(f"all {n} datasets identical to the scalar EA ({len(problems)} differences)",
          not problems)
    for msg in problems[:5]:
        print(f"      - {msg}")


def test_live(dev):
    print(f"\n[4] [{dev}] Live mode at n = {N}")
    G.configure_device(dev)
    cfg = load_cfg(n_runs=4)
    cfg["brain"]["n_neurons"] = N
    genomes = pool50(cfg["brain"])
    P = len(genomes)
    seeds = seeds_for(cfg, P)
    sim = G.sim_config_from_yaml(cfg)
    spec = build_output_spec(cfg["brain"])
    batch = encode_genomes(genomes, cfg["brain"], device=dev, dtype=torch.float32)
    K = int(batch.max_ticks.max().item())

    def run(width, compile=False, compact_below=G.COMPACT_BELOW):
        return G.eval_generation_batch(
            batch, spec, sim, seeds, width=width, mode="live", philox_rounds=10,
            contraction=G.live_contraction(dev), compile=compile,
            compact_below=compact_below)
    a = run(P * 4)
    same = lambda r: all(torch.equal(getattr(r, k), getattr(a, k)) for k in FIELDS)  # noqa: E731
    check(f"[{dev}] {a.lifespans.numel()} runs (K up to {K} brain ticks) identical back "
          f"to back", same(run(P * 4)))
    check(f"[{dev}] identical at widths {P}, {P * 2} and {P * 4}",
          same(run(P)) and same(run(P * 2)))
    off = run(P * 4, compact_below=0.0)
    check(f"[{dev}] identical with compaction off ({a.compactions} compactions when on)",
          same(off))
    try:
        c = run(P * 4, compile=True)
        check(f"[{dev}] compiled run identical back to back", all(
            torch.equal(getattr(c, k), getattr(run(P * 4, compile=True), k))
            for k in FIELDS))
        diff = int((c.lifespans != a.lifespans).sum())
        print(f"      [{dev}] compiled vs eager (reported): "
              f"{'identical' if diff == 0 else f'{diff} of {a.lifespans.numel()} lifespans differ'}")
    except RuntimeError as e:
        check(f"[{dev}] compile works here ({str(e)[:150]})", False)


def test_memory():
    print("\n[5] Memory: per-slot estimate and the genome batch in the width budget")
    live = G.estimate_slot_bytes(height=51, width_cells=51, n=11, K=22,
                                 dtype=torch.float32, mode="live", max_ticks=2000,
                                 sparse_slot_bytes=None)
    pre = G.estimate_slot_bytes(height=51, width_cells=51, n=11, K=22, dtype=torch.float64,
                                mode="predrawn", max_ticks=200, tracked_connections=30,
                                sparse_slot_bytes=None)
    check(f"n = 11 estimates unchanged ({live}, {pre} bytes)",
          live == FROZEN_SLOT_BYTES["live"] and pre == FROZEN_SLOT_BYTES["predrawn"])
    K = 2 * N
    est50 = G.estimate_slot_bytes(height=51, width_cells=51, n=N, K=K,
                                  dtype=torch.float32, mode="live", max_ticks=2000,
                                  sparse_slot_bytes=None)
    by_hand = math.ceil(G.PEAK_FACTOR * (
        51 * 51 * 2 + (N ** 3 + 3 * N * N + N + 1) * 4 + (N * N + N) * 4 + 16 * 8
        + K * N * 4 + K * 13 * 4 * 8 * 6 + 2 * 51 * 51))
    check(f"n = 50 estimate {est50 / 1024:.0f} KB/slot counts 13 Philox blocks",
          est50 == by_hand)
    genomes = pool50(brain50())
    dense = encode_genomes(genomes, brain50(), dtype=torch.float32)
    gb1 = G.genome_batch_bytes(dense)
    P = len(genomes)
    check(f"genome batch bytes = all its tensors ({gb1 / 2**20:.1f} MB for {P} genomes "
          f"with the dense Mod, {P * N ** 3 * 4 / 2**20:.1f} MB of it Mod)",
          gb1 == sum(t.numel() * t.element_size() for t in vars(dense).values()
                     if isinstance(t, torch.Tensor)) and gb1 > P * N ** 3 * 4)
    gb = gb1 * 1000 // P                             # a P = 1000 batch of such genomes
    total, slot = 8 * 2**30, est50
    budget = int(0.8 * total) - G.BASE_BYTES
    w0, _ = G.choose_width(1000, 300, slot, "cuda", batch_bytes=0, total_memory=total)
    w1, why = G.choose_width(1000, 300, slot, "cuda", batch_bytes=gb, total_memory=total)
    check(f"the width budget subtracts the genomes: {w0} slots without, {w1} with "
          f"('{why[:60]}..')", w0 == budget // slot and w1 == (budget - gb) // slot
          and w1 < w0 and "of genomes" in why)


def main():
    print("=" * 70)
    print(f"Brains larger than 12 neurons (plan Step 10, n = {N})")
    print("=" * 70)
    test_n11_unchanged()
    for dev in all_devices():
        test_n50_noise(dev)
    with tempfile.TemporaryDirectory() as tmp:
        test_scalar_vs_tensor(tmp)
    for dev in all_devices():
        test_live(dev)
    test_memory()
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
