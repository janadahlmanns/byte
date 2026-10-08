#!/usr/bin/env python3
"""Tests for mvb_torch/philox.py, the keyed live-mode randomness (plan Step 8, R1).

1. Philox4x32-10 reproduces the Random123 known-answer vectors.
2. CPU and MPS produce identical integers and identical uniforms (integer -> uniform is
   exact), so the address -> number map does not depend on the device.
3. The numbers are well distributed: uniforms and normals against their exact
   distributions (Kolmogorov-Smirnov), moments, ranges, and no correlation between
   neighbouring addresses along every axis of the address (tick, brain tick, neuron,
   seed) -- a counter generator with too few rounds fails exactly there.
4. Every number is a function of its own address only: computing it alone, or inside a
   batch, or at a different K, gives the same bits.

Usage
-----
    python -m tests.test_philox
"""

import os
import sys

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.devices import accelerators, all_devices  # noqa: E402

from mvb_torch import philox  # noqa: E402

_RESULTS = []


def check(name, condition):
    _RESULTS.append((name, bool(condition)))
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}")
    return bool(condition)


def t64(*vals, device="cpu"):
    return [torch.tensor([v], dtype=torch.int64, device=device) for v in vals]


def test_kat():
    print("\n[1] Known-answer vectors (Random123 kat_vectors, philox4x32 10 rounds)")
    vectors = [
        ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
        ((0xFFFFFFFF,) * 4, (0xFFFFFFFF,) * 2, (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
        ((0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344), (0xA4093822, 0x299F31D0),
         (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1)),
    ]
    for ctr, key, want in vectors:
        got = tuple(int(w) for w in philox.philox4x32(*t64(*ctr), *t64(*key), rounds=10))
        check(f"ctr={ctr[0]:08x}.. key={key[0]:08x}.. -> {' '.join(f'{w:08x}' for w in got)}",
              got == want)
    check("allowed round counts are exactly 7..10 (philox_rounds config key, R1.1)",
          list(philox.ALLOWED_ROUNDS) == [7, 8, 9, 10])
    bad = []
    for r in (6, 11, 0, 7.0, "10", None, True):
        try:
            philox.check_rounds(r)
            bad.append(r)
        except ValueError:
            pass
    check(f"check_rounds refuses 6, 11, 0, 7.0, '10', None, True ({bad or 'all refused'})",
          not bad)
    seed = torch.arange(64, dtype=torch.int64).view(64, 1)
    tick = torch.full((64, 1), 5, dtype=torch.int64)
    z7 = philox.standard_normals(seed, tick, 4, 11, torch.float64, rounds=7)
    z10 = philox.standard_normals(seed, tick, 4, 11, torch.float64, rounds=10)
    check("7 and 10 rounds give different numbers for the same address (replay must match)",
          not torch.equal(z7, z10))


def _words(device, n=300_000, seed=0):
    g = np.random.default_rng(seed)
    ins = [torch.as_tensor(g.integers(0, 2**32, size=n, dtype=np.uint64).astype(np.int64),
                           device=device) for _ in range(6)]
    return philox.philox4x32(*ins, rounds=10)


def test_devices():
    print("\n[2] CPU and every GPU agree bit for bit on integers and uniforms")
    if not accelerators():
        check("no GPU available -- skipped", True)
        return
    cpu = _words("cpu")
    B = 2000
    seed = torch.arange(B, dtype=torch.int64).view(B, 1) * 7919 + 12345
    tick = torch.arange(B, dtype=torch.int64).view(B, 1) % 50
    zc = philox.standard_normals(seed, tick, 22, 11, torch.float32, rounds=10)
    for dev in accelerators():                      # MPS and/or CUDA (Step 8)
        gpu = _words(dev)
        check(f"[{dev}] 300,000 random (counter, key) inputs: identical 4 x 32-bit outputs "
              f"(int64 wraparound matches the CPU)",
              all(torch.equal(a, b.cpu()) for a, b in zip(cpu, gpu)))
        ok = torch.equal(philox.open_uniform(cpu[0], torch.float32),
                         philox.open_uniform(gpu[0], torch.float32).cpu())
        ok &= torch.equal(philox.half_open_uniform(cpu[1], torch.float32),
                          philox.half_open_uniform(gpu[1], torch.float32).cpu())
        check(f"[{dev}] uniforms identical to the CPU's (float32)", ok)
        zg = philox.standard_normals(seed.to(dev), tick.to(dev), 22, 11, torch.float32,
                                     rounds=10).cpu()
        diff = (zc - zg).abs().max().item()
        print(f"      [{dev}] normals vs CPU (float32 log/cos/sin): max |diff| = {diff:.2e} "
              f"(last-bit differences are expected; exact replay needs the same device type)")
        check(f"[{dev}] normals agree with the CPU's to float32 rounding (|diff| < 1e-5)",
              diff < 1e-5)


def test_distribution(rounds):
    print(f"\n[3] Distribution and independence (CPU, float64), {rounds} rounds")
    B, K, n = 4000, 22, 11
    seed = torch.as_tensor(np.random.default_rng(1).integers(0, 2**32, size=B, dtype=np.uint64)
                           .astype(np.int64)).view(B, 1)
    tick = torch.full((B, 1), 17, dtype=torch.int64)
    z = philox.standard_normals(seed, tick, K, n, torch.float64, rounds=rounds)[:, :, 0, :]   # (K, B, n)
    flat = z.flatten().numpy()
    ks = stats.kstest(flat, "norm")
    check(f"normals: n={flat.size}, mean {flat.mean():+.4f}, var {flat.var():.4f}, "
          f"KS p = {ks.pvalue:.3f}",
          abs(flat.mean()) < 0.005 and abs(flat.var() - 1) < 0.01 and ks.pvalue > 0.001)
    # Every one of the 11 neuron columns separately (each comes from a fixed word slot).
    pcols = [stats.kstest(z[..., i].flatten().numpy(), "norm").pvalue for i in range(n)]
    check(f"each neuron column is N(0,1) (min KS p over 11 columns {min(pcols):.3f})",
          min(pcols) > 0.001)

    def corr(a, b):
        return float(np.corrcoef(a.flatten(), b.flatten())[0, 1])
    lim = 4 / np.sqrt(B * n)
    c_k = corr(z[:-1].numpy(), z[1:].numpy())                          # brain tick k vs k+1
    c_i = corr(z[..., :-1].numpy(), z[..., 1:].numpy())                # neuron i vs i+1
    c_b = corr(z[:, :-1].numpy(), z[:, 1:].numpy())                    # run b vs b+1 (seeds)
    z2 = philox.standard_normals(seed, tick + 1, K, n, torch.float64, rounds=rounds)[:, :, 0, :]
    c_t = corr(z.numpy(), z2.numpy())                                  # world tick t vs t+1
    adj = philox.standard_normals(seed + 1, tick, K, n, torch.float64, rounds=rounds)[:, :, 0, :]
    c_s = corr(z.numpy(), adj.numpy())                                 # seed s vs s+1
    check(f"no correlation between neighbouring addresses: brain tick {c_k:+.4f}, neuron "
          f"{c_i:+.4f}, run {c_b:+.4f}, world tick {c_t:+.4f}, seed+1 {c_s:+.4f} "
          f"(|r| < {lim:.4f})",
          max(abs(c) for c in (c_k, c_i, c_b, c_t, c_s)) < lim)

    u = philox.decision_uniforms(seed, tick, torch.float64, rounds=rounds).flatten().numpy()
    us = np.concatenate([philox.decision_uniforms(seed, tick + t, torch.float64, rounds=rounds).flatten().numpy()
                         for t in range(50)])
    ks_u = stats.kstest(us, "uniform")
    check(f"decision uniforms: n={us.size}, in [0, 1): min {us.min():.2e}, max {us.max():.6f}, "
          f"KS p = {ks_u.pvalue:.3f}",
          us.min() >= 0.0 and us.max() < 1.0 and ks_u.pvalue > 0.001)
    w = _words("cpu", n=1_000_000)
    uo = philox.open_uniform(w[2], torch.float32)
    uh = philox.half_open_uniform(w[3], torch.float32)
    check("open uniforms in (0, 1] and half-open in [0, 1) at float32 over 1 M words",
          bool((uo > 0).all() and (uo <= 1).all() and (uh >= 0).all() and (uh < 1).all()))
    # The failure modes sit at the extreme words, which random sampling almost never
    # hits (~128 of 2**32 words round to 1.0 if all 32 bits are used in float32), so
    # test the edges directly.
    edges = torch.tensor([0, 1, 255, 256, 2**31, 2**32 - 257, 2**32 - 129, 2**32 - 128,
                          2**32 - 2, 2**32 - 1], dtype=torch.int64)
    ok = True
    for dt in (torch.float32, torch.float64):
        eo, eh = philox.open_uniform(edges, dt), philox.half_open_uniform(edges, dt)
        ok &= bool((eo > 0).all() and (eo <= 1).all() and (eh >= 0).all() and (eh < 1).all())
    check("edge words 0 and 2**32 - 1 (and their neighbours) stay inside (0, 1] / [0, 1) "
          "in float32 and float64", ok)
    nd = philox.standard_normals(seed, tick, K, n, torch.float64, rounds=rounds)
    dd = philox.decision_uniforms(seed, tick, torch.float64, rounds=rounds)
    nd_as_u = philox.half_open_uniform(
        philox.philox4x32(tick, torch.zeros_like(seed), torch.zeros_like(seed),
                          torch.full_like(seed, philox.STREAM_NOISE), seed,
                          torch.zeros_like(seed), rounds=rounds)[0], torch.float64)
    check("noise and decision streams differ for the same seed and tick",
          not torch.equal(nd_as_u, dd))


def test_address_only():
    print("\n[4] A number depends only on its address")
    B = 500
    seed = torch.arange(B, dtype=torch.int64).view(B, 1) * 104729 + 7
    tick = (torch.arange(B, dtype=torch.int64).view(B, 1) * 3) % 200
    full = philox.standard_normals(seed, tick, 22, 11, torch.float32, rounds=10)
    alone = torch.cat([philox.standard_normals(seed[b:b + 1], tick[b:b + 1], 22, 11,
                                               torch.float32, rounds=10)
                       for b in (0, 17, 499)], dim=1)
    check("a run's noise computed alone == inside a batch of 500",
          torch.equal(alone, full[:, [0, 17, 499]]))
    short = philox.standard_normals(seed, tick, 9, 11, torch.float32, rounds=10)
    check("brain ticks 0..8 identical whether K = 9 or K = 22", torch.equal(short, full[:9]))
    perm = torch.randperm(B, generator=torch.Generator().manual_seed(0))
    check("permuting the runs permutes the numbers, nothing else",
          torch.equal(philox.standard_normals(seed[perm], tick[perm], 22, 11, torch.float32, rounds=10),
                      full[:, perm]))


def main():
    print("=" * 70)
    print("mvb_torch/philox.py test suite")
    print("=" * 70)
    test_kat()
    test_devices()
    for rounds in (10, 7):
        test_distribution(rounds)
    test_address_only()
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
