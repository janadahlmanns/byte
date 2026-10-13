#!/usr/bin/env python3
"""Why do compiled Philox normals differ from eager on the x86 CPU? (plan Step 9, G6 finding)

Observed in the Docker G6 run (x86 laptop CPU, torch 2.13.0): compiled vs eager normals
max |diff| 6.5, while decision uniforms were identical; Mac CPU 4.8e-7, CUDA/MPS 0.

Each stage of `standard_normals` is compiled and compared with eager on the same inputs,
on every device present:

  [a] the Philox words (integer only)     -- differ => integer arithmetic, e.g. the
                                             32x32-bit products overflow int64 (signed
                                             overflow is undefined behaviour in the C++
                                             that inductor generates for the CPU)
  [b] the uniforms from identical words
  [c] Box-Muller (log/sqrt/cos/sin) from identical uniforms
  [d] the whole function, as the evaluator compiles it: how many values differ, where,
      and the worst one with its inputs
  [e] whether the compiled normals still look standard normal (mean, std, tails)
  [f] hypothesis test: an overflow-free Philox (products split into 16-bit halves),
      eager == the original eager, and compiled == eager?

Usage
-----
    python -m tests.philox_compile_probe
"""

import os
import platform
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mvb_torch import philox as P  # noqa: E402
from tests.devices import all_devices  # noqa: E402

K, N, B = 22, 11, 3000
DT = torch.float32


def words_fn(seed, tick, rounds, philox_fn=None):
    """(K, B, 3, 4) Philox words, exactly as standard_normals computes them."""
    nb = P._BLOCKS_PER_BRAIN_TICK
    dev = seed.device
    k = torch.arange(K, dtype=torch.int64, device=dev).view(K, 1, 1)
    blk = torch.arange(nb, dtype=torch.int64, device=dev).view(1, 1, nb)
    shape = (K, seed.shape[0], nb)
    c0 = tick.view(1, -1, 1).expand(shape)
    c1 = (k * nb + blk).expand(shape)
    zero = torch.zeros(shape, dtype=torch.int64, device=dev)
    c3 = torch.full(shape, P.STREAM_NOISE, dtype=torch.int64, device=dev)
    k0 = seed.view(1, -1, 1).expand(shape)
    f = P.philox4x32 if philox_fn is None else philox_fn
    return torch.stack(f(c0, c1, zero, c3, k0, zero, rounds=rounds), dim=-1)


def uniforms_fn(words):
    return P.open_uniform(words.reshape(K, words.shape[1], 6, 2), DT)


def box_muller_fn(u):
    r = torch.sqrt(-2.0 * torch.log(u[..., 0]))
    theta = P._TWO_PI * u[..., 1]
    z = torch.stack((r * torch.cos(theta), r * torch.sin(theta)), dim=-1)
    return z.reshape(K, u.shape[1], 12)[..., :N].unsqueeze(2)


def _mulhilo(a, m):
    """hi, lo 32-bit halves of a * m (a < 2**32, m a 32-bit constant) without any
    intermediate exceeding 2**49: a = a_hi * 2**16 + a_lo."""
    a_hi, a_lo = a >> 16, a & 0xFFFF
    x, y = a_hi * m, a_lo * m                            # each < 2**48
    hi = ((x + (y >> 16)) >> 16) & P._M32
    lo = (((x & 0xFFFF) << 16) + y) & P._M32
    return hi, lo


def philox_no_overflow(c0, c1, c2, c3, k0, k1, *, rounds):
    for _ in range(rounds):
        hi0, lo0 = _mulhilo(c0, P._MULT_A)
        hi1, lo1 = _mulhilo(c2, P._MULT_B)
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
        k0 = (k0 + P._WEYL_A) & P._M32
        k1 = (k1 + P._WEYL_B) & P._M32
    return c0, c1, c2, c3


def words_no_overflow(seed, tick, rounds):
    return words_fn(seed, tick, rounds, philox_no_overflow)


def diff_report(name, e, c):
    e, c = e.cpu(), c.cpu()                 # float64 below; MPS has none
    if e.dtype == torch.int64:
        bad = int((e != c).sum())
        print(f"  {name}: {bad} of {e.numel()} differ")
        return bad
    d = (e.double() - c.double()).abs()
    big = int((d > 1e-5).sum())
    print(f"  {name}: max |diff| {d.max().item():.2e}; {big} of {e.numel()} differ by > 1e-5")
    return big


def probe(dev, rounds):
    print(f"\n=== {dev}, {rounds} rounds ===")
    g = torch.Generator().manual_seed(rounds)
    seed = torch.randint(0, 2**32, (B, 1), dtype=torch.int64, generator=g).to(dev)
    tick = torch.randint(0, 500, (B, 1), dtype=torch.int64, generator=g).to(dev)

    we = words_fn(seed, tick, rounds)
    wc = torch.compile(lambda s, t: words_fn(s, t, rounds))(seed, tick)
    bad_w = diff_report("[a] Philox words, compiled vs eager", we, wc)
    if bad_w:
        idx = (we != wc).nonzero()[0].tolist()
        print(f"      first differing word at (k, run, block, word) = {idx}: "
              f"eager {we[tuple(idx)].item():#010x}, compiled {wc[tuple(idx)].item():#010x}")
        neg = int((wc < 0).sum()) + int((wc > P._M32).sum())
        print(f"      compiled words outside [0, 2**32): {neg}")

    ue = uniforms_fn(we)
    diff_report("[b] uniforms from identical words", ue, torch.compile(uniforms_fn)(we))
    ze_bm = box_muller_fn(ue)
    diff_report("[c] Box-Muller from identical uniforms", ze_bm,
                torch.compile(box_muller_fn)(ue))

    ze = P.standard_normals(seed, tick, K, N, DT, rounds=rounds)
    zc = P.compiled_normals(K, N, DT, rounds)(seed, tick)
    bad_z = diff_report("[d] whole standard_normals (the evaluator's compiled function)", ze, zc)
    if bad_z:
        ze, zc = ze.cpu(), zc.cpu()
        d = (ze - zc).abs()
        w = (d == d.max()).nonzero()[0].tolist()
        kk, run, _, i = w
        print(f"      worst at brain tick {kk}, run {run}, neuron {i}: eager "
              f"{ze[tuple(w)].item():+.6f}, compiled {zc[tuple(w)].item():+.6f} "
              f"(seed {seed[run, 0].item()}, tick {tick[run, 0].item()})")
        pair = i // 2
        u_in = ue.cpu()[kk, run, pair]
        print(f"      its Box-Muller inputs u = ({u_in[0].item():.8f}, {u_in[1].item():.8f})")
        runs = sorted({r for r in (d > 1e-5).nonzero()[:, 1].tolist()})
        print(f"      differing values lie in {len(runs)} of {B} runs")

    for name, z in (("eager", ze), ("compiled", zc)):
        z = z.cpu().double()
        print(f"  [e] {name:8s} normals: mean {z.mean().item():+.4f}, std {z.std().item():.4f}, "
              f"|z|>4: {int((z.abs() > 4).sum())} (expect ~{0.0000633 * z.numel():.0f}), "
              f"min {z.min().item():+.2f}, max {z.max().item():+.2f}")

    wn = words_no_overflow(seed, tick, rounds)
    diff_report("[f] overflow-free Philox (eager) vs original eager", we, wn)
    wnc = torch.compile(lambda s, t: words_no_overflow(s, t, rounds))(seed, tick)
    diff_report("[f] overflow-free Philox compiled vs original eager", we, wnc)


def main():
    print("=" * 74)
    print(f"Philox compile probe: python {platform.python_version()}, torch {torch.__version__}, "
          f"{platform.platform()}, machine {platform.machine()}")
    print("=" * 74)
    for dev in all_devices():
        for rounds in (7, 10):
            try:
                probe(dev, rounds)
            except Exception as e:  # noqa: BLE001 -- a diagnostic: report and continue
                print(f"  FAILED on {dev}: {type(e).__name__}: {str(e)[:300]}")
    print("\nDone. Please send this whole output.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
