#!/usr/bin/env python3
"""Why are some checks bit-exact on macOS but not on Linux/CUDA? (plan Step 8)

  [1] tanh: torch.tanh (float64, CPU) and np.tanh vs Python's math.tanh, which the scalar
      simulator uses. A last-bit difference would explain |w| (plasticity) mismatches.
      Also a checksum of math.tanh itself: compare it between machines -- if it differs,
      the scalar simulator is not bit-reproducible across platforms.
  [2] Contractions on each GPU: how far the einsum differs between batch sizes, and
      whether the `sequential` contraction is batch-independent.
  [3] decide_batch on each GPU: sequential vs einsum, eager vs torch.compile -- speed,
      and bit-identity of compiled vs eager.

Usage
-----
    python -m tests.platform_probe
"""

import hashlib
import math
import os
import platform
import struct
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mvb_torch.generation as G  # noqa: E402
from mvb_torch.brain import BrainTensorState, _accumulate_input, _modulation_sum  # noqa: E402
from mvb_torch.decision import build_output_spec, decide_batch  # noqa: E402
from mvb_torch.genome_codec import encode_genomes  # noqa: E402
from tests.devices import accelerators, sync  # noqa: E402
from tests.test_generation import genome_pool, load_cfg  # noqa: E402


def tanh_probe():
    print("\n[1] tanh on the CPU (float64)")
    rng = np.random.default_rng(0)
    # The plasticity rule applies tanh to modulation sums: small multiples of weights.
    x = np.concatenate([rng.uniform(-3, 3, 400_000), rng.normal(0, 0.5, 400_000),
                        rng.uniform(-1e-3, 1e-3, 200_000)])
    ref = np.array([math.tanh(v) for v in x])
    t = torch.tanh(torch.as_tensor(x, dtype=torch.float64)).numpy()
    n = np.tanh(x)
    print(f"  torch.tanh != math.tanh: {int((t != ref).sum())} of {x.size} "
          f"(max |diff| {np.abs(t - ref).max():.1e})")
    print(f"  np.tanh    != math.tanh: {int((n != ref).sum())} of {x.size}")
    one = torch.as_tensor(x[:2000], dtype=torch.float64)
    scal = np.array([torch.tanh(v).item() for v in one])     # 0-d: non-vectorized path
    print(f"  torch.tanh one element at a time != math.tanh: "
          f"{int((scal != ref[:2000]).sum())} of 2000")
    h = hashlib.sha256(b"".join(struct.pack("<d", v) for v in ref)).hexdigest()[:16]
    print(f"  checksum of math.tanh over the same 1,000,000 inputs: {h}")
    print(f"    (macOS 27 / Apple M4 / Python 3.14: compare with this value from the "
          f"other machine; different = the scalar sim differs across platforms)")


def contraction_probe(dev):
    print(f"\n[2] Contractions on {dev}: batch-size dependence")
    gen = torch.Generator().manual_seed(0)
    B = 78_000
    act = (torch.rand(B, 1, 11, generator=gen) < 0.5).float()
    Wm = torch.rand(B, 1, 11, 11, generator=gen) * 2 - 1
    Mod = torch.rand(B, 11, 11, 11, generator=gen) * 2 - 1
    a, w, m = act.to(dev), Wm.to(dev), Mod.to(dev)
    full = torch.einsum("pri,prij->prj", a, w)
    fullm = torch.einsum("pkij,prk->prij", m, a)
    worst = worstm = 0.0
    n_bad = n_badm = 0
    for sub in ([0], [17], [B // 2], list(range(0, B, 7)), list(range(1000)),
                list(range(7_800))):
        s = torch.tensor(sub, device=dev)
        d = (torch.einsum("pri,prij->prj", a[s], w[s]) - full[s]).abs()
        dm = (torch.einsum("pkij,prk->prij", m[s], a[s]) - fullm[s]).abs()
        worst, worstm = max(worst, d.max().item()), max(worstm, dm.max().item())
        n_bad += int((d > 0).sum())
        n_badm += int((dm > 0).sum())
    print(f"  einsum net input:   {n_bad} differing values across subsets "
          f"(max |diff| {worst:.1e})")
    print(f"  einsum modulation:  {n_badm} differing values (max |diff| {worstm:.1e})")
    # `sequential`: the same products, summed elementwise in a fixed order.
    cfg = load_cfg()
    gs = genome_pool(cfg["brain"], n_random=20, n_mutant=20)
    b = encode_genomes(gs, cfg["brain"], device=dev, dtype=torch.float32)
    g = (torch.arange(B, device=dev) % len(gs)).view(B, 1)
    sb = G.slot_view(b, g)
    st = BrainTensorState(act=a, Wabs=sb.Wabs0.clone())
    W = (st.Wabs * sb.Wsign) * sb.Rel
    full_seq = _accumulate_input(sb, st, W, None, None, "sequential")
    full_mod = _modulation_sum(sb, st, "sequential")
    ok = True
    for sub in ([0], [17], [B // 2], list(range(0, B, 7)), list(range(7_800))):
        s = torch.tensor(sub, device=dev)
        sbs = G.slot_view(b, g[s])
        sts = BrainTensorState(act=a[s], Wabs=st.Wabs[s])
        Ws = (sts.Wabs * sbs.Wsign) * sbs.Rel
        ok &= torch.equal(_accumulate_input(sbs, sts, Ws, None, None, "sequential"),
                          full_seq[s])
        ok &= torch.equal(_modulation_sum(sbs, sts, "sequential"), full_mod[s])
    print(f"  sequential (net input and modulation) batch-independent: {ok}")


def decide_probe(dev):
    print(f"\n[3] decide_batch on {dev}: sequential vs einsum, eager vs compiled")
    cfg = load_cfg()
    gs = genome_pool(cfg["brain"], n_random=20, n_mutant=20)
    b = encode_genomes(gs, cfg["brain"], device=dev, dtype=torch.float32)
    spec = build_output_spec(cfg["brain"])
    K = int(b.max_ticks.max().item())
    for B in (7_800, 78_000):
        g = (torch.arange(B, device=dev) % len(gs)).view(B, 1)
        sb = G.slot_view(b, g)
        gen = torch.Generator(device=dev).manual_seed(1)
        st = BrainTensorState(act=torch.zeros(B, 1, 11, device=dev), Wabs=sb.Wabs0.clone())
        sens = (torch.rand(B, 1, len(b.sensor_keys), generator=gen, device=dev) < 0.3).float()
        nz = torch.randn(K, B, 1, 11, generator=gen, device=dev) * 0.05
        u = torch.rand(B, 1, generator=gen, device=dev)

        def run(contraction):
            s2, d = decide_batch(sb, st, spec, sens, nz, u, contraction=contraction,
                                 n_brain_ticks=K)
            return s2.act, s2.Wabs, d.action

        def bench(fn, reps=10):
            for _ in range(2):
                fn()
            sync(dev)
            t0 = time.perf_counter()
            for _ in range(reps):
                r = fn()
            sync(dev)
            return 1000 * (time.perf_counter() - t0) / reps, r
        line = []
        outs = {}
        for c in ("einsum", "sequential"):
            ms, outs[c] = bench(lambda c=c: run(c))
            line.append(f"{c} eager {ms:.1f} ms")
            try:
                comp = torch.compile(lambda c=c: run(c), dynamic=True)
                msc, oc = bench(comp)
                same = all(torch.equal(x, y) for x, y in zip(outs[c], oc))
                line.append(f"{c} compiled {msc:.1f} ms (identical to eager: {same})")
            except Exception as e:  # noqa: BLE001 -- diagnostic
                line.append(f"{c} compile FAILED ({type(e).__name__})")
        print(f"  width {B}: " + "; ".join(line))


def main():
    print("=" * 74)
    print(f"Platform probe: python {platform.python_version()}, torch {torch.__version__}, "
          f"{platform.platform()}")
    print("=" * 74)
    tanh_probe()
    for dev in accelerators():
        G.configure_device(dev)
        contraction_probe(dev)
        decide_probe(dev)
    print("\nDone. Please send this whole output.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
