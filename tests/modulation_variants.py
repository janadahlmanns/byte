#!/usr/bin/env python3
"""Which sparse-modulation layout is fastest on this GPU? (plan Step 13 follow-up)

Step 13's compact lists (prefix per position, int16) made the large run ~33% SLOWER on
the RTX 3070 than Step 11's padded lists, though they move less data. Suspected: the
per-position prefix updates (`acc[..., :E_l] += ...`) keep torch.compile from fusing
the positions into one kernel. This compares, at the production width, with the real
genomes of the large config:

  V0  Step 13 as implemented: prefix slices, int16
  V1  Step 11: every edge padded to L entries, int64
  V2  Step 11 layout with int16 indices
  V3  Step 13 lists, each position padded to E inside the computation (F.pad), so a
      compiler can keep the accumulator in registers and still read only real entries
  VD  the dense sum before Step 11 (n^3 per slot) -- after Steps 11/13 the 11-neuron
      lookup run became ~3x slower on the RTX 3070, so dense may win at small n

Each variant must equal the current `_modulation_sum` bit for bit (all are the same
additions in the same order). Timed alone (compiled) and inside the compiled decide
step (patched in), since fusion depends on the surrounding code.

Usage
-----
    python -m tests.modulation_variants                                   # large config, n = 50
    python -m tests.modulation_variants --config ea_from_lookup_tensor_cuda  # n = 11
"""

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mvb_torch.brain as B  # noqa: E402
import mvb_torch.generation as G  # noqa: E402
from mvb_torch.brain import BrainTensorState  # noqa: E402
from mvb_torch.decision import build_output_spec, decide_batch  # noqa: E402
from tests.devices import accelerators, sync  # noqa: E402
from tests.gpu_profile import DT, P, production_setup, width_rule  # noqa: E402

CURRENT = B._modulation_sum


def padded(batch, idx_dtype):
    """Step 11 arrays (P, L, E) rebuilt from the Step 13 lists."""
    E, L = batch.mod_edge.shape[1], len(batch.mod_k)
    k = torch.zeros((batch.mod_edge.shape[0], L, E), dtype=idx_dtype, device=batch.mod_edge.device)
    w = torch.zeros((batch.mod_edge.shape[0], L, E), dtype=batch.mod_w[0].dtype,
                    device=batch.mod_edge.device)
    for l_, (k_l, w_l) in enumerate(zip(batch.mod_k, batch.mod_w)):
        k[:, l_, : k_l.shape[1]] = k_l.to(idx_dtype)
        w[:, l_, : w_l.shape[1]] = w_l
    return k, w


def dense(batch, state, contraction):
    """The sequential sum before Step 11, verbatim (needs the dense Mod)."""
    P_, R, n = state.act.shape
    total = torch.zeros((P_, R, n, n), dtype=state.act.dtype, device=state.act.device)
    for k in range(n):
        total.addcmul_(state.act[..., k].unsqueeze(-1).unsqueeze(-1),
                       batch.Mod[:, k].unsqueeze(1))
    return B._tanh(total)


def finish(acc, edge, n):
    P_, R, E = acc.shape
    flat = torch.zeros((P_, R, n * n + E), dtype=acc.dtype, device=acc.device)
    flat.scatter_(2, edge.long().unsqueeze(1).expand(P_, R, E), acc)
    return B._tanh(flat[..., : n * n].reshape(P_, R, n, n))


def make_padded_variant(k, w):
    def v(batch, state, contraction):
        P_, R, n = state.act.shape
        E = batch.mod_edge.shape[-1]
        acc = torch.zeros((P_, R, E), dtype=state.act.dtype, device=state.act.device)
        for l_ in range(k.shape[1]):
            a_l = torch.gather(state.act, 2, batch._k_pad[:, l_].long().unsqueeze(1).expand(P_, R, E))
            acc.addcmul_(a_l, batch._w_pad[:, l_].unsqueeze(1))
        return finish(acc, batch.mod_edge, n)
    return v


def v3(batch, state, contraction):
    P_, R, n = state.act.shape
    E = batch.mod_edge.shape[-1]
    acc = torch.zeros((P_, R, E), dtype=state.act.dtype, device=state.act.device)
    for k_l, w_l in zip(batch.mod_k, batch.mod_w):
        E_l = k_l.shape[-1]
        a_l = torch.gather(state.act, 2, k_l.long().unsqueeze(1).expand(P_, R, E_l))
        acc = acc + F.pad(a_l * w_l.unsqueeze(1), (0, E - E_l))
    return finish(acc, batch.mod_edge, n)


class WithPad:
    """A slot batch carrying the padded arrays a padded variant reads."""

    def __init__(self, sb, k, w):
        self.__dict__.update(vars(sb))
        self._k_pad, self._w_pad = k, w


def bench(fn, reps=5):
    for _ in range(2):
        fn()
    sync_dev()
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    sync_dev()
    return 1e3 * (time.perf_counter() - t0) / reps, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ea_from_random_large_tensor_cuda")
    args = ap.parse_args()
    gpus = accelerators()
    global DEV
    DEV = gpus[0] if gpus else "cpu"
    globals()["sync_dev"] = lambda: sync(DEV)
    G.configure_device(DEV)
    cfg, _, batch = production_setup(DEV, args.config, dense_modulation=False)
    n = batch.n_neurons
    W, _, _ = width_rule(DEV, cfg, batch)
    # One width for every variant, small enough that the dense variant's per-slot n^3
    # copy fits too (n = 50: 3,000 slots x 0.5 MB).
    width = min(W, 3000 if n > 20 else 78000)
    g = (torch.arange(width, device=DEV) % P).view(width, 1)
    sb = G.slot_view(batch, g)
    widths = [k.shape[1] for k in sb.mod_k]
    print("=" * 74)
    print(f"Modulation variants on {DEV}: config {args.config}, n = {n}, width {width} "
          f"(width rule: {W}), edges per position {widths}")
    print("=" * 74)
    gen = torch.Generator().manual_seed(0)
    act = (torch.rand(width, 1, n, generator=gen) < 0.5).to(DT).to(DEV)
    st = BrainTensorState(act=act, Wabs=sb.Wabs0.clone())
    ref = CURRENT(sb, st, "sequential")
    spec = build_output_spec(cfg["brain"])
    K = int(sb.max_ticks.max().item())
    sens = (torch.rand(width, 1, len(sb.sensor_keys), generator=gen) < 0.3).to(DT).to(DEV)
    noise = torch.randn(K, width, 1, n, generator=gen).to(DT).to(DEV) * 0.05
    u = torch.rand(width, 1, generator=gen).to(DT).to(DEV)

    def build(name):
        """(function, batch it reads); each built only when measured, then freed."""
        if name.startswith("V0"):
            return CURRENT, sb
        if name.startswith("V1") or name.startswith("V2"):
            k, w = padded(sb, torch.int64 if name.startswith("V1") else torch.int16)
            return make_padded_variant(k, w), WithPad(sb, k, w)
        if name.startswith("V3"):
            return v3, sb
        _, _, dense_batch = production_setup(DEV, args.config, dense_modulation=True)
        return dense, G.slot_view(dense_batch, g)
    names = ["V0 Step 13 prefix slices, int16 (current)", "V1 Step 11 padded, int64",
             "V2 Step 11 padded, int16", "V3 Step 13 lists, F.pad per position",
             "VD dense, before Step 11"]
    print("\n  variant                                     sum alone   in decide   identical")
    first = None
    for name in names:
        torch._dynamo.reset()
        if DEV == "cuda":
            torch.cuda.empty_cache()
        line = f"  {name:42s}"
        try:
            fn, b = build(name)
            c = torch.compile(lambda s, fn=fn, b=b: fn(b, s, "sequential"))
            ms1, out = bench(lambda: c(st))
            same_sum = torch.equal(out, ref)
            del c, out
            torch._dynamo.reset()
            B._modulation_sum = (lambda bb, s_, cc, fn=fn, b=b: fn(b, s_, cc))
            try:
                ms2, o = bench(lambda: decide_batch(b, st, spec, sens, noise, u,
                                                    contraction="sequential",
                                                    n_brain_ticks=K, compile=True), reps=3)
            finally:
                B._modulation_sum = CURRENT
            key = (o[0].act, o[0].Wabs, o[1].action)
            first = first or key
            same = same_sum and all(torch.equal(x, y) for x, y in zip(key, first))
            line += f" {ms1:8.2f} ms {ms2:9.1f} ms   {same}"
            del fn, b, o
        except Exception as e:  # noqa: BLE001 -- a diagnostic: report and continue
            line += f" FAILED {type(e).__name__}: {str(e)[:160]}"
        print(line, flush=True)
    print("\n'sum alone': the compiled modulation sum by itself; 'in decide': one compiled "
          "decide step\n(K brain ticks) with that sum inside; 'identical': the sum equals the "
          "current code and the\ndecisions equal V0's, bit for bit.")
    print("\nDone. Please send this whole output.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
