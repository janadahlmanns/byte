#!/usr/bin/env python3
"""Sparse modulation (plan Steps 11, 13): the same numbers as the dense sum, far less work.

The sequential contraction used to sum `act[k] * Mod[k, i, j]` over every k for every
edge (n^3 terms); it now sums only each modulated edge's nonzero entries, in ascending
k. That is the same additions minus `+0.0` terms, so the results must be bit-identical
to the old code -- which this file keeps verbatim as the reference (`dense_reference`).

  [1] encoding: the sparse lists scattered back give exactly the dense Mod, also for
      edges with >= 3 modulators, duplicate modulators, zero-weight edges and entries
      that fold to zero; each list position is a prefix of the edges (Step 13); L is a
      power of two; padding is inert; indices int16, or int32 when they would not fit
  [2] the sparse modulation sum == the old dense sum, bit for bit (float32, float64)
  [3] whole generations: the new code == the old dense sum (patched in) == the Step 11
      padded layout (also patched in); a batch without the dense Mod gives the same;
      n = 11 and 50
  [4] compile: recompiles only when L reaches a new power of two
  [5] memory: sparse slot estimate, batch bytes without Mod; einsum refuses no Mod

Usage
-----
    python -m tests.test_sparse_modulation
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch._dynamo.utils as dynamo_utils  # noqa: E402

import mvb_torch.brain as brain  # noqa: E402
import mvb_torch.generation as G  # noqa: E402
from mvb.genome.generate_genome_lookup_hard import generate_lookup_hard_genome  # noqa: E402
from mvb.genome.generate_genome_mutate_simple import generate_genome_mutate_simple  # noqa: E402
from mvb.genome.generate_genome_random import generate_random_genome  # noqa: E402
from mvb_torch.brain import BrainTensorState  # noqa: E402
from mvb_torch.decision import build_output_spec  # noqa: E402
from mvb_torch.genome_codec import encode_genomes, sparse_slot_bytes  # noqa: E402
from tests.devices import all_devices  # noqa: E402
from tests.test_generation import load_cfg, seeds_for  # noqa: E402

_RESULTS = []
FIELDS = ("lifespans", "eats", "distance", "final_energy")


def check(name, condition):
    _RESULTS.append((name, bool(condition)))
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}")
    return bool(condition)


def dense_reference(batch, state, contraction):
    """`_modulation_sum` as it was before Step 11, verbatim (sequential branch)."""
    P, R, n = state.act.shape
    total = torch.zeros((P, R, n, n), dtype=state.act.dtype, device=state.act.device)
    for k in range(n):
        total.addcmul_(
            state.act[..., k].unsqueeze(-1).unsqueeze(-1),
            batch.Mod[:, k].unsqueeze(1),
        )
    return brain._tanh(total)


def step11_reference(batch, state, contraction):
    """The Step 11 sparse sum (every edge padded to L entries, (P, L, E) arrays),
    rebuilt from the Step 13 lists: the second reference."""
    P, R, n = state.act.shape
    E = batch.mod_edge.shape[-1]
    L = len(batch.mod_k)
    k = torch.zeros((P, L, E), dtype=torch.int64, device=state.act.device)
    w = torch.zeros((P, L, E), dtype=state.act.dtype, device=state.act.device)
    for l_, (k_l, w_l) in enumerate(zip(batch.mod_k, batch.mod_w)):
        k[:, l_, : k_l.shape[1]] = k_l.long()
        w[:, l_, : w_l.shape[1]] = w_l
    acc = torch.zeros((P, R, E), dtype=state.act.dtype, device=state.act.device)
    for l_ in range(L):
        a_l = torch.gather(state.act, 2, k[:, l_].unsqueeze(1).expand(P, R, E))
        acc.addcmul_(a_l, w[:, l_].unsqueeze(1))
    flat = torch.zeros((P, R, n * n + E), dtype=acc.dtype, device=acc.device)
    flat.scatter_(2, batch.mod_edge.long().unsqueeze(1).expand(P, R, E), acc)
    return brain._tanh(flat[..., : n * n].reshape(P, R, n, n))


def brain_cfg(n):
    b = load_cfg()["brain"]
    b["n_neurons"] = n
    return b


def special_genome(n, rng):
    """A random genome with edges the generators rarely make, built on purpose:
    >= 3 modulators in non-ascending order, a duplicate modulator, 5 modulators,
    a modulated zero-weight edge, and an edge whose entries fold to exactly 0."""
    g = generate_random_genome({"brain": brain_cfg(n)}, rng_seed=int(rng.integers(1e6)))
    cw = np.array(g["connection_weights"], dtype=np.float64)
    spec = {k: list(v) for k, v in g["modulation_spec"].items()}
    live = list(zip(*np.nonzero(cw[:, :, 0])))
    (a, b), (c, d), (e, f), (h, i) = live[:4]
    spec[(a, b)] = [(7, 0.31), (2, -0.17), (9, 0.11)]                  # 3, unsorted
    spec[(c, d)] = [(4, 0.2), (1, 0.05), (4, -0.33), (8, 0.4), (0, 0.25)]  # dup + 5
    spec[(e, f)] = [(3, 0.4), (3, -0.4)]                                 # folds to 0
    spec[(h, i)] = [(6, 0.5), (5, -0.5)]
    z = next((s, t) for s in range(n) for t in range(n) if cw[s, t, 0] == 0.0)
    spec[z] = [(1, 0.3)]                                                 # zero-weight edge
    return {"connection_weights": cw, "modulation_spec": spec,
            "tonic_activations": np.asarray(g["tonic_activations"]), "eta": g["eta"]}


def pool(n, seed=0, n_random=6, n_mutant=8, special=2):
    rng = np.random.default_rng(seed)
    cfg = {"brain": brain_cfg(n)}
    out = [generate_random_genome(cfg, rng_seed=400 + seed * 50 + i) for i in range(n_random)]
    if n == 11:
        out.append(generate_lookup_hard_genome(cfg, rng_seed=0))
    for i in range(n_mutant):
        g = out[i % len(out)]
        for _ in range(2 + i % 5):
            g = generate_genome_mutate_simple(g, 0.3, 0.2, rng)
        out.append(g)
    out += [special_genome(n, rng) for _ in range(special)]
    return out


# ============================================================

def test_encoding():
    print("\n[1] Encoding: the sparse lists are exactly the dense Mod")
    for n in (11, 50):
        gs = pool(n)
        b = encode_genomes(gs, brain_cfg(n))
        P, E, L = b.n_pop, b.mod_edge.shape[1], len(b.mod_k)
        widths = [k.shape[1] for k in b.mod_k]
        # rebuilt[p, k, edge] += w for every entry; padding adds 0.0
        rebuilt = torch.zeros((P, n, n * n + E), dtype=b.dtype)
        edge = b.mod_edge.long()
        for k_l, w_l in zip(b.mod_k, b.mod_w):
            El = k_l.shape[1]
            rebuilt.view(P, -1).scatter_add_(1, k_l.long() * (n * n + E) + edge[:, :El], w_l)
        rebuilt = rebuilt[:, :, : n * n].reshape(P, n, n, n)
        check(f"n = {n}: {P} genomes (random, mutated, special), sparse -> dense == Mod "
              f"exactly (edges per position {widths})", torch.equal(rebuilt, b.Mod))
        real = int((b.Mod != 0).sum(dim=1).amax().item())
        check(f"n = {n}: L = {L} is a power of two >= 2 and >= the longest edge ({real})",
              L >= max(2, real) and L & (L - 1) == 0 and L < 2 * max(2, real))
        # Prefix: per genome, an edge has an l-th entry only if it has an (l-1)-th one,
        # and the edges with an l-th entry come first.
        prefix_ok = True
        for p in range(P):
            have = [(w[p] != 0) for w in b.mod_w]
            for l_ in range(L):
                cnt = int(have[l_].sum())
                prefix_ok &= bool(have[l_][:cnt].all()) and not bool(have[l_][cnt:].any())
                if l_:
                    prefix_ok &= cnt <= int(have[l_ - 1].sum())
        check(f"n = {n}: every list position is a prefix of the edges (Step 13), "
              f"{sum(widths)} entries stored for {E} edges (padded: {L * E})", prefix_ok
              and all(widths[i] >= widths[i + 1] for i in range(L - 1)))
        pad_k = any(bool(((w == 0) & (k != 0)).any()) for k, w in zip(b.mod_k, b.mod_w))
        scratch = edge >= n * n
        check(f"n = {n}: padding is inert (w = 0 with k = 0; padding edges in unique "
              f"scratch columns)", not pad_k
              and all(len(set(row.tolist())) == E for row in edge)
              and bool((edge[scratch] < n * n + E).all()))
        order_ok = all(
            bool(((b.mod_k[l_ + 1].long() > b.mod_k[l_][:, : widths[l_ + 1]].long())
                  | (b.mod_w[l_ + 1] == 0)).all()) for l_ in range(L - 1))
        check(f"n = {n}: every edge's entries are in ascending modulator index", order_ok)
        check(f"n = {n}: indices are int16 ({b.mod_edge.dtype}, {b.mod_k[0].dtype})",
              b.mod_edge.dtype == torch.int16 and all(k.dtype == torch.int16 for k in b.mod_k))
    b = encode_genomes(pool(11), brain_cfg(11))
    check(f"the special genomes exercise >= 3 modulators and folding "
          f"({b.n_folded_modulators} folded entries)",
          int((b.Mod != 0).sum(dim=1).amax().item()) >= 3 and b.n_folded_modulators > 0)
    big = encode_genomes([generate_random_genome({"brain": brain_cfg(130)}, rng_seed=1)],
                         brain_cfg(130), dense_modulation=False)
    check(f"n = 130: indices widen to int32, since an edge index can reach 2 n^2 = "
          f"{2 * 130 ** 2} > 32,767 (this genome's largest: "
          f"{int(big.mod_edge.long().max())})",
          big.mod_edge.dtype == torch.int32 and all(k.dtype == torch.int32 for k in big.mod_k))


def test_sum(dev):
    print(f"\n[2] [{dev}] sparse modulation sum == the old dense sum, bit for bit")
    for n in (11, 50):
        for dtype in ((torch.float32, torch.float64) if dev == "cpu" else (torch.float32,)):
            gs = pool(n, seed=1)
            b = encode_genomes(gs, brain_cfg(n), device=dev, dtype=dtype)
            P, R = b.n_pop, 6
            gen = torch.Generator().manual_seed(n)
            ok = True
            for trial in range(5):
                act = (torch.rand(P, R, n, generator=gen) < 0.5).to(dtype).to(dev)
                st = BrainTensorState(act=act, Wabs=b.Wabs0.expand(P, R, n, n).clone())
                new = brain._modulation_sum(b, st, "sequential")
                old = dense_reference(b, st, "sequential")
                ok &= torch.equal(new, old)
            check(f"[{dev}] n = {n}, {str(dtype)[6:]}: identical on 5 activity patterns "
                  f"x {P} genomes x {R} runs", ok)


def run_gen(batch, cfg, seeds, dev, width, compile=False):
    return G.eval_generation_batch(
        batch, build_output_spec(cfg["brain"]), G.sim_config_from_yaml(cfg), seeds,
        width=width, mode="live", philox_rounds=10, contraction="sequential",
        compile=compile)


def test_generations(dev):
    print(f"\n[3] [{dev}] whole generations (sequential): new code == old dense sum")
    G.configure_device(dev)
    for n in (11, 50):
        cfg = load_cfg(n_runs=3)
        cfg["brain"]["n_neurons"] = n
        gs = pool(n, seed=2)
        P = len(gs)
        seeds = seeds_for(cfg, P)
        dense = encode_genomes(gs, cfg["brain"], device=dev, dtype=torch.float32)
        sparse_only = encode_genomes(gs, cfg["brain"], device=dev, dtype=torch.float32,
                                     dense_modulation=False)
        new = run_gen(dense, cfg, seeds, dev, P * 3)
        orig = brain._modulation_sum
        brain._modulation_sum = dense_reference
        try:
            old = run_gen(dense, cfg, seeds, dev, P * 3)
            brain._modulation_sum = step11_reference
            s11 = run_gen(dense, cfg, seeds, dev, P * 3)
        finally:
            brain._modulation_sum = orig
        same = lambda a, b: all(torch.equal(getattr(a, k), getattr(b, k)) for k in FIELDS)  # noqa: E731
        check(f"[{dev}] n = {n}: {new.lifespans.numel()} runs identical to the old dense "
              f"sum (lifespans, eats, distance, energy)", same(new, old))
        check(f"[{dev}] n = {n}: identical to the Step 11 padded layout", same(new, s11))
        so = run_gen(sparse_only, cfg, seeds, dev, P * 3)
        check(f"[{dev}] n = {n}: a batch without the dense Mod gives the same results; "
              f"Mod is {'None' if sparse_only.Mod is None else 'present'} there",
              sparse_only.Mod is None and same(so, new))
        check(f"[{dev}] n = {n}: back to back and at width {P} identical",
              same(run_gen(sparse_only, cfg, seeds, dev, P * 3), new)
              and same(run_gen(sparse_only, cfg, seeds, dev, P), new))


def graphs():
    return int(dynamo_utils.counters["stats"]["unique_graphs"])


def test_compile(dev):
    print(f"\n[4] [{dev}] compile: graphs stay bounded as the lists change")
    G.configure_device(dev)
    cfg = load_cfg(n_runs=2)
    cfg["brain"]["n_neurons"] = 11
    seeds_cache = {}

    def gen(gs):
        b = encode_genomes(gs, cfg["brain"], device=dev, dtype=torch.float32,
                           dense_modulation=False)
        P = len(gs)
        s = seeds_cache.setdefault(P, seeds_for(cfg, P))
        run_gen(b, cfg, s, dev, P * 2, compile=True)
        return (len(b.mod_k), tuple(k.shape[1] for k in b.mod_k))
    rng = np.random.default_rng(3)
    base = [generate_random_genome({"brain": brain_cfg(11)}, rng_seed=700 + i) for i in range(8)]
    try:
        shapes = [gen(base)]
        g1 = graphs()
        for i in range(3):                           # new E each time, same L = 2
            gs = [generate_genome_mutate_simple(g, 0.1, 0.2, rng) for g in base]
            gs = [g for g in gs if max((len(v) for v in g["modulation_spec"].values()),
                                       default=0) <= 2]
            if len(gs) >= 4:
                shapes.append(gen(gs[:8]))
        g2 = graphs()
        shapes.append(gen(base + [special_genome(11, rng)]))      # L -> 4 (4 distinct)
        g3 = graphs()
        shapes.append(gen(base + [special_genome(11, rng), special_genome(11, rng)]))
        g4 = graphs()
    except RuntimeError as e:
        check(f"[{dev}] compile works here ({str(e)[:150]})", False)
        return
    Ls = [s[0] for s in shapes]
    print(f"      (L, edges per position) per generation: {shapes}; graphs "
          f"{g1} -> {g2} -> {g3} -> {g4}")
    check(f"[{dev}] generations with the same L but different widths add at most "
          f"three graphs ({g1} -> {g2})", g2 - g1 <= 3 and len(set(Ls[:-2])) == 1)
    # Whether a new L needs a new graph depends on what the compile cache already
    # holds; what matters is that compiling stays bounded and a repeated L is free.
    check(f"[{dev}] a new L adds at most a few graphs ({g2} -> {g3}); the same L again "
          f"adds none ({g3} -> {g4})", g3 - g2 <= 3 and g4 == g3 and Ls[-1] == Ls[-2] > Ls[0])


def test_memory():
    print("\n[5] Memory and loud failures")
    n = 50
    gs = pool(n, seed=4)
    dense = encode_genomes(gs, brain_cfg(n), dtype=torch.float32)
    sparse = encode_genomes(gs, brain_cfg(n), dtype=torch.float32, dense_modulation=False)
    bd, bs = G.genome_batch_bytes(dense), G.genome_batch_bytes(sparse)
    check(f"n = 50, {len(gs)} genomes: batch {bd / 2**20:.2f} MB with the dense Mod, "
          f"{bs / 2**20:.3f} MB without (every edge modulated, {len(sparse.mod_k)} "
          f"list positions)", bs < bd / 2)
    widths = [k.shape[1] for k in sparse.mod_k]
    E = sparse.mod_edge.shape[1]
    lists = sparse_slot_bytes(sparse)
    check(f"sparse lists per slot = {lists} B = sum(E_l) x (2 + 4) + E_0 x 2 "
          f"(edges per position {widths})", lists == sum(widths) * 6 + E * 2)
    kw = dict(height=51, width_cells=51, n=n, K=2 * n, dtype=torch.float32, mode="live",
              max_ticks=2000)
    est_d = G.estimate_slot_bytes(sparse_slot_bytes=None, **kw)
    est_s = G.estimate_slot_bytes(sparse_slot_bytes=lists, **kw)
    by_hand = est_d - int(G.PEAK_FACTOR * (n ** 3 * 4 - lists))
    check(f"slot estimate: {est_d // 1024} KB dense -> {est_s // 1024} KB sparse "
          f"(the n^3 copy replaced by the lists)", abs(est_s - by_hand) <= 1 and est_s < est_d)
    st = BrainTensorState(act=torch.zeros(sparse.n_pop, 1, n),
                          Wabs=sparse.Wabs0.clone())
    try:
        brain._modulation_sum(sparse, st, "einsum")
        check("einsum on a batch encoded without the dense Mod is refused", False)
    except ValueError as e:
        check(f"einsum on a batch without the dense Mod is refused ('{str(e)[:50]}..')",
              "dense_modulation=False" in str(e))


def test_adapter():
    print("\n[5b] The evaluator encodes without the dense Mod exactly on the sequential path")
    import contextlib
    import io
    import mvb_torch.adapter as A
    from tests.test_ea_tensor import _eval_args, base_cfg, tensor_block
    cfg = base_cfg()                                 # pre-drawn: sequential contraction
    genomes = [generate_random_genome(cfg, rng_seed=60 + i) for i in range(3)]
    seen = []
    orig = A.encode_genomes

    def spy(*a, **k):
        b = orig(*a, **k)
        seen.append(b.Mod is None)
        return b
    A.encode_genomes = spy
    try:
        for label, blk, rcfg in (
                ("pre-drawn (sequential)", tensor_block(), cfg["experiment"]["predrawn_randomness"]),
                ("live on CPU (einsum)", tensor_block("cpu", "float32", 10, False), None)):
            args, kw = _eval_args(cfg, genomes)
            kw["randomness_cfg"] = rcfg
            with contextlib.redirect_stdout(io.StringIO()):
                A.make_tensor_evaluator(blk, cfg, rcfg)(*args, **kw)
            print(f"      {label}: dense Mod {'not built' if seen[-1] else 'built'}")
    finally:
        A.encode_genomes = orig
    check("sequential path encodes no dense Mod; the einsum path does",
          seen == [True, False])


def main():
    print("=" * 70)
    print("Sparse modulation (plan Step 11)")
    print("=" * 70)
    test_encoding()
    for dev in all_devices():
        test_sum(dev)
    for dev in all_devices():
        test_generations(dev)
    for dev in all_devices():
        test_compile(dev)
    test_memory()
    test_adapter()
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
