#!/usr/bin/env python3
"""How does the batch width -- and through it the GPU memory -- affect runtime?
(plan Steps 16a and 20c)

For each n (points of the neuron_scaling sweeps), one fixed population and seed set are
evaluated as a real generation (`eval_generation_batch`: queue, compaction, compile)
at several widths. Per width: wall time, us per run-tick, iterations, torch peak memory
and new compiled graphs. Out of memory is reported and the probe continues. The
lifespans must be identical at every width (width is scheduling only) -- the one thing
this script asserts.

Two ways to choose the widths:

  --factors 1,1.5,2   (Step 16a, the default) multiples of the width rule's own width W
                      on this card. At the end: the measured memory per slot next to
                      the estimate's terms, and decision rule 16b (fix the estimate if,
                      at n = 40 or 50, the widest width that fits is >= 10% faster per
                      run-tick than W).
  --vram-gib [auto|LIST]  (Step 20c) the width the rule would choose on a card with that
                      much memory -- `choose_width(..., total_memory=X GiB)`. Same
                      compute, bandwidth and code; only the memory-driven width changes,
                      so this isolates what VRAM does. Emulation works downwards from the
                      card it runs on: sizes above this card's memory are refused, never
                      extrapolated. `auto` = the usual card sizes up to this card's
                      memory, plus the card itself. Once all runs fit (width = P*R), a
                      bigger card cannot change anything: the first such size is run to
                      confirm, the rest are listed as "all runs fit" without rerunning.
                      Derived per n: the fastest width, the knee (smallest VRAM within 5%
                      of the best) and the VRAM at which all runs fit.

The compiled modules go to the production compile-cache folder (plan Step 18b): this
probe compiles only through `eval_generation_batch`, i.e. through the Step 18a warm-up,
and the default folder may still hold old full-width modules that run slower.

Usage
-----
    python -m tests.width_probe                                  # 16a: n = 30, 40, 50
    python -m tests.width_probe --vram-gib --csv vram.csv        # 20c: all 14 fine-sweep n
    python -m tests.width_probe --vram-gib --n 11,20,30,40,50 --csv vram.csv   # short
    python -m tests.width_probe --device mps --n 50 --runs 20 --vram-gib 2,3   # smoke
"""

import argparse
import csv
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
from mvb_torch.compile_cache import use_versioned_compile_cache  # noqa: E402
from mvb_torch.decision import build_output_spec  # noqa: E402
from mvb_torch.genome_codec import encode_genomes, sparse_slot_bytes  # noqa: E402
from tests.devices import sync  # noqa: E402

DT = torch.float32
SWEEPS = [RS.ROOT / "configs" / "sweeps" / "neuron_scaling.yaml",
          RS.ROOT / "configs" / "sweeps" / "neuron_scaling_fine.yaml"]
FINE_N = "11,14,17,20,23,26,29,32,35,38,41,44,47,50"
DECISION_MIN_GAIN = 0.10          # plan 16b, fixed before measuring
DECISION_N = (40, 50)
CARD_SIZES_GIB = (2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80)   # 20c `auto`
KNEE_TOLERANCE = 0.05             # 20c: the knee is the smallest VRAM within 5% of the best
CSV_FIELDS = ["mode", "device", "gpu", "n", "factor", "vram_gib", "width", "all_runs",
              "all_runs_fit", "status", "seconds", "us_per_run_tick", "iterations",
              "run_ticks", "peak_alloc_mib", "peak_reserved_mib", "all_runs_need_est_mib",
              "new_graphs", "lifespans_identical"]


# ============================================================
# Pure helpers (tested in tests/test_plot_scaling.py)
# ============================================================

def auto_sizes(device_gib):
    """The usual card sizes up to this card's memory, then the card itself (20c)."""
    out = [float(s) for s in CARD_SIZES_GIB if s <= device_gib + 1e-9]
    if not out or abs(out[-1] - device_gib) > 1e-6:
        out.append(round(float(device_gib), 2))
    return out


def parse_sizes(spec, device_gib):
    """`auto` or a comma list, ascending; sizes above this card's memory are refused."""
    sizes = auto_sizes(device_gib) if spec == "auto" else sorted(float(v) for v in spec.split(","))
    too_big = [s for s in sizes if s > device_gib + 1e-6]
    if too_big:
        raise SystemExit(f"--vram-gib {too_big} GiB is more than this card has "
                         f"({device_gib:.2f} GiB). Emulation only works downwards; run on a "
                         f"larger card for these sizes.")
    if len(set(sizes)) != len(sizes):
        raise SystemExit(f"--vram-gib has duplicate sizes: {sizes}")
    return sizes


def emulated_width(P, R, slot_bytes, dev, batch_bytes, gib):
    """The width the rule chooses on a card with `gib` GiB: (width, reason), or
    (None, reason) if one slot does not fit that budget."""
    try:
        return G.choose_width(P, R, slot_bytes, dev, batch_bytes=batch_bytes,
                              total_memory=int(gib * 2**30))
    except RuntimeError as e:
        return None, str(e)


def plan_vram_rows(sizes, widths, Q):
    """Which emulated sizes to run: every size until the first whose width holds all Q
    runs (run once, to confirm); later sizes give the same width and are not rerun."""
    plan, fit_seen = [], False
    for s, w in zip(sizes, widths):
        if w is None:
            plan.append((s, w, "budget too small"))
        elif fit_seen:
            plan.append((s, w, "all runs fit (not rerun)"))
        else:
            plan.append((s, w, "run"))
            fit_seen = w >= Q
    return plan


def derive_vram(rows, tolerance=KNEE_TOLERANCE):
    """Per n, from its measured rows: the fastest, the knee and where all runs fit."""
    ok = [r for r in rows if r["status"] == "ok"]
    if not ok:
        return None
    best = min(ok, key=lambda r: r["us_per_run_tick"])
    knee = min((r for r in ok if r["us_per_run_tick"] <= (1 + tolerance) * best["us_per_run_tick"]),
               key=lambda r: r["vram_gib"])
    fit = [r["vram_gib"] for r in rows if r["all_runs_fit"] and r["status"] in ("ok", "all runs fit (not rerun)")]
    return {"best_vram_gib": best["vram_gib"], "best_width": best["width"],
            "best_us": best["us_per_run_tick"], "knee_vram_gib": knee["vram_gib"],
            "knee_us": knee["us_per_run_tick"], "all_fit_vram_gib": min(fit) if fit else None}


# ============================================================
# Measurement
# ============================================================

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


class Setup:
    """One n: the population, the encoded batch, the width rule's inputs, the seeds."""

    def __init__(self, dev, n, cfg, runs, population):
        x = cfg["experiment"]
        self.P = population or int(x["population_size"])
        self.R = runs or int(x["n_runs"])
        x["population_size"], x["n_runs"] = self.P, self.R
        self.cfg, self.n, self.dev = cfg, n, dev
        self.contraction = G.live_contraction(dev)
        rng = np.random.default_rng(5)
        genomes = []
        for i in range(self.P):               # generation-1-like: random + some mutations
            g = generate_random_genome(cfg, rng_seed=i)
            for _ in range(i % 4):
                g = generate_genome_mutate_simple(g, 0.1, 0.2, rng)
            genomes.append(g)
        self.batch = encode_genomes(genomes, cfg["brain"], device=dev, dtype=DT,
                                    dense_modulation=(self.contraction == "einsum"))
        self.sim, self.spec = G.sim_config_from_yaml(cfg), build_output_spec(cfg["brain"])
        self.sparse = sparse_slot_bytes(self.batch) if self.contraction == "sequential" else None
        self.slot = G.estimate_slot_bytes(
            height=self.sim.height, width_cells=self.sim.width, n=n,
            K=int(n * float(cfg["brain"]["max_decision_delay"])), dtype=DT, mode="live",
            max_ticks=self.sim.max_ticks, sparse_slot_bytes=self.sparse)
        self.batch_bytes = G.genome_batch_bytes(self.batch)
        self.W, self.why = G.choose_width(self.P, self.R, self.slot, dev,
                                          batch_bytes=self.batch_bytes)
        self.rounds = int(x["evaluator"]["philox_rounds"])
        self.seeds = G.draw_seeds(*G.make_simulation_rngs(1), self.P, self.R)
        self.need_mib = (G.BASE_BYTES + self.batch_bytes + self.P * self.R * self.slot) / 2**20
        self.ref = None

    def warm_up(self):
        """Compile once on a few runs of the same genomes (the width is dynamic)."""
        x = self.cfg["experiment"]
        warm_sim = G.sim_config_from_yaml({**self.cfg, "experiment": {**x, "n_runs": 2}})
        t0 = time.perf_counter()
        G.eval_generation_batch(self.batch, self.spec, warm_sim,
                                G.draw_seeds(*G.make_simulation_rngs(2), self.P, 2),
                                width=min(self.W, 2 * self.P), mode="live",
                                philox_rounds=self.rounds, contraction=self.contraction,
                                compile=True)
        sync(self.dev)
        return time.perf_counter() - t0

    def measure(self, w):
        """One generation at width w -> row dict (status 'ok' or 'OOM')."""
        reset_memory(self.dev)
        g0 = int(du.counters["stats"]["unique_graphs"])
        try:
            sync(self.dev)
            t0 = time.perf_counter()
            res = G.eval_generation_batch(self.batch, self.spec, self.sim, self.seeds, width=w,
                                          mode="live", philox_rounds=self.rounds,
                                          contraction=self.contraction, compile=True)
            sync(self.dev)
            dt = time.perf_counter() - t0
        except Exception as e:  # noqa: BLE001 -- OOM is a result here, anything else is not
            if not is_oom(e):
                raise
            reset_memory(self.dev)
            return {"width": w, "status": "OOM"}
        alloc, reserved = memory(self.dev)
        ticks = int(res.lifespans.sum())
        if self.ref is None:
            self.ref = res.lifespans.clone()
        return {"width": w, "status": "ok", "seconds": dt, "us_per_run_tick": 1e6 * dt / ticks,
                "iterations": res.iterations, "run_ticks": ticks, "peak_alloc_mib": alloc,
                "peak_reserved_mib": reserved,
                "new_graphs": int(du.counters["stats"]["unique_graphs"]) - g0,
                "lifespans_identical": bool(torch.equal(res.lifespans, self.ref))}


def fmt_row(r, label):
    if r["status"] != "ok":
        return f"  {label}: width {r['width']}: {r['status']}"
    a, v = r["peak_alloc_mib"], r["peak_reserved_mib"]
    return (f"  {label}: width {r['width']:>6}: {r['seconds']:8.2f} s, {r['us_per_run_tick']:7.2f} "
            f"us/run-tick, {r['iterations']} iterations, peak alloc "
            f"{'-' if a is None else f'{a:.0f}'} MiB, reserved {'-' if v is None else f'{v:.0f}'} "
            f"MiB, new graphs {r['new_graphs']}, lifespans identical: {r['lifespans_identical']}")


def gpu_busy():
    snap = RS.gpu_snapshot()
    if isinstance(snap.get("gpus"), list):
        return ", ".join(f"GPU{g['index']} {g['util_pct']}% {g['mem_used_mib']} MiB"
                         for g in snap["gpus"])
    return ""


def probe_factors(s, factors):
    """Step 16a: widths = factors x W."""
    print(f"  warm-up (compile) {s.warm_up():.1f} s", flush=True)
    widths = sorted({min(s.P * s.R, max(2, int(round(f * s.W)))) for f in factors})
    rows = []
    for w in widths:
        busy = gpu_busy()
        r = s.measure(w)
        r.update(factor=w / s.W)
        rows.append(r)
        print(fmt_row(r, f"{w / s.W:4.2f} W") + (f"   [before: {busy}]" if busy else ""), flush=True)
    ok = [r for r in rows if r["status"] == "ok"]
    terms, est_kb = estimate_terms(s.cfg, s.n, s.slot, s.sparse)
    print(f"\n  estimate: {est_kb:.0f} KB per slot = PEAK_FACTOR {G.PEAK_FACTOR} x ("
          + ", ".join(f"{k} {v:.0f}" for k, v in terms.items()) + ") KB")
    fit = [r for r in ok if r["peak_alloc_mib"] is not None]
    slope = None
    if len(fit) >= 2:
        slope, icpt = np.polyfit([r["width"] for r in fit], [r["peak_alloc_mib"] for r in fit], 1)
        print(f"  measured: {slope * 1024:.0f} KB per slot (slope of peak allocated), "
              f"fixed part {icpt:.0f} MiB -> estimate / measured = {est_kb / (slope * 1024):.2f}")
    elif s.dev != "cuda":
        print(f"  measured per slot: unavailable on {s.dev} (no peak counter)")
    gain = None
    at_w = [r for r in ok if r["width"] == min(s.P * s.R, s.W)]
    wider = [r for r in ok if r["width"] > s.W]
    if at_w and wider:
        gain = 1 - wider[-1]["us_per_run_tick"] / at_w[0]["us_per_run_tick"]
        print(f"  widest width that fits ({wider[-1]['width']}) vs W ({s.W}): {100 * gain:+.1f}% "
              f"per run-tick (positive = faster)")
    elif not wider:
        print("  no width above W ran (all runs already fit at W, or OOM)")
    if at_w and len(ok) > 1:
        best = min(ok, key=lambda r: r["us_per_run_tick"])
        print(f"  fastest width {best['width']} ({best['width'] / s.W:.2f} W): "
              f"{100 * (1 - best['us_per_run_tick'] / at_w[0]['us_per_run_tick']):+.1f}% per "
              f"run-tick vs W")
    return rows, {"gain": gain, "slope": slope, "est_kb": est_kb}


def probe_vram(s, sizes):
    """Step 20c: the width the rule would choose on cards of these sizes."""
    widths = [emulated_width(s.P, s.R, s.slot, s.dev, s.batch_bytes, g)[0] for g in sizes]
    plan = plan_vram_rows(sizes, widths, s.P * s.R)
    print(f"  all runs need ~{s.need_mib:.0f} MiB by the width rule's estimate (an upper "
          f"bound, ~3.5x the measured need at n = 30-50)", flush=True)
    print(f"  warm-up (compile) {s.warm_up():.1f} s", flush=True)
    rows, by_width = [], {}
    for gib, w, what in plan:
        base = {"vram_gib": gib, "width": w, "all_runs_fit": w is not None and w >= s.P * s.R}
        if what != "run":
            rows.append(dict(base, status=what))
            print(f"  {gib:6.2f} GiB: width {w if w is not None else '-'}: {what}", flush=True)
            continue
        busy = gpu_busy()
        r = by_width.get(w) or s.measure(w)
        by_width[w] = r
        rows.append(dict(r, **base))
        print(fmt_row(r, f"{gib:6.2f} GiB") + (" [all runs fit]" if base["all_runs_fit"] else "")
              + (f"   [before: {busy}]" if busy else ""), flush=True)
    d = derive_vram(rows)
    if d:
        fit = "-" if d["all_fit_vram_gib"] is None else f"{d['all_fit_vram_gib']:.2f} GiB"
        print(f"  fastest: {d['best_vram_gib']:.2f} GiB (width {d['best_width']}, "
              f"{d['best_us']:.2f} us); knee (within {100 * KNEE_TOLERANCE:.0f}% of it): "
              f"{d['knee_vram_gib']:.2f} GiB; all runs fit from: {fit}")
    return rows, d


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k, "")) for k in CSV_FIELDS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", default=None, help="default: 30,40,50 (factors), the 14 fine-sweep "
                                              "points (vram)")
    ap.add_argument("--factors", default=None, help="Step 16a: multiples of W (default 1,1.5,2,2.5,3)")
    ap.add_argument("--vram-gib", nargs="?", const="auto", default=None,
                    help="Step 20c: 'auto' (default when given) or a list of GiB")
    ap.add_argument("--runs", type=int, default=None, help="default: the config's n_runs")
    ap.add_argument("--population", type=int, default=None, help="default: the config's")
    ap.add_argument("--device", default=None, help="default: the config's evaluator.device")
    ap.add_argument("--csv", default=None, help="write every row to this CSV file")
    a = ap.parse_args()
    if a.factors is not None and a.vram_gib is not None:
        raise SystemExit("give --factors or --vram-gib, not both")
    mode = "vram" if a.vram_gib is not None else "factors"
    ns = [int(v) for v in (a.n or (FINE_N if mode == "vram" else "30,40,50")).split(",")]
    points = {}
    for sw in SWEEPS:
        for _, cfg in RS.expand_sweep(sw):
            points.setdefault(cfg["brain"]["n_neurons"], cfg)
    missing = [n for n in ns if n not in points]
    if missing:
        raise SystemExit(f"n = {missing} is in no neuron_scaling sweep (points: {sorted(points)})")
    dev = a.device or points[ns[0]]["experiment"]["evaluator"]["device"]
    cache = use_versioned_compile_cache()       # before anything compiles (Step 18b)
    G.configure_device(dev)
    device_gib = G.device_memory(dev) / 2**30
    gpu = torch.cuda.get_device_name(0) if dev == "cuda" else dev
    sizes = parse_sizes(a.vram_gib, device_gib) if mode == "vram" else None
    print(f"width probe ({mode}) on {dev}: torch {torch.__version__}, {gpu}, "
          f"{device_gib:.2f} GiB; code {RS.code_version()}; compile cache {cache}")
    if sizes:
        print(f"emulated card sizes: {sizes} GiB")
    all_rows, summary = [], []
    for n in ns:
        s = Setup(dev, n, points[n], a.runs, a.population)
        print(f"\n{'=' * 78}\nn = {n}: {s.P} genomes x {s.R} runs on {dev}, contraction "
              f"{s.contraction}, compile on\n  width rule on this card: W = {s.W} ({s.why})\n"
              f"{'=' * 78}", flush=True)
        if mode == "vram":
            rows, d = probe_vram(s, sizes)
        else:
            rows, d = probe_factors(s, [float(v) for v in (a.factors or "1,1.5,2,2.5,3").split(",")])
        for r in rows:
            r.update(mode=mode, device=dev, gpu=gpu, n=n, all_runs=s.P * s.R,
                     all_runs_need_est_mib=round(s.need_mib))
        all_rows += rows
        summary.append((n, rows, d))
        if a.csv:
            write_csv(a.csv, all_rows)           # after every n: a stopped run keeps its rows

    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    for n, rows, d in summary:
        same = all(r.get("lifespans_identical", True) for r in rows if r["status"] == "ok")
        oom = [r["width"] for r in rows if r["status"] == "OOM"]
        line = f"  n = {n}: lifespans identical at every width: {same}, OOM at {oom or 'none'}"
        if mode == "vram" and d:
            line += (f"; knee {d['knee_vram_gib']:.2f} GiB, fastest {d['best_vram_gib']:.2f} GiB"
                     f" ({d['best_us']:.2f} us)")
        elif mode == "factors" and d:
            gain = "-" if d["gain"] is None else f"{100 * d['gain']:+.1f}%"
            line += f"; gain {gain}"
        print(line)
    if mode == "factors":
        decided = [d for n, _, d in summary if n in DECISION_N and d["gain"] is not None]
        if decided:
            hit = any(d["gain"] >= DECISION_MIN_GAIN for d in decided)
            verdict = ("a wider batch is >= 10% faster -> fixing the estimate is worth it"
                       if hit else "no gain >= 10% -> keep the estimate as it is")
            print(f"\n16b: {verdict}.")
    if a.csv:
        print(f"\nRows written to {a.csv}")
    bad = [n for n, rows, _ in summary
           if not all(r.get("lifespans_identical", True) for r in rows if r["status"] == "ok")]
    if bad:
        print(f"\nFAILED: lifespans differ between widths at n = {bad} -- width must never "
              f"change results.")
        return 1
    print("\nDone. Please send this whole output.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
