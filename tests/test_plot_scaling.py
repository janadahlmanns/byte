#!/usr/bin/env python3
"""Tests for the neuron-scaling figure and the VRAM emulation (plan Step 20).

  [1] aggregation on a synthetic report with known numbers: medians and IQR over
      generations 1+ (generation 0 ignored), the fitted exponent, local exponents
  [2] exclusions, each with its reason: older code (no Step 18 marker), a busy GPU, a
      failed config; --include-disturbed plots the busy one, marked
  [3] the neuron count comes from the run's data, not the config's name
  [4] the figure and CSVs are written
  [5] width_probe --vram-gib helpers: `auto` sizes for 8/24/80 GiB cards (and 7.63),
      sizes above the card refused, widths = choose_width(total_memory), "all runs fit"
      rows, the knee

Usage
-----
    python -m tests.test_plot_scaling
"""

import contextlib
import csv
import io
import json
import sys
import tempfile
from pathlib import Path

import h5py

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import analysis.plot_neuron_scaling as PS  # noqa: E402

_RESULTS = []
C = 0.02                      # synthetic: us per worm-tick = C * n^2 (exponent exactly 2)
SPREAD = [0.8, 0.9, 1.0, 1.0, 1.0, 1.1, 1.2, 0.95, 1.05]     # gens 1-9, median 1.0


def check(name, condition):
    _RESULTS.append((name, bool(condition)))
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}")
    return bool(condition)


def make_report(folder, ns, *, marker=True, busy=False, failed_n=None, misleading_names=False,
                host="testhost"):
    """A report folder as run_selected.py writes it, with known numbers."""
    folder.mkdir(parents=True)
    (folder / "results").mkdir()
    (folder / "configs").mkdir()
    (folder / "system.json").write_text(json.dumps(
        {"hostname": host, "gpus": [{"name": "NVIDIA GeForce TEST 9000"}]}))
    summary, gens, snaps = [], [], []
    for i, n in enumerate(ns, start=1):
        label = f"fam [n_neurons={999 if misleading_names else n}]"
        log = f"{i:02d}_cfg.log"
        line = ("[tensor-evaluator] backend=tensor compile=on compile_cache=/tmp/x/byte-w512-v1\n"
                if marker else "[tensor-evaluator] backend=tensor compile=on\n")
        (folder / log).write_text(line)
        copy = f"results/{i:02d}_cfg.h5"
        with h5py.File(folder / copy, "w") as f:
            f.attrs["brain_n_neurons"] = n
        summary.append({"order": i, "config": label,
                        "status": "failed (exit code 1)" if n == failed_n else "ok",
                        "log": log, "hdf5_copy": copy, "config_path": f"/elsewhere/{i:02d}_cfg.yaml",
                        "gpu_util_before_pct": "85" if busy else "0",
                        "gpu_mem_used_before_mib": "3643" if busy else "47",
                        "vram_peak_mib": "2300", "runs_queued": "yes"})
        base = C * n ** 2
        gens.append({"order": i, "config": label, "generation": 0, "seconds": 999.0,
                     "iterations": 1, "ms_per_iteration": 1, "run_ticks": 1,
                     "us_per_run_tick": base * 100})                     # the compile
        for g, f_ in enumerate(SPREAD, start=1):
            gens.append({"order": i, "config": label, "generation": g, "seconds": n * f_,
                         "iterations": 1, "ms_per_iteration": 1, "run_ticks": 1,
                         "us_per_run_tick": base * f_})
        procs = ["123, /someone/else/python, 3588 MiB"] if busy else []
        snaps.append({"order": i, "config": label, "before": {"processes": procs},
                      "after": {"processes": procs}})
    for name, rows in (("summary.csv", summary), ("generations.csv", gens)):
        with open(folder / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (folder / "gpu_snapshots.json").write_text(json.dumps(snaps))
    return folder


def test_aggregation(tmp):
    print("\n[1] Aggregation and fit on a synthetic report")
    rep = make_report(tmp / "good", [11, 20, 30, 40, 50])
    pts, excl = PS.load_report(rep)
    series = PS.series_of(pts)
    check("one series, 5 points, nothing excluded", len(series) == 1 and not excl
          and len(next(iter(series.values()))) == 5)
    v = next(iter(series.values()))
    check("medians over generations 1+ equal C * n^2 (generation 0, 100x, ignored)",
          all(abs(p["us_med"] - C * p["n"] ** 2) < 1e-9 for p in v))
    q1, med, q3 = PS.quartiles(SPREAD)
    check(f"IQR from generations 1+ ({q1:.3f}-{q3:.3f} x)",
          all(abs(p["us_q1"] - C * p["n"] ** 2 * q1) < 1e-9
              and abs(p["us_q3"] - C * p["n"] ** 2 * q3) < 1e-9 for p in v))
    k, a = PS.power_fit([p["n"] for p in v], [p["us_med"] for p in v])
    check(f"fitted exponent 2 ({k:.6f}) and prefactor C ({a:.6f})",
          abs(k - 2) < 1e-9 and abs(a - C) < 1e-9)
    loc = PS.local_exponents([p["n"] for p in v], [p["us_med"] for p in v])
    check("local exponents all 2", all(abs(e - 2) < 1e-9 for _, e in loc))
    check("seconds per generation: median over generations 1+ (n x 1.0)",
          all(abs(p["sec_med"] - p["n"]) < 1e-9 for p in v))


def test_exclusions(tmp):
    print("\n[2] Exclusions, each with its reason")
    old = make_report(tmp / "old", [11, 20], marker=False)
    busy = make_report(tmp / "busy", [11, 20], busy=True)
    fail = make_report(tmp / "fail", [11, 20, 30], failed_n=20)
    _, e_old = PS.load_report(old)
    check("older code (no Step 18 marker) excluded",
          len(e_old) == 2 and all("Step 18" in why for _, why in e_old))
    p_busy, e_busy = PS.load_report(busy)
    check("busy GPU excluded, naming the other process and the load",
          not p_busy and len(e_busy) == 2
          and all("someone/else" in why and "85%" in why for _, why in e_busy))
    p_inc, _ = PS.load_report(busy, include_disturbed=True)
    check("--include-disturbed keeps them, marked disturbed",
          len(p_inc) == 2 and all(p["disturbed"] for p in p_inc))
    p_fail, e_fail = PS.load_report(fail)
    check("a failed config is excluded, the others kept",
          [p["n"] for p in p_fail] == [11, 30] and e_fail[0][1].startswith("status failed"))
    single = make_report(tmp / "single", [30])
    check("a family with one neuron count is not a series",
          PS.series_of(PS.load_report(single)[0]) == {})


def test_n_source(tmp):
    print("\n[3] The neuron count comes from the data")
    rep = make_report(tmp / "names", [14, 23], misleading_names=True)
    pts, _ = PS.load_report(rep)
    check("n from the result file, not from a name saying 999", sorted(p["n"] for p in pts) == [14, 23])


def test_outputs(tmp):
    print("\n[4] Figure and CSVs")
    rep = make_report(tmp / "out_rep", [11, 20, 30, 40, 50])
    vram = tmp / "vram.csv"
    with open(vram, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mode", "gpu", "n", "vram_gib", "us_per_run_tick",
                                          "all_runs_fit", "status"])
        w.writeheader()
        for n in (11, 30, 50):
            for g, us in ((2, 1.3), (4, 1.05), (8, 1.0)):
                w.writerow({"mode": "vram", "gpu": "TEST", "n": n, "vram_gib": g,
                            "us_per_run_tick": us * n, "all_runs_fit": str(g == 8), "status": "ok"})
    out = tmp / "fig"
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        rc = PS.main([str(rep), "--out", str(out), "--vram-csv", str(vram)])
    files = ["neuron_scaling.png", "neuron_scaling.pdf", "points.csv", "fits.csv"]
    check(f"exit 0 and {files} written", rc == 0 and all((out / f).is_file() for f in files))
    with open(out / "fits.csv", newline="") as f:
        fit = list(csv.DictReader(f))
    check(f"fits.csv: exponent 2.000 ({fit[0]['exponent_k'] if fit else '-'})",
          len(fit) == 1 and fit[0]["exponent_k"] == "2.000")
    with open(out / "points.csv", newline="") as f:
        check("points.csv has every plotted point", len(list(csv.DictReader(f))) == 5)
    check("the printed summary names the exponent", "t ~ n^2.00" in buf.getvalue())


def test_vram_helpers():
    print("\n[5] width_probe --vram-gib helpers")
    import torch
    import mvb_torch.generation as G
    import tests.width_probe as WP
    check(f"auto, 8 GiB card: {WP.auto_sizes(8.0)}", WP.auto_sizes(8.0) == [2, 3, 4, 6, 8])
    check(f"auto, 24 GiB card ends at 24 ({WP.auto_sizes(24.0)[-3:]})",
          WP.auto_sizes(24.0) == [2, 3, 4, 6, 8, 10, 12, 16, 20, 24])
    check("auto, 80 GiB card: all 15 sizes", WP.auto_sizes(80.0) == [float(s) for s in WP.CARD_SIZES_GIB])
    check(f"auto, 7.63 GiB card (RTX 3070) ends with the card itself ({WP.auto_sizes(7.63)})",
          WP.auto_sizes(7.63) == [2, 3, 4, 6, 7.63])
    try:
        WP.parse_sizes("4,12", 8.0)
        refused = False
    except SystemExit as e:
        refused = "Emulation only works downwards" in str(e)
    check("a size above the card's memory is refused", refused)
    P, R, slot, bb = 260, 300, 200_000, 50_000_000
    ok = all(WP.emulated_width(P, R, slot, "cpu", bb, g)[0]
             == G.choose_width(P, R, slot, "cpu", batch_bytes=bb, total_memory=int(g * 2**30))[0]
             for g in (2, 4, 8, 24, 80))
    check("emulated widths equal choose_width(total_memory = X GiB)", ok)
    check("a budget too small for one slot gives no width (not a crash)",
          WP.emulated_width(P, R, 10**10, "cpu", bb, 2)[0] is None)
    plan = WP.plan_vram_rows([2, 4, 8, 16, 32], [1000, 2000, 78000, 78000, 78000], 78000)
    check(f"run until the first size where all runs fit, then 'not rerun' ({[x[2] for x in plan]})",
          [x[2] for x in plan] == ["run", "run", "run", "all runs fit (not rerun)",
                                   "all runs fit (not rerun)"])
    rows = [{"vram_gib": 2, "width": 1000, "status": "ok", "us_per_run_tick": 12.0, "all_runs_fit": False},
            {"vram_gib": 4, "width": 2000, "status": "ok", "us_per_run_tick": 10.4, "all_runs_fit": False},
            {"vram_gib": 8, "width": 4000, "status": "ok", "us_per_run_tick": 10.0, "all_runs_fit": False},
            {"vram_gib": 16, "width": 8000, "status": "ok", "us_per_run_tick": 10.6, "all_runs_fit": True},
            {"vram_gib": 32, "width": 8000, "status": "all runs fit (not rerun)", "all_runs_fit": True}]
    d = WP.derive_vram(rows)
    check(f"knee = smallest size within 5% of the best (4 GiB), best 8 GiB, all fit from 16 ({d})",
          d["knee_vram_gib"] == 4 and d["best_vram_gib"] == 8 and d["all_fit_vram_gib"] == 16)
    del torch


def main():
    with tempfile.TemporaryDirectory(prefix="test_plot_scaling_") as t:
        tmp = Path(t)
        test_aggregation(tmp)
        test_exclusions(tmp)
        test_n_source(tmp)
        test_outputs(tmp)
        test_vram_helpers()
    n_fail = sum(1 for _, ok in _RESULTS if not ok)
    if n_fail:
        print(f"\nFAILED -- {n_fail} of {len(_RESULTS)} checks failed")
        return 1
    print(f"\nOK -- {len(_RESULTS)}/{len(_RESULTS)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
