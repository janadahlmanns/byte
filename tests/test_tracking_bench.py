#!/usr/bin/env python3
"""Tests for tests/tracking_bench.py and analysis/plot_tracking.py (plan Step 31).

  [1] the case lists: groups A / B / C with the planned sizes, levels and devices
  [2] a tiny case (2 genomes x 3 runs, n = 11, CPU) at every tracking level: the right
      save functions with the right call counts; the datasets of each level present,
      the others absent
  [3] the timing wrappers change nothing: the HDF5 written with and without them is
      identical in content
  [4] status handling: per-tick at n = 50 fails on HDF5's header limit and is recorded
      ("object header message is too large"); a too-short time limit is recorded as a
      timeout; a CUDA case on a machine without CUDA is recorded as skipped; resume
      skips finished cases
  [5] the plot runs on a tiny report; its CSV agrees with the cases
  [6] scalar cases: every level runs (total time only), its file has the same dataset
      names as the tensor file; n = 50 per-tick fails on the header limit as in the
      tensor code; the CSV's speed-up = scalar total / tensor total

Usage
-----
    python -m tests.test_tracking_bench
"""

import csv
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analysis.plot_tracking as PT  # noqa: E402
import tests.tracking_bench as TB  # noqa: E402

_RESULTS = []


def check(name, cond):
    _RESULTS.append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


TINY = dict(group="A", population="elites_n11", P=2, R=3, device="cpu")


def test_cases():
    print("\n[1] Case lists")
    c = TB.case_list(["A", "B", "C"])
    check("A: 5 levels x CUDA + CPU + scalar, 10 x 300", sum(k.startswith("A_") for k in c) == 15
          and all((v["P"], v["R"]) == (10, 300) for k, v in c.items() if k.startswith("A_")))
    check("B: 5 levels on CUDA and scalar + off / run on CPU, 260 x 300", sum(k.startswith("B_") for k in c) == 12
          and all((v["P"], v["R"]) == (260, 300) for k, v in c.items() if k.startswith("B_")))
    check("C: off / run / heat / tick on CUDA and scalar, 10 x 100 at n = 50",
          sorted(v["level"] for k, v in c.items() if k.startswith("C_")) == sorted(["heat", "off", "run", "tick"] * 2)
          and all(v["population"] == "elites_n50" for k, v in c.items() if k.startswith("C_")))


def datasets(path):
    out = {}
    with h5py.File(path, "r") as f:
        f.visititems(lambda n, o: out.__setitem__(n, o[()]) if isinstance(o, h5py.Dataset) else None)
    return out


def test_levels(tmp):
    print("\n[2] Every tracking level, tiny")
    want = {"off": {}, "run": {"save_variant_summary_to_hdf5": 2, "save_wiring_to_hdf5": 2},
            "tick": {"save_per_tick_to_hdf5": 6}, "heat": {"save_heatmaps_to_hdf5": 6},
            "all": {"save_per_tick_to_hdf5": 6, "save_heatmaps_to_hdf5": 6}}
    rd = Path(tmp) / "levels"
    (rd / "cases").mkdir(parents=True)
    files = {}
    for lv, calls in want.items():
        r = TB.run_case(rd, f"t_{lv}", keep_files=True, case=dict(TINY, level=lv))
        got = {f: d["calls"] for f, d in r["breakdown"].items()}
        ok = r["status"] == "ok" and all(got.get(f) == n for f, n in calls.items())
        if lv == "off":
            ok &= not got and r["h5_mib"] == 0
        if lv in ("run", "heat"):
            ok &= "save_per_tick_to_hdf5" not in got
        if lv in ("run", "tick"):
            ok &= "save_heatmaps_to_hdf5" not in got
        check(f"{lv}: status ok, call counts {calls or 'none'} (got {got})", ok)
        h5 = list((rd / "runs" / f"t_{lv}").rglob("*.h5"))
        files[lv] = h5[0] if h5 else None
    names = {lv: set(datasets(p)) if p else set() for lv, p in files.items()}
    check("per_tick datasets only with tick / all; heat maps only with heat / all",
          any("per_tick" in n for n in names["tick"]) and not any("per_tick" in n for n in names["heat"])
          and any("staying" in n for n in names["heat"]) and not any("staying" in n for n in names["tick"])
          and any("per_tick" in n for n in names["all"]) and any("staying" in n for n in names["all"]))
    return files


def test_wrappers_change_nothing(tmp, files):
    print("\n[3] The timing wrappers change nothing")
    import mvb_torch.adapter as adapter
    from mvb_torch.generation import draw_seeds, make_simulation_rngs
    from tests.be_common import call_eval
    genomes, cfg = TB.population("elites_n11", 2)
    cfg["experiment"]["n_runs"], cfg["experiment"]["population_size"] = 3, 2
    ev = dict(cfg["experiment"]["evaluator"], device="cpu")
    ss = draw_seeds(*make_simulation_rngs(1), 2, 3)
    seeds = {k: np.asarray(getattr(ss, k)) for k in ("run_seeds", "noise_seeds", "decision_seeds")}
    folder = Path(tmp) / "plain"
    folder.mkdir()
    import importlib
    importlib.reload(adapter)                  # the module without any wrapper
    call_eval(adapter.make_tensor_evaluator(ev, cfg, None), genomes, cfg, str(folder), seeds=seeds,
              per_tick=True, per_run=True, heat_map=True)
    plain = datasets(next(folder.rglob("*.h5")))
    wrapped = datasets(files["all"])
    same = set(plain) == set(wrapped) and all(
        (np.array_equal(plain[k], wrapped[k]) if not isinstance(plain[k], bytes) else plain[k] == wrapped[k])
        for k in plain if "genome_properties" not in k)
    check(f"all {len(plain)} datasets identical with and without the wrappers", same)


def test_status(tmp):
    print("\n[4] Status handling")
    rd = Path(tmp) / "status"
    (rd / "cases").mkdir(parents=True)
    r = TB.run_case(rd, "c_tick", case=dict(group="C", population="elites_n50", P=1, R=1, level="tick", device="cpu"))
    check(f"n = 50 per-tick: recorded as failed on the header limit ({r['status'][:70]})",
          r["status"].startswith("failed") and "object header message is too large" in r["status"])
    r = TB.run_case(rd, "c_cuda", case=dict(TINY, level="run", device="cuda"))
    import torch
    if not torch.cuda.is_available():
        check("a CUDA case without CUDA: skipped, explicitly", r["status"] == "skipped: no CUDA on this machine")
    rd2 = Path(tmp) / "timeout"
    rd2.mkdir()
    (rd2 / "cases").mkdir()
    (rd2 / "run.json").write_text(json.dumps({"groups": ["A"], "cases": {"slow": dict(TINY, level="all"),
                                                                         "quick": dict(TINY, level="off")}}))
    TB.execute(rd2, timeouts={"A": 1})
    st = {n: json.loads((rd2 / "cases" / f"{n}.json").read_text())["status"] for n in ("slow", "quick")}
    check(f"a 1 s limit -> timeout recorded, the next case still runs ({st})",
          st["slow"].startswith("timeout") and st["quick"].startswith("timeout"))
    (rd2 / "cases" / "quick.json").unlink()
    before = (rd2 / "cases" / "slow.json").stat().st_mtime_ns
    TB.execute(rd2, timeouts={"A": 600})
    check("resume: the finished case is skipped, the missing one runs",
          (rd2 / "cases" / "slow.json").stat().st_mtime_ns == before
          and json.loads((rd2 / "cases" / "quick.json").read_text())["status"] == "ok")
    return rd2


def test_plot(tmp, rd2):
    print("\n[5] Plot")
    (rd2 / "system.json").write_text(json.dumps({"cpu_model": "test", "cpus_allowed": {"count": 4}, "gpus": []}))
    out = Path(tmp) / "plot"
    rc = PT.main([str(rd2), "--out", str(out)])
    with open(out / "tracking.csv") as f:
        rows = {r["case"]: r for r in csv.DictReader(f)}
    quick = json.loads((rd2 / "cases" / "quick.json").read_text())
    check("the plot and CSV are written; the CSV's sim_s equals the case's",
          rc == 0 and (out / "tracking.png").exists() and abs(float(rows["quick"]["sim_s"]) - quick["sim_s"]) < 1e-3
          and rows["slow"]["status"].startswith("timeout"))


def test_scalar(tmp, files):
    print("\n[6] Scalar cases")
    rd = Path(tmp) / "scalar"
    (rd / "cases").mkdir(parents=True)
    ok_all, names = True, {}
    for lv in TB.LEVELS:
        r = TB.run_case(rd, f"s_{lv}", keep_files=True, case=dict(TINY, level=lv, device="scalar"))
        ok_all &= r["status"] == "ok" and r["wall_s"] > 0 and "breakdown" not in r
        h5 = list((rd / "runs" / f"s_{lv}").rglob("*.h5"))
        names[lv] = set(datasets(h5[0])) if h5 else set()
        (rd / "cases" / f"s_{lv}.json").write_text(json.dumps(r))
    check("scalar: every level runs, a total time, no breakdown", ok_all)
    tensor_names = set(datasets(files["all"]))
    check(f"scalar 'all' writes the same dataset names as the tensor code ({len(names['all'])})",
          names["all"] == tensor_names)
    r = TB.run_case(rd, "s_c_tick", case=dict(group="C", population="elites_n50", P=1, R=1, level="tick", device="scalar"))
    check(f"scalar n = 50 per-tick fails on the header limit, as the tensor code ({r['status'][:60]})",
          r["status"].startswith("failed") and "header message is too large" in r["status"])
    rt = TB.run_case(rd, "t_all", keep_files=False, case=dict(TINY, level="all"))
    (rd / "cases" / "t_all.json").write_text(json.dumps(rt))
    (rd / "system.json").write_text(json.dumps({"cpu_model": "test", "cpus_allowed": {"count": 4}, "gpus": []}))
    out = Path(tmp) / "plot_scalar"
    PT.main([str(rd), "--out", str(out)])
    with open(out / "tracking.csv") as f:
        rows = {r["case"]: r for r in csv.DictReader(f)}
    s_all = json.loads((rd / "cases" / "s_all.json").read_text())
    check("the CSV's speed-up vs scalar = scalar total / tensor total",
          abs(float(rows["t_all"]["speedup_vs_scalar"]) - s_all["wall_s"] / rt["wall_s"]) < 1e-3)


def main():
    test_cases()
    with tempfile.TemporaryDirectory() as tmp:
        files = test_levels(tmp)
        test_wrappers_change_nothing(tmp, files)
        rd2 = test_status(tmp)
        test_plot(tmp, rd2)
        test_scalar(tmp, files)
    failed = [n for n, ok in _RESULTS if not ok]
    print(f"\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    for n in failed:
        print(f"  FAILED: {n}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
