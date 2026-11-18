#!/usr/bin/env python3
"""Tests for tests/tracking_bench.py and analysis/plot_tracking.py (plan Step 31).

  [1] the case lists: groups A / B / C with the planned sizes, levels and devices
  [2] a tiny case (2 genomes x 3 runs, n = 11, CPU) at every tracking level: the right
      save functions with the right call counts; the datasets of each level present,
      the others absent
  [3] the timing wrappers change nothing: the HDF5 written with and without them is
      identical in content
  [4] status handling: per-tick at n = 50 runs (split records, Step 33); a failing case
      is recorded as failed; a too-short time limit is recorded as a
      timeout; a CUDA case on a machine without CUDA is recorded as skipped; resume
      skips finished cases
  [5] the plot runs on a tiny report; its CSV agrees with the cases
  [6] scalar cases: every level runs (total time only), its file has the same dataset
      names as the tensor file; n = 50 per-tick runs, as in the tensor code; the CSV's
      speed-up = scalar total / tensor total
  [7] Step 32: writer and staging per case. A per_call case (the old writer) and a
      staged one_file case write the same datasets as the one_file case; the per_call
      case calls the save functions R times per kind, the one_file case none; phases
      (build_s, h5_s, stage_in_s / stage_out_s) recorded; the warm-up runs once per
      device and population and is not recorded; --stage-dir is required

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
    check("B: 5 levels on CUDA, CPU and scalar, 260 x 300; no _direct case without staging",
          sum(k.startswith("B_") for k in c) == 15 and "B_cuda_all_direct" not in c
          and all((v["P"], v["R"]) == (260, 300) for k, v in c.items() if k.startswith("B_")))
    check("C: all 5 levels on CUDA and scalar, 10 x 100 at n = 50",
          sorted(v["level"] for k, v in c.items() if k.startswith("C_")) == sorted(list(TB.LEVELS) * 2)
          and all(v["population"] == "elites_n50" for k, v in c.items() if k.startswith("C_")))
    c = TB.case_list(["A", "B", "C"], stage="/st", scalar=False)
    check("staged, no scalar: tensor cases one_file staged in /st; B_cuda_all_direct unstaged; no scalar case",
          not any(v["device"] == "scalar" for v in c.values())
          and all(v["writer"] == "one_file" and v["stage"] == "/st" for k, v in c.items() if k != "B_cuda_all_direct")
          and c["B_cuda_all_direct"]["stage"] == "none" and c["B_cuda_all_direct"]["level"] == "all")


def datasets(path):
    out = {}
    with h5py.File(path, "r") as f:
        f.visititems(lambda n, o: out.__setitem__(n, o[()]) if isinstance(o, h5py.Dataset) else None)
    return out


def test_levels(tmp):
    print("\n[2] Every tracking level, tiny (one_file writer)")
    rd = Path(tmp) / "levels"
    (rd / "cases").mkdir(parents=True)
    files = {}
    for lv in TB.LEVELS:
        r = TB.run_case(rd, f"t_{lv}", keep_files=True, case=dict(TINY, level=lv))
        got = {f: d["calls"] for f, d in r["breakdown"].items()}
        ok = r["status"] == "ok"
        if lv == "off":
            ok &= not got and r["h5_mib"] == 0 and not r["phases"]
        else:
            # only the setup writes go through hdf5_utils; the tracking goes through one file
            ok &= set(got) == {"create_hdf5_file", "save_genome_properties_to_hdf5"}
            ok &= r["phases"].get("h5_s", 0) > 0 and "build_s" in r["phases"]
        check(f"{lv}: status ok, phases {sorted(r['phases'])}, hdf5_utils calls {sorted(got) or 'none'}", ok)
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
    check(f"n = 50 per-tick: ok since Step 33 (split records) ({r['status'][:70]})", r["status"] == "ok")
    from simulate import hdf5_utils as H
    H._SPLIT_ABOVE["per_tick"] = 5000           # the rule above HDF5's limit: the old failure
    try:
        r = TB.run_case(rd, "c_fail", case=dict(group="C", population="elites_n50", P=1, R=1, level="tick",
                                                device="cpu"))
    finally:
        H._SPLIT_ABOVE["per_tick"] = H.PER_TICK_SPLIT_ABOVE
    check(f"a failing case is recorded as failed, not raised ({r['status'][:60]})",
          r["status"].startswith("failed") and "header message is too large" in r["status"])
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
    check(f"scalar n = 50 per-tick: ok since Step 33, as the tensor code ({r['status'][:60]})", r["status"] == "ok")
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


def test_step32(tmp, files):
    print("\n[7] Step 32: writer, staging, warm-up")
    import contextlib
    import io
    rd = Path(tmp) / "step32"
    (rd / "cases").mkdir(parents=True)
    r = TB.run_case(rd, "pc_all", keep_files=True, case=dict(TINY, level="all", writer="per_call", stage="none"))
    got = {f: d["calls"] for f, d in r["breakdown"].items()}
    h5 = list((rd / "runs" / "pc_all").rglob("*.h5"))
    check(f"per_call: the save functions are called per dataset (per_tick {got.get('save_per_tick_to_hdf5')}, "
          f"heat maps {got.get('save_heatmaps_to_hdf5')}); same dataset names as one_file",
          r["status"] == "ok" and got.get("save_per_tick_to_hdf5") == 6 and got.get("save_heatmaps_to_hdf5") == 6
          and h5 and set(datasets(h5[0])) == set(datasets(files["all"])))
    st = Path(tmp) / "stage"
    st.mkdir()
    r = TB.run_case(rd, "st_all", keep_files=True, case=dict(TINY, level="all", writer="one_file", stage=str(st)))
    h5 = list((rd / "runs" / "st_all").rglob("*.h5"))
    a, b = datasets(h5[0]) if h5 else {}, datasets(files["all"])
    same = set(a) == set(b) and all(
        (np.array_equal(a[k], b[k]) if not isinstance(a[k], bytes) else a[k] == b[k])
        for k in a if "genome_properties" not in k)
    check(f"staged: same datasets and values as unstaged; stage copies timed "
          f"({sorted(k for k in r['phases'] if k.startswith('stage'))}); stage folder empty",
          r["status"] == "ok" and same and {"stage_in_s", "stage_out_s"} <= set(r["phases"]) and not list(st.iterdir()))

    rw = Path(tmp) / "warm"
    (rw / "cases").mkdir(parents=True)
    (rw / "run.json").write_text(json.dumps({"groups": ["A"], "cases": {
        "w1": dict(TINY, level="off"), "w2": dict(TINY, level="run")}}))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        TB.execute(rw, timeouts={"A": 600})
    out = buf.getvalue()
    check(f"warm-up: once per device and population, before the first case, not recorded "
          f"({out.count('[warm-up] cpu elites_n11')} start line(s))",
          out.count("[warm-up] cpu elites_n11") == 1 and "[warm-up] done" in out
          and out.index("[warm-up]") < out.index("[w1]")
          and sorted(p.name for p in (rw / "cases").glob("*.json")) == ["w1.json", "w2.json"])
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            TB.main(["--cases", "A", "--out", str(Path(tmp) / "no_stage")])
        code = None
    except SystemExit as e:
        code = e.code
    check("a new run without --stage-dir is refused", code == 2 and not (Path(tmp) / "no_stage").exists())


def main():
    test_cases()
    with tempfile.TemporaryDirectory() as tmp:
        files = test_levels(tmp)
        test_wrappers_change_nothing(tmp, files)
        rd2 = test_status(tmp)
        test_plot(tmp, rd2)
        test_scalar(tmp, files)
        test_step32(tmp, files)
    failed = [n for n, ok in _RESULTS if not ok]
    print(f"\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    for n in failed:
        print(f"  FAILED: {n}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
