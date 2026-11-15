#!/usr/bin/env python3
"""Tests for the break-even sweeps and analysis/plot_breakeven.py (plan Step 28).

  [1] sweep files: 6 + 6 + 2 points; each family's scalar and CUDA files identical apart
      from the backend; elite_size 1 < every P; R, generations, n as planned; the random
      family equals the Step 22 population sweep apart from elite_size (so it extends
      that curve); the presets list exactly these points
  [2] run_ea accepts the smallest point (P = 2, elite 1) on both backends (tiny R)
  [3] plot on synthetic reports with exact power laws: the break-even is interpolated
      exactly; no crossing is reported as such; files written
  [4] refusals: a missing backend point, a config difference beyond the backend, a
      wrong backend in the report, mixed machines, a disturbed point (unless
      --include-disturbed)

Usage
-----
    python -m tests.test_breakeven
"""

import copy
import csv
import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analysis.plot_breakeven as PB  # noqa: E402
import simulate.run_selected as RS  # noqa: E402
from tests.test_scaling import run_ea_in  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SWEEPS = ROOT / "configs" / "sweeps"
_RESULTS = []
PLAN = {"lookup": ([2, 4, 8, 16, 32, 64], 300, 11), "random": ([2, 4, 8, 16, 25, 50], 100, 11),
        "n50": ([2, 4], 300, 50)}


def check(name, cond):
    _RESULTS.append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def points(fam, be):
    return RS.expand_sweep(SWEEPS / f"breakeven_{fam}_{be}.yaml")


def test_sweeps():
    print("\n[1] Sweep files and presets")
    for fam, (Ps, R, n) in PLAN.items():
        sc, cu = points(fam, "scalar"), points(fam, "cuda")
        check(f"{fam}: {len(Ps)} points per backend", len(sc) == len(cu) == len(Ps))
        same = all(PB._strip(a[1]) == PB._strip(b[1]) for a, b in zip(sc, cu))
        check(f"{fam}: scalar and CUDA configs identical apart from the backend", same)
        ok = True
        for (_name, c) in sc + cu:
            x = c["experiment"]
            ok &= (x["n_runs"] == R and x["evolutionary_algorithm"]["num_generations"] == 4
                   and x["evolutionary_algorithm"]["elite_size"] == 1 < x["population_size"]
                   and c["brain"]["n_neurons"] == n)
        check(f"{fam}: R {R}, 4 generations, elite 1 < P, n {n}", ok)
        check(f"{fam}: P = {Ps}", [c["experiment"]["population_size"] for _n, c in sc] == Ps)
        check(f"{fam}: scalar backend scalar, CUDA backend tensor cuda",
              all(c["experiment"]["evaluator"] == {"backend": "scalar"} for _n, c in sc)
              and all(c["experiment"]["evaluator"]["device"] == "cuda" for _n, c in cu))
    old = {c["experiment"]["population_size"]: c
           for _n, c in RS.expand_sweep(SWEEPS / "scaling_population_scalar.yaml")}
    new = {c["experiment"]["population_size"]: c for _n, c in points("random", "scalar")}
    diffs = []
    for P in (25, 50):
        a, b = copy.deepcopy(old[P]), copy.deepcopy(new[P])
        for c in (a, b):
            for k in ("experiment.evolutionary_algorithm.elite_size", "experiment.simulation_name",
                      "experiment.output_folder"):
                PB.PS._drop(c, k)
        diffs.append(a == b)
    check("random family == Step 22 population sweep apart from elite_size (P = 25, 50)", all(diffs))
    presets = RS.load_presets()
    want = [n for fam in ("lookup", "random") for be in ("scalar",) for n, _c in points(fam, be)] + \
           [n for fam in ("lookup", "random") for n, _c in points(fam, "cuda")]
    check("preset breakeven: the 24 lookup + random points, scalar first",
          list(presets["breakeven"].configs) == want)
    check("preset breakeven_n50: the 4 n = 50 points",
          sorted(presets["breakeven_n50"].configs) == sorted(n for be in ("scalar", "cuda")
                                                             for n, _c in points("n50", be)))


def test_run_ea(tmp):
    print("\n[2] run_ea accepts the smallest point (P = 2, elite 1)")
    for be in ("scalar", "cuda"):
        cfg = copy.deepcopy(points("lookup", be)[0][1])
        x = cfg["experiment"]
        x["n_runs"] = 4
        x["evolutionary_algorithm"]["num_generations"] = 2
        if be == "cuda":
            x["evaluator"]["device"] = "cpu"          # this machine; the same code path
            x["evaluator"]["compile"] = False
        h5, log, rc = run_ea_in(cfg, tmp, f"p2_{be}", ROOT)
        check(f"{be}: run_ea with P = 2, elite 1 finishes (exit {rc}) and writes its HDF5",
              rc == 0 and h5 is not None and "[EA timing]" in log)


# ---------------------------------------------------------------------------
# synthetic reports
# ---------------------------------------------------------------------------

LAW = {"scalar": (0.05, 1.0), "cuda": (0.4, 0.3)}     # seconds per generation = a * P^k


def synth(root, name, fams=("random",), machine_gpu="NVIDIA GeForce RTX 3070", cpu="i7",
          drop=(), tamper=None, lie=(), busy=(), law=LAW):
    rdir = Path(root) / name
    (rdir / "configs").mkdir(parents=True)
    (rdir / "system.json").write_text(json.dumps({"hostname": "uni", "cpu_model": cpu,
                                                  "gpus": [{"name": machine_gpu}]}))
    summary, gens, i = [], [], 0
    for fam in fams:
        for be in ("scalar", "cuda"):
            for row_name, cfg in points(fam, be):
                if row_name in drop:
                    continue
                i += 1
                cfg = copy.deepcopy(cfg)
                if tamper and tamper[0] == row_name:
                    tamper[1](cfg)
                cp = rdir / "configs" / f"{i:02d}.yaml"
                cp.write_text(yaml.safe_dump(cfg))
                log = f"{i:02d}.log"
                (rdir / log).write_text("compile_cache=/c/byte-w512-v1\n")
                P = cfg["experiment"]["population_size"]
                a, k = law[be]
                sim = a * P ** k
                backend = "scalar" if be == "scalar" else "tensor cuda"
                if row_name in lie:
                    backend = "tensor cuda" if be == "scalar" else "scalar"
                summary.append({"order": i, "config": row_name, "status": "ok", "log": log,
                                "config_path": str(cp), "backend": backend, "cpu_workers": str(min(P, 20)),
                                "disturbance": "before: GPU load 90%" if row_name in busy else "",
                                "wall_s": f"{(2 if be == 'scalar' else 8) + 4 * sim:.6f}",
                                "compile_cache_at_start": "warm" if be == "cuda" else ""})
                for g in range(4):
                    warm = 0.0 if be == "scalar" else (3.0 if g == 1 else 0.0)
                    gens.append({"order": i, "config": row_name, "generation": g,
                                 "evaluate_s": f"{sim + warm:.9f}", "genomes_s": "0.01",
                                 "select_save_s": "0.02",
                                 "simulate_s": f"{sim:.9f}" if be == "cuda" else "",
                                 "warmup_s": f"{warm:.9f}"})
    for fname, rows in (("summary.csv", summary), ("generations.csv", gens)):
        with open(rdir / fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return rdir


def run_plot(args):
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            rc = PB.main(args)
        return rc, buf.getvalue(), None
    except PB.BreakevenError as e:
        return None, buf.getvalue(), str(e)


def test_plot(tmp):
    print("\n[3] Plot on synthetic reports")
    rep = synth(tmp, "r_ok", fams=("lookup", "random"))
    out = Path(tmp) / "out_ok"
    rc, txt, err = run_plot([str(rep), "--out", str(out)])
    check(f"runs without error ({err})", rc == 0)
    exact = (8.0) ** (1 / 0.7)                       # 0.05 P = 0.4 P^0.3
    _rows, res = PB.summarise(PB.load_report(rep)[1])
    got = res["random"]["be_sim"][0]
    check(f"per-generation break-even interpolated exactly (P = {exact:.4f}; got {got})",
          len(got) == 1 and abs(got[0] - exact) < 1e-6)   # inputs written with 9 decimals
    check("both families' figures and the CSV written",
          all((out / f"breakeven_{f}.png").exists() for f in ("lookup", "random"))
          and (out / "breakeven.csv").exists())
    with open(out / "breakeven.csv") as f:
        n = sum(1 for _ in f) - 1
    check(f"CSV has one row per family and P ({n} = 12)", n == 12)
    rep2 = synth(tmp, "r_never", law={"scalar": (0.01, 0.2), "cuda": (0.5, 0.3)})
    _rows, res = PB.summarise(PB.load_report(rep2)[1])
    check(f"no crossing reported as such ({res['random']['be_sim'][1]})",
          res["random"]["be_sim"][0] == [] and "scalar faster at every" in res["random"]["be_sim"][1])


def test_refusals(tmp):
    print("\n[4] Refusals")
    name = "breakeven_random_cuda [population_size=8]"
    cases = [
        ("a missing CUDA point", dict(drop=(name,)), "no usable cuda point"),
        ("a config difference beyond the backend",
         dict(tamper=(name, lambda c: c["experiment"].__setitem__("n_runs", 99))), "differ beyond the backend"),
        ("a wrong backend in the report", dict(lie=(name,)), "the run's backend is"),
        ("a disturbed point (pair incomplete)", dict(busy=(name,)), "no usable cuda point"),
    ]
    for i, (what, kw, msg) in enumerate(cases):
        rep = synth(tmp, f"bad{i}", **kw)
        rc, _txt, err = run_plot([str(rep), "--out", str(Path(tmp) / f"o{i}")])
        check(f"{what} -> refused ({err})", rc is None and err and msg in err)
    rep = synth(tmp, "busy_ok", busy=(name,))
    rc, _t, err = run_plot([str(rep), "--out", str(Path(tmp) / "obusy"), "--include-disturbed"])
    check("... but accepted with --include-disturbed", rc == 0)
    a = synth(tmp, "m1", fams=("lookup",))
    b = synth(tmp, "m2", fams=("random",), machine_gpu="NVIDIA A100-SXM4-40GB")
    rc, _t, err = run_plot([str(a), str(b), "--out", str(Path(tmp) / "omix")])
    check(f"two machines -> refused ({err})", rc is None and err and "more than one kind of machine" in err)


def main():
    test_sweeps()
    with tempfile.TemporaryDirectory() as tmp:
        test_run_ea(tmp)
        test_plot(tmp)
        test_refusals(tmp)
    failed = [n for n, ok in _RESULTS if not ok]
    print(f"\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    for n in failed:
        print(f"  FAILED: {n}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
