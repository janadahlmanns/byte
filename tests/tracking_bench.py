#!/usr/bin/env python3
"""What tracking costs in the tensor code (plan_evotorch.md, Steps 31 and 32). Measurement only.

Each case runs the tensor evaluator exactly as `run_batch` calls it (eval_generation's
signature, the tracking flags, a fresh HDF5), in its own subprocess, and records:

  sim_s           the simulation without the compile warm-up (batch call - warm-up)
  warmup_s        the compile warm-up
  write_s         adapter.write_tracking (all HDF5 writing after the generation)
  phases          write_tracking's own timings (Step 32): build_s (per-tick tables, heat
                  maps), h5_s (HDF5 calls), stage_in_s / stage_out_s (copies, if staged)
  breakdown       per hdf5_utils save function that was called: calls and seconds
                  (timers wrapped around the functions in this process only); since
                  Step 32 only the setup writes and per_call cases call them
  wall_s, h5_mib, gpu_peak_mib, rss_peak_mib, status (ok / failed: ... / timeout)

Tracking levels: off, run (per-run), tick (per-run + per-tick), heat (per-run + heat
maps), all (all three). The scalar code (device "scalar") runs the same cases; its workers
write during the generation, so only its total time (wall_s) is recorded. Cases (uni PC):
  A  replay a few genomes: the 10 production lookup elites, n = 11, R 300; all levels,
     CUDA, CPU and scalar
  B  a whole production generation: the 260-genome Step 12 population, n = 11, R 300;
     all levels on CUDA, CPU and scalar (30-minute limit per case). With --stage-dir set,
     also B_cuda_all_direct: all levels written in place, to separate the one-file
     writer's gain from staging's
  C  large brains: the first 10 large-run elites, n = 50, R 100; off / run / heat /
     tick on CUDA and scalar (tick: expected to fail on HDF5's header limit,
     known_issues.md)

Each tensor case records its writer (Step 32: one_file; reports from before Step 32 have
none, i.e. per_call) and its staging folder. --stage-dir is required for a new run
(a local folder, or none). --skip-scalar leaves the scalar cases out (the scalar code did
not change in Step 32, so the earlier reports' scalar cases still hold). Before the first
case of each device and population, an untimed warm-up case fills the compile cache.

Usage
-----
    python -m tests.tracking_bench --cases A B C --stage-dir /tmp --skip-scalar
    python -m tests.tracking_bench --resume data/runtime_reports/<folder>
The HDF5 files are deleted after their size is recorded (--keep-files keeps them).
"""

import argparse
import csv
import datetime
import functools
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LEVELS = {"off": (False, False, False), "run": (True, False, False), "tick": (True, True, False),
          "heat": (True, False, True), "all": (True, True, True)}
SAVE_FUNCS = ("create_hdf5_file", "save_genome_properties_to_hdf5", "save_variant_summary_to_hdf5",
              "save_wiring_to_hdf5", "save_modulation_to_hdf5", "save_eta_to_hdf5",
              "save_tonic_activations_to_hdf5", "save_per_tick_to_hdf5", "save_heatmaps_to_hdf5")
TIMEOUT_S = {"A": 1800, "B": 1800, "C": 1800}


def case_list(groups, stage="none", scalar=True):
    """name -> dict(group, population, P, R, level, device[, writer, stage]).
    Tensor cases write with the one-file writer (Step 32), staged in `stage` ('none':
    in place)."""
    out = {}
    pops = {"A": ("elites_n11", 10, 300), "B": ("step12_n11", 260, 300), "C": ("elites_n50", 10, 100)}
    devices = {"A": ("cuda", "cpu"), "B": ("cuda", "cpu"), "C": ("cuda",)}
    levels = {"A": list(LEVELS), "B": list(LEVELS), "C": ["off", "run", "heat", "tick"]}
    for g in groups:
        if g not in pops:
            raise SystemExit(f"ERROR: unknown case group {g!r} (A, B, C)")
        pop, P, R = pops[g]
        for dev in devices[g] + (("scalar",) if scalar else ()):
            for lv in levels[g]:
                c = dict(group=g, population=pop, P=P, R=R, level=lv, device=dev)
                if dev != "scalar":
                    c.update(writer="one_file", stage=stage)
                out[f"{g}_{dev}_{lv}"] = c
        if g == "B" and stage != "none":
            out["B_cuda_all_direct"] = dict(group="B", population=pop, P=P, R=R, level="all", device="cuda",
                                            writer="one_file", stage="none")
    return out


def population(kind, P):
    """(genomes, config) of a case. Elites from tests/refs/behaviour (the production runs'
    final elites), or the Step 12 part a population."""
    import tests.behaviour_equivalence as BE
    from mvb.genome.generate_genome_from_file import generate_genome_from_file
    if kind == "step12_n11":
        spec = BE.PARTS["a"]
        cfg = BE.load_cfg(spec["config"])
        g, _ = BE.build_population(cfg, spec["elites"], spec["blocks"])
        if len(g) != P:
            raise SystemExit(f"ERROR: the Step 12 population has {len(g)} genomes, the case says {P}")
        return g, cfg
    part = {"elites_n11": "a", "elites_n50": "b"}[kind]
    spec = BE.PARTS[part]
    cfg = BE.load_cfg(spec["config"])
    return [generate_genome_from_file(cfg, elite_id=e, hdf5_path=str(ROOT / spec["elites"]))
            for e in range(P)], cfg


# ---------------------------------------------------------------------------
# One case, in its own process
# ---------------------------------------------------------------------------

def draw_case_seeds(P, R):
    """The seeds eval_generation draws with simulation seed 1 (as mvb_torch.generation's
    draw_seeds; numpy only, so a scalar case never imports torch before its worker pool
    forks)."""
    import numpy as np
    ss = np.random.SeedSequence(1).spawn(4)
    rng_noise, rng_decision, rng_world = (np.random.default_rng(s) for s in ss[:3])
    run_seeds = rng_world.integers(0, 2 ** 32, size=R, dtype=np.uint32)
    noise = np.stack([rng_noise.integers(0, 2 ** 32, size=R, dtype=np.uint32) for _ in range(P)])
    decision = np.stack([rng_decision.integers(0, 2 ** 32, size=R, dtype=np.uint32) for _ in range(P)])
    return {"run_seeds": run_seeds, "noise_seeds": noise, "decision_seeds": decision}


def run_scalar_case(rdir, name, case, out, genomes, cfg, keep_files):
    """The scalar code (mvb.simulation_API.eval_generation, worker processes that write
    the HDF5 themselves under a lock): total time only."""
    import mvb.simulation_API as api
    from tests.be_common import call_eval
    per_run, per_tick, heat = LEVELS[case["level"]]
    folder = Path(rdir) / "runs" / name
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    t0 = time.perf_counter()
    try:
        call_eval(api.eval_generation, genomes, cfg, str(folder), seeds=draw_case_seeds(case["P"], case["R"]),
                  per_tick=per_tick, per_run=per_run, heat_map=heat)
        out["status"] = "ok"
    except Exception as e:  # noqa: BLE001 -- recorded as the case's result
        out["status"] = f"failed: {type(e).__name__}: {str(e).splitlines()[0][:200]}"
    out["wall_s"] = time.perf_counter() - t0
    out["h5_mib"] = sum(p.stat().st_size for p in folder.rglob("*.h5")) / 2 ** 20
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out["rss_peak_mib"] = rss / 2 ** 20 if sys.platform == "darwin" else rss / 2 ** 10   # main process only
    if not keep_files:
        shutil.rmtree(folder)
    return out


def run_case(rdir, name, keep_files=False, case=None):
    rdir = Path(rdir)
    case = case or json.loads((rdir / "run.json").read_text())["cases"][name]
    out = dict(name=name, **case)
    genomes, cfg = population(case["population"], case["P"])
    cfg["experiment"]["n_runs"] = case["R"]
    cfg["experiment"]["population_size"] = case["P"]
    if case["device"] == "scalar":
        cfg["experiment"]["evaluator"] = {"backend": "scalar"}
        return run_scalar_case(rdir, name, case, out, genomes, cfg, keep_files)
    import numpy as np
    import torch
    import mvb_torch.adapter as adapter
    from mvb_torch.generation import draw_seeds, make_simulation_rngs
    from tests.be_common import call_eval
    ev = dict(cfg["experiment"]["evaluator"], device=case["device"])
    if case.get("stage", "none") != "none":
        ev["tracking_stage_dir"] = case["stage"]
    if case["device"] == "cuda" and not torch.cuda.is_available():
        out["status"] = "skipped: no CUDA on this machine"
        return out

    calls = {f: [0, 0.0] for f in SAVE_FUNCS}

    def timed(fname, fn):
        @functools.wraps(fn)
        def w(*a, **k):
            t = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                calls[fname][0] += 1
                calls[fname][1] += time.perf_counter() - t
        return w

    originals = {f: getattr(adapter, f) for f in SAVE_FUNCS + ("eval_generation_batch", "write_tracking")}
    try:
        return _run_tensor_case(rdir, name, case, out, genomes, cfg, keep_files, ev, calls, timed, adapter)
    finally:                                   # restored: cases run in one process in the tests
        for f, fn in originals.items():
            setattr(adapter, f, fn)


def _run_tensor_case(rdir, name, case, out, genomes, cfg, keep_files, ev, calls, timed, adapter):
    import numpy as np
    import torch
    from mvb_torch.generation import draw_seeds, make_simulation_rngs
    from tests.be_common import call_eval
    for f in SAVE_FUNCS:                       # the names the adapter imported
        setattr(adapter, f, timed(f, getattr(adapter, f)))
    batch = {}
    real_batch, real_write = adapter.eval_generation_batch, adapter.write_tracking

    def batch_spy(*a, **k):
        t = time.perf_counter()
        r = real_batch(*a, **k)
        batch["s"], batch["res"] = time.perf_counter() - t, r
        return r

    def write_spy(*a, **k):
        if case.get("writer", "one_file") == "per_call":
            k["per_call"] = True
        k["timings"] = phases = k.get("timings") if k.get("timings") is not None else {}
        t = time.perf_counter()
        try:
            return real_write(*a, **k)
        finally:
            batch["write_s"] = time.perf_counter() - t
            batch["phases"] = dict(phases)

    adapter.eval_generation_batch, adapter.write_tracking = batch_spy, write_spy
    per_run, per_tick, heat = LEVELS[case["level"]]
    folder = rdir / "runs" / name
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    ss = draw_seeds(*make_simulation_rngs(1), case["P"], case["R"])
    seeds = {"run_seeds": np.asarray(ss.run_seeds), "noise_seeds": np.asarray(ss.noise_seeds),
             "decision_seeds": np.asarray(ss.decision_seeds)}
    t0 = time.perf_counter()
    try:
        evaluate = adapter.make_tensor_evaluator(ev, cfg, None)
        call_eval(evaluate, genomes, cfg, str(folder), seeds=seeds, per_tick=per_tick,
                  per_run=per_run, heat_map=heat)
        out["status"] = "ok"
    except Exception as e:  # noqa: BLE001 -- recorded as the case's result
        out["status"] = f"failed: {type(e).__name__}: {str(e).splitlines()[0][:200]}"
    out["wall_s"] = time.perf_counter() - t0
    res = batch.get("res")
    if res is not None:
        out["warmup_s"] = float(res.warmup_s)
        out["sim_s"] = batch["s"] - float(res.warmup_s)
        out["run_ticks"] = int(res.lifespans.sum())
        out["iterations"] = int(res.iterations)
    out["write_s"] = batch.get("write_s", 0.0)
    out["phases"] = batch.get("phases", {})
    out["breakdown"] = {f: {"calls": c, "s": s} for f, (c, s) in calls.items() if c}
    h5 = list(folder.rglob("*.h5"))
    out["h5_mib"] = sum(p.stat().st_size for p in h5) / 2 ** 20
    if case["device"] == "cuda":
        out["gpu_peak_mib"] = torch.cuda.max_memory_reserved() / 2 ** 20
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out["rss_peak_mib"] = rss / 2 ** 20 if sys.platform == "darwin" else rss / 2 ** 10
    if not keep_files:
        shutil.rmtree(folder)
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def plan(rdir, groups, stage="none", scalar=True):
    import mvb_torch.h5_writer as W
    if stage != "none":
        stage = str(W.check_stage_dir(stage))       # refused here, before any case runs
    rdir.mkdir(parents=True)
    (rdir / "cases").mkdir()
    run = dict(groups=groups, cases=case_list(groups, stage, scalar), writer="one_file (Step 32)", stage=stage,
               created=datetime.datetime.now().isoformat(timespec="seconds"))
    (rdir / "run.json").write_text(json.dumps(run, indent=1))
    from simulate import machine_load
    from simulate.run_selected import system_info, torch_probe
    info = system_info(torch_probe())
    info["idle_check"] = machine_load.idle_check()
    info["filesystem"] = {"report": W.filesystem_of(rdir),
                          "stage": W.filesystem_of(stage) if stage != "none" else None}
    (rdir / "system.json").write_text(json.dumps(info, indent=1, default=str))
    print(f"[tracking_bench] idle check: {info['idle_check'].get('verdict')}", flush=True)
    return run


WARM_CASE = dict(group="warm", P=2, R=3, level="off")


def warm_up(rdir, case):
    """An untimed tiny case on the device and population of `case`, so the first timed
    case does not pay a cold compile cache (Step 31: A_cuda_off's warm-up 15 s, not 3 s).
    Its result is printed, not recorded; a failure is only reported (the timed case
    will show the real error)."""
    w = dict(WARM_CASE, population=case["population"], device=case["device"])
    print(f"[warm-up] {case['device']} {case['population']} (untimed) ...", flush=True)
    cmd = [sys.executable, "-m", "tests.tracking_bench", "--warm-case", json.dumps(w), "--dir", str(rdir)]
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=1800)
    print(f"[warm-up] {'done' if p.returncode == 0 else 'FAILED: ' + ((p.stderr or p.stdout).strip().splitlines() or ['?'])[-1][:200]}",
          flush=True)


def execute(rdir, keep_files=False, timeouts=None, warm=True):
    run = json.loads((rdir / "run.json").read_text())
    timeouts = timeouts or TIMEOUT_S
    warmed = set()
    for name, case in run["cases"].items():
        f = rdir / "cases" / f"{name}.json"
        if f.exists():
            print(f"[{name}] done earlier, skipped", flush=True)
            continue
        key = (case["device"], case["population"])
        if warm and case["device"] != "scalar" and key not in warmed:
            warmed.add(key)
            warm_up(rdir, case)
        print(f"[{name}] {case['group']}: {case['P']} genomes x {case['R']} runs, level {case['level']}, "
              f"{case['device']}", flush=True)
        cmd = [sys.executable, "-m", "tests.tracking_bench", "--run-case", name, "--dir", str(rdir)]
        if keep_files:
            cmd.append("--keep-files")
        t0 = time.perf_counter()
        try:
            p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                               timeout=timeouts[case["group"]])
            (rdir / "cases" / f"{name}.log").write_text(p.stdout + p.stderr)
            if p.returncode != 0 or not f.exists():
                res = dict(name=name, **case, status=f"failed: exit {p.returncode}: "
                           + ((p.stderr or p.stdout).strip().splitlines() or ["?"])[-1][:200],
                           wall_s=time.perf_counter() - t0)
                f.write_text(json.dumps(res, indent=1))
        except subprocess.TimeoutExpired:
            res = dict(name=name, **case, status=f"timeout after {timeouts[case['group']]} s",
                       wall_s=time.perf_counter() - t0)
            f.write_text(json.dumps(res, indent=1))
            shutil.rmtree(rdir / "runs" / name, ignore_errors=True)
        r = json.loads(f.read_text())
        print(f"[{name}] {r['status']}" + (f": sim {r.get('sim_s', 0):.1f} s, write {r.get('write_s', 0):.1f} s, "
                                           f"{r.get('h5_mib', 0):.0f} MiB" if r["status"] == "ok" else ""), flush=True)
    return write_tables(rdir)


def write_tables(rdir):
    run = json.loads((rdir / "run.json").read_text())
    rows, br = [], []
    for name in run["cases"]:
        f = rdir / "cases" / f"{name}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        rows.append({**{k: r.get(k, "") for k in ("name", "group", "device", "level", "P", "R", "writer", "stage",
                                                  "status", "wall_s", "warmup_s", "sim_s", "write_s")},
                     **{k: (r.get("phases") or {}).get(k, "") for k in ("build_s", "h5_s", "stage_in_s", "stage_out_s")},
                     **{k: r.get(k, "") for k in ("run_ticks", "iterations", "h5_mib", "gpu_peak_mib", "rss_peak_mib")}})
        for fn, d in (r.get("breakdown") or {}).items():
            br.append(dict(name=name, function=fn, calls=d["calls"], seconds=f"{d['s']:.4f}"))
    for fname, data in (("summary.csv", rows), ("write_breakdown.csv", br)):
        if data:
            with open(rdir / fname, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(data[0]))
                w.writeheader()
                w.writerows(data)
    bad = [r["name"] for r in rows if not str(r["status"]).startswith(("ok", "skipped"))]
    print(f"\n{len(rows)} cases written to {rdir}" + (f"; not ok: {bad}" if bad else ""))
    return rows


def default_dir():
    gpu = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0).replace(" ", "-")
    except Exception:  # noqa: BLE001 -- only names the folder
        pass
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return ROOT / "data" / "runtime_reports" / f"{ts}_{gpu}_tracking"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", nargs="+", choices=["A", "B", "C"])
    ap.add_argument("--out")
    ap.add_argument("--resume", metavar="DIR")
    ap.add_argument("--keep-files", action="store_true")
    ap.add_argument("--stage-dir", help="tensor cases stage their file here (a local folder) or 'none'; "
                                        "required for a new run")
    ap.add_argument("--skip-scalar", action="store_true",
                    help="no scalar cases (reuse the earlier reports' scalar cases in the plot)")
    ap.add_argument("--run-case", help=argparse.SUPPRESS)
    ap.add_argument("--warm-case", help=argparse.SUPPRESS)
    ap.add_argument("--dir", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.warm_case:
        res = run_case(a.dir, "_warm_up", case=json.loads(a.warm_case))
        shutil.rmtree(Path(a.dir) / "runs" / "_warm_up", ignore_errors=True)
        print(f"warm-up: {res['status']}")
        return 0 if res["status"] == "ok" else 1
    if a.run_case:
        res = run_case(a.dir, a.run_case, a.keep_files)
        (Path(a.dir) / "cases" / f"{a.run_case}.json").write_text(json.dumps(res, indent=1))
        return 0
    if a.resume:
        rdir = Path(a.resume).resolve()
    else:
        if not a.cases:
            ap.error("give --cases (A, B, C) or --resume")
        if a.stage_dir is None:
            ap.error("give --stage-dir (a local folder, e.g. /tmp, or none)")
        rdir = (Path(a.out) if a.out else default_dir()).resolve()
        if rdir.exists():
            raise SystemExit(f"ERROR: {rdir} exists; use --resume")
        plan(rdir, a.cases, a.stage_dir, scalar=not a.skip_scalar)
    execute(rdir, a.keep_files)
    return 0


if __name__ == "__main__":
    sys.exit(main())
