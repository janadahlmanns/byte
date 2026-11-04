#!/usr/bin/env python3
"""Scaling sweeps, the backend-independent timing line, the machine-load checks and the
scaling figure (plan Step 22).

  [1] the nine sweep files configs/sweeps/scaling_<dim>_<backend>.yaml: they expand,
      the point counts, identical settings apart from the varied key and the backend,
      the backend blocks, unique names, the three presets resolve
  [2] run_ea's [EA timing] line: one per generation on the scalar and the tensor-CPU
      backend, run-ticks equal the tensor adapter's, and the HDF5 results identical to
      the committed code's (HEAD, without the line): the line is a print only
  [3] run_selected reads it: parse_backend_line, ea_summary, generations.csv rows for a
      scalar config, empty columns for an old log
  [4] simulate/machine_load.py on a fake /proc: process parsing, own process tree,
      idle_check verdicts, LoadSampler counting only foreign CPU time and GPU processes
  [5] analysis/plot_scaling.py on synthetic reports: known exponents recovered, and
      every loud check (mixed machines, mismatched setting, missing timing line,
      backend mismatch, busy machine excluded, missing points listed)

Usage
-----
    python -m tests.test_scaling
"""

import copy
import csv
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import redirect_stdout
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import analysis.plot_scaling as PS  # noqa: E402
import simulate.machine_load as ML  # noqa: E402
import simulate.run_selected as RS  # noqa: E402
from tests.test_ea_tensor import diff_h5, tensor_block  # noqa: E402

_RESULTS = []
SWEEPS = ROOT / "configs" / "sweeps"
DIMS = {"neurons": "brain.n_neurons", "population": "experiment.population_size",
        "runs": "experiment.n_runs"}
BACKENDS = ("scalar", "cpu", "cuda")


def check(name, condition):
    _RESULTS.append((name, bool(condition)))
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}")
    return bool(condition)


def _get(cfg, dotted):
    for k in dotted.split("."):
        cfg = cfg[k]
    return cfg


def all_points():
    """{(dim, backend): [(row name, cfg)]} of the nine sweep files."""
    return {(d, b): RS.expand_sweep(SWEEPS / f"scaling_{d}_{b}.yaml") for d in DIMS for b in BACKENDS}


# ============================================================

def test_sweeps():
    print("\n[1] The nine scaling sweep files")
    try:
        pts = all_points()
    except RS.SweepError as e:
        check(f"all nine expand ({e})", False)
        return
    check("all nine expand", True)
    counts = {k: len(v) for k, v in pts.items()}
    want = {(d, b): 7 if b == "cuda" and d != "neurons" else 5 for d in DIMS for b in BACKENDS}
    check(f"point counts: 5 per file, CUDA population/runs 7 ({counts})", counts == want)
    for (d, b), v in pts.items():
        xs = [_get(c, DIMS[d]) for _n, c in v]
        if b != "cuda" or d == "neurons":
            ok = xs == ([11, 20, 30, 40, 50] if d == "neurons" else [25, 50, 100, 200, 400])
        else:
            ok = xs == [25, 50, 100, 200, 400, 800, 1600]
        check(f"{d}/{b}: x values {xs}", ok and max(xs) <= (50 if d == "neurons" else 1600))

    # Identical apart from the varied key, the evaluator block and the names.
    def stripped(cfg):
        c = copy.deepcopy(cfg)
        for k in ("simulation_name", "output_folder", "evaluator", "population_size", "n_runs"):
            c["experiment"].pop(k)
        c["brain"].pop("n_neurons")
        return c
    flat = [(n, c) for v in pts.values() for n, c in v]
    ref = stripped(flat[0][1])
    odd = [n for n, c in flat if stripped(c) != ref]
    check(f"all 49 points identical apart from n, P, R, the evaluator and the names ({odd[:2]})",
          not odd)
    fixed = {"brain.n_neurons": 11, "experiment.population_size": 100, "experiment.n_runs": 100}
    bad = [n for (d, _b), v in pts.items() for n, c in v
           for k, val in fixed.items() if k != DIMS[d] and _get(c, k) != val]
    check(f"the keys not varied are n = 11, P = 100, R = 100 everywhere ({bad[:2]})", not bad)
    check("4 generations, output data/scaling/",
          all(c["experiment"]["evolutionary_algorithm"]["num_generations"] == 4
              and c["experiment"]["output_folder"] == "data/scaling/" for _n, c in flat))
    ev = {b: [c["experiment"]["evaluator"] for (_d, bb), v in pts.items() if bb == b for _n, c in v]
          for b in BACKENDS}
    cuda_block = ev["cuda"][0]
    check("scalar: evaluator is exactly {backend: scalar}",
          all(e == {"backend": "scalar"} for e in ev["scalar"]))
    check("tensor CPU: the CUDA block with device cpu",
          all(e == dict(cuda_block, device="cpu") for e in ev["cpu"]))
    check("tensor CUDA: backend tensor, device cuda, compiled",
          all(e == cuda_block for e in ev["cuda"]) and cuda_block["device"] == "cuda"
          and cuda_block["backend"] == "tensor" and cuda_block["compile"] is True)
    names = [n for n, _c in flat]
    sims = [c["experiment"]["simulation_name"] for _n, c in flat]
    check("row names and simulation names unique", len(set(names)) == 49 == len(set(sims)))
    infos = {n: RS.describe_cfg(c, n, "x", {"cpu", "cuda"}) for n, c in flat}
    check("the selector shows the right device and every point is selectable",
          all(infos[n].device == ("cuda" if "_cuda " in n else "cpu") and infos[n].selectable
              for n in names))
    # Step 26: the capacity files continue the CUDA population/runs sweeps.
    cap = {d: RS.expand_sweep(SWEEPS / f"scaling_{d}_cuda_capacity.yaml") for d in ("population", "runs")}
    check("capacity sweeps: P and R = 3200, 6400, 12800, 25600",
          all([_get(c, DIMS[d]) for _n, c in v] == [3200, 6400, 12800, 25600] for d, v in cap.items()))
    odd = [n for v in cap.values() for n, c in v if stripped(c) != ref
           or c["experiment"]["evaluator"] != cuda_block]
    fixed_bad = [n for d, v in cap.items() for n, c in v
                 for k, val in fixed.items() if k != DIMS[d] and _get(c, k) != val]
    check(f"... identical to the base sweeps apart from the varied key ({odd[:1]}, {fixed_bad[:1]})",
          not odd and not fixed_bad)
    cap_names = [n for v in cap.values() for n, _c in v]
    infos.update({n: RS.describe_cfg(c, n, "x", {"cpu", "cuda"}) for v in cap.values() for n, c in v})
    check("... names and simulation names unique, also against the base sweeps",
          len(set(cap_names + names)) == 57
          and len({c["experiment"]["simulation_name"] for v in cap.values() for _n, c in v} | set(sims)) == 57)
    for preset, want_n in (("scaling_cuda", 19), ("scaling_cpu", 30), ("scaling_all", 49)):
        p = RS.load_preset(ROOT / "configs" / "presets" / f"{preset}.yaml")
        missing = [c for c in p.configs if c not in infos]
        check(f"preset {preset}: {len(p.configs)} configs, all of them sweep points ({missing[:2]})",
              len(p.configs) == want_n and not missing and len(set(p.configs)) == want_n)
    p = RS.load_preset(ROOT / "configs" / "presets" / "scaling_cuda_capacity.yaml")
    check(f"preset scaling_cuda_capacity: the 8 capacity points ({len(p.configs)})",
          p.configs == cap_names)


# ============================================================

def tiny(backend):
    """A small random-genome EA of the scaling sweeps' kind: P = 6, R = 4, 3 generations."""
    cfg = RS.expand_sweep(SWEEPS / f"scaling_neurons_{backend}.yaml")[0][1]
    x = cfg["experiment"]
    x["population_size"], x["n_runs"] = 6, 4
    x["evolutionary_algorithm"]["num_generations"] = 3
    x["evolutionary_algorithm"]["elite_size"] = 2
    x["max_ticks"] = 300
    if backend == "cpu":
        x["evaluator"]["compile"] = False             # the line is the same; compiling is slow
    return cfg


def run_ea_in(cfg, tmp, tag, cwd):
    out = Path(tmp) / tag
    out.mkdir()
    cfg = copy.deepcopy(cfg)
    cfg["experiment"]["output_folder"] = str(out) + "/"
    path = Path(tmp) / f"{tag}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    p = subprocess.run([sys.executable, "-m", "simulate.run_ea", "--config", str(path)],
                       input="n\n", text=True, capture_output=True, cwd=cwd, timeout=1800)
    h5 = list(out.glob("*.h5"))
    return (h5[0] if len(h5) == 1 else None), p.stdout + p.stderr, p.returncode


def head_checkout(tmp):
    """The committed simulate/ (without the timing line), beside links to the rest."""
    old = Path(tmp) / "head"
    old.mkdir()
    arch = subprocess.run(["git", "-C", str(ROOT), "archive", "HEAD", "simulate"],
                          capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(old)], input=arch, check=True)
    for d in ("mvb", "mvb_torch", "configs"):
        (old / d).symlink_to(ROOT / d)
    return old


def test_timing_line(tmp):
    print("\n[2] run_ea's [EA timing] line")
    old = head_checkout(tmp)
    if "EA timing" in (old / "simulate" / "run_ea.py").read_text():
        print("      (HEAD already has the line: the identity check compares HEAD with itself)")
    for backend in ("scalar", "cpu"):
        cfg = tiny(backend)
        h5, out, rc = run_ea_in(cfg, tmp, f"new_{backend}", ROOT)
        if not check(f"{backend}: run_ea exits 0 (got {rc})", rc == 0 and h5):
            print(out[-2000:])
            continue
        parsed = {"ea": [], "backend": "scalar", "cpu_workers": ""}
        for line in out.splitlines():
            RS.parse_backend_line(line, parsed)
        gens = [g["generation"] for g in parsed["ea"]]
        check(f"{backend}: one line per generation, in order ({gens})", gens == [0, 1, 2])
        check(f"{backend}: every part measured, run-ticks between 1 and P*R*max_ticks",
              all(g["evaluate_s"] > 0 and g["genomes_s"] >= 0 and g["select_save_s"] > 0
                  and 0 < g["run_ticks"] <= 6 * 4 * 300 for g in parsed["ea"]))
        check(f"{backend}: backend and CPU parallelism read ({parsed['backend']}, "
              f"{parsed['cpu_workers']})",
              parsed["backend"] == ("scalar" if backend == "scalar" else "tensor cpu")
              and parsed["cpu_workers"])
        if backend == "cpu":
            stats = [int(m.group(1)) for m in map(RS.STATS_LINE.search, out.splitlines()) if m]
            check(f"tensor: run-ticks equal the adapter's ({stats} vs "
                  f"{[g['run_ticks'] for g in parsed['ea']]})",
                  stats == [g["run_ticks"] for g in parsed["ea"]])
        h5_old, out_old, rc_old = run_ea_in(cfg, tmp, f"head_{backend}", old)
        if not check(f"{backend}: the committed run_ea runs too (rc {rc_old})", rc_old == 0 and h5_old):
            print(out_old[-2000:])
            continue
        problems, n = diff_h5(str(h5), str(h5_old))
        check(f"{backend}: all {n} datasets identical to the committed code's "
              f"({len(problems)} differences)", not problems)
        for m in problems[:3]:
            print("      -", m)


# ============================================================

def test_parsing(tmp):
    print("\n[3] run_selected reads the line")
    lines = ["[tensor-evaluator] backend=tensor mode=live device=cpu dtype=float32 width=auto "
             "contraction=einsum rng=philox4x32-10 compile=on threads=8 compile_cache=/x/byte-w512-v1",
             "[EA timing] gen 0: genomes 0.010 s, evaluate 9.000 s, run-ticks 1000, 9000.00 us/run-tick, select+save 0.020 s",
             "[EA timing] gen 1: genomes 0.500 s, evaluate 2.000 s, run-ticks 1000, 2000.00 us/run-tick, select+save 0.030 s",
             "[EA timing] gen 2: genomes 0.700 s, evaluate 4.000 s, run-ticks 3000, 1333.33 us/run-tick, select+save 0.050 s"]
    p = {"ea": [], "backend": "scalar", "cpu_workers": ""}
    for line in lines:
        RS.parse_backend_line(line, p)
    check(f"tensor cpu with its thread count ({p['backend']}, {p['cpu_workers']})",
          p["backend"] == "tensor cpu" and p["cpu_workers"] == "8 torch threads")
    s = RS.ea_summary(p["ea"])
    check(f"gens 1+ medians and us/run-tick = 6 s / 4000 ticks ({s})",
          {k: s[k] for k in ("genomes", "evaluate", "select_save", "us")}
          == {"genomes": "0.600", "evaluate": "3.000", "select_save": "0.040", "us": "1500.00"})
    q = {"ea": [], "backend": "scalar", "cpu_workers": ""}
    RS.parse_backend_line("[INFO] Parallel execution on 18 cores (20 total).", q)
    check(f"scalar workers ({q['cpu_workers']})", q["cpu_workers"] == "18 processes")
    check("an old log: nothing", all(v == "" for v in RS.ea_summary([]).values()))
    rdir = Path(tmp) / "rep3"
    rdir.mkdir()
    r = RS.Result(order=1, config="scalar_cfg", path="x", status="ok", ea_times=p["ea"],
                  backend="scalar", cpu_workers="18 processes")
    r_old = RS.Result(order=2, config="old", path="y", status="ok",
                      gen_times=[(1.0, 10, 100)], gen_stats=[])
    RS.write_reports(rdir, [r, r_old])
    with open(rdir / "generations.csv", newline="") as f:
        g = list(csv.DictReader(f))
    check("generations.csv: 3 rows for a config with only [EA timing] lines, tensor columns empty",
          [x["generation"] for x in g if x["config"] == "scalar_cfg"] == ["0", "1", "2"]
          and all(x["seconds"] == "" and x["evaluate_s"] for x in g if x["config"] == "scalar_cfg"))
    check("... and an old config's row keeps its columns, the new ones empty",
          [(x["seconds"], x["evaluate_s"]) for x in g if x["config"] == "old"] == [("1.00", "")])
    with open(rdir / "summary.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    check("summary.csv: the Step 22 columns", rows[0]["evaluate_gens_1plus_median_s"] == "3.000"
          and rows[0]["ea_us_per_run_tick_gens_1plus"] == "1500.00" and rows[0]["ea_gens_timed"] == "3"
          and rows[1]["ea_gens_timed"] == "0")
    check("scalar: no compile, simulation = evaluation (Step 25)",
          rows[0]["simulate_gens_1plus_median_s"] == "3.000" and rows[0]["warmup_gens_1plus_total_s"] == "0.000"
          and all(x["warmup_s"] == "0.000" and x["simulate_s"] == x["evaluate_s"]
                  for x in g if x["config"] == "scalar_cfg"))

    # Step 25: the warm-up line of a tensor run.
    w = {"ea": [], "backend": "scalar", "cpu_workers": ""}
    for line in lines[:1] + [lines[1], "[tensor-evaluator] compile: warm-up 8.50 s, 9 new graph(s)",
                             lines[2], "[tensor-evaluator] compile: warm-up 1.50 s, 1 new graph(s)",
                             lines[3], "[tensor-evaluator] compile: warm-up 0.10 s, 0 new graph(s)"]:
        RS.parse_backend_line(line, w)
    check(f"WARMUP_LINE read per generation ({w['warmup']})",
          w["warmup"] == [(8.5, 9), (1.5, 1), (0.1, 0)])
    s = RS.ea_summary(w["ea"], w["backend"], w["warmup"])
    check(f"simulation = evaluation - warm-up: median of 0.5 and 3.9 s, us over 4.4 s / 4000 ticks ({s})",
          s["simulate"] == "2.200" and s["sim_us"] == "1100.00" and s["warmup_total"] == "1.600")
    rt = RS.Result(order=1, config="tensor_cfg", path="x", status="ok", ea_times=w["ea"],
                   backend="tensor cpu", warmups=w["warmup"])
    rt_old = RS.Result(order=2, config="tensor_old", path="y", status="ok", ea_times=w["ea"],
                       backend="tensor cpu")
    rdir2 = Path(tmp) / "rep3b"
    rdir2.mkdir()
    RS.write_reports(rdir2, [rt, rt_old])
    with open(rdir2 / "generations.csv", newline="") as f:
        g2 = list(csv.DictReader(f))
    new = [x for x in g2 if x["config"] == "tensor_cfg"]
    check(f"generations.csv: warm-up, new graphs, simulation per generation "
          f"({[(x['warmup_s'], x['new_graphs'], x['simulate_s']) for x in new]})",
          [(x["warmup_s"], x["new_graphs"], x["simulate_s"]) for x in new]
          == [("8.500", "9", "0.500"), ("1.500", "1", "0.500"), ("0.100", "0", "3.900")])
    check("a tensor log without the line (old): the Step 25 columns empty, not 0",
          all(x["warmup_s"] == "" and x["simulate_s"] == "" for x in g2 if x["config"] == "tensor_old"))


# ============================================================

def _atomic_write(path, text):
    """Write then rename: a reader never sees a half-written fake /proc file."""
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def fake_proc(root, cpu, procs):
    """Write a /proc look-alike: `cpu` = (busy, idle) jiffies, procs = {pid: (ppid, name, jiffies)}."""
    root = Path(root)
    root.mkdir(exist_ok=True)
    busy, idle = cpu
    _atomic_write(root / "stat", f"cpu  {busy} 0 0 {idle} 0 0 0 0 0 0\ncpu0 1 2 3 4\n")
    for d in root.iterdir():
        if d.name.isdigit() and int(d.name) not in procs:
            shutil.rmtree(d)
    for pid, (ppid, name, j) in procs.items():
        (root / str(pid)).mkdir(exist_ok=True)
        rest = ["S", str(ppid)] + ["0"] * 9 + [str(j), "0"] + ["0"] * 30
        _atomic_write(root / str(pid) / "stat", f"{pid} ({name}) " + " ".join(rest) + "\n")


def test_machine_load(tmp):
    print("\n[4] simulate/machine_load.py (fake /proc)")
    tick = os.sysconf("SC_CLK_TCK")
    proc = Path(tmp) / "proc"
    me = os.getpid()
    fake_proc(proc, (100, 900), {1: (0, "init", 0), me: (1, "python", 50),
                                  5001: (me, "python worker", 10),
                                  5002: (5001, "inductor (compile) )x", 0),
                                  7000: (1, "other user's job", 0)})
    ps = ML.processes(proc)
    check(f"process names with spaces and ')' parsed ({ps.get(5002)})",
          ps[5002][:3] == (5001, 0.0, "inductor (compile) )x") and ps[me][1] == 50 / tick)
    check(f"own tree: this process and its descendants ({sorted(ML.own_tree(ps))})",
          ML.own_tree(ps) == {me, 5001, 5002})
    check("cpu totals: busy, total", ML.cpu_totals(proc) == (100, 1000))

    saved = (ML.gpu_load, ML.gpu_processes, ML.cpu_totals)

    def idle_with(util, mem, gp, cpu_seq):
        seq = iter(cpu_seq)
        ML.gpu_load = lambda: [(util, mem)]
        ML.gpu_processes = lambda: gp
        ML.cpu_totals = lambda proc=None, cpus=None: next(seq)
        try:
            return ML.idle_check(seconds=0.05, interval=0.01, proc=proc, cpus=None)
        finally:
            ML.gpu_load, ML.gpu_processes, ML.cpu_totals = saved

    r = idle_with(2, 47, [], [(100, 1000), (102, 1100)])
    check(f"idle machine: ok ({r['verdict']}, CPU {r['cpu_busy_pct']}%)",
          r["verdict"] == "ok" and r["cpu_busy_pct"] == "2.0")
    r = idle_with(85, 2100, [(4242, "python", "2000 MiB")], [(100, 1000), (102, 1100)])
    check(f"busy GPU: load, memory and the process named ({r['verdict']})",
          r["verdict"].startswith("busy") and "85%" in r["verdict"] and "2100 MiB" in r["verdict"]
          and "pid 4242" in r["verdict"])
    r = idle_with(0, 47, [], [(100, 1000), (180, 1100)])
    check(f"busy CPU: 80% ({r['verdict']})", r["verdict"].startswith("busy: CPU 80% busy"))
    r = idle_with(0, 47, "unavailable: nvidia-smi not found", [(100, 1000), (101, 1100)])
    check(f"no nvidia-smi: the CPU still judged ({r['verdict']}, {r['gpu_processes']})",
          r["verdict"] == "ok" and r["gpu_processes"].startswith(ML.UNAVAILABLE))

    # LoadSampler: the foreign job gains 2 s of CPU per 1 s, our worker 4 s per 1 s.
    gp_saved = ML.gpu_processes
    ML.gpu_processes = lambda: [(5001, "python worker", "900 MiB"), (7000, "other", "300 MiB")]
    try:
        with ML.LoadSampler(interval=0.1, proc=proc, cpus=None) as s:
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < 1.0:
                dt = time.perf_counter() - t0
                fake_proc(proc, (100, 900), {1: (0, "init", 0), me: (1, "python", 50),
                                             5001: (me, "python worker", int(4 * dt * tick)),
                                             5002: (5001, "inductor (compile) )x", 0),
                                             7000: (1, "other user's job", int(2 * dt * tick))})
                time.sleep(0.02)
    finally:
        ML.gpu_processes = gp_saved
    cores = float(s.foreign_cores())
    check(f"foreign CPU ~2 cores, our worker's 4 not counted ({s.foreign_cores()}, {s.foreign_top()})",
          1.5 < cores < 2.3 and "other user's job" in s.foreign_top() and "worker" not in s.foreign_top())
    check(f"foreign GPU process found, ours not ({s.foreign_gpu_text()})",
          "7000" in s.foreign_gpu_text() and "5001" not in s.foreign_gpu_text())
    check(f"disturbance names both ({s.disturbance()})",
          "cores" in (s.disturbance() or "") and "GPU" in (s.disturbance() or ""))
    with ML.LoadSampler(interval=0.05, proc=proc, cpus=None) as quiet:
        time.sleep(0.3)
    check(f"nothing foreign: no disturbance ({quiet.foreign_cores()}, {quiet.disturbance()})",
          quiet.disturbance() is None)
    with ML.LoadSampler(interval=0.05, proc=Path(tmp) / "no_proc", cpus=None) as mac:
        time.sleep(0.1)
    check(f"no /proc: explicit unavailable ({mac.foreign_cores()})",
          mac.foreign_cores().startswith(ML.UNAVAILABLE) and mac.disturbance() is None)


# ============================================================

LAWS = {  # backend -> (a, k) of evaluate seconds = a * x^k, per dimension the same here
    "scalar": (0.02, 1.0), "cpu": (0.01, 0.8), "cuda": (0.5, 0.3)}


def synth_report(root, name, points, machine="uni", busy=(), drop_timing=(), backend_lie=(),
                 tamper=None, gpu="NVIDIA GeForce RTX 3070", old_format=()):
    """A run_selected-like report folder with exact power-law simulation timings. Tensor
    points compile 3 s in generations 1 and 2 (new list layouts, plan Step 25; two of
    three, as in the uni PC's CUDA sweep, so the median is affected): their
    evaluation time is simulation + warm-up. `old_format`: backends written as before
    Step 25 (no warm-up columns)."""
    rdir = Path(root) / name
    (rdir / "configs").mkdir(parents=True)
    gpus = [{"name": gpu}]
    (rdir / "system.json").write_text(json.dumps({"hostname": machine, "cpu_model": "i7",
                                                  "gpus": gpus}))
    summary, gens = [], []
    for i, (row_name, cfg) in enumerate(points, start=1):
        dim = row_name.split("_")[1].split(" ")[0]
        be = row_name.split(" ")[0].split("_")[2]
        x = _get(cfg, DIMS[dim])
        cfg = copy.deepcopy(cfg)
        if tamper and tamper[0] == row_name:
            tamper[1](cfg)
        cp = rdir / "configs" / f"{i:02d}.yaml"
        cp.write_text(yaml.safe_dump(cfg))
        log = f"{i:02d}.log"
        (rdir / log).write_text("compile_cache=/c/byte-w512-v1\n")
        a, k = LAWS[be]
        ev = a * x ** k
        backend = {"scalar": "scalar", "cpu": "tensor cpu", "cuda": "tensor cuda"}[be]
        if row_name in backend_lie:
            backend = "tensor cuda" if be != "cuda" else "scalar"
        summary.append({"order": i, "config": row_name, "status": "ok", "log": log,
                        "config_path": str(cp), "backend": backend, "cpu_workers": "8",
                        "vram_peak_mib": "", "disturbance": "before: GPU load 90%" if row_name in busy else "",
                        "runs_queued": "yes" if be == "cuda" and x > 1600 else "no"})
        for g in range(4):
            jitter = 1 + 0.01 * (g - 2) if g else 5      # gen 0 slow (compile), never used
            row = {"order": i, "config": row_name, "generation": g, "evaluate_s": "",
                   "ea_us_per_run_tick": "", "genomes_s": "", "select_save_s": "",
                   "ea_run_ticks": "", "warmup_s": "", "new_graphs": "", "simulate_s": "",
                   "width": str(min(112_000, 100 * x if dim != "neurons" else 10_000))}
            if row_name not in drop_timing:
                ticks = 10_000
                sim = ev * jitter
                warm = 0.0 if be == "scalar" else (3.0 if g in (1, 2) else 0.01)
                row.update(evaluate_s=f"{sim + warm:.9f}", genomes_s="0.1", select_save_s="0.05",
                           ea_us_per_run_tick=f"{1e6 * (sim + warm) / ticks:.9f}",
                           ea_run_ticks=str(ticks))
                if be not in old_format:
                    row.update(warmup_s=f"{warm:.9f}", new_graphs="1" if g in (1, 2) and be != "scalar" else "0",
                               simulate_s=f"{sim:.9f}")
            gens.append(row)
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
            rc = PS.main(args)
        return rc, buf.getvalue(), None
    except (PS.ScalingError, SystemExit) as e:
        return None, buf.getvalue(), e


def test_plot(tmp):
    print("\n[5] analysis/plot_scaling.py on synthetic reports")
    pts = all_points()
    flat = [p for v in pts.values() for p in v]
    root = Path(tmp) / "plots"
    good = synth_report(root, "good", flat)
    out = root / "out"
    rc, text, err = run_plot([str(good), "--out", str(out)])
    if not check(f"runs on a full set of 49 points (rc {rc}, {err})", rc == 0):
        print(text[-2000:])
        return
    with open(out / "fits.csv", newline="") as f:
        fits = list(csv.DictReader(f))
    ev = {(r["dimension"], r["backend"]): float(r["exponent_k"]) for r in fits if r["metric"] == "evaluate"}
    want = {(d, b): LAWS[b][1] for d in DIMS for b in BACKENDS}
    check(f"evaluation exponents recovered for all 9 lines ({ev})",
          set(ev) == set(want) and all(abs(ev[k] - want[k]) < 2e-3 for k in want))
    usk = {(r["dimension"], r["backend"]): float(r["exponent_k"]) for r in fits if r["metric"] == "us"}
    check("us/worm-tick exponents too (fixed ticks: the same)",
          all(abs(usk[k] - want[k]) < 2e-3 for k in want))
    with open(out / "points.csv", newline="") as f:
        prow = list(csv.DictReader(f))
    check(f"points.csv: 49 rows, 3 generations each (gen 0 left out)",
          len(prow) == 49 and all(r["generations"] == "3" for r in prow))
    figs = [out / f"scaling_{n}.{e}" for n in ("neurons", "population", "runs", "all") for e in ("png", "pdf")]
    check("figures written: one per dimension and the grid, png and pdf",
          all(p.is_file() and p.stat().st_size > 10_000 for p in figs))
    check("nothing reported missing", "MISSING" not in text)

    # Loud checks.
    other = synth_report(root, "other_gpu", flat[:3], gpu="NVIDIA GeForce RTX 3080")
    rc, text, err = run_plot([str(good), str(other), "--out", str(root / "o2")])
    check(f"another GPU: refused ({str(err)[:60]})", isinstance(err, PS.ScalingError)
          and "more than one kind of machine" in str(err))
    twin = synth_report(root, "twin_node", flat, machine="tg081")
    rc, text, err = run_plot([str(good), str(twin), "--out", str(root / "o2b")])
    title_ok = (root / "o2b" / "points.csv").is_file() and "tg081" in (root / "o2b" / "points.csv").read_text()
    check(f"the same hardware under another hostname: merged, both hosts named ({err})",
          rc == 0 and title_ok)
    name = flat[7][0]
    bad = synth_report(root, "tampered", flat, tamper=(name, lambda c: c["worm"].update(speed=2)))
    rc, text, err = run_plot([str(bad), "--out", str(root / "o3")])
    check(f"a setting changed in one point: refused, naming it ({str(err)[:90]})",
          isinstance(err, PS.ScalingError) and name in str(err) and "worm.speed" in str(err))
    bad = synth_report(root, "fixed_key", flat,
                       tamper=(flat[0][0], lambda c: c["experiment"].update(population_size=120)))
    rc, text, err = run_plot([str(bad), "--out", str(root / "o3b")])
    check(f"a point of the neuron sweep with P = 120: refused ({str(err)[:80]})",
          isinstance(err, PS.ScalingError) and "population_size" in str(err))
    bad = synth_report(root, "no_timing", flat, drop_timing=(name,))
    rc, text, err = run_plot([str(bad), "--out", str(root / "o4")])
    check(f"ok run without [EA timing]: refused ({str(err)[:70]})",
          isinstance(err, PS.ScalingError) and "[EA timing]" in str(err))
    bad = synth_report(root, "lie", flat, backend_lie=(name,))
    rc, text, err = run_plot([str(bad), "--out", str(root / "o5")])
    check(f"backend column contradicts the sweep file: refused ({str(err)[:70]})",
          isinstance(err, PS.ScalingError) and "backend" in str(err))
    busy = synth_report(root, "busy", flat, busy=(name,))
    rc, text, err = run_plot([str(busy), "--out", str(root / "o6")])
    check(f"busy machine: excluded and listed, then reported missing",
          rc == 0 and f"EXCLUDED {name}: machine not free" in text and "MISSING" in text)
    rc, text, err = run_plot([str(busy), "--out", str(root / "o7"), "--include-disturbed"])
    with open(root / "o7" / "points.csv", newline="") as f:
        marked = [r for r in csv.DictReader(f) if r["disturbed"] == "True"]
    check(f"--include-disturbed: plotted and marked ({len(marked)} marked)",
          rc == 0 and len(marked) == 1)
    part = synth_report(root, "partial", [p for p in flat if "_cuda " not in p[0]])
    rc, text, err = run_plot([str(part), "--out", str(root / "o8")])
    check("CUDA points missing: listed per dimension, the rest plotted",
          rc == 0 and text.count("MISSING") == 3 and "tensor CUDA" in text)
    # Step 26: capacity points extend the CUDA lines; queued points are marked.
    capflat = [p for d in ("population", "runs")
               for p in RS.expand_sweep(SWEEPS / f"scaling_{d}_cuda_capacity.yaml")]
    capr = synth_report(root, "capacity", capflat)
    rc, text, err = run_plot([str(good), str(capr), "--out", str(root / "o13")])
    with open(root / "o13" / "points.csv", newline="") as f:
        crow = [r for r in csv.DictReader(f) if r["backend"] == "cuda" and r["dimension"] == "population"]
    with open(root / "o13" / "fits.csv", newline="") as f:
        cfit = [r for r in csv.DictReader(f) if r["backend"] == "cuda" and r["dimension"] == "population"
                and r["metric"] == "evaluate"][0]
    check(f"capacity points join the base line: 11 points, x up to 25600 ({len(crow)}, {cfit['x_max']})",
          rc == 0 and len(crow) == 11 and cfit["x_max"] == "25600" and cfit["points"] == "11"
          and abs(float(cfit["exponent_k"]) - LAWS["cuda"][1]) < 2e-3)
    check(f"queued points marked, with the width cap ({[(r['x'], r['runs_queued'], r['width_min']) for r in crow][-4:]})",
          all(r["runs_queued"] == ("True" if int(r["x"]) > 1600 else "False") for r in crow)
          and all(r["width_min"] == "112000" for r in crow if int(r["x"]) > 1600))
    check("nothing missing when the whole capacity run is there", "MISSING" not in text)
    part = synth_report(root, "capacity_part", capflat[:3])
    rc, text, err = run_plot([str(good), str(part), "--out", str(root / "o14")])
    check(f"a partial capacity run: its missing points listed, a series without any capacity "
          f"point not ({[l.strip() for l in text.splitlines() if 'MISSING' in l]})",
          rc == 0 and "MISSING population / tensor CUDA: x = [25600]" in text
          and "MISSING runs" not in text)

    # Step 25: compile time is subtracted; old reports need --include-compile <backend>.
    with open(out / "points.csv", newline="") as f:
        prow = {(r["dimension"], r["backend"], r["x"]): r for r in csv.DictReader(f)}
    some = prow[("runs", "cuda", "100")]
    check(f"points.csv: simulation without the 3 s compile, the evaluation with it, the warm-up "
          f"({some['evaluate_s_median']}, {some['evaluate_with_compile_s_median']}, {some['warmup_s_total']})",
          abs(float(some["evaluate_s_median"]) - 0.5 * 100 ** 0.3) < 0.02
          and float(some["warmup_s_total"]) > 6 and some["new_graphs_total"] == "2"
          and some["compile_included"] == "False")
    old = synth_report(root, "old_format", flat, old_format=("cpu", "cuda"))
    rc, text, err = run_plot([str(old), "--out", str(root / "o10")])
    check(f"a report from before Step 25: tensor points refused ({str(err)[:80]})",
          isinstance(err, PS.ScalingError) and "--include-compile" in str(err))
    rc, text, err = run_plot([str(old), "--out", str(root / "o11"), "--include-compile", "cpu"])
    check(f"--include-compile cpu alone: the CUDA points still refused ({str(err)[:60]})",
          isinstance(err, PS.ScalingError) and "--include-compile cuda" in str(err))
    rc, text, err = run_plot([str(old), "--out", str(root / "o12"), "--include-compile", "cpu",
                              "--include-compile", "cuda"])
    with open(root / "o12" / "points.csv", newline="") as f:
        orow = list(csv.DictReader(f))
    check("both named: plotted, tensor points marked 'compile included', scalar not",
          rc == 0 and all(r["compile_included"] == ("False" if r["backend"] == "scalar" else "True")
                          for r in orow))
    with open(root / "o12" / "fits.csv", newline="") as f:
        okk = {(r["dimension"], r["backend"]): float(r["exponent_k"]) for r in csv.DictReader(f)
               if r["metric"] == "evaluate"}
    check(f"... and with the compile included the CUDA exponents are off ({okk[('runs', 'cuda')]:.2f} "
          f"vs {LAWS['cuda'][1]})", abs(okk[("runs", "cuda")] - LAWS["cuda"][1]) > 0.05)
    rc, text, err = run_plot([str(good), str(good), "--out", str(root / "o9")])
    with open(root / "o9" / "points.csv", newline="") as f:
        rep = list(csv.DictReader(f))
    check("the same report twice: repeats pooled (6 generations per point)",
          rc == 0 and len(rep) == 49 and all(r["generations"] == "6" for r in rep))


# ============================================================

def main():
    test_sweeps()
    with tempfile.TemporaryDirectory(prefix="test_scaling_") as tmp:
        test_parsing(tmp)
        test_machine_load(tmp)
        test_plot(tmp)
        test_timing_line(tmp)
    n_fail = sum(1 for _, ok in _RESULTS if not ok)
    if n_fail:
        print(f"\nFAILED -- {n_fail} of {len(_RESULTS)} checks failed")
        return 1
    print(f"\nOK -- {len(_RESULTS)}/{len(_RESULTS)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
