#!/usr/bin/env python3
"""Tests for the CPU baseline (plan Step 30): sweeps, hpc/tinyfat_job.sh, the plot.

  [1] sweeps: per size the scalar / cpu / cuda files are identical apart from the
      evaluator; values as planned; the presets list the planned configs
  [2] tinyfat_job.sh with stub commands: runs with physical cores; stops when nproc
      differs from the request, and when allowed CPUs are hyperthreads of one core;
      several presets in one job, preset@N on the first N cores with N threads; bad
      segments refused before anything runs; a failing preset does not stop the rest
  [3] the tensor-CPU configs (n = 11 and n = 50) run on this machine (tiny, compile off)
  [4] plot on synthetic reports of four machines: speed-ups, GPU lead and core scaling
      exact; files written
  [5] refusals: a duplicate machine / config, a missing counterpart on the reference
      machine, a tampered config, no or two CUDA machines

Usage
-----
    python -m tests.test_cpu_baseline
"""

import copy
import csv
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analysis.plot_cpu_baseline as PB  # noqa: E402
import simulate.run_selected as RS  # noqa: E402
from tests.test_scaling import run_ea_in  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "hpc" / "tinyfat_job.sh"
_RESULTS = []


def check(name, cond):
    _RESULTS.append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def test_sweeps():
    print("\n[1] Sweeps and presets")
    exp = PB.expected_configs()
    for size, (P, R, gens, elite, n) in {"n11": (260, 300, 5, 10, 11), "n50": (64, 100, 4, 5, 50)}.items():
        cfgs = {be: exp[f"cpu_baseline_{size}_{be} [population_size={P}]"] for be in ("scalar", "cpu", "cuda")}
        strip = []
        for be, c in cfgs.items():
            c = copy.deepcopy(c)
            PB.PS._drop(c, "experiment.evaluator")
            PB.PS._drop(c, "experiment.simulation_name")
            strip.append(c)
        check(f"{size}: scalar / cpu / cuda identical apart from the evaluator", strip[0] == strip[1] == strip[2])
        e = cfgs["scalar"]["experiment"]
        ea = e["evolutionary_algorithm"]
        check(f"{size}: P {P}, R {R}, {gens} generations, elite {elite}, n {n}",
              (e["population_size"], e["n_runs"], ea["num_generations"], ea["elite_size"],
               cfgs["scalar"]["brain"]["n_neurons"]) == (P, R, gens, elite, n))
        check(f"{size}: evaluators scalar / tensor cpu compiled / tensor cuda",
              cfgs["scalar"]["experiment"]["evaluator"] == {"backend": "scalar"}
              and cfgs["cpu"]["experiment"]["evaluator"]["device"] == "cpu"
              and cfgs["cpu"]["experiment"]["evaluator"]["compile"] is True
              and cfgs["cuda"]["experiment"]["evaluator"]["device"] == "cuda")
    pre = RS.load_presets()
    check("presets: tinyfat 4 (no CUDA), cores 1 (n11 scalar), unipc 6",
          len(pre["cpu_baseline_tinyfat"].configs) == 4
          and not any("cuda" in c for c in pre["cpu_baseline_tinyfat"].configs)
          and pre["cpu_baseline_cores"].configs == ["cpu_baseline_n11_scalar [population_size=260]"]
          and len(pre["cpu_baseline_unipc"].configs) == 6)


def stub(dirpath, name, body):
    p = Path(dirpath) / name
    p.write_text("#!/bin/bash\n" + body + "\n")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)


def run_script(tmp, nproc, allowed, want, tag, segments=("cpu_baseline_tinyfat",), fail_preset=None):
    """Run the job script with stubs; -> (exit code, output, [one line per apptainer call:
    'OMP=<threads> CPUS=<taskset list> <apptainer args>'] or None)."""
    d = Path(tmp) / tag
    (d / "bin").mkdir(parents=True)
    (d / "configs").symlink_to(ROOT / "configs")          # the presets, as in the clone
    # 8 CPUs: 0-3 are cores 0-3, 4-7 their hyperthreads
    lscpu_p = "# CPU,Core,Socket\n" + "\n".join(f"{c},{c % 4},0" for c in range(8))
    stub(d / "bin", "nproc", f"echo {nproc}")
    stub(d / "bin", "taskset", f'if [ "$1" = "-cp" ]; then echo "pid $2\'s current affinity list: {allowed}"; '
                               f'else export STUB_CPUS=$2; shift 2; exec "$@"; fi')
    stub(d / "bin", "lscpu", f'if [ "${{1:-}}" = "-p=CPU,CORE,SOCKET" ]; then cat <<"EOF"\n{lscpu_p}\nEOF\n'
                             f'else echo "Model name: Test CPU"; fi')
    stub(d / "bin", "apptainer", f'echo "OMP=$OMP_NUM_THREADS CPUS=$STUB_CPUS $*" >> {d}/apptainer_calls\n'
                                 + (f'case "$*" in *"--preset {fail_preset}") exit 3;; esac' if fail_preset else ""))
    (d / "byte.sif").write_text("x")
    env = dict(os.environ, PATH=f"{d / 'bin'}:{os.environ['PATH']}", SLURM_CPUS_PER_TASK=str(want),
               SLURM_SUBMIT_DIR=str(d), SLURM_JOB_ID="1", WORK=str(d), BYTE_SIF=str(d / "byte.sif"),
               BYTE_CACHE=str(d / "cache"))
    p = subprocess.run(["bash", str(SCRIPT), *segments], env=env, capture_output=True, text=True, cwd=str(d))
    f = d / "apptainer_calls"
    return p.returncode, p.stdout + p.stderr, (f.read_text().strip().splitlines() if f.exists() else None)


def test_job_script(tmp):
    print("\n[2] hpc/tinyfat_job.sh with stub commands")
    check("bash -n", subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0)
    sif = Path(tmp) / "ok" / "byte.sif"
    rc, out, calls = run_script(tmp, 4, "0-3", 4, "ok")
    check(f"4 physical cores (0-3) -> runs the preset on all 4 ({calls})",
          rc == 0 and calls == [f"OMP=4 CPUS=0,1,2,3 exec {sif} python -m simulate.run_selected --preset cpu_baseline_tinyfat"])
    sif = Path(tmp) / "multi" / "byte.sif"
    rc, out, calls = run_script(tmp, 4, "0-3", 4, "multi",
                                segments=("cpu_baseline_tinyfat", "cpu_baseline_cores@2", "cpu_baseline_cores@1"))
    check(f"one job, three segments: all 4 cores, then the first 2, then the first 1 ({calls})",
          rc == 0 and calls == [
              f"OMP=4 CPUS=0,1,2,3 exec {sif} python -m simulate.run_selected --preset cpu_baseline_tinyfat",
              f"OMP=2 CPUS=0,1 exec {sif} python -m simulate.run_selected --preset cpu_baseline_cores",
              f"OMP=1 CPUS=0 exec {sif} python -m simulate.run_selected --preset cpu_baseline_cores"])
    for tag, segs, msg in (("toomany", ("cpu_baseline_tinyfat", "cpu_baseline_cores@5"), "5 cores, but the job has 4"),
                           ("notint", ("cpu_baseline_cores@x",), "positive integer"),
                           ("nopreset", ("cpu_baseline_tinyfat", "no_such_preset"), "no preset"),
                           ("flag", ("--configs",), "takes preset names")):
        rc, out, calls = run_script(tmp, 4, "0-3", 4, tag, segments=segs)
        check(f"{' '.join(segs)} -> refused before anything runs", rc != 0 and msg in out and calls is None)
    rc, out, calls = run_script(tmp, 4, "0-3", 4, "onefails", segments=("cpu_baseline_tinyfat", "cpu_baseline_cores@2"),
                                fail_preset="cpu_baseline_tinyfat")
    check("a failing preset: the next still runs, the job exits non-zero",
          rc != 0 and calls is not None and len(calls) == 2 and "failed: cpu_baseline_tinyfat@4" in out)
    rc, out, calls = run_script(tmp, 8, "0-7", 4, "mismatch")
    check("nproc 8 for a request of 4 -> stops, nothing run", rc != 0 and "nproc is 8" in out and calls is None)
    rc, out, calls = run_script(tmp, 4, "0,1,4,5", 4, "smt")
    check(f"CPUs 0,1,4,5 (two cores' hyperthreads) -> stops ({out.strip().splitlines()[-1][:70]})",
          rc != 0 and "only 2 physical cores" in out and calls is None)


def test_tensor_cpu_runs(tmp):
    print("\n[3] The tensor-CPU configs run here (tiny, compile off)")
    exp = PB.expected_configs()
    for size, P in (("n11", 260), ("n50", 64)):
        cfg = copy.deepcopy(exp[f"cpu_baseline_{size}_cpu [population_size={P}]"])
        x = cfg["experiment"]
        x["population_size"], x["n_runs"] = 6, 3
        x["evolutionary_algorithm"]["num_generations"] = 2
        x["evolutionary_algorithm"]["elite_size"] = 2
        x["evaluator"]["compile"] = False
        h5, log, rc = run_ea_in(cfg, tmp, f"tcpu_{size}", ROOT)
        check(f"{size}: run_ea on tensor CPU finishes (exit {rc})",
              rc == 0 and h5 is not None and "device=cpu" in log)


# ---------------------------------------------------------------------------
# synthetic reports
# ---------------------------------------------------------------------------

UNI = dict(cpu_model="12th Gen Intel(R) Core(TM) i7-12700", cores=20, gpu="NVIDIA GeForce RTX 3070")
EPYC = "AMD EPYC 7502 32-Core Processor"
TIMES = {  # (machine tag, config) -> seconds per whole generation
    ("uni", "n11_scalar"): 80.0, ("uni", "n11_cpu"): 100.0, ("uni", "n11_cuda"): 5.0,
    ("uni", "n50_scalar"): 300.0, ("uni", "n50_cpu"): 400.0, ("uni", "n50_cuda"): 1.0,
    ("tf64", "n11_scalar"): 40.0, ("tf64", "n11_cpu"): 50.0, ("tf64", "n50_scalar"): 150.0, ("tf64", "n50_cpu"): 200.0,
    ("tf32", "n11_scalar"): 70.0, ("tf16", "n11_scalar"): 130.0,
}


def synth(root, name, tag, cpu, cores, gpu, configs, tamper=None):
    exp = PB.expected_configs()
    rdir = Path(root) / name
    (rdir / "configs").mkdir(parents=True)
    (rdir / "system.json").write_text(json.dumps({"cpu_model": cpu, "cpus_allowed": {"count": cores},
                                                  "gpus": [{"name": gpu}] if gpu else "unavailable: none"}))
    summary, gens = [], []
    for i, short in enumerate(configs, start=1):
        size, be = short.split("_")
        P = 260 if size == "n11" else 64
        cname = f"cpu_baseline_{size}_{be} [population_size={P}]"
        cfg = copy.deepcopy(exp[cname])
        if tamper == short:
            cfg["experiment"]["n_runs"] = 7
        cp = rdir / "configs" / f"{i:02d}.yaml"
        cp.write_text(yaml.safe_dump(cfg))
        t = TIMES[(tag, short)]
        summary.append({"order": i, "config": cname, "status": "ok", "config_path": str(cp),
                        "backend": PB.BACKEND_COLUMN[be], "wall_s": f"{5 * t:.2f}", "disturbance": "",
                        "cpu_workers": str(cores)})
        for g in range(4):
            sim = t - 0.3
            gens.append({"order": i, "config": cname, "generation": g, "genomes_s": "0.2", "select_save_s": "0.1",
                         "evaluate_s": f"{sim + (2.0 if be != 'scalar' and g == 1 else 0):.6f}",
                         "simulate_s": f"{sim:.6f}" if be != "scalar" else ""})
    for fname, rows in (("summary.csv", summary), ("generations.csv", gens)):
        with open(rdir / fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return rdir


def four_machines(tmp, prefix="", **kw):
    return [synth(tmp, f"{prefix}uni", "uni", UNI["cpu_model"], 20, UNI["gpu"],
                  ["n11_scalar", "n11_cpu", "n11_cuda", "n50_scalar", "n50_cpu", "n50_cuda"], **kw),
            synth(tmp, f"{prefix}tf64", "tf64", EPYC, 64, None, ["n11_scalar", "n11_cpu", "n50_scalar", "n50_cpu"]),
            synth(tmp, f"{prefix}tf32", "tf32", EPYC, 32, None, ["n11_scalar"]),
            synth(tmp, f"{prefix}tf16", "tf16", EPYC, 16, None, ["n11_scalar"])]


def run_plot(args):
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            rc = PB.main(args)
        return rc, buf.getvalue(), None
    except PB.BaselineError as e:
        return None, buf.getvalue(), str(e)


def test_plot(tmp):
    print("\n[4] Plot on synthetic reports (uni PC + TinyFat 64 / 32 / 16)")
    reps = four_machines(tmp)
    out = Path(tmp) / "out"
    rc, txt, err = run_plot([*map(str, reps), "--out", str(out)])
    check(f"runs ({err})", rc == 0)
    points, ref = PB.collect(reps)
    rows = PB.table(points, ref)
    get = lambda size, cores, code: next(r for r in rows if r["size"] == size and r["cores"] == cores and r["code"] == code)
    check(f"reference = the CUDA machine ({ref[:30]}...)", "RTX 3070" in ref)
    check("TinyFat 64 scalar n11: 40 s, x2.0 vs the reference scalar",
          abs(get("n11", 64, "scalar")["median_s"] - 40.0) < 1e-9 and abs(get("n11", 64, "scalar")["vs_ref_scalar"] - 2.0) < 1e-9)
    check("GPU lead over the fastest CPU: n11 40/5 = 8, n50 150/1 = 150 (and the fastest CPU is TinyFat scalar)",
          abs(get("n11", 64, "scalar")["gpu_lead"] - 8.0) < 1e-9 and get("n11", 64, "scalar")["fastest_cpu"]
          and abs(get("n50", 64, "scalar")["gpu_lead"] - 150.0) < 1e-9)
    check("tensor warm-up excluded (tensor CPU 50 s, not 52)", abs(get("n11", 64, "tensor CPU")["median_s"] - 50.0) < 1e-9)
    check("core scaling rows: 16 / 32 / 64 cores present", {r["cores"] for r in rows if r["size"] == "n11"
                                                             and r["code"] == "scalar"} == {16, 20, 32, 64})
    check("figure and CSV written", (out / "cpu_baseline.png").exists() and (out / "cpu_baseline.csv").exists())


def test_refusals(tmp):
    print("\n[5] Refusals")
    reps = four_machines(tmp, "a_")
    dup = synth(tmp, "dup64", "tf64", EPYC, 64, None, ["n11_scalar"])
    rc, _t, err = run_plot([*map(str, reps), str(dup), "--out", str(Path(tmp) / "o1")])
    check(f"the same machine and core count twice -> refused", rc is None and err and "appears in two reports" in err)
    uni_partial = synth(tmp, "uni_partial", "uni", UNI["cpu_model"], 20, UNI["gpu"], ["n11_scalar", "n11_cpu", "n11_cuda"])
    rc, _t, err = run_plot([str(uni_partial), str(reps[1]), "--out", str(Path(tmp) / "o2")])
    check(f"TinyFat n50 without a uni PC n50 counterpart -> refused", rc is None and err and "no counterpart" in err)
    reps_t = four_machines(tmp, "t_", tamper="n11_cpu")
    rc, _t, err = run_plot([*map(str, reps_t), "--out", str(Path(tmp) / "o3")])
    check(f"a tampered config -> refused ({(err or '')[:50]})", rc is None and err and "differs from the sweep" in err)
    rc, _t, err = run_plot([str(reps[1]), str(reps[2]), "--out", str(Path(tmp) / "o4")])
    check("no machine with CUDA -> refused", rc is None and err and "exactly one machine with CUDA" in err)
    other_gpu = synth(tmp, "a100", "uni", "AMD EPYC 7662", 32, "NVIDIA A100-SXM4-40GB", ["n11_cuda"])
    rc, _t, err = run_plot([*map(str, reps), str(other_gpu), "--out", str(Path(tmp) / "o5")])
    check("two machines with CUDA -> refused", rc is None and err and "exactly one machine with CUDA" in err)


def main():
    test_sweeps()
    with tempfile.TemporaryDirectory() as tmp:
        test_job_script(tmp)
        test_tensor_cpu_runs(tmp)
        test_plot(tmp)
        test_refusals(tmp)
    failed = [n for n, ok in _RESULTS if not ok]
    print(f"\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    for n in failed:
        print(f"  FAILED: {n}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
