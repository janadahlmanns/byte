#!/usr/bin/env python3
"""Run the REAL `simulate.run_ea` of one code version for a few generations and print its
per-generation timings (plan Step 13 follow-up).

tests/version_bench.py times the generation loop, but with a fixed width and no
selection, and showed no slowdown between versions, while real EA runs did slow down.
This runs the EA exactly as a user does (selection, mutation accumulating over
generations, the width rule, the real evaluator), on a temporary copy of a config with
fewer generations (and optionally a smaller population or fewer runs), inside the
checkout given by --code. The copy and the output files are deleted afterwards.

Usage
-----
    git worktree add /tmp/bv_dense 0fc6608785
    python tests/version_ea.py --code /tmp/bv_dense --config ea_from_lookup_tensor_cuda --gens 8
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

import yaml

ap = argparse.ArgumentParser()
ap.add_argument("--code", required=True, help="repository checkout whose run_ea is run")
ap.add_argument("--config", required=True)
ap.add_argument("--gens", type=int, required=True)
ap.add_argument("--population", type=int, default=None, help="default: the config's")
ap.add_argument("--runs", type=int, default=None, help="default: the config's n_runs")
a = ap.parse_args()
code = os.path.abspath(a.code)
cfg = yaml.safe_load(open(os.path.join(code, "configs/experiments", f"{a.config}.yaml")))
x = cfg["experiment"]
x["evolutionary_algorithm"]["num_generations"] = a.gens
if a.population:
    x["population_size"] = a.population
if a.runs:
    x["n_runs"] = a.runs
out_dir = tempfile.mkdtemp(prefix="version_ea_")
x["output_folder"] = out_dir + "/"
cfg_path = os.path.join(out_dir, "bench.yaml")
with open(cfg_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
commit = subprocess.run(["git", "-C", code, "log", "-1", "--format=%h"], text=True,
                        capture_output=True).stdout.strip()
print(f"version {commit}: run_ea --config {a.config}, {a.gens} generations, population "
      f"{x['population_size']}, {x['n_runs']} runs", flush=True)
try:
    p = subprocess.Popen([sys.executable, "-m", "simulate.run_ea", "--config", cfg_path],
                         cwd=code, text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    p.stdin.write("n\n")
    p.stdin.close()
    total, n = 0.0, 0
    for line in p.stdout:
        m = re.search(r"in ([\d.]+)s \((\d+) iterations, (\d+) ms/iteration.*width (\d+) ->", line)
        if m:
            print(f"  gen {n}: {float(m.group(1)):8.2f} s, {m.group(2)} iterations, "
                  f"{m.group(3)} ms/iteration, width {m.group(4)}", flush=True)
            if n:
                total += float(m.group(1))
            n += 1
        elif "Error" in line or "Traceback" in line:
            print("  " + line.rstrip(), flush=True)
    p.wait()
    print(f"  generations 1-{n - 1}: {total:.2f} s (exit code {p.returncode})")
finally:
    shutil.rmtree(out_dir, ignore_errors=True)
