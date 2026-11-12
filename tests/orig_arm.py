#!/usr/bin/env python3
"""One arm of the behavioural-equivalence test on ANOTHER commit's scalar code (plan
Step 12c): the original code (8e17e694c7) or, as the positive control, a worktree of
the current commit.

Run by tests/behaviour_equivalence.py, never by hand:

    cd <worktree> && PYTHONPATH=<worktree> python /abs/byte/tests/orig_arm.py \
        --input arm_input.npz --out O.npz --folder <scratch> --commit <sha>

Python puts this file's folder (byte/tests/) first on sys.path. It holds no `mvb` or
`simulate` package, so those imports resolve to the worktree, which is checked before
anything runs: a run of the wrong code would be worse than none.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from be_common import (EquivalenceError, call_eval, check_seeds_recorded, find_h5,
                       genome_hash, loaded_files, read_summaries, unflatten_genomes,
                       write_arm)


def git(worktree, *args):
    p = subprocess.run(["git", "-C", str(worktree), *args], text=True, capture_output=True)
    if p.returncode != 0:
        raise EquivalenceError(f"git {' '.join(args)} in {worktree} failed: {p.stderr.strip()}")
    return p.stdout.strip()


def check_worktree(worktree, commit):
    worktree = Path(worktree).resolve()
    if Path.cwd().resolve() != worktree:
        raise EquivalenceError(f"must run with the worktree as working directory "
                               f"({worktree}), not {Path.cwd()}")
    head = git(worktree, "rev-parse", "HEAD")
    want = git(worktree, "rev-parse", f"{commit}^{{commit}}")
    if head != want:
        raise EquivalenceError(f"worktree {worktree} is at {head[:10]}, not {commit}")
    dirty = git(worktree, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise EquivalenceError(f"worktree {worktree} has local changes:\n{dirty}")
    import mvb
    import simulate
    for mod in (mvb, simulate):
        where = Path(mod.__file__).resolve()
        if worktree not in where.parents:
            raise EquivalenceError(f"{mod.__name__} was imported from {where}, not from the "
                                   f"worktree {worktree} (check PYTHONPATH)")
    return head


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folder", required=True, help="scratch folder for the HDF5")
    ap.add_argument("--commit", required=True)
    ap.add_argument("--arm", required=True)
    a = ap.parse_args(argv)

    head = check_worktree(Path.cwd(), a.commit)
    import mvb.simulation_API as api        # the worktree's (checked above)

    with np.load(a.input) as z:
        flat = {k: z[k] for k in z.files}
    cfg = json.loads(bytes(flat.pop("cfg_json")).decode())
    meta_in = json.loads(bytes(flat.pop("meta_json")).decode())
    seeds = {k: flat.pop(k) for k in ("run_seeds", "noise_seeds", "decision_seeds")}
    genomes = unflatten_genomes(flat, source=f"orig_arm {a.arm}")
    got = genome_hash(genomes)
    if got != meta_in["population_hash"]:
        raise EquivalenceError(f"rebuilt population hash {got[:12]} != "
                               f"{meta_in['population_hash'][:12]}")
    P, R = len(genomes), int(cfg["experiment"]["n_runs"])
    print(f"[orig_arm {a.arm}] code {head[:10]} from {Path.cwd()}; {P} genomes x {R} runs, "
          f"per_tick={meta_in['per_tick']}", flush=True)

    os.makedirs(a.folder, exist_ok=True)
    t0 = time.perf_counter()
    call_eval(api.eval_generation, genomes, cfg, a.folder, seeds=seeds,
              per_tick=meta_in["per_tick"], with_randomness_kw=False)
    wall = time.perf_counter() - t0
    h5 = find_h5(a.folder)
    res = read_summaries(h5, P, R)
    check_seeds_recorded(res, seeds, a.arm)
    meta = dict(meta_in, wall_s=wall, code_commit=head, code_dirty=False,
                code_root=str(Path.cwd()), h5=str(h5), python=sys.version.split()[0],
                numpy=np.__version__, modules=loaded_files(Path.cwd()))
    write_arm(a.out, results=res, meta=meta)
    print(f"[orig_arm {a.arm}] done in {wall:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except EquivalenceError as e:
        sys.exit(f"ERROR: {e}")
