#!/usr/bin/env python3
"""Replay stored elite runs of a live tensor EA file and compare with the stored lifespans.

Used for Step 9, G5 (Docker): a file made in the container is replayed natively and vice
versa, and a source under an extra mount is replayed in the container. Replay goes through
`simulate.run_batch` exactly as a user would. run_batch writes its output next to the
source under a fixed name (no timestamp), so the path is read from its own
"[batch] writing to" line. The output is deleted afterwards only if this run created it;
a file of that name that already existed has been overwritten by run_batch, and is kept.

Usage
-----
    python -m tests.replay_check --file data/first_ea/<run>.h5 --device cuda --compile true --rounds 10 --last 10
    python -m tests.replay_check --file /inputs/<run>.h5 --device cuda --compile true --rounds 10 --last 3
"""

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_replay import load, run_module  # noqa: E402


def run_replay(src, device, ids, rounds, compile_, tmp):
    """run_batch replay of elites `ids`, all runs; returns ({elite: lifespans}, output, rc, path)."""
    cfg = load("genome_from_file")
    x = cfg["experiment"]
    x.update(viz_enabled=False, viz_brain_enabled=False, enable_per_run_tracking=True,
             enable_per_tick_tracking=False, enable_heat_map_tracking=False)
    x["from_file_source"].update(file_folder=str(Path(src).parent), filename=Path(src).stem,
                                 genome_ID=list(ids), runs_to_load="all")
    x["evaluator"] = {"backend": "tensor", "device": device, "dtype": "float32",
                      "philox_rounds": rounds, "compile": compile_}
    folder = Path(src).parent
    before = {p.name for p in folder.glob("*.h5")}
    rc, out = run_module("simulate.run_batch", cfg, tmp, "replay_check")
    m = re.search(r"\[batch\] writing to (\S+\.h5)", out)
    if rc != 0 or m is None:
        return {}, out, rc, None
    path = Path(m.group(1))
    res = {}
    with h5py.File(path, "r") as f:
        for i, e in enumerate(ids):
            res[e] = f[f"variant_{i}/summary"]["lifetime_ticks"][:].astype(np.int64)
    if path.name in before:
        print(f"note: {path} already existed and was overwritten by run_batch; kept")
    else:
        os.remove(path)
    return res, out, rc, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--compile", required=True, choices=["true", "false"])
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--last", type=int, required=True,
                    help="replay the last N stored elites, all of their runs")
    a = ap.parse_args()
    with h5py.File(a.file, "r") as f:
        stored = f["elite_genomes/lifespans"][:].astype(np.int64)
    n_el, R = stored.shape
    ids = list(range(max(0, n_el - a.last), n_el))
    print(f"{a.file}: {n_el} stored elites x {R} runs; replaying elites {ids[0]}..{ids[-1]} "
          f"on {a.device}, compile={a.compile}, philox_rounds={a.rounds}")
    with tempfile.TemporaryDirectory() as tmp:
        got, out, rc, _ = run_replay(a.file, a.device, ids, a.rounds, a.compile == "true", tmp)
    if not got:
        print(out[-3000:])
        print(f"FAILED: run_batch exited with {rc}"
              + ("" if rc else " but printed no '[batch] writing to <file>.h5' line"))
        return 1
    bad = sum(int((got[e] != stored[e]).sum()) for e in ids)
    print(f"{len(ids) * R} runs replayed: {bad} lifespans differ from the stored ones")
    print("OK -- identical" if bad == 0 else "FAILED -- not identical")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
