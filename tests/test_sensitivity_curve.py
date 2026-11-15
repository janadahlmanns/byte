#!/usr/bin/env python3
"""Tests for tests/sensitivity_curve.py (plan Step 12d), on this machine's CPU.

  [1] a small run (12 genomes x 8 runs; only the arms the curve needs): the identity
      arm equals the run's T bit for bit, every perturbation arm is written with its
      change and only its change, --analyse reproduces summary.json byte for byte, and
      the run's own files are untouched
  [2] refusals: another commit's simulation code, a run without its scalar arms
  [3] continuing: a missing arm is rerun, the others reused

Usage
-----
    python -m tests.test_sensitivity_curve
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tests.behaviour_equivalence as BE  # noqa: E402
import tests.sensitivity_curve as SC  # noqa: E402
from tests.be_common import EquivalenceError, read_arm  # noqa: E402
from tests.test_behaviour_equivalence import (ORIG, SMALL, make_worktrees,  # noqa: E402
                                              raises, remove_worktrees)

ROOT = BE.ROOT
_RESULTS = []
NEEDED = ("T", "S1", "S2", "S3", "S1d", "S2d", "Td")


def check(name, cond):
    _RESULTS.append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def digest(folder):
    h = hashlib.sha256()
    for p in sorted(Path(folder).rglob("*")):
        if p.is_file() and "sensitivity" not in p.parts:
            h.update(str(p.relative_to(folder)).encode() + p.read_bytes())
    return h.hexdigest()


def small_run(base, wts):
    out = base / "run"
    BE.plan_run("a", SMALL, out, wts["orig"], ORIG, wts["cur"], device_override="cpu")
    for n in NEEDED:
        subprocess.run([sys.executable, "-m", "tests.behaviour_equivalence", "--run-arm", n,
                        "--dir", str(out)], cwd=str(ROOT), check=True, capture_output=True)
    return out


def test_curve(out):
    print("\n[1] Sensitivity curve on a small run")
    before = digest(out)
    rc = SC.main([str(out)])
    sens = out / "sensitivity"
    S = json.loads((sens / "summary.json").read_text())
    check(f"exit code 0 (identity check holds), got {rc}", rc == 0)
    check("identity arm equals the run's T bit for bit", S["checks"]["identity_equals_T"])
    check(f"all {len(SC.PERTURBATIONS)} perturbation arms written",
          all((sens / "arms" / f"{n}.npz").exists() for n in SC.PERTURBATIONS))
    run = json.loads((sens / "run.json").read_text())
    base_cfg = BE.load_cfg(run["config"])
    ok = True
    for n, (key, fn, *_r) in SC.PERTURBATIONS.items():
        cfg = BE.arm_cfg(base_cfg, run["spec"], run["arms"][n])
        sect, field = key.split(".")
        want = fn(base_cfg[sect][field])
        others_same = all(cfg[s] == base_cfg[s] for s in base_cfg if s not in ("brain", "experiment"))
        ok &= cfg[sect][field] == want and others_same
    check("each arm changes exactly its own key (noise_level or threshold)", ok)
    check("the perturbed arms really differ from T (the change reached the simulation)",
          all(not BE.results_equal(read_arm(out / "arms" / "T.npz"), read_arm(sens / "arms" / f"{n}.npz"))
              for n in ("N5", "TH0.55")))
    first = (sens / "summary.json").read_bytes()
    SC.main(["--analyse", str(out)])
    check("--analyse reproduces summary.json byte for byte", (sens / "summary.json").read_bytes() == first)
    check("figure written", (sens / "sensitivity.png").exists())
    check("the run's own files are untouched", digest(out) == before)


def test_refusals(out, base):
    print("\n[2] Refusals")
    e = raises(lambda: SC.check_sim_code(ORIG), EquivalenceError)
    check(f"another commit's simulation code -> refused ({(e or '').splitlines()[0][:70]})",
          e and "differs from the run's commit" in e)
    d = base / "no_scalar"
    shutil.copytree(out, d, ignore=shutil.ignore_patterns("sensitivity"))
    (d / "arms" / "S2.npz").unlink()
    e = raises(lambda: SC.plan(d, d / "sensitivity"), EquivalenceError)
    check("a run without its arm S2 -> refused", e and "S2" in e)


def test_continue(out):
    print("\n[3] Continuing")
    sens = out / "sensitivity"
    before = {p.name: p.stat().st_mtime_ns for p in (sens / "arms").glob("*.npz")}
    (sens / "arms" / "N3.npz").unlink()
    SC.main([str(out)])
    after = {p.name: p.stat().st_mtime_ns for p in (sens / "arms").glob("*.npz")}
    check("a missing arm (N3) is rerun, every other arm reused untouched",
          "N3.npz" in after and all(after[n] == t for n, t in before.items() if n != "N3.npz"))


def main():
    base = Path(tempfile.mkdtemp(prefix="sens_test_"))
    wts = {}
    try:
        wts = make_worktrees(base)
        out = small_run(base, wts)
        test_curve(out)
        test_refusals(out, base)
        test_continue(out)
    finally:
        remove_worktrees(wts)
        shutil.rmtree(base, ignore_errors=True)
    failed = [n for n, ok in _RESULTS if not ok]
    print(f"\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    for n in failed:
        print(f"  FAILED: {n}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
