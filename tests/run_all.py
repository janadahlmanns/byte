#!/usr/bin/env python3
"""Run every test suite of the tensor port and print one summary line each.

Device-parametrised suites run on the CPU plus every GPU present (CUDA and/or MPS; see
tests/devices.py), so on an NVIDIA machine this is the CUDA validation.

Usage
-----
    python -m tests.run_all            # everything (~15-25 minutes)
    python -m tests.run_all --fast     # skips the slowest scalar-regression parts
"""

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUITES = [
    ("tests.test_genome_codec", False),
    ("tests.test_brain_tick", False),
    ("tests.test_decision", False),
    ("tests.test_world_worm", False),
    ("tests.test_world_clock", False),
    ("tests.test_philox", False),
    ("tests.test_generation", True),
    ("tests.test_scheduling", False),
    ("tests.test_compile", False),
    ("tests.test_replay", False),
    ("tests.test_io_tensor", True),
    ("tests.test_ea_tensor", True),
    ("tests.test_predrawn", False),
]


def main():
    fast = "--fast" in sys.argv
    failed = []
    t_all = time.perf_counter()
    for module, has_fast in SUITES:
        args = [sys.executable, "-m", module] + (["--fast"] if fast and has_fast else [])
        t0 = time.perf_counter()
        # UTF-8 for the child's piped output: on Windows a pipe otherwise uses the
        # legacy code page, which cannot encode every character a test might print.
        env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        p = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, env=env,
                           encoding="utf-8", errors="replace")
        lines = p.stdout.strip().splitlines()
        summary = next((l.strip() for l in reversed(lines)
                        if "checks passed" in l or l.startswith("FAILED")
                        or "passed" in l), None)
        ok = p.returncode == 0
        if summary is None:
            summary = "CRASH: " + ((p.stderr.strip().splitlines() or ["?"])[-1][:150])
            ok = False
        print(f"{'OK  ' if ok else 'FAIL'} {module:28s} {summary}  ({time.perf_counter() - t0:.0f} s)",
              flush=True)
        if not ok:
            failed.append(module)
            for l in lines:
                if "[FAIL]" in l:
                    print(f"       {l.strip()[:160]}")
            if p.returncode and "Traceback" in p.stderr:
                print("       " + "\n       ".join(p.stderr.strip().splitlines()[-6:]))
    print(f"\n{len(SUITES) - len(failed)}/{len(SUITES)} suites passed in "
          f"{time.perf_counter() - t_all:.0f} s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
