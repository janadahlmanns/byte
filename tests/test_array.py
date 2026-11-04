#!/usr/bin/env python3
"""One job per GPU: Slurm job arrays over a preset (plan Step 24).

  [1] 24a  --count; --task i runs exactly entry i with its reference comparison, and
           system.json records it; every misuse refused before anything runs; the
           warning when an array leaves entries out
  [2] 24b  names: report folders with the job / array task id, two array tasks in the
           same second get two folders; the HDF5 gets _j<job id> in a job; an
           existing HDF5 is never overwritten (FileExistsError, bytes unchanged)
  [3] 24c  hpc/tinygpu_job.sh with stub apptainer/nvidia-smi: an array task passes
           --task $SLURM_ARRAY_TASK_ID, a plain job the preset, an array with
           --configs is refused

Usage
-----
    python -m tests.test_array
"""

import contextlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import simulate.run_ea as EA  # noqa: E402
import simulate.run_selected as RS  # noqa: E402
from tests.test_ea_tensor import run_ea  # noqa: E402
from tests.test_run_selected import base_cfg, read_summary, write_cfg  # noqa: E402

_RESULTS = []
SLURM = ["SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_ARRAY_TASK_MIN",
         "SLURM_ARRAY_TASK_MAX"]


def check(name, condition):
    _RESULTS.append((name, bool(condition)))
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}")
    return bool(condition)


@contextlib.contextmanager
def slurm_env(**kw):
    """Exactly these SLURM_* variables (the others removed), restored afterwards."""
    saved = {k: os.environ.get(k) for k in SLURM}
    try:
        for k in SLURM:
            os.environ.pop(k, None)
        os.environ.update({k: str(v) for k, v in kw.items()})
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def main_quiet(args):
    """RS.main -> (exit code or SystemExit message, stdout)."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rc = RS.main(args)
    except SystemExit as e:
        rc = str(e.code)
    except Exception as e:  # noqa: BLE001 -- a crash instead of a clean refusal: a failure
        rc = f"crashed: {type(e).__name__}: {e}"
    return rc, buf.getvalue()


# ============================================================

def test_task(tmp):
    print("\n[1] 24a: --count and --task")
    cfgs = [write_cfg(base_cfg(), tmp, f"t{i}") for i in range(3)]
    pdir = Path(tmp) / "presets"
    pdir.mkdir()
    ref = Path(tmp) / "ref.h5"
    (pdir / "three.yaml").write_text(yaml.safe_dump(
        {"description": "three", "configs": [str(c) for c in cfgs], "references": {str(cfgs[2]): str(ref)}}))
    old = RS.PRESET_DIR
    RS.PRESET_DIR = pdir
    try:
        rc, out = main_quiet(["--preset", "three", "--count"])
        check(f"--count prints the number of entries ({out.strip()!r})", rc == 0 and out.strip() == "3")
        for task in (0, 2):
            root = Path(tmp) / f"rep_task{task}"
            with slurm_env():
                rc, out = main_quiet(["--preset", "three", "--task", str(task), "--report-dir", str(root)])
            rdir, rows = read_summary(root)
            check(f"--task {task}: exit 0, exactly entry {task} ran ({[r['config'] for r in rows]})",
                  rc == 0 and [r["config"] for r in rows] == [str(cfgs[task])] and rows[0]["status"] == "ok")
            s = json.loads((rdir / "system.json").read_text())["run"]
            check(f"--task {task}: system.json records it ({s['selection']}, task {s.get('task')} of {s.get('of')})",
                  s["selection"] == f"preset three, task {task} of 3" and s["task"] == task and s["of"] == 3
                  and s["selected"] == [str(cfgs[task])])
            if task == 2:
                check(f"the entry's reference is compared ({rows[0]['matches_reference'][:40]})",
                      rows[0]["matches_reference"].startswith("unavailable: reference not found"))
                import shutil
                shutil.copy(rdir / rows[0]["hdf5_copy"], ref)
                root2 = Path(tmp) / "rep_task2b"
                with slurm_env():
                    main_quiet(["--preset", "three", "--task", "2", "--report-dir", str(root2)])
                _, rows2 = read_summary(root2)
                check(f"... and with the reference in place: {rows2[0]['matches_reference']!r}",
                      rows2[0]["matches_reference"].startswith("identical ("))

        root = Path(tmp) / "rep_bad"
        for args, want, label in (
                (["--preset", "three", "--task", "3"], "--array=0-2", "task = N"),
                (["--preset", "three", "--task", "-1"], "--array=0-2", "negative task"),
                (["--task", "0"], "need --preset", "--task without --preset"),
                (["--configs", str(cfgs[0]), "--task", "0"], "need --preset", "--task with --configs"),
                (["--preset", "three", "--task", "0", "--count"], "not both", "--task with --count")):
            rc, out = main_quiet(args + ["--report-dir", str(root)])
            check(f"{label}: refused ({str(rc)[:70]}), nothing ran",
                  want in str(rc) and not str(rc).startswith("crashed") and not root.exists())

        p = RS.load_preset(pdir / "three.yaml")
        for lo, hi, warn in ((0, 1, True), (0, 2, False), (1, 2, True)):
            buf = io.StringIO()
            with slurm_env(SLURM_ARRAY_TASK_ID=lo, SLURM_ARRAY_TASK_MIN=lo, SLURM_ARRAY_TASK_MAX=hi), \
                    contextlib.redirect_stdout(buf):
                RS.preset_task(p, lo)
            left = [i for i in range(3) if not lo <= i <= hi]
            check(f"array {lo}-{hi} over 3 entries: {'warns about ' + str(left) if warn else 'no warning'} "
                  f"({buf.getvalue().strip()[:60]!r})",
                  ("WARNING" in buf.getvalue() and str(left) in buf.getvalue()) if warn
                  else buf.getvalue() == "")
        buf = io.StringIO()
        with slurm_env(SLURM_ARRAY_TASK_MIN=0, SLURM_ARRAY_TASK_MAX=0), contextlib.redirect_stdout(buf):
            RS.preset_task(p, 0)
        check("not an array task: no warning", buf.getvalue() == "")
    finally:
        RS.PRESET_DIR = old


# ============================================================

PROBE = {"python": "3", "torch": "t", "torch_cuda": "", "cuda_available": False,
         "mps_available": False, "gpus": []}


def test_names(tmp):
    print("\n[2] 24b: names")
    for env, want in (({"SLURM_JOB_ID": 501, "SLURM_ARRAY_JOB_ID": 500, "SLURM_ARRAY_TASK_ID": 3}, "_job500_3"),
                      ({"SLURM_JOB_ID": 777}, "_job777"), ({}, "")):
        with slurm_env(**env):
            check(f"report-folder suffix {env or 'outside Slurm'}: {RS.slurm_suffix()!r}",
                  RS.slurm_suffix() == want)
    root = Path(tmp) / "same_second"
    import datetime as dt
    real = RS.datetime.datetime

    class Frozen(real):
        @classmethod
        def now(cls, tz=None):
            return real(2026, 11, 4, 12, 0, 0)
    RS.datetime.datetime = Frozen
    try:
        made = []
        try:
            for task in (0, 1):
                with slurm_env(SLURM_JOB_ID=600 + task, SLURM_ARRAY_JOB_ID=600, SLURM_ARRAY_TASK_ID=task), \
                        contextlib.redirect_stdout(io.StringIO()):
                    made.append(RS.run_queue([], root, PROBE)[1])
        except FileExistsError:
            pass                                   # the second task collided: checked below
        check(f"two array tasks in the same second: two folders ({[m.name for m in made]})",
              len({m.name for m in made}) == 2 and all(m.is_dir() for m in made))
        try:
            with slurm_env(), contextlib.redirect_stdout(io.StringIO()):
                RS.run_queue([], root, PROBE)
                RS.run_queue([], root, PROBE)
            collided = False
        except FileExistsError:
            collided = True
        check("outside Slurm, two queues in the same second still collide loudly (as before)", collided)
    finally:
        RS.datetime.datetime = real

    cfg = base_cfg()
    with slurm_env(SLURM_JOB_ID=4242):
        h5, out, rc = run_ea(cfg, tmp, "h5_job")
    check(f"in a Slurm job the HDF5 name ends in _j4242 ({Path(h5).name if h5 else out[-300:]})",
          rc == 0 and h5 and Path(h5).name.endswith("_j4242.h5"))
    with slurm_env():
        h5, out, rc = run_ea(cfg, tmp, "h5_plain")
    check(f"outside Slurm the name is unchanged ({Path(h5).name if h5 else '-'})",
          rc == 0 and h5 and not re.search(r"_j\d+\.h5$", Path(h5).name))

    folder = Path(tmp) / "never_overwrite"
    real_ea = EA.datetime

    class FrozenEA(real_ea):
        @classmethod
        def now(cls, tz=None):
            return real_ea(2026, 11, 4, 12, 0, 0)
    EA.datetime = FrozenEA
    try:
        with slurm_env(), contextlib.redirect_stdout(io.StringIO()):
            first = Path(EA.initialize_hdf5_file(str(folder), "same", cfg, 2))
            before = first.read_bytes()
            try:
                EA.initialize_hdf5_file(str(folder), "same", cfg, 2)
                raised = None
            except FileExistsError as e:
                raised = e
    finally:
        EA.datetime = real_ea
    check(f"the same name in the same second: FileExistsError ({type(raised).__name__ if raised else 'none'})",
          isinstance(raised, FileExistsError))
    check("... and the existing file is untouched (same bytes)", first.read_bytes() == before)


# ============================================================

def test_job_script(tmp):
    print("\n[3] 24c: hpc/tinygpu_job.sh")
    stub = Path(tmp) / "bin"
    stub.mkdir()
    log = Path(tmp) / "apptainer_args"
    (stub / "apptainer").write_text(f'#!/bin/bash\nprintf "%s\\n" "$@" > {log}\n')
    (stub / "nvidia-smi").write_text("#!/bin/bash\necho 'NVIDIA A100, 615.71.10, 40960 MiB'\n")
    (stub / "nproc").write_text("#!/bin/bash\necho 32\n")
    for f in stub.iterdir():
        f.chmod(0o755)
    work = Path(tmp) / "work"
    work.mkdir()
    (work / "byte.sif").write_text("")

    def run(args, **env):
        log.unlink(missing_ok=True)
        e = {"PATH": f"{stub}{os.pathsep}{os.environ['PATH']}", "HOME": os.environ.get("HOME", "/tmp"),
             "WORK": str(work), "SLURM_SUBMIT_DIR": str(ROOT), "SLURM_JOB_ID": "9",
             "SLURM_JOB_PARTITION": "a100"}
        e.update({k: str(v) for k, v in env.items()})
        p = subprocess.run(["bash", str(ROOT / "hpc" / "tinygpu_job.sh")] + args, env=e,
                           capture_output=True, text=True, timeout=60)
        got = log.read_text().splitlines() if log.exists() else None
        return p.returncode, got, p.stdout + p.stderr

    rc, got, out = run(["scaling_cuda"], SLURM_ARRAY_JOB_ID=8, SLURM_ARRAY_TASK_ID=5)
    check(f"array task 5: --preset scaling_cuda --task 5 ({got and got[-5:]})",
          rc == 0 and got[-5:] == ["simulate.run_selected", "--preset", "scaling_cuda", "--task", "5"]
          and "(array 8, task 5)" in out)
    rc, got, out = run(["scaling_cuda"])
    check(f"plain job: --preset scaling_cuda ({got and got[-3:]})",
          rc == 0 and got[-3:] == ["simulate.run_selected", "--preset", "scaling_cuda"])
    rc, got, out = run(["--configs", "a [n=1],b"])
    check(f"plain job with --configs: passed through, names with spaces intact ({got and got[-2:]})",
          rc == 0 and got[-2:] == ["--configs", "a [n=1],b"])
    rc, got, out = run(["--configs", "a"], SLURM_ARRAY_JOB_ID=8, SLURM_ARRAY_TASK_ID=0)
    check(f"an array with --configs: refused, nothing started ({out.strip()[-50:]!r})",
          rc != 0 and got is None and "needs a preset" in out)
    rc, got, out = run([])
    check("no argument: refused", rc != 0 and got is None)


# ============================================================

def main():
    with tempfile.TemporaryDirectory(prefix="test_array_") as tmp:
        test_names(tmp)
        test_job_script(tmp)
        test_task(tmp)
    n_fail = sum(1 for _, ok in _RESULTS if not ok)
    if n_fail:
        print(f"\nFAILED -- {n_fail} of {len(_RESULTS)} checks failed")
        return 1
    print(f"\nOK -- {len(_RESULTS)}/{len(_RESULTS)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
