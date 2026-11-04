#!/usr/bin/env python3
"""Shared cluster nodes: only the job's cores and memory count (plan Step 23).

  [1] 23a  the scalar worker count follows os.sched_getaffinity (a Slurm job, taskset),
           os.cpu_count() without it; unchanged on this machine; and the results do not
           depend on the worker count (serial, 2 and 6 workers: identical HDF5)
  [2] 23b  machine_load on a fake /proc: CPU busy over the job's cores only, a process
           pinned to other cores is not foreign, one that may share them is, unknown
           counts as foreign; a core without a /proc/stat line is explicit
  [3] 23d  cgroup memory limit, v2 and v1, limits on a parent, "no limit" forms, a
           container's "/" path; the tensor-CPU width plans with min(RAM, limit);
           run_selected reports the same limit
  [4] 23e  system.json: cpus_allowed, the Slurm allocation, the container kind

Usage
-----
    python -m tests.test_cluster
"""

import copy
import io
import os
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mvb.simulation_helper_functions as SH  # noqa: E402
import mvb_torch.generation as G  # noqa: E402
import mvb_torch.host_limits as HL  # noqa: E402
import simulate.machine_load as ML  # noqa: E402
import simulate.run_selected as RS  # noqa: E402
from tests.test_ea_tensor import diff_h5  # noqa: E402

_RESULTS = []
GiB = 2 ** 30


def check(name, condition):
    _RESULTS.append((name, bool(condition)))
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}")
    return bool(condition)


# ============================================================

def workers_with(affinity):
    """get_num_workers with os.sched_getaffinity replaced (None: removed, as on macOS)."""
    had = hasattr(os, "sched_getaffinity")
    saved = getattr(os, "sched_getaffinity", None)
    try:
        if affinity is None:
            if had:
                del os.sched_getaffinity
        else:
            os.sched_getaffinity = lambda pid: set(affinity)
        buf = io.StringIO()
        with redirect_stdout(buf):
            n = SH.get_num_workers(False, False)
        return n, buf.getvalue().strip()
    finally:
        if had:
            os.sched_getaffinity = saved
        elif hasattr(os, "sched_getaffinity"):
            del os.sched_getaffinity


WORKER_RUN = """
import sys
import mvb.simulation_API as A
n = {n!r}
A.get_num_workers = lambda *a: n
sys.argv = ["run_ea", "--config", {cfg!r}]
from simulate import run_ea
run_ea.main()
"""


def run_with_workers(cfg, tmp, n):
    out = Path(tmp) / f"w{n}"
    out.mkdir()
    cfg = copy.deepcopy(cfg)
    cfg["experiment"]["output_folder"] = str(out) + "/"
    path = Path(tmp) / f"w{n}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    p = subprocess.run([sys.executable, "-c", WORKER_RUN.format(n=n, cfg=str(path))],
                       input="n\n", text=True, capture_output=True, cwd=ROOT, timeout=1800)
    h5 = list(out.glob("*.h5"))
    return (h5[0] if len(h5) == 1 else None), p.stdout + p.stderr, p.returncode


def test_workers(tmp):
    print("\n[1] 23a: scalar worker count")
    n, line = workers_with(range(6))
    check(f"6 allowed cores (a Slurm job): 4 workers ({line})", n == 4 and "6 allowed" in line)
    n, line = workers_with(range(2))
    check(f"2 allowed cores: serial ({n}, {line})", n is None and "serially" in line)
    n, line = workers_with(None)
    check(f"no sched_getaffinity (macOS, Windows): os.cpu_count() - 2 ({n})",
          n == max(1, os.cpu_count() - 2) if os.cpu_count() > 2 else n is None)
    n, line = workers_with(range(os.cpu_count()))
    check(f"every core allowed (an ordinary machine): the same as before ({n} = {os.cpu_count()} - 2)",
          n == os.cpu_count() - 2)
    here = SH.allowed_cpu_count()
    print(f"      this machine: {here} allowed of {os.cpu_count()}")
    check("WORKERS_LINE still reads the new line",
          RS.WORKERS_LINE.search(workers_with(range(6))[1]) is not None)

    import tests.test_scaling as TS
    cfg = TS.tiny("scalar")
    cfg["experiment"]["evolutionary_algorithm"]["num_generations"] = 2
    runs = {}
    for k in (None, 2, 6):
        h5, out, rc = run_with_workers(cfg, tmp, k)
        if not check(f"workers {k or 'serial'}: run_ea exits 0", rc == 0 and h5):
            print(out[-1500:])
            return
        runs[k] = h5
    for k in (2, 6):
        problems, n = diff_h5(str(runs[None]), str(runs[k]))
        check(f"{k} workers vs serial: all {n} datasets identical ({len(problems)} differences)",
              not problems)


# ============================================================

def write_proc(root, cores, procs):
    """A fake /proc: cores = [(busy, idle)] per core; procs = {pid: (ppid, name, jiffies,
    Cpus_allowed_list or None)}."""
    root = Path(root)
    root.mkdir(exist_ok=True)
    tb, ti = sum(b for b, _i in cores), sum(i for _b, i in cores)
    lines = [f"cpu  {tb} 0 0 {ti} 0 0 0 0 0 0"] + [f"cpu{c} {b} 0 0 {i} 0 0 0 0 0 0"
                                                     for c, (b, i) in enumerate(cores)]
    (root / "stat").write_text("\n".join(lines) + "\nintr 1\n")
    for pid, (ppid, name, j, allowed) in procs.items():
        d = root / str(pid)
        d.mkdir(exist_ok=True)
        (d / "stat").write_text(f"{pid} ({name}) S {ppid} " + " ".join(["0"] * 9)
                                + f" {j} 0 " + " ".join(["0"] * 30) + "\n")
        st = d / "status"
        if allowed is None:
            st.unlink(missing_ok=True)
        else:
            st.write_text(f"Name:\t{name}\nCpus_allowed_list:\t{allowed}\n")


def test_machine_load(tmp):
    print("\n[2] 23b: only the job's cores")
    check("parse_cpu_list", ML.parse_cpu_list("0-3,8,10-11\n") == frozenset({0, 1, 2, 3, 8, 10, 11})
          and ML.parse_cpu_list("5") == frozenset({5}))
    proc = Path(tmp) / "proc"
    me = os.getpid()
    tick = os.sysconf("SC_CLK_TCK")
    write_proc(proc, [(0, 1000), (0, 1000), (0, 1000), (0, 1000)], {me: (1, "python", 0, "0-3")})
    check("cpu_totals over cores {2, 3}, and over the whole machine",
          ML.cpu_totals(proc, frozenset({2, 3})) == (0, 2000) and ML.cpu_totals(proc) == (0, 4000))
    check("a core of the job without a /proc/stat line: None (reported unavailable)",
          ML.cpu_totals(proc, frozenset({2, 9})) is None)

    # Cores 0-1 (another job's) are fully busy during the window, ours (2-3) idle.
    state = {"t": 0}

    def advance():
        state["t"] += 1
        t = state["t"]
        write_proc(proc, [(100 * t, 1000), (100 * t, 1000), (1, 1000 + 100 * t), (1, 1000 + 100 * t)],
                   {me: (1, "python", 0, "0-3"),
                    7001: (1, "lmp other job", int(2 * t * tick), "0-1"),
                    7002: (1, "daemon", int(0.05 * t * tick), "0-3"),
                    7003: (1, "unknown", 0, None)})
        return [(0, 1)]

    saved = (ML.gpu_load, ML.gpu_processes)
    ML.gpu_load, ML.gpu_processes = advance, (lambda: [])
    try:
        mine = ML.idle_check(seconds=0.05, interval=0.01, proc=proc, cpus={2, 3})
        whole = ML.idle_check(seconds=0.05, interval=0.01, proc=proc, cpus=None)
    finally:
        ML.gpu_load, ML.gpu_processes = saved
    check(f"another job busy on cores 0-1: our cores 2-3 are idle ({mine['verdict']}, "
          f"{mine['cpu_busy_pct']}% over {mine['cpus_judged']} cores)",
          mine["verdict"] == "ok" and float(mine["cpu_busy_pct"]) < 1 and mine["cpus_judged"] == "2")
    check(f"... the whole machine would be busy ({whole['verdict']})",
          whole["verdict"].startswith("busy: CPU"))
    check(f"... and the other job is not among our busiest processes ({mine['cpu_top']!r})",
          "lmp" not in mine["cpu_top"])

    # LoadSampler: the pinned job is not foreign; a process that may share our cores is.
    advance()
    with ML.LoadSampler(interval=0.05, proc=proc, cpus={2, 3}) as s:
        for _ in range(20):
            advance()
            import time
            time.sleep(0.02)
    top = s.foreign_top()
    check(f"pinned to other cores: not counted ({top!r})",
          "lmp" not in top and "lmp other job" not in s.by_name)
    check(f"a process allowed on our cores is counted, and only that ({s.by_name})",
          s.by_name.get("daemon", 0) > 0 and set(s.by_name) <= {"daemon", "unknown"}
          and abs(s.foreign_cpu_s - sum(s.by_name.values())) < 1e-9)
    check("may_share: unknown on either side counts as sharing",
          ML.may_share(None, frozenset({1})) and ML.may_share(frozenset({1}), None)
          and not ML.may_share(frozenset({0}), frozenset({1})))
    with ML.LoadSampler(interval=0.05, proc=proc, cpus=None) as s2:
        for _ in range(20):
            advance()
            time.sleep(0.02)
    check(f"unrestricted (cpus=None): the other job counts, as in Step 22 ({s2.foreign_top()!r})",
          "lmp" in s2.foreign_top())
    check(f"the default `cpus` is this process's affinity ({ML.resolve_cpus(ML.JOB_CPUS)})",
          ML.resolve_cpus(ML.JOB_CPUS) == ML.allowed_cpus())


# ============================================================

def cg(tmp, name, self_cgroup, files):
    d = Path(tmp) / name
    (d / "root").mkdir(parents=True)
    (d / "self_cgroup").write_text(self_cgroup)
    for rel, val in files.items():
        f = d / "root" / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(val + "\n")
    return HL.cgroup_memory_limit(d / "self_cgroup", d / "root")


def test_memory(tmp):
    print("\n[3] 23d: the job's memory")
    v2 = cg(tmp, "v2", "0::/system.slice/slurmstepd.scope/job_1/step_0/user/task_0\n",
            {"system.slice/slurmstepd.scope/job_1/memory.max": str(46 * GiB),
             "system.slice/slurmstepd.scope/job_1/step_0/memory.max": "max",
             "memory.max": "max"})
    check(f"cgroup v2: the job's limit on a parent counts ({v2})", v2 == 46 * GiB)
    v2b = cg(tmp, "v2b", "0::/a/b\n", {"a/memory.max": str(8 * GiB), "a/b/memory.max": str(4 * GiB)})
    check("cgroup v2: the smallest limit on the path", v2b == 4 * GiB)
    v1 = cg(tmp, "v1", "12:pids:/slurm/uid_1/job_2\n11:memory:/slurm/uid_1/job_2/step_0/task_0\n"
            "4:cpu,cpuacct:/slurm/uid_1/job_2\n",
            {"memory/slurm/uid_1/job_2/memory.limit_in_bytes": str(46 * GiB),
             "memory/slurm/uid_1/job_2/step_0/memory.limit_in_bytes": "9223372036854771712",
             "memory/memory.limit_in_bytes": "9223372036854771712"})
    check(f"cgroup v1 (TinyGPU, Ubuntu 20.04): the job's limit ({v1})", v1 == 46 * GiB)
    v1c = cg(tmp, "v1c", "5:memory,hugetlb:/docker/abc\n",
             {"memory/docker/abc/memory.limit_in_bytes": str(2 * GiB)})
    check("cgroup v1 with combined controllers", v1c == 2 * GiB)
    none = cg(tmp, "none", "0::/user.slice\n", {"user.slice/memory.max": "max", "memory.max": "max"})
    check("no limit ('max', the v1 huge number): None", none is None and
          cg(tmp, "none1", "3:memory:/\n", {"memory/memory.limit_in_bytes": "9223372036854771712"}) is None)
    cont = cg(tmp, "container", "0::/\n", {"memory.max": str(3 * GiB)})
    check("a container (path '/'): the root's limit", cont == 3 * GiB)
    gone = cg(tmp, "gone", "0::/not/mounted/here\n", {"memory.max": str(5 * GiB)})
    check("a path not mounted here: its parents up to the root are tried", gone == 5 * GiB)
    check("no /proc/self/cgroup (macOS): None",
          HL.cgroup_memory_limit(Path(tmp) / "missing", Path(tmp)) is None)

    ram = G.physical_ram()
    saved = G.cgroup_memory_limit
    try:
        for lim, want, label in ((1 * GiB, 1 * GiB, "a limit below the RAM"),
                                 (None, ram, "no limit"),
                                 (ram * 4, ram, "a limit above the RAM")):
            G.cgroup_memory_limit = lambda lim=lim: lim
            check(f"device_memory('cpu') with {label}: {G.device_memory('cpu') / GiB:.1f} GiB",
                  G.device_memory("cpu") == want)
        G.cgroup_memory_limit = lambda: 4 * GiB
        w, why = G.choose_width(2600, 3000, 100_000, "cpu", batch_bytes=0)
        w_ram, _ = G.choose_width(2600, 3000, 100_000, "cpu", batch_bytes=0, total_memory=ram)
        check(f"the width rule plans with the limit ({w} slots at 4 GiB, {w_ram} with the RAM; {why[:30]})",
              w == (int(G.MEMORY_FRACTION["cpu"] * 4 * GiB) - G.BASE_BYTES) // 100_000 and w < w_ram)
    finally:
        G.cgroup_memory_limit = saved
    lim = HL.cgroup_memory_limit()
    rs = RS.memory_limit()
    check(f"run_selected reports the same limit as the width rule uses ({rs})",
          rs == ("none" if lim is None else round(lim / GiB, 1))
          or (not HL.PROC_SELF_CGROUP.exists() and str(rs).startswith(RS.UNAVAILABLE)))


# ============================================================

def test_system():
    print("\n[4] 23e: the allocation in system.json")
    env = {"SLURM_JOB_ID": "1863021", "SLURM_JOB_PARTITION": "rtx3080", "SLURM_JOB_NODELIST": "tg080",
           "SLURM_CPUS_ON_NODE": "16", "SLURM_MEM_PER_GPU": "47104", "CUDA_VISIBLE_DEVICES": "0",
           "SLURM_CLUSTER_NAME": "tinygpu"}
    saved = {k: os.environ.get(k) for k in env}
    try:
        os.environ.update(env)
        info = RS.slurm_info()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    check(f"inside a Slurm job: the allocation ({info})", info == env)
    check("outside: 'not a Slurm job'", "SLURM_JOB_ID" in os.environ or RS.slurm_info() == "not a Slurm job")
    ca = RS.cpus_allowed()
    check(f"cpus_allowed: count and list, or explicit unavailable ({ca})",
          (isinstance(ca, dict) and ca["count"] == len(ML.allowed_cpus())
           and ML.parse_cpu_list(ca["list"]) == ML.allowed_cpus())
          or (ML.allowed_cpus() is None and ca.startswith(RS.UNAVAILABLE)))
    check("cpu ranges written compactly", RS._cpu_ranges({0, 1, 2, 3, 8, 10, 11}) == "0-3,8,10-11")
    os.environ["APPTAINER_CONTAINER"] = "/home/woody/x/byte.sif"
    try:
        kind = RS.container()
    finally:
        del os.environ["APPTAINER_CONTAINER"]
    check(f"an Apptainer container is recognised ({kind})", kind == "apptainer: /home/woody/x/byte.sif")
    s = RS.system_info({"python": "3", "torch": "t", "torch_cuda": "", "cuda_available": False,
                        "mps_available": False, "gpus": []})
    check("system_info carries cpus_allowed, container and slurm",
          {"cpus_allowed", "container", "slurm"} <= set(s))


# ============================================================

def main():
    with tempfile.TemporaryDirectory(prefix="test_cluster_") as tmp:
        test_machine_load(tmp)
        test_memory(tmp)
        test_system()
        test_workers(tmp)
    n_fail = sum(1 for _, ok in _RESULTS if not ok)
    if n_fail:
        print(f"\nFAILED -- {n_fail} of {len(_RESULTS)} checks failed")
        return 1
    print(f"\nOK -- {len(_RESULTS)}/{len(_RESULTS)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
