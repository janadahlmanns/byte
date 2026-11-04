#!/usr/bin/env python3
"""Is the machine free? -- before each config and while it runs (plan Step 22e).

A runtime is only comparable when nothing else uses the machine. Two tools, both used
by simulate/run_selected.py; neither imports torch:

  idle_check()   a few seconds of sampling *before* a config: GPU load and memory in
                 use, the GPU's compute processes, and how busy the CPU is (all cores).
                 Returns the numbers and a verdict, "ok" or "busy: <reasons>".
  LoadSampler    a thread *while* a config runs: CPU time used by processes that are
                 not this runner or its children (another user's job, an update), in
                 cores, and GPU compute processes that are not ours.

CPU numbers come from /proc (Linux). Elsewhere (macOS) they say "unavailable: ...".

Only this job's CPUs count (plan Step 23b): on a shared cluster node (TinyGPU) other
users' jobs run on their own cores, which is no disturbance. So CPU busy is measured on
the cores this process may run on (os.sched_getaffinity), and a process is foreign only
if it may run on one of them (its Cpus_allowed_list). Unrestricted, that is every core
and every process: the same as before.
Inside a container /proc and nvidia-smi only see the container's own processes, so
other users' work is invisible there; the GPU load and memory are still the whole card's.
"""

import os
import subprocess
import threading
import time
from pathlib import Path

UNAVAILABLE = "unavailable: "

# Before a config, above these the machine counts as busy. An idle RTX 3070 shows ~47
# MiB and 0-3% load; an idle desktop CPU a few percent.
IDLE_GPU_UTIL_PCT = 10
IDLE_GPU_MEM_MIB = 300
IDLE_CPU_BUSY_PCT = 10
# While a config runs: foreign CPU use above this many cores is a disturbance.
FOREIGN_CORES_MAX = 0.5

PROC = Path("/proc")
JOB_CPUS = "job"            # the default `cpus`: the cores this process may run on


def _cmd(args, timeout=20):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return UNAVAILABLE + f"{args[0]} not found"
    except subprocess.TimeoutExpired:
        return UNAVAILABLE + f"{args[0]} timed out"
    if p.returncode != 0:
        return UNAVAILABLE + f"{args[0]} exit code {p.returncode}"
    return p.stdout.strip()


# ============================================================
# CPU, from /proc
# ============================================================

def allowed_cpus():
    """The cores this process may run on, or None where the OS does not say (macOS)."""
    if hasattr(os, "sched_getaffinity"):
        return frozenset(os.sched_getaffinity(0))
    return None


def resolve_cpus(cpus):
    """JOB_CPUS -> allowed_cpus(); None (the whole machine) or a set stays as given."""
    return allowed_cpus() if cpus == JOB_CPUS else (None if cpus is None else frozenset(cpus))


def parse_cpu_list(text):
    """"0-3,8,10-11" -> frozenset({0, 1, 2, 3, 8, 10, 11})."""
    out = set()
    for part in text.strip().split(","):
        if not part:
            continue
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return frozenset(out)


def _busy_total(fields):
    v = [int(x) for x in fields]
    idle = v[3] + (v[4] if len(v) > 4 else 0)            # idle + iowait
    total = sum(v[:8])                                    # without guest (counted in user)
    return total - idle, total


def cpu_totals(proc=PROC, cpus=None):
    """(busy, total) jiffies of the cores `cpus` together (None: the whole machine, the
    `cpu` line), or None without /proc/stat or when a core of `cpus` has no line."""
    try:
        lines = (proc / "stat").read_text().splitlines()
    except OSError:
        return None
    rows = {}
    for line in lines:
        f = line.split()
        if f and f[0].startswith("cpu"):
            rows[f[0]] = f[1:]
    if cpus is None:
        return _busy_total(rows["cpu"]) if "cpu" in rows else None
    if not cpus or any(f"cpu{c}" not in rows for c in cpus):
        return None
    parts = [_busy_total(rows[f"cpu{c}"]) for c in sorted(cpus)]
    return sum(b for b, _t in parts), sum(t for _b, t in parts)


def processes(proc=PROC):
    """{pid: (ppid, cpu seconds so far, name, allowed cores or None)} of every process
    /proc shows (allowed: its Cpus_allowed_list; None if unreadable)."""
    tick = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
    out = {}
    if not (proc / "stat").is_file():
        return out
    for d in proc.iterdir():
        if not d.name.isdigit():
            continue
        try:
            raw = (d / "stat").read_text()
        except OSError:
            continue                                      # ended meanwhile
        # The name is in parentheses and may contain spaces: split after the last ')'.
        name = raw[raw.index("(") + 1:raw.rindex(")")]
        rest = raw[raw.rindex(")") + 2:].split()
        ppid, utime, stime = int(rest[1]), int(rest[11]), int(rest[12])
        allowed = None
        try:
            for line in (d / "status").read_text().splitlines():
                if line.startswith("Cpus_allowed_list:"):
                    allowed = parse_cpu_list(line.split(":", 1)[1])
                    break
        except (OSError, ValueError):
            pass
        out[int(d.name)] = (ppid, (utime + stime) / tick, name, allowed)
    return out


def may_share(allowed, cpus):
    """Whether a process allowed on `allowed` may run on our cores `cpus`. Unknown
    (None) on either side counts as yes: never miss a disturbance for lack of data."""
    return allowed is None or cpus is None or bool(allowed & cpus)


def own_tree(procs, root=None):
    """The pids of `root` (default: this process) and all its descendants."""
    root = os.getpid() if root is None else root
    children = {}
    for pid, (ppid, *_rest) in procs.items():
        children.setdefault(ppid, []).append(pid)
    tree, todo = set(), [root]
    while todo:
        p = todo.pop()
        if p not in tree:
            tree.add(p)
            todo += children.get(p, [])
    return tree


# ============================================================
# GPU, from nvidia-smi
# ============================================================

def gpu_load():
    """[(util %, memory used MiB)] per GPU, or an 'unavailable: ...' string."""
    q = _cmd(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
              "--format=csv,noheader,nounits"])
    if q.startswith(UNAVAILABLE):
        return q
    try:
        return [tuple(int(float(x)) for x in line.split(",")) for line in q.splitlines() if line.strip()]
    except ValueError:
        return UNAVAILABLE + f"nvidia-smi output not understood: {q[:80]!r}"


def gpu_processes():
    """[(pid, name, used memory)] of the GPU's compute processes, or 'unavailable: ...'."""
    q = _cmd(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
              "--format=csv,noheader"])
    if q.startswith(UNAVAILABLE):
        return q
    out = []
    for line in q.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 3 and parts[0].isdigit():
            out.append((int(parts[0]), parts[1], parts[2]))
    return out


# ============================================================
# Before a config
# ============================================================

def idle_check(seconds=5.0, interval=1.0, proc=PROC, cpus=JOB_CPUS):
    """Sample for `seconds` and judge whether the machine is idle.

    Returns a dict: gpu_util_max_pct and gpu_mem_used_max_mib (per GPU, '|'-joined),
    gpu_processes, cpu_busy_pct (over `cpus`, default this job's cores; the whole
    window), cpus_judged, cpu_top (the busiest processes that may run on those cores,
    in cores), verdict ("ok" | "busy: ..." | "unavailable: ...")."""
    cpus = resolve_cpus(cpus)
    procs0, cpu0 = processes(proc), cpu_totals(proc, cpus)
    loads, t0 = [], time.perf_counter()
    while True:
        loads.append(gpu_load())
        if time.perf_counter() - t0 >= seconds:
            break
        time.sleep(interval)
    elapsed = time.perf_counter() - t0
    procs1, cpu1 = processes(proc), cpu_totals(proc, cpus)
    gp = gpu_processes()

    res, reasons = {"cpus_judged": "all" if cpus is None else str(len(cpus))}, []
    ok_loads = [x for x in loads if isinstance(x, list)]
    if ok_loads:
        util = [max(s[g][0] for s in ok_loads) for g in range(len(ok_loads[0]))]
        mem = [max(s[g][1] for s in ok_loads) for g in range(len(ok_loads[0]))]
        res["gpu_util_max_pct"] = "|".join(map(str, util))
        res["gpu_mem_used_max_mib"] = "|".join(map(str, mem))
        if max(util) > IDLE_GPU_UTIL_PCT:
            reasons.append(f"GPU load up to {max(util)}%")
        if max(mem) > IDLE_GPU_MEM_MIB:
            reasons.append(f"{max(mem)} MiB GPU memory in use")
    else:
        res["gpu_util_max_pct"] = res["gpu_mem_used_max_mib"] = loads[0]
    if isinstance(gp, list):
        res["gpu_processes"] = "; ".join(f"{p} {n} ({m})" for p, n, m in gp)
        if gp:
            reasons.append(f"{len(gp)} GPU process(es): {gp[0][1]} (pid {gp[0][0]})")
    else:
        res["gpu_processes"] = gp
    if cpu0 and cpu1 and cpu1[1] > cpu0[1]:
        busy = 100.0 * (cpu1[0] - cpu0[0]) / (cpu1[1] - cpu0[1])
        res["cpu_busy_pct"] = f"{busy:.1f}"
        top = sorted(((procs1[p][1] - procs0[p][1]) / elapsed, procs1[p][2], p)
                     for p in procs1 if p in procs0 and may_share(procs1[p][3], cpus))[::-1][:3]
        res["cpu_top"] = "; ".join(f"{n} (pid {p}) {c:.2f} cores" for c, n, p in top if c >= 0.05)
        if busy > IDLE_CPU_BUSY_PCT:
            reasons.append(f"CPU {busy:.0f}% busy"
                           + (f" (top: {res['cpu_top'].split(';')[0]})" if res["cpu_top"] else ""))
    else:
        res["cpu_busy_pct"] = UNAVAILABLE + ("no /proc/stat (not Linux)" if not (proc / "stat").is_file()
                                             else "a core of this job has no line in /proc/stat")
        res["cpu_top"] = res["cpu_busy_pct"]
    if reasons:
        res["verdict"] = "busy: " + "; ".join(reasons)
    elif not ok_loads and not cpu0:
        res["verdict"] = UNAVAILABLE + "neither nvidia-smi nor /proc available"
    else:
        res["verdict"] = "ok"
    return res


# ============================================================
# While a config runs
# ============================================================

class LoadSampler:
    """Polls every `interval` s in a thread: CPU seconds of processes outside this
    process's tree (this runner, run_ea, its worker and compile processes) that may run
    on this job's cores (`cpus`), and GPU compute processes outside the tree. Processes that start and end between two samples are
    missed; with the default 2 s that is only very short work."""

    def __init__(self, interval=2.0, proc=PROC, root=None, cpus=JOB_CPUS):
        self.interval, self.proc, self.root = interval, proc, root
        self.cpus = resolve_cpus(cpus)
        self.foreign_cpu_s = 0.0
        self.by_name = {}                     # process name -> foreign CPU seconds
        self.foreign_gpu = {}                 # pid -> "name (memory)"
        self.gpu_visible = None               # nvidia-smi answered at least once
        self.cpu_visible = (proc / "stat").is_file()
        self.elapsed = 0.0
        self._last = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self):
        procs = processes(self.proc) if self.cpu_visible else {}
        mine = own_tree(procs, self.root) if procs else set()
        if self._last is not None:
            for pid, (_pp, cpu, name, allowed) in procs.items():
                if pid in mine or not may_share(allowed, self.cpus):
                    continue
                before = self._last.get(pid)
                # A process seen for the first time counts from 0 only if it started
                # after the previous sample; we cannot tell, so count its delta only
                # from the next sample on (conservative: never over-counts).
                if before is not None and before[2] == name:
                    d = max(0.0, cpu - before[1])
                    self.foreign_cpu_s += d
                    self.by_name[name] = self.by_name.get(name, 0.0) + d
        self._last = procs
        gp = gpu_processes()
        if isinstance(gp, list):
            self.gpu_visible = True
            for pid, name, mem in gp:
                if procs and pid in mine:
                    continue
                if not procs and self.root is None:
                    continue                  # no /proc: cannot tell ours from foreign
                self.foreign_gpu[pid] = f"{name} ({mem})"
        elif self.gpu_visible is None:
            self.gpu_visible = False

    def _run(self):
        t0 = time.perf_counter()
        while True:
            self._sample()
            self.elapsed = time.perf_counter() - t0
            if self._stop.wait(self.interval):
                break
        self._sample()
        self.elapsed = time.perf_counter() - t0

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=60)
        return False

    def foreign_cores(self):
        """Mean foreign CPU use over the run, in cores, as text; or 'unavailable: ...'."""
        if not self.cpu_visible:
            return UNAVAILABLE + "no /proc (not Linux)"
        return f"{self.foreign_cpu_s / max(self.elapsed, 1e-9):.2f}"

    def foreign_top(self, k=3):
        if not self.cpu_visible:
            return UNAVAILABLE + "no /proc (not Linux)"
        top = sorted(self.by_name.items(), key=lambda kv: -kv[1])[:k]
        return "; ".join(f"{n} {s:.0f} cpu-s" for n, s in top if s >= 1.0)

    def foreign_gpu_text(self):
        if not self.gpu_visible:
            return UNAVAILABLE + "nvidia-smi not available"
        if not self.cpu_visible:
            return UNAVAILABLE + "no /proc: cannot tell this run's GPU processes from others"
        return "; ".join(f"{p} {v}" for p, v in sorted(self.foreign_gpu.items()))

    def disturbance(self):
        """None, or why this run's timing is not clean."""
        reasons = []
        if self.cpu_visible and self.foreign_cpu_s / max(self.elapsed, 1e-9) > FOREIGN_CORES_MAX:
            reasons.append(f"other processes used {self.foreign_cores()} cores on average "
                           f"({self.foreign_top()})")
        if self.foreign_gpu and self.cpu_visible:
            reasons.append(f"other GPU process(es): {self.foreign_gpu_text()}")
        return "; ".join(reasons) or None
