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

def cpu_totals(proc=PROC):
    """(busy, total) jiffies of all cores together, or None without /proc/stat."""
    try:
        first = (proc / "stat").read_text().splitlines()[0].split()
    except (OSError, IndexError):
        return None
    v = [int(x) for x in first[1:]]
    idle = v[3] + (v[4] if len(v) > 4 else 0)            # idle + iowait
    total = sum(v[:8])                                    # without guest (counted in user)
    return total - idle, total


def processes(proc=PROC):
    """{pid: (ppid, cpu seconds so far, name)} of every process /proc shows."""
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
        out[int(d.name)] = (ppid, (utime + stime) / tick, name)
    return out


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

def idle_check(seconds=5.0, interval=1.0, proc=PROC):
    """Sample for `seconds` and judge whether the machine is idle.

    Returns a dict: gpu_util_max_pct and gpu_mem_used_max_mib (per GPU, '|'-joined),
    gpu_processes, cpu_busy_pct (all cores, the whole window), cpu_top (the busiest
    processes in the window, cores), verdict ("ok" | "busy: ..." | "unavailable: ...")."""
    procs0, cpu0 = processes(proc), cpu_totals(proc)
    loads, t0 = [], time.perf_counter()
    while True:
        loads.append(gpu_load())
        if time.perf_counter() - t0 >= seconds:
            break
        time.sleep(interval)
    elapsed = time.perf_counter() - t0
    procs1, cpu1 = processes(proc), cpu_totals(proc)
    gp = gpu_processes()

    res, reasons = {}, []
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
                     for p in procs1 if p in procs0)[::-1][:3]
        res["cpu_top"] = "; ".join(f"{n} (pid {p}) {c:.2f} cores" for c, n, p in top if c >= 0.05)
        if busy > IDLE_CPU_BUSY_PCT:
            reasons.append(f"CPU {busy:.0f}% busy"
                           + (f" (top: {res['cpu_top'].split(';')[0]})" if res["cpu_top"] else ""))
    else:
        res["cpu_busy_pct"] = UNAVAILABLE + "no /proc/stat (not Linux)"
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
    process's tree (this runner, run_ea, its worker and compile processes), and GPU
    compute processes outside it. Processes that start and end between two samples are
    missed; with the default 2 s that is only very short work."""

    def __init__(self, interval=2.0, proc=PROC, root=None):
        self.interval, self.proc, self.root = interval, proc, root
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
            for pid, (_pp, cpu, name) in procs.items():
                if pid in mine:
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
