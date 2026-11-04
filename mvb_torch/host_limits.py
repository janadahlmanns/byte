"""The memory this process may use: below the machine's RAM in a Slurm job or a
container with a memory limit (plan Step 23d).

On a shared cluster node (TinyGPU rtx3080: 384 GB per node, 46 GB per GPU job) the
tensor-CPU width rule must plan with the job's limit, not the node's RAM, or the job is
killed. Slurm and Docker enforce limits through the memory cgroup:

  cgroup v2  one tree; /proc/self/cgroup has "0::<path>", the limit is
             <root>/<path>/memory.max ("max" = none)
  cgroup v1  one tree per controller; the memory line is "<n>:memory:<path>" (or with
             other controllers, e.g. "memory,..."), the limit is
             <root>/memory/<path>/memory.limit_in_bytes (a huge number = none)

A limit set on a parent cgroup (Slurm: the job; the process sits in a step below it)
applies too, so the smallest limit on the path and its parents counts. Inside a
container the path may not exist under <root>; then its parents up to <root> are tried.

Deliberately torch-free: simulate/run_selected.py loads this file directly.

Check the limit of a job by hand (as a script, not with -m: the package's __init__
imports this module through generation.py, so -m would load it twice and warn):
    python mvb_torch/host_limits.py
"""

import os
from pathlib import Path
from typing import Optional

CGROUP_ROOT = Path("/sys/fs/cgroup")
PROC_SELF_CGROUP = Path("/proc/self/cgroup")
# cgroup v1 reports "no limit" as the largest page-aligned 64-bit value; anything this
# large is no limit.
_V1_UNLIMITED = 2 ** 60


def _parents(path: str):
    """'/a/b/c' -> ['/a/b/c', '/a/b', '/a', '/']"""
    parts = [p for p in path.split("/") if p]
    return ["/" + "/".join(parts[:i]) for i in range(len(parts), 0, -1)] + ["/"]


def _read_limit(f: Path) -> Optional[int]:
    try:
        v = f.read_text().strip()
    except OSError:
        return None
    if v == "max":
        return None
    try:
        n = int(v)
    except ValueError:
        return None
    return None if n >= _V1_UNLIMITED else n


def cgroup_memory_limit(proc_self_cgroup=PROC_SELF_CGROUP, root=CGROUP_ROOT) -> Optional[int]:
    """The smallest memory limit on this process's cgroup path, in bytes; None if
    there is none or no cgroup information (macOS, Windows)."""
    try:
        lines = Path(proc_self_cgroup).read_text().splitlines()
    except OSError:
        return None
    root = Path(root)
    found = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _hier, controllers, path = parts
        if controllers == "":                                   # v2
            base, name = root, "memory.max"
        elif "memory" in controllers.split(","):                # v1
            base, name = root / "memory", "memory.limit_in_bytes"
        else:
            continue
        for p in _parents(path):
            lim = _read_limit(base / p.lstrip("/") / name)
            if lim is not None:
                found.append(lim)
    return min(found) if found else None


def describe_limit(limit: Optional[int]) -> str:
    return "none" if limit is None else f"{limit / 2**30:.1f}"


if __name__ == "__main__":
    lim = cgroup_memory_limit()
    print(f"cgroup memory limit: {describe_limit(lim)} GiB"
          + ("" if lim is None else f" ({lim} bytes)") + f"; pid {os.getpid()}")
