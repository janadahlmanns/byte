"""The tensor code's tracking writer (plan_evotorch.md, Step 32).

`simulate/hdf5_utils.py`'s `save_*` functions open and close the HDF5 file on every call:
2 x P x R + 5 x P calls per generation with all tracking on. On a network filesystem
(the uni PC's `data/` is nfs4) each open/close costs ~6 ms, which made writing 100x
slower than the simulation (Step 31). The scalar code keeps those functions (its workers
write under a lock); the tensor code writes through ONE open file per generation.

Same content as the `save_*` functions, call by call: the same groups and dataset names,
dtypes, values, compression (gzip at h5py's default level; `create_dataset` picks the
same chunk shape for the same array) and the same order of writes.
tests/test_h5_writer.py compares the two files dataset by dataset.

Staging (optional, `experiment.evaluator.tracking_stage_dir`): the file is copied to a
local folder, written there, and copied back once closed. Worth it only when the
destination is a network filesystem; it is never switched on automatically.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional

import h5py
import numpy as np

NETWORK_FS = {"nfs", "nfs4", "cifs", "smb3", "smbfs", "lustre", "gpfs", "beegfs", "ceph",
              "fuse.sshfs", "afs", "9p", "webdav", "fuse.glusterfs"}
RAM_FS = {"tmpfs", "ramfs"}


# ---------------------------------------------------------------------------
# Filesystem of a path
# ---------------------------------------------------------------------------

def _unescape_mount(s: str) -> str:
    return s.replace("\\040", " ").replace("\\011", "\t").replace("\\012", "\n").replace("\\134", "\\")


def mount_table():
    """[(mount point, fstype)] of this machine. Linux: /proc/mounts; macOS: `mount`
    (a non-local mount gets '(not local)' appended). Empty when it cannot be read."""
    mounts = []
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/mounts") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 3:
                        mounts.append((_unescape_mount(parts[1]), parts[2]))
        elif sys.platform == "darwin":
            out = subprocess.run(["mount"], capture_output=True, text=True, timeout=10).stdout
            for line in out.splitlines():
                # "/dev/disk3s5 on /System/Volumes/Data (apfs, local, journaled)"
                if " on " in line and " (" in line:
                    mp = line.split(" on ", 1)[1].rsplit(" (", 1)
                    opts = [o.strip() for o in mp[1].rstrip(")").split(",")]
                    fstype = opts[0] if opts else "unknown"
                    if "local" not in opts and fstype not in RAM_FS:
                        fstype = f"{fstype}(not local)"
                    mounts.append((mp[0], fstype))
    except (OSError, subprocess.SubprocessError):
        mounts = []
    return mounts


def filesystem_of(path, mounts=None) -> Dict[str, object]:
    """{fstype, mount, network, ram} of the filesystem holding `path` (the deepest
    mount point above it). When the mount table cannot be read: fstype 'unknown'
    (network and ram False -- said in `fstype`)."""
    p = os.path.realpath(path)
    mounts = mount_table() if mounts is None else mounts
    best = None
    for mp, fstype in mounts:
        if p == mp or p.startswith(mp.rstrip("/") + "/") or mp == "/":
            if best is None or len(mp) > len(best[0]):
                best = (mp, fstype)
    if best is None:
        return {"fstype": "unknown", "mount": "", "network": False, "ram": False}
    fstype = best[1]
    network = fstype in NETWORK_FS or fstype.startswith("nfs") or fstype.endswith("(not local)")
    return {"fstype": fstype, "mount": best[0], "network": network, "ram": fstype in RAM_FS}


def _mem_available_bytes() -> Optional[int]:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


# ---------------------------------------------------------------------------
# Staging folder: checked when the evaluator is built, and before writing
# ---------------------------------------------------------------------------

def check_stage_dir(value) -> Optional[Path]:
    """The `tracking_stage_dir` setting: None / 'none' -> no staging; otherwise an
    existing, writable folder on a local filesystem. Fails loudly, before any simulation."""
    if value is None or (isinstance(value, str) and value.strip().lower() == "none"):
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"[ERROR] experiment.evaluator.tracking_stage_dir must be a folder "
                         f"path or 'none', got {value!r}")
    p = Path(os.path.expandvars(os.path.expanduser(value))).resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"[ERROR] tracking_stage_dir {p} does not exist or is not a folder")
    if not os.access(p, os.W_OK):
        raise PermissionError(f"[ERROR] tracking_stage_dir {p} is not writable")
    fs = filesystem_of(p)
    if fs["network"]:
        raise ValueError(f"[ERROR] tracking_stage_dir {p} is on a network filesystem "
                         f"({fs['fstype']}); staging only helps on a local disk. Pick a local "
                         f"folder (e.g. /tmp, /var/tmp, a node's $TMPDIR) or 'none'.")
    return p


def check_stage_space(stage_dir: Path, need_bytes: int, existing_bytes: int = 0) -> None:
    """Before writing: the staged file must fit. `need_bytes` is an upper bound (the
    uncompressed size); on a RAM filesystem (tmpfs) it must also fit in free RAM."""
    need = int(need_bytes) + int(existing_bytes)
    free = shutil.disk_usage(stage_dir).free
    fs = filesystem_of(stage_dir)
    mem = _mem_available_bytes() if fs["ram"] else None
    gib = lambda b: b / 2 ** 30   # noqa: E731
    if need > free:
        raise RuntimeError(
            f"[ERROR] tracking_stage_dir {stage_dir}: the tracking file may need up to "
            f"{gib(need):.2f} GiB (uncompressed bound), only {gib(free):.2f} GiB free. Pick "
            f"another local folder or set tracking_stage_dir: none.")
    if mem is not None and need > mem:
        raise RuntimeError(
            f"[ERROR] tracking_stage_dir {stage_dir} is a RAM filesystem ({fs['fstype']}); the "
            f"tracking file may need up to {gib(need):.2f} GiB (uncompressed bound), only "
            f"{gib(mem):.2f} GiB of RAM available. Pick a folder on disk (e.g. /var/tmp) or "
            f"set tracking_stage_dir: none.")


@contextmanager
def staged(final: Path, stage_dir: Optional[Path], timings: Dict[str, float]):
    """Yield the path to write to. With a stage folder: copy `final` there, yield the
    copy, and once the caller is done copy it back (to a temporary name next to `final`,
    then an atomic rename). On an error the staged copy is removed and the error raised;
    `final` keeps what it had before."""
    final = Path(final)
    if stage_dir is None:
        yield final
        return
    stage = Path(stage_dir) / f"{final.name}.{os.getpid()}.staging"
    partial = final.with_name(final.name + ".staging_partial")
    t = time.perf_counter()
    shutil.copyfile(final, stage)
    timings["stage_in_s"] = time.perf_counter() - t
    try:
        yield stage
        t = time.perf_counter()
        shutil.copyfile(stage, partial)
        os.replace(partial, final)
        timings["stage_out_s"] = time.perf_counter() - t
    finally:
        for f in (stage, partial):
            try:
                f.unlink()
            except FileNotFoundError:
                pass


# ---------------------------------------------------------------------------
# One open file
# ---------------------------------------------------------------------------

class TrackingFile:
    """The `save_*` functions of simulate/hdf5_utils.py, through one open file."""

    def __init__(self, path):
        self.path = Path(path)
        self.f = None

    def __enter__(self):
        self.f = h5py.File(self.path, "a")
        return self

    def __exit__(self, *exc):
        self.f.close()
        self.f = None
        return False

    def _group(self, *names):
        g = self.f
        for n in names:
            g = g[n] if n in g else g.create_group(n)
        return g

    def _put(self, grp, name, data, **kw):
        if name in grp:
            del grp[name]
        grp.create_dataset(name, data=data, **kw)

    def variant_summary(self, v, summary):
        self._put(self._group(f"variant_{v}"), "summary", summary, compression="gzip")

    def wiring(self, v, wiring):
        self._put(self._group(f"variant_{v}"), "wiring", wiring, compression="gzip")

    def modulation(self, v, modulation):
        self._put(self._group(f"variant_{v}"), "modulation", modulation, compression="gzip")

    def eta(self, v, eta):
        self._put(self._group(f"variant_{v}"), "eta", np.float32(eta))

    def tonic_activations(self, v, tonic):
        self._put(self._group(f"variant_{v}"), "tonic_activations", tonic, compression="gzip")

    def per_tick(self, v, r, arr):
        self._put(self._group(f"variant_{v}", f"run_{r}"), "per_tick", arr, compression="gzip")

    def heatmap(self, v, r, staying):
        self._put(self._group(f"variant_{v}", f"run_{r}"), "staying", staying, compression="gzip")
