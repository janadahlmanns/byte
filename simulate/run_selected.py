#!/usr/bin/env python3
"""Pick experiment configs in a terminal screen and run them one after another
(plan_evotorch.md Step 14).

Made for the Docker image that testers build on their own machines: it lists every
config with its key facts, runs the ticked ones in the order they were ticked, and
writes one folder to send back -- the machine's hardware, a summary with each config's
runtime, per-generation times and one log per config.

Each config runs as its own `python -m simulate.run_ea --config <path>` process, i.e.
exactly the command a person would type, so results are identical to a manual run and
no state carries over from one config to the next. Its stdin gets one "n" (the
"Display generation statistics plot?" question) and is then closed: any other question
fails the run loudly (EOFError) instead of hanging.

Usage
-----
    python -m simulate.run_selected                     # selection screen (needs a terminal)
    python -m simulate.run_selected --list              # print the table and exit
    python -m simulate.run_selected --configs a,b       # run these, no screen

Sweeps (plan Step 15): a file in configs/sweeps/ names a base config, settings for
every point (`set`) and the values to vary (`vary`); each point is a row of its own.
"""

import argparse
import copy
import csv
import datetime
import itertools
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs" / "experiments"
EXTRA_CONFIG_DIR = Path("/my_configs")          # mounted by the tester (README, option 2)
SWEEP_DIR = ROOT / "configs" / "sweeps"
PRESET_DIR = ROOT / "configs" / "presets"
DEFAULT_PRESET = "tester_default"           # pre-ticked on the screen (plan Step 17)
REFERENCE_IGNORED_ATTRS = {"experiment_output_folder"}
DEFAULT_REPORT_DIR = ROOT / "data" / "runtime_reports"

# Lines of the run_ea output this runner reads. The timing line is printed by the
# tensor evaluator (mvb_torch/adapter.py) once per generation.
GEN_LINE = re.compile(r"\[tensor-evaluator\].* in ([\d.]+)s \((\d+) iterations, (\d+) ms/iteration")
STATS_LINE = re.compile(
    r"\[tensor-evaluator\] stats: run-ticks (\d+), [\d.]+ us/run-tick; memory: (.*); "
    r"width (\d+) of (\d+), all runs need ~(\d+) MiB \(est\.\)")
HDF5_LINE = re.compile(r"\[HDF5 Init\] File created: (.+)$")
DONE_LINE = "Results saved to:"

UNAVAILABLE = "unavailable: "


# ============================================================
# Config facts
# ============================================================

@dataclass
class ConfigInfo:
    name: str                       # shown in the list; what --configs accepts
    path: Path
    variants: str = "?"
    runs: str = "?"
    gens: str = "?"
    genome: str = "?"
    device: str = "?"
    neurons: str = "?"
    reasons: List[str] = field(default_factory=list)   # empty => can be selected
    cfg: Optional[dict] = None      # a sweep point: the expanded config (no file yet)

    @property
    def selectable(self):
        return not self.reasons


def _get(cfg, keys, info):
    """cfg[k0][k1]...; on a missing key, record it as a reason and return None."""
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            info.reasons.append("missing key " + ".".join(keys))
            return None
        cur = cur[k]
    return cur


def describe(path, name, available_devices):
    """The facts shown for one config, and the reasons it cannot run here (if any).

    `available_devices`: e.g. {"cpu", "cuda"} -- the devices torch sees on this machine.
    Every value is read from the YAML; a missing key shows as "?" and makes the config
    unselectable, so nothing is guessed.
    """
    try:
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001 -- shown on the screen as the reason
        return ConfigInfo(name=name, path=Path(path),
                          reasons=[f"YAML cannot be read: {type(e).__name__}"])
    return describe_cfg(cfg, name, path, available_devices)


def describe_cfg(cfg, name, path, available_devices):
    """`describe` for a config already loaded (or expanded from a sweep)."""
    info = ConfigInfo(name=name, path=Path(path))
    if not isinstance(cfg, dict):
        info.reasons.append("YAML is not a mapping")
        return info

    exp = _get(cfg, ["experiment"], info)
    if exp is None:
        return info

    v = _get(cfg, ["experiment", "population_size"], info)
    info.variants = "?" if v is None else str(v)
    if exp.get("simulation_seed") == "from_file":
        info.runs = "file"                      # benchmark mode: taken from the source HDF5
    else:
        v = _get(cfg, ["experiment", "n_runs"], info)
        info.runs = "?" if v is None else str(v)

    is_ea = exp.get("evolutionary_algorithm_enabled") is True
    if is_ea:
        v = _get(cfg, ["experiment", "evolutionary_algorithm", "num_generations"], info)
        info.gens = "?" if v is None else str(v)
    else:
        info.gens = "-"
        info.reasons.append("not an EA config")

    g = _get(cfg, ["experiment", "genome_type"], info)
    if g is not None:
        g = str(g)
        info.genome = "lookup" if g.startswith("lookup") else g
        if g == "from_file":
            info.reasons.append("needs input files (genome_type: from_file)")
    if exp.get("simulation_seed") == "from_file":
        info.reasons.append("needs input files (simulation_seed: from_file)")

    ev = exp.get("evaluator")
    if ev is None or (isinstance(ev, dict) and ev.get("backend") == "scalar"):
        info.device = "cpu"
    elif isinstance(ev, dict) and ev.get("backend") == "tensor":
        d = _get(cfg, ["experiment", "evaluator", "device"], info)
        if d is not None:
            info.device = str(d)
            if info.device not in available_devices:
                info.reasons.append({
                    "cuda": "CUDA not available (Docker: add --gpus all)",
                    "mps": "MPS not available on this machine",
                }.get(info.device, f"device {info.device!r} not available"))
    else:
        info.reasons.append(f"unknown evaluator block {ev!r}")

    v = _get(cfg, ["brain", "n_neurons"], info)
    info.neurons = "?" if v is None else str(v)

    if exp.get("viz_enabled") or exp.get("viz_brain_enabled"):
        info.reasons.append("visualisation on (no screen here)")
    if is_ea and any(exp.get(k) for k in ("enable_per_run_tracking", "enable_per_tick_tracking",
                                           "enable_heat_map_tracking")):
        # run_ea would ask whether to continue without tracking.
        info.reasons.append("tracking on (run_ea would ask a question)")
    return info


def config_files():
    """(display name, path) of every config in configs/experiments, then /my_configs."""
    if not CONFIG_DIR.is_dir():
        raise FileNotFoundError(f"config folder not found: {CONFIG_DIR}")
    out = []
    for folder, prefix in ((CONFIG_DIR, ""), (EXTRA_CONFIG_DIR, "my_configs/")):
        if folder.is_dir():
            files = sorted(list(folder.glob("*.yaml")) + list(folder.glob("*.yml")))
            out += [(prefix + p.stem, p) for p in files]
    return out


# ============================================================
# Sweeps (plan Step 15)
# ============================================================

class SweepError(ValueError):
    pass


def _lookup(cfg, dotted):
    """(parent dict, last key) of a dotted key that must already exist in `cfg`."""
    keys = dotted.split(".")
    cur = cfg
    for k in keys[:-1]:
        if not isinstance(cur, dict) or k not in cur:
            raise SweepError(f"unknown key {dotted} (no {k!r} in the base config)")
        cur = cur[k]
    if not isinstance(cur, dict) or keys[-1] not in cur:
        raise SweepError(f"unknown key {dotted} (not in the base config)")
    return cur, keys[-1]


def _type_ok(new, old):
    # Exactly the base value's type: an int stays an int, a bool stays a bool.
    return type(new) is type(old)


def expand_sweep(path, config_dir=CONFIG_DIR):
    """[(row name, expanded config)] of one sweep file; SweepError on any mistake.

    Never adds a key: every `set`/`vary` key must exist in the base, with a value of
    the same type. `simulation_name` gets `_<leaf>-<value>` per varied key, so the
    points never write the same file name."""
    path = Path(path)
    try:
        sw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        raise SweepError(f"YAML cannot be read: {type(e).__name__}") from e
    if not isinstance(sw, dict):
        raise SweepError("a sweep file must be a mapping with base, set, vary")
    extra = set(sw) - {"base", "set", "vary"}
    if extra:
        raise SweepError(f"unknown top-level keys {sorted(extra)} (allowed: base, set, vary)")
    for k in ("base", "vary"):
        if k not in sw:
            raise SweepError(f"missing {k!r}")
    base_path = None
    for ext in (".yaml", ".yml"):
        if (Path(config_dir) / f"{sw['base']}{ext}").is_file():
            base_path = Path(config_dir) / f"{sw['base']}{ext}"
            break
    if base_path is None:
        raise SweepError(f"base config {sw['base']!r} not found in {config_dir}")
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    settings = sw.get("set") or {}
    vary = sw["vary"]
    if not isinstance(settings, dict):
        raise SweepError("'set' must be a mapping of dotted keys to values")
    if not isinstance(vary, dict) or not vary:
        raise SweepError("'vary' must be a non-empty mapping of dotted keys to lists")
    both = set(settings) & set(vary)
    if both:
        raise SweepError(f"keys in both set and vary: {sorted(both)}")

    def assign(cfg, dotted, value):
        parent, leaf = _lookup(cfg, dotted)
        if not _type_ok(value, parent[leaf]):
            raise SweepError(f"{dotted}: {value!r} is {type(value).__name__}, the base "
                             f"value {parent[leaf]!r} is {type(parent[leaf]).__name__}")
        parent[leaf] = value

    common = copy.deepcopy(base)
    for k, v in settings.items():
        assign(common, k, v)
    for k, values in vary.items():
        if not isinstance(values, list) or not values:
            raise SweepError(f"vary.{k} must be a non-empty list")
        if len(set(map(repr, values))) != len(values):
            raise SweepError(f"vary.{k} has duplicate values: {values}")
        _lookup(common, k)
    _lookup(common, "experiment.simulation_name")

    points = []
    for combo in itertools.product(*vary.values()):
        cfg = copy.deepcopy(common)
        label, suffix = [], ""
        for k, v in zip(vary, combo):
            assign(cfg, k, v)
            leaf = k.split(".")[-1]
            label.append(f"{leaf}={v}")
            suffix += f"_{leaf}-{v}"
        cfg["experiment"]["simulation_name"] = str(cfg["experiment"]["simulation_name"]) + suffix
        points.append((f"{path.stem} [{' '.join(label)}]", cfg))
    return points


def sweep_infos(available, sweep_dir=SWEEP_DIR):
    """One row per sweep point; a broken sweep file is one greyed-out row."""
    out = []
    if not Path(sweep_dir).is_dir():
        return out
    for p in sorted(list(Path(sweep_dir).glob("*.yaml")) + list(Path(sweep_dir).glob("*.yml"))):
        try:
            points = expand_sweep(p)
        except SweepError as e:
            out.append(ConfigInfo(name=f"{p.stem} (sweep)", path=p,
                                  reasons=[f"broken sweep file: {e}"]))
            continue
        for name, cfg in points:
            info = describe_cfg(cfg, name, p, available)
            info.cfg = cfg
            out.append(info)
    return out


def all_infos(available):
    return [describe(p, n, available) for n, p in config_files()] + sweep_infos(available)


# ============================================================
# Presets and reference results (plan Step 17)
# ============================================================

class PresetError(ValueError):
    pass


@dataclass
class Preset:
    name: str
    description: str
    configs: List[str]
    references: dict                 # config name -> Path of a reference HDF5


def load_preset(path):
    """A preset file: description, configs (the run order), optional references."""
    path = Path(path)
    try:
        d = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        raise PresetError(f"{path.name}: YAML cannot be read: {type(e).__name__}") from e
    if not isinstance(d, dict):
        raise PresetError(f"{path.name}: must be a mapping with description, configs")
    extra = set(d) - {"description", "configs", "references"}
    if extra:
        raise PresetError(f"{path.name}: unknown keys {sorted(extra)} "
                          f"(allowed: description, configs, references)")
    for k in ("description", "configs"):
        if k not in d:
            raise PresetError(f"{path.name}: missing {k!r}")
    cfgs = d["configs"]
    if not isinstance(cfgs, list) or not cfgs or not all(isinstance(c, str) for c in cfgs):
        raise PresetError(f"{path.name}: 'configs' must be a non-empty list of names")
    dup = sorted({c for c in cfgs if cfgs.count(c) > 1})
    if dup:
        raise PresetError(f"{path.name}: duplicate configs {dup}")
    refs = d.get("references") or {}
    if not isinstance(refs, dict):
        raise PresetError(f"{path.name}: 'references' must map config names to files")
    unknown = sorted(set(refs) - set(cfgs))
    if unknown:
        raise PresetError(f"{path.name}: references for configs not in the preset: {unknown}")
    return Preset(path.stem, str(d["description"]), list(cfgs),
                  {k: (ROOT / v if not Path(v).is_absolute() else Path(v))
                   for k, v in refs.items()})


def load_presets(preset_dir=None):
    preset_dir = PRESET_DIR if preset_dir is None else preset_dir
    if not Path(preset_dir).is_dir():
        return {}
    return {p.stem: load_preset(p) for p in sorted(Path(preset_dir).glob("*.yaml"))}


def preset_infos(preset, infos, devices):
    """The preset's rows in order (names as in --list, or YAML paths, like --configs);
    PresetError if any entry does not resolve."""
    out, missing = [], []
    for c in preset.configs:
        try:
            out.append(resolve(c, infos, devices))
        except FileNotFoundError:
            missing.append(c)
    if missing:
        raise PresetError(f"preset {preset.name}: no config named {missing} "
                          f"(see `python -m simulate.run_selected --list`)")
    return out


def _walk_datasets(f):
    out = {}
    f.visititems(lambda n, o: out.__setitem__(n, o) if hasattr(o, "dtype") else None)
    return out


def _same_value(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def compare_to_reference(path, ref):
    """One line for summary.csv: is the result file bit-identical to the reference?

    The config attributes must match (except the output folder), then every dataset is
    compared bit for bit. Reports only; never fails a run."""
    import h5py
    ref = Path(ref)
    if not ref.is_file():
        return UNAVAILABLE + f"reference not found: {ref}"
    path = Path(path) if Path(path).is_absolute() else ROOT / path
    if not path.is_file():
        return UNAVAILABLE + f"result file not found: {path}"
    try:
        h5py.File(ref, "r").close()
    except OSError as e:
        return UNAVAILABLE + (f"cannot read the reference {ref.name} ({e}); a git-LFS "
                              f"pointer instead of the file? Run `git lfs pull` and rebuild")
    with h5py.File(path, "r") as a, h5py.File(ref, "r") as b:
        keys = (set(a.attrs) | set(b.attrs)) - REFERENCE_IGNORED_ATTRS
        diff = sorted(k for k in keys if k not in a.attrs or k not in b.attrs
                      or not _same_value(a.attrs[k], b.attrs[k]))
        if diff:
            return f"not comparable: config differs from the reference ({', '.join(diff[:4])})"
        da, db = _walk_datasets(a), _walk_datasets(b)
        only = sorted(set(da) ^ set(db))
        common = sorted(set(da) & set(db))
        bad = [n for n in common if not _same_value(da[n][()], db[n][()])]
        if not bad and not only:
            return f"identical ({len(common)} datasets)"
        msg = f"differs: {len(bad)} of {len(common)} datasets"
        if bad:
            msg += f" ({', '.join(bad[:3])}{', ...' if len(bad) > 3 else ''})"
        if only:
            msg += f"; only in one file: {', '.join(only[:3])}"
        if "generation_stats" in bad:
            sa, sb = da["generation_stats"][()], db["generation_stats"][()]
            first = next((i for i in range(min(len(sa), len(sb)))
                          if sa[i].tobytes() != sb[i].tobytes()), None)
            if first is not None:
                g = (int(sa[first]["generation"]) if "generation" in (sa.dtype.names or ())
                     else first)
                msg += f"; generation stats first differ at generation {g}"
        return msg


def resolve(entry, infos, devices):
    """A --configs entry: a listed name, or a path to a YAML file."""
    by_name = {i.name: i for i in infos}
    if entry in by_name:
        return by_name[entry]
    p = Path(entry)
    if p.suffix in (".yaml", ".yml") and p.is_file():
        return describe(p.resolve(), str(p), devices)
    raise FileNotFoundError(f"no config named {entry!r} (and no such YAML file); "
                            f"see `python -m simulate.run_selected --list`")


# ============================================================
# The machine
# ============================================================

# Run in a child process: initialising CUDA here would keep a context (GPU memory)
# alive in this process for the whole queue, and the runs size their batches by it.
_TORCH_PROBE = r"""
import json, platform, torch
d = {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
     "python": platform.python_version(),
     "cuda_available": torch.cuda.is_available(),
     "mps_available": torch.backends.mps.is_available(), "gpus": []}
if d["cuda_available"]:
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        d["gpus"].append({"name": p.name, "memory_gib": round(p.total_memory / 2**30, 2),
                          "capability": f"{p.major}.{p.minor}"})
print(json.dumps(d))
"""


def torch_probe():
    p = subprocess.run([sys.executable, "-c", _TORCH_PROBE], text=True,
                       capture_output=True, cwd=ROOT)
    if p.returncode != 0:
        raise RuntimeError("could not query torch for the devices of this machine:\n"
                           + p.stderr[-2000:])
    return json.loads(p.stdout.strip().splitlines()[-1])


def available_devices(probe):
    return {"cpu"} | ({"cuda"} if probe["cuda_available"] else set()) \
        | ({"mps"} if probe["mps_available"] else set())


def _cmd(args):
    """stdout of a command, or an explicit 'unavailable: <reason>'."""
    try:
        p = subprocess.run(args, text=True, capture_output=True, timeout=30)
    except FileNotFoundError:
        return UNAVAILABLE + f"{args[0]} not found"
    except subprocess.TimeoutExpired:
        return UNAVAILABLE + f"{args[0]} timed out"
    if p.returncode != 0:
        return UNAVAILABLE + f"{args[0]} exit code {p.returncode}: {p.stderr.strip()[:200]}"
    return p.stdout.strip()


def _proc_field(path, key):
    try:
        for line in Path(path).read_text().splitlines():
            if line.startswith(key):
                return line.split(":", 1)[1].strip()
    except OSError as e:
        return UNAVAILABLE + str(e)
    return UNAVAILABLE + f"{key!r} not in {path}"


def cpu_model():
    if sys.platform == "darwin":
        return _cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
    if Path("/proc/cpuinfo").exists():
        return _proc_field("/proc/cpuinfo", "model name")
    return platform.processor() or UNAVAILABLE + "unknown platform"


def ram_gib():
    if sys.platform == "darwin":
        v = _cmd(["sysctl", "-n", "hw.memsize"])
        return v if v.startswith(UNAVAILABLE) else round(int(v) / 2**30, 1)
    v = _proc_field("/proc/meminfo", "MemTotal")
    return v if v.startswith(UNAVAILABLE) else round(int(v.split()[0]) / 2**20, 1)


def memory_limit():
    """A container's memory limit (cgroup v2), which can be below the machine's RAM."""
    p = Path("/sys/fs/cgroup/memory.max")
    if not p.exists():
        return UNAVAILABLE + "no cgroup v2 memory.max"
    v = p.read_text().strip()
    return "none" if v == "max" else round(int(v) / 2**30, 1)


def code_version():
    env = os.environ.get("BYTE_CODE_VERSION")
    if env:
        return env
    head = _cmd(["git", "-C", str(ROOT), "rev-parse", "--short=10", "HEAD"])
    if head.startswith(UNAVAILABLE):
        return head
    dirty = _cmd(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"])
    return head + ("+uncommitted changes" if dirty and not dirty.startswith(UNAVAILABLE) else "")


def gpu_snapshot():
    """Utilisation and memory in use of every GPU, and the compute processes nvidia-smi
    can see. Inside a container it usually cannot see other containers' or the host's
    processes, but utilisation and memory in use are always the whole GPU's."""
    q = _cmd(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
              "--format=csv,noheader,nounits"])
    if q.startswith(UNAVAILABLE):
        return {"gpus": q, "processes": q}
    gpus = []
    for line in q.splitlines():
        i, util, used, total = [x.strip() for x in line.split(",")]
        gpus.append({"index": i, "util_pct": util, "mem_used_mib": used, "mem_total_mib": total})
    procs = _cmd(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                  "--format=csv,noheader"])
    return {"gpus": gpus, "processes": procs.splitlines() if procs else []}


def system_info(probe):
    smi = _cmd(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    return {
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "code_version": code_version(),
        "in_container": Path("/.dockerenv").exists(),
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "cpu_model": cpu_model(),
        "cpu_count": os.cpu_count(),
        "ram_gib": ram_gib(),
        "memory_limit_gib": memory_limit(),
        "python": probe["python"],
        "torch": probe["torch"],
        "torch_cuda": probe["torch_cuda"] or UNAVAILABLE + "torch built without CUDA",
        "cuda_available": probe["cuda_available"],
        "mps_available": probe["mps_available"],
        "gpus": probe["gpus"] or UNAVAILABLE + "no CUDA GPU visible to torch",
        "nvidia_smi_gpus": smi,
        "gpu_processes_note": "nvidia-smi inside a container usually does not list "
                              "processes of the host or other containers; gpu_util and "
                              "gpu_mem_used in summary.csv are the whole GPU's",
    }


def _compile_cache_module():
    """mvb_torch/compile_cache.py, loaded from its file: importing the mvb_torch package
    would import torch into the selector process."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_byte_compile_cache", ROOT / "mvb_torch" / "compile_cache.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compile_cache_dir():
    """The production path's compile-cache folder (plan Step 18b)."""
    return Path(_compile_cache_module().versioned_cache_dir())


def compile_cache_state():
    d = compile_cache_dir()
    return "warm" if d.is_dir() and any(d.iterdir()) else "empty"


# ============================================================
# Running the queue
# ============================================================

@dataclass
class Result:
    order: int
    config: str
    path: str
    status: str = "not run"
    exit_code: Optional[int] = None
    wall_s: Optional[float] = None
    gen_times: List[tuple] = field(default_factory=list)   # (seconds, iterations, ms/it)
    gen_stats: List[dict] = field(default_factory=list)    # from the stats line, per generation
    hdf5: str = ""
    compile_cache_at_start: str = ""
    gpu_before: dict = field(default_factory=dict)
    gpu_after: dict = field(default_factory=dict)
    gpu_mem_used_peak: str = ""
    gpu_sm_clock: str = ""
    gpu_temp_max: str = ""
    gpu_power_max: str = ""
    gpu_throttle: str = ""
    matches_reference: str = ""
    hdf5_copy: str = ""
    log: str = ""


def parse_stats(m):
    """One STATS_LINE match -> dict. The memory part is one of the adapter's three forms."""
    mem = m.group(2)
    peak = re.search(r"peak reserved (\d+) MiB", mem)
    drv = re.search(r"driver allocated (\d+) MiB", mem)
    if peak:
        measure, mib = "cuda peak reserved", int(peak.group(1))
    elif drv:
        measure, mib = "mps driver allocated at end of generation (no peak counter)", int(drv.group(1))
    else:
        measure, mib = mem, None                           # "unavailable on cpu"
    return {"run_ticks": int(m.group(1)), "vram_measure": measure, "vram_mib": mib,
            "width": int(m.group(3)), "all_runs": int(m.group(4)),
            "all_runs_need_mib": int(m.group(5))}


def memory_summary(stats):
    """The per-config memory columns from the per-generation stats.

    vram_peak_mib: the highest over the generations. runs_queued: in some generation
    not every run fit at once. vram_needed_all_runs_est_mib: then the width rule's
    estimate for holding them all (highest over the generations), otherwise empty."""
    if not stats:
        return {"vram_measure": "", "vram_peak_mib": "", "runs_queued": "",
                "vram_needed_all_runs_est_mib": ""}
    vals = [s["vram_mib"] for s in stats if s["vram_mib"] is not None]
    queued = any(s["width"] < s["all_runs"] for s in stats)
    return {"vram_measure": stats[0]["vram_measure"],
            "vram_peak_mib": max(vals) if vals else "",
            "runs_queued": "yes" if queued else "no",
            "vram_needed_all_runs_est_mib":
                max(s["all_runs_need_mib"] for s in stats) if queued else ""}


def us_per_run_tick_gens_1plus(r):
    """Total time / total run-ticks over generations 1+ (gen 0 includes compiling)."""
    t = [g[0] for g in r.gen_times[1:]]
    ticks = [s["run_ticks"] for s in r.gen_stats[1:]]
    if not t or len(t) != len(ticks) or sum(ticks) == 0:
        return ""
    return f"{1e6 * sum(t) / sum(ticks):.2f}"


# Clock-event (throttle) reason bits of nvidia-smi / NVML.
THROTTLE_BITS = {0x1: "gpu_idle", 0x2: "applications_clocks_setting", 0x4: "sw_power_cap",
                 0x8: "hw_slowdown", 0x10: "sync_boost", 0x20: "sw_thermal_slowdown",
                 0x40: "hw_thermal_slowdown", 0x80: "hw_power_brake_slowdown",
                 0x100: "display_clock_setting"}
# Optional fields, sampled when this nvidia-smi knows them (names changed across driver
# versions: clocks_event_reasons replaced clocks_throttle_reasons).
OPTIONAL_FIELDS = {"sm_clock_mhz": ["clocks.sm"], "temp_c": ["temperature.gpu"],
                   "power_w": ["power.draw"],
                   "throttle": ["clocks_event_reasons.active", "clocks_throttle_reasons.active"]}


def decode_throttle(mask):
    names = [v for bit, v in THROTTLE_BITS.items() if mask & bit]
    rest = mask & ~sum(THROTTLE_BITS)
    return "+".join(names + ([hex(rest)] if rest else [])) or "none"


class GpuSampler:
    """Polls nvidia-smi once per second in a thread, while a config runs (whole card:
    CUDA context and other users included): peak memory.used, and -- where this
    nvidia-smi offers them -- the SM clock range, peak temperature, peak power and every
    clock-event (throttle) reason seen. Throttling makes runtimes incomparable, so it is
    recorded rather than guessed afterwards."""

    def __init__(self, interval=1.0):
        self.interval = interval
        self.peaks = None
        self.error = None
        self.fields = {}             # key -> the nvidia-smi field name that works here
        self.stats = {}              # key -> per GPU aggregates
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _probe_fields(self):
        for key, names in OPTIONAL_FIELDS.items():
            for name in names:
                q = _cmd(["nvidia-smi", f"--query-gpu={name}", "--format=csv,noheader,nounits"])
                if not q.startswith(UNAVAILABLE) and "not supported" not in q.lower() \
                        and "invalid" not in q.lower():
                    self.fields[key] = name
                    break

    def _sample(self):
        names = ["memory.used"] + list(self.fields.values())
        q = _cmd(["nvidia-smi", f"--query-gpu={','.join(names)}", "--format=csv,noheader,nounits"])
        if q.startswith(UNAVAILABLE):
            self.error = q
            return False
        rows = [[x.strip() for x in line.split(",")] for line in q.splitlines() if line.strip()]
        mem = [int(r[0]) for r in rows]
        self.peaks = mem if self.peaks is None else [max(a, b) for a, b in zip(self.peaks, mem)]
        for j, key in enumerate(self.fields, start=1):
            for g, r in enumerate(rows):
                st = self.stats.setdefault(key, {}).setdefault(g, {})
                raw = r[j]
                try:
                    v = int(raw, 16) if key == "throttle" else float(raw)
                except ValueError:
                    continue                   # "[N/A]" on this sample
                if key == "throttle":
                    st["mask"] = st.get("mask", 0) | v
                else:
                    st["min"] = min(st.get("min", v), v)
                    st["max"] = max(st.get("max", v), v)
        return True

    def _run(self):
        if not self._sample():
            return
        self._probe_fields()
        while not self._stop.is_set():
            if not self._sample():
                return
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=60)
        return False

    @property
    def peak(self):
        if self.peaks is None:
            return self.error or UNAVAILABLE + "no sample taken"
        return "|".join(map(str, self.peaks))

    def column(self, key):
        """One summary.csv value per GPU, joined by '|'; explicit when unavailable."""
        if self.peaks is None:
            return self.error or UNAVAILABLE + "no sample taken"
        if key not in self.fields:
            return UNAVAILABLE + "not reported by this nvidia-smi"
        out = []
        for g in sorted(self.stats.get(key, {})):
            st = self.stats[key][g]
            if key == "throttle":
                out.append(decode_throttle(st["mask"]) if "mask" in st else "n/a")
            elif "min" not in st:
                out.append("n/a")
            elif key == "sm_clock_mhz":
                out.append(f"{st['min']:.0f}-{st['max']:.0f}")
            else:
                out.append(f"{st['max']:.0f}")
        return "|".join(out) or UNAVAILABLE + "no sample with this field"


def run_one(cmd, log_path, echo=True):
    """Run `cmd`, mirror its output to the terminal and `log_path`, and parse it.

    Returns (exit code or None if interrupted, wall seconds, parsed): parsed has the
    generation times, the per-generation stats, the HDF5 path and whether run_ea
    reported its results saved."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    t0 = time.perf_counter()
    parsed = {"gens": [], "stats": [], "hdf5": "", "done": False}
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, text=True, encoding="utf-8",
                             errors="replace", stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        # The plot question at the end; then EOF, so any other question fails loudly.
        p.stdin.write("n\n")
        p.stdin.close()
        try:
            for line in p.stdout:
                log.write(line)
                log.flush()
                if echo:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                m = GEN_LINE.search(line)
                if m:
                    parsed["gens"].append((float(m.group(1)), int(m.group(2)), int(m.group(3))))
                m = STATS_LINE.search(line)
                if m:
                    parsed["stats"].append(parse_stats(m))
                m = HDF5_LINE.search(line.rstrip())
                if m:
                    parsed["hdf5"] = m.group(1).strip()
                if DONE_LINE in line:
                    parsed["done"] = True
            rc = p.wait()
        except KeyboardInterrupt:
            p.terminate()
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
            log.write("\n[run_selected] interrupted (Ctrl-C)\n")
            parsed["done"] = False
            return None, time.perf_counter() - t0, parsed
    return rc, time.perf_counter() - t0, parsed


SUMMARY_FIELDS = ["order", "config", "status", "exit_code", "wall_s", "gen0_s",
                  "gens_1plus_mean_s", "gens_timed", "us_per_run_tick_gens_1plus",
                  "vram_measure", "vram_peak_mib", "gpu_mem_used_peak_mib",
                  "gpu_sm_clock_min_max_mhz", "gpu_temp_max_c", "gpu_power_max_w",
                  "gpu_throttle_reasons", "runs_queued",
                  "vram_needed_all_runs_est_mib", "matches_reference", "hdf5",
                  "hdf5_copy", "compile_cache_at_start",
                  "gpu_util_before_pct", "gpu_mem_used_before_mib", "gpu_util_after_pct",
                  "gpu_mem_used_after_mib", "config_path", "log"]


def _gpu_cols(snap):
    g = snap.get("gpus")
    if not isinstance(g, list):
        return g or "", g or ""
    return "|".join(x["util_pct"] for x in g), "|".join(x["mem_used_mib"] for x in g)


def write_reports(report_dir, results):
    """summary.csv, generations.csv and gpu_snapshots.json; rewritten after each config,
    so a crash or Ctrl-C keeps what has finished."""
    with open(report_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(SUMMARY_FIELDS)
        for r in results:
            t = [g[0] for g in r.gen_times]
            ub, mb = _gpu_cols(r.gpu_before)
            ua, ma = _gpu_cols(r.gpu_after)
            mem = memory_summary(r.gen_stats)
            w.writerow([r.order, r.config, r.status, "" if r.exit_code is None else r.exit_code,
                        "" if r.wall_s is None else f"{r.wall_s:.2f}",
                        f"{t[0]:.2f}" if t else "",
                        f"{sum(t[1:]) / len(t[1:]):.2f}" if len(t) > 1 else "",
                        len(t), us_per_run_tick_gens_1plus(r),
                        mem["vram_measure"], mem["vram_peak_mib"], r.gpu_mem_used_peak,
                        r.gpu_sm_clock, r.gpu_temp_max, r.gpu_power_max, r.gpu_throttle,
                        mem["runs_queued"], mem["vram_needed_all_runs_est_mib"],
                        r.matches_reference, r.hdf5, r.hdf5_copy, r.compile_cache_at_start, ub, mb, ua, ma, r.path, r.log])
    with open(report_dir / "generations.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["order", "config", "generation", "seconds", "iterations",
                    "ms_per_iteration", "run_ticks", "us_per_run_tick"])
        for r in results:
            for i, (sec, it, ms) in enumerate(r.gen_times):
                st = r.gen_stats[i] if i < len(r.gen_stats) else None
                w.writerow([r.order, r.config, i, f"{sec:.2f}", it, ms,
                            st["run_ticks"] if st else "",
                            f"{1e6 * sec / st['run_ticks']:.2f}" if st and st["run_ticks"] else ""])
    (report_dir / "gpu_snapshots.json").write_text(json.dumps(
        [{"order": r.order, "config": r.config, "before": r.gpu_before, "after": r.gpu_after,
          "mem_used_peak_mib": r.gpu_mem_used_peak} for r in results], indent=2))


def _slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")[:40] or "machine"


def run_queue(infos, report_root, probe, echo=True, references=None, run_info=None):
    """Run `infos` in order; returns (results, report folder).

    `references`: config name -> reference HDF5, compared after the config ran ok.
    `run_info`: how the selection was made, stored in system.json."""
    references = references or {}
    sysinfo = system_info(probe)
    sysinfo["run"] = run_info or {"selection": "not recorded"}
    hw = probe["gpus"][0]["name"] if probe["gpus"] else sysinfo["cpu_model"]
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_dir = Path(report_root) / f"{ts}_{_slug(str(hw))}"
    report_dir.mkdir(parents=True, exist_ok=False)
    (report_dir / "system.json").write_text(json.dumps(sysinfo, indent=2))

    results = []
    for i, c in enumerate(infos):
        path = c.path
        if c.cfg is not None:
            # A sweep point: the expanded config is written here and run from this file.
            (report_dir / "configs").mkdir(exist_ok=True)
            path = report_dir / "configs" / f"{i + 1:02d}_{_slug(c.name)}.yaml"
            path.write_text(f"# expanded from {c.path}, point {c.name} "
                            f"(simulate/run_selected.py)\n"
                            + yaml.safe_dump(c.cfg, sort_keys=False), encoding="utf-8")
        results.append(Result(order=i + 1, config=c.name, path=str(path)))
    write_reports(report_dir, results)
    for r in results:
        print(f"\n{'=' * 78}\n[run_selected] {r.order}/{len(results)}: {r.config}\n{'=' * 78}",
              flush=True)
        r.log = f"{r.order:02d}_{_slug(r.config)}.log"
        r.compile_cache_at_start = compile_cache_state()
        r.gpu_before = gpu_snapshot()
        cmd = [sys.executable, "-m", "simulate.run_ea", "--config", r.path]
        with GpuSampler() as sampler:
            rc, r.wall_s, parsed = run_one(cmd, report_dir / r.log, echo)
        r.gpu_mem_used_peak = sampler.peak
        r.gpu_sm_clock = sampler.column("sm_clock_mhz")
        r.gpu_temp_max = sampler.column("temp_c")
        r.gpu_power_max = sampler.column("power_w")
        r.gpu_throttle = sampler.column("throttle")
        r.gen_times, r.gen_stats, r.hdf5 = parsed["gens"], parsed["stats"], parsed["hdf5"]
        r.gpu_after = gpu_snapshot()
        r.exit_code = rc
        if rc is None:
            r.status = "interrupted"
        elif rc != 0:
            r.status = f"failed (exit code {rc})"
        elif not parsed["done"]:
            r.status = "failed (exit code 0 but no results written)"
        else:
            r.status = "ok"
        if r.hdf5:
            src = Path(r.hdf5) if Path(r.hdf5).is_absolute() else ROOT / r.hdf5
            if src.is_file():
                (report_dir / "results").mkdir(exist_ok=True)
                dst = report_dir / "results" / f"{r.order:02d}_{_slug(r.config)}.h5"
                shutil.copy2(src, dst)
                r.hdf5_copy = str(dst.relative_to(report_dir))
            else:
                r.hdf5_copy = UNAVAILABLE + f"result file not found: {src}"
        if r.config in references:
            r.matches_reference = (compare_to_reference(r.hdf5, references[r.config])
                                   if r.status == "ok" and r.hdf5
                                   else UNAVAILABLE + "the run did not finish")
        write_reports(report_dir, results)
        if rc is None:
            print("\n[run_selected] Ctrl-C: the queue is stopped.", flush=True)
            break
    return results, report_dir


def print_summary(results, report_dir):
    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    print(f"{'#':>2}  {'config':40s} {'status':14s} {'wall':>8s} {'gen 0':>8s} {'gens 1+':>8s} "
          f"{'us/tick':>8s} {'VRAM MiB':>9s} {'all runs':>9s}")
    for r in results:
        t = [g[0] for g in r.gen_times]
        mem = memory_summary(r.gen_stats)
        need = mem["vram_needed_all_runs_est_mib"]
        print(f"{r.order:>2}  {r.config[:40]:40s} {r.status[:14]:14s} "
              f"{(f'{r.wall_s:.0f} s' if r.wall_s is not None else '-'):>8s} "
              f"{(f'{t[0]:.1f} s' if t else '-'):>8s} "
              f"{(f'{sum(t[1:]) / len(t[1:]):.1f} s' if len(t) > 1 else '-'):>8s} "
              f"{us_per_run_tick_gens_1plus(r) or '-':>8s} "
              f"{str(mem['vram_peak_mib']) or '-':>9s} "
              f"{(f'~{need}' if need != '' else 'fit' if mem['runs_queued'] == 'no' else '-'):>9s}")
    print("\n'gen 0' includes compiling; 'gens 1+' is the mean of the later generations;"
          "\n'us/tick' is microseconds per worm-tick over generations 1+; 'VRAM MiB' is the"
          "\npeak torch held; 'all runs' is 'fit' if every run fit at once, otherwise the"
          "\nestimated MiB that would need (the runs were queued).")
    for r in results:
        # The power cap is normal under full load (boost runs up to it); thermal and
        # hardware slowdowns are what makes a card slower than it should be.
        reasons = set(r.gpu_throttle.replace("|", "+").split("+")) & {
            "hw_slowdown", "sw_thermal_slowdown", "hw_thermal_slowdown", "hw_power_brake_slowdown"}
        if reasons and not r.gpu_throttle.startswith(UNAVAILABLE):
            print(f"\nWARNING: {r.config} ran with the GPU slowed down ({r.gpu_throttle}; SM clock "
                  f"{r.gpu_sm_clock} MHz, up to {r.gpu_temp_max} C); its times are slower "
                  f"than this card can do.")
    print(f"\nSend this folder: {report_dir}\n")


# ============================================================
# The selection screen
# ============================================================

class Selection:
    """Cursor and ticks, independent of curses so it can be tested with scripted keys.

    Keys: "up", "down", "space" (tick/untick), "a" (tick all; again: untick all),
    "enter" (run), "q" (quit). Tick order is run order."""

    def __init__(self, infos, preticked=(), message=""):
        self.infos = infos
        self.cursor = 0
        self.order: List[int] = [infos.index(i) for i in preticked if i in infos and i.selectable]
        self.message = message

    def key(self, k):
        self.message = ""
        n = len(self.infos)
        if k == "up":
            self.cursor = (self.cursor - 1) % n
        elif k == "down":
            self.cursor = (self.cursor + 1) % n
        elif k == "space":
            i, info = self.cursor, self.infos[self.cursor]
            if i in self.order:
                self.order.remove(i)
            elif info.selectable:
                self.order.append(i)
            else:
                self.message = f"{info.name} cannot run here: {'; '.join(info.reasons)}"
        elif k == "a":
            can = [i for i, c in enumerate(self.infos) if c.selectable]
            if can and all(i in self.order for i in can):
                self.order = []
            else:
                self.order += [i for i in can if i not in self.order]
        elif k == "enter":
            if not self.order:
                self.message = "Nothing selected: tick configs with space first."
            else:
                return "run"
        elif k == "q":
            return "quit"
        return None

    def chosen(self):
        return [self.infos[i] for i in self.order]


COLUMNS = ("variants", "runs", "gens", "genome", "device", "neurons")


def table_lines(infos, name_width):
    head = f"{'config':{name_width}s} " + " ".join(f"{c:>9s}" for c in COLUMNS)
    rows = [f"{c.name[:name_width]:{name_width}s} "
            + " ".join(f"{getattr(c, k):>9s}" for k in COLUMNS)
            + ("" if c.selectable else "   -- " + "; ".join(c.reasons)) for c in infos]
    return head, rows


def _screen(stdscr, sel, name_width):
    import curses
    curses.curs_set(0)
    try:
        curses.use_default_colors()
    except curses.error:
        pass
    keys = {curses.KEY_UP: "up", ord("k"): "up", curses.KEY_DOWN: "down", ord("j"): "down",
            ord(" "): "space", ord("a"): "a", ord("q"): "q",
            curses.KEY_ENTER: "enter", 10: "enter", 13: "enter"}
    head, rows = table_lines(sel.infos, name_width)
    top = 0
    while True:
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        body = max(1, h - 6)
        top = min(max(top, sel.cursor - body + 1), sel.cursor)

        def put(y, text, attr=0):
            if 0 <= y < h:
                try:
                    stdscr.addnstr(y, 0, text, max(1, w - 1), attr)
                except curses.error:
                    pass
        put(0, "Select the configs to run -- they run one after another, in the order "
               "you tick them.", curses.A_BOLD)
        put(1, "      " + head, curses.A_UNDERLINE)
        for y, i in enumerate(range(top, min(len(rows), top + body))):
            mark = f"[{sel.order.index(i) + 1:>2}]" if i in sel.order else "[  ]"
            attr = (curses.A_REVERSE if i == sel.cursor else 0) \
                | (0 if sel.infos[i].selectable else curses.A_DIM)
            put(2 + y, f"{mark}  {rows[i]}", attr)
        put(h - 3, sel.message, curses.A_BOLD)
        put(h - 2, f"{len(sel.order)} selected.   up/down: move   space: tick   "
                   "a: tick all   enter: run   q: quit")
        stdscr.refresh()
        k = keys.get(stdscr.getch())
        if k:
            r = sel.key(k)
            if r:
                return r


def preset_selection(infos, preset, devices):
    """A Selection pre-ticked with `preset`, with a message about it."""
    if preset is None:
        return Selection(infos)
    rows = preset_infos(preset, infos, devices)
    skipped = [r for r in rows if not r.selectable]
    msg = f"Pre-ticked: preset {preset.name} ({len(rows) - len(skipped)} configs). Enter runs it."
    if skipped:
        msg = (f"Preset {preset.name}: {len(skipped)} cannot run here and are not ticked: "
               + "; ".join(f"{r.name} ({r.reasons[0]})" for r in skipped))
    return Selection(infos, preticked=rows, message=msg)


def select_interactively(infos, preset=None, devices=("cpu",)):
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise SystemExit(
            "[run_selected] ERROR: the selection screen needs a terminal. With Docker, "
            "add -it:\n  docker run -it --rm --gpus all ... byte-sim:$TAG\n"
            "Without a terminal, name the configs: python -m simulate.run_selected "
            "--configs a,b  (list them with --list)")
    import curses
    sel = preset_selection(infos, preset, devices)
    name_width = min(42, max(len("config"), *(len(i.name) for i in infos)))
    if curses.wrapper(_screen, sel, name_width) == "quit":
        return []
    return sel.chosen()


# ============================================================
# Entry point
# ============================================================

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", help="comma-separated config names or YAML paths; "
                                      "runs them in this order without the screen")
    ap.add_argument("--preset", help=f"run a preset from {PRESET_DIR} without the screen")
    ap.add_argument("--list", action="store_true", help="print the config table and exit")
    ap.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR),
                    help=f"where the report folder is made (default: {DEFAULT_REPORT_DIR})")
    a = ap.parse_args(argv)

    probe = torch_probe()
    devices = available_devices(probe)
    infos = all_infos(devices)
    presets = load_presets()                 # a broken preset file stops here, loudly
    default = presets.get(DEFAULT_PRESET)
    if default is not None:
        preset_infos(default, infos, devices)    # every name must exist

    if a.list:
        head, rows = table_lines(infos, min(42, max(len(i.name) for i in infos)))
        print(head)
        print("\n".join(rows))
        return 0

    if a.configs and a.preset:
        raise SystemExit("[run_selected] ERROR: give --configs or --preset, not both")
    preset_used = None
    if a.preset:
        if a.preset not in presets:
            raise SystemExit(f"[run_selected] ERROR: no preset {a.preset!r} in {PRESET_DIR} "
                             f"(presets: {sorted(presets)})")
        preset_used = presets[a.preset]
        chosen = preset_infos(preset_used, infos, devices)
    elif a.configs:
        chosen = [resolve(e.strip(), infos, devices) for e in a.configs.split(",") if e.strip()]
    if a.preset or a.configs:
        bad = [c for c in chosen if not c.selectable]
        if bad:
            for c in bad:
                print(f"[run_selected] ERROR: {c.name} cannot run here: {'; '.join(c.reasons)}")
            print("[run_selected] Nothing was run.")
            return 2
    else:
        chosen = select_interactively(infos, default, devices)
        if not chosen:
            print("[run_selected] Nothing selected; nothing was run.")
            return 0
        preset_used = default

    names = [c.name for c in chosen]
    if preset_used is not None:
        same = names == preset_used.configs
        selection = (f"preset {preset_used.name} (unchanged)" if same
                     else f"preset {preset_used.name} (modified)")
    else:
        selection = "--configs" if a.configs else "manual"
    run_info = {"selection": selection, "selected": names,
                "preset_configs": preset_used.configs if preset_used else None}
    references = dict(preset_used.references) if preset_used else {}
    if default is not None:
        for k, v in default.references.items():
            references.setdefault(k, v)

    print("[run_selected] Run order:")
    for i, c in enumerate(chosen):
        print(f"  {i + 1}. {c.name}  ({c.variants} variants x {c.runs} runs, {c.gens} "
              f"generations, {c.genome}, {c.device}, {c.neurons} neurons)")
    if shutil.which("nvidia-smi") is None and any(c.device == "cuda" for c in chosen):
        print("[run_selected] note: nvidia-smi not found; GPU load will be recorded as "
              "unavailable.")
    print(f"[run_selected] Selection: {selection}")
    results, report_dir = run_queue(chosen, a.report_dir, probe, references=references,
                                    run_info=run_info)
    print_summary(results, report_dir)
    return 0 if all(r.status == "ok" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
