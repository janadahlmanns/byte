#!/usr/bin/env python3
"""Single-run replay from the HDF5 file alone (plan Step 8, R1 -- the user's requirement).

1. Storage: an elite genome written by run_ea and read back by the replay loader is
   identical to the original, modulation weights included (they used to be stored as
   float32).
2. End to end, per device (CPU, and MPS when available), in LIVE mode:
   a small EA runs with the tensor evaluator; then run_batch's replay mode re-simulates
   the stored elites from nothing but the file -- all runs at once, and single runs on
   their own -- and every replayed lifespan must equal the stored one bit for bit.
3. Cross-device replay (a GPU file -- MPS or CUDA -- replayed on CPU) is reported, not asserted: float32
   rounding differs between device types, so it is only expected to agree statistically.

Usage
-----
    python -m tests.test_replay
"""

import copy
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.devices import accelerators, all_devices  # noqa: E402

from mvb.genome.generate_genome_from_file import generate_genome_from_file  # noqa: E402
from mvb.genome.generate_genome_mutate_simple import generate_genome_mutate_simple  # noqa: E402
from mvb.genome.generate_genome_random import generate_random_genome  # noqa: E402
from mvb_torch.adapter import MARKER  # noqa: E402
from simulate.run_ea import save_elite_to_hdf5  # noqa: E402

_RESULTS = []
PY = sys.executable


def check(name, condition):
    _RESULTS.append((name, bool(condition)))
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}")
    return bool(condition)


def load(name):
    with open(ROOT / "configs" / "experiments" / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def run_module(module, cfg, tmp, tag):
    path = Path(tmp) / f"{tag}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    proc = subprocess.run([PY, "-m", module, "--config", str(path)], cwd=ROOT, text=True,
                          capture_output=True, input="n\n", timeout=3600)
    return proc.returncode, proc.stdout + proc.stderr


# ============================================================
# 1. Storage round trip
# ============================================================

def test_storage(tmp):
    print("\n[1] An elite genome survives save_elite_to_hdf5 -> replay loader exactly")
    cfg = load("test_ea")
    rng = np.random.default_rng(3)
    genomes = []
    for i in range(6):
        g = generate_random_genome(cfg, rng_seed=50 + i)
        for _ in range(i % 3):
            g = generate_genome_mutate_simple(g, 0.1, 0.2, rng)
        genomes.append(g)
    path = Path(tmp) / "store.h5"
    with h5py.File(path, "w"):
        pass
    R = 3
    save_elite_to_hdf5(path, genomes, [np.zeros(R)] * 6, [np.arange(R)] * 6,
                       [np.arange(R)] * 6)
    n_lossy = sum(float(np.float32(w)) != w for g in genomes
                  for mods in g.modulation_spec.values() for _, w in mods)
    ok = True
    for e, g in enumerate(genomes):
        back = generate_genome_from_file(cfg, elite_id=e, hdf5_path=str(path))
        ok &= np.array_equal(back["connection_weights"], g["connection_weights"])
        ok &= np.array_equal(back["tonic_activations"], g["tonic_activations"])
        ok &= back["eta"] == g["eta"]
        ok &= ({k: list(v) for k, v in back["modulation_spec"].items()}
               == {k: [(int(m), float(w)) for m, w in v] for k, v in g.modulation_spec.items()})
    check(f"6 genomes identical after the round trip, modulation weights included "
          f"({n_lossy} of them would have been rounded by the old float32 storage)",
          ok and n_lossy > 0)


# ============================================================
# 2. End-to-end replay, live mode
# ============================================================

def make_ea(tmp, device, tag, rounds=10):
    cfg = load("test_ea")
    x = cfg["experiment"]
    x["output_folder"] = str(Path(tmp) / tag) + "/"
    x["population_size"] = 40
    x["evolutionary_algorithm"]["num_generations"] = 3
    x["evaluator"] = {"backend": "tensor", "device": device, "dtype": "float32",
                      "philox_rounds": rounds}
    rc, out = run_module("simulate.run_ea", cfg, tmp, f"ea_{tag}")
    files = glob.glob(str(Path(tmp) / tag / "*.h5"))
    return (files[0] if rc == 0 and len(files) == 1 else None), out


def replay(src, device, genome_ids, runs, tmp, tag, rounds=10):
    """run_batch replay mode on `src`; returns ({elite_id: lifespans}, output, returncode)."""
    cfg = load("genome_from_file")
    x = cfg["experiment"]
    x.update(viz_enabled=False, viz_brain_enabled=False, enable_per_run_tracking=True,
             enable_per_tick_tracking=False, enable_heat_map_tracking=False)
    x["from_file_source"].update(file_folder=str(Path(src).parent), filename=Path(src).stem,
                                 genome_ID=list(genome_ids), runs_to_load=runs)
    x["evaluator"] = {"backend": "tensor", "device": device, "dtype": "float32",
                      "philox_rounds": rounds}
    before = set(glob.glob(str(Path(src).parent / "*.h5")))
    rc, out = run_module("simulate.run_batch", cfg, tmp, tag)
    new = [p for p in glob.glob(str(Path(src).parent / "*.h5")) if p not in before]
    res = {}
    if rc == 0 and len(new) == 1:
        with h5py.File(new[0], "r") as f:
            for i, e in enumerate(genome_ids):
                res[e] = f[f"variant_{i}/summary"]["lifetime_ticks"][:].astype(np.int64)
    for p in new:
        os.remove(p)
    return res, out, rc


def test_replay(tmp, device):
    print(f"\n[2] Live EA on {device}, then replay from the HDF5 alone")
    src, out = make_ea(tmp, device, f"ea_{device}")
    if not check(f"[{device}] live tensor EA produced a file", src is not None):
        print(out[-2000:])
        return None
    check(f"[{device}] the EA used keyed live randomness",
          "mode=live" in out and f"rng=philox4x32-" in out)
    with h5py.File(src, "r") as f:
        stored = f["elite_genomes/lifespans"][:].astype(np.int64)
    n_el, R = stored.shape

    got, rout, _ = replay(src, device, range(n_el), "all", tmp, f"rep_all_{device}")
    if not check(f"[{device}] replay ran through the tensor evaluator",
                 bool(got) and MARKER in rout and "mode=live" in rout):
        print(rout[-2000:])
        return src
    bad = sum(int((got[e] != stored[e]).sum()) for e in range(n_el))
    check(f"[{device}] all {n_el} elites x {R} runs replayed together: {bad} lifespans differ",
          bad == 0)

    picks = [(0, 0), (0, R - 1), (n_el // 2, 1), (n_el - 1, R // 2), (n_el - 1, R - 1)]
    single_bad = []
    for e, r in picks:
        res, _, _ = replay(src, device, [e], [r], tmp, f"rep_{device}_{e}_{r}")
        if not res or int(res[e][0]) != int(stored[e, r]):
            single_bad.append((e, r, None if not res else int(res[e][0]), int(stored[e, r])))
    check(f"[{device}] {len(picks)} single runs replayed ALONE (batch of one): "
          f"{len(single_bad)} differ {single_bad if single_bad else ''}", not single_bad)
    print(f"      stored lifespans span {stored.min()}..{stored.max()} ticks")
    return src


def test_rounds(tmp):
    print("\n[4] philox_rounds (R1.1): recorded in the file, honoured and checked by replay")
    src, out = make_ea(tmp, "cpu", "ea_r7", rounds=7)
    if not check("7-round live EA produced a file", src is not None):
        print(out[-2000:])
        return
    with h5py.File(src, "r") as f:
        stored = f["elite_genomes/lifespans"][:].astype(np.int64)
        attr = f.attrs.get("experiment_evaluator_philox_rounds", None)
    check(f"the EA printed rng=philox4x32-7 and the file records philox_rounds = {attr}",
          "rng=philox4x32-7" in out and attr is not None and int(attr) == 7)
    n_el, R = stored.shape
    got, _, _ = replay(src, "cpu", range(n_el), "all", tmp, "rep_r7_all", rounds=7)
    bad = sum(int((got[e] != stored[e]).sum()) for e in range(n_el)) if got else -1
    check(f"replayed with 7 rounds: all {n_el} x {R} runs identical ({bad} differ)", bad == 0)
    one, _, _ = replay(src, "cpu", [n_el - 1], [R - 1], tmp, "rep_r7_one", rounds=7)
    check("one run replayed alone with 7 rounds: identical",
          bool(one) and int(one[n_el - 1][0]) == int(stored[n_el - 1, R - 1]))
    _, out10, rc10 = replay(src, "cpu", [0], [0], tmp, "rep_r7_as_10", rounds=10)
    check("replaying the 7-round file with philox_rounds: 10 is refused, naming the count",
          rc10 != 0 and "philox_rounds=7" in out10)
    legacy = Path(tmp) / "legacy" / "no_rounds.h5"
    legacy.parent.mkdir()
    shutil.copy(src, legacy)
    with h5py.File(legacy, "a") as f:
        del f.attrs["experiment_evaluator_philox_rounds"]
    _, outl, rcl = replay(str(legacy), "cpu", [0], [0], tmp, "rep_legacy", rounds=10)
    check("a tensor live file that does not record its round count is refused",
          rcl != 0 and "does not record" in outl)


def report_cross_device(src_gpu, tmp, dev):
    print(f"\n[3] Cross-device replay (reported, not asserted): {dev} file replayed on CPU")
    with h5py.File(src_gpu, "r") as f:
        stored = f["elite_genomes/lifespans"][:].astype(np.int64)
    got, _, _ = replay(src_gpu, "cpu", range(stored.shape[0]), "all", tmp, f"rep_cross_{dev}")
    if got:
        diff = sum(int((got[e] != stored[e]).sum()) for e in range(stored.shape[0]))
        print(f"      {diff} of {stored.size} runs differ "
              f"(float32 rounding differs between device types)")


def main():
    print("=" * 70)
    print("Single-run replay from the HDF5 (plan Step 8, R1)")
    print("=" * 70)
    devices = all_devices()
    with tempfile.TemporaryDirectory() as tmp:
        test_storage(tmp)
        srcs = {d: test_replay(tmp, d) for d in devices}
        test_rounds(tmp)
        for dev in accelerators():
            if srcs.get(dev):
                report_cross_device(srcs[dev], tmp, dev)
    failed = [nm for nm, ok in _RESULTS if not ok]
    print("\n" + "=" * 70)
    if failed:
        print(f"FAILED {len(failed)}/{len(_RESULTS)}")
        for nm in failed:
            print(f"  - {nm}")
        return 1
    print(f"OK -- {len(_RESULTS)}/{len(_RESULTS)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
