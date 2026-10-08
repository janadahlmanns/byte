#!/usr/bin/env python3
"""Tests for tracking in the tensor evaluator (plan_evotorch.md Step 7).

The headline test (T1) runs the real `simulate.run_batch` twice on pre-drawn
randomness -- scalar evaluator, then tensor evaluator -- with every tracking flag on,
and requires the two HDF5 files to be identical: every dataset, every field, every
attribute except the evaluator's own config keys and the output folder. Config
variants pin the recorder quirks Q1 (movement at the grid edge) and Q2 (manhattan
axes), sensors that the brain does not read, regrowth and phase switches, and the
tracking-flag combinations.

Config variants and outputs go to a temporary directory; nothing is written into the
repository.

Usage
-----
    python -m tests.test_io_tensor            # everything
    python -m tests.test_io_tensor --fast     # skip the scalar-regression ea_drift run
"""

import copy
import glob
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

from mvb_torch.adapter import MARKER  # noqa: E402
from mvb_torch.decision import build_output_spec  # noqa: E402
from mvb_torch.generation import eval_generation_batch, sim_config_from_yaml  # noqa: E402
from mvb_torch.genome_codec import encode_genomes  # noqa: E402
from mvb_torch import tracking as trk  # noqa: E402
from tests.test_ea_tensor import diff_h5  # noqa: E402
from tests.test_generation import genome_pool, seeds_for  # noqa: E402

PREDRAWN = ROOT / "configs" / "experiments" / "test_predrawn.yaml"
BEHAVIOUR = ROOT / "configs" / "experiments" / "test_behaviour.yaml"
MAX_BRAIN_TICKS = 32
_RESULTS = []


def check(name, condition):
    _RESULTS.append((name, bool(condition)))
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}")
    return bool(condition)


def load(path):
    with open(path) as f:
        return yaml.safe_load(f)


def tensor_block(device="cpu", dtype="float64", philox_rounds=None):
    """Evaluator block. Live mode needs philox_rounds (R1.1); pre-drawn mode refuses it.
    No width key: the width is chosen automatically (R3)."""
    blk = {"backend": "tensor", "device": device, "dtype": dtype}
    if philox_rounds is not None:
        blk["philox_rounds"] = philox_rounds
    return blk


def run_batch(cfg, tmp, tag):
    """Run simulate.run_batch on `cfg` in a fresh output folder. Returns (h5, output)."""
    out_dir = Path(tmp) / tag
    out_dir.mkdir()
    cfg = copy.deepcopy(cfg)
    cfg["experiment"]["output_folder"] = str(out_dir) + "/"
    cfg_path = Path(tmp) / f"{tag}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    proc = subprocess.run(
        [sys.executable, "-m", "simulate.run_batch", "--config", str(cfg_path)],
        input="n\n", text=True, capture_output=True, cwd=ROOT, timeout=1800,
    )
    h5s = glob.glob(str(out_dir / "*.h5"))
    return (h5s[0] if len(h5s) == 1 else None), proc.stdout + proc.stderr


def dataset_names(path):
    with h5py.File(path, "r") as f:
        names = []
        f.visititems(lambda n, o: names.append(n) if isinstance(o, h5py.Dataset) else None)
    return sorted(names)


# ============================================================
# T1. Exact file equality, pre-drawn mode
# ============================================================

def _variant_edge(cfg):
    # Q1: a 7x7 world, so worms reach the edge and wrap within their ~30-tick life.
    cfg["world"].update(grid_width=7, grid_height=7, start_pos=[3, 3])


def _variant_asymmetric(cfg):
    # Q2: with start_pos [5, 40] the recorder's (y, x) start differs from the worm's.
    cfg["world"]["start_pos"] = [5, 40]


def _variant_regrow(cfg):
    cfg["food"] = [
        {"phase_from": 0, "feeding_paradigm": {"initial": True, "regrow": True},
         "initial_fraction_per_cell": 0.3, "regrow_time": 5},
        {"phase_from": 15, "feeding_paradigm": {"initial": False, "regrow": True},
         "initial_fraction_per_cell": 0.3, "regrow_time": 3},
    ]


def _variant_current_field_only(cfg):
    cfg["worm"]["sensors"]["active"] = ["current_field"]


def _variant_unmapped_sensor(cfg):
    # adjacent_binary stays active, but the brain is not fed food_north: the recorder
    # must still see it (plan D4).
    del cfg["brain"]["sensory_mapping"]["food_north"]


def _flags(per_run, per_tick, heat):
    def f(cfg):
        cfg["experiment"].update(enable_per_run_tracking=per_run,
                                 enable_per_tick_tracking=per_tick,
                                 enable_heat_map_tracking=heat)
    return f


T1_VARIANTS = [
    ("base (test_predrawn: phase switch with regrow at tick 40)", None),
    ("Q1: 7x7 grid, edge wrapping", _variant_edge),
    ("Q2: asymmetric start_pos [5, 40]", _variant_asymmetric),
    ("regrow from tick 0 + initial:false switch at 15", _variant_regrow),
    ("only current_field active", _variant_current_field_only),
    ("adjacent_binary active but food_north unmapped", _variant_unmapped_sensor),
    ("flags: per-run only", _flags(True, False, False)),
    ("flags: per-run + heat map", _flags(True, False, True)),
    ("flags: per-run + per-tick", _flags(True, True, False)),
]


def test_exact_files(tmp):
    print("\n[T1] run_batch, pre-drawn: tensor file == scalar file")
    for i, (name, mutate) in enumerate(T1_VARIANTS):
        cfg = load(PREDRAWN)
        if mutate is not None:
            mutate(cfg)
        a, out_a = run_batch(cfg, tmp, f"t1_{i}_scalar")
        t = copy.deepcopy(cfg)
        t["experiment"]["evaluator"] = tensor_block()
        b, out_b = run_batch(t, tmp, f"t1_{i}_tensor")
        if a is None or b is None:
            check(f"{name}: both runs produced a file", False)
            print((out_a if a is None else out_b)[-1500:])
            continue
        problems, n = diff_h5(a, b)
        for p in problems[:5]:
            print(f"      {p}")
        check(f"{name}: {n} datasets identical, tensor path ran",
              not problems and n > 0 and MARKER in out_b)


# ============================================================
# T2. Live mode: schema, genome-level datasets and per-run invariants
# ============================================================

def _live_cfg():
    cfg = load(BEHAVIOUR)
    cfg["experiment"].update(population_size=30, n_runs=10)
    return cfg


def _invariants(path, cfg):
    """Per-run consistency of the tracked data. Returns a list of problems."""
    problems = []
    cap = cfg["worm"]["energy_capacity"]
    H, W = cfg["world"]["grid_height"], cfg["world"]["grid_width"]
    with h5py.File(path, "r") as f:
        for v in range(cfg["experiment"]["population_size"]):
            summ = f[f"variant_{v}/summary"][()]
            for r in range(cfg["experiment"]["n_runs"]):
                pt = f[f"variant_{v}/run_{r}/per_tick"][()]
                hm = f[f"variant_{v}/run_{r}/staying"][()]
                L = int(summ["lifetime_ticks"][r])
                tag = f"variant {v} run {r}"
                if len(pt) != L + 1:
                    problems.append(f"{tag}: {len(pt)} rows for lifetime {L}")
                    continue
                if not np.array_equal(pt["tick"], np.arange(L + 1)):
                    problems.append(f"{tag}: ticks not 0..L")
                if pt["energy"][0] != cap or pt["energy"][-1] != summ["final_energy"][r]:
                    problems.append(f"{tag}: energy endpoints")
                if int(pt["food_consumed"].sum()) != int(summ["foods"][r]):
                    problems.append(f"{tag}: food_consumed sum != foods")
                if int((pt["movement"] != b"stay").sum()) != int(summ["distance"][r]):
                    problems.append(f"{tag}: moves != distance")
                if int(hm.sum()) != L + 1 or hm.shape != (H, W):
                    problems.append(f"{tag}: heat map sum/shape")
                if pt["decision_made"][0] != 0 or not np.all(pt["decision_made"][1:] == 1):
                    problems.append(f"{tag}: decision_made")
    return problems


GENOME_LEVEL = ("run_seeds", "genome_properties", "modulation", "eta", "tonic_activations")


def test_live(tmp, devices):
    print("\n[T2] run_batch, live: same schema and genome-level data as the scalar; "
          "per-run invariants hold")
    cfg = _live_cfg()
    a, out_a = run_batch(cfg, tmp, "t2_scalar")
    if not check("scalar live run produced a file", a is not None):
        print(out_a[-1500:])
        return
    names_a = dataset_names(a)
    check("scalar file passes the invariants (sanity of the invariants themselves)",
          not _invariants(a, cfg))
    for device in devices:
        t = copy.deepcopy(cfg)
        t["experiment"]["evaluator"] = tensor_block(device=device, dtype="float32",
                                                    philox_rounds=10)
        b, out_b = run_batch(t, tmp, f"t2_tensor_{device}")
        if not check(f"[{device}] tensor live run produced a file", b is not None):
            print(out_b[-1500:])
            continue
        check(f"[{device}] tensor path ran in live mode",
              MARKER in out_b and "mode=live" in out_b)
        check(f"[{device}] identical dataset names ({len(names_a)})",
              dataset_names(b) == names_a)
        bad = []
        with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
            for n in names_a:
                da, db = fa[n], fb[n]
                if da.dtype != db.dtype:
                    bad.append(f"dtype {n}")
                base = n.split("/")[-1]
                if base in GENOME_LEVEL and not np.array_equal(da[()], db[()]):
                    bad.append(f"value {n}")
                if base == "wiring":
                    for fld in ("src", "tgt", "weight_initial", "reliability"):
                        if not np.array_equal(da[fld], db[fld]):
                            bad.append(f"value {n}.{fld}")
                if base == "summary":
                    for fld in ("run_id", "seed_noise", "seed_decision"):
                        if not np.array_equal(da[fld], db[fld]):
                            bad.append(f"value {n}.{fld}")
        for x in bad[:5]:
            print(f"      {x}")
        check(f"[{device}] every dtype identical; genome-level datasets, wiring "
              f"preconditions and summary seeds equal", not bad)
        inv = _invariants(b, cfg)
        for x in inv[:5]:
            print(f"      {x}")
        check(f"[{device}] per-run invariants hold on the tensor file", not inv)


# ============================================================
# T3 / T4. In-process: tracking does not perturb; schedule invariance
# ============================================================

def _run(cfg, genomes, seeds, *, S, F=64, mode="predrawn", flags=None):
    bcfg = cfg["brain"]
    sim = sim_config_from_yaml(cfg)
    batch = encode_genomes(genomes, bcfg)
    tracker = None
    if flags is not None:
        tracker = trk.Tracker(genomes, flags, sim.n_runs,
                              min(len(genomes) * S, len(genomes) * sim.n_runs),
                              sim.active_sensors,
                              batch.device, batch.dtype, flush_every=F)
    res = eval_generation_batch(batch, build_output_spec(bcfg), sim, seeds,
                                width=min(len(genomes) * S, len(genomes) * sim.n_runs),
                                mode=mode,
                                max_brain_ticks=MAX_BRAIN_TICKS, tracker=tracker,
                                philox_rounds=10 if mode == "live" else None)
    out = {k: getattr(res, k).cpu().numpy()
           for k in ("lifespans", "eats", "distance", "final_energy")}
    if tracker is not None:
        P, R = len(genomes), sim.n_runs
        out["final_w"] = tracker.final_w[: P * R].cpu().numpy()
        for v, r, ints, w in tracker.per_run_rows():
            out[f"pt_{v}_{r}"] = trk.build_per_tick(ints, w, genomes[v], tracker.conns[v],
                                                    sim.start_pos, sim.energy_capacity)
            out[f"hm_{v}_{r}"] = trk.build_heat_map(ints, sim.start_pos, sim.height,
                                                    sim.width)
    return out


def test_no_perturbation():
    print("\n[T3] tracking on == tracking off (it only reads state)")
    cfg = load(PREDRAWN)
    cfg["experiment"]["n_runs"] = 6
    genomes = genome_pool(cfg["brain"])
    seeds = seeds_for(cfg, len(genomes))
    all_on = trk.TrackingFlags.effective(True, True, True)
    keys = ["lifespans", "eats", "distance", "final_energy"]
    for mode in ("predrawn", "live"):
        off = _run(cfg, genomes, seeds, S=3, mode=mode)
        on = _run(cfg, genomes, seeds, S=3, mode=mode, flags=all_on)
        diff = [k for k in keys if not np.array_equal(off[k], on[k])]
        check(f"{mode}: lifespans, eats, distance, energy identical", not diff)


def test_schedule_invariance():
    print("\n[T4] tracked data independent of S (slots) and F (flush interval)")
    cfg = load(PREDRAWN)
    cfg["experiment"]["n_runs"] = 6
    genomes = genome_pool(cfg["brain"])
    seeds = seeds_for(cfg, len(genomes))
    all_on = trk.TrackingFlags.effective(True, True, True)
    ref = _run(cfg, genomes, seeds, S=6, F=64, flags=all_on)
    n_pt = sum(k.startswith("pt_") for k in ref)
    check(f"reference has per-tick data for all {len(genomes) * 6} runs",
          n_pt == len(genomes) * 6)
    for S, F in ((1, 1), (1, 7), (3, 7), (3, 64), (6, 1)):
        got = _run(cfg, genomes, seeds, S=S, F=F, flags=all_on)
        diff = [k for k in ref if k not in got or not np.array_equal(ref[k], got[k])]
        check(f"S={S}, F={F}: identical ({len(ref)} arrays)",
              not diff and set(got) == set(ref))


# ============================================================
# T5 / T6. Scalar path untouched; flags and validation
# ============================================================

def test_scalar_untouched():
    print("\n[T5] run_batch without an evaluator block is unchanged")
    proc = subprocess.run(
        [sys.executable, "-m", "tests.ea_drift", "--reference",
         "tests/refs/test_behaviour_ref.h5", "--runner", "simulate.run_batch",
         "--config", "test_behaviour", "--sim-name", "test_behaviour",
         "--mode", "behaviour"],
        text=True, capture_output=True, cwd=ROOT, timeout=1800,
    )
    check("ea_drift --mode behaviour vs test_behaviour_ref.h5 passes",
          proc.returncode == 0 and "[PASS]" in proc.stdout)
    check("the scalar run never loaded the tensor evaluator", MARKER not in proc.stdout)
    probe = subprocess.run(
        [sys.executable, "-c",
         "import sys, simulate.run_batch; print('torch' in sys.modules)"],
        text=True, capture_output=True, cwd=ROOT, timeout=300,
    )
    check("importing simulate.run_batch does not import torch",
          probe.stdout.strip() == "False")


def test_flags():
    print("\n[T6] effective tracking flags follow eval_variant's nesting")
    e = trk.TrackingFlags.effective
    check("per-run off => nothing tracked", e(False, True, True) == e(False, False, False)
          and not e(False, True, True).any)
    check("per-run on keeps the detail flags",
          e(True, True, False) == trk.TrackingFlags(True, True, False))


def main():
    fast = "--fast" in sys.argv
    devices = all_devices()
    with tempfile.TemporaryDirectory() as tmp:
        test_exact_files(tmp)
        test_live(tmp, devices)
    test_no_perturbation()
    test_schedule_invariance()
    test_flags()
    if not fast:
        test_scalar_untouched()
    n_fail = sum(not ok for _, ok in _RESULTS)
    print(f"\n{len(_RESULTS) - n_fail}/{len(_RESULTS)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
