#!/usr/bin/env python3
"""End-to-end tests for the tensor evaluator inside simulate/run_ea.py (plan Step 6).

The headline test runs the real EA twice on pre-drawn randomness -- once with the
scalar evaluator, once with the tensor evaluator -- and requires every dataset in the
two HDF5 files to be identical: elite genomes, lifespan vectors, seeds, run seeds and
generation statistics, in every generation. Selection feeds each generation's lifespans
into the next generation's genomes, so any single difference anywhere would compound.

Config variants and outputs go to a temporary directory; nothing is written into the
repository.

Usage
-----
    python -m tests.test_ea_tensor            # everything (a few minutes)
    python -m tests.test_ea_tensor --fast     # skip the scalar-regression ea_drift run
"""

import contextlib
import copy
import glob
import io
import os
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

import mvb.simulation_API as api  # noqa: E402
from mvb_torch.adapter import MARKER, make_tensor_evaluator, validate_evaluator_cfg  # noqa: E402
from mvb_torch.generation import draw_seeds, make_simulation_rngs  # noqa: E402
from tests.ea_drift import _values_equal, _walk_attrs, _walk_datasets  # noqa: E402

BASE = ROOT / "configs" / "experiments" / "test_ea_predrawn.yaml"
_RESULTS = []


def check(name, condition):
    _RESULTS.append((name, bool(condition)))
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}")
    return bool(condition)


def base_cfg():
    with open(BASE) as f:
        return yaml.safe_load(f)


def tensor_block(device="cpu", dtype="float64", philox_rounds=None):
    """Evaluator block. Live mode needs philox_rounds (R1.1); pre-drawn mode refuses it.
    No width key: the width is chosen automatically (R3)."""
    blk = {"backend": "tensor", "device": device, "dtype": dtype}
    if philox_rounds is not None:
        blk["philox_rounds"] = philox_rounds
    return blk


def run_ea(cfg, tmp, tag):
    """Run simulate.run_ea on `cfg` in a fresh output folder. Returns (h5, stdout, rc)."""
    out_dir = Path(tmp) / tag
    out_dir.mkdir()
    cfg = copy.deepcopy(cfg)
    cfg["experiment"]["output_folder"] = str(out_dir) + "/"
    cfg_path = Path(tmp) / f"{tag}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    proc = subprocess.run(
        [sys.executable, "-m", "simulate.run_ea", "--config", str(cfg_path)],
        input="n\n", text=True, capture_output=True, cwd=ROOT, timeout=1800,
    )
    h5s = glob.glob(str(out_dir / "*.h5"))
    return (h5s[0] if len(h5s) == 1 else None), proc.stdout + proc.stderr, proc.returncode


def diff_h5(a, b, ignore_attr_prefix=("experiment_evaluator", "experiment_output_folder")):
    """Every dataset exactly; every attribute exactly except two that legitimately
    differ: the evaluator's own config keys, and the output folder (this test gives
    each run its own temp folder, and run_ea saves the config into the attributes)."""
    problems = []
    with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
        da, db = _walk_datasets(fa), _walk_datasets(fb)
        for k in sorted(set(da) ^ set(db)):
            problems.append(f"dataset only in one file: {k}")
        for k in sorted(set(da) & set(db)):
            ok, msg = _values_equal(da[k][()], db[k][()], False, 0.0, 0.0)
            if not ok:
                problems.append(f"dataset {k}: {msg}")
        aa, ab = _walk_attrs(fa), _walk_attrs(fb)
        for path in sorted(set(aa) & set(ab)):
            keys = set(aa[path]) | set(ab[path])
            for key in sorted(keys):
                if key.startswith(ignore_attr_prefix):
                    continue
                if key not in aa[path] or key not in ab[path]:
                    problems.append(f"attr {path}@{key} only in one file")
                    continue
                ok, msg = _values_equal(aa[path][key], ab[path][key], False, 0.0, 0.0)
                if not ok:
                    problems.append(f"attr {path}@{key}: {msg}")
        n_datasets = len(da)
    return problems, n_datasets


# ============================================================
# 1. The scalar path is untouched
# ============================================================

def test_scalar_untouched():
    print("\n[1] Scalar path untouched: ea_drift vs tests/refs/test_ea_ref.h5")
    proc = subprocess.run(
        [sys.executable, "-m", "tests.ea_drift", "--reference", "tests/refs/test_ea_ref.h5"],
        text=True, capture_output=True, cwd=ROOT, timeout=1800,
    )
    check("run_ea with no evaluator block is bit-identical to the reference",
          proc.returncode == 0 and "[PASS]" in proc.stdout)
    check("the scalar run never loaded the tensor evaluator", MARKER not in proc.stdout)


# ============================================================
# 2. End-to-end EA equivalence
# ============================================================

def test_end_to_end(tmp):
    print("\n[2] Whole EA run: scalar evaluator vs tensor evaluator, pre-drawn")
    cfg = base_cfg()
    R = cfg["experiment"]["n_runs"]
    ref, out_s, rc = run_ea(cfg, tmp, "scalar")
    if not check("scalar run completed", rc == 0 and ref is not None):
        print(out_s[-2000:])
        return
    check("scalar run did not use the tensor evaluator", MARKER not in out_s)

    # R3: the width is automatic (every run at once here); width invariance is tested
    # in-process (test_generation, test_scheduling), so one EA run suffices.
    for S in ("auto",):
        c = base_cfg()
        c["experiment"]["evaluator"] = tensor_block()
        h5, out_t, rc = run_ea(c, tmp, f"tensor_S{S}")
        if not check(f"tensor run S={S} completed", rc == 0 and h5 is not None):
            print(out_t[-2000:])
            continue
        n_gens = cfg["experiment"]["evolutionary_algorithm"]["num_generations"]
        calls = out_t.count(f"{MARKER} {cfg['experiment']['population_size']} genomes")
        check(f"S={S}: the tensor evaluator really ran, once per generation "
              f"({calls} calls for {n_gens} generations)", calls == n_gens)
        # The EA comparison cannot see summation order (threshold margins absorb a
        # 1e-16 difference), so assert the bit-exact contraction was actually chosen.
        check(f"S={S}: pre-drawn mode used the bit-exact `sequential` contraction",
              "mode=predrawn" in out_t and "contraction=sequential" in out_t)
        problems, n = diff_h5(ref, h5)
        check(f"S={S}: all {n} datasets identical to the scalar EA "
              f"({len(problems)} differences)", not problems)
        for msg in problems[:5]:
            print(f"      - {msg}")

    # An explicit `backend: scalar` must be the same as no block at all.
    c = base_cfg()
    c["experiment"]["evaluator"] = {"backend": "scalar"}
    h5, out, rc = run_ea(c, tmp, "explicit_scalar")
    problems, _ = diff_h5(ref, h5) if h5 else (["no output"], 0)
    check("`backend: scalar` is identical to omitting the block", rc == 0 and not problems)
    return ref


# ============================================================
# 3. Replay: pre-computed seeds through both evaluators
# ============================================================

def _eval_args(cfg, genomes, sim_seed=1):
    e, w, wm = cfg["experiment"], cfg["world"], cfg["worm"]
    phases = sorted(cfg["food"], key=lambda p: p["phase_from"])
    rn, rd, rw = make_simulation_rngs(sim_seed)
    args = (genomes, cfg, "unused", "unused", False, False, False, False, False, 4, 4,
            len(genomes), rn, rd, rw, wm["decisionmaking"]["version"], e["max_ticks"],
            e["n_runs"], w["grid_width"], w["grid_height"], w["start_pos"], wm["speed"],
            wm["energy_capacity"], wm["metabolic_rate"], wm["movement_cost"],
            wm["sensors"]["active"], phases[0], cfg["brain"])
    return args, dict(switch_phases=phases[1:], randomness_cfg=e["predrawn_randomness"])


def test_replay():
    print("\n[3] Replay mode: pre-computed seeds give identical lifespans")
    from mvb.genome.generate_genome_random import generate_random_genome
    cfg = base_cfg()
    genomes = [generate_random_genome(cfg, rng_seed=50 + i) for i in range(5)]
    seeds = draw_seeds(*make_simulation_rngs(99), 5, cfg["experiment"]["n_runs"])
    pre = {v: {"run_seeds": seeds.run_seeds, "noise_seeds": seeds.noise_seeds[v],
               "decision_seeds": seeds.decision_seeds[v]} for v in range(5)}
    args, kw = _eval_args(cfg, genomes)
    with contextlib.redirect_stdout(io.StringIO()):
        s_out, s_rs = api.eval_generation(*args, pre_computed_seeds_dict=pre, **kw)
        ev = make_tensor_evaluator(tensor_block(), cfg, kw["randomness_cfg"])
        args2, _ = _eval_args(cfg, genomes)
        t_out, t_rs = ev(*args2, pre_computed_seeds_dict=pre, **kw)
    check("run_seeds taken from the replay dict by both",
          np.array_equal(s_rs, t_rs) and np.array_equal(t_rs, seeds.run_seeds))
    check("lifespans identical for every genome",
          all(np.array_equal(s_out[v]["lifespan_vector"], t_out[v]["lifespan_vector"])
              for v in range(5)))
    check("lifespan dtype matches eval_generation's ('i4')",
          all(t_out[v]["lifespan_vector"].dtype == s_out[v]["lifespan_vector"].dtype
              for v in range(5)))
    check("returned seed arrays are the replayed ones",
          all(np.array_equal(t_out[v]["seeds_noise_all_runs"], seeds.noise_seeds[v])
              and np.array_equal(t_out[v]["seeds_decision_all_runs"],
                                 seeds.decision_seeds[v]) for v in range(5)))


# ============================================================
# 4. Live smoke runs
# ============================================================

def test_live_smoke(tmp):
    print("\n[4] Live mode: full EA runs on CPU and MPS (smoke, not equivalence)")
    devices = all_devices()
    for dev in devices:
        c = base_cfg()
        del c["experiment"]["predrawn_randomness"]
        c["experiment"]["evaluator"] = tensor_block(device=dev, dtype="float32",
                                                    philox_rounds=10)
        h5, out, rc = run_ea(c, tmp, f"live_{dev}")
        if not check(f"{dev}: live EA run completed", rc == 0 and h5 is not None):
            print(out[-2000:])
            continue
        with h5py.File(h5, "r") as f:
            stats = f["generation_stats"][()]
            L = f["elite_genomes/lifespans"][()]
        T = c["experiment"]["max_ticks"]
        check(f"{dev}: generation stats finite and positive for all "
              f"{len(stats)} generations",
              all(np.isfinite(stats[k]).all() and (stats[k] > 0).all()
                  for k in ("mean", "max")))
        check(f"{dev}: elite lifespans within [1, max_ticks]",
              bool(((L >= 1) & (L <= T)).all()))
        check(f"{dev}: reported mode=live contraction=einsum",
              "mode=live" in out and "contraction=einsum" in out)


# ============================================================
# 5. Configuration validation
# ============================================================

def test_validation(tmp):
    print("\n[5] Unsupported configurations fail loudly")
    cfg = base_cfg()
    rc_pre = cfg["experiment"]["predrawn_randomness"]

    def raises(name, exc, fn):
        try:
            fn()
        except exc:
            return check(name, True)
        except Exception as e:  # noqa: BLE001
            return check(f"{name} (wrong exception {type(e).__name__}: {e})", False)
        return check(f"{name} (no exception)", False)

    blk = tensor_block()
    raises("missing key", KeyError,
           lambda: validate_evaluator_cfg({k: v for k, v in blk.items()
                                           if k != "device"}, rc_pre))
    raises("unknown key (e.g. a second randomness switch)", KeyError,
           lambda: validate_evaluator_cfg({**blk, "mode": "live"}, rc_pre))
    # R3: the key was removed; an old config that still has it must be refused, loudly.
    try:
        validate_evaluator_cfg({**blk, "slots_per_genome": 30}, rc_pre)
        check("slots_per_genome (removed in R3) is refused", False)
    except KeyError as e:
        check("slots_per_genome (removed in R3) is refused, with the reason",
              "chosen automatically" in str(e))
    raises("bad dtype", ValueError,
           lambda: validate_evaluator_cfg({**blk, "dtype": "float16"}, rc_pre))
    raises("pre-drawn on float32", ValueError,
           lambda: validate_evaluator_cfg({**blk, "dtype": "float32"}, rc_pre))
    if torch.backends.mps.is_available():
        raises("pre-drawn on mps", ValueError,
               lambda: validate_evaluator_cfg({**blk, "device": "mps"}, rc_pre))
        raises("float64 on mps (live)", ValueError,
               lambda: validate_evaluator_cfg({**blk, "device": "mps", "philox_rounds": 10},
                                              None))

    # philox_rounds (R1.1): required in live mode, 7..10 only, refused in pre-drawn mode.
    live = tensor_block(dtype="float32", philox_rounds=10)
    raises("live mode without philox_rounds", KeyError,
           lambda: validate_evaluator_cfg({k: v for k, v in live.items()
                                           if k != "philox_rounds"}, None))
    for bad in (6, 11, 7.5, "10"):
        raises(f"philox_rounds = {bad!r}", ValueError,
               lambda bad=bad: validate_evaluator_cfg({**live, "philox_rounds": bad}, None))
    raises("philox_rounds in pre-drawn mode", KeyError,
           lambda: validate_evaluator_cfg({**blk, "philox_rounds": 10}, rc_pre))
    check("live mode accepts philox_rounds 7 and 10",
          validate_evaluator_cfg({**live, "philox_rounds": 7}, None)[2] == 7
          and validate_evaluator_cfg(live, None)[2] == 10)

    genomes_cfg = base_cfg()
    from mvb.genome.generate_genome_random import generate_random_genome
    g = [generate_random_genome(genomes_cfg, rng_seed=1)]
    args, kw = _eval_args(genomes_cfg, g)
    with contextlib.redirect_stdout(io.StringIO()):
        ev = make_tensor_evaluator(blk, genomes_cfg, rc_pre)
    # Tracking is supported since Step 7 (tests/test_io_tensor.py); visualisation is not.
    viz = list(args)
    viz[7] = True                           # VIZ_ENABLED
    raises("visualisation refused", NotImplementedError, lambda: ev(*viz, **kw))

    for tag, block, needle in [
        ("bad_backend", {"backend": "gpu"}, "backend must be"),
        ("scalar_extra", {"backend": "scalar", "device": "cpu"}, "other evaluator keys"),
    ]:
        c = base_cfg()
        c["experiment"]["evaluator"] = block
        _, out, rc = run_ea(c, tmp, tag)
        check(f"run_ea refuses {block}", rc != 0 and needle in out)


def main():
    fast = "--fast" in sys.argv
    print("=" * 70)
    print("tensor evaluator inside run_ea -- test suite")
    print("=" * 70)
    with tempfile.TemporaryDirectory() as tmp:
        if fast:
            print("\n[1] SKIPPED (--fast)")
        else:
            test_scalar_untouched()
        test_end_to_end(tmp)
        test_replay()
        test_live_smoke(tmp)
        test_validation(tmp)

    failed = [n for n, ok in _RESULTS if not ok]
    print("\n" + "=" * 70)
    if failed:
        print(f"FAILED {len(failed)}/{len(_RESULTS)}")
        for n in failed:
            print(f"  - {n}")
        return 1
    print(f"OK -- {len(_RESULTS)}/{len(_RESULTS)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
