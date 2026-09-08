#!/usr/bin/env python3
"""Simulation drift regression test: run a deterministic simulation and compare
its HDF5 output against a known-good reference file.

By default this runs `simulate.run_ea --config test_ea`, but the runner, config,
and output simulation_name are all overridable (see --runner/--config/--sim-name)
so the same harness can guard other deterministic entrypoints — e.g.
`simulate.run_batch` with per-run tracking enabled.

The targeted runs are fully deterministic (fixed simulation_seed, per-variant
seeds drawn in variant_id order), so a correct refactor must produce a
bit-identical HDF5 file. This script guards that invariant while the parallelism
is optimized.

Usage
-----
Generate a reference from the current (known-good) code, once:
    python -m tests.ea_drift --generate-reference tests/refs/test_ea_ref.h5

Compare the current code against that reference (default mode):
    python -m tests.ea_drift --reference tests/refs/test_ea_ref.h5

Drive a different runner/config (must match the reference it's compared to):
    python -m tests.ea_drift --reference tests/refs/test_batch_tracking_ref.h5 \
        --runner simulate.run_batch --config test_batch_tracking \
        --sim-name test_batch_tracking

Check only that the worm BEHAVED the same, ignoring float drift in the weights:
    python -m tests.ea_drift --reference tests/refs/test_behaviour_ref.h5 \
        --runner simulate.run_batch --config test_behaviour \
        --sim-name test_behaviour --mode behaviour

Useful flags:
    --mode MODE      exact (default) or behaviour. See "Two modes" below.
    --runner MODULE  Module to run with -m (default: simulate.run_ea).
    --config NAME    Experiment config name (default: test_ea).
    --sim-name NAME  experiment.simulation_name in the config, used to locate the
                     produced H5 (default: test_task_switching).
    --tol            Compare floats with np.allclose instead of exact equality.
    --rtol / --atol  Tolerances used with --tol (defaults: 1e-7 / 0.0).
    --keep           Keep the HDF5 file produced by the run (default: delete it).
    --timeout SEC    Subprocess timeout in seconds (default: 600).

Two modes
---------
exact (default)
    Every dataset and attribute must be bit-identical. This is the strongest
    guarantee and the right default, but it is deliberately broken by the
    float64/math.tanh refactors (commits 740a3413dd, 3dfcd3167b), which changed
    the last bits of the weight trajectory on purpose.

behaviour
    Asks the different question: did those refactors change what the worm DID?
    Compares the per-tick decision record -- sensed food, chosen movement,
    food consumed, energy, distance, whether a decision was made -- plus the
    occupancy heat map and the per-run outcome summary, all exactly, while
    ignoring the tracked connection-weight columns where the drift lives.

    Genomes and seeds are checked first as a precondition: if they differ, the
    two runs simulated different animals and the script aborts rather than
    reporting a behaviour verdict it cannot support.

    Requires enable_per_tick_tracking in the config, so it does NOT work against
    test_ea_ref.h5 (that config has tracking off and stores no decision record).
    Use configs/experiments/test_behaviour.yaml or test_behaviour_lookup.yaml.

Exit code is 0 when output matches the reference, 1 otherwise.
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import h5py

# Mirrors the chosen config's experiment.output_folder. The produced file is
# "{timestamp}_{simulation_name}.h5".
OUTPUT_DIR = Path("data/temp")

# Defaults reproduce the original EA drift test. Override via CLI to drive a
# different runner/config (e.g. run_batch with per-run tracking enabled).
DEFAULT_RUNNER = "simulate.run_ea"
DEFAULT_CONFIG = "test_ea"
DEFAULT_SIM_NAME = "test_task_switching"

# ============================================================
# Behaviour-mode field classification
# ============================================================
# In behaviour mode every field of the output falls into exactly one of three
# classes. The split exists because the float64 / math.tanh / arctanh refactors
# deliberately changed the last bits of the weight trajectory (see
# reproducibility.md) while leaving -- we hope -- every decision intact. Only the
# behavioural class decides pass/fail.
#
#   precondition  must be bit-identical or the comparison is MEANINGLESS, not
#                 merely failing: a different seed or genome means the two runs
#                 simulated different animals in different worlds. Aborts.
#   behavioural   the assertion. What the worm sensed, chose, ate, and where it
#                 went. Compared exactly -- no tolerance, ever.
#   numerical     connection weights. Reported, never asserted.

# per_tick columns that describe behaviour. Anything else in per_tick must look
# like a tracked connection weight ("<src>_<tgt>"); an unrecognised field is a
# loud failure rather than a silent addition to the ignored set, so that a new
# tracking column cannot quietly escape the assertion.
BEHAVIOURAL_PER_TICK_FIELDS = frozenset({
    "tick", "food_sensed_N", "food_sensed_E", "food_sensed_S", "food_sensed_W",
    "movement", "food_consumed", "energy", "manhattan_dist", "decision_made",
})
WEIGHT_COLUMN = re.compile(r"^\d+_\d+$")

# summary/ is per-run outcome. The seeds are inputs, not outcomes.
BEHAVIOURAL_SUMMARY_FIELDS = frozenset({
    "run_id", "lifetime_ticks", "foods", "distance", "final_energy",
})
PRECONDITION_SUMMARY_FIELDS = frozenset({"seed_noise", "seed_decision"})

# Datasets that define WHICH animal ran in WHICH world. Matched by basename.
PRECONDITION_DATASETS = frozenset({
    "run_seeds", "genome_properties", "modulation", "tonic_activations", "eta",
})

# `wiring` is MIXED and must be split field-wise: it stores the genome's initial
# wiring alongside the post-plasticity weight each run finished with. The latter
# is an output that carries the drift, so classifying the whole dataset as a
# precondition would abort every comparison for exactly the reason the mode
# exists to tolerate.
WIRING_PRECONDITION_FIELDS = frozenset({
    "src", "tgt", "weight_initial", "reliability",
})
WIRING_NUMERICAL_FIELD = re.compile(r"^weight_final_run_\d+$")


def _existing_outputs(sim_name: str) -> set:
    """Set of output H5 files for sim_name currently in OUTPUT_DIR."""
    if not OUTPUT_DIR.exists():
        return set()
    return set(OUTPUT_DIR.glob(f"*_{sim_name}.h5"))


def run_simulation(timeout: int, runner: str, config: str, sim_name: str) -> Path:
    """Run the simulation and return the path to the H5 file it produced.

    Identifies the new file by diffing the OUTPUT_DIR listing taken before and
    after the run, so pre-existing files in data/temp are not mistaken for it.
    """
    run_cmd = [sys.executable, "-m", runner, "--config", config]
    before = _existing_outputs(sim_name)
    print(f"[run] {' '.join(run_cmd)}")
    start = time.time()
    try:
        # Feed "n" to any "Display ... plot? (y/n)" prompt (run_ea) so the run
        # never blocks waiting on stdin; runners without the prompt ignore it.
        proc = subprocess.run(
            run_cmd,
            input="n\n",
            text=True,
            timeout=timeout,
            capture_output=True,
        )
    except subprocess.TimeoutExpired:
        print(f"[FAIL] run exceeded timeout of {timeout}s", file=sys.stderr)
        sys.exit(1)
    elapsed = time.time() - start
    print(f"[run] finished in {elapsed:.1f}s (exit {proc.returncode})")

    if proc.returncode != 0:
        print(f"[FAIL] {runner} exited non-zero. Captured output:", file=sys.stderr)
        sys.stderr.write(proc.stdout or "")
        sys.stderr.write(proc.stderr or "")
        sys.exit(1)

    new_files = _existing_outputs(sim_name) - before
    if len(new_files) == 0:
        print(f"[FAIL] no new *_{sim_name}.h5 produced in {OUTPUT_DIR}",
              file=sys.stderr)
        sys.exit(1)
    if len(new_files) > 1:
        # Fall back to newest by mtime, but warn — concurrent runs are unexpected.
        newest = max(new_files, key=lambda p: p.stat().st_mtime)
        print(f"[WARN] multiple new files {sorted(map(str, new_files))}; "
              f"using newest: {newest}")
        return newest
    produced = new_files.pop()
    print(f"[run] produced: {produced}")
    return produced


def _walk_datasets(h5_file):
    """Map of {path: dataset} for every dataset in the file."""
    found = {}
    h5_file.visititems(
        lambda name, obj: found.__setitem__(name, obj)
        if isinstance(obj, h5py.Dataset) else None
    )
    return found


def _walk_attrs(h5_file):
    """Map of {group_or_dataset_path: {attr: value}} including the root ('/')."""
    attrs = {"/": dict(h5_file.attrs)}
    h5_file.visititems(
        lambda name, obj: attrs.__setitem__(name, dict(obj.attrs))
    )
    return attrs


def _values_equal(a, b, use_tol, rtol, atol):
    """Compare two arrays/scalars; tolerance only applies to floating types."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False, f"shape {a.shape} != {b.shape}"
    if a.dtype != b.dtype:
        return False, f"dtype {a.dtype} != {b.dtype}"

    is_float = np.issubdtype(a.dtype, np.floating)
    # Structured dtypes (e.g. generation_stats, modulation_spec): compare field
    # by field so float fields can honor tolerance.
    if a.dtype.names:
        for field in a.dtype.names:
            ok, msg = _values_equal(a[field], b[field], use_tol, rtol, atol)
            if not ok:
                return False, f"field '{field}': {msg}"
        return True, ""

    if use_tol and is_float:
        if np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
            return True, ""
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        return False, f"max abs diff {np.nanmax(diff):.3e} exceeds tol"

    if np.array_equal(a, b, equal_nan=is_float):
        return True, ""
    if is_float:
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        return False, f"not bit-identical; max abs diff {np.nanmax(diff):.3e}"
    return False, "values differ"


# ============================================================
# Behaviour comparison
# ============================================================

def _first_true(mask):
    """Index of the first True in a 1-D boolean array, or None."""
    hits = np.flatnonzero(mask)
    return int(hits[0]) if hits.size else None


def _check_preconditions(fp, fr, mismatches, drift=None):
    """Verify the two runs simulated the same genomes in the same worlds.

    A failure here is not a behavioural difference -- it means the comparison
    itself is void, so the caller aborts instead of reporting a behaviour verdict.

    Numerical (drift-carrying) fields encountered along the way are folded into
    `drift`, a single-entry {"max": float} accumulator, so they are reported
    rather than silently skipped.
    """
    prod_ds, ref_ds = _walk_datasets(fp), _walk_datasets(fr)

    missing = sorted(set(ref_ds) - set(prod_ds))
    extra = sorted(set(prod_ds) - set(ref_ds))
    for path in missing:
        mismatches.append(f"dataset MISSING in produced: {path}")
    for path in extra:
        mismatches.append(f"dataset EXTRA in produced:   {path}")

    for path in sorted(set(prod_ds) & set(ref_ds)):
        base = path.rsplit("/", 1)[-1]
        if base in PRECONDITION_DATASETS:
            ok, msg = _values_equal(prod_ds[path][()], ref_ds[path][()],
                                    False, 0.0, 0.0)
            if not ok:
                mismatches.append(f"input DIFFERS {path}: {msg}")
        elif base == "wiring":
            a, b = prod_ds[path][()], ref_ds[path][()]
            if a.shape != b.shape or a.dtype.names != b.dtype.names:
                mismatches.append(f"input DIFFERS {path}: wiring layout changed")
                continue
            for field in a.dtype.names:
                if field in WIRING_PRECONDITION_FIELDS:
                    if not np.array_equal(a[field], b[field]):
                        mismatches.append(
                            f"input DIFFERS {path}.{field}: genome wiring changed")
                elif WIRING_NUMERICAL_FIELD.match(field):
                    # Post-plasticity weight: an output, not an input. Report only.
                    if drift is not None:
                        d = np.abs(a[field].astype(np.float64)
                                   - b[field].astype(np.float64))
                        if d.size:
                            drift["max"] = max(drift["max"], float(d.max()))
                else:
                    mismatches.append(
                        f"{path}: unrecognised wiring field '{field}'. Classify it "
                        f"in WIRING_PRECONDITION_FIELDS or WIRING_NUMERICAL_FIELD "
                        f"in tests/ea_drift.py -- refusing to guess.")
        elif base == "summary":
            a, b = prod_ds[path][()], ref_ds[path][()]
            if a.shape != b.shape:
                mismatches.append(f"input DIFFERS {path}: shape {a.shape} != {b.shape}")
                continue
            for field in PRECONDITION_SUMMARY_FIELDS & set(a.dtype.names):
                if not np.array_equal(a[field], b[field]):
                    mismatches.append(f"input DIFFERS {path}.{field}: seeds do not match")
    return mismatches


def _split_per_tick_fields(dtype, path):
    """Partition per_tick columns into (behavioural, numerical).

    Raises on an unrecognised column so a newly tracked field cannot silently
    land in the ignored bucket.
    """
    behavioural, numerical, unknown = [], [], []
    for name in dtype.names:
        if name in BEHAVIOURAL_PER_TICK_FIELDS:
            behavioural.append(name)
        elif WEIGHT_COLUMN.match(name):
            numerical.append(name)
        else:
            unknown.append(name)
    if unknown:
        raise ValueError(
            f"{path}: unrecognised per_tick field(s) {unknown}. Add them to "
            f"BEHAVIOURAL_PER_TICK_FIELDS in tests/ea_drift.py if they describe "
            f"behaviour, or confirm they are connection weights named <src>_<tgt>. "
            f"Refusing to guess -- an ignored field is an unasserted field."
        )
    return behavioural, numerical


def compare_behaviour(produced_path: Path, reference_path: Path) -> bool:
    """Compare what the worm DID, ignoring float drift in the weight columns.

    Returns True when every trajectory is behaviourally identical.
    """
    with h5py.File(produced_path, "r") as fp, h5py.File(reference_path, "r") as fr:
        drift = {"max": 0.0}
        blocking = _check_preconditions(fp, fr, [], drift)
        if blocking:
            print(f"\n[ABORT] {len(blocking)} input mismatch(es) -- the two runs did "
                  f"not simulate the same genomes/worlds, so a behavioural "
                  f"comparison would be meaningless:")
            for m in blocking[:20]:
                print(f"   - {m}")
            if len(blocking) > 20:
                print(f"   ... and {len(blocking) - 20} more")
            return False

        trajectories = sorted(
            path for path in _walk_datasets(fp)
            if path.endswith("/per_tick")
        )
        if not trajectories:
            print("\n[ABORT] no per_tick datasets found. Behaviour mode needs a "
                  "config with experiment.enable_per_tick_tracking: true "
                  "(e.g. --config test_behaviour).")
            return False

        n_ident = 0
        n_ticks = 0
        weight_max_diff = drift["max"]   # seeded with wiring's weight_final_*
        length_mismatch = 0
        divergences = []       # (path, field, tick)
        field_counts = {}
        weight_first_tick = None

        for path in trajectories:
            a, b = fp[path][()], fr[path][()]
            behavioural, numerical = _split_per_tick_fields(a.dtype, path)
            m = min(len(a), len(b))
            n_ticks += m

            same_length = len(a) == len(b)
            if not same_length:
                length_mismatch += 1

            first_bad, bad_field = None, None
            for field in behavioural:
                diff = a[field][:m] != b[field][:m]
                if diff.any():
                    field_counts[field] = field_counts.get(field, 0) + 1
                    t = _first_true(diff)
                    if first_bad is None or t < first_bad:
                        first_bad, bad_field = t, field

            if first_bad is None and same_length:
                n_ident += 1
            else:
                divergences.append((path, bad_field, first_bad,
                                    len(a), len(b)))

            # Numerical drift: reported so it is never silently hidden.
            for field in numerical:
                d = np.abs(a[field][:m].astype(np.float64)
                           - b[field][:m].astype(np.float64))
                if d.size and d.max() > 0.0:
                    weight_max_diff = max(weight_max_diff, float(d.max()))
                    t = _first_true(d > 0.0)
                    if t is not None and (weight_first_tick is None
                                          or t < weight_first_tick):
                        weight_first_tick = t

        # Order-free spatial and outcome checks, alongside the tick-by-tick one.
        outcome_bad = []
        for path in sorted(p for p in _walk_datasets(fp) if p.endswith("/staying")):
            if not np.array_equal(fp[path][()], fr[path][()]):
                outcome_bad.append(f"{path}: occupancy heat map differs")
        for path in sorted(p for p in _walk_datasets(fp) if p.endswith("/summary")):
            a, b = fp[path][()], fr[path][()]
            if a.shape != b.shape:
                outcome_bad.append(f"{path}: shape {a.shape} != {b.shape}")
                continue
            for field in sorted(BEHAVIOURAL_SUMMARY_FIELDS & set(a.dtype.names)):
                if not np.array_equal(a[field], b[field]):
                    n_bad = int((a[field] != b[field]).sum())
                    outcome_bad.append(f"{path}.{field}: {n_bad} run(s) differ")

    total = len(trajectories)
    print(f"\n[precondition] genomes, seeds and config match the reference")
    print(f"[behaviour]    trajectories compared: {total} ({n_ticks:,} ticks)")
    print(f"[behaviour]    behaviourally identical: {n_ident}/{total}")
    print(f"[behaviour]    differing lifetimes: {length_mismatch}")
    if weight_first_tick is None and weight_max_diff == 0.0:
        print(f"[numerical]    weight columns: bit-identical")
    elif weight_first_tick is None:
        print(f"[numerical]    weight drift: max |diff| {weight_max_diff:.3e} "
              f"(final weights only; ignored by design)")
    else:
        print(f"[numerical]    weight drift: max |diff| {weight_max_diff:.3e}, "
              f"from tick {weight_first_tick} (ignored by design)")

    if divergences or outcome_bad:
        print(f"\n[FAIL] behaviour DIVERGED from the reference")
        if field_counts:
            print("   fields that differ (trajectory count):")
            for field, count in sorted(field_counts.items(),
                                       key=lambda kv: -kv[1]):
                print(f"     {field:16s} {count}")
        for path, field, tick, la, lb in divergences[:15]:
            where = (f"first at tick {tick} in '{field}'" if field is not None
                     else "same prefix, different length")
            print(f"   - {path}: {where}  (len {la} vs {lb})")
        if len(divergences) > 15:
            print(f"   ... and {len(divergences) - 15} more trajectories")
        for msg in outcome_bad[:10]:
            print(f"   - {msg}")
        if len(outcome_bad) > 10:
            print(f"   ... and {len(outcome_bad) - 10} more outcome mismatches")
        return False

    print(f"\n[PASS] worm behaviour is IDENTICAL to reference "
          f"({reference_path})\n       {total} trajectories, {n_ticks:,} ticks, "
          f"every decision matched despite {weight_max_diff:.1e} weight drift")
    return True


def compare(produced_path: Path, reference_path: Path, use_tol, rtol, atol) -> bool:
    """Return True if produced matches reference. Prints a diff summary."""
    mismatches = []

    with h5py.File(produced_path, "r") as fp, h5py.File(reference_path, "r") as fr:
        prod_ds = _walk_datasets(fp)
        ref_ds = _walk_datasets(fr)

        missing = sorted(set(ref_ds) - set(prod_ds))
        extra = sorted(set(prod_ds) - set(ref_ds))
        for path in missing:
            mismatches.append(f"dataset MISSING in produced: {path}")
        for path in extra:
            mismatches.append(f"dataset EXTRA in produced:   {path}")

        for path in sorted(set(prod_ds) & set(ref_ds)):
            ok, msg = _values_equal(prod_ds[path][()], ref_ds[path][()],
                                    use_tol, rtol, atol)
            if not ok:
                mismatches.append(f"dataset DIFFERS {path}: {msg}")

        # Attributes (the saved config lives here).
        prod_attrs = _walk_attrs(fp)
        ref_attrs = _walk_attrs(fr)
        for path in sorted(set(ref_attrs) & set(prod_attrs)):
            pa, ra = prod_attrs[path], ref_attrs[path]
            for key in sorted(set(ra) - set(pa)):
                mismatches.append(f"attr MISSING in produced: {path}@{key}")
            for key in sorted(set(pa) - set(ra)):
                mismatches.append(f"attr EXTRA in produced:   {path}@{key}")
            for key in sorted(set(pa) & set(ra)):
                ok, msg = _values_equal(pa[key], ra[key], use_tol, rtol, atol)
                if not ok:
                    mismatches.append(f"attr DIFFERS {path}@{key}: {msg}")

    if mismatches:
        print(f"\n[FAIL] {len(mismatches)} mismatch(es) vs reference:")
        for m in mismatches[:40]:
            print(f"   - {m}")
        if len(mismatches) > 40:
            print(f"   ... and {len(mismatches) - 40} more")
        return False

    print(f"\n[PASS] produced output is identical to reference "
          f"({reference_path})")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--reference", type=str,
                       help="Reference H5 to compare the run against.")
    group.add_argument("--generate-reference", type=str, metavar="PATH",
                       help="Run and save the output as a new reference file.")
    parser.add_argument("--mode", choices=("exact", "behaviour"), default="exact",
                        help="exact (default): every dataset must be bit-identical. "
                             "behaviour: assert only that the worm made the same "
                             "decisions, ignoring float drift in the tracked "
                             "connection weights. Needs a config with "
                             "enable_per_tick_tracking (e.g. --config test_behaviour).")
    parser.add_argument("--tol", action="store_true",
                        help="Compare floats with tolerance instead of exact.")
    parser.add_argument("--rtol", type=float, default=1e-7)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--keep", action="store_true",
                        help="Keep the H5 file the run produced.")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Subprocess timeout in seconds (default 600).")
    parser.add_argument("--runner", default=DEFAULT_RUNNER,
                        help=f"Module to run (default {DEFAULT_RUNNER}). "
                             f"Use simulate.run_batch for the per-run-tracking path.")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help=f"Experiment config name (default {DEFAULT_CONFIG}).")
    parser.add_argument("--sim-name", default=DEFAULT_SIM_NAME,
                        help=f"experiment.simulation_name in the config, used to "
                             f"locate the produced H5 (default {DEFAULT_SIM_NAME}).")
    args = parser.parse_args()

    produced = run_simulation(args.timeout, args.runner, args.config, args.sim_name)

    if args.generate_reference:
        ref_path = Path(args.generate_reference)
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        # Copy bytes rather than move, so behavior matches a normal run.
        ref_path.write_bytes(produced.read_bytes())
        print(f"\n[OK] reference written: {ref_path}")
        if not args.keep:
            produced.unlink()
        sys.exit(0)

    ref_path = Path(args.reference)
    if not ref_path.exists():
        print(f"[FAIL] reference not found: {ref_path}", file=sys.stderr)
        sys.exit(1)

    if args.mode == "behaviour":
        if args.tol:
            print("[FAIL] --tol is meaningless with --mode behaviour: behavioural "
                  "fields are compared exactly and weight drift is ignored "
                  "entirely.", file=sys.stderr)
            sys.exit(1)
        ok = compare_behaviour(produced, ref_path)
    else:
        ok = compare(produced, ref_path, args.tol, args.rtol, args.atol)
    if not args.keep:
        produced.unlink()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
