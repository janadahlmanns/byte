#!/usr/bin/env python3
"""EA regression test: run `simulate.run_ea --config test_ea` and compare its
HDF5 output against a known-good reference file.

The test_ea run is fully deterministic (fixed simulation_seed, per-variant seeds
drawn in variant_id order), so a correct refactor must produce a bit-identical
HDF5 file. This script guards that invariant while the parallelism is optimized.

Usage
-----
Generate a reference from the current (known-good) code, once:
    python -m tests.ea_drift --generate-reference tests/refs/test_ea_ref.h5

Compare the current code against that reference (default mode):
    python -m tests.ea_drift --reference tests/refs/test_ea_ref.h5

Useful flags:
    --tol            Compare floats with np.allclose instead of exact equality.
    --rtol / --atol  Tolerances used with --tol (defaults: 1e-7 / 0.0).
    --keep           Keep the HDF5 file produced by the run (default: delete it).
    --timeout SEC    Subprocess timeout in seconds (default: 600).

Exit code is 0 when output matches the reference, 1 otherwise.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import h5py

# Mirrors configs/experiments/test_ea.yaml: experiment.output_folder and
# experiment.simulation_name. The produced file is "{timestamp}_{name}.h5".
OUTPUT_DIR = Path("data/temp")
SIMULATION_NAME = "test_task_switching"
RUN_CMD = [sys.executable, "-m", "simulate.run_ea", "--config", "test_ea"]


def _existing_outputs() -> set:
    """Set of EA output H5 files currently in OUTPUT_DIR."""
    if not OUTPUT_DIR.exists():
        return set()
    return set(OUTPUT_DIR.glob(f"*_{SIMULATION_NAME}.h5"))


def run_simulation(timeout: int) -> Path:
    """Run the EA and return the path to the H5 file it produced.

    Identifies the new file by diffing the OUTPUT_DIR listing taken before and
    after the run, so pre-existing files in data/temp are not mistaken for it.
    """
    before = _existing_outputs()
    print(f"[run] {' '.join(RUN_CMD)}")
    start = time.time()
    try:
        # Feed "n" to the unconditional "Display ... plot? (y/n)" prompt so the
        # run never blocks waiting on stdin.
        proc = subprocess.run(
            RUN_CMD,
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
        print("[FAIL] run_ea exited non-zero. Captured output:", file=sys.stderr)
        sys.stderr.write(proc.stdout or "")
        sys.stderr.write(proc.stderr or "")
        sys.exit(1)

    new_files = _existing_outputs() - before
    if len(new_files) == 0:
        print(f"[FAIL] no new *_{SIMULATION_NAME}.h5 produced in {OUTPUT_DIR}",
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
    parser.add_argument("--tol", action="store_true",
                        help="Compare floats with tolerance instead of exact.")
    parser.add_argument("--rtol", type=float, default=1e-7)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--keep", action="store_true",
                        help="Keep the H5 file the run produced.")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Subprocess timeout in seconds (default 600).")
    args = parser.parse_args()

    produced = run_simulation(args.timeout)

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

    ok = compare(produced, ref_path, args.tol, args.rtol, args.atol)
    if not args.keep:
        produced.unlink()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
