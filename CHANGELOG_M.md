# Changelog

## 2026-06-18

### Added

**Added ea_drift.py to test whether code changes have caused calculation mistakes**

A standalone regression test that runs `python -m simulate.run_ea --config test_ea` and compares the resulting HDF5 file against a known-good reference. It exists to guard numerical correctness while the runtime/parallelism is being optimized: the `test_ea` run is fully deterministic (fixed `simulation_seed`, per-variant seeds drawn in `variant_id` order), so any correct refactor must reproduce a bit-identical output file. A difference means the change altered the calculation.

How it works:
- Runs the EA as a subprocess, feeding `"n"` to the unconditional plot prompt so the run never blocks on stdin, with a timeout and exit-code check.
- Locates the produced file by diffing the `data/temp/` listing before/after the run, so the timestamped output is identified without matching on the (non-deterministic) filename and pre-existing files are ignored.
- Recursively compares **every dataset** (`generation_stats`, all `elite_genomes/*` arrays, `lifespans`, `run_seeds`) including structured-dtype fields, plus **all config attributes**. Reports missing/extra paths and per-path value differences.
- Comparison is **exact by default** (`np.array_equal`); `--tol` switches floats to `np.allclose` with `--rtol`/`--atol` as an escape hatch. Exit code is `0` on match and `1` on any mismatch, so it is automation-friendly.
- Cleans up the HDF5 file the run produced unless `--keep` is passed.

Two modes:
- `--generate-reference PATH` — run once on known-good code to capture the baseline.
- `--reference PATH` — run after each change to verify the output is unchanged.

#### Changes

- **`tests/ea_drift.py`** (new)
  - Standalone CLI test script; no pytest dependency. Run with the project interpreter so the subprocess inherits the right environment, e.g. `pyenv/bin/python -m tests.ea_drift --reference tests/refs/test_ea_ref.h5`.

- **`tests/refs/`** (new)
  - Holds the committed reference HDF5 used as the correctness baseline.

---

**Added ea_timer.py to track runtime across code versions**

A lightweight timing harness that runs a simulation, measures wall-clock time, and appends it to a history file so runtimes can be compared across refactors and machines. It defaults to `simulate.run_ea --config test_ea` but the runner, config, and output simulation_name are overridable via `--runner`/`--config`/`--sim-name` (mirroring `ea_drift.py`), so any entrypoint — e.g. `simulate.run_batch` — can be timed. It is meant as a quick "is there any gain at all?" screen, not a controlled benchmark — environment noise (CPU load, temperature) is deliberately not controlled.

How it works:
- Times the simulation subprocess end-to-end (feeding `"n"` to the run_ea plot prompt, with a timeout and exit-code check) and appends one CSV row per rep to `tests/bench/runtimes.csv`.
- Each row is auto-tagged with the git short commit, branch, a **dirty-tree flag**, the **hostname**, the **runner and config**, and an optional `--note`. The dirty flag matters because an uncommitted refactor would otherwise be recorded under the previous commit's hash; `--note` provides a human label to distinguish such versions; recording the config keeps timings for different configs from being conflated.
- `--reps N` records multiple timings per invocation; produced HDF5/PNG files are cleaned up unless `--keep-output`.
- `--plot` renders the accumulated history to `tests/bench/runtimes.png`: x-axis is the version (`config:note`, else `config:commit`), one line per host, with min/max error bars when multiple reps exist. The per-host grouping lets results from different computers be compared on the same graph.
- The history file is migrated automatically: if an older `runtimes.csv` lacks the `runner`/`config` columns, it is rewritten under the current schema with those columns backfilled, preserving previously recorded baseline rows.

Note on interpretation: the measurement is end-to-end, so constant serial overhead (imports, HDF5 write, the EA's own plot) is included and slightly understates the percentage gain of compute-only optimizations. On fanless machines, sustained back-to-back reps drift upward from thermal throttling — compare cold first-runs (and prefer the min) there, and save precise numbers for a machine with active cooling.

#### Changes

- **`tests/ea_timer.py`** (new)
  - Standalone CLI; no pytest dependency. Run with the project interpreter, e.g. `pyenv/bin/python -m tests.ea_timer --note "baseline"`, and `python -m tests.ea_timer --plot` to graph the history.

- **`tests/bench/`** (new)
  - Holds the accumulated `runtimes.csv` history and the generated `runtimes.png` graph.

---


## 2026-06-12

### Fixed

**SIGABRT on macOS when running with visualization enabled**

`pynput.keyboard.Listener` was being started in a background thread at the same time Qt's `QApplication` was running on the main thread. Both use macOS's Text Input Services (Carbon/CoreFoundation) API, which macOS forbids calling from two threads concurrently, causing an immediate SIGABRT.

Root cause: `simulation_helper_functions.py:get_num_workers()` called `init_pause_manager()` — which spawned the pynput listener thread — when visualization was enabled.

Fix: Stripped pynput from `PauseManager` entirely, turning it into a pure state holder (3 boolean flags, no threads). Qt renderers' `keyPressEvent` now writes directly to the `PauseManager` singleton; `simulate_run()` passes `app.processEvents` to `check_pause()` so the UI stays responsive during pause.

---

**Pause, step, resume, and cancel not working correctly**

Each Qt window (`world_renderer_qt`, `brain_renderer_qt`) held its own independent `paused`/`exit_requested` flags that nothing read — `simulate_run` only checked the pynput-based `PauseManager`. Keys pressed in either window had no effect. Additionally, the `'n'` step key was clearing `paused` directly, which caused the simulation to resume fully instead of advancing one tick.

Fix: Both renderers now write to the shared `PauseManager` singleton. Step (`'n'`) sets `step_requested` without touching `paused`, so the simulation advances exactly one tick and re-pauses.

---

#### Changes

- **`simulate/pause_manager.py`**
  - Removed all pynput imports and the `_start_listener()` method.
  - `PauseManager` is now a pure state holder — no threads, no keyboard listener.
  - `check_pause()` accepts an optional `process_events` callable, invoked during the pause spin-loop to keep the Qt event loop delivering key events.
  - `cleanup()` resets flags instead of stopping a listener.
  - API surface unchanged: `init_pause_manager()`, `get_pause_manager()`, `cleanup_pause_manager()`, `PauseManagerExit`.

- **`mvb/simulation_API.py`**
  - `simulate_run()` resolves `app.processEvents` from the worm's renderer and passes it to `check_pause()`.
  - Moved `cleanup_pause_manager` to a top-level import (was a late `from` import in the `finally` block).

- **`mvb/simulation_helper_functions.py`**
  - Removed unused `PauseManagerExit` from import.

- **`mvb/world_renderer_qt.py`**
  - Removed independent `self.paused`/`self.exit_requested` instance variables.
  - `keyPressEvent` writes to the `PauseManager` singleton: `'p'` toggles `_paused`, `'n'` sets `_step_requested` without touching `_paused`, `'c'` sets `_exit_requested`.
  - `closeEvent` sets `_exit_requested` on the singleton.

- **`mvb/brain_renderer_qt.py`**
  - Same changes as `world_renderer_qt.py`. Key events in either window now update the same shared `PauseManager` state.

- **`mvb/brains/decisionmaking_plasticity.py`**
  - Updated the brain-beat pause checkpoint to pass `process_events` from the brain renderer's Qt app (was previously calling `check_pause()` without it, which would have blocked Qt event delivery during pause).

- **`simulate/run_batch.py`**
  - Removed unused `init_pause_manager`/`cleanup_pause_manager` import.

- **`simulate/run_ea.py`**
  - Removed unused `init_pause_manager`/`cleanup_pause_manager` import.
