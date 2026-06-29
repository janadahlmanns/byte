#!/usr/bin/env python3
"""Time a simulation run and append it to a history file.

By default this times `simulate.run_ea --config test_ea`, but the runner, config,
and output simulation_name are all overridable (see --runner/--config/--sim-name),
mirroring tests/ea_drift.py, so the same harness can time other entrypoints.

Workflow: run this on the current code, refactor, run it again. Each invocation
appends a row to tests/bench/runtimes.csv (timestamped, tagged with git
commit/branch and hostname). Use --plot to render the accumulated history as a
graph so you can eyeball whether a refactor actually got faster.

This is a quick relative comparison across versions, not an absolute benchmark:
environment noise (CPU load, temperature) is not controlled. Good enough to spot
whether there's any gain on a given machine before validating on a stronger one.

Usage
-----
    python -m tests.ea_timer                       # time once, append to history
    python -m tests.ea_timer --reps 3 --note "pool reuse"
    python -m tests.ea_timer --plot                # render history graph, no run

Time a different runner/config:
    python -m tests.ea_timer --runner simulate.run_batch \
        --config test_batch_tracking --sim-name test_batch_tracking

Each row records: timestamp, hostname, git commit, branch, dirty flag, runner,
config, note, rep index, and elapsed wall-clock seconds.
"""

import argparse
import csv
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

OUTPUT_DIR = Path("data/temp")

# Defaults reproduce the original EA timing. Override via CLI to time a
# different runner/config (matching tests/ea_drift.py's syntax).
DEFAULT_RUNNER = "simulate.run_ea"
DEFAULT_CONFIG = "test_ea"
DEFAULT_SIM_NAME = "test_task_switching"

HISTORY_PATH = Path("tests/bench/runtimes.csv")
PLOT_PATH = Path("tests/bench/runtimes.png")
CSV_FIELDS = ["timestamp", "host", "commit", "branch", "dirty", "runner",
              "config", "note", "rep", "seconds"]


def _git(*args) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def git_tag() -> dict:
    """Identify the code version: short commit, branch, and dirty-tree flag."""
    commit = _git("rev-parse", "--short", "HEAD") or "unknown"
    branch = _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown"
    dirty = bool(_git("status", "--porcelain"))
    return {"commit": commit, "branch": branch, "dirty": dirty}


def _outputs(sim_name: str) -> set:
    """Set of output files for sim_name currently in OUTPUT_DIR (.h5 and .png)."""
    if not OUTPUT_DIR.exists():
        return set()
    return set(OUTPUT_DIR.glob(f"*_{sim_name}.*"))


def time_one_run(timeout: int, keep_output: bool,
                 runner: str, config: str, sim_name: str) -> float:
    """Run the simulation once, return elapsed wall-clock seconds.

    Exits on timeout or non-zero exit.
    """
    run_cmd = [sys.executable, "-m", runner, "--config", config]
    before = _outputs(sim_name)
    start = time.perf_counter()
    try:
        # Feed "n" to any "Display ... plot? (y/n)" prompt (run_ea) so the run
        # never blocks waiting on stdin; runners without the prompt ignore it.
        proc = subprocess.run(run_cmd, input="n\n", text=True,
                              timeout=timeout, capture_output=True)
    except subprocess.TimeoutExpired:
        print(f"[FAIL] run exceeded timeout of {timeout}s", file=sys.stderr)
        sys.exit(1)
    elapsed = time.perf_counter() - start

    if proc.returncode != 0:
        print(f"[FAIL] {runner} exited non-zero. Captured output:", file=sys.stderr)
        sys.stderr.write(proc.stdout or "")
        sys.stderr.write(proc.stderr or "")
        sys.exit(1)

    if not keep_output:
        for f in _outputs(sim_name) - before:
            f.unlink()
    return elapsed


def append_history(rows: list):
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    old_fields, old_rows = None, []
    if HISTORY_PATH.exists():
        with HISTORY_PATH.open(newline="") as f:
            reader = csv.DictReader(f)
            old_fields = reader.fieldnames
            old_rows = list(reader)

    # If an older history used a different schema (e.g. before runner/config
    # columns existed), rewrite it under the current schema, backfilling missing
    # columns with "" so the accumulated baseline data is preserved.
    if old_fields is not None and old_fields != CSV_FIELDS:
        with HISTORY_PATH.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for r in old_rows:
                writer.writerow({k: r.get(k, "") for k in CSV_FIELDS})
            writer.writerows(rows)
        return

    with HISTORY_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if old_fields is None:
            writer.writeheader()
        writer.writerows(rows)


def do_run(args):
    tag = git_tag()
    if tag["dirty"]:
        print("[note] working tree is dirty — timing uncommitted changes; "
              "use --note to label this version.")
    print(f"[timer] {args.runner} --config {args.config} | "
          f"{tag['branch']}@{tag['commit']}"
          f"{' (dirty)' if tag['dirty'] else ''} — {args.reps} rep(s)")

    rows, times = [], []
    for rep in range(args.reps):
        secs = time_one_run(args.timeout, args.keep_output,
                            args.runner, args.config, args.sim_name)
        times.append(secs)
        print(f"  rep {rep + 1}/{args.reps}: {secs:.1f}s")
        rows.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "host": platform.node(),
            "commit": tag["commit"],
            "branch": tag["branch"],
            "dirty": int(tag["dirty"]),
            "runner": args.runner,
            "config": args.config,
            "note": args.note,
            "rep": rep,
            "seconds": f"{secs:.3f}",
        })

    append_history(rows)
    med = sorted(times)[len(times) // 2]
    print(f"[timer] median {med:.1f}s, best {min(times):.1f}s "
          f"-> appended {len(rows)} row(s) to {HISTORY_PATH}")


def do_plot(_args):
    """Render history grouped by host; one marker per (host, version)."""
    import collections
    import matplotlib.pyplot as plt

    if not HISTORY_PATH.exists():
        print(f"[FAIL] no history at {HISTORY_PATH}", file=sys.stderr)
        sys.exit(1)

    with HISTORY_PATH.open(newline="") as f:
        records = list(csv.DictReader(f))
    if not records:
        print("[FAIL] history is empty", file=sys.stderr)
        sys.exit(1)

    # Version label = note if present, else commit; prefix the config when one
    # is recorded so multiple configs stay on separate x positions. Keep
    # first-seen order.
    def label(r):
        version = r["note"] or r["commit"]
        config = r.get("config") or ""
        return f"{config}:{version}" if config else version

    order, seen = [], set()
    for r in records:
        lbl = label(r)
        if lbl not in seen:
            seen.add(lbl)
            order.append(lbl)

    # Group seconds by (host, label).
    grouped = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in records:
        grouped[r["host"]][label(r)].append(float(r["seconds"]))

    fig, ax = plt.subplots(figsize=(max(8, len(order) * 1.2), 6))
    x = range(len(order))
    for host, by_label in sorted(grouped.items()):
        ys, lo, hi = [], [], []
        for lbl in order:
            vals = by_label.get(lbl, [])
            if vals:
                m = sum(vals) / len(vals)
                ys.append(m); lo.append(m - min(vals)); hi.append(max(vals) - m)
            else:
                ys.append(float("nan")); lo.append(0); hi.append(0)
        ax.errorbar(x, ys, yerr=[lo, hi], marker="o", capsize=4, label=host)

    ax.set_xticks(list(x))
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylabel("Runtime [s]")
    ax.set_xlabel("Version (config:note, else config:commit)")
    ax.set_title("Simulation runtime by version")
    ax.grid(True, alpha=0.3)
    ax.legend(title="host")
    fig.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=150)
    print(f"[plot] saved {PLOT_PATH} ({len(order)} version(s), "
          f"{len(grouped)} host(s))")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reps", type=int, default=1,
                        help="Number of timed runs to record (default 1).")
    parser.add_argument("--note", type=str, default="",
                        help="Label for this version (recommended when the "
                             "working tree is dirty).")
    parser.add_argument("--timeout", type=int, default=1200,
                        help="Per-run subprocess timeout in seconds.")
    parser.add_argument("--keep-output", action="store_true",
                        help="Keep the HDF5/PNG files the run produced.")
    parser.add_argument("--runner", default=DEFAULT_RUNNER,
                        help=f"Module to run with -m (default {DEFAULT_RUNNER}).")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help=f"Experiment config name (default {DEFAULT_CONFIG}).")
    parser.add_argument("--sim-name", default=DEFAULT_SIM_NAME,
                        help=f"experiment.simulation_name in the config, used to "
                             f"locate produced output files (default "
                             f"{DEFAULT_SIM_NAME}).")
    parser.add_argument("--plot", action="store_true",
                        help="Render the history graph and exit (no run).")
    args = parser.parse_args()

    if args.plot:
        do_plot(args)
    else:
        do_run(args)


if __name__ == "__main__":
    main()
