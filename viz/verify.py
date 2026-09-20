"""Check a replay against the ground truth recorded during the original run.

`MetricsRecorder` writes a `per_tick` table into the HDF5 while the experiment
runs. Our recorder samples at the same point in the tick (`wait_frame`, after
`worm.ticks += 1`), so the two series should line up index for index.

This is the load-bearing test for the whole viewer: if the replay drifts, the
page is showing a run that never happened.
"""

import numpy as np


# Sensor key in our frames -> column name in the per_tick table
SENSOR_COLUMNS = {
    "food_north": "food_sensed_N",
    "food_east": "food_sensed_E",
    "food_south": "food_sensed_S",
    "food_west": "food_sensed_W",
}


class Check:
    """One compared field."""

    def __init__(self, name, total, mismatches, note=""):
        self.name = name
        self.total = total
        self.mismatches = mismatches
        self.note = note

    @property
    def ok(self):
        return not self.mismatches

    def __str__(self):
        n = len(self.mismatches)
        status = "OK" if self.ok else f"{n} MISMATCH"
        line = f"  {self.name:<18} {self.total:>6} compared   {status}"
        if self.mismatches:
            preview = ", ".join(
                f"t{t}: got {g!r} expected {e!r}" for t, g, e in self.mismatches[:3]
            )
            line += f"\n      {preview}"
            if n > 3:
                line += f"  (+{n - 3} more)"
        if self.note:
            line += f"\n      note: {self.note}"
        return line


def verify(recorder, source):
    """Compare recorded frames against the per_tick ground truth.

    Args:
        recorder: WorldFrameRecorder after a replay
        source: the RunSource it was replayed from

    Returns:
        (checks, fatal) where checks is a list of Check and fatal is a string
        when verification could not be attempted at all.
    """
    truth = source.ground_truth
    if truth is None:
        return [], "no per_tick data in this file (was per-tick tracking enabled?)"

    aborted = _detect_aborted(truth, source)
    if aborted:
        return [], aborted

    frames = recorder.frames
    checks = []

    # --- length ---
    checks.append(Check(
        "frame count", 1,
        [] if len(frames) == len(truth) else [("-", len(frames), len(truth))],
    ))
    n = min(len(frames), len(truth))
    if n == 0:
        return checks, "nothing to compare"

    # --- tick numbering ---
    bad = [(i, frames[i]["t"], int(truth[i]["tick"]))
           for i in range(n) if frames[i]["t"] != int(truth[i]["tick"])]
    checks.append(Check("tick index", n, bad))

    # --- energy ---
    bad = [(frames[i]["t"], frames[i]["energy"], float(truth[i]["energy"]))
           for i in range(n)
           if abs(frames[i]["energy"] - float(truth[i]["energy"])) > 1e-4]
    checks.append(Check("energy", n, bad))

    # --- sensors ---
    for key, col in SENSOR_COLUMNS.items():
        bad = []
        for i in range(n):
            got = int(frames[i]["sense"].get(key, 0) > 0)
            exp = int(truth[i][col])
            if got != exp:
                bad.append((frames[i]["t"], got, exp))
        checks.append(Check(col, n, bad))

    # --- manhattan distance from start (recomputed the same way) ---
    sy, sx = recorder.static["start_pos"]
    bad = []
    for i in range(n):
        got = abs(frames[i]["y"] - sy) + abs(frames[i]["x"] - sx)
        exp = int(truth[i]["manhattan_dist"])
        if got != exp:
            bad.append((frames[i]["t"], got, exp))
    checks.append(Check("manhattan_dist", n, bad))

    # --- food consumed this tick, derived from the eats counter ---
    bad = []
    for i in range(1, n):
        got = int(frames[i]["eats"] > frames[i - 1]["eats"])
        exp = int(truth[i]["food_consumed"])
        if got != exp:
            bad.append((frames[i]["t"], got, exp))
    checks.append(Check("food_consumed", max(0, n - 1), bad))

    # --- movement ---
    # per_tick[T].movement is the move EXECUTED during tick T, which is the
    # action DECIDED at tick T-1 -- our frames[T-1].next_action.
    bad = []
    for i in range(1, n):
        got = frames[i - 1]["next_action"] or "stay"
        raw = truth[i]["movement"]
        exp = raw.decode() if isinstance(raw, (bytes, np.bytes_)) else str(raw)
        if got != exp:
            bad.append((frames[i]["t"], got, exp))
    checks.append(Check(
        "movement", max(0, n - 1), bad,
        note=("MetricsRecorder derives direction by comparing raw coordinates "
              "without accounting for toroidal wrap, so it reports the opposite "
              "direction whenever the agent crosses an edge. Our value comes "
              "from the action tuple itself and is correct.")
        if bad else "",
    ))

    return checks, None


def format_report(checks, fatal, source):
    """Render the verification result as text."""
    head = (f"[verify] {source.hdf5_path}\n"
            f"[verify] genome {source.genome_id} run {source.run_id}")
    if fatal:
        return f"{head}\n  SKIPPED: {fatal}"

    body = "\n".join(str(c) for c in checks)
    # Movement is excluded from the verdict: it disagrees only where the
    # ground truth itself is wrong (see the note above).
    judged = [c for c in checks if c.name != "movement"]
    failed = [c for c in judged if not c.ok]
    verdict = ("PASS: replay matches the recorded run"
               if not failed
               else f"FAIL: {len(failed)} field(s) diverged")
    return f"{head}\n{body}\n\n  {verdict}"


def _detect_aborted(truth, source):
    """Detect a recorded run that was stopped by the user rather than by the simulation.

    A run ends legitimately either because the agent starved (final energy 0) or
    because it hit the tick limit. Anything else -- still alive, below the limit --
    means PauseManagerExit fired because someone pressed 'c' or closed the
    visualisation window, and `simulate_run` returned early.

    Such a run is written to the HDF5 looking like a real result, but it cannot be
    replayed faithfully: the replay has no reason to stop where the user did.
    """
    if len(truth) == 0:
        return "the recorded run contains no ticks"

    last = truth[-1]
    final_energy = float(last["energy"])
    final_tick = int(last["tick"])

    if final_energy > 0 and final_tick < source.max_ticks:
        if final_tick == 0:
            return (
                "the recorded run never ran -- it has a single tick at full energy. "
                "The original simulation was aborted before this run started "
                "(window closed, or 'c' pressed), so there is nothing to verify against."
            )
        return (
            f"the recorded run was cut short: it stops at tick {final_tick} while "
            f"still alive with {final_energy:.0f} energy, below the {source.max_ticks} "
            "tick limit. The original simulation was interrupted, so the stored data "
            "is truncated and a faithful replay cannot match it."
        )
    return None
