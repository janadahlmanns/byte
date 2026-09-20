"""Check a replay against the per_tick data recorded during the original run.

`MetricsRecorder` writes per_tick into the HDF5 while the experiment runs, and
the replay recorder samples at the same point in the tick, so the two series
line up index for index. A divergence means the replay is not reproducing the
run it claims to.
"""

import numpy as np


# Sensor key in a recorded frame -> column name in the per_tick table
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
    """Compare recorded frames against the per_tick data.

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
    # per_tick[T].movement is the move executed during tick T, which is the
    # action decided at tick T-1, i.e. frames[T-1].next_action.
    bad = []
    for i in range(1, n):
        got = frames[i - 1]["next_action"] or "stay"
        raw = truth[i]["movement"]
        exp = raw.decode() if isinstance(raw, (bytes, np.bytes_)) else str(raw)
        if got != exp:
            bad.append((frames[i]["t"], got, exp))
    checks.append(Check(
        "movement", max(0, n - 1), bad,
        note=("MetricsRecorder derives direction from raw coordinate differences "
              "without accounting for toroidal wrap, so it reports the opposite "
              "direction when the agent crosses an edge. The value compared here "
              "is taken from the action tuple.")
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
    # Movement is excluded from the verdict; see the note attached to that check.
    judged = [c for c in checks if c.name != "movement"]
    failed = [c for c in judged if not c.ok]
    verdict = ("PASS: replay matches the recorded run"
               if not failed
               else f"FAIL: {len(failed)} field(s) diverged")
    return f"{head}\n{body}\n\n  {verdict}"


def _detect_aborted(truth, source):
    """Detect a recorded run that was interrupted rather than ended by the simulation.

    A run ends legitimately when the agent starves (final energy 0) or hits the
    tick limit. Still alive and below the limit means PauseManagerExit fired and
    `simulate_run` returned early. Such a run is stored like any other result but
    is truncated, so a replay cannot match it.
    """
    if len(truth) == 0:
        return "the recorded run contains no ticks"

    last = truth[-1]
    final_energy = float(last["energy"])
    final_tick = int(last["tick"])

    if final_energy > 0 and final_tick < source.max_ticks:
        if final_tick == 0:
            return (
                "the recorded run has a single tick at full energy: the original "
                "simulation was interrupted before this run started, so there is "
                "nothing to compare against."
            )
        return (
            f"the recorded run stops at tick {final_tick} while still alive with "
            f"{final_energy:.0f} energy, below the {source.max_ticks} tick limit. "
            "The original simulation was interrupted, so the stored data is "
            "truncated and a replay cannot match it."
        )
    return None
