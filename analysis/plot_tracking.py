#!/usr/bin/env python3
"""Tracked batch runs: tensor code before / after the Step 32 writer, and the scalar code
(plan_evotorch.md, Steps 31 and 32; tests/tracking_bench.py).

Reads one or more tracking reports from the same machine. Each case is one of:
  scalar        the scalar code (device "scalar"; total time only)
  per_call      tensor code, the writer before Step 32 (reports without a writer field)
  one_file      tensor code, Step 32's one-file writer, written in place
  one_file+stage  the same, staged in a local folder and copied back

  (a)-(c) total time per tracking level for case groups A, B and C: scalar, CPU and
          CUDA, before (hatched) and after (solid; the staged run where there is one)
  (d)     B with all tracking on: where the time goes (simulation, building the tables,
          HDF5, staging copies, the rest) for each writer, against the scalar total

Failed and timed-out cases are marked, not hidden. Output: tracking.png / .pdf,
tracking.csv (one row per case, with the speed-up against the scalar code and against
the writer before Step 32).

Usage
-----
    python -m analysis.plot_tracking data/runtime_reports/<..._tracking> [<more>] --out <folder>
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import analysis.plot_scaling as PS
from analysis.plot_neuron_scaling import INK, INK2, SERIES, SURFACE, _style

LEVEL_ORDER = ["off", "run", "tick", "heat", "all"]
LEVEL_NAME = {"off": "off", "run": "per-run", "tick": "+per-tick", "heat": "+heat map", "all": "all"}
GROUP_TEXT = {"A": "A: 10 lookup elites x 300 runs, n = 11", "B": "B: 260 genomes x 300 runs, n = 11",
              "C": "C: 10 large-run elites x 100 runs, n = 50"}
DEVICE_COLOR = {"cuda": SERIES[0], "cpu": SERIES[1], "scalar": SERIES[2]}


def impl_of(c):
    if c["device"] == "scalar":
        return "scalar"
    w = c.get("writer") or "per_call"           # reports from before Step 32 have no writer
    return w + ("+stage" if c.get("stage", "none") != "none" else "")


def ok(c):
    return c.get("status") == "ok"


def load(reports):
    machines, cases = set(), {}
    for r in reports:
        r = Path(r)
        machines.add(PS.machine_of(json.loads((r / "system.json").read_text()))[0])
        for f in sorted((r / "cases").glob("*.json")):
            c = json.loads(f.read_text())
            c["impl"] = impl_of(c)
            c["report"] = r.name
            key = (c["impl"], c["name"])
            if key in cases:
                raise SystemExit(f"ERROR: case {c['name']} ({c['impl']}) in more than one report: "
                                 f"{cases[key]['report']}, {r.name}")
            cases[key] = c
    if len(machines) != 1:
        raise SystemExit(f"ERROR: reports from {len(machines)} machines: {sorted(machines)}")
    if not cases:
        raise SystemExit("ERROR: no finished case")
    return next(iter(machines)), cases


def find(cases, impl, group, device, level, name=None):
    for c in cases.values():
        if (c["impl"], c["group"], c["device"], c["level"]) == (impl, group, device, level) and \
                (name is None or c["name"] == name) and not c["name"].endswith("_direct"):
            return c
    return None


def after_impl(cases, group, device, level):
    """The Step 32 case shown as "after": the staged one where there is one."""
    return (find(cases, "one_file+stage", group, device, level)
            or find(cases, "one_file", group, device, level))


def speedups(cases):
    """(impl, name) -> (vs scalar, vs per_call): total / total of the same group, device
    (for per_call) and level; both cases ok."""
    out = {}
    for k, c in cases.items():
        if c["impl"] == "scalar" or not ok(c) or not c.get("wall_s"):
            continue
        sc = find(cases, "scalar", c["group"], "scalar", c["level"])
        bf = find(cases, "per_call", c["group"], c["device"], c["level"])
        out[k] = (sc["wall_s"] / c["wall_s"] if sc and ok(sc) else None,
                  bf["wall_s"] / c["wall_s"] if bf and ok(bf) and c["impl"] != "per_call" else None)
    return out


def _status_word(c):
    s = str(c["status"])
    return "timeout" if s.startswith("timeout") else "skipped" if s.startswith("skipped") else "failed"


def draw_group(ax, cases, group, title):
    series = [("scalar", "scalar", "scalar code", False)]
    for dev in ("cpu", "cuda"):
        series += [(dev, "per_call", f"{dev.upper()} before", True), (dev, "after", f"{dev.upper()} after", False)]
    present = []
    for dev, impl, label, hatch in series:
        cs = {lv: (after_impl(cases, group, dev, lv) if impl == "after" else find(cases, impl, group, dev, lv))
              for lv in LEVEL_ORDER}
        if any(cs.values()):
            present.append((dev, label, hatch, cs))
    levels = [lv for lv in LEVEL_ORDER if any(cs[lv] for *_, cs in present)]
    if not present:
        ax.set_visible(False)
        return
    w = 0.8 / len(present)
    top = 1
    for j, (dev, label, hatch, cs) in enumerate(present):
        for i, lv in enumerate(levels):
            c = cs[lv]
            x = i + (j - (len(present) - 1) / 2) * w
            if c is None:
                continue
            if ok(c):
                col = DEVICE_COLOR[dev]
                ax.bar(x, c["wall_s"], w * 0.92, color=SURFACE if hatch else col, edgecolor=col,
                       hatch="////" if hatch else None, linewidth=1.0)
                ax.text(x, c["wall_s"] * 1.06, f"{c['wall_s']:.3g}", ha="center", va="bottom", fontsize=5.5,
                        color=INK, rotation=90)
                top = max(top, c["wall_s"])
            else:
                ax.text(x, 0.02, _status_word(c), transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                        fontsize=6, color=INK2, rotation=90)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([LEVEL_NAME[lv] for lv in levels], fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(top=top * 4)
    _style(ax, title, "tracking level", "total seconds (log)")
    return [(DEVICE_COLOR[dev], label, hatch) for dev, label, hatch, _ in present]


def draw_breakdown(ax, cases):
    rows = []
    for impl, label in (("per_call", "CUDA, before\n(open/close per dataset)"),
                        ("one_file", "CUDA, one file\nin place"),
                        ("one_file+stage", "CUDA, one file\nstaged")):
        c = find(cases, impl, "B", "cuda", "all")
        if impl == "one_file":
            c = next((d for d in cases.values() if d["impl"] == "one_file" and d["group"] == "B"
                      and d["device"] == "cuda" and d["level"] == "all"), None)
        if c and ok(c):
            rows.append((label, c))
    for impl, label in (("per_call", "CPU, before"), ("after", "CPU, after")):
        c = after_impl(cases, "B", "cpu", "all") if impl == "after" else find(cases, impl, "B", "cpu", "all")
        if c and ok(c):
            rows.append((label, c))
    sc = find(cases, "scalar", "B", "scalar", "all")
    # neutral tones: blue / orange / green mean CUDA / CPU / scalar in (a)-(c)
    parts = [("simulation", "#a3a29d", lambda c: c.get("sim_s") or 0),
             ("building tables", "#d4d3ce", lambda c: (c.get("phases") or {}).get("build_s", 0)),
             ("HDF5", INK, lambda c: (c.get("phases") or {}).get("h5_s", c.get("write_s", 0)
                                                                        if not c.get("phases") else 0)),
             ("staging copies", INK2, lambda c: sum((c.get("phases") or {}).get(k, 0)
                                                        for k in ("stage_in_s", "stage_out_s")))]
    for i, (label, c) in enumerate(rows):
        left = 0
        for _, col, f in parts:
            v = f(c)
            ax.barh(i, v, 0.6, left=left, color=col, edgecolor=SURFACE, linewidth=1)
            left += v
        rest = max(c["wall_s"] - left, 0)
        ax.barh(i, rest, 0.6, left=left, color=SURFACE, edgecolor=INK2, linewidth=0.6)
        ax.text(c["wall_s"], i, f"  {c['wall_s']:.0f} s", va="center", fontsize=7, color=INK)
    if sc and ok(sc):
        ax.axvline(sc["wall_s"], color=INK, linestyle="--", linewidth=1)
        ax.text(sc["wall_s"], -0.7, f"scalar code {sc['wall_s']:.0f} s", ha="center", va="bottom", fontsize=7,
                color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.invert_yaxis()
    ax.set_ylim(len(rows) - 0.4, -1.1)
    if rows:
        ax.set_xlim(0, max(max(r[1]["wall_s"] for r in rows), sc["wall_s"] if sc and ok(sc) else 0) * 1.18)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=col, label=n) for n, col, _ in parts]
              + [Patch(facecolor=SURFACE, edgecolor=INK2, label="rest (compile warm-up, setup)")],
              fontsize=6.5, frameon=False, labelcolor=INK, loc="lower right")
    _style(ax, "(d) B, all tracking: where the time goes", "seconds", "")


def draw(cases, machine, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), facecolor=SURFACE)
    legend = {}
    for ax, g, letter in zip(axes.flat[:3], "ABC", "abc"):
        got = draw_group(ax, cases, g, f"({letter}) {GROUP_TEXT[g]}")
        for col, label, hatch in got or []:
            legend[label] = (col, hatch)
    draw_breakdown(axes[1, 1], cases)
    fig.legend(handles=[Patch(facecolor=SURFACE if h else col, edgecolor=col, hatch="////" if h else None, label=l)
                        for l, (col, h) in legend.items()],
               loc="upper right", ncol=len(legend), fontsize=8, frameon=False, labelcolor=INK,
               bbox_to_anchor=(0.99, 0.965))
    fig.suptitle("Tracked batch runs: the tensor code before and after the one-file writer (Step 32), "
                 "and the scalar code", x=0.01, y=0.99, ha="left", fontsize=11, color=INK, fontweight="bold")
    fig.text(0.01, 0.008, f"{machine}. Before: open/close per dataset (hdf5_utils). After: one open file per "
             f"generation, staged locally where marked. Totals include the compile warm-up. Data: tracking.csv.",
             fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94), h_pad=2.5, w_pad=2)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"tracking.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    machine, cases = load(a.reports)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sp = speedups(cases)
    fmt = lambda v, d=3: f"{v:.{d}f}" if isinstance(v, (int, float)) else ""   # noqa: E731
    order = sorted(cases, key=lambda k: (cases[k]["group"], cases[k]["device"], k[0],
                                         LEVEL_ORDER.index(cases[k]["level"]), k[1]))
    with open(out / "tracking.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "impl", "group", "device", "level", "P", "R", "status", "sim_s", "build_s", "h5_s",
                    "stage_in_s", "stage_out_s", "write_s", "wall_s", "speedup_vs_scalar", "speedup_vs_before",
                    "h5_mib", "gpu_peak_mib", "rss_peak_mib", "report"])
        for k in order:
            c = cases[k]
            ph = c.get("phases") or {}
            vs = sp.get(k, (None, None))
            w.writerow([c["name"], c["impl"], c["group"], c["device"], c["level"], c["P"], c["R"], c["status"],
                        fmt(c.get("sim_s")), fmt(ph.get("build_s")), fmt(ph.get("h5_s")), fmt(ph.get("stage_in_s")),
                        fmt(ph.get("stage_out_s")), fmt(c.get("write_s")), fmt(c.get("wall_s")),
                        fmt(vs[0]), fmt(vs[1]), fmt(c.get("h5_mib"), 1), fmt(c.get("gpu_peak_mib"), 1),
                        fmt(c.get("rss_peak_mib"), 1), c["report"]])
    draw(cases, machine, out)
    print(machine)
    for k in order:
        c = cases[k]
        line = f"  {c['name']:18s} {c['impl']:15s} {c['status'][:30]:30s}"
        if ok(c):
            line += f" total {c['wall_s']:8.2f} s"
            if c.get("sim_s") is not None:
                line += f" (sim {c['sim_s']:.2f}, write {c.get('write_s', 0):.2f})"
            vs = sp.get(k, (None, None))
            if vs[0]:
                line += f"  {vs[0]:.2f}x vs scalar"
            if vs[1]:
                line += f"  {vs[1]:.1f}x vs before"
        print(line)
    print(f"written: {out / 'tracking.png'}, .pdf, tracking.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
