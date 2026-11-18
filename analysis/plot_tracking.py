#!/usr/bin/env python3
"""What tracking costs in the tensor code (plan_evotorch.md, Step 31; tests/tracking_bench.py).

  (a) per case: simulation, HDF5 writing and the rest (compile warm-up, setup) stacked;
      grouped by case group (A / B / C) and device, one bar per tracking level
  (b) where the writing goes: seconds per HDF5 save function, for the largest finished
      case of each tracking level
  (c) the simulation's slowdown from tracking (sim_s / sim_s without tracking), i.e. the
      tracker's own cost on the device, separate from writing
The scalar code (device "scalar") writes during the generation, so it has a total only;
tracking.csv gives each tensor case's speed-up against it (scalar total / tensor total,
same group and level).

Failed and timed-out cases are marked, not hidden. Output: tracking.png / .pdf,
tracking.csv.

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


def load(reports):
    machines, cases = set(), {}
    for r in reports:
        r = Path(r)
        machines.add(PS.machine_of(json.loads((r / "system.json").read_text()))[0])
        for f in sorted((r / "cases").glob("*.json")):
            c = json.loads(f.read_text())
            if c["name"] in cases:
                raise SystemExit(f"ERROR: case {c['name']} in more than one report")
            cases[c["name"]] = c
    if len(machines) != 1:
        raise SystemExit(f"ERROR: reports from {len(machines)} machines: {sorted(machines)}")
    if not cases:
        raise SystemExit("ERROR: no finished case")
    return next(iter(machines)), cases


def ok(c):
    return c.get("status") == "ok"


def slowdowns(cases):
    out = {}
    for c in cases.values():
        base = next((b for b in cases.values() if b["group"] == c["group"] and b["device"] == c["device"]
                     and b["level"] == "off" and ok(b)), None)
        if ok(c) and c.get("sim_s") and base and base.get("sim_s"):
            out[c["name"]] = c["sim_s"] / base["sim_s"]
    return out


def vs_scalar(cases):
    """tensor case -> scalar wall / tensor wall (same group and level), both ok."""
    out = {}
    for c in cases.values():
        if c["device"] == "scalar" or not ok(c):
            continue
        sc = next((s for s in cases.values() if s["device"] == "scalar" and s["group"] == c["group"]
                   and s["level"] == c["level"]), None)
        if sc and ok(sc) and c.get("wall_s"):
            out[c["name"]] = sc["wall_s"] / c["wall_s"]
    return out


def draw(cases, machine, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    groups = sorted({(c["group"], c["device"]) for c in cases.values()})
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), facecolor=SURFACE, gridspec_kw={"width_ratios": [2.2, 1.3, 1]})
    ax = axes[0]
    x, ticks, labels = 0, [], []
    for g, dev in groups:
        for lv in LEVEL_ORDER:
            c = next((c for c in cases.values() if (c["group"], c["device"], c["level"]) == (g, dev, lv)), None)
            if c is None:
                continue
            if ok(c) and dev == "scalar":
                ax.bar(x, c["wall_s"], 0.75, color=SERIES[2], edgecolor=SURFACE, linewidth=1.5)
                ax.text(x, c["wall_s"], f"{c['wall_s']:.3g} s", ha="center", va="bottom", fontsize=6, color=INK)
            elif ok(c):
                sim, wr = c.get("sim_s", 0), c.get("write_s", 0)
                rest = max(c.get("wall_s", 0) - sim - wr, 0)
                ax.bar(x, sim, 0.75, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5)
                ax.bar(x, wr, 0.75, bottom=sim, color=SERIES[1], edgecolor=SURFACE, linewidth=1.5)
                ax.bar(x, rest, 0.75, bottom=sim + wr, color=INK2, edgecolor=SURFACE, linewidth=1.5)
                ax.text(x, sim + wr + rest, f"{sim + wr + rest:.3g} s", ha="center", va="bottom", fontsize=6, color=INK)
            else:
                ax.text(x, 0, ("timeout" if str(c["status"]).startswith("timeout") else
                               "skipped" if str(c["status"]).startswith("skipped") else "failed"),
                        rotation=90, ha="center", va="bottom", fontsize=7, color=INK2)
            ticks.append(x)
            labels.append(f"{LEVEL_NAME[lv]}")
            x += 1
        ax.axvline(x - 0.5, color=INK2, linewidth=0.5)
        ax.text(x - 0.6, 1.0, f"{g} {dev}", transform=ax.get_xaxis_transform(), ha="right", va="top",
                fontsize=7.5, color=INK, fontweight="bold")
        x += 0.6
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=60, fontsize=6.5)
    ax.set_yscale("log")
    ax.legend(handles=[Patch(color=SERIES[0], label="simulation (no compile)"), Patch(color=SERIES[1], label="HDF5 writing"),
                       Patch(color=INK2, label="rest (compile warm-up, setup)"),
                       Patch(color=SERIES[2], label="scalar code: total (writes during the run)")], fontsize=7, frameon=False,
              labelcolor=INK, loc="upper left")
    _style(ax, "(a) Time per case", "", "seconds (log)")

    ax = axes[1]
    rows = []
    for lv in LEVEL_ORDER[1:]:
        cs = [c for c in cases.values() if c["level"] == lv and ok(c) and c.get("breakdown")]
        if not cs:
            continue
        c = max(cs, key=lambda c: c["P"] * c["R"])
        for fn, d in sorted(c["breakdown"].items(), key=lambda kv: -kv[1]["s"])[:4]:
            rows.append((f"{c['name']}: {fn.replace('save_', '').replace('_to_hdf5', '')} ({d['calls']}×)", d["s"]))
    if rows:
        ax.barh(range(len(rows)), [r[1] for r in rows], color=SERIES[1])
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows], fontsize=6)
        ax.invert_yaxis()
        ax.set_xscale("log")
    _style(ax, "(b) Writing per save function (largest case per level)", "seconds (log)", "")

    ax = axes[2]
    sd = slowdowns(cases)
    names = [n for n in sorted(sd, key=lambda n: (cases[n]["group"], cases[n]["device"],
                                                   LEVEL_ORDER.index(cases[n]["level"]))) if cases[n]["level"] != "off"]
    ax.barh(range(len(names)), [sd[n] for n in names], color=SERIES[0])
    ax.axvline(1, color=INK2, linewidth=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{cases[n]['group']} {cases[n]['device']} {LEVEL_NAME[cases[n]['level']]}" for n in names],
                       fontsize=6.5)
    ax.invert_yaxis()
    for i, n in enumerate(names):
        ax.text(sd[n], i, f" ×{sd[n]:.2f}", va="center", fontsize=6.5, color=INK)
    _style(ax, "(c) Simulation slowdown from tracking", "sim time / sim time without tracking", "")
    fig.suptitle("Tracking in the tensor code: simulation, writing and the tracker's own cost", x=0.01, ha="left",
                 fontsize=11, color=INK, fontweight="bold")
    fig.text(0.01, 0.005, f"{machine}. A: 10 lookup elites x 300 runs (n = 11); B: 260 genomes x 300 runs (n = 11); "
             f"C: 10 large-run elites x 100 runs (n = 50). Data: tracking.csv.", fontsize=6.5, color=INK2, wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
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
    sd = slowdowns(cases)
    with open(out / "tracking.csv", "w", newline="") as f:
        w = csv.writer(f)
        vs = vs_scalar(cases)
        w.writerow(["case", "group", "device", "level", "P", "R", "status", "sim_s", "write_s", "wall_s",
                    "speedup_vs_scalar", "sim_slowdown_vs_off", "write_share_of_wall", "h5_mib", "gpu_peak_mib",
                    "rss_peak_mib"])
        for c in cases.values():
            wall = c.get("wall_s") or 0
            w.writerow([c["name"], c["group"], c["device"], c["level"], c["P"], c["R"], c["status"],
                        *(f"{c[k]:.3f}" if isinstance(c.get(k), (int, float)) else "" for k in ("sim_s", "write_s", "wall_s")),
                        f"{vs[c['name']]:.3f}" if c["name"] in vs else "",
                        f"{sd[c['name']]:.3f}" if c["name"] in sd else "",
                        f"{c.get('write_s', 0) / wall:.3f}" if ok(c) and wall else "",
                        *(f"{c[k]:.1f}" if isinstance(c.get(k), (int, float)) else "" for k in ("h5_mib", "gpu_peak_mib", "rss_peak_mib"))])
    draw(cases, machine, out)
    vs = vs_scalar(cases)
    print(machine)
    for c in cases.values():
        line = f"  {c['name']:16s} {c['status'][:40]:40s}"
        if ok(c):
            line += f" total {c['wall_s']:8.2f} s"
            if c.get("sim_s") is not None:
                line += f" (sim {c['sim_s']:.2f}, write {c['write_s']:.2f})"
            line += f"  {c.get('h5_mib', 0):8.1f} MiB"
            if c["name"] in vs:
                line += f"  {vs[c['name']]:.1f}x faster than scalar"
        print(line)
    print(f"written: {out / 'tracking.png'}, .pdf, tracking.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
