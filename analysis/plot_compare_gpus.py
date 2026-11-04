#!/usr/bin/env python3
"""The same CUDA scaling configs on two (or more) machines, side by side.

analysis/plot_scaling.py draws one machine per figure and refuses to mix them; this
puts machines next to each other. Each machine is a set of report folders that
plot_scaling accepts as one machine (same hardware, same configs), read with its own
rules (failed and busy runs left out, compile warm-up subtracted). Two figures, one
panel per dimension (neurons, population, runs), one line per machine:

  compare_us_per_tick.png/.pdf   us per worm-tick of the simulation (compile excluded),
                                 median over generations 1+ with the interquartile range
  compare_total_time.png/.pdf    the whole run of each config: its wall time in
                                 summary.csv (start-up, compile, generation 0, all
                                 generations, mutation, selection, HDF5)

Hollow markers: runs queued (not all fit on that device at once). compare.csv holds
every number, including the ratio of the first machine to each other one.

Usage
-----
    python -m analysis.plot_compare_gpus \\
        --machine "RTX 3070" data/runtime_reports/<A> data/runtime_reports/<B> \\
        --machine "A100" data/runtime_reports/<C> --out analysis/out/compare_3070_a100
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

import analysis.plot_scaling as PS
from analysis.plot_neuron_scaling import INK, INK2, MARKERS, SERIES, SURFACE, _style

DIM_ORDER = ["neurons", "population", "runs"]


class CompareError(ValueError):
    pass


def load_machine(label, reports, include_disturbed=False):
    """-> (title, {(dim, x): entry}) for one machine's CUDA points; entry holds the
    us/tick quartiles, the wall times, queued. plot_scaling's checks apply."""
    machines, hosts, points = {}, set(), []
    for r in reports:
        (machine, host), pts, excl = PS.load_report(r, include_disturbed)
        machines.setdefault(machine, []).append(Path(r).name)
        hosts.add(host)
        for name, why in excl:
            print(f"  [{label}] EXCLUDED {name}: {why}")
        points += [p for p in pts if p["backend"] == "cuda"]
    if len(machines) > 1:
        raise CompareError(f"{label}: reports from more than one kind of machine: "
                           + "; ".join(f"{m}: {', '.join(rs)}" for m, rs in machines.items()))
    if not points:
        raise CompareError(f"{label}: no usable CUDA scaling point in {reports}")
    PS.check_consistent(points)
    wall = {}
    for r in reports:
        with open(Path(r) / "summary.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        if rows and "wall_s" not in rows[0]:
            raise CompareError(f"{label}: {Path(r).name}/summary.csv has no wall_s column")
        for row in rows:
            wall.setdefault(row["config"], []).append(row)
    out = {}
    for e in [e for v in PS.series_of(points).values() for e in v]:
        names = [p["config"] for p in points if (p["dim"], p["x"]) == (e["dim"], e["x"])]
        walls = [float(row["wall_s"]) for n in names for row in wall.get(n, [])
                 if row["status"] == "ok" and row["wall_s"]]
        out[(e["dim"], e["x"])] = {"us_q": e["us_q"], "wall": walls, "queued": e["queued"],
                                   "width": e["width"], "disturbed": e["disturbed"]}
    return f"{label}: {next(iter(machines))} ({', '.join(sorted(hosts))})", out


def _axes(ax, xs, title, ylabel, dim):
    from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter, NullLocator
    ax.set_xscale("log")
    ax.set_yscale("log")
    if len(xs) > 8:
        kept = xs[::2]
        if kept[-1] != xs[-1]:
            kept[-1] = xs[-1]
        xs = kept
    plain = FuncFormatter(lambda v, _: f"{v:g}")
    ax.xaxis.set_major_locator(FixedLocator(xs))
    ax.xaxis.set_major_formatter(plain)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(plain)
    ax.yaxis.set_minor_formatter(NullFormatter())
    _style(ax, title, PS.DIMS[dim][1] + " (log)", ylabel)


def draw(machines, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if len(machines) > len(SERIES):
        raise CompareError(f"at most {len(SERIES)} machines per figure")
    dims = [d for d in DIM_ORDER if any(k[0] == d for _t, pts in machines for k in pts)]
    written = []
    for kind, title, ylabel in (
            ("us_per_tick", "Time per worm-tick (simulation, compile excluded)",
             "µs per worm-tick (log), median, IQR"),
            ("total_time", "Total time of each config's run",
             "wall seconds per config run (log)")):
        fig, axes = plt.subplots(1, len(dims), figsize=(5.2 * len(dims), 4.6), facecolor=SURFACE,
                                 squeeze=False)
        for ax, dim in zip(axes[0], dims):
            xs_all = sorted({k[1] for _t, pts in machines for k in pts if k[0] == dim})
            for i, (label, pts) in enumerate(machines):
                xs = sorted(x for d, x in pts if d == dim)
                if not xs:
                    continue
                es = [pts[(dim, x)] for x in xs]
                if kind == "us_per_tick":
                    ys = [e["us_q"][1] for e in es]
                    err = [[e["us_q"][1] - e["us_q"][0] for e in es], [e["us_q"][2] - e["us_q"][1] for e in es]]
                else:
                    ys = [float(np.median(e["wall"])) if e["wall"] else np.nan for e in es]
                    err = None
                ax.errorbar(xs, ys, yerr=err, color=SERIES[i], marker=MARKERS[i], markersize=5.5,
                            linewidth=1.6, capsize=3, elinewidth=1, label=label.split(":")[0])
                qx = [x for x, e in zip(xs, es) if e["queued"]]
                qy = [y for y, e in zip(ys, es) if e["queued"]]
                if qx:
                    ax.plot(qx, qy, linestyle="none", marker=MARKERS[i], markersize=5.5,
                            markerfacecolor=SURFACE, markeredgecolor=SERIES[i], markeredgewidth=1.5,
                            zorder=4)
            _axes(ax, xs_all, {"neurons": "Neurons n", "population": "Population P",
                               "runs": "Runs per genome R"}[dim], ylabel if dim == dims[0] else "", dim)
            ax.legend(fontsize=8, frameon=False, labelcolor=INK, loc="best")
        fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK, fontweight="bold")
        fig.text(0.01, 0.005,
                 "  |  ".join(t for t, _p in machines)
                 + ".  Base: P = 100, R = 100, n = 11 apart from the varied key; hollow: runs queued "
                   "(not all fit on that device at once). Data: compare.csv.",
                 fontsize=6.5, color=INK2, wrap=True)
        fig.tight_layout(rect=(0, 0.05, 1, 0.94))
        for ext in ("png", "pdf"):
            fig.savefig(out / f"compare_{kind}.{ext}", dpi=200, facecolor=SURFACE)
            written.append(out / f"compare_{kind}.{ext}")
        plt.close(fig)
    return written


def write_csv(machines, out):
    keys = sorted({k for _t, pts in machines for k in pts}, key=lambda k: (DIM_ORDER.index(k[0]), k[1]))
    first = machines[0][0].split(":")[0]
    with open(out / "compare.csv", "w", newline="") as f:
        w = csv.writer(f)
        head = ["dimension", "x"]
        for t, _p in machines:
            m = t.split(":")[0]
            head += [f"{m} us_per_tick", f"{m} wall_s", f"{m} queued", f"{m} width"]
        for t, _p in machines[1:]:
            m = t.split(":")[0]
            head += [f"{first}/{m} us_per_tick", f"{first}/{m} wall_s"]
        w.writerow(head)
        for k in keys:
            row = [k[0], k[1]]
            vals = []
            for _t, pts in machines:
                e = pts.get(k)
                us = e["us_q"][1] if e else None
                wall = float(np.median(e["wall"])) if e and e["wall"] else None
                vals.append((us, wall))
                row += ["" if us is None else f"{us:.4f}", "" if wall is None else f"{wall:.2f}",
                        "" if not e else e["queued"], "" if not e or e["width"] is None else e["width"]]
            for us, wall in vals[1:]:
                row += [f"{vals[0][0] / us:.3f}" if us and vals[0][0] else "",
                        f"{vals[0][1] / wall:.3f}" if wall and vals[0][1] else ""]
            w.writerow(row)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--machine", nargs="+", action="append", required=True, metavar=("LABEL", "REPORT"),
                    help="a label, then that machine's report folders; give it once per machine")
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-disturbed", action="store_true",
                    help="use runs flagged as on a busy machine too (see plot_scaling)")
    a = ap.parse_args(argv)
    if len(a.machine) < 2:
        raise CompareError("give --machine at least twice")
    if any(len(m) < 2 for m in a.machine):
        raise CompareError("each --machine needs a label and at least one report folder")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    machines = [load_machine(m[0], m[1:], a.include_disturbed) for m in a.machine]
    write_csv(machines, out)
    written = draw(machines, out)
    for t, pts in machines:
        print(f"{t}: {len(pts)} CUDA points")
    print("written: " + ", ".join(str(p) for p in written) + f", {out / 'compare.csv'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (CompareError, PS.ScalingError) as e:
        sys.exit(f"ERROR: {e}")
