#!/usr/bin/env python3
"""Figures for the version runs (plan_evotorch.md, Step 29; tests/version_runs.py).

  versions_original.png/.pdf   item 3: seconds per generation of the original scalar
                               code, today's scalar and today's CUDA per population size
                               (n = 11), and the n = 50 point, with speed-ups
  versions_waterfall_a.png/.pdf item 4: one bar per commit (median seconds per whole EA
                               generation, IQR), with the speed-up against the previous
                               commit and against the original; and the total wall time
  versions.csv                 every job's numbers

Per generation = the interval between run_ea's "[Gen k/N] Saved" lines (generations
2..N-1), the same measure for every commit. Failed or disturbed jobs are shown as
missing (disturbed: unless --include-disturbed) and listed.

Usage
-----
    python -m analysis.plot_versions data/runtime_reports/<versions report> [...] --out <folder>
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

import analysis.plot_scaling as PS
from analysis.plot_neuron_scaling import INK, INK2, MARKERS, SERIES, SURFACE, _style


class VersionPlotError(ValueError):
    pass


def load(reports, include_disturbed=False):
    machines, jobs, order = {}, {}, []
    for r in reports:
        r = Path(r)
        system = json.loads((r / "system.json").read_text())
        machines.setdefault(PS.machine_of(system)[0], []).append(r.name)
        run = json.loads((r / "run.json").read_text())
        for j in run["jobs"]:
            f = r / "jobs" / f"{j['name']}.json"
            if not f.exists():
                print(f"  NOT RUN {j['name']} ({r.name})")
                continue
            res = json.loads(f.read_text())
            if j["name"] in jobs:
                raise VersionPlotError(f"job {j['name']} in more than one report")
            res["sha"] = j["sha"]
            res["usable"] = res["status"] == "ok" and (include_disturbed or not res.get("disturbance"))
            if res["status"] != "ok":
                print(f"  FAILED {j['name']}: {res['status']}")
            elif res.get("disturbance") and not include_disturbed:
                print(f"  EXCLUDED {j['name']}: machine not free: {res['disturbance']}")
            jobs[j["name"]] = res
            order.append(j["name"])
    if len(machines) > 1:
        raise VersionPlotError("reports from more than one kind of machine: "
                               + "; ".join(f"{m}: {', '.join(v)}" for m, v in machines.items()))
    if not jobs:
        raise VersionPlotError("no finished job in the given reports")
    return next(iter(machines)), [jobs[n] for n in order]


def q(res):
    return [float(x) for x in np.percentile(res["seconds"], [25, 50, 75])]


def draw_original(jobs, machine, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    labels = ["original scalar", "today scalar", "today CUDA"]
    n11 = [j for j in jobs if j["group"] == "original_n11" and j["usable"]]
    n50 = [j for j in jobs if j["group"] == "original_n50" and j["usable"]]
    if not n11 and not n50:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    for i, lab in enumerate(labels):
        pts = sorted((j for j in n11 if j["label"] == lab), key=lambda j: j["P"])
        if not pts:
            continue
        qs = [q(j) for j in pts]
        ax.errorbar([j["P"] for j in pts], [x[1] for x in qs],
                    yerr=[[x[1] - x[0] for x in qs], [x[2] - x[1] for x in qs]],
                    color=SERIES[i], marker=MARKERS[i], linewidth=1.6, capsize=3, label=lab)
    Ps = sorted({j["P"] for j in n11})
    ax.set_xscale("log")
    ax.set_yscale("log")
    if Ps:
        ax.xaxis.set_major_locator(FixedLocator(Ps))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    _style(ax, "n = 11, lookup config, R = 300", "population size P (log)", "seconds per EA generation (log)")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK)
    ax = axes[1]
    xs, ys = [], []
    for i, lab in enumerate(labels):
        j = next((j for j in n50 if j["label"] == lab), None)
        if j:
            xs.append(i)
            ys.append(q(j)[1])
            ax.bar(i, q(j)[1], color=SERIES[i], width=0.6)
            ax.text(i, q(j)[1], f"{q(j)[1]:.3g} s", ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
    _style(ax, "n = 50, random genomes, P = 2, R = 300", "", "seconds per EA generation (log)")
    fig.suptitle("The original code (8e17e694c7) against today's code", x=0.01, ha="left",
                 fontsize=11, color=INK, fontweight="bold")
    fig.text(0.01, 0.005, f"{machine}. Median of generations 2-4 (IQR), each the interval between run_ea's "
             f"'[Gen k/N] Saved' lines. Data: versions.csv.", fontsize=6.5, color=INK2, wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    written = []
    for ext in ("png", "pdf"):
        fig.savefig(out / f"versions_original.{ext}", dpi=200, facecolor=SURFACE)
        written.append(out / f"versions_original.{ext}")
    plt.close(fig)
    return written


def waterfall_rows(jobs):
    wf = [j for j in jobs if j["group"] == "waterfall_a"]
    rows, prev, first = [], None, None
    for j in wf:
        med = q(j)[1] if j["usable"] else None
        if first is None and med is not None:
            first = med
        rows.append(dict(job=j, med=med,
                         vs_prev=(prev / med if med and prev else None),
                         vs_original=(first / med if med and first else None)))
        if med is not None:
            prev = med
    return rows


def draw_waterfall(jobs, machine, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = waterfall_rows(jobs)
    if not rows:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), facecolor=SURFACE)
    import textwrap
    names = [r['job']['sha'][:7] + "\n" + "\n".join(textwrap.wrap(r['job']['label'], 13)) for r in rows]
    for ax, key, title, ylabel in ((axes[0], "gen", "Seconds per EA generation (median, IQR)", "seconds (log)"),
                                   (axes[1], "wall", "Total wall time (5 generations)", "seconds (log)")):
        for i, r in enumerate(rows):
            j = r["job"]
            col = SERIES[0] if j["backend"] == "scalar" else SERIES[1]
            if not j["usable"]:
                ax.text(i, 1, "failed" if j["status"] != "ok" else "disturbed", ha="center", fontsize=7,
                        color=INK2, rotation=90)
                continue
            if key == "gen":
                lo, med, hi = q(j)
                ax.bar(i, med, color=col, width=0.65, yerr=[[med - lo], [hi - med]], capsize=3)
                txt = f"{med:.3g} s"
                if r["vs_prev"]:
                    txt += f"\n×{r['vs_prev']:.2g} prev"
                if r["vs_original"] and i:
                    txt += f"\n×{r['vs_original']:.3g} orig"
                ax.text(i, med * 1.08, txt, ha="center", va="bottom", fontsize=6.5, color=INK)
            else:
                ax.bar(i, j["wall_s"], color=col, width=0.65)
                ax.text(i, j["wall_s"] * 1.08, f"{j['wall_s']:.0f} s", ha="center", va="bottom", fontsize=6.5, color=INK)
        ax.set_yscale("log")
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(names, fontsize=6.5)
        _style(ax, title, "", ylabel)
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color=SERIES[0], label="scalar CPU"), Patch(color=SERIES[1], label="tensor CUDA")],
                   fontsize=7, frameon=False, labelcolor=INK, loc="upper right")
    fig.suptitle("Version waterfall: production lookup config (P 260, R 300, n = 11)", x=0.01, ha="left",
                 fontsize=11, color=INK, fontweight="bold")
    fig.text(0.01, 0.005, f"{machine}. One run per commit, 5 generations; per generation = interval between "
             f"run_ea's '[Gen k/N] Saved' lines, generations 2-4. Data: versions.csv.",
             fontsize=6.5, color=INK2, wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    written = []
    for ext in ("png", "pdf"):
        fig.savefig(out / f"versions_waterfall_a.{ext}", dpi=200, facecolor=SURFACE)
        written.append(out / f"versions_waterfall_a.{ext}")
    plt.close(fig)
    return written


def write_csv(jobs, out):
    wf = {id(r["job"]): r for r in waterfall_rows(jobs)}
    with open(out / "versions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["job", "group", "label", "commit", "backend", "P", "status", "usable", "wall_s",
                    "median_s_per_gen", "q25", "q75", "speedup_vs_previous", "speedup_vs_original",
                    "final_mean_lifespan", "disturbance"])
        for j in jobs:
            qs = q(j) if j["status"] == "ok" else ["", "", ""]
            r = wf.get(id(j), {})
            w.writerow([j["name"], j["group"], j["label"], j["sha"][:10], j["backend"], j["P"], j["status"],
                        j["usable"], f"{j['wall_s']:.2f}", *(f"{x:.4f}" if x != "" else "" for x in (qs[1], qs[0], qs[2])),
                        f"{r['vs_prev']:.3f}" if r.get("vs_prev") else "",
                        f"{r['vs_original']:.3f}" if r.get("vs_original") else "",
                        (j.get("final") or {}).get("mean", ""), j.get("disturbance", "")])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-disturbed", action="store_true")
    a = ap.parse_args(argv)
    machine, jobs = load(a.reports, a.include_disturbed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(jobs, out)
    written = draw_original(jobs, machine, out) + draw_waterfall(jobs, machine, out)
    print(machine)
    for r in waterfall_rows(jobs):
        j = r["job"]
        print(f"  {j['sha'][:7]} {j['label']:32s} " + (f"{r['med']:8.2f} s/gen" if r["med"] else "   (no data)")
              + (f"  x{r['vs_prev']:.2f} vs previous" if r["vs_prev"] else "")
              + (f"  x{r['vs_original']:.1f} vs original" if r["vs_original"] else ""))
    print("written: " + ", ".join(str(p) for p in written) + f", {out / 'versions.csv'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except VersionPlotError as e:
        sys.exit(f"ERROR: {e}")
