#!/usr/bin/env python3
"""Break-even of the CUDA tensor code against the scalar code (plan_evotorch.md, Step 28).

Reads run_selected report folders of ONE machine that contain the break-even sweeps
(configs/sweeps/breakeven_<family>_<scalar|cuda>.yaml) and draws, per family (lookup,
random, n50):

  (a) seconds per generation, median of generations 1-3 (IQR as error bars):
      scalar evaluation; CUDA simulation without the compile warm-up (Step 25); and
      both backends' whole EA generation (mutation + evaluation + selection + HDF5)
  (b) total wall time of each config's run (summary.csv wall_s: start-up, compile,
      all four generations), with the CUDA compile-cache state at the start
  (c) speed-up scalar / CUDA for both, with the break-even (speed-up 1) interpolated
      log-log between the measured points

`--scaling REPORT...` adds the Step 22 population sweep (scaling_population_*, same
base as the random family but elite_size 10) as a separate reference series in the
random family's panels; it is never merged with the new points.

Usage
-----
    python -m analysis.plot_breakeven data/runtime_reports/<report> [<report> ...] \\
        --out analysis/out/breakeven [--scaling data/runtime_reports/<scaling report>]
"""

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import yaml

import analysis.plot_scaling as PS
from analysis.plot_neuron_scaling import INK, INK2, MARKERS, SERIES, SURFACE, _style

ROW = re.compile(r"^breakeven_(lookup|random|n50)_(scalar|cuda) \[population_size=(\d+)\]$")
FAMILIES = {
    "lookup": "production lookup config, n = 11, R = 300",
    "random": "random genomes (Step 22 base), n = 11, R = 100",
    "n50": "random genomes, n = 50, R = 300",
}
BACKEND_COLUMN = {"scalar": "scalar", "cuda": "tensor cuda"}
# Keys that may differ between a family's scalar and CUDA configs.
BACKEND_KEYS = ("experiment.evaluator", "experiment.simulation_name")


class BreakevenError(ValueError):
    pass


def _quart(xs):
    q = np.percentile(xs, [25, 50, 75])
    return [float(v) for v in q]


def load_report(report, include_disturbed=False):
    """-> (machine, points, excluded). One point per usable break-even config."""
    report = Path(report)
    system = json.loads((report / "system.json").read_text())
    with open(report / "summary.csv", newline="") as f:
        summary = list(csv.DictReader(f))
    with open(report / "generations.csv", newline="") as f:
        gens = list(csv.DictReader(f))
    points, excluded = [], []
    for row in summary:
        m = ROW.match(row["config"])
        if not m:
            continue
        fam, be, p = m.group(1), m.group(2), int(m.group(3))
        name = row["config"]
        if row["status"] != "ok":
            excluded.append((name, f"status {row['status']}"))
            continue
        if not row.get("backend", "").startswith(BACKEND_COLUMN[be]):
            raise BreakevenError(f"{name} in {report.name}: the run's backend is "
                                 f"{row.get('backend')!r}, the sweep file says {BACKEND_COLUMN[be]!r}")
        if be == "cuda":
            log = report / row["log"]
            if not (log.is_file() and PS.COMPILE_FIX_MARKER.search(log.read_text(errors="replace"))):
                excluded.append((name, "no Step 18 compile fix in the log"))
                continue
        disturbed = row.get("disturbance", "")
        if disturbed and not include_disturbed:
            excluded.append((name, f"machine not free: {disturbed}"))
            continue
        g = [r for r in gens if r["order"] == row["order"] and int(r["generation"]) >= 1
             and r.get("evaluate_s")]
        if not g:
            raise BreakevenError(f"{name} in {report.name} ran ok but has no [EA timing] line "
                                 f"for generation 1+")
        if be == "cuda" and not all(r.get("simulate_s") for r in g):
            raise BreakevenError(f"{name} in {report.name}: no compile warm-up columns (a report "
                                 f"from before plan Step 25)")
        cfg_path = report / "configs" / Path(row["config_path"]).name
        if not cfg_path.is_file():
            raise BreakevenError(f"{name} in {report.name}: expanded config {cfg_path} missing")
        cfg = yaml.safe_load(cfg_path.read_text())
        if int(PS._get(cfg, "experiment.population_size")) != p:
            raise BreakevenError(f"{name}: the expanded config has another population_size")
        sim = [float(r["simulate_s"] if be == "cuda" else r["evaluate_s"]) for r in g]
        whole = [float(r["genomes_s"]) + float(r["evaluate_s"]) + float(r["select_save_s"]) for r in g]
        points.append(dict(family=fam, backend=be, P=p, config=name, report=report.name, cfg=cfg,
                           sim_q=_quart(sim), whole_q=_quart(whole), wall=float(row["wall_s"]),
                           cache=row.get("compile_cache_at_start", ""), disturbed=bool(disturbed),
                           workers=row.get("cpu_workers", "")))
    return PS.machine_of(system), points, excluded


def check_pairs(points):
    """Every family/P present for both backends; a family's scalar and CUDA configs differ
    only in BACKEND_KEYS; one point per (family, backend, P)."""
    seen = {}
    for p in points:
        key = (p["family"], p["backend"], p["P"])
        if key in seen:
            raise BreakevenError(f"{p['config']} appears twice ({seen[key]['report']}, {p['report']}); "
                                 f"give one report per point")
        seen[key] = p
    for fam in sorted({p["family"] for p in points}):
        ps = sorted({p["P"] for p in points if p["family"] == fam})
        for P in ps:
            missing = [be for be in ("scalar", "cuda") if (fam, be, P) not in seen]
            if missing:
                raise BreakevenError(f"breakeven_{fam} P = {P}: no usable {missing[0]} point "
                                     f"(each population size needs both backends)")
            a, b = (dict(seen[(fam, be, P)]["cfg"]) for be in ("scalar", "cuda"))
            a, b = _strip(a), _strip(b)
            if a != b:
                raise BreakevenError(f"breakeven_{fam} P = {P}: the scalar and CUDA configs differ "
                                     f"beyond the backend: {PS._first_diff(a, b)}")


def _strip(cfg):
    import copy
    cfg = copy.deepcopy(cfg)
    for k in BACKEND_KEYS:
        PS._drop(cfg, k)
    return cfg


def crossings(Ps, speedup):
    """Population sizes where speed-up (scalar time / CUDA time) crosses 1, interpolated
    log-log between neighbours; or a statement when it never does."""
    out = []
    for (p1, s1), (p2, s2) in zip(zip(Ps, speedup), zip(Ps[1:], speedup[1:])):
        if (s1 - 1) * (s2 - 1) < 0:
            l1, l2 = math.log(s1), math.log(s2)
            out.append(math.exp(math.log(p1) + (0 - l1) * (math.log(p2) - math.log(p1)) / (l2 - l1)))
        elif s2 == 1:
            out.append(float(p2))
    if out:
        return out, ", ".join(f"P ≈ {x:.1f}" for x in out)
    if all(s > 1 for s in speedup):
        return [], f"CUDA faster at every measured P (from P = {Ps[0]})"
    return [], f"scalar faster at every measured P (up to P = {Ps[-1]})"


def series(points, fam):
    Ps = sorted({p["P"] for p in points if p["family"] == fam})
    by = {(p["backend"], p["P"]): p for p in points if p["family"] == fam}
    return Ps, {be: [by[(be, P)] for P in Ps] for be in ("scalar", "cuda")}


def summarise(points):
    rows, results = [], {}
    for fam in [f for f in FAMILIES if any(p["family"] == f for p in points)]:
        Ps, s = series(points, fam)
        sp_sim = [a["sim_q"][1] / b["sim_q"][1] for a, b in zip(s["scalar"], s["cuda"])]
        sp_wall = [a["wall"] / b["wall"] for a, b in zip(s["scalar"], s["cuda"])]
        results[fam] = dict(Ps=Ps, s=s, sp_sim=sp_sim, sp_wall=sp_wall,
                            be_sim=crossings(Ps, sp_sim), be_wall=crossings(Ps, sp_wall))
        for i, P in enumerate(Ps):
            a, b = s["scalar"][i], s["cuda"][i]
            rows.append([fam, P, f"{a['sim_q'][1]:.4f}", f"{b['sim_q'][1]:.4f}", f"{a['whole_q'][1]:.4f}",
                         f"{b['whole_q'][1]:.4f}", f"{a['wall']:.2f}", f"{b['wall']:.2f}",
                         f"{sp_sim[i]:.3f}", f"{sp_wall[i]:.3f}", a["workers"], b["cache"]])
    return rows, results


def draw(results, machine, out, scaling=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    written = []
    plain = FuncFormatter(lambda v, _: f"{v:g}")
    for fam, r in results.items():
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), facecolor=SURFACE)
        Ps = r["Ps"]
        ax = axes[0]
        for i, (be, lab, key) in enumerate((("scalar", "scalar: evaluation", "sim_q"),
                                            ("cuda", "CUDA: simulation (no compile)", "sim_q"))):
            q = [p[key] for p in r["s"][be]]
            ax.errorbar(Ps, [x[1] for x in q], yerr=[[x[1] - x[0] for x in q], [x[2] - x[1] for x in q]],
                        color=SERIES[i], marker=MARKERS[i], linewidth=1.6, capsize=3, label=lab)
            w = [p["whole_q"][1] for p in r["s"][be]]
            ax.plot(Ps, w, color=SERIES[i], linestyle="--", linewidth=1, label=f"{be}: whole EA generation")
        if scaling and fam == "random":
            for i, be in enumerate(("scalar", "cuda")):
                pts = sorted((p for p in scaling if p["backend"] == be), key=lambda p: p["x"])
                if pts:
                    ax.plot([p["x"] for p in pts], [float(np.median(p["evaluate"])) for p in pts],
                            linestyle="none", marker=MARKERS[i], markerfacecolor=SURFACE,
                            markeredgecolor=SERIES[i], label=f"Step 22 {be} (elite 10)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        _style(ax, "(a) Seconds per generation (gens 1-3, median, IQR)", "population size P (log)", "seconds (log)")
        ax.legend(fontsize=7, frameon=False, labelcolor=INK)
        ax = axes[1]
        for i, be in enumerate(("scalar", "cuda")):
            ax.plot(Ps, [p["wall"] for p in r["s"][be]], color=SERIES[i], marker=MARKERS[i], linewidth=1.6,
                    label=f"{be}")
        caches = sorted({p["cache"] for p in r["s"]["cuda"] if p["cache"]})
        ax.set_xscale("log")
        ax.set_yscale("log")
        _style(ax, "(b) Total wall time per config (4 generations)", "population size P (log)", "seconds (log)")
        ax.legend(fontsize=7, frameon=False, labelcolor=INK,
                  title=f"CUDA compile cache at start: {', '.join(caches) or 'unknown'}", title_fontsize=6.5)
        ax = axes[2]
        ax.plot(Ps, r["sp_sim"], color=SERIES[0], marker=MARKERS[0], linewidth=1.6, label="per generation (simulation)")
        ax.plot(Ps, r["sp_wall"], color=SERIES[1], marker=MARKERS[1], linewidth=1.6, label="total wall time")
        ax.axhline(1, color=INK2, linewidth=0.8)
        for x in r["be_sim"][0]:
            ax.axvline(x, color=SERIES[0], linestyle=":", linewidth=0.9)
        for x in r["be_wall"][0]:
            ax.axvline(x, color=SERIES[1], linestyle=":", linewidth=0.9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        _style(ax, "(c) Speed-up scalar / CUDA (>1: CUDA faster)", "population size P (log)", "speed-up (log)")
        ax.legend(fontsize=7, frameon=False, labelcolor=INK)
        for ax in axes:
            ax.xaxis.set_major_locator(FixedLocator(Ps))
            ax.xaxis.set_major_formatter(plain)
            ax.xaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_major_formatter(plain)
            ax.yaxis.set_minor_formatter(plain)       # log axes label minor ticks too
        fig.suptitle(f"Break-even, {FAMILIES[fam]}", x=0.01, ha="left", fontsize=11, color=INK,
                     fontweight="bold")
        fig.text(0.01, 0.005, f"{machine}. Break-even per generation: {r['be_sim'][1]}; total: "
                 f"{r['be_wall'][1]}. The scalar code runs one genome per worker (min(P, cores) busy). "
                 f"Data: breakeven.csv.", fontsize=6.5, color=INK2, wrap=True)
        fig.tight_layout(rect=(0, 0.05, 1, 0.94))
        for ext in ("png", "pdf"):
            fig.savefig(out / f"breakeven_{fam}.{ext}", dpi=200, facecolor=SURFACE)
            written.append(out / f"breakeven_{fam}.{ext}")
        plt.close(fig)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scaling", nargs="+", default=[],
                    help="Step 22 report folder(s) with scaling_population_*: reference series")
    ap.add_argument("--include-disturbed", action="store_true")
    a = ap.parse_args(argv)
    machines, points = {}, []
    for r in a.reports:
        (machine, host), pts, excl = load_report(r, a.include_disturbed)
        machines.setdefault(machine, []).append(Path(r).name)
        for name, why in excl:
            print(f"  EXCLUDED {name}: {why}")
        points += pts
    if len(machines) > 1:
        raise BreakevenError("reports from more than one kind of machine: "
                             + "; ".join(f"{m}: {', '.join(rs)}" for m, rs in machines.items()))
    if not points:
        raise BreakevenError("no usable break-even point in the given reports")
    check_pairs(points)
    machine = next(iter(machines))
    scaling = []
    for r in a.scaling:
        # Only the scalar and CUDA population points are used. The tensor-CPU rows of a
        # Step 22 report may predate Step 25 (no warm-up columns); they are read with the
        # compile included and dropped below, so they cannot block the report. CUDA rows
        # without warm-up data still raise.
        (m2, _h), pts, _e = PS.load_report(r, a.include_disturbed, include_compile=("cpu",))
        if m2 != machine:
            raise BreakevenError(f"--scaling {Path(r).name} is from another machine ({m2})")
        scaling += [p for p in pts if p["dim"] == "population" and p["backend"] in ("scalar", "cuda")]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, results = summarise(points)
    with open(out / "breakeven.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "P", "scalar_s_per_gen", "cuda_sim_s_per_gen", "scalar_whole_gen_s",
                    "cuda_whole_gen_s", "scalar_wall_s", "cuda_wall_s", "speedup_per_gen",
                    "speedup_wall", "scalar_workers", "cuda_compile_cache_at_start"])
        w.writerows(rows)
    written = draw(results, machine, out, scaling)
    print(f"{machine}")
    for fam, r in results.items():
        print(f"  {fam:7s} break-even per generation: {r['be_sim'][1]}; total wall time: {r['be_wall'][1]}")
    print("written: " + ", ".join(str(p) for p in written) + f", {out / 'breakeven.csv'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (BreakevenError, PS.ScalingError) as e:
        sys.exit(f"ERROR: {e}")
