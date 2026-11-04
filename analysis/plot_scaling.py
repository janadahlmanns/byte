#!/usr/bin/env python3
"""Runtime growth with neurons, population and runs: scalar CPU vs tensor CPU vs CUDA
(plan Step 22d).

Reads report folders of simulate/run_selected.py that ran the nine sweeps
configs/sweeps/scaling_<neurons|population|runs>_<scalar|cpu|cuda>.yaml, and draws, per
dimension (scaling_<dim>.png/.pdf), three log-log panels with one line per backend:

  (a) simulation time per generation (s): the evaluation minus the compile warm-up
      (plan Step 25; random genomes compile new graphs in later generations too);
  (b) us per worm-tick -- (a) divided by the worm-ticks simulated, which removes the
      differing run lengths (scalar and tensor results are statistically equivalent,
      not identical);
  (c) the EA part per generation (creating the genomes + selection and HDF5 writes):
      the same code on every backend, it grows with P.

Points are medians over generations 1+ with the interquartile range (generation 0
holds the compile), the fitted power law t ~ x^k in the legend. scaling_all.png/.pdf is
the 3 x 3 grid of all of it; points.csv and fits.csv hold every number.

All numbers come from run_ea's backend-independent [EA timing] line. Loud, never silent:
  - reports from different hardware raise (one machine is the point of the comparison).
    Hardware = CPU model, the cores the run was allowed, GPU model: two identical
    cluster nodes (TinyGPU tg080, tg081) are the same machine (plan Step 23e);
  - configs must be equal except the varied key, the evaluator block, simulation_name
    and the output folder -- across all points, all backends and all dimensions;
  - a config that ran ok but has no [EA timing] line raises;
  - a tensor point without warm-up data (before Step 25) raises, unless its backend is
    named in --include-compile: then it is plotted with the compile included, marked;
  - failed runs, runs on a busy machine (summary.csv `disturbance`, Step 22e) and tensor
    runs without the Step 18 compile fix are left out and listed (--include-disturbed
    plots busy runs, dashed);
  - points of a sweep file that are missing are listed.

Usage
-----
    python -m analysis.plot_scaling data/runtime_reports/<folder> [...] --out analysis/out/scaling
"""

import argparse
import copy
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

from analysis.plot_neuron_scaling import (COMPILE_FIX_MARKER, INK, INK2, MARKERS,
                                          SERIES, SURFACE, _style, local_exponents,
                                          power_fit, quartiles)

ROOT = Path(__file__).resolve().parent.parent
SWEEP_DIR = ROOT / "configs" / "sweeps"
ROW = re.compile(r"^scaling_(neurons|population|runs)_(scalar|cpu|cuda) \[")

DIMS = {  # dimension -> (dotted config key, axis label, symbol)
    "neurons": ("brain.n_neurons", "neurons n", "n"),
    "population": ("experiment.population_size", "population size P (genomes)", "P"),
    "runs": ("experiment.n_runs", "runs per genome R", "R"),
}
BACKENDS = {  # file tag -> (legend name, what run_selected's `backend` column must say)
    "scalar": ("scalar CPU", "scalar"),
    "cpu": ("tensor CPU", "tensor cpu"),
    "cuda": ("tensor CUDA", "tensor cuda"),
}
# Keys that may differ between points; everything else must be identical.
FREE_KEYS = ["experiment.simulation_name", "experiment.output_folder", "experiment.evaluator"]
METRICS = {  # key -> (panel title, y label)
    "evaluate": ("Simulation time per generation (compile excluded)", "seconds per generation (log)"),
    "us": ("Time per worm-tick", "µs per worm-tick (log)"),
    "ea": ("EA part per generation", "genomes + selection + HDF5, s (log)"),
}


class ScalingError(ValueError):
    pass


def _get(cfg, dotted):
    cur = cfg
    for k in dotted.split("."):
        cur = cur[k]
    return cur


def _drop(cfg, dotted):
    *head, leaf = dotted.split(".")
    cur = cfg
    for k in head:
        cur = cur.get(k, {})
    cur.pop(leaf, None)


# ============================================================
# Reading reports
# ============================================================

def machine_of(system):
    """(hardware key, hostname). Reports before Step 23 have no cpus_allowed: the
    machine's CPU count stands in (they ran unrestricted)."""
    gpus = system.get("gpus")
    gpu = gpus[0]["name"] if isinstance(gpus, list) and gpus else "no GPU"
    allowed = system.get("cpus_allowed")
    cores = allowed["count"] if isinstance(allowed, dict) else system.get("cpu_count", "?")
    return f"{system.get('cpu_model', '?')} ({cores} cores), {gpu}", system.get("hostname", "?")


def load_report(report, include_disturbed=False, include_compile=()):
    """-> ((hardware, hostname), points, excluded): one point per usable scaling config."""
    report = Path(report)
    system = json.loads((report / "system.json").read_text())
    with open(report / "summary.csv", newline="") as f:
        summary = list(csv.DictReader(f))
    with open(report / "generations.csv", newline="") as f:
        gens = list(csv.DictReader(f))
    points, excluded = [], []
    for row in summary:
        name = row["config"]
        m = ROW.match(name)
        if not m:
            continue                                    # not one of the scaling sweeps
        dim, be = m.groups()
        if row["status"] != "ok":
            excluded.append((name, f"status {row['status']}"))
            continue
        if "backend" not in row:
            raise ScalingError(f"{report.name}: summary.csv has no Step 22 columns -- the "
                               f"report was written by older code")
        if not row["backend"].startswith(BACKENDS[be][1]):
            raise ScalingError(f"{name} in {report.name}: the run's backend is "
                               f"{row['backend']!r}, the sweep file says {BACKENDS[be][1]!r}")
        if be != "scalar":
            log = report / row["log"]
            if not (log.is_file() and COMPILE_FIX_MARKER.search(log.read_text(errors="replace"))):
                excluded.append((name, "no Step 18 compile fix in the log"))
                continue
        disturbed = row.get("disturbance", "")
        if disturbed and not include_disturbed:
            excluded.append((name, f"machine not free: {disturbed}"))
            continue
        g = [r for r in gens if r["order"] == row["order"] and int(r["generation"]) >= 1
             and r.get("evaluate_s")]
        if not g:
            raise ScalingError(f"{name} in {report.name} ran ok but has no [EA timing] line "
                               f"for generation 1+ (generations.csv evaluate_s)")
        cfg_path = report / "configs" / Path(row["config_path"]).name
        if not cfg_path.is_file():
            raise ScalingError(f"{name} in {report.name}: expanded config {cfg_path} missing")
        cfg = yaml.safe_load(cfg_path.read_text())
        # Step 25: the simulation without the compile warm-up. Scalar has no compile;
        # a tensor point from an older report has no warm-up data: only with
        # --include-compile <backend>, then its evaluation time (compile included).
        if be == "scalar":
            sim, compile_included = [float(r["evaluate_s"]) for r in g], False
        elif all(r.get("simulate_s") for r in g):
            sim, compile_included = [float(r["simulate_s"]) for r in g], False
        elif be in include_compile:
            sim, compile_included = [float(r["evaluate_s"]) for r in g], True
        else:
            raise ScalingError(
                f"{name} in {report.name}: no compile warm-up data (a report from before plan "
                f"Step 25), so its time includes compiling. Re-run it, or pass "
                f"--include-compile {be} to plot it with the compile included")
        ticks = [int(r["ea_run_ticks"]) for r in g]
        points.append({
            "dim": dim, "backend": be, "config": name, "report": report.name,
            "x": int(_get(cfg, DIMS[dim][0])), "cfg": cfg, "disturbed": bool(disturbed),
            "evaluate": sim,
            "us": [1e6 * t / k for t, k in zip(sim, ticks)],
            "with_compile": [float(r["evaluate_s"]) for r in g],
            "warmup_s": sum(float(r["warmup_s"]) for r in g if r.get("warmup_s")),
            "new_graphs": sum(int(r["new_graphs"]) for r in g if r.get("new_graphs")),
            "compile_included": compile_included,
            "ea": [float(r["genomes_s"]) + float(r["select_save_s"]) for r in g],
            "cpu_workers": row.get("cpu_workers", ""),
            "vram_peak_mib": row.get("vram_peak_mib", "")})
    return machine_of(system), points, excluded


def check_consistent(points):
    """Every config equal apart from FREE_KEYS and the three dimension keys, and within
    a dimension the two keys not varied equal across all its points (every backend)."""
    def stripped(cfg, keep=()):
        c = copy.deepcopy(cfg)
        for k in FREE_KEYS + [v[0] for d, v in DIMS.items() if d not in keep]:
            _drop(c, k)
        return c
    ref = None
    for p in points:
        s = stripped(p["cfg"])
        if ref is None:
            ref = (s, p)
        elif s != ref[0]:
            diff = _first_diff(ref[0], s)
            raise ScalingError(f"{p['config']} ({p['report']}) differs from {ref[1]['config']} "
                               f"({ref[1]['report']}) at {diff}: only the varied key and the "
                               f"backend may differ")
    for dim in DIMS:
        fixed = [d for d in DIMS if d != dim]
        seen = {}
        for p in points:
            if p["dim"] != dim:
                continue
            vals = tuple(_get(p["cfg"], DIMS[d][0]) for d in fixed)
            seen.setdefault(vals, []).append(p["config"])
        if len(seen) > 1:
            raise ScalingError(f"the {dim} points do not share {', '.join(DIMS[d][0] for d in fixed)}: "
                               + "; ".join(f"{v} in {c[0]}" for v, c in seen.items()))


def _first_diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                return f"{path}{k} (only in one)"
            if a[k] != b[k]:
                return _first_diff(a[k], b[k], f"{path}{k}.")
    return f"{path.rstrip('.')} ({a!r} vs {b!r})"


def expected_points(sweep_dir=SWEEP_DIR):
    """{(dim, backend): [x values]} from the nine sweep files."""
    out = {}
    for dim, (key, *_rest) in DIMS.items():
        for be in BACKENDS:
            f = Path(sweep_dir) / f"scaling_{dim}_{be}.yaml"
            if f.is_file():
                out[(dim, be)] = list(yaml.safe_load(f.read_text())["vary"][key])
    return out


def series_of(points):
    """{(dim, backend): [one entry per x, repeats pooled]} sorted by x."""
    pooled = {}
    for p in points:
        e = pooled.setdefault((p["dim"], p["backend"], p["x"]), {
            "dim": p["dim"], "backend": p["backend"], "x": p["x"], "reports": [],
            "evaluate": [], "us": [], "ea": [], "disturbed": False, "compile_included": False,
            "with_compile": [], "warmup_s": 0.0, "new_graphs": 0,
            "cpu_workers": p["cpu_workers"], "vram_peak_mib": p["vram_peak_mib"]})
        for k in METRICS:
            e[k] += p[k]
        e["reports"].append(p["report"])
        e["disturbed"] |= p["disturbed"]
        e["compile_included"] |= p["compile_included"]
        e["with_compile"] += p["with_compile"]
        e["warmup_s"] += p["warmup_s"]
        e["new_graphs"] += p["new_graphs"]
    out = {}
    for e in pooled.values():
        for k in METRICS:
            e[k + "_q"] = quartiles(e[k])
        out.setdefault((e["dim"], e["backend"]), []).append(e)
    return {k: sorted(v, key=lambda e: e["x"]) for k, v in out.items()}


def fits_of(series):
    """{(dim, backend, metric): (k, a)} for series with at least 2 x values."""
    out = {}
    for (dim, be), v in series.items():
        if len(v) < 2:
            continue
        for k in METRICS:
            out[(dim, be, k)] = power_fit([e["x"] for e in v], [e[k + "_q"][1] for e in v])
    return out


# ============================================================
# Figure
# ============================================================

def _panel(ax, series, fits, dim, metric, legend=True):
    from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter, NullLocator
    xs_all = sorted({e["x"] for (d, _b), v in series.items() if d == dim for e in v})
    ys = []
    for i, be in enumerate(BACKENDS):
        v = series.get((dim, be))
        if not v:
            continue
        xs = [e["x"] for e in v]
        q = [e[metric + "_q"] for e in v]
        dis = any(e["disturbed"] for e in v)
        comp = any(e["compile_included"] for e in v)
        fit = fits.get((dim, be, metric))
        ys += [y for t in q for y in t]
        lab = BACKENDS[be][0] + (f"  (∝ {DIMS[dim][2]}^{round(fit[0], 2) + 0.0:.2f})" if fit else "") + ("  [busy machine]" if dis else "") + ("  [compile included]" if comp else "")
        ax.errorbar(xs, [m for _a, m, _b in q], yerr=[[m - a for a, m, _b in q], [b - m for _a, m, b in q]],
                    color=SERIES[i], marker=MARKERS[i], markersize=5.5, linewidth=1.5, capsize=3,
                    elinewidth=1, linestyle="--" if dis else "-", label=lab)
        if fit:
            gx = np.geomspace(min(xs), max(xs), 50)
            ax.plot(gx, fit[1] * gx ** fit[0], color=SERIES[i], linewidth=0.8, alpha=0.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    if ys and min(ys) > 0 and max(ys) / min(ys) < 4:
        # A nearly flat line: at least a factor 4 of range, so log ticks exist.
        mid = (max(ys) * min(ys)) ** 0.5
        ax.set_ylim(mid / 2, mid * 2)
    plain = FuncFormatter(lambda v, _: f"{v:g}")
    if xs_all:
        ax.xaxis.set_major_locator(FixedLocator(xs_all))
        ax.xaxis.set_major_formatter(plain)
        ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(plain)
    ax.yaxis.set_minor_formatter(NullFormatter())
    _style(ax, METRICS[metric][0], DIMS[dim][1] + " (log)", METRICS[metric][1])
    if legend:
        ax.legend(fontsize=7, frameon=False, labelcolor=INK, loc="best")


NOTE = ("Median over generations 1+, compile warm-ups subtracted (plan Step 25), bars: interquartile range; "
        "thin lines: power-law fits. Base: P = 100, R = 100, n = 11 apart from the varied key. "
        "Data: points.csv, fits.csv.")


def draw(series, fits, machine, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    written = []
    dims = [d for d in DIMS if any(k[0] == d for k in series)]
    for dim in dims:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), facecolor=SURFACE)
        for ax, metric in zip(axes, METRICS):
            _panel(ax, series, fits, dim, metric)
        fig.suptitle(f"Scaling with {DIMS[dim][1]} — {machine}", x=0.01, ha="left",
                     fontsize=10.5, color=INK, fontweight="bold")
        fig.text(0.01, 0.005, NOTE, fontsize=7, color=INK2)
        fig.tight_layout(rect=(0, 0.03, 1, 0.95))
        for ext in ("png", "pdf"):
            fig.savefig(out / f"scaling_{dim}.{ext}", dpi=200, facecolor=SURFACE)
            written.append(out / f"scaling_{dim}.{ext}")
        plt.close(fig)
    if dims:
        fig, axes = plt.subplots(len(dims), 3, figsize=(14, 4.2 * len(dims)), facecolor=SURFACE,
                                 squeeze=False)
        for r, dim in enumerate(dims):
            for c, metric in enumerate(METRICS):
                _panel(axes[r][c], series, fits, dim, metric)
        fig.suptitle(f"Runtime scaling — {machine}", x=0.01, ha="left", fontsize=11,
                     color=INK, fontweight="bold")
        fig.text(0.01, 0.003, NOTE, fontsize=7, color=INK2)
        fig.tight_layout(rect=(0, 0.015, 1, 0.97))
        for ext in ("png", "pdf"):
            fig.savefig(out / f"scaling_all.{ext}", dpi=200, facecolor=SURFACE)
            written.append(out / f"scaling_all.{ext}")
        plt.close(fig)
    return written


# ============================================================
# Entry point
# ============================================================

def write_csvs(series, fits, machine, out):
    with open(out / "points.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["machine", "dimension", "backend", "x", "reports", "generations", "disturbed",
                    "evaluate_s_median", "evaluate_s_q1", "evaluate_s_q3",
                    "us_per_tick_median", "us_q1", "us_q3",
                    "ea_part_s_median", "ea_q1", "ea_q3", "cpu_workers", "vram_peak_mib",
                    "compile_included", "evaluate_with_compile_s_median", "warmup_s_total",
                    "new_graphs_total"])
        for (dim, be), v in series.items():
            for e in v:
                w.writerow([machine, dim, be, e["x"], " ".join(e["reports"]), len(e["evaluate"]),
                            e["disturbed"]]
                           + [f"{e[k + '_q'][i]:.4f}" for k in METRICS for i in (1, 0, 2)]
                           + [e["cpu_workers"], e["vram_peak_mib"], e["compile_included"],
                              f"{float(np.median(e['with_compile'])):.4f}", f"{e['warmup_s']:.3f}",
                              e["new_graphs"]])
    with open(out / "fits.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["machine", "dimension", "backend", "metric", "exponent_k", "prefactor",
                    "x_min", "x_max", "points", "local_exponents (x_mid:k)"])
        for (dim, be, metric), (k, a) in fits.items():
            v = series[(dim, be)]
            loc = local_exponents([e["x"] for e in v], [e[metric + "_q"][1] for e in v])
            w.writerow([machine, dim, be, metric, f"{k:.3f}", f"{a:.6g}", v[0]["x"], v[-1]["x"],
                        len(v), " ".join(f"{x:.0f}:{e:.2f}" for x, e in loc)])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("reports", nargs="+", help="report folders (data/runtime_reports/...)")
    ap.add_argument("--out", required=True, help="output folder for the figures and CSVs")
    ap.add_argument("--include-disturbed", action="store_true",
                    help="plot runs on a busy machine too (dashed, marked)")
    ap.add_argument("--include-compile", action="append", default=[], choices=["cpu", "cuda"],
                    help="plot this tensor backend's points from reports before plan Step 25 "
                         "(no warm-up data) with the compile included (marked); repeatable")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    machines, hosts, points = {}, set(), []
    for r in a.reports:
        (machine, host), pts, excl = load_report(r, a.include_disturbed, a.include_compile)
        machines.setdefault(machine, []).append(Path(r).name)
        hosts.add(host)
        print(f"{Path(r).name}: {len(pts)} scaling configs used")
        for name, why in excl:
            print(f"  EXCLUDED {name}: {why}")
        points += pts
    if len(machines) > 1:
        raise ScalingError("reports from more than one kind of machine; the comparison is only "
                           "meaningful on one. Plot each machine on its own:\n  "
                           + "\n  ".join(f"{m}: {', '.join(rs)}" for m, rs in machines.items()))
    if not points:
        raise SystemExit("nothing to plot: no usable scaling config in these reports")
    check_consistent(points)
    machine = f"{next(iter(machines))} — {', '.join(sorted(hosts))}"
    series = series_of(points)
    for key, xs in expected_points().items():
        have = {e["x"] for e in series.get(key, [])}
        missing = [x for x in xs if x not in have]
        if missing:
            print(f"  MISSING {key[0]} / {BACKENDS[key[1]][0]}: x = {missing}")
    fits = fits_of(series)
    write_csvs(series, fits, machine, out)
    for (dim, be, metric), (k, _a) in fits.items():
        if metric != "ea":
            v = series[(dim, be)]
            print(f"{dim:10s} {BACKENDS[be][0]:12s} {metric:8s} ∝ x^{k:.2f} "
                  f"(x = {v[0]['x']}-{v[-1]['x']}, {len(v)} points)")
    written = draw(series, fits, machine, out)
    print("written: " + ", ".join(str(p) for p in written) + f", {out / 'points.csv'}, "
          f"{out / 'fits.csv'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ScalingError as e:
        sys.exit(f"ERROR: {e}")
