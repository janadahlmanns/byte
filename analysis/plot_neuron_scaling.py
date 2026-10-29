#!/usr/bin/env python3
"""How runtime grows with the number of neurons -- the figure (plan Step 20a).

Reads report folders written by simulate/run_selected.py (data/runtime_reports/...),
one or more machines, and draws:

  1. us per worm-tick vs n (log-log), median over generations 1+ with the
     interquartile range, one line per machine and sweep, the fitted exponent in the
     legend (t ~ n^k);
  2. seconds per generation vs n -- the number a reader feels;
  3. the local exponent between neighbouring points -- does the growth rate grow?
  4. (with --vram-csv, from tests/width_probe.py --vram-gib) time per worm-tick
     relative to the fastest emulated VRAM size, one line per n (at most 5 shown;
     all are in the probe's CSV).

Which data counts is decided per config and printed, never silently:
  - only code with the Step 18 compile fix (the log's `compile_cache=.../byte-w512-...`
    line); older runs had a bimodal, history-dependent speed;
  - only an idle GPU: no foreign process in gpu_snapshots.json and idle load/memory
    before the run (`--include-disturbed` plots them anyway, marked "disturbed");
  - only configs that finished ok, and only families with at least 2 neuron counts.
Generation 0 is never used: it holds the compile.

Writes <out>/neuron_scaling.png and .pdf, points.csv (every plotted number) and
fits.csv (exponents), so the numbers in the text match the figure.

Usage
-----
    python -m analysis.plot_neuron_scaling data/runtime_reports/<folder> [...] --out analysis/out
    python -m analysis.plot_neuron_scaling <folders> --out analysis/out --vram-csv vram.csv
"""

import argparse
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path

import numpy as np

COMPILE_FIX_MARKER = re.compile(r"compile_cache=\S*byte-w512")
IDLE_UTIL_PCT = 10          # load before a config above this: another user's work
IDLE_MEM_MIB = 300          # memory in use before a config above this (idle card: ~47 MiB)

# Light-mode print figure. Categorical slots in fixed order (machines), validated
# light-mode (dataviz palette check: worst adjacent CVD dE 9.2); an ordinal blue ramp
# for n in panel 4 (5 steps, validated). Text in neutral ink, never series colour.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "D"]
NRAMP = ["#86b6ef", "#5598e7", "#256abf", "#184f95", "#0d366b"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


# ============================================================
# Reading reports
# ============================================================

def _n_of(report, row):
    """The neuron count from the run's own data: the result file's config attribute,
    else the expanded config saved in the report. Never from the config's name."""
    copy = row.get("hdf5_copy", "")
    if copy and not copy.startswith("unavailable") and (report / copy).is_file():
        import h5py
        with h5py.File(report / copy, "r") as f:
            if "brain_n_neurons" in f.attrs:
                return int(f.attrs["brain_n_neurons"])
    cfg = report / "configs" / Path(row.get("config_path", "")).name
    if cfg.is_file():
        import yaml
        return int(yaml.safe_load(cfg.read_text())["brain"]["n_neurons"])
    raise ValueError(f"cannot tell the neuron count of {row['config']!r} in {report}: no "
                     f"result copy with brain_n_neurons and no expanded config")


def _disturbance(row, snap):
    """None if the GPU was idle around this config, else the reason."""
    reasons = []
    for when in ("before", "after"):
        procs = (snap or {}).get(when, {}).get("processes")
        if isinstance(procs, list) and procs:
            reasons.append(f"other GPU process {when}: {procs[0]}")
    try:
        if float(row.get("gpu_util_before_pct", "0").split("|")[0] or 0) > IDLE_UTIL_PCT:
            reasons.append(f"GPU load {row['gpu_util_before_pct']}% before")
        if float(row.get("gpu_mem_used_before_mib", "0").split("|")[0] or 0) > IDLE_MEM_MIB:
            reasons.append(f"{row['gpu_mem_used_before_mib']} MiB in use before")
    except ValueError:
        pass                  # "unavailable: ..." (no nvidia-smi): nothing to judge by
    return "; ".join(reasons) or None


def load_report(report, include_disturbed=False):
    """-> (points, excluded): one point per usable config; excluded = (config, reason)."""
    report = Path(report)
    system = json.loads((report / "system.json").read_text())
    gpus = system.get("gpus")
    gpu = gpus[0]["name"] if isinstance(gpus, list) and gpus else str(system.get("cpu_model"))
    machine = f"{gpu.replace('NVIDIA GeForce ', '')} ({system.get('hostname', '?')})"
    with open(report / "summary.csv", newline="") as f:
        summary = list(csv.DictReader(f))
    with open(report / "generations.csv", newline="") as f:
        gens = list(csv.DictReader(f))
    snaps = {}
    if (report / "gpu_snapshots.json").is_file():
        snaps = {str(x["order"]): x for x in json.loads((report / "gpu_snapshots.json").read_text())}
    points, excluded = [], []
    for row in summary:
        name = row["config"]
        if row["status"] != "ok":
            excluded.append((name, f"status {row['status']}"))
            continue
        log = report / row["log"]
        if not (log.is_file() and COMPILE_FIX_MARKER.search(log.read_text(errors="replace"))):
            excluded.append((name, "no Step 18 compile fix in the log (older code: bimodal speed)"))
            continue
        disturbed = _disturbance(row, snaps.get(row["order"]))
        if disturbed and not include_disturbed:
            excluded.append((name, f"GPU not idle: {disturbed}"))
            continue
        g = [r for r in gens if r["order"] == row["order"] and int(r["generation"]) >= 1
             and r.get("us_per_run_tick")]
        if not g:
            excluded.append((name, "no generation 1+ with run-ticks"))
            continue
        us = [float(r["us_per_run_tick"]) for r in g]
        sec = [float(r["seconds"]) for r in g]
        points.append({
            "machine": machine, "family": re.sub(r"\s*\[.*\]$", "", name), "config": name,
            "n": _n_of(report, row), "report": report.name, "disturbed": bool(disturbed),
            "us": us, "sec": sec, "vram_peak_mib": row.get("vram_peak_mib", ""),
            "runs_queued": row.get("runs_queued", "")})
    return points, excluded


# ============================================================
# Numbers
# ============================================================

def quartiles(xs):
    q = statistics.quantiles(xs, n=4, method="inclusive") if len(xs) > 1 else [xs[0]] * 3
    return q[0], statistics.median(xs), q[2]


def power_fit(ns, ts):
    """Least squares on log t = log a + k log n -> (k, a)."""
    k, loga = np.polyfit(np.log(ns), np.log(ts), 1)
    return float(k), float(math.exp(loga))


def local_exponents(ns, ts):
    return [(math.sqrt(ns[i - 1] * ns[i]), math.log(ts[i] / ts[i - 1]) / math.log(ns[i] / ns[i - 1]))
            for i in range(1, len(ns))]


def series_of(points):
    """{(machine, family): points sorted by n}, families with >= 2 neuron counts only."""
    out = {}
    for p in points:
        q1, med, q3 = quartiles(p["us"])
        s1, smed, s3 = quartiles(p["sec"])
        p.update(us_q1=q1, us_med=med, us_q3=q3, sec_q1=s1, sec_med=smed, sec_q3=s3)
        out.setdefault((p["machine"], p["family"]), []).append(p)
    return {k: sorted(v, key=lambda p: p["n"]) for k, v in out.items()
            if len({p["n"] for p in v}) >= 2}


# ============================================================
# Figure
# ============================================================

def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold")
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.6, which="major")
    ax.tick_params(colors=INK2, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def draw(series, fits, out, vram_rows=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter, NullLocator

    rows = 2 if vram_rows else 1
    fig, axes = plt.subplots(rows, 3 if not vram_rows else 2, figsize=(13, 4.4 * rows),
                             facecolor=SURFACE, squeeze=False)
    ax1, ax2 = axes[0][0], axes[0][1]
    ax3 = axes[0][2] if not vram_rows else axes[1][0]
    ax4 = axes[1][1] if vram_rows else None
    keys = list(series)
    if len(keys) > len(SERIES):
        raise SystemExit(f"{len(keys)} series; the figure shows at most {len(SERIES)} distinct "
                         f"colours. Plot fewer reports at once (the CSVs take any number).")
    all_n = sorted({p["n"] for v in series.values() for p in v})
    for i, key in enumerate(keys):
        pts, c, m = series[key], SERIES[i], MARKERS[i]
        ns = [p["n"] for p in pts]
        k, a = fits[key]
        lab = f"{key[0]}, {key[1]}  (t ∝ n^{k:.2f})"
        dis = any(p["disturbed"] for p in pts)
        ax1.errorbar(ns, [p["us_med"] for p in pts],
                     yerr=[[p["us_med"] - p["us_q1"] for p in pts], [p["us_q3"] - p["us_med"] for p in pts]],
                     color=c, marker=m, markersize=6, linewidth=1.5, capsize=3, elinewidth=1,
                     label=lab + ("  [disturbed]" if dis else ""),
                     linestyle="--" if dis else "-")
        xs = np.linspace(min(ns), max(ns), 50)
        ax1.plot(xs, a * xs ** k, color=c, linewidth=0.8, alpha=0.45)
        ax2.errorbar(ns, [p["sec_med"] for p in pts],
                     yerr=[[p["sec_med"] - p["sec_q1"] for p in pts], [p["sec_q3"] - p["sec_med"] for p in pts]],
                     color=c, marker=m, markersize=6, linewidth=1.5, capsize=3, elinewidth=1,
                     linestyle="--" if dis else "-")
        loc = local_exponents(ns, [p["us_med"] for p in pts])
        ax3.plot([x for x, _ in loc], [e for _, e in loc], color=c, marker=m, markersize=6,
                 linewidth=1.5, linestyle="--" if dis else "-")
        ax3.axhline(k, color=c, linewidth=0.8, alpha=0.45)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    plain = FuncFormatter(lambda v, _: f"{v:g}")
    ticks = all_n if len(all_n) <= 8 else all_n[::2] + ([all_n[-1]] if len(all_n) % 2 == 0 else [])
    ax1.xaxis.set_major_locator(FixedLocator(ticks))
    ax1.xaxis.set_major_formatter(plain)
    ax1.xaxis.set_minor_locator(NullLocator())
    ax1.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax1.yaxis.set_major_formatter(plain)
    ax1.yaxis.set_minor_formatter(NullFormatter())
    _style(ax1, "Time per worm-tick", "neurons n (log)", "µs per worm-tick (log), median, IQR")
    _style(ax2, "Time per generation", "neurons n", "seconds per generation, median, IQR")
    _style(ax3, "Growth rate between neighbouring n", "neurons n",
           "local exponent  d ln t / d ln n  (line: overall fit)")
    ax1.legend(fontsize=7.5, frameon=False, labelcolor=INK, loc="upper left")
    if ax4 is not None:
        draw_vram(ax4, vram_rows)
    fig.text(0.01, 0.005, "Generations 1+ (generation 0 holds the compile). Data: points.csv, fits.csv.",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"neuron_scaling.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def pick_n(ns, k=5):
    """At most k neuron counts, evenly spaced, always the smallest and the largest."""
    ns = sorted(ns)
    if len(ns) <= k:
        return ns
    idx = sorted({round(i * (len(ns) - 1) / (k - 1)) for i in range(k)})
    return [ns[i] for i in idx]


def draw_vram(ax, rows):
    """Runtime relative to the fastest emulated size at each n (1.00 = fastest), on a
    fixed minimum range, so a 0.3% difference does not fill the panel."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ok = [r for r in rows if r["status"] == "ok"]
    shown = pick_n({int(r["n"]) for r in ok})
    ramp = NRAMP[-len(shown):] if len(shown) < len(NRAMP) else NRAMP
    top = 1.25
    for n, c in zip(shown, ramp):
        pts = sorted((float(r["vram_gib"]), float(r["us_per_run_tick"]), r["all_runs_fit"] == "True")
                     for r in ok if int(r["n"]) == n)
        best = min(p[1] for p in pts)
        rel = [p[1] / best for p in pts]
        top = max(top, max(rel) * 1.03)
        ax.plot([p[0] for p in pts], rel, color=c, marker="o", markersize=5, linewidth=1.5,
                label=f"n = {n}  (fastest {best:.1f} µs)")
        fit = [(p[0], r) for p, r in zip(pts, rel) if p[2]]
        if fit:
            ax.plot([fit[0][0]], [fit[0][1]], marker="o", markersize=10, markerfacecolor="none",
                    markeredgecolor=c, linewidth=0)
    ax.axhline(1.05, color=INK2, linewidth=0.8, linestyle=":")
    ax.text(ax.get_xlim()[0], 1.05, " knee threshold: +5%", fontsize=7, color=INK2, va="bottom")
    gpu = rows[0].get("gpu", "") if rows else ""
    ax.set_xscale("log", base=2)
    ax.set_ylim(0.95, top)
    ax.xaxis.set_major_locator(FixedLocator(sorted({float(r["vram_gib"]) for r in ok})))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    _style(ax, f"VRAM vs runtime (emulated on {gpu})", "emulated card memory, GiB (log)",
           "time per worm-tick relative to the fastest size")
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK,
              title="open circle: all runs fit", title_fontsize=7)


# ============================================================
# Entry point
# ============================================================

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("reports", nargs="+", help="report folders (data/runtime_reports/...)")
    ap.add_argument("--out", required=True, help="output folder for the figure and CSVs")
    ap.add_argument("--include-disturbed", action="store_true",
                    help="plot configs run on a busy GPU too (dashed, marked)")
    ap.add_argument("--vram-csv", default=None, help="tests/width_probe.py --vram-gib --csv output")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    points = []
    for r in a.reports:
        pts, excl = load_report(r, a.include_disturbed)
        print(f"{Path(r).name}: {len(pts)} configs used")
        for name, why in excl:
            print(f"  EXCLUDED {name}: {why}")
        points += pts
    series = series_of(points)
    for p in points:
        if (p["machine"], p["family"]) not in series:
            print(f"  NOT PLOTTED {p['config']} ({p['report']}): its family has fewer than 2 "
                  f"neuron counts")
    if not series:
        raise SystemExit("nothing to plot: no family with at least 2 usable neuron counts")
    fits = {k: power_fit([p["n"] for p in v], [p["us_med"] for p in v]) for k, v in series.items()}

    with open(out / "points.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["machine", "family", "n", "report", "disturbed", "generations",
                    "us_per_tick_median", "us_q1", "us_q3", "sec_per_gen_median", "sec_q1",
                    "sec_q3", "vram_peak_mib", "runs_queued"])
        for (mach, fam), v in series.items():
            for p in v:
                w.writerow([mach, fam, p["n"], p["report"], p["disturbed"], len(p["us"]),
                            f"{p['us_med']:.3f}", f"{p['us_q1']:.3f}", f"{p['us_q3']:.3f}",
                            f"{p['sec_med']:.2f}", f"{p['sec_q1']:.2f}", f"{p['sec_q3']:.2f}",
                            p["vram_peak_mib"], p["runs_queued"]])
    with open(out / "fits.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["machine", "family", "exponent_k", "prefactor_us", "n_min", "n_max", "points",
                    "local_exponents (n_mid:k)"])
        for key, v in series.items():
            k, a_ = fits[key]
            loc = local_exponents([p["n"] for p in v], [p["us_med"] for p in v])
            w.writerow([key[0], key[1], f"{k:.3f}", f"{a_:.5f}", v[0]["n"], v[-1]["n"], len(v),
                        " ".join(f"{x:.1f}:{e:.2f}" for x, e in loc)])
            print(f"{key[0]}, {key[1]}: t ~ n^{k:.2f} over n = {v[0]['n']}-{v[-1]['n']} "
                  f"({len(v)} points)")
    vram_rows = None
    if a.vram_csv:
        with open(a.vram_csv, newline="") as f:
            vram_rows = [r for r in csv.DictReader(f) if r["mode"] == "vram"]
        if not vram_rows:
            raise SystemExit(f"{a.vram_csv} has no --vram-gib rows")
    draw(series, fits, out, vram_rows)
    print(f"written: {out / 'neuron_scaling.png'}, .pdf, points.csv, fits.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
