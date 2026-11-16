#!/usr/bin/env python3
"""CPU baseline on a many-core server against the uni PC's CPU and GPU (plan Step 30).

Reads run_selected report folders from several machines ON PURPOSE (unlike
plot_scaling): each is labelled from its system.json (CPU model, allowed cores, GPU).
Only the cpu_baseline_<n11|n50>_<scalar|cpu|cuda> configs are used.

  (a) n = 11, (b) n = 50: seconds per whole EA generation, median of generations 1+
      (mutation + evaluation + selection/HDF5 from the [EA timing] line; tensor without
      the compile warm-up, plan Step 25), one bar per machine and code
  (c) core scaling: the n = 11 scalar run against the allowed cores of each machine with
      the same CPU model (TinyFat 16 / 32 / 64), the ideal-linear line from the fewest
      cores, and the other machines' scalar points for reference

The reference machine is the one with CUDA results (exactly one is required); speed-ups
are given against its scalar run, and the GPU's lead against the fastest CPU bar.

Usage
-----
    python -m analysis.plot_cpu_baseline data/runtime_reports/<A> <B> ... --out <folder>
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

import analysis.plot_scaling as PS
import simulate.run_selected as RS
from analysis.plot_neuron_scaling import INK, INK2, SERIES, SURFACE, _style

ROW = re.compile(r"^cpu_baseline_(n11|n50)_(scalar|cpu|cuda) \[population_size=(\d+)\]$")
BACKEND_COLUMN = {"scalar": "scalar", "cpu": "tensor cpu", "cuda": "tensor cuda"}
CODE = {"scalar": "scalar", "cpu": "tensor CPU", "cuda": "tensor CUDA"}
SWEEPS = RS.SWEEP_DIR


class BaselineError(ValueError):
    pass


def short_cpu(model):
    """'12th Gen Intel(R) Core(TM) i7-12700' -> 'i7-12700'; 'AMD EPYC 7502 32-Core
    Processor' -> 'EPYC 7502'; otherwise the first 14 characters."""
    m = re.search(r"(i\d-\d+\w*|EPYC \d+\w*|Xeon[\w() ]*? \d{4}\w*|Ryzen[\w ]*? \d{4}\w*)", model)
    return m.group(1) if m else model[:14]


def machine_key(system):
    gpus = system.get("gpus")
    gpu = gpus[0]["name"] if isinstance(gpus, list) and gpus else None
    allowed = system.get("cpus_allowed")
    cores = allowed["count"] if isinstance(allowed, dict) else system.get("cpu_count")
    return dict(cpu=system.get("cpu_model", "?"), cores=int(cores), gpu=gpu,
                label=f"{system.get('cpu_model', '?')} ({cores} cores)" + (f" + {gpu}" if gpu else ""))


def expected_configs():
    out = {}
    for size in ("n11", "n50"):
        for be in ("scalar", "cpu", "cuda"):
            for name, cfg in RS.expand_sweep(SWEEPS / f"cpu_baseline_{size}_{be}.yaml"):
                out[name] = cfg
    return out


def _strip(cfg):
    cfg = copy.deepcopy(cfg)
    PS._drop(cfg, "experiment.output_folder")
    return cfg


def load_report(report, expected, include_disturbed=False):
    report = Path(report)
    system = json.loads((report / "system.json").read_text())
    m = machine_key(system)
    with open(report / "summary.csv", newline="") as f:
        summary = list(csv.DictReader(f))
    with open(report / "generations.csv", newline="") as f:
        gens = list(csv.DictReader(f))
    points, excluded = [], []
    for row in summary:
        mm = ROW.match(row["config"])
        if not mm:
            continue
        size, be = mm.group(1), mm.group(2)
        name = row["config"]
        if row["status"] != "ok":
            excluded.append((name, f"status {row['status']}"))
            continue
        if not row.get("backend", "").startswith(BACKEND_COLUMN[be]):
            raise BaselineError(f"{name} in {report.name}: the run's backend is {row.get('backend')!r}, "
                                f"the sweep file says {BACKEND_COLUMN[be]!r}")
        if row.get("disturbance") and not include_disturbed:
            excluded.append((name, f"machine not free: {row['disturbance']}"))
            continue
        cfg = yaml.safe_load((report / "configs" / Path(row["config_path"]).name).read_text())
        if _strip(cfg) != _strip(expected[name]):
            raise BaselineError(f"{name} in {report.name}: its config differs from the sweep file's: "
                                f"{PS._first_diff(_strip(cfg), _strip(expected[name]))}")
        g = [r for r in gens if r["order"] == row["order"] and int(r["generation"]) >= 1 and r.get("evaluate_s")]
        if not g:
            raise BaselineError(f"{name} in {report.name}: no [EA timing] line for generation 1+")
        if be != "scalar" and not all(r.get("simulate_s") for r in g):
            raise BaselineError(f"{name} in {report.name}: no compile warm-up columns (before Step 25)")
        whole = [float(r["genomes_s"]) + float(r["simulate_s"] if be != "scalar" else r["evaluate_s"])
                 + float(r["select_save_s"]) for r in g]
        points.append(dict(machine=m, size=size, backend=be, config=name, report=report.name,
                           q=[float(x) for x in np.percentile(whole, [25, 50, 75])],
                           wall=float(row["wall_s"]), workers=row.get("cpu_workers", "")))
    return points, excluded


def collect(reports, include_disturbed=False):
    expected = expected_configs()
    points = []
    for r in reports:
        pts, excl = load_report(r, expected, include_disturbed)
        for name, why in excl:
            print(f"  EXCLUDED {name} ({Path(r).name}): {why}")
        points += pts
    seen = {}
    for p in points:
        key = (p["machine"]["label"], p["config"])
        if key in seen:
            raise BaselineError(f"{p['config']} on {p['machine']['label']} appears in two reports "
                                f"({seen[key]}, {p['report']}); give one per machine and core count")
        seen[key] = p["report"]
    refs = sorted({p["machine"]["label"] for p in points if p["backend"] == "cuda"})
    if len(refs) != 1:
        raise BaselineError(f"exactly one machine with CUDA results is needed as the reference, got {refs}")
    ref = refs[0]
    for p in points:
        if p["backend"] == "cuda":
            continue
        if (ref, p["config"]) not in seen:
            raise BaselineError(f"{p['config']} ran on {p['machine']['label']} but not on the reference "
                                f"machine {ref}: no counterpart to compare with")
    return points, ref


def table(points, ref):
    rows = []
    for size in ("n11", "n50"):
        pts = [p for p in points if p["size"] == size]
        if not pts:
            continue
        ref_scalar = next((p for p in pts if p["machine"]["label"] == ref and p["backend"] == "scalar"), None)
        cuda = next((p for p in pts if p["backend"] == "cuda"), None)
        best_cpu = min((p for p in pts if p["backend"] != "cuda"), key=lambda p: p["q"][1], default=None)
        for p in sorted(pts, key=lambda p: (p["backend"] == "cuda", -p["q"][1])):
            rows.append(dict(size=size, machine=p["machine"]["label"], cores=p["machine"]["cores"],
                             code=CODE[p["backend"]], median_s=p["q"][1], q25=p["q"][0], q75=p["q"][2],
                             wall_s=p["wall"],
                             vs_ref_scalar=(ref_scalar["q"][1] / p["q"][1]) if ref_scalar else None,
                             gpu_lead=(p["q"][1] / cuda["q"][1]) if cuda and p["backend"] != "cuda" else None,
                             fastest_cpu=(p is best_cpu), point=p))
    return rows


def draw(points, rows, ref, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    sizes = [s for s in ("n11", "n50") if any(r["size"] == s for r in rows)]
    fig, axes = plt.subplots(1, len(sizes) + 1, figsize=(6.2 * (len(sizes) + 1), 5), facecolor=SURFACE)
    titles = {"n11": "(a) n = 11, lookup config, P 260, R 300", "n50": "(b) n = 50, P 64, R 100"}
    colour = {"scalar": SERIES[0], "cpu": SERIES[2], "cuda": SERIES[1]}
    for ax, size in zip(axes, sizes):
        rs = [r for r in rows if r["size"] == size]
        for i, r in enumerate(rs):
            p = r["point"]
            ax.bar(i, r["median_s"], color=colour[p["backend"]], width=0.65,
                   yerr=[[r["median_s"] - r["q25"]], [r["q75"] - r["median_s"]]], capsize=3)
            txt = f"{r['median_s']:.3g} s"
            if r["vs_ref_scalar"] and not (p["machine"]["label"] == ref and p["backend"] == "scalar"):
                txt += f"\n×{r['vs_ref_scalar']:.2g} vs ref scalar"
            if r["gpu_lead"] and r["fastest_cpu"]:
                txt += f"\nGPU {r['gpu_lead']:.3g}× faster"
            ax.text(i, r["median_s"] * 1.06, txt, ha="center", va="bottom", fontsize=6.5, color=INK)
        ax.set_xticks(range(len(rs)))
        ax.set_xticklabels([f"{r['code']}\n{short_cpu(r['point']['machine']['cpu'])}\n{r['cores']} cores"
                            for r in rs], fontsize=6.5)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        _style(ax, titles[size], "", "seconds per EA generation (log), median, IQR")
    ax = axes[-1]
    sc = [p for p in points if p["size"] == "n11" and p["backend"] == "scalar"]
    by_cpu = {}
    for p in sc:
        by_cpu.setdefault(p["machine"]["cpu"], []).append(p)
    for i, (cpu, ps) in enumerate(sorted(by_cpu.items(), key=lambda kv: -len(kv[1]))):
        ps = sorted(ps, key=lambda p: p["machine"]["cores"])
        xs, ys = [p["machine"]["cores"] for p in ps], [p["q"][1] for p in ps]
        ax.plot(xs, ys, marker="o", color=SERIES[i % len(SERIES)], linewidth=1.6, label=short_cpu(cpu))
        if len(ps) > 1:
            ax.plot(xs, [ys[0] * xs[0] / x for x in xs], linestyle=":", color=SERIES[i % len(SERIES)],
                    linewidth=1, label=f"ideal linear from {xs[0]} cores")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.legend(fontsize=7, frameon=False, labelcolor=INK)
    _style(ax, "(c) Scalar code vs cores (n = 11)", "allowed cores (log2)", "seconds per EA generation (log)")
    fig.suptitle("CPU baseline: many-core server CPU vs the uni PC's CPU and GPU", x=0.01, ha="left",
                 fontsize=11, color=INK, fontweight="bold")
    fig.text(0.01, 0.005, f"Reference (speed-ups): {ref}. Whole EA generation from [EA timing], generations 1+, "
             f"tensor without compile warm-up. The scalar code runs one genome per worker. Data: cpu_baseline.csv.",
             fontsize=6.5, color=INK2, wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    written = []
    for ext in ("png", "pdf"):
        fig.savefig(out / f"cpu_baseline.{ext}", dpi=200, facecolor=SURFACE)
        written.append(out / f"cpu_baseline.{ext}")
    plt.close(fig)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-disturbed", action="store_true")
    a = ap.parse_args(argv)
    points, ref = collect(a.reports, a.include_disturbed)
    rows = table(points, ref)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "cpu_baseline.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["size", "machine", "cores", "code", "median_s_per_gen", "q25", "q75", "wall_s",
                    "speedup_vs_reference_scalar", "gpu_lead_over_this", "fastest_cpu"])
        for r in rows:
            w.writerow([r["size"], r["machine"], r["cores"], r["code"], f"{r['median_s']:.4f}", f"{r['q25']:.4f}",
                        f"{r['q75']:.4f}", f"{r['wall_s']:.2f}",
                        f"{r['vs_ref_scalar']:.3f}" if r["vs_ref_scalar"] else "",
                        f"{r['gpu_lead']:.3f}" if r["gpu_lead"] else "", r["fastest_cpu"]])
    written = draw(points, rows, ref, out)
    print(f"reference: {ref}")
    for r in rows:
        print(f"  {r['size']} {r['code']:11s} {r['machine'][:48]:48s} {r['median_s']:9.3f} s/gen"
              + (f"  x{r['vs_ref_scalar']:.2f} vs ref scalar" if r["vs_ref_scalar"] else "")
              + (f"  (GPU {r['gpu_lead']:.1f}x faster)" if r["gpu_lead"] else ""))
    print("written: " + ", ".join(str(p) for p in written) + f", {out / 'cpu_baseline.csv'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (BaselineError, PS.ScalingError) as e:
        sys.exit(f"ERROR: {e}")
