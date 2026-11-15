#!/usr/bin/env python3
"""Sensitivity curve for a finished behavioural-equivalence run (plan Step 12d).

How large a change to the model does the equivalence test detect? For an existing run
folder (tests/behaviour_equivalence.py), this runs extra tensor production arms in which
the model is deliberately changed (noise_level x 1.5 / 2 / 3 / 5, threshold 0.51 / 0.52 /
0.55), judges each against the run's own scalar arms S1, S2 and S3 with the unchanged
criteria, and reports which criteria detect which change. An identity arm (no change)
must equal the run's T bit for bit: it proves the new arms use the same population,
seeds and code path as the run.

A post-hoc analysis: it does not change the run's verdict and does not touch the run's
files. Everything goes to <run>/sensitivity/. It refuses to run if mvb/, mvb_torch/ or
simulate/ differ from the run's commit, because the new arms are compared with arms the
run computed.

Also reported: the decision statistics paired run by run (same genome, same world), from
the run's per-tick arms S1d, S2d and Td.

Usage
-----
    python -m tests.sensitivity_curve data/behaviour_equivalence/<run folder>
    python -m tests.sensitivity_curve --analyse data/behaviour_equivalence/<run folder>
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tests.behaviour_equivalence as BE  # noqa: E402
from tests.be_common import EquivalenceError, read_arm, read_per_tick  # noqa: E402

ROOT = BE.ROOT
SIM_CODE = ("mvb", "mvb_torch", "simulate")

# name -> (config key, change, label, kind, size). Added to BE.CONTROL_PATCHES in the
# processes of this script only.
PERTURBATIONS = {
    "N1.5": ("brain.noise_level", lambda v: v * 1.5, "noise_level x1.5", "noise", 1.5),
    "N2": ("brain.noise_level", lambda v: v * 2.0, "noise_level x2", "noise", 2.0),
    "N3": ("brain.noise_level", lambda v: v * 3.0, "noise_level x3", "noise", 3.0),
    "N5": ("brain.noise_level", lambda v: v * 5.0, "noise_level x5", "noise", 5.0),
    "TH0.51": ("brain.threshold", lambda v: 0.51, "threshold 0.51", "threshold", 0.51),
    "TH0.52": ("brain.threshold", lambda v: 0.52, "threshold 0.52", "threshold", 0.52),
    "TH0.55": ("brain.threshold", lambda v: 0.55, "threshold 0.55", "threshold", 0.55),
}
IDENTITY = "ID"


def register():
    for name, (key, fn, label, _k, _s) in PERTURBATIONS.items():
        BE.CONTROL_PATCHES[name] = (key, fn, label)


def git(*args):
    p = subprocess.run(["git", "-C", str(ROOT), *args], text=True, capture_output=True)
    if p.returncode != 0:
        raise EquivalenceError(f"git {' '.join(args)} failed: {p.stderr.strip()}")
    return p.stdout.strip()


def check_sim_code(commit):
    """The simulation code here must be the run's: no committed and no uncommitted change
    in mvb/, mvb_torch/, simulate/ since `commit`."""
    changed = git("diff", "--name-only", commit, "--", *SIM_CODE)
    if changed:
        raise EquivalenceError(f"the simulation code differs from the run's commit {commit[:10]}:\n"
                               f"{changed}\nthe new arms would not be comparable with the run's arms")


def plan(run_dir, sens_dir):
    run_dir, sens_dir = Path(run_dir), Path(sens_dir)
    run = json.loads((run_dir / "run.json").read_text())
    for n in ("T", "S1", "S2", "S3", "S1d", "S2d", "Td"):
        if not (run_dir / "arms" / f"{n}.npz").exists():
            raise EquivalenceError(f"{run_dir} has no finished arm {n}")
    check_sim_code(run["code_commit"])
    if run.get("device_override"):
        print(f"[sensitivity] NOTE: the run used device override {run['device_override']!r} "
              f"(a test run, not the production device)", flush=True)
    base_cfg = BE.load_cfg(run["config"])
    t = run["arms"]["T"]
    arms = {IDENTITY: dict(t, role="identity: must equal the run's T bit for bit")}
    for name, (_k, _f, label, _kind, _s) in PERTURBATIONS.items():
        arms[name] = dict(t, patch=name, role=label)
    for a in arms.values():
        a["config_hash"] = BE.json_hash(BE.arm_cfg(base_cfg, run["spec"], a))
    if arms[IDENTITY]["config_hash"] != t["config_hash"]:
        raise EquivalenceError("the identity arm's config differs from the run's T")
    sens = dict(run, arms=arms, parent_run=str(run_dir.resolve()), sensitivity=True)
    head = git("rev-parse", "HEAD")
    sens["code_commit"] = head
    sens_dir.mkdir(parents=True)
    (sens_dir / "arms").mkdir()
    shutil.copy(run_dir / "seeds.npz", sens_dir / "seeds.npz")
    (sens_dir / "run.json").write_text(json.dumps(sens, indent=1))
    return sens


def execute(sens_dir):
    sens = json.loads((Path(sens_dir) / "run.json").read_text())
    for name, arm in sens["arms"].items():
        if (Path(sens_dir) / "arms" / f"{name}.npz").exists():
            print(f"[arm {name}] reused", flush=True)
            continue
        print(f"\n[arm {name}] {arm['role']}", flush=True)
        p = subprocess.run([sys.executable, "-m", "tests.sensitivity_curve", "--run-arm", name,
                            "--dir", str(sens_dir)], cwd=str(ROOT))
        if p.returncode != 0:
            raise EquivalenceError(f"arm {name} failed (exit {p.returncode}); run the same "
                                   f"command again to continue")


def paired_decisions(run_dir, run):
    """Per-run differences (same genome, same world) of the decision features, averaged
    per genome, with a 95% CI over genomes: Td - S1d and the floor S2d - S1d."""
    P, k = len(run["labels"]), run["spec"]["decision_runs"]
    feats = {}
    for n in ("S1d", "S2d", "Td"):
        a = read_arm(Path(run_dir) / "arms" / f"{n}.npz")
        rows = BE.decision_features(read_per_tick(BE.arm_h5(run_dir, a), P, k, weights=False))
        feats[n] = {(r["v"], r["r"]): r for r in rows}
    out = {}
    for feat in BE.FEATURES:
        s1 = {key: r[feat] for key, r in feats["S1d"].items()}
        out[feat] = {}
        for lab, n in (("T_minus_S1", "Td"), ("S2_minus_S1", "S2d")):
            per_g = []
            for v in range(P):
                d = [feats[n][(v, r)][feat] - s1[(v, r)] for r in range(k)
                     if not (np.isnan(feats[n][(v, r)][feat]) or np.isnan(s1[(v, r)]))]
                if d:
                    per_g.append(np.mean(d))
            per_g = np.array(per_g)
            m = float(per_g.mean()) if len(per_g) else float("nan")
            se = float(per_g.std(ddof=1) / np.sqrt(len(per_g))) if len(per_g) > 1 else float("nan")
            out[feat][lab] = dict(mean=m, ci95=[m - 1.96 * se, m + 1.96 * se], genomes=len(per_g),
                                  ref_mean=float(np.nanmean(list(s1.values()))))
    return out


def analyse(sens_dir):
    sens_dir = Path(sens_dir)
    sens = json.loads((sens_dir / "run.json").read_text())
    run_dir = Path(sens["parent_run"])
    if not run_dir.exists():                      # copied elsewhere: the parent of sensitivity/
        run_dir = sens_dir.parent
    mt, k = sens["max_ticks"], sens["spec"]["topk"]
    ref = {n: read_arm(run_dir / "arms" / f"{n}.npz") for n in ("T", "S1", "S2", "S3")}
    arms = {n: read_arm(sens_dir / "arms" / f"{n}.npz") for n in sens["arms"]}
    S = {"part": sens["part"], "run": str(run_dir.name), "checks": {}, "arms": {},
         "device_override": sens.get("device_override")}
    S["checks"]["identity_equals_T"] = BE.results_equal(ref["T"], arms[IDENTITY])
    fl = [BE.measures(ref["S1"], ref[n], "lifespans", k, mt) for n in ("S2", "S3")]
    mT = BE.measures(ref["S1"], ref["T"], "lifespans", k, mt)
    S["T"] = dict(measures=mT, criteria=BE.judge(mT, fl, "lifespans"))
    S["floors"] = fl
    for name, (_k, _f, label, kind, size) in PERTURBATIONS.items():
        m = BE.measures(ref["S1"], arms[name], "lifespans", k, mt)
        c = BE.judge(m, fl, "lifespans")
        S["arms"][name] = dict(label=label, kind=kind, size=size, measures=m, criteria=c,
                               bias_pct=100 * m["bias"] / m["ref_mean"],
                               detected_by=[x for x, v in c.items() if v is False])
    S["paired_decisions"] = paired_decisions(run_dir, sens)
    S["verdict_note"] = ("post-hoc sensitivity analysis (plan Step 12d); the run's own verdict "
                         "is unchanged")
    (sens_dir / "summary.json").write_text(json.dumps(S, indent=1, sort_keys=True, default=BE._jsonable))
    (sens_dir / "report.txt").write_text(report(S))
    draw(sens_dir, S)
    print((sens_dir / "report.txt").read_text())
    return 0 if all(S["checks"].values()) else 1


def report(S):
    L = [f"Sensitivity curve, part {S['part']} (plan Step 12d; post-hoc, the run's verdict is unchanged)",
         f"  run {S['run']}" + (f"  [TEST RUN: device override {S['device_override']}]"
                                if S.get("device_override") else ""), "",
         f"  [{'OK ' if S['checks']['identity_equals_T'] else 'BAD'}] identity arm equals the run's T bit for bit",
         "", "Lifespan, judged against S1 with the floors S2/S3 and the unchanged criteria",
         f"  {'arm':22s} {'bias':>8s} {'bias %':>7s} {'rho':>7s} {'KS share':>9s} {'D':>7s}  detected by"]

    def row(label, m, det):
        return (f"  {label:22s} {m['bias']:+8.3f} {100 * m['bias'] / m['ref_mean']:+6.2f}% {m['rho']:7.4f} "
                f"{100 * m['ks_share']:8.2f}% {m['pooled_D']:7.4f}  {', '.join(det) if det else '-'}")
    L.append(row("T (production, no change)", S["T"]["measures"],
                 [x for x, v in S["T"]["criteria"].items() if v is False]))
    for f, n in zip(S["floors"], ("S2", "S3")):
        L.append(row(f"{n} (floor)", f, []))
    for a in S["arms"].values():
        L.append(row(a["label"], a["measures"], a["detected_by"]))
    L += ["", "Decision statistics, paired run by run (same genome, same world); mean difference, 95% CI over genomes"]
    for feat, d in S["paired_decisions"].items():
        t, f = d["T_minus_S1"], d["S2_minus_S1"]
        L.append(f"  {feat:14s} (S1 mean {t['ref_mean']:.4f})  T-S1 {t['mean']:+.4f} ({t['ci95'][0]:+.4f}..{t['ci95'][1]:+.4f})"
                 f"   floor S2-S1 {f['mean']:+.4f} ({f['ci95'][0]:+.4f}..{f['ci95'][1]:+.4f})")
    return "\n".join(L) + "\n"


def draw(sens_dir, S):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from analysis.plot_neuron_scaling import INK, INK2, SERIES, SURFACE, _style
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURFACE)
    floor_bias = max(abs(100 * f["bias"] / f["ref_mean"]) for f in S["floors"])
    floor_ks = max(100 * f["ks_share"] for f in S["floors"])
    for ax, kind, xlabel in ((axes[0], "noise", "noise_level factor"), (axes[1], "threshold", "threshold")):
        arms = sorted((a for a in S["arms"].values() if a["kind"] == kind), key=lambda a: a["size"])
        xs = [a["size"] for a in arms]
        ax.plot(xs, [abs(a["bias_pct"]) for a in arms], marker="o", color=SERIES[0], label="|bias| % of mean")
        ax.plot(xs, [100 * a["measures"]["ks_share"] for a in arms], marker="s", color=SERIES[1],
                label="KS share %")
        for a, x in zip(arms, xs):
            if a["detected_by"]:
                ax.annotate("detected", (x, abs(a["bias_pct"])), textcoords="offset points", xytext=(0, 6),
                            fontsize=7, color=INK, ha="center")
        ax.axhline(100 * BE.MARGIN, color=SERIES[0], linestyle="--", linewidth=0.8, label="bias margin")
        ax.axhline(floor_ks + 100 * BE.KS_SLACK, color=SERIES[1], linestyle=":", linewidth=0.8,
                   label="KS limit (floor + 5)")
        ax.axhline(floor_bias, color=INK2, linewidth=0.6)
        ax.legend(fontsize=7, frameon=False, labelcolor=INK)
        _style(ax, f"Part {S['part']}: {kind}", xlabel, "%")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(Path(sens_dir) / f"sensitivity.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", nargs="?", help="a finished behavioural-equivalence run folder")
    ap.add_argument("--analyse", metavar="RUN", help="only redo the report from <RUN>/sensitivity")
    ap.add_argument("--run-arm", help=argparse.SUPPRESS)
    ap.add_argument("--dir", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    register()
    if a.run_arm:
        return BE.run_arm(str(Path(a.dir).resolve()), a.run_arm)
    if a.analyse:
        return analyse(Path(a.analyse).resolve() / "sensitivity")
    if not a.run:
        ap.error("give a run folder")
    run_dir = Path(a.run).resolve()
    sens_dir = run_dir / "sensitivity"
    if not sens_dir.exists():
        plan(run_dir, sens_dir)
    else:
        check_sim_code(json.loads((run_dir / "run.json").read_text())["code_commit"])
        print(f"[sensitivity] continuing in {sens_dir}", flush=True)
    execute(sens_dir)
    return analyse(sens_dir)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except EquivalenceError as e:
        sys.exit(f"ERROR: {e}")
