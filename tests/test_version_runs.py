#!/usr/bin/env python3
"""Tests for tests/version_runs.py and analysis/plot_versions.py (plan Step 29).

  [1] every job of both job files builds: model sections equal the canonical config,
      the job's values applied, only allowed differences; the per-commit evaluator as
      expected (none before ef34, no compile at ef34, compile added at b88b, per_child
      at 93de and HEAD)
  [2] refusals while building: a set key the commit does not know, an added key outside
      the evaluator, a CUDA job on a scalar template, an old experiment key that
      changes the config
  [3] generation markers: intervals measured exactly; a missing marker is an error
  [4] end to end on this machine (scalar, tiny): the original code and HEAD run from a
      temporary worktree, 2 generations timed each, resume skips finished jobs, the
      plot is written
  [5] refusals while running: a dirty worktree, this checkout's HEAD changed since
      planning; waterfall speed-ups computed against the previous and first commit

Usage
-----
    python -m tests.test_version_runs
"""

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analysis.plot_versions as PV  # noqa: E402
import tests.version_runs as VR  # noqa: E402

ROOT = VR.ROOT
_RESULTS = []


def check(name, cond):
    _RESULTS.append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def raises(fn):
    try:
        fn()
    except VR.VersionRunError as e:
        return str(e)
    return None


def build(job):
    head = VR.resolve("HEAD")
    sha = head if job["commit"] == "HEAD" else VR.resolve(job["commit"])
    t = VR.show_yaml(sha, f"configs/experiments/{job['template']}.yaml")
    c = VR.show_yaml(head, f"configs/experiments/{job['canonical']}.yaml")
    return VR.build_config(t, c, job), c, t


def test_build():
    print("\n[1] Every job of both job files builds")
    for jf in ("original", "waterfall_a"):
        jobs = yaml.safe_load((ROOT / "configs" / "version_runs" / f"{jf}.yaml").read_text())["jobs"]
        ok = True
        for j in jobs:
            (cfg, _d), canon, _t = build(j)
            ok &= all(cfg[s] == canon[s] for s in VR.MODEL_SECTIONS)
            for k, v in (j.get("set") or {}).items():
                cur = cfg
                for part in k.split("."):
                    cur = cur[part]
                ok &= cur == v
        check(f"{jf}: {len(jobs)} jobs build; model sections == canonical; set values applied", ok)
    jobs = {j["name"]: j for j in yaml.safe_load(
        (ROOT / "configs" / "version_runs" / "waterfall_a.yaml").read_text())["jobs"]}
    ev = {n: build(j)[0][0]["experiment"].get("evaluator") for n, j in jobs.items()}
    rng = {n: build(j)[0][0]["experiment"]["evolutionary_algorithm"].get("mutation_rng") for n, j in jobs.items()}
    check("scalar commits (8e17, 740a, 3dfc): no evaluator block (their code has none)",
          all(ev[n] is None for n in ("wfA_0_8e17e69", "wfA_1_740a341", "wfA_2_3dfcd31")))
    check("ef34: tensor CUDA without compile (its code does not know the key)",
          ev["wfA_3_ef34271"]["device"] == "cuda" and "compile" not in ev["wfA_3_ef34271"])
    check("b88b: compile added (true)", ev["wfA_4_b88ba9d"].get("compile") is True)
    check("93de and HEAD: per_child; 35f2: no mutation_rng (predates it)",
          rng["wfA_6_93de867"] == rng["wfA_7_HEAD"] == "per_child" and rng["wfA_5_35f2840"] is None)
    check("waterfall A: production values (P 260, R 300, elite 10, n 11, 5 generations)",
          all((lambda e: e["population_size"] == 260 and e["n_runs"] == 300
               and e["evolutionary_algorithm"]["elite_size"] == 10
               and e["evolutionary_algorithm"]["num_generations"] == 5)(build(j)[0][0]["experiment"])
              and build(j)[0][0]["brain"]["n_neurons"] == 11 for j in jobs.values()))


def test_build_refusals():
    print("\n[2] Refusals while building")
    base = dict(name="x", commit="8e17e694c7", template="ea_from_lookup", canonical="ea_from_lookup_tensor_cuda",
                backend="scalar", group="g", label="l", P=2)
    e = raises(lambda: build(dict(base, set={"experiment.evolutionary_algorithm.mutation_rng": "per_child"})))
    check(f"a key the commit does not know (mutation_rng at 8e17) -> refused ({(e or '')[:60]})",
          e and "does not exist in this commit's template" in e)
    e = raises(lambda: build(dict(base, commit="b88ba9ddd9", template="ea_from_lookup_tensor_cuda",
                                  backend="cuda", add={"brain.noise_level": 0.1})))
    check("an added key outside the evaluator -> refused", e and "only experiment.evaluator" in e)
    e = raises(lambda: build(dict(base, backend="cuda")))
    check("a CUDA job on a scalar template -> refused", e and "no tensor evaluator" in e)
    head = VR.resolve("HEAD")
    t = VR.show_yaml("8e17e694c7", "configs/experiments/ea_from_lookup.yaml")
    t["experiment"]["old_key"] = 1
    c = VR.show_yaml(head, "configs/experiments/ea_from_lookup_tensor_cuda.yaml")
    try:
        VR.build_config(t, c, base)
        e = None
    except VR.VersionRunError as err:
        e = str(err)
    check("an experiment key only the old template has -> refused", e and "old_key" in e)


def test_markers():
    print("\n[3] Generation markers")
    line = "[12:00:00] [Gen {k}/5] Saved. Max: 300.00, Median: 50.00, Mean: {m:.2f}, Std: 1.00"
    lines = [(0.5, "start"), (10.0, line.format(k=2, m=40)), (13.5, line.format(k=3, m=41)),
             (17.25, line.format(k=4, m=42)), (21.0, line.format(k=5, m=43)), (21.1, "Done.")]
    r = VR.parse_markers(lines, 5)
    check(f"generations 2-4 timed exactly ({r['gen']}, {r['seconds']})",
          r["gen"] == [2, 3, 4] and r["seconds"] == [3.5, 3.75, 3.75] and r["mean_lifespan"] == [41, 42, 43])
    try:
        VR.parse_markers(lines[:3] + lines[4:], 5)
        e = None
    except VR.VersionRunError as err:
        e = str(err)
    check(f"a missing marker -> error ({e})", e and "expected" in e)


def tiny_jobfile(tmp):
    jobs = []
    for commit, lab in (("8e17e694c7", "original scalar"), ("HEAD", "today scalar")):
        jobs.append(dict(name=f"tiny_{lab.split()[0]}", commit=commit, template="ea_from_lookup",
                         canonical="ea_from_lookup_tensor_cuda", backend="scalar", group="original_n11",
                         label=lab, P=2, set={"experiment.population_size": 2, "experiment.n_runs": 4,
                                              "experiment.evolutionary_algorithm.num_generations": 4,
                                              "experiment.evolutionary_algorithm.elite_size": 1}))
    p = Path(tmp) / "tiny.yaml"
    p.write_text(yaml.safe_dump({"description": "test", "jobs": jobs}, sort_keys=False))
    return p


def test_end_to_end(tmp):
    print("\n[4] End to end (scalar, tiny), from a temporary worktree")
    wt, out = Path(tmp) / "wt", Path(tmp) / "report"
    jf = tiny_jobfile(tmp)
    try:
        rc = VR.main([str(jf), "--worktree", str(wt), "--out", str(out), "--quiet"])
        check(f"both jobs ok (exit {rc})", rc == 0)
        res = {p.stem: json.loads(p.read_text()) for p in (out / "jobs").glob("*.json")}
        check("the original job ran 8e17e694c7, the other HEAD",
              res["tiny_original"]["commit"].startswith("8e17e694c7")
              and res["tiny_today"]["commit"] == VR.resolve("HEAD"))
        check("2 generations timed per job (4 generations: indices 2 and 3)",
              all(r["gen"] == [2, 3] and len(r["seconds"]) == 2 for r in res.values()))
        log = (out / "logs" / "tiny_original.log").read_text().splitlines()
        check("the log holds timestamped lines", log and all(" | " in x for x in log[:5]))
        check("summary.csv and generations.csv written",
              (out / "summary.csv").exists() and (out / "generations.csv").exists())
        before = {p.name: p.stat().st_mtime_ns for p in (out / "jobs").glob("*.json")}
        rc = VR.main(["--resume", str(out), "--quiet"])
        after = {p.name: p.stat().st_mtime_ns for p in (out / "jobs").glob("*.json")}
        check("resume skips finished jobs", rc == 0 and before == after)
        rc = PV.main([str(out), "--out", str(Path(tmp) / "plot")])
        check("plot_versions writes the original figure and the CSV",
              rc == 0 and (Path(tmp) / "plot" / "versions_original.png").exists()
              and (Path(tmp) / "plot" / "versions.csv").exists())
        test_run_refusals(tmp, wt, out)
    finally:
        subprocess.run(["git", "-C", str(ROOT), "worktree", "remove", "--force", str(wt)],
                       capture_output=True)


def test_run_refusals(tmp, wt, out):
    print("\n[5] Refusals while running; waterfall arithmetic")
    sha = VR.git("rev-parse", "HEAD", cwd=wt)
    f = wt / "mvb" / "world.py"
    text = f.read_text()
    try:
        f.write_text(text + "\n# local change\n")
        e = raises(lambda: VR.checkout(wt, sha))
        check("a worktree with a local change -> refused", e and "local changes" in e)
    finally:
        f.write_text(text)
    d = Path(tmp) / "moved_head"
    shutil.copytree(out, d)
    run = json.loads((d / "run.json").read_text())
    run["head"] = VR.resolve("8e17e694c7")
    (d / "run.json").write_text(json.dumps(run))
    e = raises(lambda: VR.execute(d, echo=False))
    check("this checkout's HEAD changed since planning -> refused", e and "HEAD changed" in e)
    jobs = [dict(group="waterfall_a", usable=True, status="ok", seconds=s, label=str(i), sha="0" * 10,
                 backend="scalar", wall_s=1.0) for i, s in enumerate(([8, 8, 8], [4, 4, 4], [1, 1, 1]))]
    jobs.insert(2, dict(jobs[1], usable=False, status="failed: x"))
    rows = PV.waterfall_rows(jobs)
    check("waterfall speed-ups: x2 vs previous, x8 vs original; a failed commit is skipped over",
          rows[1]["vs_prev"] == 2 and rows[2]["med"] is None and rows[3]["vs_prev"] == 4
          and rows[3]["vs_original"] == 8)


def main():
    test_build()
    test_build_refusals()
    test_markers()
    with tempfile.TemporaryDirectory() as tmp:
        test_end_to_end(tmp)
    failed = [n for n, ok in _RESULTS if not ok]
    print(f"\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    for n in failed:
        print(f"  FAILED: {n}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
