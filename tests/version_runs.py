#!/usr/bin/env python3
"""Run plain run_ea of older commits for timing (plan_evotorch.md, Step 29).

Items 3 (the original code) and 4 (the version waterfall). Each job runs
`python -m simulate.run_ea --config <file>` of ONE commit, from a git worktree that is
switched from commit to commit (this checkout is never touched), with a config built
from that commit's own template file and the canonical values (this checkout's HEAD):

  - model sections (world, food, worm, brain) and the experiment / EA values are copied
    from the canonical config into the commit's template, key by key;
  - the job's `set` keys are applied last;
  - the result may differ from canonical + set ONLY in the allowed keys (the evaluator
    block, mutation_rng where the commit predates it, names and the output folder),
    otherwise the job is refused before anything runs.

Timing: run_ea prints "[Gen k/N] Saved" at the end of every generation from index 1 on,
in every commit. The runner timestamps every output line on arrival (the child runs with
PYTHONUNBUFFERED=1); the interval between consecutive markers is one whole EA
generation (mutation, evaluation, selection, HDF5), measured the same way for every
commit. Generation 0 and 1 are not timed (start-up, compile). Also: the total wall time.

Usage
-----
    python -m tests.version_runs configs/version_runs/original.yaml --worktree ../byte_versions
    python -m tests.version_runs --resume data/runtime_reports/<report folder>
The worktree is created on first use (git worktree add --detach, LFS skipped).
"""

import argparse
import copy
import csv
import datetime
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MARKER = re.compile(r"\[Gen (\d+)/(\d+)\] Saved\. Max: ([-\d.]+), Median: ([-\d.]+), Mean: ([-\d.]+)")
MODEL_SECTIONS = ("world", "food", "worm", "brain")
# Keys of the experiment block taken from the canonical config (where the template has them).
# May differ between the built config and canonical + set:
ALLOWED_DIFF = re.compile(r"^experiment\.(evaluator(\..*)?|evolutionary_algorithm\.mutation_rng|"
                          r"simulation_name|output_folder)$")
OFF_FLAGS = ("viz_enabled", "viz_brain_enabled", "enable_per_run_tracking",
             "enable_per_tick_tracking", "enable_heat_map_tracking")


class VersionRunError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# git
# ---------------------------------------------------------------------------

def git(*args, cwd=ROOT, env=None):
    p = subprocess.run(["git", "-C", str(cwd), *args], text=True, capture_output=True,
                       env=dict(os.environ, **(env or {})))
    if p.returncode != 0:
        raise VersionRunError(f"git {' '.join(args)} failed: {p.stderr.strip()}")
    return p.stdout.strip()


def resolve(commit):
    return git("rev-parse", f"{'HEAD' if commit == 'HEAD' else commit}^{{commit}}")


def show_yaml(commit, path):
    p = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{path}"], text=True,
                       capture_output=True)
    if p.returncode != 0:
        raise VersionRunError(f"{path} does not exist at {commit[:10]}: {p.stderr.strip()}")
    return yaml.safe_load(p.stdout)


def prepare_worktree(wt):
    wt = Path(wt).resolve()
    if wt == ROOT:
        raise VersionRunError("--worktree must be a separate worktree, not this checkout")
    if not wt.exists():
        git("worktree", "add", "--detach", str(wt), "HEAD", env={"GIT_LFS_SKIP_SMUDGE": "1"})
    if not (wt / ".git").exists():
        raise VersionRunError(f"{wt} exists but is not a git worktree")
    return wt


def checkout(wt, sha):
    git("-c", "advice.detachedHead=false", "checkout", "--detach", sha, cwd=wt,
        env={"GIT_LFS_SKIP_SMUDGE": "1"})
    head = git("rev-parse", "HEAD", cwd=wt)
    if head != sha:
        raise VersionRunError(f"worktree {wt} is at {head[:10]}, expected {sha[:10]}")
    dirty = git("status", "--porcelain", "--untracked-files=no", cwd=wt)
    if dirty:
        raise VersionRunError(f"worktree {wt} has local changes:\n{dirty}")


# ---------------------------------------------------------------------------
# configs
# ---------------------------------------------------------------------------

def flat(d, prefix=""):
    out = {}
    if isinstance(d, dict) and d:
        for k, v in d.items():
            out.update(flat(v, f"{prefix}{k}."))
    elif isinstance(d, list) and d and all(isinstance(x, dict) for x in d):
        for i, v in enumerate(d):
            out.update(flat(v, f"{prefix}{i}."))
    else:
        out[prefix[:-1]] = d
    return out


def _set(cfg, dotted, value, must_exist=True):
    cur = cfg
    *head, leaf = dotted.split(".")
    for k in head:
        if k not in cur:
            raise VersionRunError(f"key {dotted}: {k!r} missing")
        cur = cur[k]
    if must_exist and leaf not in cur:
        raise VersionRunError(f"key {dotted} does not exist in this commit's template "
                              f"(the commit would ignore it)")
    cur[leaf] = value


def build_config(template, canonical, job):
    """The commit's template with the canonical values and the job's `set` keys; refuses
    any difference from canonical + set beyond ALLOWED_DIFF."""
    cfg = copy.deepcopy(template)
    for sec in MODEL_SECTIONS:
        if sec not in canonical:
            raise VersionRunError(f"canonical config has no {sec!r} section")
        cfg[sec] = copy.deepcopy(canonical[sec])
    ce, te = canonical["experiment"], cfg["experiment"]
    for k, v in ce.items():
        if k in ("evaluator", "simulation_name", "output_folder"):
            continue
        if k == "evolutionary_algorithm":
            for kk, vv in v.items():
                if kk in te.get(k, {}):
                    te[k][kk] = copy.deepcopy(vv)
            continue
        if k in te:
            te[k] = copy.deepcopy(v)
    for k in OFF_FLAGS:
        if k in te:
            te[k] = False
    want = copy.deepcopy(canonical)
    for dotted, value in (job.get("set") or {}).items():
        _set(cfg, dotted, value)
        _set(want, dotted, value, must_exist=False)
    # `add`: keys the commit's code needs but its template lacks (b88b requires
    # evaluator.compile, its config file predates the key). Only inside the evaluator
    # block, which every commit's adapter validates itself (unknown keys are refused).
    for dotted, value in (job.get("add") or {}).items():
        if not dotted.startswith("experiment.evaluator."):
            raise VersionRunError(f"job {job['name']}: add.{dotted}: only experiment.evaluator.* "
                                  f"keys may be added")
        if job["backend"] != "cuda":
            raise VersionRunError(f"job {job['name']}: add is for CUDA jobs")
        _set(cfg, dotted, value, must_exist=False)
    if job["backend"] == "scalar":
        ev = te.get("evaluator")
        if ev is not None:
            te["evaluator"] = {"backend": "scalar"}
    else:
        ev = te.get("evaluator")
        if not ev or ev.get("backend") != "tensor":
            raise VersionRunError(f"job {job['name']}: template {job['template']} has no tensor "
                                  f"evaluator; it cannot run on CUDA")
        if ev.get("device") != "cuda":
            raise VersionRunError(f"job {job['name']}: template evaluator device is {ev.get('device')!r}")
    for k in OFF_FLAGS:
        if k in want["experiment"]:
            want["experiment"][k] = False
    fc, fw = flat(cfg), flat(want)
    diffs = {k: [fc.get(k, "<absent>"), fw.get(k, "<absent>")]
             for k in sorted(set(fc) | set(fw)) if fc.get(k, "<absent>") != fw.get(k, "<absent>")}
    bad = {k: v for k, v in diffs.items() if not ALLOWED_DIFF.match(k)}
    if bad:
        raise VersionRunError(f"job {job['name']}: the config at {job['commit']} differs from the "
                              f"canonical one beyond the allowed keys: {bad}")
    return cfg, diffs


# ---------------------------------------------------------------------------
# output parsing
# ---------------------------------------------------------------------------

def parse_markers(timed_lines, n_generations):
    """[(t, line)] -> {"gen": [index], "seconds": [...], "mean_lifespan": [...]} for the
    timed generations (index 2 .. N-1), and the per-marker statistics."""
    marks = []
    for t, line in timed_lines:
        m = MARKER.search(line)
        if m:
            k, n = int(m.group(1)), int(m.group(2))
            if n != n_generations:
                raise VersionRunError(f"marker says {n} generations, the config {n_generations}")
            marks.append((k - 1, t, float(m.group(3)), float(m.group(4)), float(m.group(5))))
    idx = [m[0] for m in marks]
    if idx != list(range(1, n_generations)):
        raise VersionRunError(f"generation markers {idx}, expected {list(range(1, n_generations))}")
    gens, secs, means = [], [], []
    for (g0, t0, *_a), (g1, t1, _mx, _md, mean) in zip(marks, marks[1:]):
        gens.append(g1)
        secs.append(t1 - t0)
        means.append(mean)
    final = dict(zip(("max", "median", "mean"), marks[-1][2:]))
    return {"gen": gens, "seconds": secs, "mean_lifespan": means, "final": final}


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------

def run_job(job, wt, rdir, echo=True):
    from simulate import machine_load as ML
    jdir = rdir / "runs" / job["name"]
    if jdir.exists():
        import shutil
        shutil.rmtree(jdir)                       # an interrupted earlier attempt
    jdir.mkdir(parents=True)
    cfg = yaml.safe_load((rdir / "configs" / f"{job['name']}.yaml").read_text())
    cfg["experiment"]["output_folder"] = str(jdir) + "/"
    cfg_path = jdir / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    checkout(wt, job["sha"])
    idle = ML.idle_check()
    env = dict(os.environ, PYTHONPATH=str(wt), PYTHONUNBUFFERED="1")
    log = open(rdir / "logs" / f"{job['name']}.log", "w")
    lines = []
    t_start = time.perf_counter()
    with ML.LoadSampler() as load:
        p = subprocess.Popen([sys.executable, "-m", "simulate.run_ea", "--config", str(cfg_path)],
                             cwd=str(wt), env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        p.stdin.write("n\n")                      # "Display generation statistics plot?"
        p.stdin.close()
        for line in p.stdout:
            t = time.perf_counter() - t_start
            lines.append((t, line.rstrip("\n")))
            log.write(f"{t:10.3f} | {line}")
            log.flush()
            if echo and (MARKER.search(line) or "Error" in line or "Traceback" in line):
                print(f"    {t:8.1f} s  {line.rstrip()}", flush=True)
        rc = p.wait()
    wall = time.perf_counter() - t_start
    log.close()
    res = dict(name=job["name"], commit=job["sha"], backend=job["backend"], group=job["group"],
               label=job["label"], P=job["P"], wall_s=wall, exit_code=rc,
               idle_check=idle.get("verdict"),
               disturbance="; ".join(x for x in (
                   ("before: " + idle["verdict"][len("busy: "):]) if idle["verdict"].startswith("busy") else "",
                   ("during: " + load.disturbance()) if load.disturbance() else "") if x),
               foreign_cores=load.foreign_cores())
    try:
        if rc != 0:
            raise VersionRunError(f"run_ea exited with {rc}")
        if len(list(jdir.glob("*.h5"))) != 1:
            raise VersionRunError("run_ea wrote no HDF5 (or more than one)")
        res.update(parse_markers(lines, cfg["experiment"]["evolutionary_algorithm"]["num_generations"]))
        res["status"] = "ok"
    except VersionRunError as e:
        res["status"] = f"failed: {e}"
    (rdir / "jobs" / f"{job['name']}.json").write_text(json.dumps(res, indent=1))
    return res


def plan(jobfile, rdir, wt):
    jf = yaml.safe_load(Path(jobfile).read_text())
    jobs = jf["jobs"]
    names = [j["name"] for j in jobs]
    if len(set(names)) != len(names):
        raise VersionRunError("duplicate job names")
    head = resolve("HEAD")
    for d in ("configs", "logs", "jobs", "runs"):
        (rdir / d).mkdir(parents=True, exist_ok=True)
    diffs = {}
    for j in jobs:
        for k in ("name", "commit", "template", "canonical", "backend", "group", "label", "P"):
            if k not in j:
                raise VersionRunError(f"job {j.get('name')}: missing {k!r}")
        if j["backend"] not in ("scalar", "cuda"):
            raise VersionRunError(f"job {j['name']}: backend must be scalar or cuda")
        j["sha"] = head if j["commit"] == "HEAD" else resolve(j["commit"])
        template = show_yaml(j["sha"], f"configs/experiments/{j['template']}.yaml")
        canonical = show_yaml(head, f"configs/experiments/{j['canonical']}.yaml")
        cfg, d = build_config(template, canonical, j)
        cfg["experiment"]["simulation_name"] = j["name"]
        (rdir / "configs" / f"{j['name']}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        diffs[j["name"]] = d
    (rdir / "config_diffs.json").write_text(json.dumps(diffs, indent=1, default=str))
    run = dict(jobfile=str(jobfile), description=jf.get("description", ""), head=head,
               worktree=str(wt), jobs=jobs, created=datetime.datetime.now().isoformat(timespec="seconds"))
    (rdir / "run.json").write_text(json.dumps(run, indent=1))
    from simulate.run_selected import system_info, torch_probe
    (rdir / "system.json").write_text(json.dumps(system_info(torch_probe()), indent=1, default=str))
    return run


def write_tables(rdir):
    run = json.loads((rdir / "run.json").read_text())
    rows, gens = [], []
    for j in run["jobs"]:
        f = rdir / "jobs" / f"{j['name']}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        secs = r.get("seconds") or []
        med = sorted(secs)[len(secs) // 2] if secs else ""
        rows.append(dict(name=r["name"], group=r["group"], label=r["label"], P=r["P"],
                         commit=r["commit"][:10], backend=r["backend"], status=r["status"],
                         wall_s=f"{r['wall_s']:.2f}", gens_timed=len(secs),
                         median_s_per_gen=(f"{med:.3f}" if secs else ""),
                         final_mean_lifespan=(r.get("final") or {}).get("mean", ""),
                         idle_check=r["idle_check"], disturbance=r["disturbance"]))
        for g, s, m in zip(r.get("gen", []), secs, r.get("mean_lifespan", [])):
            gens.append(dict(name=r["name"], generation=g, seconds=f"{s:.4f}", mean_lifespan=m))
    for fname, data in (("summary.csv", rows), ("generations.csv", gens)):
        if data:
            with open(rdir / fname, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0]))
                w.writeheader()
                w.writerows(data)
    return rows


def execute(rdir, echo=True):
    run = json.loads((rdir / "run.json").read_text())
    if resolve("HEAD") != run["head"]:
        raise VersionRunError(f"this checkout's HEAD changed since the run was planned "
                              f"({run['head'][:10]} -> {resolve('HEAD')[:10]}); the canonical "
                              f"configs and 'HEAD' jobs would differ")
    wt = prepare_worktree(run["worktree"])
    first = {}
    for i, j in enumerate(run["jobs"]):
        first.setdefault(j["sha"], i)
    # Jobs of one commit together (fewer worktree switches), commits in file order.
    order = sorted(range(len(run["jobs"])), key=lambda i: (first[run["jobs"][i]["sha"]], i))
    for i in order:
        j = run["jobs"][i]
        if (rdir / "jobs" / f"{j['name']}.json").exists():
            print(f"[{j['name']}] done earlier, skipped", flush=True)
            continue
        print(f"\n[{j['name']}] {j['label']}, {j['sha'][:10]}, {j['backend']}, P {j['P']}", flush=True)
        r = run_job(j, wt, rdir, echo)
        med = sorted(r.get("seconds") or [0])[len(r.get("seconds") or [0]) // 2]
        print(f"[{j['name']}] {r['status']}: wall {r['wall_s']:.1f} s"
              + (f", median {med:.2f} s per generation" if r["status"] == "ok" else ""), flush=True)
    rows = write_tables(rdir)
    failed = [r["name"] for r in rows if r["status"] != "ok"]
    print(f"\n{len(rows) - len(failed)}/{len(rows)} jobs ok" + (f"; failed: {failed}" if failed else "")
          + f"\nreport: {rdir}")
    return 1 if failed else 0


def default_dir(jobfile):
    gpu = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0).replace(" ", "-")
    except Exception:  # noqa: BLE001 -- only names the folder
        pass
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return ROOT / "data" / "runtime_reports" / f"{ts}_{gpu}_versions_{Path(jobfile).stem}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("jobfile", nargs="?")
    ap.add_argument("--worktree")
    ap.add_argument("--out")
    ap.add_argument("--resume", metavar="DIR")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    if a.resume:
        rdir = Path(a.resume).resolve()
    else:
        if not (a.jobfile and a.worktree):
            ap.error("a new run needs a job file and --worktree")
        rdir = (Path(a.out) if a.out else default_dir(a.jobfile)).resolve()
        if rdir.exists():
            raise VersionRunError(f"{rdir} exists; use --resume {rdir}")
        wt = prepare_worktree(a.worktree)
        plan(a.jobfile, rdir, wt)
        print(f"[version_runs] planned {a.jobfile} -> {rdir}", flush=True)
    return execute(rdir, echo=not a.quiet)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except VersionRunError as e:
        sys.exit(f"ERROR: {e}")
