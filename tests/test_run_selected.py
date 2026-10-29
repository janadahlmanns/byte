#!/usr/bin/env python3
"""Tests for the config selector simulate/run_selected.py (plan Step 14).

  [1] the facts shown for every config equal the values in its YAML
  [2] configs that cannot run here are unselectable, with the right reason
  [3] a queue of two configs runs in order, and each result is bit-identical to a
      direct `run_ea` run of the same config
  [4] system.json has every field; nothing unknown is left out
  [5] the selection logic (scripted keys)
  [6] negative controls: a failing config does not stop the queue; a forced
      unselectable config is refused before anything runs; a question no one answers
      fails instead of hanging; "exit 0 but no results" is a failure; no terminal ->
      the screen refuses with the -it hint

Everything is written to a temporary folder.

Usage
-----
    python -m tests.test_run_selected
"""

import contextlib
import copy
import csv
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import simulate.run_selected as RS  # noqa: E402
from tests.test_ea_tensor import diff_h5, run_ea, tensor_block  # noqa: E402

_RESULTS = []
BASE = ROOT / "configs" / "experiments" / "test_ea_predrawn.yaml"


def check(name, condition):
    _RESULTS.append((name, bool(condition)))
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}")
    return bool(condition)


def base_cfg(gens=2):
    cfg = yaml.safe_load(BASE.read_text())
    cfg["experiment"]["evolutionary_algorithm"]["num_generations"] = gens
    return cfg


def write_cfg(cfg, tmp, tag):
    out = Path(tmp) / f"out_{tag}"
    out.mkdir(exist_ok=True)
    cfg = copy.deepcopy(cfg)
    cfg["experiment"]["output_folder"] = str(out) + "/"
    p = Path(tmp) / f"{tag}.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return p


def selector(args, timeout=1800):
    return subprocess.run([sys.executable, "-m", "simulate.run_selected"] + args, cwd=ROOT,
                          text=True, capture_output=True, timeout=timeout,
                          stdin=subprocess.DEVNULL)


def read_summary(report_root):
    dirs = sorted(Path(report_root).iterdir())
    if len(dirs) != 1:
        return None, []
    with open(dirs[0] / "summary.csv", newline="") as f:
        return dirs[0], list(csv.DictReader(f))


# ============================================================

def test_facts():
    print("\n[1] Config facts equal the YAML")
    every = ["cpu", "cuda", "mps"]
    files = RS.config_files()
    check(f"every YAML in configs/experiments is listed ({len(files)})",
          len([f for f in files if f[1].parent == RS.CONFIG_DIR])
          == len(list(RS.CONFIG_DIR.glob("*.yaml")) + list(RS.CONFIG_DIR.glob("*.yml"))))
    bad = []
    for name, path in files:
        info = RS.describe(path, name, set(every))
        e = yaml.safe_load(path.read_text())["experiment"]
        if e.get("evolutionary_algorithm_enabled") is not True:
            continue
        ev = e.get("evaluator") or {"backend": "scalar"}
        want = (str(e["population_size"]), str(e["n_runs"]),
                str(e["evolutionary_algorithm"]["num_generations"]),
                "lookup" if e["genome_type"].startswith("lookup") else e["genome_type"],
                ev["device"] if ev["backend"] == "tensor" else "cpu",
                str(yaml.safe_load(path.read_text())["brain"]["n_neurons"]))
        got = (info.variants, info.runs, info.gens, info.genome, info.device, info.neurons)
        if got != want:
            bad.append((name, got, want))
    check(f"every EA config's row equals its YAML ({len(bad)} mismatches)", not bad)
    for b in bad:
        print("      ", b)
    rows = {n: RS.describe(p, n, set(every)) for n, p in files}
    for name, want in [
        ("ea_from_lookup_tensor_cuda", ("260", "300", "50", "lookup", "cuda", "11")),
        ("ea_from_random_large_tensor_cuda", ("1000", "300", "150", "random", "cuda", "50")),
        ("ea_from_random", ("260", "300", "70", "random", "cpu", "11")),
    ]:
        i = rows[name]
        check(f"{name}: {want}",
              (i.variants, i.runs, i.gens, i.genome, i.device, i.neurons) == want
              and i.selectable)


def test_reasons(tmp):
    print("\n[2] Unselectable configs and their reasons")
    cases = []
    c = base_cfg(); c["experiment"]["evaluator"] = tensor_block(device="mps")
    cases.append(("mps", c, "MPS not available"))
    c = base_cfg(); c["experiment"]["evaluator"] = tensor_block(device="cuda")
    cases.append(("cuda", c, "CUDA not available (Docker: add --gpus all)"))
    c = base_cfg(); c["experiment"]["viz_enabled"] = True
    cases.append(("viz", c, "visualisation on"))
    c = base_cfg(); c["experiment"]["genome_type"] = "from_file"
    cases.append(("from_file", c, "needs input files"))
    c = base_cfg(); c["experiment"]["evolutionary_algorithm_enabled"] = False
    cases.append(("not_ea", c, "not an EA config"))
    c = base_cfg(); c["experiment"]["enable_per_run_tracking"] = True
    cases.append(("tracking", c, "tracking on"))
    c = base_cfg(); del c["experiment"]["n_runs"]
    cases.append(("missing", c, "missing key experiment.n_runs"))
    for tag, cfg, reason in cases:
        info = RS.describe(write_cfg(cfg, tmp, f"r_{tag}"), tag, {"cpu"})
        check(f"{tag}: unselectable, reason {reason!r} (got {info.reasons})",
              not info.selectable and any(reason in r for r in info.reasons))
    info = RS.describe(write_cfg(base_cfg(), tmp, "r_ok"), "ok", {"cpu"})
    check("the unchanged base config is selectable", info.selectable)
    info = RS.describe(write_cfg(cases[0][1], tmp, "r_mps_ok"), "mps", {"cpu", "mps"})
    check("an mps config is selectable where MPS is available", info.selectable)
    missing = RS.describe(write_cfg(cases[-1][1], tmp, "r_missing2"), "m", {"cpu"})
    check("a missing value is shown as '?'", missing.runs == "?")


def test_queue(tmp):
    print("\n[3] A queue of two configs: order, report, identical results")
    scalar = base_cfg()
    tensor = base_cfg(); tensor["experiment"]["evaluator"] = tensor_block()
    paths = [write_cfg(tensor, tmp, "q_tensor"), write_cfg(scalar, tmp, "q_scalar")]
    root = Path(tmp) / "reports_q"
    p = selector(["--configs", ",".join(map(str, paths)), "--report-dir", str(root)])
    if not check(f"the selector exits 0 (got {p.returncode})", p.returncode == 0):
        print(p.stdout[-3000:], p.stderr[-2000:])
        return None
    rdir, rows = read_summary(root)
    check("two summary rows, in the given order",
          [r["config"] for r in rows] == [str(x) for x in paths])
    check("both ok", [r["status"] for r in rows] == ["ok", "ok"])
    check("each HDF5 exists", all(r["hdf5"] and Path(r["hdf5"]).exists() for r in rows))
    check("tensor config: 2 timed generations; scalar config: none (wall time only)",
          rows[0]["gens_timed"] == "2" and rows[0]["gen0_s"] and rows[0]["gens_1plus_mean_s"]
          and rows[1]["gens_timed"] == "0" and float(rows[1]["wall_s"]) > 0)
    with open(rdir / "generations.csv", newline="") as f:
        g = list(csv.DictReader(f))
    check("generations.csv has the tensor config's 2 generations",
          [(r["config"], r["generation"]) for r in g] == [(str(paths[0]), "0"),
                                                         (str(paths[0]), "1")])
    check("one log per config, with the run's output",
          all("EVOLUTIONARY ALGORITHM COMPLETED" in (rdir / r["log"]).read_text()
              for r in rows))
    check("the summary names the report folder", f"Send this folder: {rdir}" in p.stdout)
    check("compile cache state recorded", all(r["compile_cache_at_start"] in ("empty", "warm")
                                              for r in rows))
    for cfg, row, tag in ((tensor, rows[0], "tensor"), (scalar, rows[1], "scalar")):
        h5, out, rc = run_ea(cfg, tmp, f"direct_{tag}")
        problems, n = diff_h5(row["hdf5"], h5) if h5 else (["no output"], 0)
        check(f"{tag}: all {n} datasets identical to a direct run_ea run "
              f"({len(problems)} differences)", rc == 0 and not problems)
        for m in problems[:5]:
            print("      -", m)
    return rdir


def test_system(rdir):
    print("\n[4] system.json")
    if rdir is None:
        check("report folder from [3] exists", False)
        return
    s = json.loads((rdir / "system.json").read_text())
    keys = ["time", "code_version", "in_container", "hostname", "os", "cpu_model",
            "cpu_count", "ram_gib", "memory_limit_gib", "python", "torch", "torch_cuda",
            "cuda_available", "mps_available", "gpus", "nvidia_smi_gpus"]
    check(f"every field present ({sorted(set(keys) - set(s))} missing)", set(keys) <= set(s))
    empty = [k for k in keys if s.get(k) in (None, "", [])]
    check(f"no field is empty ({empty})", not empty)
    check("unreadable fields are explicit 'unavailable: <reason>'",
          all(len(v) > len(RS.UNAVAILABLE) for v in s.values()
              if isinstance(v, str) and v.startswith("unavailable")))
    check(f"code version recorded ({s['code_version']})",
          not str(s["code_version"]).startswith(RS.UNAVAILABLE))
    snaps = json.loads((rdir / "gpu_snapshots.json").read_text())
    check("a GPU snapshot before and after each config",
          len(snaps) == 2 and all("before" in x and "after" in x for x in snaps))


def test_selection():
    print("\n[5] Selection logic")
    mk = lambda n, ok: RS.ConfigInfo(n, Path(n), reasons=[] if ok else ["no"])
    infos = [mk("a", True), mk("b", False), mk("c", True), mk("d", True)]
    s = RS.Selection(infos)
    for k in ["down", "down", "space", "up", "up", "space"]:
        s.key(k)
    check("tick order is run order (c, then a)", [i.name for i in s.chosen()] == ["c", "a"])
    s.key("down"); s.key("space")
    check("an unselectable config cannot be ticked, with a message",
          [i.name for i in s.chosen()] == ["c", "a"] and "cannot run here" in s.message)
    s.key("down"); s.key("space")
    check("unticking removes it and renumbers", [i.name for i in s.chosen()] == ["a"])
    s.key("a")
    check("'a' adds every selectable config, keeping the earlier order",
          [i.name for i in s.chosen()] == ["a", "c", "d"])
    s.key("a")
    check("'a' again unticks all", s.chosen() == [])
    check("enter with nothing ticked does not run", s.key("enter") is None and "Nothing" in s.message)
    s.key("space")
    check("enter runs, q quits", s.key("enter") == "run" and s.key("q") == "quit")
    s.cursor = 0
    s.key("up")
    check("the cursor wraps around", s.cursor == len(infos) - 1)


def test_negative(tmp):
    print("\n[6] Negative controls")
    bad = base_cfg(); bad["experiment"]["evaluator"] = {"backend": "tensor", "device": "cpu",
                                                        "dtype": "bogus"}
    paths = [write_cfg(bad, tmp, "n_bad"), write_cfg(base_cfg(), tmp, "n_good")]
    root = Path(tmp) / "reports_n"
    p = selector(["--configs", ",".join(map(str, paths)), "--report-dir", str(root)])
    _, rows = read_summary(root)
    check(f"a failing config: exit code non-zero (got {p.returncode})", p.returncode == 1)
    check("... it is recorded as failed, and the next config still ran ok",
          [r["status"].split(" ")[0] for r in rows] == ["failed", "ok"])

    viz = base_cfg(); viz["experiment"]["viz_enabled"] = True
    root = Path(tmp) / "reports_v"
    p = selector(["--configs", str(write_cfg(viz, tmp, "n_viz")), "--report-dir", str(root)])
    check("a forced viz config is refused before anything runs",
          p.returncode == 2 and "visualisation on" in p.stdout and not root.exists())

    log = Path(tmp) / "eof.log"
    rc, wall, *_ = RS.run_one([sys.executable, "-c", "input(); input()"], log, echo=False)
    check(f"a second question gets EOF and fails (exit code {rc}), no hang",
          rc not in (0, None) and "EOFError" in log.read_text())

    track = base_cfg(); track["experiment"]["enable_per_run_tracking"] = True
    info = RS.ConfigInfo("track", write_cfg(track, tmp, "n_track"))     # bypasses describe
    results, _ = RS.run_queue([info], Path(tmp) / "reports_t", RS.torch_probe(), echo=False)
    check(f"run_ea answering 'n' to its tracking question exits 0 without results -> "
          f"failed ({results[0].status})",
          results[0].exit_code == 0 and results[0].status.startswith("failed")
          and "no results" in results[0].status)

    p = selector([])
    check("no terminal: the screen refuses with the -it hint",
          p.returncode != 0 and "add -it" in (p.stdout + p.stderr))


def test_presets(tmp):
    print("\n[7] Presets: loading, errors, pre-ticking")
    d = Path(tmp) / "presets"
    d.mkdir()
    good = {"description": "x", "configs": ["test_ea_predrawn", "test_ea"]}
    cases = [("unknown_key", {**good, "config": []}, "unknown keys ['config']"),
             ("no_configs", {"description": "x"}, "missing 'configs'"),
             ("empty", {**good, "configs": []}, "non-empty list"),
             ("dup", {**good, "configs": ["test_ea", "test_ea"]}, "duplicate configs"),
             ("ref_outside", {**good, "references": {"ea_from_lookup": "x.h5"}},
              "references for configs not in the preset")]
    for tag, d_, msg in cases:
        p = d / f"{tag}.yaml"
        p.write_text(yaml.safe_dump(d_))
        try:
            RS.load_preset(p)
            got = "no error"
        except RS.PresetError as e:
            got = str(e)
        check(f"{tag}: PresetError containing {msg!r} (got {got!r})", msg in got)
    devs = {"cpu"}
    infos = RS.all_infos(devs)
    p = d / "typo.yaml"
    p.write_text(yaml.safe_dump({**good, "configs": ["test_ea", "test_eaa"]}))
    try:
        RS.preset_infos(RS.load_preset(p), infos, devs)
        got = "no error"
    except RS.PresetError as e:
        got = str(e)
    check(f"an unknown name is refused, not skipped ({got!r})", "test_eaa" in got)
    p = d / "mixed.yaml"
    p.write_text(yaml.safe_dump({**good, "configs": ["test_ea", "test_ea_tensor", "test_ea_predrawn"]}))
    sel = RS.preset_selection(infos, RS.load_preset(p), devs)
    check("pre-ticked in preset order, the unselectable entry (mps here) skipped",
          [i.name for i in sel.chosen()] == ["test_ea", "test_ea_predrawn"])
    check(f"... and the screen says so ({sel.message[:70]}...)",
          "cannot run here" in sel.message and "test_ea_tensor" in sel.message)
    both = {"cpu", "mps"}
    sel = RS.preset_selection(RS.all_infos(both), RS.load_preset(p), both)
    check("where everything can run, the message says enter runs it",
          "Enter runs it" in sel.message)


def test_preset_run(tmp):
    print("\n[8] --preset: order, selection record, HDF5 copies, reference")
    a = write_cfg(base_cfg(), tmp, "p_a")
    b = write_cfg(base_cfg(), tmp, "p_b")
    pdir = Path(tmp) / "presets_run"
    pdir.mkdir()
    ref = Path(tmp) / "ref.h5"
    (pdir / "small.yaml").write_text(yaml.safe_dump(
        {"description": "two", "configs": [str(a), str(b)], "references": {str(a): str(ref)}}))
    old = RS.PRESET_DIR
    RS.PRESET_DIR = pdir
    root = Path(tmp) / "reports_p"
    try:
        # 1st run: the reference does not exist yet -> explicit "unavailable"
        with contextlib.redirect_stdout(io.StringIO()):
            rc = RS.main(["--preset", "small", "--report-dir", str(root)])
        rdir, rows = read_summary(root)
        check(f"exit code 0 and both ok ({rc}, {[r['status'] for r in rows]})",
              rc == 0 and [r["status"] for r in rows] == ["ok", "ok"])
        check("run in the preset's order", [r["config"] for r in rows] == [str(a), str(b)])
        s = json.loads((rdir / "system.json").read_text())
        check(f"system.json: selection 'preset small (unchanged)' ({s['run']['selection']})",
              s["run"]["selection"] == "preset small (unchanged)"
              and s["run"]["selected"] == [str(a), str(b)])
        copies = [rdir / r["hdf5_copy"] for r in rows]
        check("each HDF5 copied into results/, byte-identical to the original",
              all(c.parent == rdir / "results" and c.read_bytes()
                  == (RS.ROOT / r["hdf5"] if not Path(r["hdf5"]).is_absolute()
                      else Path(r["hdf5"])).read_bytes() for c, r in zip(copies, rows)))
        check("missing reference -> 'unavailable: reference not found'",
              rows[0]["matches_reference"].startswith("unavailable: reference not found"))
        check("a config without a reference -> empty column", rows[1]["matches_reference"] == "")
        # 2nd run: the reference is the first run's result -> identical
        import shutil
        shutil.copy(copies[0], ref)
        shutil.rmtree(root)
        with contextlib.redirect_stdout(io.StringIO()):
            RS.main(["--preset", "small", "--report-dir", str(root)])
        _, rows = read_summary(root)
        check(f"rerun vs the first run as reference: {rows[0]['matches_reference']!r}",
              rows[0]["matches_reference"].startswith("identical ("))
    finally:
        RS.PRESET_DIR = old


def test_compare(tmp):
    print("\n[9] Reference comparison")
    import h5py
    import shutil
    a = write_cfg(base_cfg(), tmp, "c_a")
    h5, _, rc = run_ea(yaml.safe_load(a.read_text()), tmp, "c_run")
    if not check("a result file to compare", rc == 0 and h5):
        return
    ref = Path(tmp) / "c_ref.h5"
    shutil.copy(h5, ref)
    check("identical to a copy of itself", RS.compare_to_reference(h5, ref).startswith("identical ("))
    changed = Path(tmp) / "c_changed.h5"
    shutil.copy(h5, changed)
    with h5py.File(changed, "a") as f:
        st = f["generation_stats"][()]
        st[1]["mean"] += 1.0
        f["generation_stats"][...] = st
    got = RS.compare_to_reference(changed, ref)
    check(f"one changed generation -> differs, with the generation ({got!r})",
          got.startswith("differs: 1 of") and "generation_stats" in got
          and f"generation {int(st[1]['generation'])}" in got)
    cfgdiff = Path(tmp) / "c_cfg.h5"
    shutil.copy(h5, cfgdiff)
    with h5py.File(cfgdiff, "a") as f:
        f.attrs["experiment_n_runs"] = 999
    got = RS.compare_to_reference(cfgdiff, ref)
    check(f"a changed config attribute -> not comparable ({got!r})",
          got.startswith("not comparable") and "experiment_n_runs" in got)
    folder = Path(tmp) / "c_folder.h5"
    shutil.copy(h5, folder)
    with h5py.File(folder, "a") as f:
        f.attrs["experiment_output_folder"] = "somewhere/else/"
    check("only the output folder differs -> still identical",
          RS.compare_to_reference(folder, ref).startswith("identical ("))
    lfs = Path(tmp) / "c_lfs.h5"
    lfs.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:0\nsize 1\n")
    got = RS.compare_to_reference(h5, lfs)
    check(f"an LFS pointer as reference -> unavailable, says git lfs pull ({got[:60]}...)",
          got.startswith("unavailable") and "git lfs pull" in got)
    check("missing reference -> unavailable",
          RS.compare_to_reference(h5, Path(tmp) / "nope.h5").startswith("unavailable: reference not found"))


def test_default_preset():
    print("\n[10] The shipped default preset")
    presets = RS.load_presets()
    if not check("configs/presets/tester_default.yaml exists and loads", RS.DEFAULT_PRESET in presets):
        return
    p = presets[RS.DEFAULT_PRESET]
    devs = {"cpu", "cuda", "mps"}
    rows = RS.preset_infos(p, RS.all_infos(devs), devs)
    check(f"every entry resolves and can run on a CUDA machine ({len(rows)} configs)",
          all(r.selectable for r in rows))
    check("the lookup reproducibility run is in the preset, with a reference to compare against",
          "ea_from_lookup_tensor_cuda" in p.configs and "ea_from_lookup_tensor_cuda" in p.references)
    for name, ref in p.references.items():
        check(f"reference for {name} exists: {ref.relative_to(RS.ROOT)} "
              f"(add the university PC's result file; see plan Step 17)", ref.is_file())


def main():
    with tempfile.TemporaryDirectory(prefix="test_run_selected_") as tmp:
        test_facts()
        test_reasons(tmp)
        rdir = test_queue(tmp)
        test_system(rdir)
        test_selection()
        test_negative(tmp)
        test_presets(tmp)
        test_preset_run(tmp)
        test_compare(tmp)
        test_default_preset()
    n_fail = sum(1 for _, ok in _RESULTS if not ok)
    if n_fail:
        print(f"\nFAILED -- {n_fail} of {len(_RESULTS)} checks failed")
        return 1
    print(f"\nOK -- {len(_RESULTS)}/{len(_RESULTS)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
