#!/usr/bin/env python3
"""Tests for sweep files and the per-generation stats line (plan Step 15).

  [1] configs/sweeps/neuron_scaling.yaml expands to 7 points, each equal to its base
      config except the keys the sweep sets
  [2] every kind of sweep mistake fails loudly, and the selector greys the file out
  [3] a small sweep run through the selector is bit-identical to direct run_ea runs of
      handwritten configs with the same values
  [4] the stats line: memory form per device; on CUDA the peaks are plausible
  [5] run-ticks equal the sum of all lifespans
  [6] runs queued by the memory cap: the "all runs need" estimate equals the width
      rule's own numbers, and the summary column is empty exactly when nothing queued
  [7] the nvidia-smi sampler: explicit "unavailable" without nvidia-smi; the thread
      stops when the block ends, also on an exception

Usage
-----
    python -m tests.test_sweeps
"""

import contextlib
import copy
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mvb_torch.adapter as A  # noqa: E402
import simulate.run_selected as RS  # noqa: E402
from mvb_torch.generation import BASE_BYTES, make_simulation_rngs  # noqa: E402
from tests.devices import all_devices  # noqa: E402
from tests.test_ea_tensor import diff_h5, run_ea  # noqa: E402

_RESULTS = []
EXP = ROOT / "configs" / "experiments"


def check(name, condition):
    _RESULTS.append((name, bool(condition)))
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}")
    return bool(condition)


def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            out.update(flat(v, key + "."))
        else:
            out[key] = v
    return out


def load(name):
    return yaml.safe_load((EXP / f"{name}.yaml").read_text())


# ============================================================

def test_expansion():
    print("\n[1] neuron_scaling expands as written")
    path = ROOT / "configs" / "sweeps" / "neuron_scaling.yaml"
    sw = yaml.safe_load(path.read_text())
    points = RS.expand_sweep(path)
    ns = sw["vary"]["brain.n_neurons"]
    check(f"{len(ns)} points, named by n", [p[0] for p in points]
          == [f"neuron_scaling [n_neurons={n}]" for n in ns])
    base = flat(load(sw["base"]))
    bad = []
    for (name, cfg), n in zip(points, ns):
        f = flat(cfg)
        want = dict(base)
        want.update(sw["set"])
        want["brain.n_neurons"] = n
        want["experiment.simulation_name"] = base["experiment.simulation_name"] + f"_n_neurons-{n}"
        if f != want:
            bad.append((name, sorted(k for k in set(f) | set(want) if f.get(k) != want.get(k))))
    check(f"every point equals the base except set, n_neurons and simulation_name "
          f"({bad[:3]})", not bad)
    names = [cfg["experiment"]["simulation_name"] for _, cfg in points]
    check("simulation names are unique", len(set(names)) == len(names))
    infos = [i for i in RS.sweep_infos({"cpu", "cuda"}) if i.name.startswith("neuron_scaling [")]
    check("the selector shows them with n, 260 variants, 300 runs, 10 gens, cuda",
          [(i.neurons, i.variants, i.runs, i.gens, i.device) for i in infos]
          == [(str(n), "260", "300", "10", "cuda") for n in ns] and all(i.selectable for i in infos))


def test_errors(tmp):
    print("\n[2] Sweep mistakes fail loudly")
    d = Path(tmp) / "sweeps_bad"
    d.mkdir()
    good = {"base": "test_ea_tensor", "set": {"experiment.n_runs": 4},
            "vary": {"brain.n_neurons": [11, 14]}}
    cases = [
        ("typo", {**good, "vary": {"brain.n_nerons": [11]}}, "unknown key brain.n_nerons"),
        ("typo_set", {**good, "set": {"experiment.n_run": 4}}, "unknown key experiment.n_run"),
        ("type", {**good, "vary": {"brain.n_neurons": [11, 20.0]}}, "is float"),
        ("bool", {**good, "set": {"experiment.n_runs": True}}, "is bool"),
        ("base", {**good, "base": "no_such_config"}, "base config 'no_such_config' not found"),
        ("empty", {**good, "vary": {}}, "'vary' must be a non-empty mapping"),
        ("emptylist", {**good, "vary": {"brain.n_neurons": []}}, "must be a non-empty list"),
        ("dup", {**good, "vary": {"brain.n_neurons": [11, 11]}}, "duplicate values"),
        ("both", {**good, "set": {"brain.n_neurons": 12}}, "keys in both set and vary"),
        ("toplevel", {**good, "sett": {}}, "unknown top-level keys ['sett']"),
        ("novary", {"base": "test_ea_tensor"}, "missing 'vary'"),
    ]
    for tag, sw, msg in cases:
        p = d / f"{tag}.yaml"
        p.write_text(yaml.safe_dump(sw))
        try:
            RS.expand_sweep(p)
            got = "no error"
        except RS.SweepError as e:
            got = str(e)
        check(f"{tag}: SweepError containing {msg!r} (got {got!r})", msg in got)
    (d / "zz_good.yaml").write_text(yaml.safe_dump(good))
    infos = RS.sweep_infos({"cpu"}, sweep_dir=d)
    broken = [i for i in infos if i.name.endswith("(sweep)")]
    check(f"the selector shows each broken file as one greyed row ({len(broken)} of "
          f"{len(cases)})", len(broken) == len(cases)
          and all(not i.selectable and "broken sweep file" in i.reasons[0] for i in broken))
    check("the valid file next to them still gives its 2 points",
          [i.name for i in infos if not i.name.endswith("(sweep)")]
          == ["zz_good [n_neurons=11]", "zz_good [n_neurons=14]"])


def small_set(out):
    return {"experiment.output_folder": str(out) + "/",
            "experiment.population_size": 20,
            "experiment.n_runs": 4,
            "experiment.max_ticks": 200,
            "experiment.evolutionary_algorithm.num_generations": 2,
            "experiment.evolutionary_algorithm.elite_size": 4,
            "experiment.evaluator.device": "cpu",
            "experiment.evaluator.compile": False}


def test_identity(tmp):
    print("\n[3] A sweep through the selector == direct run_ea of handwritten configs")
    d = Path(tmp) / "sweeps_small"
    d.mkdir()
    out = Path(tmp) / "out_sweep"
    out.mkdir()
    sw = {"base": "test_ea_tensor", "set": small_set(out), "vary": {"brain.n_neurons": [11, 14]}}
    (d / "small.yaml").write_text(yaml.safe_dump(sw, sort_keys=False))
    infos = RS.sweep_infos({"cpu"}, sweep_dir=d)
    if not check("2 selectable points", len(infos) == 2 and all(i.selectable for i in infos)):
        for i in infos:
            print("      ", i.name, i.reasons)
        return
    with contextlib.redirect_stdout(io.StringIO()):
        results, rdir = RS.run_queue(infos, Path(tmp) / "reports_sweep", RS.torch_probe(),
                                     echo=False)
    check("both ok", [r.status for r in results] == ["ok", "ok"])
    check("the expanded configs were written into the report folder and run from there",
          all(Path(r.path).parent == rdir / "configs" and Path(r.path).is_file() for r in results))
    check("each HDF5 name carries its point's suffix",
          all(f"_n_neurons-{n}" in r.hdf5 for r, n in zip(results, (11, 14))))
    for r, n in zip(results, (11, 14)):
        cfg = load("test_ea_tensor")                   # handwritten, independently
        e = cfg["experiment"]
        e["population_size"], e["n_runs"], e["max_ticks"] = 20, 4, 200
        e["evolutionary_algorithm"]["num_generations"] = 2
        e["evolutionary_algorithm"]["elite_size"] = 4
        e["evaluator"]["device"], e["evaluator"]["compile"] = "cpu", False
        cfg["brain"]["n_neurons"] = n
        e["simulation_name"] += f"_n_neurons-{n}"
        h5, log, rc = run_ea(cfg, tmp, f"hand_{n}")
        problems, k = diff_h5(r.hdf5, h5) if h5 else (["no output"], 0)
        check(f"n={n}: all {k} datasets identical to the handwritten config's run "
              f"({len(problems)} differences)", rc == 0 and not problems)
        for m in problems[:5]:
            print("      -", m)
    mem = [RS.memory_summary(r.gen_stats) for r in results]
    check("stats parsed for both generations of both points",
          all(len(r.gen_stats) == 2 == len(r.gen_times) for r in results))
    check("cpu: memory 'unavailable on cpu', not queued, no estimate column",
          all(m["vram_measure"] == "unavailable on cpu" and m["vram_peak_mib"] == ""
              and m["runs_queued"] == "no" and m["vram_needed_all_runs_est_mib"] == ""
              for m in mem))
    check("us per run-tick over gens 1+ in the summary",
          all(float(RS.us_per_run_tick_gens_1plus(r)) > 0 for r in results))


def live_eval(dev, P=6, gens=1, patch_width=None):
    """One in-process tensor generation; returns (stdout, all lifespans)."""
    from mvb.genome.generate_genome_random import generate_random_genome
    cfg = load("test_ea_predrawn")
    del cfg["experiment"]["predrawn_randomness"]
    e, w, wm = cfg["experiment"], cfg["world"], cfg["worm"]
    ev_cfg = {"backend": "tensor", "device": dev, "dtype": "float32", "philox_rounds": 10,
              "compile": False}
    genomes = [generate_random_genome(cfg, rng_seed=70 + i) for i in range(P)]
    phases = sorted(cfg["food"], key=lambda p: p["phase_from"])
    rn, rd, rw = make_simulation_rngs(3)
    args = (genomes, cfg, "unused", "unused", False, False, False, False, False, 4, 4,
            P, rn, rd, rw, wm["decisionmaking"]["version"], e["max_ticks"], e["n_runs"],
            w["grid_width"], w["grid_height"], w["start_pos"], wm["speed"],
            wm["energy_capacity"], wm["metabolic_rate"], wm["movement_cost"],
            wm["sensors"]["active"], phases[0], cfg["brain"])
    buf = io.StringIO()
    orig = A.choose_width
    try:
        if patch_width:
            A.choose_width = patch_width
        with contextlib.redirect_stdout(buf):
            ev = A.make_tensor_evaluator(ev_cfg, cfg, None)
            out, _ = ev(*args, switch_phases=phases[1:], randomness_cfg=None)
    finally:
        A.choose_width = orig
    return buf.getvalue(), np.stack([out[v]["lifespan_vector"] for v in range(P)])


def stats_of(text):
    return [RS.parse_stats(m) for m in RS.STATS_LINE.finditer(text)]


def test_stats_line():
    print("\n[4]+[5] The stats line on every device; run-ticks")
    import torch
    for dev in all_devices():
        text, L = live_eval(dev)
        st = stats_of(text)
        if not check(f"{dev}: one stats line, parsed", len(st) == 1):
            print(text[-1500:])
            continue
        s = st[0]
        check(f"{dev}: run-ticks {s['run_ticks']} == sum of all lifespans {int(L.sum())}",
              s["run_ticks"] == int(L.sum()))
        if dev == "cpu":
            check("cpu: 'unavailable on cpu'", s["vram_measure"] == "unavailable on cpu"
                  and s["vram_mib"] is None)
        elif dev == "mps":
            check(f"mps: driver allocation, labelled as no peak ({s['vram_mib']} MiB)",
                  s["vram_measure"].startswith("mps driver allocated") and s["vram_mib"] > 0)
        else:
            m = RS.re.search(r"peak reserved (\d+) MiB, peak allocated (\d+) MiB", text)
            total = torch.cuda.get_device_properties(0).total_memory / 2**20
            check(f"cuda: 0 < allocated <= reserved <= card ({m and m.groups()})",
                  m and 0 < int(m.group(2)) <= int(m.group(1)) <= total)


def test_queued():
    print("\n[6] Queued runs: the all-runs estimate")
    seen = {}

    def tiny(P, R, slot_bytes, device, *, batch_bytes):
        seen.update(P=P, R=R, slot=slot_bytes, batch=batch_bytes)
        return A.__dict__["_orig_choose"](P, R, slot_bytes, device, batch_bytes=batch_bytes,
                                          memory_fraction=1.0,
                                          total_memory=BASE_BYTES + batch_bytes + 3 * slot_bytes)
    A._orig_choose = A.choose_width
    try:
        text, _ = live_eval("cpu", patch_width=tiny)
    finally:
        del A._orig_choose
    st = stats_of(text)
    if not check("one stats line", len(st) == 1):
        return
    s = st[0]
    want = int(round((BASE_BYTES + seen["batch"] + seen["P"] * seen["R"] * seen["slot"]) / 2**20))
    check(f"width capped at 3 of {seen['P'] * seen['R']}", s["width"] == 3
          and s["all_runs"] == seen["P"] * seen["R"])
    check(f"all runs need ~{s['all_runs_need_mib']} MiB == BASE + genomes + P*R*slot ({want})",
          s["all_runs_need_mib"] == want)
    m = RS.memory_summary(st)
    check("summary: runs_queued yes, estimate filled in",
          m["runs_queued"] == "yes" and m["vram_needed_all_runs_est_mib"] == want)
    text, _ = live_eval("cpu")
    m = RS.memory_summary(stats_of(text))
    check("unpatched: all fit -> runs_queued no, estimate empty",
          m["runs_queued"] == "no" and m["vram_needed_all_runs_est_mib"] == "")
    a = dict(st[0])
    b = dict(st[0], width=a["all_runs"], all_runs_need_mib=a["all_runs_need_mib"] + 100)
    m = RS.memory_summary([b, a])
    check("queued in only one generation still counts; the estimate is the highest",
          m["runs_queued"] == "yes" and m["vram_needed_all_runs_est_mib"] == a["all_runs_need_mib"] + 100)


def test_sampler():
    print("\n[7] nvidia-smi sampler")
    import shutil
    with RS.GpuSampler(interval=0.05) as s:
        pass
    check("the thread stops when the block ends", not s._thread.is_alive())
    if shutil.which("nvidia-smi") is None:
        check(f"no nvidia-smi: explicit 'unavailable' ({s.peak})",
              s.peak.startswith(RS.UNAVAILABLE) and "nvidia-smi not found" in s.peak)
    else:
        check(f"nvidia-smi: a number per GPU ({s.peak})",
              all(x.isdigit() for x in s.peak.split("|")))
    try:
        with RS.GpuSampler(interval=0.05) as s2:
            raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    check("... also when the block is left by Ctrl-C", not s2._thread.is_alive())

    # A stub nvidia-smi: an older driver (no clocks_event_reasons field), values that
    # change between samples, and one "[N/A]".
    import os
    import time
    d = Path(tempfile.mkdtemp(prefix="fake_smi_"))
    stub = d / "nvidia-smi"
    stub.write_text("""#!/usr/bin/env python3
import sys, pathlib
q = [a for a in sys.argv if a.startswith("--query-gpu=")][0].split("=", 1)[1].split(",")
c = pathlib.Path(__file__).with_name("count"); n = int(c.read_text()) if c.exists() else 0
if "clocks_event_reasons.active" in q:
    print("Field clocks_event_reasons.active is not a valid field to query."); sys.exit(2)
if len(q) > 1:
    c.write_text(str(n + 1))
hot = n >= 2
vals = {"memory.used": "1200" if hot else "1000", "clocks.sm": "1200" if hot else "1500",
        "temperature.gpu": "83" if hot else "70", "power.draw": "[N/A]" if hot else "150.4",
        "clocks_throttle_reasons.active": "0x0000000000000024" if hot else "0x0000000000000000"}
print(", ".join(vals[f] for f in q))
""")
    stub.chmod(0o755)
    old_path = os.environ["PATH"]
    os.environ["PATH"] = f"{d}{os.pathsep}{old_path}"
    try:
        with RS.GpuSampler(interval=0.05) as s3:
            time.sleep(1.0)
    finally:
        os.environ["PATH"] = old_path
    cols = {k: s3.column(k) for k in ("sm_clock_mhz", "temp_c", "power_w", "throttle")}
    check(f"stub nvidia-smi: the old throttle field name is used ({s3.fields.get('throttle')})",
          s3.fields.get("throttle") == "clocks_throttle_reasons.active")
    check(f"peak memory, clock range, peak temperature, power without the [N/A] samples "
          f"({s3.peak}, {cols})",
          s3.peak == "1200" and cols["sm_clock_mhz"] == "1200-1500" and cols["temp_c"] == "83"
          and cols["power_w"] == "150")
    check(f"throttle reasons decoded and combined over samples ({cols['throttle']})",
          cols["throttle"] == "sw_power_cap+sw_thermal_slowdown")
    check("without nvidia-smi every new column says unavailable",
          all(s.column(k).startswith(RS.UNAVAILABLE) for k in ("sm_clock_mhz", "throttle")))


def main():
    with tempfile.TemporaryDirectory(prefix="test_sweeps_") as tmp:
        test_expansion()
        test_errors(tmp)
        test_identity(tmp)
        test_stats_line()
        test_queued()
        test_sampler()
    n_fail = sum(1 for _, ok in _RESULTS if not ok)
    if n_fail:
        print(f"\nFAILED -- {n_fail} of {len(_RESULTS)} checks failed")
        return 1
    print(f"\nOK -- {len(_RESULTS)}/{len(_RESULTS)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
