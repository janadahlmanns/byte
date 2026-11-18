#!/usr/bin/env python3
"""Where the tracking writer's time goes on this machine (plan_evotorch.md, Step 32.0).

Builds real per-tick tables and heat maps (the n = 11 lookup elites, live, CPU), then
writes N of each six ways and prints milliseconds per dataset:

  per_call   @ dest   simulate/hdf5_utils.py's functions (open/close per dataset): the
                      tensor code before Step 32, and the scalar code
  one_file   @ dest   one open file (Step 32.1)
  no_gzip    @ dest   one open file without compression (separates compression from
                      the metadata and I/O cost; not a format anyone writes)
  per_call   @ local
  one_file   @ local
  staged              one open file at local, then copied to dest (Step 32.3)

dest is the repo's data/ folder (where run_batch writes), local the system's temp folder
($TMPDIR, else /tmp); the filesystem type of each is printed. ~2-5 min on the uni PC.
A measurement tool: no file of the repository changes; the probe's files are deleted.

Usage
-----
    python -m tests.write_probe                     # N = 2000
    python -m tests.write_probe --n 500 --local /var/tmp
"""

import argparse
import datetime
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def make_arrays(n_genomes=4, R=50):
    """Per-tick tables and heat maps of real runs: n_genomes lookup elites x R runs."""
    import numpy as np
    import mvb_torch.adapter as adapter
    import mvb_torch.tracking as trk
    import tests.tracking_bench as TB
    from mvb_torch.generation import draw_seeds, make_simulation_rngs
    from tests.be_common import call_eval
    genomes, cfg = TB.population("elites_n11", n_genomes)
    cfg["experiment"]["n_runs"], cfg["experiment"]["population_size"] = R, n_genomes
    ev = dict(cfg["experiment"]["evaluator"], device="cpu")
    ss = draw_seeds(*make_simulation_rngs(1), n_genomes, R)
    seeds = {k: np.asarray(getattr(ss, k)) for k in ("run_seeds", "noise_seeds", "decision_seeds")}
    got = {}
    real = adapter.write_tracking

    def spy(hdf5_path, genomes_, res, seeds_, tracker, sim, **k):
        got.update(tracker=tracker, sim=sim, genomes=genomes_)

    adapter.write_tracking = spy
    with tempfile.TemporaryDirectory() as tmp:
        try:
            call_eval(adapter.make_tensor_evaluator(ev, cfg, None), genomes, cfg, tmp, seeds=seeds,
                      per_tick=True, per_run=True, heat_map=True)
        finally:
            adapter.write_tracking = real
    tr, sim = got["tracker"], got["sim"]
    ticks, heats = [], []
    for v, r, ints, w in tr.per_run_rows():
        ticks.append(trk.build_per_tick(ints, w, got["genomes"][v], tr.conns[v], sim.start_pos, sim.energy_capacity))
        heats.append(trk.build_heat_map(ints, sim.start_pos, sim.height, sim.width))
    return ticks, heats


def write_all(sink, ticks, heats, N):
    t = time.perf_counter()
    with sink:
        for i in range(N):
            sink.per_tick(i // 300, i % 300, ticks[i % len(ticks)])
        for i in range(N):
            sink.heatmap(i // 300, i % 300, heats[i % len(heats)])
    return time.perf_counter() - t


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=2000, help="datasets of each kind (default 2000)")
    ap.add_argument("--dest", default=str(ROOT / "data"), help="the destination folder (default: data/)")
    ap.add_argument("--local", default=tempfile.gettempdir(), help="a local folder (default: the temp folder)")
    a = ap.parse_args(argv)
    import h5py
    import mvb_torch.h5_writer as W
    from mvb_torch.adapter import _PerCallSink

    class NoGzip(W.TrackingFile):
        def per_tick(self, v, r, arr):
            self._put(self._group(f"variant_{v}", f"run_{r}"), "per_tick", arr)

        def heatmap(self, v, r, staying):
            self._put(self._group(f"variant_{v}", f"run_{r}"), "staying", staying)

    print("[probe] building real per-tick tables and heat maps (4 elites x 50 runs, CPU) ...", flush=True)
    ticks, heats = make_arrays()
    mean_len = sum(len(t) for t in ticks) / len(ticks)
    print(f"[probe] {len(ticks)} runs, mean {mean_len:.0f} ticks, per-tick row {ticks[0].dtype.itemsize} B, "
          f"heat map {heats[0].shape}", flush=True)

    places = {}
    for label, base in (("dest", a.dest), ("local", a.local)):
        d = Path(base) / f"write_probe_{os.getpid()}"
        d.mkdir(parents=True)
        places[label] = d
    results = {}
    try:
        for label, d in places.items():
            fs = W.filesystem_of(d)
            results[f"fs_{label}"] = dict(path=str(d.parent), **fs)
            print(f"[probe] {label}: {d.parent} -> {fs['fstype']} at {fs['mount']}", flush=True)

        def fresh(d, name):
            p = d / f"{name}.h5"
            h5py.File(p, "w").close()
            return p

        runs = [("per_call", "dest", _PerCallSink), ("one_file", "dest", W.TrackingFile),
                ("no_gzip", "dest", NoGzip), ("per_call", "local", _PerCallSink),
                ("one_file", "local", W.TrackingFile)]
        for how, where, sink in runs:
            p = fresh(places[where], how)
            s = write_all(sink(p), ticks, heats, a.n)
            results[f"{how}@{where}"] = dict(total_s=s, ms_per_dataset=1e3 * s / (2 * a.n),
                                             mib=p.stat().st_size / 2 ** 20)
            p.unlink()
            print(f"  {how:9s} @ {where:5s}: {1e3 * s / (2 * a.n):7.3f} ms per dataset ({s:7.2f} s)", flush=True)
        p_dest = fresh(places["dest"], "staged")
        t = time.perf_counter()
        tw = {}
        with W.staged(p_dest, places["local"], tw) as path:
            write_all(W.TrackingFile(path), ticks, heats, a.n)
        s = time.perf_counter() - t
        results["staged"] = dict(total_s=s, ms_per_dataset=1e3 * s / (2 * a.n), **tw,
                                 mib=p_dest.stat().st_size / 2 ** 20)
        print(f"  staged (one_file @ local, copied to dest): {1e3 * s / (2 * a.n):7.3f} ms per dataset ({s:7.2f} s; "
              f"copy back {tw.get('stage_out_s', 0):.2f} s)", flush=True)
    finally:
        for d in places.values():
            shutil.rmtree(d, ignore_errors=True)

    from simulate.run_selected import system_info, torch_probe
    out = ROOT / "data" / "runtime_reports" / f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}_write_probe"
    out.mkdir(parents=True)
    (out / "system.json").write_text(json.dumps(system_info(torch_probe()), indent=1, default=str))
    (out / "probe.json").write_text(json.dumps(dict(n=a.n, mean_ticks=mean_len, results=results), indent=1))
    base = results["per_call@dest"]["ms_per_dataset"]
    print(f"\n[probe] per_call @ dest -> one_file @ dest: {base / results['one_file@dest']['ms_per_dataset']:.1f}x; "
          f"-> staged: {base / results['staged']['ms_per_dataset']:.1f}x")
    print(f"[probe] written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
