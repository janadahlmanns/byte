#!/usr/bin/env python3
"""Tests for the tensor code's tracking writer (plan_evotorch.md, Step 32).

  [1] same file: one generation (3 lookup elites x 8 runs, n = 11, CPU, all tracking)
      written by the old per-call functions and by the one-file writer; every dataset
      equal in name, dtype, shape, values, compression, level, shuffle and chunk shape,
      and every attribute equal. Negative controls, each must fail the comparison:
      a dropped heat map, per-tick without gzip, one changed value
  [2] streaming: each variant's per-tick tables are built after the previous variant
      was written (at most one variant in memory); a missing run still fails loudly
  [3] staging: a staged write gives the same file as an unstaged one, the stage folder
      is empty afterwards, the copy times are reported; an error during a staged write
      leaves the destination as it was and no staged or partial file behind
  [4] the staging folder's checks: 'none' -> off; missing folder, a file, a network
      filesystem -> refused; too little space, too little RAM on a tmpfs -> refused
  [5] filesystem detection from a mount table: deepest mount point, nfs4 = network,
      tmpfs = RAM, a mount point name that is only a prefix does not match, escaped
      spaces
  [6] the evaluator config: tracking_stage_dir accepted (and printed), unknown keys
      still refused

Usage
-----
    python -m tests.test_h5_writer
"""

import contextlib
import io
import os
import shutil
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mvb_torch.adapter as adapter  # noqa: E402
import mvb_torch.h5_writer as W  # noqa: E402
import mvb_torch.tracking as trk  # noqa: E402

_RESULTS = []


def check(name, cond):
    _RESULTS.append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def describe(path):
    """Everything about a file that the writer could change: per object its kind and
    attributes; per dataset dtype, shape, values and storage settings."""
    out = {}

    def visit(name, obj):
        attrs = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in obj.attrs.items()}
        if isinstance(obj, h5py.Dataset):
            out[name] = dict(kind="dataset", attrs=attrs, dtype=obj.dtype.descr if obj.dtype.names else str(obj.dtype),
                             shape=obj.shape, compression=obj.compression, opts=obj.compression_opts,
                             shuffle=obj.shuffle, chunks=obj.chunks, fletcher32=obj.fletcher32,
                             value=obj[()])
        else:
            out[name] = dict(kind="group", attrs=attrs)

    with h5py.File(path, "r") as f:
        out["/"] = dict(kind="root", attrs={k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                            for k, v in f.attrs.items()})
        f.visititems(visit)
    return out


def differences(a, b):
    """List of differences between two describe() results (empty = same content)."""
    diff = [f"only in first: {k}" for k in sorted(set(a) - set(b))]
    diff += [f"only in second: {k}" for k in sorted(set(b) - set(a))]
    for k in sorted(set(a) & set(b)):
        x, y = a[k], b[k]
        for key in x:
            if key == "value":
                vx, vy = x[key], y[key]
                same = (vx == vy) if isinstance(vx, bytes) else (
                    vx.dtype == vy.dtype and vx.shape == vy.shape and vx.tobytes() == vy.tobytes())
                if not same:
                    diff.append(f"{k}: values differ")
            elif x[key] != y[key]:
                diff.append(f"{k}: {key} {x[key]!r} != {y[key]!r}")
    return diff


# ---------------------------------------------------------------------------
# One captured generation, rewritten with different writers
# ---------------------------------------------------------------------------

def capture(tmp):
    """Run 3 elites x 8 runs with all tracking on the CPU; return the write_tracking
    arguments and the file as it was before the tracking was written."""
    import tests.tracking_bench as TB
    from mvb_torch.generation import draw_seeds, make_simulation_rngs
    from tests.be_common import call_eval
    genomes, cfg = TB.population("elites_n11", 3)
    cfg["experiment"]["n_runs"], cfg["experiment"]["population_size"] = 8, 3
    ev = dict(cfg["experiment"]["evaluator"], device="cpu")
    ss = draw_seeds(*make_simulation_rngs(1), 3, 8)
    seeds = {k: np.asarray(getattr(ss, k)) for k in ("run_seeds", "noise_seeds", "decision_seeds")}
    got = {}
    real = adapter.write_tracking

    def spy(hdf5_path, *a, **k):
        got["pre"] = Path(tmp) / "pre.h5"
        shutil.copyfile(hdf5_path, got["pre"])
        got["args"] = a
        return real(hdf5_path, *a, **k)

    adapter.write_tracking = spy
    try:
        folder = Path(tmp) / "run"
        folder.mkdir()
        call_eval(adapter.make_tensor_evaluator(ev, cfg, None), genomes, cfg, str(folder), seeds=seeds,
                  per_tick=True, per_run=True, heat_map=True)
    finally:
        adapter.write_tracking = real
    return got


def rewrite(tmp, got, name, **kw):
    p = Path(tmp) / f"{name}.h5"
    shutil.copyfile(got["pre"], p)
    tw = {}
    adapter.write_tracking(p, *got["args"], timings=tw, **kw)
    return p, tw


@contextlib.contextmanager
def patched(obj, attr, value):
    old = getattr(obj, attr)
    setattr(obj, attr, value)
    try:
        yield
    finally:
        setattr(obj, attr, old)


def test_same_file(tmp, got):
    print("\n[1] The one-file writer writes the same file as the per-call functions")
    old, _ = rewrite(tmp, got, "per_call", per_call=True)
    new, tw = rewrite(tmp, got, "one_file")
    a, b = describe(old), describe(new)
    n_ds = sum(1 for v in a.values() if v["kind"] == "dataset")
    d = differences(a, b)
    check(f"all {n_ds} datasets and {len(a) - n_ds} groups / root equal "
          f"(per_tick and heat maps included: {sum('per_tick' in k for k in a)}, {sum('staying' in k for k in a)})"
          + (f"; first difference: {d[0]}" if d else ""),
          not d and any("per_tick" in k for k in a) and any("staying" in k for k in a))
    check(f"timings reported: build_s {tw.get('build_s', -1):.3f}, h5_s {tw.get('h5_s', -1):.3f}, no stage keys",
          tw.get("build_s", 0) > 0 and tw.get("h5_s", 0) > 0 and "stage_in_s" not in tw)

    class DropHeat(W.TrackingFile):
        def heatmap(self, v, r, staying):
            if (v, r) != (1, 3):
                super().heatmap(v, r, staying)

    class NoGzip(W.TrackingFile):
        def per_tick(self, v, r, arr):
            self._put(self._group(f"variant_{v}", f"run_{r}"), "per_tick", arr)

    class OneValue(W.TrackingFile):
        def per_tick(self, v, r, arr):
            if (v, r) == (2, 5):
                arr = arr.copy()
                arr["energy"][-1] += 1
            super().per_tick(v, r, arr)

    for label, cls, expect in (("a dropped heat map", DropHeat, "only in first"),
                               ("per-tick without gzip", NoGzip, "compression"),
                               ("one changed value", OneValue, "values differ")):
        with patched(W, "TrackingFile", cls):
            bad, _ = rewrite(tmp, got, f"neg_{cls.__name__}")
        d = differences(a, describe(bad))
        check(f"negative control, {label}: caught ({d[0] if d else 'NOT caught'})",
              bool(d) and any(expect in x for x in d))


def test_streaming(tmp, got):
    print("\n[2] Streaming: one variant at a time")
    events = []
    real_build = trk.build_per_tick

    def build(ints, w, genome, conns, *a):
        events.append(("build", int(ints[0, trk._GEN]) if hasattr(trk, "_GEN") else None))
        return real_build(ints, w, genome, conns, *a)

    class Rec(W.TrackingFile):
        def per_tick(self, v, r, arr):
            events.append(("write", v))
            super().per_tick(v, r, arr)

    with patched(trk, "build_per_tick", build), patched(W, "TrackingFile", Rec):
        rewrite(tmp, got, "stream")
    builds = [v for e, v in events if e == "build"]
    order_ok = builds and all(v is not None for v in builds)
    if order_ok:
        last_write_of = {}
        first_build_of = {}
        for i, (e, v) in enumerate(events):
            if e == "write":
                last_write_of[v] = i
            elif v not in first_build_of:
                first_build_of[v] = i
        order_ok = all(first_build_of[v] > last_write_of[v - 1] for v in first_build_of if v > 0)
    check(f"variant v's tables are built only after variant v-1 is written ({len(builds)} builds, "
          f"{sum(e == 'write' for e, _ in events)} writes)", order_ok)

    tracker = got["args"][3]
    real_rows = tracker.per_run_rows

    def drop_one():
        for row in real_rows():
            if (row[0], row[1]) != (1, 4):
                yield row

    with patched(tracker, "per_run_rows", drop_one):
        try:
            rewrite(tmp, got, "missing")
            msg = None
        except RuntimeError as e:
            msg = str(e)
    check(f"a missing run fails loudly ({msg})", msg is not None and "7 of 8 runs for genome 1" in msg)


def test_staging(tmp, got):
    print("\n[3] Staging")
    stage = Path(tmp) / "stage"
    stage.mkdir()
    plain, _ = rewrite(tmp, got, "unstaged")
    staged, tw = rewrite(tmp, got, "staged", stage_dir=stage)
    d = differences(describe(plain), describe(staged))
    check(f"staged file == unstaged file ({'same' if not d else d[0]})", not d)
    check(f"stage folder empty afterwards; copy times reported (in {tw.get('stage_in_s', -1):.4f} s, "
          f"out {tw.get('stage_out_s', -1):.4f} s)",
          not list(stage.iterdir()) and "stage_in_s" in tw and "stage_out_s" in tw)

    p = Path(tmp) / "fails.h5"
    shutil.copyfile(got["pre"], p)
    before = p.read_bytes()

    class Boom(W.TrackingFile):
        def heatmap(self, v, r, staying):
            if (v, r) == (2, 0):
                raise OSError("disk full (simulated)")
            super().heatmap(v, r, staying)

    with patched(W, "TrackingFile", Boom):
        try:
            adapter.write_tracking(p, *got["args"], stage_dir=stage)
            err = None
        except OSError as e:
            err = str(e)
    check(f"an error during a staged write is raised ({err}); the destination is unchanged; "
          f"no staged or partial file left",
          err == "disk full (simulated)" and p.read_bytes() == before and not list(stage.iterdir())
          and not list(p.parent.glob("*.staging_partial")))


def test_stage_checks(tmp):
    print("\n[4] The staging folder's checks")
    check("'none' and None -> no staging", W.check_stage_dir("none") is None and W.check_stage_dir(None) is None)
    d = Path(tmp) / "ok_stage"
    d.mkdir()
    check("an existing local folder is accepted", W.check_stage_dir(str(d)) == d.resolve())

    def refused(fn, *a):
        try:
            fn(*a)
            return None
        except (ValueError, FileNotFoundError, PermissionError, RuntimeError) as e:
            return str(e)

    m = refused(W.check_stage_dir, str(Path(tmp) / "missing"))
    check(f"a missing folder is refused ({m[:60] if m else m})", m and "does not exist" in m)
    f = Path(tmp) / "a_file"
    f.write_text("x")
    m = refused(W.check_stage_dir, str(f))
    check("a file is refused", m and "not a folder" in m)
    with patched(W, "filesystem_of", lambda p, mounts=None: {"fstype": "nfs4", "mount": "/proj", "network": True, "ram": False}):
        m = refused(W.check_stage_dir, str(d))
    check(f"a folder on a network filesystem is refused ({m[:70] if m else m}...)", m and "network filesystem" in m)
    m = refused(W.check_stage_space, d, 1 << 60)
    check("more than the free space is refused", m and "GiB free" in m)
    with patched(W, "filesystem_of", lambda p, mounts=None: {"fstype": "tmpfs", "mount": "/tmp", "network": False, "ram": True}), \
            patched(W, "_mem_available_bytes", lambda: 1 << 20):
        m = refused(W.check_stage_space, d, 10 << 20)
        ok_small = refused(W.check_stage_space, d, 1 << 10) is None
    check(f"on a tmpfs: more than the free RAM is refused, less is accepted ({m[:60] if m else m}...)",
          m and "RAM filesystem" in m and ok_small)


def test_filesystem():
    print("\n[5] Filesystem detection")
    m = [("/", "ext4"), ("/proj/ciptmp", "nfs4"), ("/scratch", "tmpfs"), ("/proj/cip data", "nfs")]
    fs = W.filesystem_of("/proj/ciptmp/user/byte/data", mounts=m)
    check("the deepest mount point; nfs4 is a network filesystem",
          fs["mount"] == "/proj/ciptmp" and fs["network"] and not fs["ram"])
    fs = W.filesystem_of("/scratch/x", mounts=m)
    check("tmpfs is a RAM filesystem, not network", fs["ram"] and not fs["network"])
    fs = W.filesystem_of("/proj/ciptmpX/a", mounts=m)
    check("a mount point that is only a name prefix does not match", fs["mount"] == "/")
    lines = io.StringIO("server:/x /proj/cip\\040data nfs rw 0 0\n")
    parsed = [(W._unescape_mount(ln.split()[1]), ln.split()[2]) for ln in lines]
    fs = W.filesystem_of("/proj/cip data/b", mounts=parsed)
    check("an escaped space in /proc/mounts is understood", fs["mount"] == "/proj/cip data" and fs["network"])
    real = W.filesystem_of(".")
    check(f"this machine's working folder: {real['fstype']} at {real['mount']!r}", real["fstype"] != "")


def test_config(tmp):
    print("\n[6] Evaluator config")
    ev = {"backend": "tensor", "device": "cpu", "dtype": "float32", "philox_rounds": 10, "compile": False}
    import tests.tracking_bench as TB
    _, cfg = TB.population("elites_n11", 1)
    d = Path(tmp) / "cfg_stage"
    d.mkdir()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        adapter.make_tensor_evaluator(dict(ev, tracking_stage_dir=str(d)), cfg, None)
        adapter.make_tensor_evaluator(ev, cfg, None)
    out = buf.getvalue()
    check("tracking_stage_dir accepted and printed; absent -> printed as none",
          f"tracking_stage={d.resolve()}" in out and "tracking_stage=none" in out)
    try:
        adapter.validate_evaluator_cfg(dict(ev, tracking_stage=str(d)), None)
        m = None
    except KeyError as e:
        m = str(e)
    check("an unknown key is still refused", m and "unknown keys" in m)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        got = capture(tmp)
        test_same_file(tmp, got)
        test_streaming(tmp, got)
        test_staging(tmp, got)
        test_stage_checks(tmp)
        test_filesystem()
        test_config(tmp)
    failed = [n for n, ok in _RESULTS if not ok]
    print(f"\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    for n in failed:
        print(f"  FAILED: {n}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
