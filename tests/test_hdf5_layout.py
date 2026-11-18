#!/usr/bin/env python3
"""Tests for the split layout of large tracking records and open_tracking
(plan_evotorch.md Step 33, known_issues.md 1).

  [1] the limits: HDF5's header limit re-measured (per-tick columns, wiring runs); the
      split constants must be below it
  [2] per_tick round trip for 10 / 1,000 / 1,001 / 2,500 weight columns: written with
      save_per_tick_to_hdf5, read with open_tracking (whole, slices, a field, nested
      access, keys) and read_per_tick; equal to the input in dtype and bytes. Up to
      1,000 columns the file is exactly what create_dataset(gzip) wrote before Step 33
  [3] wiring round trip for 300 / 800 / 801 / 2,000 runs, likewise
  [4] open_tracking on files without split records reads exactly what h5py reads
      (the n = 11 reference files: every dataset, every group's keys)
  [5] rewriting a run (split <-> compound) leaves no stale parts
  [6] the real case: n = 50, 2 large-run elites x 2 runs, all tracking, on the scalar
      code and on the tensor code (CPU): both succeed; their records have the same
      fields; the tensor file's records equal the arrays handed to the writer
  [7] negative controls: plain h5py on a split record fails (no silent partial read);
      a rule above the limit fails on the header limit; a split marker without the
      weights fails loudly on reading; weight columns that are not the float32 tail
      are refused on writing

Usage
-----
    python -m tests.test_hdf5_layout
"""

import os
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mvb_torch.tracking import PER_TICK_BASE_DTYPE, WIRING_BASE_DTYPE  # noqa: E402
from simulate import hdf5_utils as H  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
_RESULTS = []


def check(name, cond):
    _RESULTS.append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def raises(fn, exc):
    try:
        fn()
    except exc as e:
        return str(e) or type(e).__name__
    return None


def per_tick_record(C, L=40, seed=0):
    rng = np.random.default_rng(seed)
    dt = np.dtype(list(PER_TICK_BASE_DTYPE) + [(f"{i // 50}_{i % 50}", "f4") for i in range(C)])
    a = np.zeros(L, dt)
    a["tick"] = np.arange(L)
    a["movement"] = rng.choice([b"N", b"E", b"S", b"W"], L)
    a["energy"] = rng.random(L, dtype=np.float32)
    for n in dt.names[len(PER_TICK_BASE_DTYPE):]:
        a[n] = rng.random(L, dtype=np.float32)
    return a


def wiring_record(R, C=30, seed=1):
    rng = np.random.default_rng(seed)
    dt = np.dtype(list(WIRING_BASE_DTYPE) + [(f"weight_final_run_{r:04d}", "f4") for r in range(R)])
    a = np.zeros(C, dt)
    a["src"], a["tgt"] = np.arange(C), np.arange(C)[::-1]
    a["weight_initial"] = rng.random(C, dtype=np.float32)
    for n in dt.names[len(WIRING_BASE_DTYPE):]:
        a[n] = rng.random(C, dtype=np.float32)
    return a


def fits(dt):
    with h5py.File("probe.h5", "w", driver="core", backing_store=False) as f:
        try:
            f.create_dataset("d", data=np.zeros(2, dt), compression="gzip")
            return True
        except ValueError:
            return False


def limit(make_dtype, hi=4000):
    lo = 1
    while lo < hi:
        m = (lo + hi + 1) // 2
        if fits(make_dtype(m)):
            lo = m
        else:
            hi = m - 1
    return lo


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def test_limits():
    print("\n[1] HDF5's header limit, re-measured")
    lp = limit(lambda k: np.dtype(list(PER_TICK_BASE_DTYPE)
                                  + [(f"{i % 50:02d}_{(i // 50) % 50:02d}", "f4") for i in range(k)]))
    lw = limit(lambda k: np.dtype(list(WIRING_BASE_DTYPE) + [(f"weight_final_run_{r:04d}", "f4") for r in range(k)]))
    check(f"per-tick: {lp} columns of 5-character names fit (n <= 50: names up to '49_49'); "
          f"the rule splits above {H.PER_TICK_SPLIT_ABOVE}", H.PER_TICK_SPLIT_ABOVE < lp)
    check(f"wiring: {lw} run columns fit; the rule splits above {H.WIRING_SPLIT_ABOVE}", H.WIRING_SPLIT_ABOVE < lw)


def test_per_tick(tmp):
    print("\n[2] per_tick round trip")
    p = Path(tmp) / "pt.h5"
    ref = Path(tmp) / "pt_ref.h5"
    h5py.File(p, "w").close()
    for C in (10, 1000, 1001, 2500):
        a = per_tick_record(C, seed=C)
        H.save_per_tick_to_hdf5(p, 0, C, a)
        with H.open_tracking(p) as f:
            whole = f[f"variant_0/run_{C}/per_tick"][()]
            nested = f["variant_0"][f"run_{C}"]["per_tick"]
            part = nested[5:12]
            field = nested["energy"]
            wcol = nested[a.dtype.names[-1]]
            keys = list(f[f"variant_0/run_{C}"].keys())
            rp = H.read_per_tick(f[f"variant_0/run_{C}"])
        split = C > H.PER_TICK_SPLIT_ABOVE
        ok = (same(whole, a) and same(part, a[5:12]) and np.array_equal(field, a["energy"])
              and np.array_equal(wcol, a[a.dtype.names[-1]]) and keys == ["per_tick"] and same(rp, a))
        with h5py.File(p, "r") as f:
            g = f[f"variant_0/run_{C}"]
            on_disk = sorted(g.keys())
            ok &= same(H.read_per_tick(g), a)
            if not split:
                with h5py.File(ref, "w") as r:          # how the file looked before Step 33
                    r.create_dataset("per_tick", data=a, compression="gzip")
                    d0 = r["per_tick"]
                    d1 = g["per_tick"]
                    ok &= (d0.chunks, d0.compression, d0.compression_opts) == (d1.chunks, d1.compression,
                                                                               d1.compression_opts)
                    ok &= same(d1[()], a) and "per_tick_layout" not in g.attrs
            else:
                ok &= g.attrs.get("per_tick_layout") == "split"
        check(f"{C} columns: {'split' if split else 'one compound dataset, as before'} on disk {on_disk}; "
              f"open_tracking and read_per_tick return the record exactly", ok)


def test_wiring(tmp):
    print("\n[3] wiring round trip")
    p = Path(tmp) / "w.h5"
    h5py.File(p, "w").close()
    for R in (300, 800, 801, 2000):
        a = wiring_record(R, seed=R)
        H.save_wiring_to_hdf5(p, R, a)
        with H.open_tracking(p) as f:
            whole = f[f"variant_{R}/wiring"][()]
            col = f[f"variant_{R}"]["wiring"]["weight_final_run_0007"]
            rw = H.read_wiring(f[f"variant_{R}"])
        with h5py.File(p, "r") as f:
            on_disk = sorted(f[f"variant_{R}"].keys())
            layout = f[f"variant_{R}"].attrs.get("wiring_layout")
        split = R > H.WIRING_SPLIT_ABOVE
        check(f"{R} runs: {'split' if split else 'one compound dataset'} on disk {on_disk}; read back exactly",
              same(whole, a) and same(rw, a) and np.array_equal(col, a["weight_final_run_0007"])
              and (layout == "split") == split and (("wiring" in on_disk) != split))


def test_reference_files():
    print("\n[4] open_tracking == h5py on files without split records")
    for name in ("test_behaviour_lookup_ref.h5", "test_predrawn_ref.h5"):
        p = ROOT / "tests" / "refs" / name
        if not p.exists():
            check(f"{name} present", False)
            continue
        ds, groups = [], []
        with h5py.File(p, "r") as f:
            f.visititems(lambda n, o: (ds if isinstance(o, h5py.Dataset) else groups).append(n))
            plain = {n: f[n][()] for n in ds}
            gkeys = {n: list(f[n].keys()) for n in groups}
        with H.open_tracking(p) as t:
            ok = all((same(t[n][()], v) if isinstance(v, np.ndarray) else t[n][()] == v) for n, v in plain.items())
            ok &= all(list(t[n].keys()) == k for n, k in gkeys.items())
        n_pt = sum(n.endswith("per_tick") for n in ds)
        check(f"{name}: all {len(ds)} datasets ({n_pt} per_tick) and {len(groups)} groups' keys identical", ok)


def test_rewrite(tmp):
    print("\n[5] Rewriting a run")
    p = Path(tmp) / "rw.h5"
    h5py.File(p, "w").close()
    big, small = per_tick_record(1500), per_tick_record(20)
    H.save_per_tick_to_hdf5(p, 0, 0, big)
    H.save_per_tick_to_hdf5(p, 0, 0, small)
    with h5py.File(p, "r") as f:
        g = f["variant_0/run_0"]
        k1, a1 = sorted(g.keys()), dict(g.attrs)
    H.save_per_tick_to_hdf5(p, 0, 0, big)
    with h5py.File(p, "r") as f:
        k2 = sorted(f["variant_0/run_0"].keys())
    with H.open_tracking(p) as f:
        back = f["variant_0/run_0/per_tick"][()]
    check(f"split -> compound: only {k1}, no layout attribute; compound -> split: {k2}; read back exactly",
          k1 == ["per_tick"] and "per_tick_layout" not in a1
          and k2 == ["per_tick_base", "per_tick_weight_names", "per_tick_weights"] and same(back, big))


def test_real(tmp):
    print("\n[6] n = 50, all tracking, scalar and tensor")
    import tests.tracking_bench as TB
    rd = Path(tmp) / "real"
    (rd / "cases").mkdir(parents=True)
    handed = {}
    real_wpt = H.write_per_tick

    def spy(grp, arr):
        handed[grp.name] = arr.copy()
        return real_wpt(grp, arr)

    case = dict(group="C", population="elites_n50", P=2, R=2, level="all")
    H.write_per_tick = spy                      # the tensor run only
    try:
        res = {"cpu": TB.run_case(rd, "n50_cpu", keep_files=True, case=dict(case, device="cpu"))}
    finally:
        H.write_per_tick = real_wpt
    res["scalar"] = TB.run_case(rd, "n50_scalar", keep_files=True, case=dict(case, device="scalar"))
    recs, names = {}, {}
    for dev in res:
        h5 = list((rd / "runs" / f"n50_{dev}").rglob("*.h5"))
        recs[dev] = {}
        if h5:
            with H.open_tracking(h5[0]) as f:
                for v in range(2):
                    for r in range(2):
                        recs[dev][(v, r)] = f[f"variant_{v}/run_{r}/per_tick"][()]
                names[dev] = list(f["variant_0/run_0"].keys())
    ncols = len(recs["cpu"][(0, 0)].dtype.names) - len(PER_TICK_BASE_DTYPE) if recs["cpu"] else 0
    check(f"both succeed ({res['cpu']['status']}, {res['scalar']['status']}); "
          f"{ncols} connection columns, read through open_tracking",
          all(r["status"] == "ok" for r in res.values()) and ncols > H.PER_TICK_SPLIT_ABOVE)
    check("scalar and tensor records have the same fields, in the same order",
          recs["cpu"] and all(recs["cpu"][k].dtype.names == recs["scalar"][k].dtype.names for k in recs["cpu"]))
    tensor_ok = len(handed) == 4 and all(same(recs["cpu"][(v, r)], handed[f"/variant_{v}/run_{r}"])
                                         for v in range(2) for r in range(2))
    check("the tensor file's records equal the arrays handed to the writer", tensor_ok)


def test_negative(tmp):
    print("\n[7] Negative controls")
    p = Path(tmp) / "neg.h5"
    h5py.File(p, "w").close()
    H.save_per_tick_to_hdf5(p, 0, 0, per_tick_record(1500))
    with h5py.File(p, "r") as f:
        e = raises(lambda: f["variant_0/run_0/per_tick"][()], KeyError)
    check(f"plain h5py on a split record: KeyError, not a partial record ({(e or 'no error')[:50]})", e is not None)

    old = H._SPLIT_ABOVE["per_tick"]
    H._SPLIT_ABOVE["per_tick"] = 1100
    try:
        e = raises(lambda: H.save_per_tick_to_hdf5(p, 0, 1, per_tick_record(1090)), ValueError)
    finally:
        H._SPLIT_ABOVE["per_tick"] = old
    check(f"a rule above the limit (1,100) fails on the header limit ({(e or 'no error')[:45]})",
          e is not None and "header message is too large" in e)

    with h5py.File(p, "a") as f:
        del f["variant_0/run_0/per_tick_weights"]
    with H.open_tracking(p) as f:
        e = raises(lambda: f["variant_0/run_0/per_tick"][()], H.TrackingLayoutError)
    check(f"a split marker without the weights fails loudly ({(e or 'no error')[:60]})",
          e is not None and "per_tick_weights is missing" in e)

    good = per_tick_record(1200)
    names = list(good.dtype.names)
    moved = good[[names[0]] + names[10:12] + names[1:10] + names[12:]]   # weights before base fields
    moved = np.array(moved.tolist(), dtype=[(n, good.dtype[n]) for n in moved.dtype.names])
    e1 = raises(lambda: H.save_per_tick_to_hdf5(p, 0, 2, moved), H.TrackingLayoutError)
    f64 = np.zeros(3, np.dtype(list(PER_TICK_BASE_DTYPE) + [(f"{i}_{i}", "f8") for i in range(1200)]))
    e2 = raises(lambda: H.save_per_tick_to_hdf5(p, 0, 3, f64), H.TrackingLayoutError)
    check(f"weight columns not the tail, or not float32: refused on writing ({(e1 or '-')[:40]}; {(e2 or '-')[:40]})",
          e1 is not None and e2 is not None)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        test_limits()
        test_per_tick(tmp)
        test_wiring(tmp)
        test_reference_files()
        test_rewrite(tmp)
        test_real(tmp)
        test_negative(tmp)
    failed = [n for n, ok in _RESULTS if not ok]
    print(f"\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    for n in failed:
        print(f"  FAILED: {n}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
