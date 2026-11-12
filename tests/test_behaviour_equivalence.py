#!/usr/bin/env python3
"""Tests for tests/behaviour_equivalence.py (plan Step 12), on this machine's CPU.

  [1] statistics on synthetic data: one distribution passes; a 5% shift fails (2); a
      different shape at the same mean fails (3)/(4); ties and the cap are handled
  [2] population: deterministic hash, block sizes, block A == run_ea's generation 0,
      one changed weight changes the hash, flatten/unflatten exact (empty lists kept)
  [3] seeds: S2/S3 share S1's worlds only; seed set 1 == what eval_generation records
  [4] end to end on CPU (12 genomes, 8 runs, every arm incl. the original code in a
      temporary worktree): all hard checks hold, --analyse reproduces summary.json
  [5] controls: C1's change reaches only C1; doctored arms make --analyse FAIL
  [6] resume: a mismatching arm is refused, a missing arm is rerun alone
  [7] the original-code runner refuses the wrong code: our mvb on the path, a dirty
      worktree, another commit

Usage
-----
    python -m tests.test_behaviour_equivalence
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tests.behaviour_equivalence as BE  # noqa: E402
from tests.be_common import (EquivalenceError, flatten_genomes, genome_hash,  # noqa: E402
                             read_arm, unflatten_genomes, write_arm)

ROOT = BE.ROOT
ORIG = "8e17e694c7"
_RESULTS = []


def check(name, cond):
    _RESULTS.append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def raises(fn, *exc):
    try:
        fn()
    except exc or (Exception,) as e:  # noqa: BLE001
        return str(e) or type(e).__name__
    return None


SMALL = dict(BE.PARTS["a"], n_runs=8, decision_runs=4, predrawn_runs=2, topk=3, controls=["C1"],
             blocks=dict(kind="lookup", copies=2, chain_lengths=[1, 3], chains_per_length=2,
                         elites=2, mutants_per_elite=2))


# ---------------------------------------------------------------------------
# [1] statistics
# ---------------------------------------------------------------------------

def synth(rng, mu, R, scale=1.0, cap=2000):
    return {"lifespans": np.minimum(rng.geometric(1 / (mu[:, None] * scale), (len(mu), R)), cap),
            "eats": rng.poisson(mu[:, None] / 10, (len(mu), R)),
            "distance": rng.poisson(mu[:, None] / 3, (len(mu), R))}


def test_statistics():
    print("\n[1] Statistics on synthetic data")
    # Large enough that the +-2% margin is decided by the data, not by noise (the CI
    # half-width here is ~0.4% of the mean).
    rng = np.random.default_rng(3)
    P, R = 150, 1500
    mu = rng.uniform(40, 300, P)
    A, F1, F2 = (synth(rng, mu, R) for _ in range(3))
    same = synth(rng, mu, R)
    fl = [BE.measures(A, F, "lifespans", 10, 2000) for F in (F1, F2)]
    m = BE.measures(A, same, "lifespans", 10, 2000)
    c = BE.judge(m, fl, "lifespans")
    check(f"one distribution: every criterion holds ({c})", all(v is True for v in c.values()))
    shifted = synth(rng, mu, R, scale=1.05)
    c = BE.judge(BE.measures(A, shifted, "lifespans", 10, 2000), fl, "lifespans")
    check("a 5% longer mean lifespan fails (2) the margin", c["2_bias"] is False)
    # The same mean, another shape: uniform on [1, 2 mu - 1] instead of geometric.
    shape = {"lifespans": np.stack([rng.integers(1, int(2 * x), R) for x in mu]),
             "eats": same["eats"], "distance": same["distance"]}
    c = BE.judge(BE.measures(A, shape, "lifespans", 10, 2000), fl, "lifespans")
    check("another shape at the same mean fails (3) and (4)",
          c["3_ks_share"] is False and c["4_pooled_D"] is False)
    capped = {k: v.copy() for k, v in same.items()}
    capped["lifespans"][:, :R // 10] = 2000
    mc = BE.measures(A, capped, "lifespans", 10, 2000)
    check(f"capped share counted ({mc['capped_share']:.3f} vs ref {mc['capped_share_ref']:.4f}) "
          f"and a 10-point jump fails (4)", abs(mc["capped_share"] - 0.1) < 0.01
          and BE.judge(mc, fl, "lifespans")["4_capped"] is False)
    tight = [dict(f, equivalent=False) for f in fl]
    check("a floor that misses the margin makes (2) n/a (None), never PASS",
          BE.judge(m, tight, "lifespans")["2_bias"] is None)
    ties = {"lifespans": np.full((P, R), 7), "eats": same["eats"], "distance": same["distance"]}
    mt = BE.measures(ties, ties, "lifespans", 10, 2000)
    check("all-tied identical arrays: bias 0, D 0, KS share 0 (no crash)",
          mt["bias"] == 0 and mt["pooled_D"] == 0 and mt["ks_share"] == 0)
    check("Spearman of a constant is nan, without a warning", np.isnan(BE.spearman(np.ones(5), np.arange(5))))
    a = {"lifespans": np.array([[5, 6]]), "eats": np.array([[1, 1]]), "distance": np.array([[2, 3]])}
    b = {"lifespans": np.array([[5, 9]]), "eats": np.array([[1, 2]]), "distance": np.array([[2, 3]])}
    d = BE.per_run_diff(a, b, 2000)
    check("per-run differences counted (lifespan 1 of 2, eats 1 of 2, distance 0)",
          d["share_runs_lifespans_differ"] == 0.5 and d["share_runs_eats_differ"] == 0.5
          and d["share_runs_distance_differ"] == 0.0)


# ---------------------------------------------------------------------------
# [2] population
# ---------------------------------------------------------------------------

def test_population():
    print("\n[2] Population")
    from simulate.parallel_genomes import GenomePool
    from mvb.genome.generate_genome_lookup_hard import generate_lookup_hard_genome
    from mvb.genome.generate_genome_random import generate_random_genome
    for part, gen in (("a", generate_lookup_hard_genome), ("b", generate_random_genome)):
        sp = BE.PARTS[part]
        cfg = BE.load_cfg(sp["config"])
        g1, labels = BE.build_population(cfg, sp["elites"], sp["blocks"])
        g2, _ = BE.build_population(cfg, sp["elites"], sp["blocks"])
        check(f"part {part}: built twice -> the same hash", genome_hash(g1) == genome_hash(g2))
        counts = {b: sum(lab.startswith(b) for lab in labels) for b in "ABC"}
        want = {"a": {"A": 10, "B": 90, "C": 160}, "b": {"A": 20, "B": 0, "C": 80}}[part]
        check(f"part {part}: blocks {counts}", counts == want and len(g1) == sum(want.values()))
        nA = want["A"]
        seed0 = cfg["experiment"]["random_genome"]["wiring_randomization_seed"] if part == "b" else 0
        ref = GenomePool(1).generate(gen, cfg, [seed0 + v for v in range(nA)])
        check(f"part {part}: block A == run_ea's generation 0 (GenomePool.generate)",
              genome_hash(g1[:nA]) == genome_hash(ref))
        r = unflatten_genomes(flatten_genomes(g1))
        check(f"part {part}: flatten -> unflatten gives the same hash", genome_hash(r) == genome_hash(g1))
    lookup = g = BE.build_population(BE.load_cfg(SMALL["config"]), SMALL["elites"], SMALL["blocks"])[0][0]
    empty = [k for k, v in g.modulation_spec.items() if not v]
    back = unflatten_genomes(flatten_genomes([lookup]))[0]
    check(f"keys with empty modulation lists survive ({empty})",
          empty and list(back.modulation_spec) == list(lookup.modulation_spec))
    w = back.connection_weights.copy()
    w.flat[np.flatnonzero(w)[0]] += np.float32(1e-6)
    back.connection_weights = w
    check("one weight changed by 1e-6 -> another hash", genome_hash([back]) != genome_hash([lookup]))


# ---------------------------------------------------------------------------
# [3] seeds
# ---------------------------------------------------------------------------

def test_seeds():
    print("\n[3] Seeds")
    import mvb.simulation_API as api
    from mvb_torch.generation import make_simulation_rngs
    from tests.be_common import call_eval, find_h5, read_summaries
    P, R = 3, 5
    s = BE.draw_all_seeds(P, R)
    check("S2 and S3 use S1's worlds", all(np.array_equal(s[1]["run_seeds"], s[k]["run_seeds"]) for k in (2, 3)))
    check("... and their own noise / decision seeds",
          all(not np.array_equal(s[1][x], s[k][x]) for k in (2, 3) for x in ("noise_seeds", "decision_seeds")))
    cfg = BE.load_cfg(SMALL["config"])
    cfg["experiment"]["n_runs"] = R
    genomes = BE.build_population(cfg, SMALL["elites"], SMALL["blocks"])[0][:P]
    from tests.be_common import eval_args
    e = eval_args(cfg)
    with tempfile.TemporaryDirectory() as tmp:
        rn, rd, rw = make_simulation_rngs(1)
        api.eval_generation(genomes, cfg, tmp, "seed_check", True, False, False, False, False, 4, 4, P,
                            rn, rd, rw, e["brain_module_name"], e["MAX_TICKS"], R, e["grid_width"],
                            e["grid_height"], e["start_pos"], e["worm_speed"], e["worm_energy_capacity"],
                            e["worm_metabolic_rate"], e["worm_movement_cost"], e["sensor_cfg"],
                            e["initial_feeding_cfg"], e["brain_cfg"], switch_phases=e["switch_phases"])
        rec = read_summaries(find_h5(tmp), P, R)
    check("seed set 1 == the seeds eval_generation draws with simulation seed 1 (run_ea gen 0)",
          all(np.array_equal(rec[k], s[1][k].astype(np.int64)) for k in ("run_seeds", "noise_seeds", "decision_seeds")))
    with tempfile.TemporaryDirectory() as tmp:
        call_eval(api.eval_generation, genomes, cfg, tmp, seeds=s[2], per_tick=False)
        rec2 = read_summaries(find_h5(tmp), P, R)
    check("call_eval passes the given seeds through (recorded == given)",
          all(np.array_equal(rec2[k], s[2][k].astype(np.int64)) for k in ("run_seeds", "noise_seeds", "decision_seeds")))


# ---------------------------------------------------------------------------
# [4]-[7] end to end
# ---------------------------------------------------------------------------

def make_worktrees(base):
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
    wts = {}
    for name, rev in (("orig", ORIG), ("cur", "HEAD")):
        p = base / f"wt_{name}"
        subprocess.run(["git", "-C", str(ROOT), "worktree", "add", "--detach", str(p), rev],
                       check=True, capture_output=True, env=env)
        wts[name] = p
    return wts


def remove_worktrees(wts):
    for p in wts.values():
        subprocess.run(["git", "-C", str(ROOT), "worktree", "remove", "--force", str(p)],
                       capture_output=True)


def test_end_to_end(base, wts):
    print("\n[4] End to end on CPU (every arm, incl. the original code)")
    out = base / "run"
    run = BE.plan_run("a", SMALL, out, wts["orig"], ORIG, wts["cur"], device_override="cpu")
    BE.execute(out)
    verdict = BE.analyse(out)
    import json
    S = json.loads((out / "summary.json").read_text())
    check(f"all {len(run['arms'])} arms written", all((out / "arms" / f"{n}.npz").exists() for n in run["arms"]))
    bad = [k for k, v in S["checks"].items() if not v]
    check(f"every hard check holds ({len(S['checks'])}; failing: {bad})", not bad)
    check("L1: 0 runs and 0 ticks differ", S["levels"]["L1_model"]["decisions"]["ticks_differ"] == 0
          and S["levels"]["L1_model"]["per_run"]["share_runs_lifespans_differ"] == 0)
    check(f"a verdict is produced ({verdict}; meaningless at this size)", verdict in ("PASS", "FAIL", "INCONCLUSIVE"))
    first = (out / "summary.json").read_bytes()
    BE.analyse(out)
    check("--analyse reproduces summary.json byte for byte", (out / "summary.json").read_bytes() == first)
    check("figures written", all((out / f"{n}.png").exists() for n in
                                 ("genome_means", "bland_altman", "lifespan_ecdf", "per_run_difference")))
    m = read_arm(out / "arms" / "O.npz")["meta"]
    check(f"O ran {ORIG} from the worktree", m["code_commit"].startswith(ORIG) and Path(m["code_root"]) == wts["orig"].resolve())
    return out


def doctored(out, base, name, fn):
    d = base / f"doctored_{name}"
    shutil.copytree(out, d)
    a = read_arm(d / "arms" / f"{name}.npz")
    fn(a)
    meta = a.pop("meta")
    write_arm(d / "arms" / f"{name}.npz", results=a, meta=meta)
    return d


def test_controls(out, base):
    print("\n[5] Controls")
    run = __import__("json").loads((out / "run.json").read_text())
    cfg = BE.load_cfg(SMALL["config"])
    c1 = BE.arm_cfg(cfg, SMALL, run["arms"]["C1"])
    t = BE.arm_cfg(cfg, SMALL, run["arms"]["T"])
    check("C1: noise_level doubled in C1's config only",
          c1["brain"]["noise_level"] == 2 * cfg["brain"]["noise_level"]
          and t["brain"]["noise_level"] == cfg["brain"]["noise_level"])
    check("C1's config hash differs from T's; T == Tr == T2's (no patch leaks)",
          run["arms"]["C1"]["config_hash"] != run["arms"]["T"]["config_hash"]
          and run["arms"]["T"]["config_hash"] == run["arms"]["Tr"]["config_hash"] == run["arms"]["T2"]["config_hash"])

    def bump_one(a):
        a["lifespans"][0, 0] += 1
    d = doctored(out, base, "Tp", bump_one)
    check("one pre-drawn tensor lifespan +1 -> exit 1 (L1 not identical)",
          BE.main(["--analyse", str(d)]) == 1)
    d = doctored(out, base, "T", bump_one)
    check("T changed but not Tr -> exit 1 (determinism check)", BE.main(["--analyse", str(d)]) == 1)
    d = doctored(out, base, "O0", bump_one)
    check("O0 differs from S1d -> exit 1 (runner's positive control)", BE.main(["--analyse", str(d)]) == 1)


def test_resume(out, base):
    print("\n[6] Resume")
    d = base / "resume"
    shutil.copytree(out, d)
    a = read_arm(d / "arms" / "S2.npz")
    meta = a.pop("meta")
    meta["config_hash"] = "0" * 64
    write_arm(d / "arms" / "S2.npz", results=a, meta=meta)
    e = raises(lambda: BE.execute(d, resume=True), EquivalenceError)
    check(f"an arm with another config hash is refused ({e})", e and "S2" in e)
    shutil.rmtree(d)
    shutil.copytree(out, d)
    e = raises(lambda: BE.execute(d, resume=False), EquivalenceError)
    check("without --resume, existing arms are refused", e and "--resume" in e)
    before = {p.name: p.stat().st_mtime_ns for p in (d / "arms").glob("*.npz")}
    (d / "arms" / "Tr.npz").unlink()
    BE.execute(d, resume=True)
    after = {p.name: p.stat().st_mtime_ns for p in (d / "arms").glob("*.npz")}
    check("a missing arm (Tr) is rerun, every other arm reused untouched",
          "Tr.npz" in after and all(after[n] == t for n, t in before.items() if n != "Tr.npz"))
    check("the rerun Tr equals T bit for bit",
          BE.results_equal(read_arm(d / "arms" / "T.npz"), read_arm(d / "arms" / "Tr.npz")))


def run_orig(out, wt, commit, env_path):
    inp = out / "arms" / "O" / "arm_input.npz"
    tmp = out.parent / "orig_refusal"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    p = subprocess.run([sys.executable, str(BE.ORIG_RUNNER), "--input", str(inp), "--out", str(tmp / "x.npz"),
                        "--folder", str(tmp / "h5"), "--commit", commit, "--arm", "X"],
                       cwd=str(wt), env=dict(os.environ, PYTHONPATH=str(env_path)),
                       capture_output=True, text=True)
    return p.returncode, (p.stderr + p.stdout)[-400:], (tmp / "x.npz").exists()


def test_orig_refusals(out, wts):
    print("\n[7] The original-code runner refuses the wrong code")
    rc, msg, made = run_orig(out, wts["orig"], ORIG, ROOT)
    check(f"PYTHONPATH at this checkout -> refused ({msg.strip().splitlines()[-1] if msg.strip() else ''})",
          rc != 0 and "not from the worktree" in msg and not made)
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    rc, msg, made = run_orig(out, wts["orig"], head, wts["orig"])
    check("another commit than the worktree's -> refused", rc != 0 and "is at" in msg and not made)
    f = wts["orig"] / "mvb" / "world.py"
    orig_text = f.read_text()
    try:
        f.write_text(orig_text + "\n# local change\n")
        rc, msg, made = run_orig(out, wts["orig"], ORIG, wts["orig"])
        check("a worktree with a local change -> refused", rc != 0 and "local changes" in msg and not made)
    finally:
        f.write_text(orig_text)
    e = raises(lambda: BE.check_worktree_arg(ROOT, "HEAD", "--current-worktree"), EquivalenceError)
    check(f"this checkout itself as a worktree argument -> refused ({e})", e and "separate worktree" in e)
    e = raises(lambda: BE.check_worktree_arg(wts["orig"], "HEAD", "--current-worktree"), EquivalenceError)
    check("a worktree at another commit than asked -> refused", e and "expected" in e)


def main():
    test_statistics()
    test_population()
    test_seeds()
    base = Path(tempfile.mkdtemp(prefix="be_test_"))
    wts = {}
    try:
        wts = make_worktrees(base)
        out = test_end_to_end(base, wts)
        test_controls(out, base)
        test_resume(out, base)
        test_orig_refusals(out, wts)
    finally:
        remove_worktrees(wts)
        shutil.rmtree(base, ignore_errors=True)
    failed = [n for n, ok in _RESULTS if not ok]
    print(f"\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    for n in failed:
        print(f"  FAILED: {n}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
