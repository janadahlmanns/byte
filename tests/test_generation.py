#!/usr/bin/env python3
"""Tests for mvb_torch/generation.py (plan_evotorch.md Step 5).

The oracle is the real `eval_generation`, called exactly as run_batch / run_ea call it
and with per-run tracking on, so the HDF5 summary also yields the seeds it drew.

Usage
-----
    python -m tests.test_generation            # everything (~1-2 min)
    python -m tests.test_generation --fast     # skip the live-mode statistics
"""

import contextlib
import copy
import glob
import io
import os
import sys
import tempfile

import h5py
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mvb.simulation_API as api  # noqa: E402
from mvb.feeding import seed_food  # noqa: E402
from mvb.genome.generate_genome_lookup_hard import generate_lookup_hard_genome  # noqa: E402
from mvb.genome.generate_genome_mutate_simple import generate_genome_mutate_simple  # noqa: E402
from mvb.genome.generate_genome_random import generate_random_genome  # noqa: E402
from mvb.predrawn import make_bundle  # noqa: E402
from mvb.world import World  # noqa: E402
from mvb_torch.decision import build_output_spec  # noqa: E402
from mvb_torch.generation import (  # noqa: E402
    build_phase_table,
    build_world_table,
    draw_seeds,
    eval_generation_batch,
    make_simulation_rngs,
    reset_slots,
    setup_generation,
    sim_config_from_yaml,
    slot_view,
)
from mvb_torch.genome_codec import encode_genomes  # noqa: E402
from tests.test_brain_tick import make_genomes  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, "configs", "experiments", "test_predrawn.yaml")
MAX_BRAIN_TICKS = 32          # test_predrawn.yaml's predrawn_randomness.max_brain_ticks
SIM_SEED = 1

_RESULTS = []


def check(name, condition):
    _RESULTS.append((name, bool(condition)))
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}")
    return bool(condition)


def load_cfg(n_runs=5, food=None):
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    cfg["experiment"]["n_runs"] = n_runs
    if food is not None:
        cfg["food"] = food
    return cfg


# ============================================================
# Scalar oracle: the real eval_generation
# ============================================================

def scalar_eval(cfg, genomes, *, predrawn, seeds=None, sim_seed=SIM_SEED):
    """Run eval_generation as run_batch does. Returns a dict of (P, R) arrays plus the
    run_seeds it returned and the per-run seeds it recorded."""
    sc = sim_config_from_yaml(cfg)
    rng_noise, rng_decision, rng_world = make_simulation_rngs(sim_seed)
    P = len(genomes)
    pre = None
    if seeds is not None:
        pre = {v: {"run_seeds": seeds.run_seeds,
                   "noise_seeds": seeds.noise_seeds[v],
                   "decision_seeds": seeds.decision_seeds[v]} for v in range(P)}
    rcfg = {"enabled": True, "max_brain_ticks": MAX_BRAIN_TICKS} if predrawn else None
    wm = cfg["worm"]
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        lifespans, run_seeds = api.eval_generation(
            genomes, cfg, tmp, "gen_test",
            True, False, False,            # per-run tracking on: the summary has the seeds
            False, False, 4, 4, P,
            rng_noise, rng_decision, rng_world,
            cfg["worm"]["decisionmaking"]["version"],
            sc.max_ticks, sc.n_runs, sc.width, sc.height, cfg["world"]["start_pos"],
            wm["speed"], wm["energy_capacity"], wm["metabolic_rate"], wm["movement_cost"],
            list(sc.active_sensors), sc.feeding_cfg, cfg["brain"],
            pre_computed_seeds_dict=pre, replay_info=None,
            switch_phases=list(sc.switch_phases), randomness_cfg=rcfg,
        )
        (h5,) = glob.glob(os.path.join(tmp, "*.h5"))
        out = {k: np.zeros((P, sc.n_runs), dtype=np.int64)
               for k in ("lifespans", "eats", "distance", "final_energy",
                         "noise_seeds", "decision_seeds")}
        with h5py.File(h5, "r") as f:
            for v in range(P):
                s = f[f"variant_{v}/summary"][()]
                out["eats"][v] = s["foods"]
                out["distance"][v] = s["distance"]
                out["final_energy"][v] = s["final_energy"]
                out["noise_seeds"][v] = s["seed_noise"]
                out["decision_seeds"][v] = s["seed_decision"]
    for v in range(P):
        # eval_generation returns {variant_id: {"lifespan_vector": ..., ...}}
        out["lifespans"][v] = lifespans[v]["lifespan_vector"]
    out["run_seeds"] = np.asarray(run_seeds)
    return out


def tensor_eval(cfg, genomes, seeds, *, S, mode="predrawn", contraction="sequential"):
    bcfg = cfg["brain"]
    batch = encode_genomes(genomes, bcfg)
    res = eval_generation_batch(
        batch, build_output_spec(bcfg), sim_config_from_yaml(cfg), seeds,
        # R3: the width replaces slots_per_genome; the S sweeps use width = P*S,
        # clamped to P*R (a width beyond every run is refused).
        width=min(len(genomes) * S, len(genomes) * cfg["experiment"]["n_runs"]),
        mode=mode, max_brain_ticks=MAX_BRAIN_TICKS,
        philox_rounds=10 if mode == "live" else None, contraction=contraction,
    )
    return {k: getattr(res, k).cpu().numpy() for k in
            ("lifespans", "eats", "distance", "final_energy")}


def seeds_for(cfg, P, sim_seed=SIM_SEED):
    return draw_seeds(*make_simulation_rngs(sim_seed), P, cfg["experiment"]["n_runs"])


# ============================================================
# 1. Seeds
# ============================================================

def test_seeds(scalar, seeds):
    print("\n[1] Seeds are drawn exactly as eval_generation draws them")
    check("run_seeds identical to what eval_generation returned",
          np.array_equal(seeds.run_seeds, scalar["run_seeds"]))
    check("noise seeds identical, per genome and run (read from the HDF5 summary)",
          np.array_equal(seeds.noise_seeds.astype(np.int64), scalar["noise_seeds"]))
    check("decision seeds identical, per genome and run",
          np.array_equal(seeds.decision_seeds.astype(np.int64), scalar["decision_seeds"]))
    a = np.random.SeedSequence(7).spawn(3)
    b = np.random.SeedSequence(7).spawn(4)
    check("SeedSequence children depend only on index (run_batch spawn(3) == run_ea "
          "spawn(4)[:3])",
          all(np.random.default_rng(x).integers(0, 2**32, 8).tolist()
              == np.random.default_rng(y).integers(0, 2**32, 8).tolist()
              for x, y in zip(a, b)))


# ============================================================
# 2. Worlds
# ============================================================

def test_world_table():
    print("\n[2] World table == production food seeding (both modes)")
    food = [
        {"phase_from": 0, "feeding_paradigm": {"initial": True, "regrow": False},
         "initial_fraction_per_cell": 0.1, "regrow_time": 200},
        {"phase_from": 40, "feeding_paradigm": {"initial": True, "regrow": True},
         "initial_fraction_per_cell": 0.05, "regrow_time": 4},
    ]
    H, W = 51, 51
    run_seeds = np.random.default_rng(3).integers(0, 2**32, 6, dtype=np.uint32)
    phases = build_phase_table(food[0], food[1:])
    table = build_world_table(run_seeds, phases, H, W, "cpu").numpy()

    ok_live = ok_pre = True
    for r, seed in enumerate(run_seeds):
        # Production, live: the real seed_food on a World whose rng is default_rng(seed).
        w = World(W, H, (26, 26), 0)
        w.rng_world_run = np.random.default_rng(seed)
        w.reset_food()
        seed_food(w, food[0])
        ok_live &= np.array_equal(w.food, table[r, 0])
        seed_food(w, food[1])                      # the phase switch's re-seed
        ok_live &= np.array_equal(w.food, table[r, 1])
        # Production, pre-drawn: make_bundle as eval_variant calls it.
        b = make_bundle(run_seed=seed, noise_seed=1, decision_seed=2, max_ticks=3,
                        max_brain_ticks=2, n_neurons=11, grid_shape=(H, W),
                        n_seed_events=1 + len(food) - 1)
        ok_pre &= np.array_equal((b.food_uniform[0] < 0.1).astype(np.int8), table[r, 0])
        ok_pre &= np.array_equal((b.food_uniform[1] < 0.05).astype(np.int8), table[r, 1])
    check("every world grid matches the live path's seed_food, incl. the phase re-seed",
          ok_live)
    check("every world grid matches make_bundle's food_uniform -- one table serves both",
          ok_pre)


# ============================================================
# 3 + 4. Pre-drawn equivalence and chaining invariance
# ============================================================

def compare(a, b, keys=("lifespans", "eats", "distance", "final_energy")):
    return {k: int((a[k] != b[k]).sum()) for k in keys}


def test_predrawn_equivalence(cfg, genomes, scalar, seeds):
    print("\n[3] Pre-drawn: a full generation is bit-exact vs eval_generation")
    t = tensor_eval(cfg, genomes, seeds, S=2)
    diff = compare(scalar, t)
    for k, d in diff.items():
        check(f"{k}: {scalar[k].size} runs, {d} differ", d == 0)
    lt = scalar["lifespans"]
    check(f"lifespans genuinely vary (min {lt.min()}, max {lt.max()}) so refills happen "
          f"at different iterations", lt.min() != lt.max())
    return t


def test_chaining_invariance(cfg, genomes, seeds, reference):
    print("\n[4] Chaining invariance: identical results for every S")
    R = cfg["experiment"]["n_runs"]
    for S in (1, 3, R, R + 2):
        t = tensor_eval(cfg, genomes, seeds, S=S)
        diff = compare(reference, t)
        note = {1: "every run refilled", 3: "does not divide R",
                R: "one run per slot, no refills", R + 2: "clamped to P*R"}[S]
        check(f"S={S} ({note}): {sum(diff.values())} fields differ",
              sum(diff.values()) == 0)


# ============================================================
# 5 + 9. Reset: a dirty slot must come back bit-identical to a clean one
# ============================================================

def test_dirty_refill(cfg, genomes):
    print("\n[9] Dirty refill: reset_slots restores every field")
    bcfg = cfg["brain"]
    batch = encode_genomes(genomes, bcfg)
    sc = sim_config_from_yaml(cfg)
    seeds = seeds_for(cfg, len(genomes))
    kw = dict(width=len(genomes) * 3, mode="predrawn", max_brain_ticks=MAX_BRAIN_TICKS)
    clean, ctx, src_clean = setup_generation(batch, sc, seeds, **kw)
    dirty, _, src_dirty = setup_generation(batch, sc, seeds, **kw)

    g = torch.Generator().manual_seed(0)
    w = dirty.worm
    for name in ("y", "x", "energy", "eats", "distance", "ticks"):
        setattr(w, name, torch.randint(0, 40, w.y.shape, generator=g))
    w.action = torch.randint(0, 5, w.y.shape, generator=g)
    w.alive = torch.zeros_like(w.alive)
    dirty.world.avail.random_(0, 50, generator=g)      # regrow clock world (R4)
    dirty.world.clock = torch.randint(1, 40, dirty.world.clock.shape, generator=g)
    dirty.phase_idx = torch.ones_like(dirty.phase_idx)
    dirty.brain.act.random_(0, 2, generator=g)
    dirty.brain.Wabs.mul_(0.5)
    src_dirty.z.normal_(generator=g)
    src_dirty.u.uniform_(generator=g)

    reset_slots(dirty, torch.ones_like(dirty.phase_idx, dtype=torch.bool), ctx, src_dirty,
                slot_view(batch, dirty.slot_genome))

    pairs = {
        "worm.y": (clean.worm.y, dirty.worm.y), "worm.x": (clean.worm.x, dirty.worm.x),
        "worm.energy": (clean.worm.energy, dirty.worm.energy),
        "worm.eats": (clean.worm.eats, dirty.worm.eats),
        "worm.distance": (clean.worm.distance, dirty.worm.distance),
        "worm.ticks": (clean.worm.ticks, dirty.worm.ticks),
        "worm.alive": (clean.worm.alive, dirty.worm.alive),
        "worm.action": (clean.worm.action, dirty.worm.action),
        "world.avail": (clean.world.avail, dirty.world.avail),
        "world.clock": (clean.world.clock, dirty.world.clock),
        "phase_idx": (clean.phase_idx, dirty.phase_idx),
        "brain.act": (clean.brain.act, dirty.brain.act),
        "brain.Wabs (== Wabs0, F5.1)": (clean.brain.Wabs, dirty.brain.Wabs),
        "predrawn noise buffer": (src_clean.z, src_dirty.z),
        "predrawn decision buffer": (src_clean.u, src_dirty.u),
    }
    bad = [k for k, (a, b) in pairs.items() if not torch.equal(a, b)]
    check(f"all {len(pairs)} per-run fields restored exactly "
          f"({'none' if not bad else ', '.join(bad)} differ)", not bad)
    # Slots are shared by all genomes (Step 8): each must hold ITS genome's Wabs0.
    g = dirty.slot_genome.reshape(-1)
    check("[5] a reset brain is exactly its slot genome's Wabs0 with zero activity",
          torch.equal(dirty.brain.Wabs, batch.Wabs0[g])
          and bool((dirty.brain.act == 0).all())
          and len(set(g.tolist())) > 1)


# ============================================================
# 6. CRN
# ============================================================

def test_crn(cfg, genomes):
    print("\n[6] Common random numbers: every genome faces the same world on run r")
    batch = encode_genomes(genomes, cfg["brain"])
    R = cfg["experiment"]["n_runs"]
    state, _, _ = setup_generation(batch, sim_config_from_yaml(cfg),
                                   seeds_for(cfg, len(genomes)),
                                   width=len(genomes) * R, mode="predrawn",
                                   max_brain_ticks=MAX_BRAIN_TICKS)
    # With S = R the queue's first P*R items fill every slot: slot b holds genome
    # b // R, run b % R (genome-major order, Step 8).
    P = len(genomes)
    check("slot b holds queue item b (genome b // R, run b % R)",
          torch.equal(state.slot_genome.reshape(P, R),
                      torch.arange(P).view(P, 1).expand(P, R))
          and torch.equal(state.run_idx.reshape(P, R),
                          torch.arange(R).view(1, R).expand(P, R)))
    food = state.world.derived_food()                    # clock world (R4)
    f = food.reshape(P, R, *food.shape[-2:])
    check("food[p, r] identical for every genome p, on every run r",
          all(torch.equal(f[p], f[0]) for p in range(P)))
    check("different runs get different worlds",
          not torch.equal(f[0, 0], f[0, 1]))


# ============================================================
# 8. Per-slot phase switches, including a non-seeding phase
# ============================================================

def test_per_slot_phases(genomes):
    print("\n[8] Per-slot phase switches after refills (early switches, 3 phases)")
    food = [
        {"phase_from": 0, "feeding_paradigm": {"initial": True, "regrow": False},
         "initial_fraction_per_cell": 0.1, "regrow_time": 200},
        {"phase_from": 6, "feeding_paradigm": {"initial": True, "regrow": True},
         "initial_fraction_per_cell": 0.5, "regrow_time": 4},
        {"phase_from": 11, "feeding_paradigm": {"initial": False, "regrow": True},
         "initial_fraction_per_cell": 0.0, "regrow_time": 3},
    ]
    cfg = load_cfg(n_runs=4, food=food)
    seeds = seeds_for(cfg, len(genomes))
    scalar = scalar_eval(cfg, genomes, predrawn=True, seeds=seeds)
    ok = True
    for S in (1, 4):
        t = tensor_eval(cfg, genomes, seeds, S=S)
        d = compare(scalar, t)
        ok &= check(f"S={S}: bit-exact vs eval_generation ({sum(d.values())} fields "
                    f"differ)", sum(d.values()) == 0)
    check(f"every run crossed both switches (min lifespan {scalar['lifespans'].min()} "
          f"> 11), so S=1's refills genuinely exercise per-slot clocks",
          scalar["lifespans"].min() > 11)


def test_frozen_regrow(genomes):
    print("\n[8b] Regrow frozen mid-run: regrow:true -> regrow:false WITHOUT reseeding -> "
          "regrow:true")
    # Cells eaten in phase 0 are still waiting to regrow when phase 1 (no reseed, no
    # regrow) freezes every timer; phase 2 lets them continue. In the regrow-clock world
    # (Step 8, R4) a frozen phase must not advance the clock -- otherwise food would come
    # back during the freeze. No other schedule here has a pending cell enter a
    # non-reseeding regrow:false phase (found by an R4 negative control).
    food = [
        {"phase_from": 0, "feeding_paradigm": {"initial": True, "regrow": True},
         "initial_fraction_per_cell": 0.35, "regrow_time": 8},
        {"phase_from": 10, "feeding_paradigm": {"initial": False, "regrow": False},
         "initial_fraction_per_cell": 0.0, "regrow_time": 3},
        {"phase_from": 30, "feeding_paradigm": {"initial": False, "regrow": True},
         "initial_fraction_per_cell": 0.0, "regrow_time": 2},
    ]
    cfg = load_cfg(n_runs=6, food=food)
    seeds = seeds_for(cfg, len(genomes))
    scalar = scalar_eval(cfg, genomes, predrawn=True, seeds=seeds)
    for S in (1, 6):
        d = compare(scalar, tensor_eval(cfg, genomes, seeds, S=S))
        check(f"S={S}: bit-exact vs eval_generation ({sum(d.values())} fields differ)",
              sum(d.values()) == 0)
    check(f"runs reach the freeze and beyond (median lifespan "
          f"{int(np.median(scalar['lifespans']))}, {int((scalar['lifespans'] > 30).sum())} "
          f"runs pass tick 30)", int((scalar["lifespans"] > 30).sum()) > 0)


# ============================================================
# Sensors
# ============================================================

def test_inactive_sensor(genomes):
    print("\n[10] An inactive sensor feeds 0.0, as InputSource.update does")
    cfg = load_cfg(n_runs=3)
    cfg["worm"]["sensors"]["active"] = ["current_field"]
    seeds = seeds_for(cfg, len(genomes))
    scalar = scalar_eval(cfg, genomes, predrawn=True, seeds=seeds)
    t = tensor_eval(cfg, genomes, seeds, S=2)
    d = compare(scalar, t)
    check(f"adjacent_binary off: bit-exact vs eval_generation ({sum(d.values())} "
          f"differ)", sum(d.values()) == 0)


# ============================================================
# 7. Live mode, calibrated against a scalar noise floor
# ============================================================

def genome_pool(bcfg, n_random=10, n_mutant=14):
    cfg = {"brain": bcfg}
    pool = [generate_random_genome(cfg, rng_seed=100 + i) for i in range(n_random)]
    rng = np.random.default_rng(5)
    for i in range(n_mutant):
        g = generate_lookup_hard_genome(cfg, rng_seed=0)
        for _ in range(1 + i % 7):
            g = generate_genome_mutate_simple(g, 0.1, 0.2, rng)
        pool.append(g)
    return pool


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def test_live_statistics(R=40, n_floor=3):
    print("\n[7] Live mode: agrees with the scalar at least as well as the scalar "
          "agrees with itself")
    cfg = load_cfg(n_runs=R)
    genomes = genome_pool(cfg["brain"])
    P = len(genomes)
    base = seeds_for(cfg, P, sim_seed=11)
    A = scalar_eval(cfg, genomes, predrawn=False, seeds=base)["lifespans"]

    floors = []
    for k in range(n_floor):
        other = seeds_for(cfg, P, sim_seed=100 + k)
        s = type(base)(run_seeds=base.run_seeds,            # SAME worlds (paired)
                       noise_seeds=other.noise_seeds, decision_seeds=other.decision_seeds)
        floors.append(scalar_eval(cfg, genomes, predrawn=False, seeds=s)["lifespans"])

    # Live mode is keyed by the per-run seeds in `base` (Step 8, R1): no generator.
    T = tensor_eval(cfg, genomes, base, S=10, mode="live")["lifespans"]

    mA, mT = A.mean(1), T.mean(1)
    floor_rho = [spearman(B.mean(1), mA) for B in floors]
    rho = spearman(mT, mA)
    print(f"      per-genome mean lifespan: range {mA.min():.1f}..{mA.max():.1f}")
    print(f"      Spearman(tensor, scalar) = {rho:.3f}; scalar-vs-scalar floor = "
          f"{', '.join(f'{x:.3f}' for x in floor_rho)}")
    check("rank agreement >= the scalar's own floor minus 0.05 (threshold set before "
          "running)", rho >= min(floor_rho) - 0.05)

    d_floor = np.concatenate([(B - A).ravel() for B in floors])
    se = d_floor.std() / np.sqrt(A.size)
    bias = float((T - A).mean())
    print(f"      mean paired difference (tensor - scalar) = {bias:+.3f} ticks; "
          f"floor SE = {se:.3f}")
    check("no systematic bias: |mean paired difference| < 3 SE of the scalar floor",
          abs(bias) < 3 * se)


def main():
    fast = "--fast" in sys.argv
    print("=" * 70)
    print("mvb_torch/generation.py test suite")
    print("=" * 70)
    cfg = load_cfg(n_runs=5)
    genomes = list(make_genomes(cfg["brain"]).values())
    seeds = seeds_for(cfg, len(genomes))
    scalar = scalar_eval(cfg, genomes, predrawn=True)        # draws its own seeds

    test_seeds(scalar, seeds)
    test_world_table()
    ref = test_predrawn_equivalence(cfg, genomes, scalar, seeds)
    test_chaining_invariance(cfg, genomes, seeds, ref)
    test_crn(cfg, genomes)
    test_per_slot_phases(genomes)
    test_frozen_regrow(genomes)
    test_dirty_refill(cfg, genomes)
    test_inactive_sensor(genomes)
    if not fast:
        test_live_statistics()
    else:
        print("\n[7] SKIPPED (--fast)")

    failed = [n for n, ok in _RESULTS if not ok]
    print("\n" + "=" * 70)
    if failed:
        print(f"FAILED {len(failed)}/{len(_RESULTS)}")
        for n in failed:
            print(f"  - {n}")
        return 1
    print(f"OK -- {len(_RESULTS)}/{len(_RESULTS)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
