#!/usr/bin/env python3
"""Tests for mvb_torch/{world,worm,run}.py (plan_evotorch.md Step 4).

The oracle is the real `simulate_run`, driven through `mvb.predrawn` so both sides
consume identical food uniforms, neuron noise and decision uniforms. That is what
Step 0 built the pre-drawn path for.

Usage
-----
    python -m tests.test_world_worm
"""

import copy
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mvb.brains.decisionmaking_plasticity as dp  # noqa: E402
from mvb.feeding import feeding_tick, on_eat, seed_food  # noqa: E402
from mvb.predrawn import PredrawnBundle, PredrawnRandomness  # noqa: E402
from mvb.sensory import perceive  # noqa: E402
from mvb.simulation_API import simulate_run  # noqa: E402
from mvb.world import World  # noqa: E402
from mvb.worm import Worm  # noqa: E402
from mvb_torch.decision import FALLBACK_ACTIONS, build_output_spec  # noqa: E402
from mvb_torch.genome_codec import encode_genomes  # noqa: E402
from mvb_torch.run import simulate_batch  # noqa: E402
from mvb_torch.world import (  # noqa: E402
    feeding_tick_batch,
    init_world,
    regrow_batch,
    seed_food_batch,
    sense_batch,
)
from mvb_torch.worm import NO_ACTION, act_batch, init_worm  # noqa: E402
from tests.test_brain_tick import load_brain_cfg, make_genomes  # noqa: E402

CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "configs", "experiments", "test_predrawn.yaml")

_RESULTS = []


def check(name, condition):
    _RESULTS.append((name, bool(condition)))
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}")
    return bool(condition)


def load_cfg():
    with open(CONFIG) as f:
        return yaml.safe_load(f)


def phases_from(cfg):
    ordered = sorted(cfg["food"], key=lambda p: p["phase_from"])
    return ordered[0], ordered[1:]


# ============================================================
# 1. Seeding and regrow timing
# ============================================================

def test_seeding_and_regrow():
    print("\n[1] Seeding and regrow timing")
    H, W = 6, 7
    rng = np.random.default_rng(0)
    u = rng.random((2, 3, H, W))
    for p in (0.0, 0.1, 0.5, 1.0):
        world = init_world(2, 3, H, W, "cpu")
        seed_food_batch(world, p, torch.tensor(u))
        ref = (u < p).astype(np.int8)
        if not check(f"seed at fraction {p} matches setup_food_initially",
                     np.array_equal(world.food.numpy(), ref)):
            break
    check("seeding zeroes the regrow timers",
          bool((world.regrow_timer == 0).all()))

    # F4.6: food returns regrow_time - 1 ticks after being eaten; R <= 1 never.
    rows = []
    for R in (0, 1, 2, 3, 4, 5):
        w = World(3, 3, (1, 1), 0)
        w.food[:] = 0
        w.food[1, 1] = 1
        cfg = {"feeding_paradigm": {"initial": False, "regrow": True},
               "regrow_time": R, "initial_fraction_per_cell": 0.0}
        w.feeding_cfg = cfg
        w.switch_phases = []
        on_eat(w, cfg, 1, 1)
        scalar_back = None
        for t in range(1, 12):
            w.ticks = t
            feeding_tick(w, cfg)
            if w.food[1, 1] > 0:
                scalar_back = t
                break
        tw = init_world(1, 1, 3, 3, "cpu")
        tw.food[0, 0, 1, 1] = 1
        tw.food[0, 0, 1, 1] = 0
        tw.regrow_timer[0, 0, 1, 1] = R
        tensor_back = None
        for t in range(1, 12):
            regrow_batch(tw)
            if tw.food[0, 0, 1, 1] > 0:
                tensor_back = t
                break
        rows.append((R, scalar_back, tensor_back))
    check("regrow timing matches the scalar for regrow_time 0..5 "
          f"{[(R, s) for R, s, _ in rows]}",
          all(s == t for _, s, t in rows))
    check("regrow_time <= 1 never regrows (F4.6)",
          rows[0][1] is None and rows[1][1] is None)
    check("regrow_time R >= 2 returns after exactly R-1 ticks",
          all(s == R - 1 for R, s, _ in rows[2:]))


# ============================================================
# 2. Sensing, including wrap
# ============================================================

def test_sensing():
    print("\n[2] Sensing vs perceive(), including toroidal wrap")
    H, W = 7, 9
    cfg = load_cfg()
    keys = list(cfg["brain"]["sensory_mapping"].keys())
    rng = np.random.default_rng(5)
    food = (rng.random((H, W)) < 0.4).astype(np.int8)

    positions = [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1),
                 (0, 4), (H - 1, 4), (3, 0), (3, W - 1), (3, 4)]
    world = init_world(1, len(positions), H, W, "cpu")
    world.food[:] = torch.tensor(food)
    ys = torch.tensor([[p[0] for p in positions]])
    xs = torch.tensor([[p[1] for p in positions]])
    got = sense_batch(world, ys, xs, keys, torch.float64)[0].numpy()

    ok = True
    for i, (y, x) in enumerate(positions):
        w = World(W, H, (0, 0), 0)
        w.food = food.copy()
        worm = type("W", (), {})()
        worm.y, worm.x = y, x
        worm.active_sensors = ["current_field", "adjacent_binary"]
        ref = perceive(w, worm)
        ok &= all(float(ref[k]) == got[i, j] for j, k in enumerate(keys))
    check(f"all {len(positions)} positions (edges + corners) match perceive()", ok)
    check("sensed values are exactly 0.0 or 1.0",
          bool(np.all((got == 0.0) | (got == 1.0))))


# ============================================================
# 3. Acting: the F4.1 rule
# ============================================================

def test_acting():
    print("\n[3] Acting -- eating happens ONLY on stay (F4.1)")
    H, W = 5, 5
    world = init_world(1, 2, H, W, "cpu")
    world.food[:, :, 0, 1] = 1
    worm = init_worm(1, 2, (0, 0), 30, "cpu")   # start_pos is (x, y)
    worm.energy[:] = 10
    running = torch.ones(1, 2, dtype=torch.bool)

    worm.action[:] = FALLBACK_ACTIONS.index("east")
    act_batch(world, worm, running, 1, 30, False, 0)
    check("move onto food does NOT eat it",
          int(worm.eats[0, 0]) == 0 and int(world.food[0, 0, 0, 1]) == 1)
    check("move costs movement_cost and increments distance",
          int(worm.energy[0, 0]) == 9 and int(worm.distance[0, 0]) == 1)

    worm.action[:] = FALLBACK_ACTIONS.index("stay")
    act_batch(world, worm, running, 1, 30, False, 0)
    check("stay on food eats it and resets energy to capacity (F4.2)",
          int(worm.eats[0, 0]) == 1 and int(worm.energy[0, 0]) == 30
          and int(world.food[0, 0, 0, 1]) == 0)
    check("stay does not increment distance", int(worm.distance[0, 0]) == 1)

    worm.action[:] = FALLBACK_ACTIONS.index("stay")
    e_before = int(worm.energy[0, 0])
    act_batch(world, worm, running, 1, 30, False, 0)
    check("stay on an empty cell does nothing",
          int(worm.eats[0, 0]) == 1 and int(worm.energy[0, 0]) == e_before)

    # Toroidal wrap off each edge, from an explicit corner each time.
    for name, (sy, sx), (ey, ex) in [
        ("north", (0, 0), (H - 1, 0)),
        ("south", (H - 1, 0), (0, 0)),
        ("west", (0, 0), (0, W - 1)),
        ("east", (0, W - 1), (0, 0)),
    ]:
        worm.y[:] = sy
        worm.x[:] = sx
        worm.action[:] = FALLBACK_ACTIONS.index(name)
        act_batch(world, worm, running, 0, 30, False, 0)
        check(f"wrap {name}: ({sy},{sx}) -> ({ey},{ex})",
              int(worm.y[0, 0]) == ey and int(worm.x[0, 0]) == ex)

    # start_pos is unpacked as (x, y) by Worm.reset -- verify with an asymmetric one.
    w2 = init_worm(1, 1, (3, 7), 30, "cpu")
    check("start_pos (3,7) -> x=3, y=7, matching Worm.reset's (x, y) unpacking",
          int(w2.x[0, 0]) == 3 and int(w2.y[0, 0]) == 7)


# ============================================================
# 4. Full-run lockstep vs simulate_run
# ============================================================

def _bundle_for(T, K, n, H, W, E, rng):
    return PredrawnBundle(
        neuron_noise=rng.normal(0.0, 1.0, size=(T, K, n)),
        decision_uniform=rng.random(T),
        food_uniform=rng.random((E, H, W)),
    )


def scalar_run(genome, cfg, bundle, max_ticks):
    """One full simulate_run on pre-drawn randomness."""
    wcfg, fcfg, bcfg = cfg["world"], cfg["food"], cfg["brain"]
    first, rest = phases_from(cfg)
    H, W = wcfg["grid_height"], wcfg["grid_width"]
    world = World(W, H, tuple(wcfg["start_pos"]), 0)
    world.feeding_cfg = first
    world.switch_phases = [dict(p) for p in rest]
    src = PredrawnRandomness(bundle)
    world.rng_world_run = src
    wm = cfg["worm"]
    worm = Worm(wm["speed"], wm["energy_capacity"], wm["metabolic_rate"],
                wm["movement_cost"], world)
    worm.active_sensors = wm["sensors"]["active"]
    worm.brain = dp
    dp.init_brain(genome, bcfg)
    # bind() before ANY draw: eval_variant binds right after init_brain, and the
    # food seeding below is the first thing that consumes randomness.
    src.bind(worm, dp._brain_state)
    world.reset_food()
    seed_food(world, first)
    worm.reset()
    simulate_run(world, worm, None, src, src, max_ticks, None)
    return world, worm


def test_full_run_lockstep():
    print("\n[4] Full-run lockstep vs simulate_run (pre-drawn randomness)")
    cfg = load_cfg()
    bcfg = cfg["brain"]
    wcfg, wmcfg = cfg["world"], cfg["worm"]
    H, W = wcfg["grid_height"], wcfg["grid_width"]
    n = bcfg["n_neurons"]
    first, rest = phases_from(cfg)
    E = 1 + sum(1 for p in rest if p["feeding_paradigm"].get("initial", False))
    T = int(cfg["experiment"]["max_ticks"])
    spec = build_output_spec(bcfg)

    genomes = make_genomes(bcfg)
    labels = list(genomes)
    gs = [genomes[k] for k in labels]
    batch = encode_genomes(gs, bcfg)
    P = len(gs)
    K = int(batch.max_ticks.max())

    R = 3
    bundles = {}
    rng = np.random.default_rng(99)
    for p in range(P):
        for r in range(R):
            bundles[(p, r)] = _bundle_for(T, K, n, H, W, E, rng)

    noise = torch.tensor(np.stack(
        [np.stack([bundles[(p, r)].neuron_noise for r in range(R)], 0) for p in range(P)],
        0).transpose(2, 3, 0, 1, 4) * bcfg["noise_level"])
    du = torch.tensor(np.stack(
        [np.stack([bundles[(p, r)].decision_uniform for r in range(R)], 0)
         for p in range(P)], 0).transpose(2, 0, 1))
    fu = torch.tensor(np.stack(
        [np.stack([bundles[(p, r)].food_uniform for r in range(R)], 0) for p in range(P)],
        0).transpose(2, 0, 1, 3, 4))

    world_t, worm_t, brain_t, hist = simulate_batch(
        batch, spec, n_runs=R, height=H, width=W,
        start_pos=wcfg["start_pos"],
        energy_capacity=wmcfg["energy_capacity"],
        metabolic_rate=wmcfg["metabolic_rate"],
        movement_cost=wmcfg["movement_cost"],
        feeding_cfg=first, switch_phases=rest, max_ticks=T,
        food_uniform=fu, noise=noise, decision_uniform=du,
        contraction="sequential", record=True,
    )

    bad = {"y": 0, "x": 0, "energy": 0, "eats": 0, "distance": 0,
           "ticks": 0, "alive": 0, "food": 0}
    lifetimes = []
    for p in range(P):
        for r in range(R):
            sw, sworm = scalar_run(gs[p], cfg, bundles[(p, r)], T)
            lifetimes.append(sworm.ticks)
            if sworm.y != int(worm_t.y[p, r]):
                bad["y"] += 1
            if sworm.x != int(worm_t.x[p, r]):
                bad["x"] += 1
            if sworm.energy != int(worm_t.energy[p, r]):
                bad["energy"] += 1
            if sworm.eats != int(worm_t.eats[p, r]):
                bad["eats"] += 1
            if sworm.distance != int(worm_t.distance[p, r]):
                bad["distance"] += 1
            if sworm.ticks != int(worm_t.ticks[p, r]):
                bad["ticks"] += 1
            if bool(sworm.alive) != bool(worm_t.alive[p, r]):
                bad["alive"] += 1
            if not np.array_equal(sw.food, world_t.food[p, r].numpy()):
                bad["food"] += 1

    total = P * R
    for field in ("y", "x", "energy", "eats", "distance", "ticks", "alive"):
        check(f"final {field} matches in all {total} runs ({bad[field]} wrong)",
              bad[field] == 0)
    check(f"final FOOD GRID matches in all {total} runs ({bad['food']} wrong)",
          bad["food"] == 0)

    lt = np.array(lifetimes)
    check(f"lifetimes vary across runs (min {lt.min()}, max {lt.max()}) -- "
          f"divergent termination is exercised", lt.min() != lt.max())
    check("energy is an exact integer in [0, capacity] throughout (F4.2)",
          all(bool(((e >= 0) & (e <= wmcfg["energy_capacity"])).all())
              for e in hist.energy))
    check(f"history has lifetime+1 rows (F4.9): {len(hist.y)} rows for "
          f"max lifetime {lt.max()}", len(hist.y) == int(lt.max()) + 1)


# ============================================================
# 5. Dead agents are frozen
# ============================================================

def test_frozen_after_death():
    print("\n[5] Dead agents freeze")
    cfg = load_cfg()
    bcfg, wcfg, wmcfg = cfg["brain"], cfg["world"], cfg["worm"]
    H, W = wcfg["grid_height"], wcfg["grid_width"]
    first, rest = phases_from(cfg)
    n = bcfg["n_neurons"]
    T = 60
    spec = build_output_spec(bcfg)
    genomes = make_genomes(bcfg)
    gs = list(genomes.values())
    batch = encode_genomes(gs, bcfg)
    P, R = len(gs), 2
    K = int(batch.max_ticks.max())
    E = 1 + sum(1 for p in rest if p["feeding_paradigm"].get("initial", False))
    rng = np.random.default_rng(4)
    noise = torch.tensor(rng.normal(0, bcfg["noise_level"], size=(T, K, P, R, n)))
    du = torch.tensor(rng.random((T, P, R)))
    fu = torch.tensor(rng.random((E, P, R, H, W)))
    _, worm, _, hist = simulate_batch(
        batch, spec, n_runs=R, height=H, width=W, start_pos=wcfg["start_pos"],
        energy_capacity=wmcfg["energy_capacity"],
        metabolic_rate=wmcfg["metabolic_rate"],
        movement_cost=wmcfg["movement_cost"],
        feeding_cfg=first, switch_phases=rest, max_ticks=T,
        food_uniform=fu, noise=noise, decision_uniform=du, record=True)

    Y = torch.stack(hist.y)
    alive = torch.stack(hist.alive)
    died = ~alive[-1]
    check("some agents died before the loop ended", bool(died.any()))
    ok = True
    for p in range(P):
        for r in range(R):
            a = alive[:, p, r]
            if bool(a.all()):
                continue
            d = int((~a).nonzero()[0])          # first row where it is dead
            ok &= bool((Y[d:, p, r] == Y[d, p, r]).all())
            ok &= bool((torch.stack(hist.energy)[d:, p, r]
                        == torch.stack(hist.energy)[d, p, r]).all())
            ok &= bool((torch.stack(hist.food_sum)[d:, p, r]
                        == torch.stack(hist.food_sum)[d, p, r]).all())
    check("position, energy and food grid stop changing after death", ok)
    check("ticks stop advancing after death",
          bool((worm.ticks[died] < T).all()) if bool(died.any()) else True)


# ============================================================
# 6. Phase switch
# ============================================================

def test_phase_switch():
    print("\n[6] Phase switch: re-seeds and skips that tick's regrow (F4.7)")
    H, W = 8, 8
    rng = np.random.default_rng(11)
    first = {"phase_from": 0, "feeding_paradigm": {"initial": True, "regrow": True},
             "initial_fraction_per_cell": 0.3, "regrow_time": 4}
    second = {"phase_from": 5, "feeding_paradigm": {"initial": True, "regrow": True},
              "initial_fraction_per_cell": 0.8, "regrow_time": 4}
    fu = torch.tensor(rng.random((2, 1, 1, H, W)))

    world = init_world(1, 1, H, W, "cpu")
    seed_food_batch(world, first["initial_fraction_per_cell"], fu[0])
    before = world.food.sum().item()
    cur, pending, ev = first, [dict(second)], 1
    sums = []
    for t in range(1, 9):
        switching = bool(pending) and t == pending[0]["phase_from"]
        su = fu[ev] if switching else None
        cur = feeding_tick_batch(world, t, cur, pending, su, None)
        if switching:
            ev += 1
        sums.append(world.food.sum().item())
    after = (fu[1] < 0.8).sum().item()
    check(f"grid is re-seeded at the switch tick (sum {sums[4]} == {after})",
          sums[4] == after)
    check("fraction actually changed (0.3 -> 0.8)", before < after)
    check("the phase list is consumed exactly once", pending == [])

    # Regrow must be skipped on the switch tick: a cell due exactly then must not
    # come back until the following tick.
    w = World(4, 4, (0, 0), 0)
    w.food[:] = 0
    w.food[1, 1] = 1
    cfg_r = {"phase_from": 0, "feeding_paradigm": {"initial": False, "regrow": True},
             "regrow_time": 3, "initial_fraction_per_cell": 0.0}
    w.feeding_cfg = cfg_r
    sw = {"phase_from": 3, "feeding_paradigm": {"initial": False, "regrow": True},
          "regrow_time": 3, "initial_fraction_per_cell": 0.0}
    w.switch_phases = [dict(sw)]
    on_eat(w, cfg_r, 1, 1)                       # eaten at tick 1 -> due at tick 3
    back = None
    for t in range(2, 9):
        w.ticks = t
        feeding_tick(w, w.feeding_cfg)
        if w.food[1, 1] > 0:
            back = t
            break
    check(f"scalar: a switch on the due tick delays regrow by 1 (back at {back}, "
          f"not 3)", back == 4)

    tw = init_world(1, 1, 4, 4, "cpu")
    tw.regrow_timer[0, 0, 1, 1] = 3
    cur2, pend2 = dict(cfg_r), [dict(sw)]
    tback = None
    for t in range(2, 9):
        switching = bool(pend2) and t == pend2[0]["phase_from"]
        cur2 = feeding_tick_batch(tw, t, cur2, pend2, None, None)
        if tw.food[0, 0, 1, 1] > 0:
            tback = t
            break
    check(f"tensor reproduces the delay exactly (back at {tback})", tback == back)


def main():
    print("=" * 70)
    print("mvb_torch world / worm / run test suite")
    print("=" * 70)
    test_seeding_and_regrow()
    test_sensing()
    test_acting()
    test_full_run_lockstep()
    test_frozen_after_death()
    test_phase_switch()

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
