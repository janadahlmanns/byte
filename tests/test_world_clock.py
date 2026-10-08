#!/usr/bin/env python3
"""The regrow clock (plan Step 8, R4) against the faithful two-grid world, cell for cell.

The two-grid `WorldState` (food + regrow_timer) reproduces the scalar world by
construction and is proven against it in Steps 4-5. The clock form
(`ClockWorldState`: avail + clock) is a storage change. Here both are driven through
identical random histories -- phase schedules with regrow on/off, regrow times
{0, 1, 2, 3, 5, 40}, switches with initial true/false, eating, refills -- with the
generation loop's exact order (switch/reseed or regrow pass, then eat), and after every
tick every cell must agree: food == (avail <= clock). Sensing and has_food_at must
agree too.

Coverage counters assert that the cases that broke the first `available_from` formula
actually happen: a switch on the very tick a cell is due, eats on regrown cells, eats
in no-regrow phases, refills.

Usage
-----
    python -m tests.test_world_clock
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mvb_torch.world import (  # noqa: E402
    AVAIL_INF,
    ClockWorldState,
    WorldState,
    avail_from_seed,
    consume_at,
    has_food_at,
    regrow_batch,
    sense_batch,
)

_RESULTS = []
KEYS = ("on_food", "food_north", "food_east", "food_south", "food_west")
R_CHOICES = (0, 1, 2, 3, 5, 40)


def check(name, condition):
    _RESULTS.append((name, bool(condition)))
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}")
    return bool(condition)


def random_schedule(rng):
    n = int(rng.integers(1, 5))
    starts = sorted(set([0] + list(rng.integers(2, 45, size=n - 1))))
    phases = []
    for i, start in enumerate(starts):
        phases.append(dict(start=int(start),
                           initial=True if i == 0 else bool(rng.integers(0, 2)),
                           regrow=bool(rng.random() < 0.7),
                           R=int(rng.choice(R_CHOICES)),
                           frac=float(rng.uniform(0.1, 0.6))))
    return phases


def run_trial(rng, B=8, H=7, W=7, T=60, cover=None):
    """One random history; returns the number of mismatching cell-ticks."""
    sched = random_schedule(rng)
    nq = len(sched)
    w2 = WorldState(food=torch.zeros((B, 1, H, W), dtype=torch.int8),
                    regrow_timer=torch.zeros((B, 1, H, W), dtype=torch.int16))
    wc = ClockWorldState(avail=torch.full((B, 1, H, W), AVAIL_INF, dtype=torch.int16),
                         clock=torch.zeros((B, 1), dtype=torch.int64))
    tick = np.zeros(B, dtype=np.int64)
    phase = np.zeros(B, dtype=np.int64)
    eaten_before = torch.zeros((B, 1, H, W), dtype=torch.bool)

    def reseed(mask_slots, frac):
        seed = torch.as_tensor(rng.random((B, 1, H, W)) < frac).to(torch.int8)
        m = torch.as_tensor(mask_slots).view(B, 1, 1, 1)
        w2.food.copy_(torch.where(m, seed, w2.food))
        w2.regrow_timer.copy_(torch.where(m, torch.zeros_like(w2.regrow_timer), w2.regrow_timer))
        wc.avail.copy_(torch.where(m, avail_from_seed(seed), wc.avail))
        eaten_before.copy_(eaten_before & ~m)

    def refill(mask_slots):
        tick[mask_slots] = 0
        phase[mask_slots] = 0
        reseed(mask_slots, sched[0]["frac"])
        mm = torch.as_tensor(mask_slots).view(B, 1)
        wc.clock = torch.where(mm, torch.zeros_like(wc.clock), wc.clock)

    refill(np.ones(B, dtype=bool))
    bad = 0
    for _ in range(T):
        # 1. world.step(): a switch (reseeding if initial) returns early; else regrow.
        world_tick = tick + 1
        switching = np.array([phase[b] + 1 < nq and world_tick[b] == sched[phase[b] + 1]["start"]
                              for b in range(B)])
        if cover is not None and switching.any():
            due = (w2.regrow_timer == 2).view(B, -1).any(1).numpy()    # would regrow now
            cover["switch_on_due_tick"] += int((due & switching).sum())
        phase = phase + switching
        rs = switching & np.array([sched[q]["initial"] for q in phase])
        for q in range(nq):          # a reseed draws its own fraction per phase
            sel = rs & (phase == q)
            if sel.any():
                reseed(sel, sched[q]["frac"])
        regrow_mask = torch.as_tensor(~switching & np.array([sched[q]["regrow"] for q in phase]))
        regrow_batch(w2, regrow_mask.view(B, 1))
        wc.clock = wc.clock + regrow_mask.view(B, 1).to(torch.int64)

        # 2. eat (only on stay, only if food): mostly where there is food.
        has2 = w2.food.view(B, -1) > 0
        y = np.empty(B, dtype=np.int64)
        x = np.empty(B, dtype=np.int64)
        for b in range(B):
            cells = has2[b].nonzero().flatten().numpy()
            c = int(rng.choice(cells)) if len(cells) and rng.random() < 0.6 else int(rng.integers(H * W))
            y[b], x[b] = divmod(c, W)
        yt, xt = torch.as_tensor(y).view(B, 1), torch.as_tensor(x).view(B, 1)
        hf2, hfc = has_food_at(w2, yt, xt), has_food_at(wc, yt, xt)
        bad += int((hf2 != hfc).sum())
        eat = hf2 & torch.as_tensor(rng.random((B, 1)) < 0.7)
        en = torch.as_tensor([sched[q]["regrow"] for q in phase]).view(B, 1)
        rt = torch.as_tensor([sched[q]["R"] for q in phase]).view(B, 1)
        if cover is not None:
            idx = (yt * W + xt).view(B, 1)
            prev = eaten_before.view(B, -1).gather(1, idx)
            cover["eat_on_regrown"] += int((eat & prev).sum())
            cover["eat_no_regrow"] += int((eat & ~en).sum())
            for r in R_CHOICES:
                cover[f"eat_R{r}"] += int((eat & (rt == r) & en).sum())
            eaten_before.view(B, -1).scatter_(1, idx, prev | eat)
        consume_at(w2, yt, xt, eat, en, rt)
        consume_at(wc, yt, xt, eat, en, rt)

        tick = tick + 1
        # 3. compare every cell, and the senses at the worm's cell
        bad += int((w2.food != wc.derived_food()).sum())
        bad += int((sense_batch(w2, yt, xt, KEYS, torch.float64)
                    != sense_batch(wc, yt, xt, KEYS, torch.float64)).sum())
        # 4. random refills: a new run starts in some slots (clock back to 0)
        ref = rng.random(B) < 0.04
        if ref.any():
            if cover is not None:
                cover["refills"] += int(ref.sum())
            refill(ref)
    return bad


def main():
    print("=" * 70)
    print("Regrow clock vs two-grid world (plan Step 8, R4)")
    print("=" * 70)
    rng = np.random.default_rng(2026)
    cover = {k: 0 for k in ("switch_on_due_tick", "eat_on_regrown", "eat_no_regrow", "refills")}
    cover.update({f"eat_R{r}": 0 for r in R_CHOICES})
    trials, bad = 300, 0
    for _ in range(trials):
        bad += run_trial(rng, cover=cover)
    print(f"\n[1] {trials} random histories x 60 ticks x 8 slots x 7x7 cells")
    check(f"every cell, has_food_at and every sense identical after every tick "
          f"({bad} mismatches)", bad == 0)
    print(f"      coverage: {cover}")
    check("covered: switches on the exact tick a cell was due, eats on regrown cells, "
          "eats in no-regrow phases, refills, and every R in {0, 1, 2, 3, 5, 40}",
          all(v > 0 for v in cover.values()))

    print("\n[2] Saturation")
    wc = ClockWorldState(avail=torch.zeros((1, 1, 3, 3), dtype=torch.int16),
                         clock=torch.full((1, 1), 30000, dtype=torch.int64))
    one = torch.zeros((1, 1), dtype=torch.int64)
    consume_at(wc, one, one, torch.ones((1, 1), dtype=torch.bool), True, 5000)
    check("a return beyond AVAIL_INF saturates to AVAIL_INF (never seen within a run)",
          int(wc.avail[0, 0, 0, 0]) == AVAIL_INF)

    failed = [nm for nm, ok in _RESULTS if not ok]
    print("\n" + "=" * 70)
    if failed:
        print(f"FAILED {len(failed)}/{len(_RESULTS)}")
        for nm in failed:
            print(f"  - {nm}")
        return 1
    print(f"OK -- {len(_RESULTS)}/{len(_RESULTS)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
