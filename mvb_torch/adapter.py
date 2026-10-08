"""Drop-in replacement for `mvb.simulation_API.eval_generation`, backed by the batched
evaluator (plan_evotorch.md Step 6, option A).

`simulate/run_ea.py` calls `eval_generation` in exactly two places and, with tracking
off, uses only three fields of its result per genome. So the batched evaluator can take
over those two calls while selection, mutation and every HDF5 write stay the scalar
code. `make_tensor_evaluator` returns a function with `eval_generation`'s signature and
return structure; `run_ea` does not know which one answered.

Randomness follows the config knob that already exists: with
`experiment.predrawn_randomness.enabled` the evaluator runs in pre-drawn mode on
CPU/float64 with the bit-exact `sequential` contraction; otherwise it runs live with
`einsum` (plan 5.4). There is deliberately no second switch for this.

Tracking (plan Step 7): with per-run tracking on, the evaluator writes the same HDF5
file `eval_generation` writes, through the scalar's own `simulate/hdf5_utils.py`
functions, from the main process after the generation (no lock). The arrays come from
`mvb_torch.tracking`. `run_ea` never tracks (it switches tracking off); `run_batch`
does.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import h5py
import numpy as np
import torch

from mvb.simulation_helper_functions import make_experiment_dir
from simulate.hdf5_utils import (
    create_hdf5_file,
    save_eta_to_hdf5,
    save_genome_properties_to_hdf5,
    save_heatmaps_to_hdf5,
    save_modulation_to_hdf5,
    save_per_tick_to_hdf5,
    save_tonic_activations_to_hdf5,
    save_variant_summary_to_hdf5,
    save_wiring_to_hdf5,
)

from . import philox
from . import tracking as trk
from .decision import build_output_spec
from .generation import (SeedSet, SimConfig, choose_width, configure_device, draw_seeds,
                         estimate_slot_bytes, eval_generation_batch, live_contraction)
from .genome_codec import encode_genomes

# Printed once per generation. tests/test_ea_tensor.py looks for it to prove the tensor
# path actually ran -- a toggle that silently did nothing would otherwise make the
# scalar-vs-tensor comparison pass by comparing the scalar EA with itself.
MARKER = "[tensor-evaluator]"

_REQUIRED = ("backend", "device", "dtype")
# Keys that existed once and were removed: refused with an explanation, never ignored.
_REMOVED = {
    "slots_per_genome": "slots_per_genome was removed (plan_evotorch.md Step 8, R3): the "
                        "width is now chosen automatically from device memory -- every run "
                        "at once unless memory caps it. Delete the key.",
}
# Live mode only: the Philox round count (plan Step 8, R1.1). Required there, refused in
# pre-drawn mode, where it would have no effect.
_LIVE_ONLY = ("philox_rounds", "compile")
_DTYPES = {"float32": torch.float32, "float64": torch.float64}

def _predrawn_enabled(randomness_cfg: Optional[Dict[str, Any]]) -> bool:
    return randomness_cfg is not None and bool(randomness_cfg.get("enabled", False))


def validate_evaluator_cfg(evaluator_cfg: Dict[str, Any],
                           randomness_cfg: Optional[Dict[str, Any]]):
    """Check the `experiment.evaluator` block. No defaults: every key is required.
    Returns (device, dtype, philox_rounds or None in pre-drawn mode)."""
    for key, why in _REMOVED.items():
        if key in evaluator_cfg:
            raise KeyError(f"[ERROR] experiment.evaluator.{why}")
    predrawn = _predrawn_enabled(randomness_cfg)
    required = _REQUIRED if predrawn else _REQUIRED + _LIVE_ONLY
    missing = [k for k in required if k not in evaluator_cfg]
    if missing:
        raise KeyError(f"[ERROR] experiment.evaluator is missing {missing}")
    if predrawn and any(k in evaluator_cfg for k in _LIVE_ONLY):
        bad = [k for k in _LIVE_ONLY if k in evaluator_cfg]
        raise KeyError(
            f"[ERROR] experiment.evaluator {bad} have no effect with predrawn_randomness "
            f"enabled (pre-drawn bundles are not Philox, and the bit-exact reference path "
            f"is never compiled); remove them."
        )
    extra = [k for k in evaluator_cfg if k not in _REQUIRED + _LIVE_ONLY]
    if extra:
        raise KeyError(
            f"[ERROR] experiment.evaluator has unknown keys {extra}. Randomness is not "
            f"set here: it follows experiment.predrawn_randomness."
        )
    if evaluator_cfg["backend"] != "tensor":
        raise ValueError("validate_evaluator_cfg is for backend: tensor")

    dtype_name = evaluator_cfg["dtype"]
    if dtype_name not in _DTYPES:
        raise ValueError(f"[ERROR] evaluator.dtype must be one of {sorted(_DTYPES)}")
    device = torch.device(evaluator_cfg["device"])
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("[ERROR] evaluator.device is mps but MPS is not available")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("[ERROR] evaluator.device is cuda but CUDA is not available")
    if device.type not in ("cpu", "mps", "cuda"):
        raise ValueError(f"[ERROR] unsupported evaluator.device {device}")
    if device.type == "mps" and dtype_name == "float64":
        raise ValueError("[ERROR] MPS has no float64 (reproducibility.md 3.1); use "
                         "dtype: float32 on mps, or device: cpu for float64")


    if predrawn:
        if device.type != "cpu" or dtype_name != "float64":
            raise ValueError(
                "[ERROR] predrawn_randomness with the tensor evaluator requires "
                "device: cpu and dtype: float64 -- pre-drawn mode is the bit-exact "
                "reference path (plan 5.4, reproducibility.md 3)."
            )
        rounds, compile_ = None, False
    else:
        rounds = philox.check_rounds(evaluator_cfg["philox_rounds"])
        compile_ = evaluator_cfg["compile"]
        if not isinstance(compile_, bool):
            raise ValueError(f"[ERROR] evaluator.compile must be true or false, got "
                             f"{compile_!r}")
    return device, _DTYPES[dtype_name], rounds, compile_


def make_tensor_evaluator(evaluator_cfg: Dict[str, Any], cfg: Dict[str, Any],
                          randomness_cfg: Optional[Dict[str, Any]]):
    """Return a function with `eval_generation`'s signature, backed by the tensor
    evaluator. Validates the whole configuration up front, so a bad combination fails
    before any simulation runs.

    Live mode needs no generator of its own: its random numbers are computed from each
    run's `noise_seed` / `decision_seed` (`mvb_torch.philox`), the per-run seeds this
    evaluator draws from run_ea's streams and returns -- and run_ea stores -- exactly as
    `eval_generation` does. So a stored run can be replayed alone (plan Step 8, R1)."""
    device, dtype, rounds, compile_ = validate_evaluator_cfg(evaluator_cfg, randomness_cfg)
    configure_device(device)      # CUDA: TF32 off, reproducible cuBLAS -- before any work
    predrawn = _predrawn_enabled(randomness_cfg)
    spec = build_output_spec(cfg["brain"])

    mode = "predrawn" if predrawn else "live"
    contraction = "sequential" if predrawn else live_contraction(device)
    rng = "predrawn-bundles" if predrawn else f"philox4x32-{rounds}"
    print(f"{MARKER} backend=tensor mode={mode} device={device} "
          f"dtype={str(dtype).split('.')[-1]} width=auto "
          f"contraction={contraction} rng={rng} compile={'on' if compile_ else 'off'}"
          + (f" tf32=off gpu={torch.cuda.get_device_name(device)}" if device.type == "cuda" else ""))

    def evaluate(genomes, cfg_, EXPERIMENT_FOLDER, SIMULATION_NAME,
                 ENABLE_PER_RUN_TRACKING, ENABLE_PER_TICK_TRACKING,
                 ENABLE_HEAT_MAP_TRACKING, VIZ_ENABLED, VIZ_BRAIN_ENABLED, VIZ_FPS,
                 VIZ_BRAIN_FPS, N_VARIANTS, rng_noise, rng_decision, rng_world,
                 brain_module_name, MAX_TICKS, N_RUNS, grid_width, grid_height,
                 start_pos, worm_speed, worm_energy_capacity, worm_metabolic_rate,
                 worm_movement_cost, sensor_cfg, feeding_cfg, brain_cfg,
                 pre_computed_seeds_dict=None, replay_info=None, switch_phases=None,
                 randomness_cfg=None):
        # --- what the tensor path cannot do: fail, never skip silently ------------
        if VIZ_ENABLED or VIZ_BRAIN_ENABLED:
            raise NotImplementedError("visualisation needs the scalar evaluator")
        if brain_module_name != "plasticity":
            raise ValueError(f"the tensor evaluator implements only the 'plasticity' "
                             f"brain, not {brain_module_name!r}")
        if int(worm_speed) != 1:
            raise ValueError(f"worm speed must be 1 (the scalar model's v1), got {worm_speed}")
        if _predrawn_enabled(randomness_cfg) != predrawn:
            raise ValueError("randomness_cfg changed between setup and evaluation")
        if len(genomes) != N_VARIANTS:
            raise ValueError(f"{len(genomes)} genomes but N_VARIANTS={N_VARIANTS}")
        P, R = N_VARIANTS, N_RUNS
        flags = trk.TrackingFlags.effective(
            ENABLE_PER_RUN_TRACKING, ENABLE_PER_TICK_TRACKING, ENABLE_HEAT_MAP_TRACKING)

        # --- one-time HDF5 initialisation, as eval_generation does it -------------
        hdf5_path = None
        if flags.per_run:
            skip_timestamp = replay_info is not None and replay_info.get('mode') != 'test'
            hdf5_path = make_experiment_dir(EXPERIMENT_FOLDER, SIMULATION_NAME,
                                            skip_timestamp=skip_timestamp)
            print(f"[batch] writing to {hdf5_path}\n")
            create_hdf5_file(hdf5_path, cfg_)
            with h5py.File(hdf5_path, 'a') as f:
                f.attrs['tracking_per_run_enabled'] = int(ENABLE_PER_RUN_TRACKING)
                f.attrs['tracking_per_tick_enabled'] = int(ENABLE_PER_TICK_TRACKING)
                f.attrs['tracking_heatmap_enabled'] = int(ENABLE_HEAT_MAP_TRACKING)

        # --- seeds: from the SAME generators run_ea passes, so the streams advance
        #     across generations exactly as they do for eval_generation (F5.3) ------
        if pre_computed_seeds_dict is not None:
            seeds = SeedSet(
                run_seeds=np.asarray(pre_computed_seeds_dict[0]["run_seeds"]),
                noise_seeds=np.stack([np.asarray(pre_computed_seeds_dict[v]["noise_seeds"])
                                      for v in range(P)]),
                decision_seeds=np.stack([np.asarray(pre_computed_seeds_dict[v]["decision_seeds"])
                                         for v in range(P)]),
            )
        else:
            seeds = draw_seeds(rng_noise, rng_decision, rng_world, P, R)

        if flags.per_run:
            save_genome_properties_to_hdf5(hdf5_path, genomes, replay_info=replay_info)
            with h5py.File(hdf5_path, 'a') as f:
                if 'elite_genomes' in f:
                    f['elite_genomes'].create_dataset('run_seeds', data=seeds.run_seeds)
                else:
                    f.create_dataset('run_seeds', data=seeds.run_seeds)

        sim = SimConfig(
            max_ticks=int(MAX_TICKS), n_runs=int(R),
            height=int(grid_height), width=int(grid_width),
            start_pos=(int(start_pos[0]), int(start_pos[1])),
            energy_capacity=int(worm_energy_capacity),
            metabolic_rate=int(worm_metabolic_rate),
            movement_cost=int(worm_movement_cost),
            active_sensors=tuple(sensor_cfg),
            feeding_cfg=feeding_cfg,
            switch_phases=tuple(switch_phases or ()),
            noise_level=float(brain_cfg["noise_level"]),
        )

        # Width (R3): from the config and the device's total memory only, so it is the
        # same every generation and every run on this machine. Upper bounds (K from
        # n_neurons x max_decision_delay, all n^2 connections tracked) keep it
        # independent of the particular genomes.
        n = int(brain_cfg["n_neurons"])
        slot_bytes = estimate_slot_bytes(
            height=sim.height, width_cells=sim.width, n=n,
            K=int(n * float(brain_cfg["max_decision_delay"])), dtype=dtype, mode=mode,
            max_ticks=sim.max_ticks,
            tracked_connections=(n * n if flags.per_tick else 0 if flags.heat_map else None),
        )
        width, width_reason = choose_width(P, R, slot_bytes, device)

        t0 = time.perf_counter()
        batch = encode_genomes(genomes, brain_cfg, device=device, dtype=dtype)
        tracker = (trk.Tracker(genomes, flags, R, width, sim.active_sensors, device, dtype)
                   if flags.per_run else None)
        res = eval_generation_batch(
            batch, spec, sim, seeds,
            width=width, mode=mode,
            max_brain_ticks=int(randomness_cfg["max_brain_ticks"]) if predrawn else None,
            philox_rounds=rounds, contraction=contraction, compile=compile_,
            tracker=tracker,
        )
        lifespans = res.lifespans.cpu().numpy()
        wall = time.perf_counter() - t0
        print(f"{MARKER} {P} genomes x {R} runs in {wall:.2f}s ({res.iterations} "
              f"iterations, {1000 * wall / res.iterations:.0f} ms/iteration, slot "
              f"utilisation {res.utilisation:.0%}, longest run {int(lifespans.max())} ticks, "
              f"{res.compactions} compactions, width {res.n_slots} -> {res.final_width}; "
              f"width: {width_reason})")

        if flags.per_run:
            t1 = time.perf_counter()
            write_tracking(hdf5_path, genomes, res, seeds, tracker, sim)
            print(f"{MARKER} tracking written in {time.perf_counter() - t1:.2f}s")
            print(f"[batch] Simulation completed. Saved to {hdf5_path.name}")

        # --- the structure run_ea reads (F6.5), with eval_generation's dtypes ----
        all_lifespans = {
            v: {
                "lifespan_vector": lifespans[v].astype("i4"),
                "seeds_noise_all_runs": seeds.noise_seeds[v],
                "seeds_decision_all_runs": seeds.decision_seeds[v],
            }
            for v in range(P)
        }
        return all_lifespans, seeds.run_seeds

    return evaluate


def write_tracking(hdf5_path, genomes, res, seeds: SeedSet, tracker, sim: SimConfig):
    """Write every variant's tracking datasets, in the order and with the functions
    `eval_variant` uses (plan D7)."""
    flags = tracker.flags
    P, R = len(genomes), sim.n_runs
    lifespans = res.lifespans.cpu().numpy()
    eats = res.eats.cpu().numpy()
    distance = res.distance.cpu().numpy()
    final_energy = res.final_energy.cpu().numpy()
    final_w = tracker.final_w[: P * R].cpu().numpy().reshape(P, R, tracker.C)

    per_tick: Dict[int, Dict[int, np.ndarray]] = {v: {} for v in range(P)}
    heat: Dict[int, Dict[int, np.ndarray]] = {v: {} for v in range(P)}
    if tracker.logging:
        for v, r, ints, w in tracker.per_run_rows():
            if flags.per_tick:
                per_tick[v][r] = trk.build_per_tick(ints, w, genomes[v], tracker.conns[v],
                                                    sim.start_pos, sim.energy_capacity)
            if flags.heat_map:
                heat[v][r] = trk.build_heat_map(ints, sim.start_pos, sim.height, sim.width)
        for v in range(P):
            got = len(per_tick[v]) if flags.per_tick else len(heat[v])
            if got != R:
                raise RuntimeError(f"tracking assembled {got} of {R} runs for genome {v}")

    for v in range(P):
        summary = np.zeros(R, dtype=trk.SUMMARY_DTYPE)
        summary['run_id'] = np.arange(R)
        summary['lifetime_ticks'] = lifespans[v]
        summary['foods'] = eats[v]
        summary['distance'] = distance[v]
        summary['final_energy'] = final_energy[v]
        summary['seed_noise'] = seeds.noise_seeds[v]
        summary['seed_decision'] = seeds.decision_seeds[v]

        wiring = trk.wiring_template(genomes[v], R)
        for c in range(len(tracker.conns[v])):
            for r in range(R):
                wiring[c][f'weight_final_run_{r:04d}'] = final_w[v, r, c]

        save_variant_summary_to_hdf5(hdf5_path, v, summary)
        save_wiring_to_hdf5(hdf5_path, v, wiring)
        save_modulation_to_hdf5(hdf5_path, v, trk.modulation_array(genomes[v]))
        save_eta_to_hdf5(hdf5_path, v, trk.genome_eta(genomes[v]))
        save_tonic_activations_to_hdf5(hdf5_path, v, trk.genome_tonic(genomes[v]))
        if flags.per_tick:
            for r in range(R):
                save_per_tick_to_hdf5(hdf5_path, v, r, per_tick[v][r])
        if flags.heat_map:
            for r in range(R):
                save_heatmaps_to_hdf5(hdf5_path, v, r, heat[v][r])
