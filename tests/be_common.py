"""Shared pieces of the behavioural-equivalence test (plan Step 12).

Standalone on purpose: it imports only the standard library, numpy and h5py, never
`mvb`, `mvb_torch` or `simulate`. `tests/orig_arm.py` runs with the ORIGINAL code's
worktree on its path and imports this file, so anything imported here from the
repository would silently come from the original code instead.

  eval_args(cfg)          the positional arguments run_ea passes to eval_generation,
                          built from the config exactly as run_ea builds them
  genome_hash(genomes)    SHA-256 over every genome's numbers, in order
  flatten / unflatten     genomes <-> plain arrays (for the original code; no pickles
                          of today's classes)
  read_summaries(h5)      the per-run summary (P, R) arrays eval_generation writes
  write_arm / read_arm    one arm's result file (.npz)
"""

import dataclasses
import glob
import hashlib
import json
import os
import struct
from typing import Any, Dict, List

import h5py
import numpy as np

SUMMARY_FIELDS = (("lifespans", "lifetime_ticks"), ("eats", "foods"),
                  ("distance", "distance"), ("final_energy", "final_energy"))
RESULT_KEYS = tuple(k for k, _ in SUMMARY_FIELDS)
# Per-tick fields compared decision by decision (the connection-weight columns follow).
PER_TICK_BASE = ("tick", "food_sensed_N", "food_sensed_E", "food_sensed_S",
                 "food_sensed_W", "movement", "food_consumed", "energy",
                 "manhattan_dist", "decision_made")
MOVES = (b"stay", b"N", b"E", b"S", b"W")


class EquivalenceError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# eval_generation's arguments, as run_ea builds them
# ---------------------------------------------------------------------------

def eval_args(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """The values run_ea passes to eval_generation, from the config, built line by line
    as in simulate/run_ea.py (identical at 8e17e694c7 and today)."""
    e = cfg["experiment"]
    phases = sorted(cfg["food"], key=lambda p: p["phase_from"])
    return dict(
        brain_module_name=str(cfg["worm"]["decisionmaking"]["version"]),
        MAX_TICKS=e["max_ticks"], N_RUNS=e["n_runs"],
        grid_width=cfg["world"]["grid_width"], grid_height=cfg["world"]["grid_height"],
        start_pos=cfg["world"]["start_pos"], worm_speed=cfg["worm"]["speed"],
        worm_energy_capacity=cfg["worm"]["energy_capacity"],
        worm_metabolic_rate=cfg["worm"]["metabolic_rate"],
        worm_movement_cost=cfg["worm"]["movement_cost"],
        sensor_cfg=cfg["worm"]["sensors"]["active"],
        initial_feeding_cfg=phases[0], switch_phases=phases[1:],
        brain_cfg=cfg["brain"],
    )


def call_eval(fn, genomes, cfg, folder, *, seeds, per_tick, randomness_cfg=None,
              with_randomness_kw=True, per_run=True):
    """Call an eval_generation-compatible function as run_ea does, with per-run tracking
    on unless per_run=False (and per-tick if asked), and the given seeds as
    pre_computed_seeds_dict.
    `seeds` = dict(run_seeds (R,), noise_seeds (P, R), decision_seeds (P, R))."""
    a = eval_args(cfg)
    P = len(genomes)
    pre = {v: {"run_seeds": np.asarray(seeds["run_seeds"]),
               "noise_seeds": np.asarray(seeds["noise_seeds"][v]),
               "decision_seeds": np.asarray(seeds["decision_seeds"][v])} for v in range(P)}
    # The rng_* arguments are unused with pre_computed_seeds_dict (both codes take
    # every seed from the dict); generators that would fail if touched are not
    # available in numpy, so pass fresh ones and check below that the seeds recorded
    # in the HDF5 are the ones given.
    rngs = [np.random.default_rng(0) for _ in range(3)]
    kw = dict(pre_computed_seeds_dict=pre, replay_info=None, switch_phases=a["switch_phases"])
    if per_tick and not per_run:
        raise EquivalenceError("per-tick tracking needs per-run tracking")
    if with_randomness_kw:
        kw["randomness_cfg"] = randomness_cfg
    elif randomness_cfg is not None:
        raise EquivalenceError("this eval_generation has no randomness_cfg")
    out = fn(genomes, cfg, folder, "be_arm",
             bool(per_run), bool(per_tick), False,     # per-run, per-tick as asked, no heat map
             False, False, 4, 4, P,
             *rngs,
             a["brain_module_name"], a["MAX_TICKS"], a["N_RUNS"], a["grid_width"],
             a["grid_height"], a["start_pos"], a["worm_speed"], a["worm_energy_capacity"],
             a["worm_metabolic_rate"], a["worm_movement_cost"], a["sensor_cfg"],
             a["initial_feeding_cfg"], a["brain_cfg"], **kw)
    return out


def find_h5(folder):
    found = glob.glob(os.path.join(folder, "**", "*.h5"), recursive=True)
    if len(found) != 1:
        raise EquivalenceError(f"expected exactly one HDF5 under {folder}, found {found}")
    return found[0]


def read_summaries(h5_path, P, R) -> Dict[str, np.ndarray]:
    """(P, R) arrays of lifespans, eats, distance, final_energy and the per-run seeds,
    from variant_<v>/summary (written by eval_generation and by the tensor adapter)."""
    out = {k: np.zeros((P, R), dtype=np.float64 if k == "final_energy" else np.int64)
           for k in RESULT_KEYS}
    out["noise_seeds"] = np.zeros((P, R), dtype=np.int64)
    out["decision_seeds"] = np.zeros((P, R), dtype=np.int64)
    with h5py.File(h5_path, "r") as f:
        for v in range(P):
            s = f[f"variant_{v}/summary"][()]
            if len(s) != R:
                raise EquivalenceError(f"variant {v}: {len(s)} runs in the summary, expected {R}")
            if not np.array_equal(s["run_id"], np.arange(R)):
                raise EquivalenceError(f"variant {v}: summary rows not in run order")
            for k, col in SUMMARY_FIELDS:
                out[k][v] = s[col]
            out["noise_seeds"][v] = s["seed_noise"]
            out["decision_seeds"][v] = s["seed_decision"]
        out["run_seeds"] = np.asarray(f["run_seeds"][()]).astype(np.int64)
    return out


def check_seeds_recorded(rec, seeds, arm):
    for k in ("run_seeds", "noise_seeds", "decision_seeds"):
        if not np.array_equal(np.asarray(rec[k]).astype(np.int64),
                              np.asarray(seeds[k]).astype(np.int64)):
            raise EquivalenceError(f"arm {arm}: the {k} recorded in the HDF5 are not the "
                                   f"ones passed in")


# ---------------------------------------------------------------------------
# Genomes: hash, flatten, rebuild
# ---------------------------------------------------------------------------

def _f(x) -> bytes:
    return struct.pack("<d", float(x))


def genome_digest(g) -> bytes:
    h = hashlib.sha256()
    for key in ("connection_weights", "tonic_activations"):
        a = np.ascontiguousarray(np.asarray(g[key]))
        h.update(f"{key}:{a.dtype.str}:{a.shape}".encode())
        h.update(a.tobytes())
    h.update(b"eta:" + _f(g["eta"]))
    for (s, t), mods in g["modulation_spec"].items():
        h.update(f"m{int(s)},{int(t)}:{len(mods)}".encode())
        for m, w in mods:
            h.update(f"{int(m)}".encode() + _f(w))
    return h.digest()


def genome_hash(genomes) -> str:
    h = hashlib.sha256()
    for g in genomes:
        h.update(genome_digest(g))
    return h.hexdigest()


def flatten_genomes(genomes) -> Dict[str, np.ndarray]:
    """Plain arrays: weights (P, n, n, 2), tonic (P, n), eta (P,), the modulation keys
    in dict order (genome, src, tgt, count) and their entries (modulator, weight)."""
    keys, mods = [], []
    for v, g in enumerate(genomes):
        for (s, t), lst in g["modulation_spec"].items():
            keys.append((v, int(s), int(t), len(lst)))
            mods += [(int(m), float(w)) for m, w in lst]
    return {
        "weights": np.stack([np.asarray(g["connection_weights"]) for g in genomes]),
        "tonic": np.stack([np.asarray(g["tonic_activations"]) for g in genomes]),
        "eta": np.array([float(g["eta"]) for g in genomes], dtype=np.float64),
        "mod_keys": np.array(keys, dtype=np.int64).reshape(-1, 4),
        "mod_index": np.array([m for m, _ in mods], dtype=np.int64),
        "mod_weight": np.array([w for _, w in mods], dtype=np.float64),
    }


@dataclasses.dataclass
class FlatGenomeParams:
    """`.params` of a rebuilt genome (save_genome_properties_to_hdf5 reads genome 0's)."""
    source: str


class FlatGenome:
    """A genome rebuilt from plain arrays: the four values the scalar code reads, by key
    and by attribute, with Python types as the generators produce them."""

    def __init__(self, connection_weights, modulation_spec, tonic_activations, eta, source):
        self.connection_weights = connection_weights
        self.modulation_spec = modulation_spec
        self.tonic_activations = tonic_activations
        self.eta = eta
        self.params = FlatGenomeParams(source=source)

    def __getitem__(self, key):
        if key in ("connection_weights", "modulation_spec", "tonic_activations", "eta", "params"):
            return getattr(self, key)
        raise KeyError(key)


def unflatten_genomes(flat, source="behaviour_equivalence") -> List[FlatGenome]:
    P = flat["weights"].shape[0]
    specs = [dict() for _ in range(P)]
    i = 0
    for v, s, t, c in flat["mod_keys"].tolist():
        specs[v][(int(s), int(t))] = [(int(flat["mod_index"][j]), float(flat["mod_weight"][j]))
                                      for j in range(i, i + c)]
        i += c
    if i != len(flat["mod_index"]):
        raise EquivalenceError("modulation entries do not match their keys")
    return [FlatGenome(flat["weights"][v].copy(), specs[v], flat["tonic"][v].copy(),
                       float(flat["eta"][v]), source) for v in range(P)]


# ---------------------------------------------------------------------------
# Arm result files
# ---------------------------------------------------------------------------

def write_arm(path, *, results, meta):
    arrays = {k: np.asarray(results[k]) for k in RESULT_KEYS}
    for k in ("run_seeds", "noise_seeds", "decision_seeds"):
        arrays[k] = np.asarray(results[k]).astype(np.int64)
    arrays["meta_json"] = np.frombuffer(json.dumps(meta, sort_keys=True).encode(), dtype=np.uint8)
    tmp = str(path) + ".partial.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)          # a crash never leaves a half-written arm behind


def loaded_files(root):
    """{path relative to root: sha256} of every imported module file under root."""
    import sys
    from pathlib import Path
    root = Path(root).resolve()
    out = {}
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if not f:
            continue
        p = Path(f).resolve()
        # Some extension shims (PySide6's signature_bootstrap) report a __file__ that
        # does not exist; only real files can be compared.
        if root in p.parents and p.suffix == ".py" and p.is_file():
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return dict(sorted(out.items()))


def read_arm(path):
    with np.load(path) as z:
        out = {k: z[k] for k in z.files if k != "meta_json"}
        out["meta"] = json.loads(bytes(z["meta_json"]).decode())
    return out


def read_per_tick(h5_path, P, R, columns=None):
    """{(v, r): structured per-tick array} for every run (fields as the recorder writes
    them). `columns` limits the fields read."""
    out = {}
    with h5py.File(h5_path, "r") as f:
        for v in range(P):
            for r in range(R):
                d = f[f"variant_{v}/run_{r}/per_tick"]
                out[(v, r)] = d.fields(list(columns))[()] if columns else d[()]
    return out
