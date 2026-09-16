"""Load a single saved run out of an HDF5 result file.

Two different layouts are written by the two runners, and both are replayable:

  run_ea.py     elite_genomes/elite_<id>/{connection_weights,tonic_activations,
                eta,modulation_spec,seeds_noise,seeds_decision}
                elite_genomes/run_seeds

  run_batch.py  variant_<id>/{wiring,modulation,eta,tonic_activations,summary}
                run_seeds                      (top level)
                variant_<id>/run_<n>/per_tick  (ground truth for verification)

The batch layout stores the genome as a sparse `wiring` table rather than a
dense matrix, so it is rebuilt here into the same dict shape the brain module
expects. Either way the caller gets an identical `RunSource`.
"""

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np


@dataclass
class RunSource:
    """Everything needed to deterministically replay one run."""
    hdf5_path: str
    layout: str                  # "elite" or "variant"
    genome_id: int
    run_id: int
    genome: dict                 # connection_weights, modulation_spec, tonic_activations, eta
    cfg: dict                    # reconstructed world/food/worm/brain config
    world_seed: int
    noise_seed: int
    decision_seed: int
    max_ticks: int
    expected_lifetime: int = -1  # from summary, -1 when unknown
    ground_truth: np.ndarray = None  # per_tick array, None when absent
    available_genomes: list = field(default_factory=list)
    available_runs: int = 0


# ============================================================
# Path resolution
# ============================================================

def resolve_h5_path(name: str, search_root: str = "data") -> Path:
    """Resolve a file argument to an actual .h5 path.

    Accepts a full path, a path without the .h5 suffix, or a bare filename to
    search for under `search_root`.
    """
    candidates = [Path(name), Path(f"{name}.h5")]
    for c in candidates:
        if c.is_file():
            return c

    stem = Path(name).stem
    root = Path(search_root)
    if root.exists():
        matches = sorted(p for p in root.rglob("*.h5") if p.stem == stem)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            listing = "\n  ".join(str(m) for m in matches)
            raise ValueError(f"[ERROR] '{name}' is ambiguous, matches:\n  {listing}")

    available = sorted(str(p) for p in root.rglob("*.h5")) if root.exists() else []
    listing = "\n  ".join(available) if available else "(none found)"
    raise FileNotFoundError(
        f"[ERROR] HDF5 file not found: {name}\nAvailable files under '{search_root}':\n  {listing}"
    )


# ============================================================
# Loading
# ============================================================

def detect_layout(f: h5py.File) -> str:
    """Return 'elite' for run_ea output, 'variant' for run_batch output."""
    if "elite_genomes" in f and any(k.startswith("elite_") for k in f["elite_genomes"]):
        return "elite"
    if any(k.startswith("variant_") for k in f.keys()):
        return "variant"
    raise ValueError(
        "[ERROR] Unrecognised HDF5 layout: found neither 'elite_genomes/elite_*' "
        "(run_ea output) nor 'variant_*' (run_batch output). "
        "Was this file written with per-run tracking enabled?"
    )


def load_run_source(hdf5_path, genome_id: int, run_id: int) -> RunSource:
    """Load one run's genome, seeds and config from a result file.

    Args:
        hdf5_path: Path to the .h5 file
        genome_id: Elite id (EA output) or variant id (batch output)
        run_id: Which run of that genome to replay

    Returns:
        RunSource ready to hand to the replayer.
    """
    from simulate.run_batch import load_and_reconstruct_hdf5_cfg

    hdf5_path = str(hdf5_path)
    cfg = load_and_reconstruct_hdf5_cfg(hdf5_path)

    with h5py.File(hdf5_path, "r") as f:
        layout = detect_layout(f)
        max_ticks = int(f.attrs.get("experiment_max_ticks", 2000))
        n_neurons = int(f.attrs.get("brain_n_neurons", 0))

        if layout == "elite":
            available = sorted(
                int(k.split("_")[1]) for k in f["elite_genomes"] if k.startswith("elite_")
            )
            _require(genome_id, available, "genome")
            grp = f[f"elite_genomes/elite_{genome_id}"]

            genome = {
                "connection_weights": grp["connection_weights"][:],
                "tonic_activations": grp["tonic_activations"][:],
                "eta": _scalar(grp["eta"]),
                "modulation_spec": _modulation_from_rows(
                    grp["modulation_spec"][:] if "modulation_spec" in grp else [],
                    ("source", "target", "modulating_neuron", "modulation_weight"),
                ),
            }

            run_seeds = f["elite_genomes/run_seeds"][:]
            noise_seeds = grp["seeds_noise"][:]
            decision_seeds = grp["seeds_decision"][:]
            n_runs = len(run_seeds)
            _require(run_id, list(range(n_runs)), "run")

            world_seed = int(run_seeds[run_id])
            noise_seed = int(noise_seeds[run_id])
            decision_seed = int(decision_seeds[run_id])
            expected, truth = -1, None

        else:  # variant layout
            available = sorted(
                int(k.split("_")[1]) for k in f.keys() if k.startswith("variant_")
            )
            _require(genome_id, available, "genome")
            grp = f[f"variant_{genome_id}"]

            genome = {
                "connection_weights": _weights_from_wiring(grp["wiring"][:], n_neurons),
                "tonic_activations": grp["tonic_activations"][:],
                "eta": _scalar(grp["eta"]),
                "modulation_spec": _modulation_from_rows(
                    grp["modulation"][:] if "modulation" in grp else [],
                    ("target_src", "target_tgt", "modulator_src", "modulation_weight"),
                ),
            }

            run_seeds = f["run_seeds"][:]
            summary = grp["summary"][:]
            n_runs = len(run_seeds)
            _require(run_id, list(range(n_runs)), "run")

            world_seed = int(run_seeds[run_id])
            row = summary[summary["run_id"] == run_id]
            if len(row) == 0:
                raise ValueError(f"[ERROR] No summary row for run {run_id}")
            noise_seed = int(row[0]["seed_noise"])
            decision_seed = int(row[0]["seed_decision"])
            expected = int(row[0]["lifetime_ticks"])

            truth_key = f"variant_{genome_id}/run_{run_id}/per_tick"
            truth = f[truth_key][:] if truth_key in f else None

    return RunSource(
        hdf5_path=hdf5_path,
        layout=layout,
        genome_id=genome_id,
        run_id=run_id,
        genome=genome,
        cfg=cfg,
        world_seed=world_seed,
        noise_seed=noise_seed,
        decision_seed=decision_seed,
        max_ticks=max_ticks,
        expected_lifetime=expected,
        ground_truth=truth,
        available_genomes=available,
        available_runs=n_runs,
    )


# ============================================================
# Helpers
# ============================================================

def _require(value, available, label):
    if value not in available:
        raise ValueError(
            f"[ERROR] {label} {value} not found. Available {label}s: {available}"
        )


def _scalar(dataset):
    """Read eta, which is stored as a bare scalar in one layout and a 1-element array in the other."""
    value = dataset[()]
    arr = np.atleast_1d(value)
    return float(arr[0])


def _weights_from_wiring(wiring, n_neurons):
    """Rebuild the dense (n, n, 2) weight matrix from the sparse wiring table.

    The batch writer only stores connections whose weight is non-zero, so every
    absent (src, tgt) pair is a genuine zero.
    """
    if n_neurons <= 0:
        raise ValueError("[ERROR] brain_n_neurons missing from HDF5 attrs; cannot rebuild weights")
    weights = np.zeros((n_neurons, n_neurons, 2), dtype=np.float32)
    for row in wiring:
        weights[int(row["src"]), int(row["tgt"]), 0] = float(row["weight_initial"])
        weights[int(row["src"]), int(row["tgt"]), 1] = float(row["reliability"])
    return weights


def _modulation_from_rows(rows, fields):
    """Rebuild modulation_spec as {(src, tgt): [(modulator, weight), ...]}."""
    src_f, tgt_f, mod_f, w_f = fields
    spec = {}
    for row in rows:
        key = (int(row[src_f]), int(row[tgt_f]))
        spec.setdefault(key, []).append((int(row[mod_f]), float(row[w_f])))
    return spec
