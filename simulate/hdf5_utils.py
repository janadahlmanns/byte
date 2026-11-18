"""
HDF5 utilities for saving experiment data.

Provides functions to save data in HDF5 format, mirroring the folder structure:
- Root level: experiment metadata (attributes)
- /variant_XX/ groups: summary, wiring, modulation, heatmaps datasets
- /variant_XX/per_tick_run_XXXX/ groups: per-tick data datasets

Supports parallel writing with multiprocessing.Lock() to synchronize concurrent HDF5 writes.

Large records (plan_evotorch.md Step 33, known_issues.md 1): HDF5 cannot store a compound
type with more than ~1,082 per-tick connection columns or ~854 wiring run columns (one
header message is limited to 64 KiB). Records above PER_TICK_SPLIT_ABOVE /
WIRING_SPLIT_ABOVE columns are written split: the base fields under `<name>_base`, the
weight columns as one float32 matrix `<name>_weights` with their names in
`<name>_weight_names`, and the group attribute `<name>_layout = "split"`. Smaller records
keep the single compound dataset, byte for byte as before. `open_tracking(path)` reads
both layouts with the same code: `f["variant_0/run_3/per_tick"][()]` returns the full
record either way.
"""

import json
import re
import sys
from pathlib import Path
import numpy as np
import h5py
from multiprocessing import Lock

# Split above these many weight columns (measured limits 2026-11-18: 1,082 per-tick
# columns with names up to "49_49", which n <= 50 guarantees; 854 wiring columns
# "weight_final_run_NNNN"). tests/test_hdf5_layout.py re-measures the limits.
PER_TICK_SPLIT_ABOVE = 1000
WIRING_SPLIT_ABOVE = 800
SPLIT = "split"
_WEIGHT_COLUMN = {"per_tick": re.compile(r"^\d+_\d+$"), "wiring": re.compile(r"^weight_final_run_\d+$")}
_SPLIT_ABOVE = {"per_tick": PER_TICK_SPLIT_ABOVE, "wiring": WIRING_SPLIT_ABOVE}
_SPLIT_NOTED = set()


class TrackingLayoutError(RuntimeError):
    """A tracking record that cannot be split or reassembled as documented."""


def _weight_columns(name: str, dtype: np.dtype):
    """(base field names, weight column names) of a per_tick / wiring record. The weight
    columns must be the float32 tail of the record (as both simulations build it)."""
    names = dtype.names or ()
    is_w = [bool(_WEIGHT_COLUMN[name].match(n)) for n in names]
    first = is_w.index(True) if True in is_w else len(names)
    if not all(is_w[first:]):
        raise TrackingLayoutError(f"{name}: weight columns are not the tail of the record "
                                  f"({[n for n, w in zip(names, is_w) if not w][:5]} after them)")
    wcols = list(names[first:])
    bad = [c for c in wcols if dtype[c] != np.float32]
    if bad:
        raise TrackingLayoutError(f"{name}: weight columns not float32: {bad[:5]}")
    return list(names[:first]), wcols


def _write_record(grp, name: str, arr: np.ndarray):
    """Write `arr` as dataset `name` in `grp`: one compound dataset when it has at most
    _SPLIT_ABOVE[name] weight columns (the format of every file so far), else split."""
    attr = f"{name}_layout"
    for k in (name, f"{name}_base", f"{name}_weights", f"{name}_weight_names"):
        if k in grp:
            del grp[k]
    if attr in grp.attrs:
        del grp.attrs[attr]
    base_names, wcols = _weight_columns(name, arr.dtype)
    if len(wcols) <= _SPLIT_ABOVE[name]:
        grp.create_dataset(name, data=arr, compression='gzip')
        return
    if name not in _SPLIT_NOTED:
        _SPLIT_NOTED.add(name)
        print(f"[hdf5] {name}: {len(wcols)} weight columns > {_SPLIT_ABOVE[name]} (HDF5's header "
              f"limit): written split ({name}_base + {name}_weights); read it with "
              f"simulate.hdf5_utils.open_tracking", file=sys.stderr, flush=True)
    base = np.empty(arr.shape, dtype=[(f, arr.dtype[f]) for f in base_names])
    for f in base_names:
        base[f] = arr[f]
    w = np.empty(arr.shape + (len(wcols),), dtype=np.float32)
    for j, c in enumerate(wcols):
        w[..., j] = arr[c]
    grp.create_dataset(f"{name}_base", data=base, compression='gzip')
    grp.create_dataset(f"{name}_weights", data=w, compression='gzip')
    grp.create_dataset(f"{name}_weight_names",
                       data=np.array([c.encode() for c in wcols], dtype=f"S{max(len(c) for c in wcols)}"))
    grp.attrs[attr] = SPLIT


def write_per_tick(grp, per_tick_array: np.ndarray):
    """One run's per-tick record into an open run group (see the module docstring)."""
    _write_record(grp, "per_tick", per_tick_array)


def write_wiring(grp, wiring_array: np.ndarray):
    """One variant's wiring record into an open variant group (see the module docstring)."""
    _write_record(grp, "wiring", wiring_array)


# ---------------------------------------------------------------------------
# Reading: one code path for both layouts
# ---------------------------------------------------------------------------

class JoinedRecord:
    """A split per_tick / wiring record, read like the h5py dataset it replaces:
    `[()]`, `[...]`, slices, integer and index selections along the rows, a field name
    (`ds["energy"]`), `.shape`, `.dtype`, `len()`, `.attrs` (of the base dataset)."""

    def __init__(self, grp, name: str):
        self._grp, self.name = grp, name
        for k in (f"{name}_base", f"{name}_weights", f"{name}_weight_names"):
            if k not in grp:
                raise TrackingLayoutError(f"{grp.name}: {name}_layout is 'split' but {k} is missing")
        self._base = grp[f"{name}_base"]
        self._w = grp[f"{name}_weights"]
        self._wcols = [c.decode() for c in grp[f"{name}_weight_names"][()]]
        if self._w.shape != self._base.shape + (len(self._wcols),):
            raise TrackingLayoutError(f"{grp.name}: {name}_weights has shape {self._w.shape}, "
                                      f"expected {self._base.shape + (len(self._wcols),)}")
        self.dtype = np.dtype([(f, self._base.dtype[f]) for f in self._base.dtype.names]
                              + [(c, np.float32) for c in self._wcols])
        self.shape = self._base.shape
        self.attrs = self._base.attrs

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, sel):
        if isinstance(sel, str):
            if sel in self._base.dtype.names:
                return self._base[sel]
            if sel in self._wcols:
                return self._w[:, self._wcols.index(sel)]
            raise ValueError(f"no field {sel!r} in {self._grp.name}/{self.name}")
        if sel == () or sel is Ellipsis:
            base, w = self._base[()], self._w[()]
        else:
            base, w = self._base[sel], self._w[sel]
        out = np.empty(np.shape(base), dtype=self.dtype)
        for f in self._base.dtype.names:
            out[f] = base[f]
        for j, c in enumerate(self._wcols):
            out[c] = w[..., j]
        return out

    def __array__(self, dtype=None, copy=None):
        a = self[()]
        return a if dtype is None else a.astype(dtype)


class TrackingGroup:
    """An h5py group seen through `open_tracking`: identical, except that a split
    per_tick / wiring record appears as one dataset under its usual name."""

    def __init__(self, grp):
        self._g = grp

    def _logical(self, name):
        return self._g.attrs.get(f"{name}_layout") == SPLIT

    def __getitem__(self, key):
        parts = [p for p in str(key).split("/") if p]
        if str(key).startswith("/"):
            node = TrackingGroup(self._g.file)
        else:
            node = self
        if not parts:
            return node
        for i, p in enumerate(parts):
            if not isinstance(node, TrackingGroup):
                raise KeyError(f"{key}: {p!r} below a dataset")
            node = node._child(p)
        return node

    def _child(self, p):
        if p in ("per_tick", "wiring") and self._logical(p):
            return JoinedRecord(self._g, p)
        obj = self._g[p]
        return TrackingGroup(obj) if isinstance(obj, h5py.Group) else obj

    def _names(self):
        hidden = set()
        extra = []
        for n in ("per_tick", "wiring"):
            if self._logical(n):
                hidden |= {f"{n}_base", f"{n}_weights", f"{n}_weight_names"}
                extra.append(n)
        names = [k for k in self._g.keys() if k not in hidden]
        return names + [n for n in extra if n not in names]

    def __contains__(self, key):
        try:
            self[key]
            return True
        except KeyError:
            return False

    def keys(self):
        return self._names()

    def __iter__(self):
        return iter(self._names())

    def __len__(self):
        return len(self._names())

    def items(self):
        return [(k, self._child(k)) for k in self._names()]

    def values(self):
        return [self._child(k) for k in self._names()]

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __getattr__(self, attr):          # attrs, name, file, visititems, ... of the group
        return getattr(self._g, attr)

    def __repr__(self):
        return f"<TrackingGroup {self._g.name!r}>"


class TrackingReader(TrackingGroup):
    """`open_tracking(path)`: an h5py.File (read-only by default) whose per_tick and
    wiring records read the same for both layouts. Use like h5py.File, incl. `with`."""

    def __init__(self, path, mode="r"):
        if mode != "r":
            raise ValueError("open_tracking reads; write with the save_* functions")
        super().__init__(h5py.File(path, mode))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._g.close()
        return False

    def close(self):
        self._g.close()


def open_tracking(path, mode="r") -> "TrackingReader":
    """Open a tracking HDF5 file so that `f["variant_V/run_R/per_tick"][()]` and
    `f["variant_V/wiring"][()]` return the full record for either layout (one compound
    dataset, or split above the header limit). For files without split records it reads
    exactly what h5py.File reads."""
    return TrackingReader(path, mode)


def read_per_tick(run_group) -> np.ndarray:
    """The full per-tick record of an open run group (h5py or open_tracking), either layout."""
    g = run_group._g if isinstance(run_group, TrackingGroup) else run_group
    return JoinedRecord(g, "per_tick")[()] if g.attrs.get("per_tick_layout") == SPLIT else g["per_tick"][()]


def read_wiring(variant_group) -> np.ndarray:
    """The full wiring record of an open variant group, either layout."""
    g = variant_group._g if isinstance(variant_group, TrackingGroup) else variant_group
    return JoinedRecord(g, "wiring")[()] if g.attrs.get("wiring_layout") == SPLIT else g["wiring"][()]


def _flatten_config(config: dict) -> dict:
    """
    Recursively flatten YAML config with hierarchical prefixes for all properties.
    
    Excludes only: viz_enabled, viz_fps, viz_brain_enabled, viz_brain_fps
    
    All other YAML parameters are written with prefixes based on their hierarchy.
    Example: brain.output_mapping.5 -> brain_output_mapping_5
             experiment.max_ticks -> experiment_max_ticks
    
    Args:
        config: The parsed YAML config dict
    
    Returns:
        Flat dict ready for HDF5 attributes (all values HDF5-compatible)
    """
    flattened = {}
    excluded_keys = {
        "viz_enabled",
        "viz_fps",
        "viz_brain_enabled",
        "viz_brain_fps"
    }
    
    def _make_hdf5_compatible(value):
        """Convert a value to HDF5-compatible type."""
        if isinstance(value, bool):
            return int(value)  # Convert bool to 0/1
        elif isinstance(value, list):
            return json.dumps(value)  # Convert lists to JSON
        elif value is None:
            return ""  # Convert None to empty string
        else:
            return value
    
    def _recursive_flatten(d, prefix=""):
        """Recursively flatten nested dicts with hierarchical prefix."""
        for key, value in d.items():
            if key in excluded_keys:
                continue
            
            key_str = str(key)  # Ensure key is string
            
            # Build full key with prefix
            if prefix:
                full_key = f"{prefix}_{key_str}"
            else:
                full_key = key_str
            
            if isinstance(value, dict):
                # Recurse into nested dicts with updated prefix
                _recursive_flatten(value, full_key)
            else:
                # Convert value to HDF5-compatible type
                flattened[full_key] = _make_hdf5_compatible(value)
    
    _recursive_flatten(config)
    return flattened


def create_hdf5_file(hdf5_path: Path, yaml_config: dict):
    """
    Create and initialize an HDF5 file with YAML configuration as top-level attributes.
    
    All config values are flattened and stored as attributes (except viz parameters).
    
    Args:
        hdf5_path: Path to create the HDF5 file
        yaml_config: The parsed YAML config dict
    """
    with h5py.File(hdf5_path, 'w') as f:
        # Flatten the YAML config and store as individual attributes
        flattened = _flatten_config(yaml_config)
        
        for attr_name, attr_value in flattened.items():
            try:
                f.attrs[str(attr_name)] = attr_value
            except (TypeError, ValueError):
                # Fallback: convert to string if type is incompatible
                f.attrs[str(attr_name)] = str(attr_value)


def save_genome_properties_to_hdf5(hdf5_path: Path, genomes: list, replay_info: dict = None):
    """
    Save genome generation parameters to HDF5 as a vertical list dataset.
    
    Normal mode:          Saves genome params from the first genome (shared across all variants).
    Test mode:            Saves genome params + test_loaded_from_file / test_source_filename /
                          test_genome_ids entries recording where the genomes were loaded from.
    Replay/Benchmark mode: Saves only source metadata (loaded_from_file, source_filename,
                           genome_ids, runs_to_load).
    
    Saves as a structured array with name-value pairs (one row per parameter).
    
    Args:
        hdf5_path: Path to HDF5 file
        genomes: List of genome objects (each has a .params attribute)
        replay_info: Optional dict with metadata. Required keys:
            - 'mode': 'test', 'replay', or 'benchmark'
            - 'source_path': path to source HDF5 file
            - 'source_filename': filename without extension
            - 'genome_ids': genome IDs loaded ('all', int, or list)
            - 'runs_to_load': run indices loaded ('all', list, or 'n/a' for test mode)
    """
    if not genomes and replay_info is None:
        return
    
    # Create structured array with name-value pairs (both as byte strings)
    dtype = np.dtype([
        ('name', 'S100'),   # byte string, up to 100 chars
        ('value', 'S256'),  # byte string, up to 256 chars (for numeric or text values)
    ])
    
    if replay_info is not None and replay_info.get('mode') != 'test':
        # REPLAY / BENCHMARK MODE: Save metadata about where genomes were loaded from
        properties = {
            'loaded_from_file': replay_info['source_path'],
            'source_filename': replay_info['source_filename'],
            'genome_ids': str(replay_info['genome_ids']),
            'runs_to_load': str(replay_info['runs_to_load']),
        }
    else:
        # NORMAL / TEST MODE: Extract params from first genome
        params = genomes[0].params
        
        # Convert dataclass to dict
        params_dict = {f.name: getattr(params, f.name) for f in params.__dataclass_fields__.values()}
        
        # Flatten the params dict
        flattened = _flatten_config(params_dict)
        properties = flattened
        
        # TEST MODE: also record the genome source
        if replay_info is not None:
            properties['test_loaded_from_file'] = replay_info['source_path']
            properties['test_source_filename'] = replay_info['source_filename']
            properties['test_genome_ids'] = str(replay_info['genome_ids'])
    
    # Create array with one row per property
    genome_props = np.zeros(len(properties), dtype=dtype)
    
    for i, (key, value) in enumerate(sorted(properties.items())):
        # Convert both key and value to bytes
        genome_props[i] = (key.encode('utf-8'), str(value).encode('utf-8'))
    
    # Write to HDF5
    with h5py.File(hdf5_path, 'a') as f:
        # Delete if exists
        if 'genome_properties' in f:
            del f['genome_properties']
        
        # Create dataset with compression
        f.create_dataset(
            'genome_properties',
            data=genome_props,
            compression='gzip',
            compression_opts=4,
        )


def save_variant_summary_to_hdf5(hdf5_path: Path, variant_id: int, summary_array: np.ndarray, lock=None):
    """Save variant summary stats to HDF5.
    
    Args:
        hdf5_path: Path to HDF5 file
        variant_id: Variant identifier (0-based)
        summary_array: Data to save
        lock: Optional multiprocessing.Lock() for synchronized parallel writes
    """
    def _write():
        group_name = f'variant_{variant_id}'
        with h5py.File(hdf5_path, 'a') as f:
            if group_name not in f:
                grp = f.create_group(group_name)
            else:
                grp = f[group_name]
            if 'summary' in grp:
                del grp['summary']
            grp.create_dataset('summary', data=summary_array, compression='gzip')
    
    if lock is not None:
        with lock:
            _write()
    else:
        _write()


def save_wiring_to_hdf5(hdf5_path: Path, variant_id: int, wiring_array: np.ndarray, lock=None):
    """Save wiring (initial and final weights per run) to HDF5.
    
    Args:
        hdf5_path: Path to HDF5 file
        variant_id: Variant identifier (0-based)
        wiring_array: Data to save
        lock: Optional multiprocessing.Lock() for synchronized parallel writes
    """
    def _write():
        group_name = f'variant_{variant_id}'
        with h5py.File(hdf5_path, 'a') as f:
            if group_name not in f:
                grp = f.create_group(group_name)
            else:
                grp = f[group_name]
            write_wiring(grp, wiring_array)
    
    if lock is not None:
        with lock:
            _write()
    else:
        _write()


def save_modulation_to_hdf5(hdf5_path: Path, variant_id: int, modulation_array: np.ndarray, lock=None):
    """Save modulation specs to HDF5.
    
    Args:
        hdf5_path: Path to HDF5 file
        variant_id: Variant identifier (0-based)
        modulation_array: Data to save
        lock: Optional multiprocessing.Lock() for synchronized parallel writes
    """
    def _write():
        group_name = f'variant_{variant_id}'
        with h5py.File(hdf5_path, 'a') as f:
            if group_name not in f:
                grp = f.create_group(group_name)
            else:
                grp = f[group_name]
            if 'modulation' in grp:
                del grp['modulation']
            grp.create_dataset('modulation', data=modulation_array, compression='gzip')
    
    if lock is not None:
        with lock:
            _write()
    else:
        _write()


def save_eta_to_hdf5(hdf5_path: Path, variant_id: int, eta: float, lock=None):
    """Save eta (global plasticity factor) as a scalar dataset to HDF5.
    
    Args:
        hdf5_path: Path to HDF5 file
        variant_id: Variant identifier (0-based)
        eta: Global plasticity factor
        lock: Optional multiprocessing.Lock() for synchronized parallel writes
    """
    def _write():
        group_name = f'variant_{variant_id}'
        with h5py.File(hdf5_path, 'a') as f:
            if group_name not in f:
                grp = f.create_group(group_name)
            else:
                grp = f[group_name]
            if 'eta' in grp:
                del grp['eta']
            grp.create_dataset('eta', data=np.float32(eta))
    
    if lock is not None:
        with lock:
            _write()
    else:
        _write()


def save_tonic_activations_to_hdf5(hdf5_path: Path, variant_id: int, tonic_activations: np.ndarray, lock=None):
    """Save tonic activations vector to HDF5.
    
    Args:
        hdf5_path: Path to HDF5 file
        variant_id: Variant identifier (0-based)
        tonic_activations: 1-D array of tonic activation values per neuron
        lock: Optional multiprocessing.Lock() for synchronized parallel writes
    """
    def _write():
        group_name = f'variant_{variant_id}'
        with h5py.File(hdf5_path, 'a') as f:
            if group_name not in f:
                grp = f.create_group(group_name)
            else:
                grp = f[group_name]
            if 'tonic_activations' in grp:
                del grp['tonic_activations']
            grp.create_dataset('tonic_activations', data=tonic_activations, compression='gzip')
    
    if lock is not None:
        with lock:
            _write()
    else:
        _write()


def save_heatmaps_to_hdf5(hdf5_path: Path, variant_id: int, run_id: int, staying_heatmap: np.ndarray, lock=None):
    """Save heatmap (2D array) to HDF5.
    
    Args:
        hdf5_path: Path to HDF5 file
        variant_id: Variant identifier
        run_id: Run identifier
        staying_heatmap: Staying heatmap data
        lock: Optional multiprocessing.Lock() for synchronized parallel writes
    """
    def _write():
        variant_group = f'variant_{variant_id}'
        run_group = f'run_{run_id}'
        with h5py.File(hdf5_path, 'a') as f:
            if variant_group not in f:
                grp_variant = f.create_group(variant_group)
            else:
                grp_variant = f[variant_group]
            if run_group not in grp_variant:
                grp_run = grp_variant.create_group(run_group)
            else:
                grp_run = grp_variant[run_group]
            if 'staying' in grp_run:
                del grp_run['staying']
            grp_run.create_dataset('staying', data=staying_heatmap, compression='gzip')
    
    if lock is not None:
        with lock:
            _write()
    else:
        _write()


def save_per_tick_to_hdf5(hdf5_path: Path, variant_id: int, run_id: int, per_tick_array: np.ndarray, lock=None):
    """Save per-tick tracking data to HDF5.
    
    Args:
        hdf5_path: Path to HDF5 file
        variant_id: Variant identifier
        run_id: Run identifier
        per_tick_array: Per-tick tracking data
        lock: Optional multiprocessing.Lock() for synchronized parallel writes
    """
    def _write():
        variant_group = f'variant_{variant_id}'
        run_group = f'run_{run_id}'
        with h5py.File(hdf5_path, 'a') as f:
            if variant_group not in f:
                grp_variant = f.create_group(variant_group)
            else:
                grp_variant = f[variant_group]
            if run_group not in grp_variant:
                grp_run = grp_variant.create_group(run_group)
            else:
                grp_run = grp_variant[run_group]
            write_per_tick(grp_run, per_tick_array)
    
    if lock is not None:
        with lock:
            _write()
    else:
        _write()

