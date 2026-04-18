"""
HDF5 utilities for saving experiment data.

Provides functions to save data in HDF5 format, mirroring the folder structure:
- Root level: experiment metadata (attributes)
- /variant_XX/ groups: summary, wiring, modulation, heatmaps datasets
- /variant_XX/per_tick_run_XXXX/ groups: per-tick data datasets

Supports parallel writing with multiprocessing.Lock() to synchronize concurrent HDF5 writes.
"""

import json
from pathlib import Path
import numpy as np
import h5py
from multiprocessing import Lock


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


def save_genome_properties_to_hdf5(hdf5_path: Path, genomes: list):
    """
    Save genome generation parameters to HDF5 as a vertical list dataset.
    
    Extracts the params from the first genome (shared across all variants)
    and saves as a structured array with name-value pairs (one row per parameter).
    
    Args:
        hdf5_path: Path to HDF5 file
        genomes: List of genome objects (each has a .params attribute)
    """
    if not genomes:
        return
    
    # Extract params from first genome (same generation params for all variants)
    params = genomes[0].params
    
    # Convert dataclass to dict
    params_dict = {f.name: getattr(params, f.name) for f in params.__dataclass_fields__.values()}
    
    # Flatten the params dict
    flattened = _flatten_config(params_dict)
    
    # Create structured array with name-value pairs (both as byte strings)
    dtype = np.dtype([
        ('name', 'S100'),   # byte string, up to 100 chars
        ('value', 'S256'),  # byte string, up to 256 chars (for numeric or text values)
    ])
    
    # Create array with one row per parameter
    genome_props = np.zeros(len(flattened), dtype=dtype)
    
    for i, (key, value) in enumerate(sorted(flattened.items())):
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
        variant_id: Variant identifier
        summary_array: Data to save
        lock: Optional multiprocessing.Lock() for synchronized parallel writes
    """
    def _write():
        group_name = f'variant_{variant_id:02d}'
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
        variant_id: Variant identifier
        wiring_array: Data to save
        lock: Optional multiprocessing.Lock() for synchronized parallel writes
    """
    def _write():
        group_name = f'variant_{variant_id:02d}'
        with h5py.File(hdf5_path, 'a') as f:
            if group_name not in f:
                grp = f.create_group(group_name)
            else:
                grp = f[group_name]
            if 'wiring' in grp:
                del grp['wiring']
            grp.create_dataset('wiring', data=wiring_array, compression='gzip')
    
    if lock is not None:
        with lock:
            _write()
    else:
        _write()


def save_modulation_to_hdf5(hdf5_path: Path, variant_id: int, modulation_array: np.ndarray, lock=None):
    """Save modulation specs to HDF5.
    
    Args:
        hdf5_path: Path to HDF5 file
        variant_id: Variant identifier
        modulation_array: Data to save
        lock: Optional multiprocessing.Lock() for synchronized parallel writes
    """
    def _write():
        group_name = f'variant_{variant_id:02d}'
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
        variant_group = f'variant_{variant_id:02d}'
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
        variant_group = f'variant_{variant_id:02d}'
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
            if 'per_tick' in grp_run:
                del grp_run['per_tick']
            grp_run.create_dataset('per_tick', data=per_tick_array, compression='gzip')
    
    if lock is not None:
        with lock:
            _write()
    else:
        _write()

