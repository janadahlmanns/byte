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
    Flatten config dict into a flat set of HDF5 attributes with explicit naming.
    
    Uses simple, unambiguous attribute names without hierarchical prefixes.
    The viz section is skipped entirely.
    
    Args:
        config: Dict with nested sections to flatten
    
    Returns:
        Flat dict ready for HDF5 attributes
    """
    flattened = {}
    
    # experiment_metadata section
    if "experiment_metadata" in config:
        em = config["experiment_metadata"]
        flattened["experiment_folder"] = em.get("experiment_folder", "")
        flattened["simulation_name"] = em.get("simulation_name", "")
        flattened["simulation_config_path"] = em.get("simulation_config_path", "")
        flattened["brain_init_type"] = em.get("brain_init_type", "")
    
    # wiring_randomization section
    if "wiring_randomization" in config:
        wr = config["wiring_randomization"]
        flattened["connectivity_degree_excitatory"] = wr.get("connectivity_degree_excitatory")
        flattened["connectivity_degree_inhibitory"] = wr.get("connectivity_degree_inhibitory")
        flattened["modulation_degree_potentiation"] = wr.get("modulation_degree_potentiation")
        flattened["modulation_degree_depression"] = wr.get("modulation_degree_depression")
        flattened["wiring_randomization_seed_base"] = wr.get("wiring_randomization_seed_base")
        flattened["n_variants"] = wr.get("n_variants")
    
    # simulation_parameters section
    if "simulation_parameters" in config:
        sp = config["simulation_parameters"]
        flattened["max_ticks"] = sp.get("max_ticks")
        flattened["n_runs"] = sp.get("n_runs")
        flattened["feeding_initial_fraction_per_cell"] = sp.get("feeding_initial_fraction_per_cell")
        flattened["feeding_regrow_time"] = sp.get("feeding_regrow_time")
    
    # data_tracking section
    if "data_tracking" in config:
        dt = config["data_tracking"]
        flattened["enable_per_run_tracking"] = dt.get("enable_per_run_tracking")
        flattened["enable_per_tick_tracking"] = dt.get("enable_per_tick_tracking")
        flattened["enable_heat_map_tracking"] = dt.get("enable_heat_map_tracking")
    
    # brain_architecture section
    if "brain_architecture" in config:
        ba = config["brain_architecture"]
        flattened["n_neurons"] = ba.get("n_neurons")
        flattened["max_decision_delay"] = ba.get("max_decision_delay")
        flattened["eta"] = ba.get("eta")
        flattened["excitatory_weight"] = ba.get("excitatory_weight")
        flattened["inhibitory_weight"] = ba.get("inhibitory_weight")
        flattened["n_input_neurons"] = ba.get("n_input_neurons")
        flattened["n_output_neurons"] = ba.get("n_output_neurons")
        flattened["n_always_on_neurons"] = ba.get("n_always_on_neurons")
    
    # World, food, worm, sensors, decisionmaking from top-level or nested
    # (These are already flattened with prefixes by _rename_world_config_keys)
    for key in ["world_grid_width", "world_grid_height", "world_start_pos", "world_rng_seed",
                "feeding_initial", "feeding_regrow",
                "worm_speed", "worm_energy_capacity", "worm_metabolic_rate", "worm_sensors_active",
                "decisionmaking_version", "brain"]:
        if key in config:
            flattened[key] = config[key]
    
    # Convert values to HDF5-compatible types
    for key, value in flattened.items():
        if isinstance(value, bool):
            flattened[key] = int(value)
        elif isinstance(value, list):
            flattened[key] = json.dumps(value)
        elif value is None:
            flattened[key] = ""
    
    return flattened


def create_hdf5_file(hdf5_path: Path, comprehensive_config: dict):
    """
    Create and initialize an HDF5 file with experiment metadata.
    
    All config values are flattened into individual attributes with explicit names.
    The 'viz' section is completely skipped.
    
    Args:
        hdf5_path: Path to create the HDF5 file
        comprehensive_config: Dict with experiment config to store as attributes
    """
    with h5py.File(hdf5_path, 'w') as f:
        # Flatten the entire config and store as individual attributes
        flattened = _flatten_config(comprehensive_config)
        
        for attr_name, attr_value in flattened.items():
            try:
                f.attrs[attr_name] = attr_value
            except Exception as e:
                # Fallback: convert to string if type is incompatible
                f.attrs[attr_name] = str(attr_value)


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


def save_heatmaps_to_hdf5(hdf5_path: Path, variant_id: int, run_id: int, entering_heatmap: np.ndarray, staying_heatmap: np.ndarray, lock=None):
    """Save heatmaps (2D arrays) to HDF5.
    
    Args:
        hdf5_path: Path to HDF5 file
        variant_id: Variant identifier
        run_id: Run identifier
        entering_heatmap: Entering heatmap data
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
            if 'entering' in grp_run:
                del grp_run['entering']
            if 'staying' in grp_run:
                del grp_run['staying']
            grp_run.create_dataset('entering', data=entering_heatmap, compression='gzip')
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
