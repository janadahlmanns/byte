"""
HDF5 utilities for saving experiment data.

Provides functions to save data in HDF5 format, mirroring the folder structure:
- Root level: experiment metadata (attributes)
- /variant_XX/ groups: summary, wiring, modulation, heatmaps datasets
- /variant_XX/per_tick_run_XXXX/ groups: per-tick data datasets
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import h5py


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


def save_variant_summary_to_hdf5(hdf5_path: Path, variant_id: int, summary_df: pd.DataFrame):
    """
    Save variant summary stats to HDF5.
    
    Args:
        hdf5_path: Path to HDF5 file
        variant_id: Variant number (1-indexed)
        summary_df: Summary DataFrame with columns: run_id, lifetime_ticks, foods, distance, etc.
    """
    group_name = f'variant_{variant_id:02d}'
    
    with h5py.File(hdf5_path, 'a') as f:
        if group_name not in f:
            grp = f.create_group(group_name)
        else:
            grp = f[group_name]
        
        # Convert DataFrame to structured array and save
        # First, convert all columns to float or int as appropriate
        data_dict = {}
        for col in summary_df.columns:
            if col == 'run_id':
                data_dict[col] = summary_df[col].astype(int)
            else:
                data_dict[col] = summary_df[col].astype(float)
        
        # Create dataset
        summary_array = np.array([(row['run_id'], row['lifetime_ticks'], row['foods'], row['distance'], 
                                   row['final_energy'], row['moves_north'], row['moves_south'], 
                                   row['moves_east'], row['moves_west'], row['food_sensed_north'], 
                                   row['food_sensed_east'], row['food_sensed_south'], row['food_sensed_west'],
                                   row['decisions'], row['correct_decisions'])
                                  for _, row in summary_df.iterrows()],
                               dtype=[('run_id', 'i4'), ('lifetime_ticks', 'i4'), ('foods', 'i4'), 
                                     ('distance', 'f4'), ('final_energy', 'i4'), ('moves_north', 'i4'),
                                     ('moves_south', 'i4'), ('moves_east', 'i4'), ('moves_west', 'i4'),
                                     ('food_sensed_north', 'i4'), ('food_sensed_east', 'i4'),
                                     ('food_sensed_south', 'i4'), ('food_sensed_west', 'i4'),
                                     ('decisions', 'i4'), ('correct_decisions', 'i4')])
        
        if 'summary' in grp:
            del grp['summary']
        grp.create_dataset('summary', data=summary_array, compression='gzip')


def save_wiring_to_hdf5(hdf5_path: Path, variant_id: int, wiring_df: pd.DataFrame):
    """
    Save wiring (initial and final weights per run) to HDF5 as a single dataset.
    
    Args:
        hdf5_path: Path to HDF5 file
        variant_id: Variant number (1-indexed)
        wiring_df: DataFrame with columns: src, tgt, weight_initial, weight_final_run_XXXX, ...
    """
    group_name = f'variant_{variant_id:02d}'
    
    with h5py.File(hdf5_path, 'a') as f:
        if group_name not in f:
            grp = f.create_group(group_name)
        else:
            grp = f[group_name]
        
        # Delete existing dataset if present
        if 'wiring' in grp:
            del grp['wiring']
        
        # Convert DataFrame columns to appropriate types
        # Build a list of (dtype_field, data) for the structured array
        dtype_fields = []
        data_rows = []
        
        for _, row in wiring_df.iterrows():
            row_dict = {}
            row_dict['src'] = int(row['src'])
            row_dict['tgt'] = int(row['tgt'])
            row_dict['weight_initial'] = float(row['weight_initial'])
            
            # Add final weights for each run
            for col in wiring_df.columns:
                if col.startswith('weight_final_run_'):
                    run_label = col.replace('weight_final_run_', '')
                    row_dict[f'weight_final_run_{run_label}'] = float(row[col])
            
            data_rows.append(row_dict)
        
        # Create dtype dynamically based on columns
        dtype_fields = [('src', 'i2'), ('tgt', 'i2'), ('weight_initial', 'f4')]
        for col in wiring_df.columns:
            if col.startswith('weight_final_run_'):
                run_label = col.replace('weight_final_run_', '')
                dtype_fields.append((f'weight_final_run_{run_label}', 'f4'))
        
        # Create structured array
        wiring_array = np.array([tuple(row.get(field[0], 0) for field in dtype_fields) 
                                 for row in data_rows],
                                dtype=dtype_fields)
        
        # Save as dataset
        grp.create_dataset('wiring', data=wiring_array, compression='gzip')


def save_modulation_to_hdf5(hdf5_path: Path, variant_id: int, modulation_df: pd.DataFrame):
    """
    Save modulation specs to HDF5.
    
    Args:
        hdf5_path: Path to HDF5 file
        variant_id: Variant number (1-indexed)
        modulation_df: DataFrame with columns: target_src, target_tgt, modulator_src, modulation_weight
    """
    group_name = f'variant_{variant_id:02d}'
    
    with h5py.File(hdf5_path, 'a') as f:
        if group_name not in f:
            grp = f.create_group(group_name)
        else:
            grp = f[group_name]
        
        if 'modulation' in grp:
            del grp['modulation']
        
        if len(modulation_df) > 0:
            # Save as structured array
            mod_array = np.array([(int(row['target_src']), int(row['target_tgt']), 
                                   int(row['modulator_src']), float(row['modulation_weight']))
                                  for _, row in modulation_df.iterrows()],
                                dtype=[('target_src', 'i2'), ('target_tgt', 'i2'), 
                                       ('modulator_src', 'i2'), ('modulation_weight', 'f4')])
            grp.create_dataset('modulation', data=mod_array, compression='gzip')
        else:
            # Empty modulation - create empty dataset
            grp.create_dataset('modulation', data=np.array([], dtype=[('target_src', 'i2'), ('target_tgt', 'i2'), 
                                                                        ('modulator_src', 'i2'), ('modulation_weight', 'f4')]))


def save_heatmaps_to_hdf5(hdf5_path: Path, variant_id: int, heatmaps_df: pd.DataFrame):
    """
    Save heatmaps to HDF5 as a single dataset.
    
    Expected DataFrame has columns: field_y, field_x, entering_run_XXXX, staying_run_XXXX, ...
    Each row is a grid cell with entering/staying counts for each run.
    
    Args:
        hdf5_path: Path to HDF5 file
        variant_id: Variant number (1-indexed)
        heatmaps_df: DataFrame with field coordinates and entering/staying counts per run
    """
    group_name = f'variant_{variant_id:02d}'
    
    with h5py.File(hdf5_path, 'a') as f:
        if group_name not in f:
            grp = f.create_group(group_name)
        else:
            grp = f[group_name]
        
        # Delete existing dataset if present
        if 'heatmaps' in grp:
            del grp['heatmaps']
        
        # Build dtype dynamically based on all columns
        dtype_fields = [('field_y', 'i2'), ('field_x', 'i2')]
        for col in heatmaps_df.columns:
            if col.startswith('entering_run_') or col.startswith('staying_run_'):
                dtype_fields.append((col, 'i4'))
        
        # Create structured array
        data_rows = []
        for _, row in heatmaps_df.iterrows():
            row_data = (int(row['field_y']), int(row['field_x']))
            for col in heatmaps_df.columns:
                if col.startswith('entering_run_') or col.startswith('staying_run_'):
                    row_data += (int(row[col]),)
            data_rows.append(row_data)
        
        heatmaps_array = np.array(data_rows, dtype=dtype_fields)
        
        # Save as dataset
        grp.create_dataset('heatmaps', data=heatmaps_array, compression='gzip')


def save_per_tick_to_hdf5(hdf5_path: Path, variant_id: int, run_id: int, per_tick_df: pd.DataFrame):
    """
    Save per-tick tracking data to HDF5.
    
    Creates variant_XX/run_X/ groups (only if ENABLE_PER_TICK_TRACKING is true) with:
    - simulation dataset: tick, food_sensed_N/E/S/W, movement, food_consumed, energy, manhattan_dist, decision_made
    - weights dataset: conn_src_tgt columns (connection weights over time)
    
    Args:
        hdf5_path: Path to HDF5 file
        variant_id: Variant number (1-indexed)
        run_id: Run number (1-indexed)
        per_tick_df: DataFrame with tick, food_sensed_*, movement, energy, distance, decision, conn_*, etc.
    """
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
        
        # === SIMULATION DATASET ===
        # Standard columns: tick, food sensing, movement, consumption, energy, distance, decision
        standard_cols = ['tick', 'food_sensed_N', 'food_sensed_E', 'food_sensed_S', 'food_sensed_W',
                        'movement', 'food_consumed', 'energy', 'manhattan_dist', 'decision_made']
        
        # Create structured array for simulation data
        dtype_sim = [('tick', 'i4'), ('food_sensed_N', 'u1'), ('food_sensed_E', 'u1'),
                     ('food_sensed_S', 'u1'), ('food_sensed_W', 'u1'),
                     ('movement', 'S4'), ('food_consumed', 'u1'), ('energy', 'f4'),
                     ('manhattan_dist', 'u2'), ('decision_made', 'u1')]
        
        sim_data = np.array([(int(row['tick']), int(row['food_sensed_N']), int(row['food_sensed_E']),
                              int(row['food_sensed_S']), int(row['food_sensed_W']),
                              str(row['movement']), int(row['food_consumed']), float(row['energy']),
                              int(row['manhattan_dist']), int(row['decision_made']))
                             for _, row in per_tick_df[standard_cols].iterrows()],
                            dtype=dtype_sim)
        
        if 'simulation' in grp_run:
            del grp_run['simulation']
        grp_run.create_dataset('simulation', data=sim_data, compression='gzip')
        
        # === WEIGHTS DATASET ===
        # Collect all connection weight columns (conn_src_tgt format)
        conn_cols = [col for col in per_tick_df.columns if col.startswith('conn_')]
        
        if conn_cols:
            # Build dtype for weights with one field per connection
            dtype_weights = [(col.replace('conn_', ''), 'f4') for col in conn_cols]
            
            # Create structured array for weights
            weights_data = np.array([tuple(float(row[col]) for col in conn_cols)
                                     for _, row in per_tick_df[conn_cols].iterrows()],
                                    dtype=dtype_weights)
            
            if 'weights' in grp_run:
                del grp_run['weights']
            grp_run.create_dataset('weights', data=weights_data, compression='gzip')
        else:
            # No connection weights to save
            pass
