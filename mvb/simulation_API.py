"""Simulation API - Clean interface for running simulations.

Provides the primary entry point for executing simulations:
- eval_generation(): Run all variants across multiple processes or serially
- eval_variant(): Execute runs for a single genome variant
- simulate_run(): Execute a single simulation run

This module is designed to be imported by runner scripts.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager

import numpy as np
import h5py

from mvb.world import World
from mvb.feeding import seed_food
from mvb.worm import Worm
from mvb.world_renderer_qt import QtRenderer
from mvb.brain_renderer_qt import BrainQtRenderer
from simulate.pause_manager import get_pause_manager, PauseManagerExit
from simulate.hdf5_utils import (
    create_hdf5_file,
    save_variant_summary_to_hdf5,
    save_wiring_to_hdf5,
    save_modulation_to_hdf5,
    save_heatmaps_to_hdf5,
    save_per_tick_to_hdf5,
    save_genome_properties_to_hdf5,
)

from mvb.simulation_helper_functions import (
    MetricsRecorder,
    load_brain_module,
    get_connection_weight,
    get_num_workers,
    make_experiment_dir,
)


def simulate_run(world, worm, rec, rng_worker_decision, rng_worker_neuron_noise, max_ticks, pause_mgr):
    """Execute a single simulation run with all ticks until worm dies or max_ticks reached.
    
    Args:
        world: World instance
        worm: Worm instance
        rec: MetricsRecorder instance for tracking metrics, or None if no tracking details
        rng_worker_decision: RNG for decision-making
        rng_worker_neuron_noise: RNG for neuron noise during brain computation
        max_ticks: Maximum simulation ticks
        pause_mgr: Optional PauseManager for pause/exit handling
    
    Returns:
        Tuple of (world, worm, rec, pause_mgr) - all objects modified in place during simulation
    """
    try:
        while worm.alive and worm.ticks < max_ticks:
            # Check pause/exit at start of each tick
            if pause_mgr is not None:
                pause_mgr.check_pause()
            
            world.step()
            worm.step_day(rng_worker_decision, rng_worker_neuron_noise)
            worm.ticks += 1
            if rec is not None:
                rec.record(worm)
            
            # Double-check exit flag after each step
            if pause_mgr is not None and pause_mgr.should_exit():
                raise PauseManagerExit("Exit requested during simulation")
            
            # Wait to maintain FPS if visualization is enabled
            if worm.renderer is not None:
                worm.renderer.wait_frame()
    except PauseManagerExit:
        pass  # Exit simulation gracefully
    
    return world, worm, rec, pause_mgr
def eval_variant(
    variant_id,
    brain_module_name,
    grid_width,
    grid_height,
    start_pos,
    enable_per_run_tracking,
    enable_per_tick_tracking,
    enable_heat_map_tracking,
    feeding_cfg,
    worm_speed,
    worm_energy_capacity,
    worm_metabolic_rate,
    worm_movement_cost,
    n_runs,
    genome,
    brain_cfg,
    viz_enabled,
    viz_fps,
    viz_brain_enabled,
    viz_brain_fps,
    run_seeds,
    seeds_noise_variant,
    seeds_decision_variant,
    max_ticks,
    sensor_cfg,
):
    """Execute all runs for a single variant and return tracking results.
    
    Args:
        variant_id: Index of this variant (for indexing into seeds_*_variant arrays)
        brain_module_name: Name of brain module to import
        grid_width: World grid width
        grid_height: World grid height
        start_pos: Worm starting position (tuple)
        enable_per_run_tracking: Whether to track per-run metrics
        enable_per_tick_tracking: Whether to track per-tick data
        enable_heat_map_tracking: Whether to track heatmaps
        feeding_cfg: Feed config dict with keys: feeding_paradigm, initial_fraction_per_cell, regrow_time
        worm_speed: Worm speed parameter
        worm_energy_capacity: Worm energy capacity
        worm_metabolic_rate: Worm metabolic rate
        worm_movement_cost: Worm movement cost
        n_runs: Number of runs
        genome: Pre-generated genome dict
        brain_cfg: Brain configuration dict
        viz_enabled: Whether to enable world visualization
        viz_fps: World visualization FPS
        viz_brain_enabled: Whether to enable brain visualization
        viz_brain_fps: Brain visualization FPS
        run_seeds: Array of N_RUNS seeds for world initialization
        seeds_noise_variant: Array of RNG seeds for neuron noise (one per variant)
        seeds_decision_variant: Array of RNG seeds for decision-making (one per variant)
        max_ticks: Maximum simulation ticks
        sensor_cfg: Sensor configuration list
    
    Returns:
        Dict with tracking results containing:
        - lifespan_vector: Always included
        - summary_array, wiring_array, modulation_array: If enable_per_run_tracking
        - per_tick_all_runs: If enable_per_tick_tracking
        - heatmaps_all_runs: If enable_heat_map_tracking
    """
    rec = None
    # Always allocate lifespan vector (always available, no tracking flags needed)
    lifespan_vector = np.zeros(n_runs, dtype='i4')

    # Conditionally allocate full summary array (only if per-run tracking enabled)
    if enable_per_run_tracking:
        # Extract genome components for tracking
        connection_weights = genome["connection_weights"]
        connections_to_track = []
        for src in range(connection_weights.shape[0]):
            for tgt in range(connection_weights.shape[1]):
                if connection_weights[src, tgt, 0] != 0.0:
                    connections_to_track.append((src, tgt))
        summary_array = None
        dtype_summary = [('run_id', 'i2'), ('lifetime_ticks', 'i4'), ('foods', 'i4'),
                         ('distance', 'i4'), ('final_energy', 'f4'), ('seed_noise', 'u4'), ('seed_decision', 'u4')]
        summary_array = np.zeros(n_runs, dtype=dtype_summary)

        # Pre-allocate wiring array with columns for all run final weights
        dtype_wiring = [('src', 'i2'), ('tgt', 'i2'), ('weight_initial', 'f4')]
        for run_id in range(n_runs):
            dtype_wiring.append((f'weight_final_run_{run_id:04d}', 'f4'))
        wiring_array = np.zeros(len(connections_to_track), dtype=dtype_wiring)

        for idx, (src, tgt) in enumerate(connections_to_track):
            wiring_array[idx]['src'] = src
            wiring_array[idx]['tgt'] = tgt
            wiring_array[idx]['weight_initial'] = connection_weights[src, tgt, 0]

        # Pre-allocate modulation array
        dtype_modulation = [('target_src', 'i2'), ('target_tgt', 'i2'), ('modulator_src', 'i2'), ('modulation_weight', 'f4')]
        modulation_list = []
        modulator_spec = genome["modulation_spec"]
        for (target_src, target_tgt), modulators in modulator_spec.items():
            for mod_src, mod_weight in modulators:
                modulation_list.append((target_src, target_tgt, mod_src, mod_weight))
        modulation_array = np.array(modulation_list, dtype=dtype_modulation) if modulation_list else np.array([], dtype=dtype_modulation)

    # ============================================================
    # Create MetricsRecorder to track per-tick data (only if tracking enabled)
    # ============================================================
  
        if enable_per_tick_tracking or enable_heat_map_tracking:
            rec = MetricsRecorder.empty(genome, start_pos[0], start_pos[1], grid_height, grid_width, enable_per_run_tracking, enable_per_tick_tracking, enable_heat_map_tracking, max_ticks)     
        if enable_per_tick_tracking:
            per_tick_all_runs = {}
        if enable_heat_map_tracking:
            heatmaps_all_runs = {}
    else:
        summary_array = None
        wiring_array = None
        modulation_array = None
    


    # ============================================================
    # Load Brain Module
    # ============================================================
    brain_module = load_brain_module(brain_module_name)

    # ============================================================
    # Create temporary world and worm (will be reset per run so seed doesnt matter)
    # ============================================================
    world = World(grid_width, grid_height, start_pos, 0)
    world.feeding_cfg = feeding_cfg
    
    worm = Worm(worm_speed, worm_energy_capacity, worm_metabolic_rate, worm_movement_cost, world)
    worm.active_sensors = sensor_cfg
    worm.brain = brain_module

    # ============================================================
    # Simulate runs. For each run do:
    # ============================================================
    for run_id in range(n_runs):

        # ============================================================
        # Set rngs for this run
        # ============================================================
        rng_world_run = np.random.default_rng(run_seeds[run_id])
        rng_noise_run = np.random.default_rng(seeds_noise_variant[run_id])
        rng_decision_run = np.random.default_rng(seeds_decision_variant[run_id])

        # ============================================================
        # Call brain_module.init_brain(genome, brain_cfg) for a clean reset
        # ============================================================
        brain_module.init_brain(genome, brain_cfg)

        # ============================================================
        # Reset worm & simulation, reset world with the according run rng 
        # ============================================================
        world.reset_food()
        seed_food(world, feeding_cfg, rng_world_run)
        worm.reset()
        if rec is not None:
            rec.reset()
            rec.record(worm)

        # Create world renderer independently if world visualization is enabled
        if viz_enabled:
            try:
                if viz_fps > 0:
                    worm.renderer = QtRenderer(world, worm, viz_fps)
                    print(f"[viz] Created world renderer at {viz_fps} FPS")
                else:
                    worm.renderer = None
            except Exception as e:
                print(f"[WARNING] Failed to create world renderer: {e}.")
                worm.renderer = None
        else:
            worm.renderer = None       
        # Create brain renderer independently if brain visualization is enabled
        if viz_brain_enabled:
            try:
                if viz_brain_fps > 0:
                    brain_renderer = BrainQtRenderer(fps=viz_brain_fps)
                    brain_module._brain_renderer = brain_renderer
                    print(f"[viz] Created brain renderer at {viz_brain_fps} FPS")
                else:
                    brain_module._brain_renderer = None
            except Exception as e:
                print(f"[WARNING] Failed to create brain renderer: {e}.")
                brain_module._brain_renderer = None
        else:
            brain_module._brain_renderer = None
        # Get pause manager for checkpoints (if any visualization enabled)
        pause_mgr = None
        if viz_enabled or viz_brain_enabled:
            try:
                pause_mgr = get_pause_manager()
            except RuntimeError:
                pass

        # ============================================================
        # Simulate
        # ============================================================
        world, worm, rec, pause_mgr = simulate_run(world, worm, rec, rng_decision_run, rng_noise_run, max_ticks, pause_mgr)
         
        # Always record lifespan
        lifespan_vector[run_id] = worm.ticks 
        if enable_per_run_tracking:
            # Conditionally record full summary metrics
            for idx, (src, tgt) in enumerate(connections_to_track):
                w = get_connection_weight(worm.brain, src, tgt)
                wiring_array[idx][f'weight_final_run_{run_id:04d}'] = w
            
            summary_array[run_id]['run_id'] = run_id
            summary_array[run_id]['lifetime_ticks'] = worm.ticks
            summary_array[run_id]['foods'] = worm.eats
            summary_array[run_id]['distance'] = worm.distance
            summary_array[run_id]['final_energy'] = worm.energy
            summary_array[run_id]['seed_noise'] = seeds_noise_variant[run_id]
            summary_array[run_id]['seed_decision'] = seeds_decision_variant[run_id]
            if enable_per_tick_tracking:
                per_tick_data = rec.per_tick_data[:rec.per_tick_count]
                per_tick_all_runs[run_id] = per_tick_data
            
            if enable_heat_map_tracking:
                heatmaps_all_runs[run_id] = rec.staying_heatmap.copy()
    
    # Build tracking results dict
    tracking_results = {
        'lifespan_vector': lifespan_vector,
        'seeds_noise_all_runs': seeds_noise_variant,
        'seeds_decision_all_runs': seeds_decision_variant
    }
    
    if enable_per_run_tracking:
        tracking_results['summary_array'] = summary_array
        tracking_results['wiring_array'] = wiring_array
        tracking_results['modulation_array'] = modulation_array
        
        if enable_per_tick_tracking:
            tracking_results['per_tick_all_runs'] = per_tick_all_runs
        
        if enable_heat_map_tracking:
            tracking_results['heatmaps_all_runs'] = heatmaps_all_runs
    
    return tracking_results


def run_variant_worker(
    variant_id,
    brain_module_name,
    genome,
    viz_enabled,
    enable_per_run_tracking,
    enable_per_tick_tracking,
    enable_heat_map_tracking,
    max_ticks,
    n_runs,
    viz_fps,
    viz_brain_enabled,
    viz_brain_fps,
    grid_width,
    grid_height,
    start_pos,
    worm_speed,
    worm_energy_capacity,
    worm_metabolic_rate,
    worm_movement_cost,
    sensor_cfg,
    feeding_cfg,
    brain_cfg,
    seeds_noise_variant,
    seeds_decision_variant,
    run_seeds,
    hdf5_path=None,
    hdf5_lock=None,
):
    """Execute a single variant's simulation runs and write data directly to HDF5.
    
    Args:
        variant_id: Index of this variant
        brain_module_name: Name of brain module to import
        genome: Pre-generated genome dict with connection_weights, modulation_spec, etc.
        viz_enabled: Whether to create and display renderer visualization
        enable_per_run_tracking: Whether to track per-run metrics
        enable_per_tick_tracking: Whether to track per-tick data
        enable_heat_map_tracking: Whether to track heatmaps
        max_ticks: Maximum simulation ticks
        n_runs: Number of runs per variant
        viz_fps: World visualization FPS
        viz_brain_enabled: Whether to enable brain visualization
        viz_brain_fps: Brain visualization FPS
        grid_width: World grid width
        grid_height: World grid height
        start_pos: Worm starting position (tuple)
        worm_speed: Worm speed parameter
        worm_energy_capacity: Worm energy capacity
        worm_metabolic_rate: Worm metabolic rate
        worm_movement_cost: Worm movement cost
        sensor_cfg: Sensor configuration list
        feeding_cfg: Feed config dict with keys: feeding_paradigm, initial_fraction_per_cell, regrow_time
        brain_cfg: Dict with brain configuration (n_neurons, threshold, noise_level, sensory_mapping, output_mapping, max_decision_delay)
        seeds_noise_variant: Array of RNG seeds for neuron noise (one per variant)
        seeds_decision_variant: Array of RNG seeds for decision-making (one per variant)
        run_seeds: Array of N_RUNS seeds for world initialization
        hdf5_path: Path to HDF5 file to write to
        hdf5_lock: multiprocessing.Lock() for synchronized writes
    """
    
    # ============================================================
    # Prepare simulation arrays
    # ============================================================
    tracking_results = eval_variant(
        variant_id,
        brain_module_name,
        grid_width,
        grid_height,
        start_pos,
        enable_per_run_tracking,
        enable_per_tick_tracking,
        enable_heat_map_tracking,
        feeding_cfg,
        worm_speed,
        worm_energy_capacity,
        worm_metabolic_rate,
        worm_movement_cost,
        n_runs,
        genome,
        brain_cfg,
        viz_enabled,
        viz_fps,
        viz_brain_enabled,
        viz_brain_fps,
        run_seeds,
        seeds_noise_variant,
        seeds_decision_variant,
        max_ticks,
        sensor_cfg,
    )

    # ============================================================
    # Batch write to HDF5
    # ============================================================
    # Extract tracking results
    lifespan_results = {
        'lifespan_vector': tracking_results['lifespan_vector'],
        'seeds_noise_all_runs': tracking_results['seeds_noise_all_runs'],
        'seeds_decision_all_runs': tracking_results['seeds_decision_all_runs']
    }
    
    # Batch write all variant data after all runs complete (only if per-run tracking enabled)
    if enable_per_run_tracking:
        summary_array = tracking_results['summary_array']
        wiring_array = tracking_results['wiring_array']
        modulation_array = tracking_results['modulation_array']
        
        with hdf5_lock:
            # Write summary array only if per-run tracking is enabled
            if summary_array is not None:
                save_variant_summary_to_hdf5(hdf5_path, variant_id, summary_array)
                save_wiring_to_hdf5(hdf5_path, variant_id, wiring_array)
                save_modulation_to_hdf5(hdf5_path, variant_id, modulation_array)
                
                # Write accumulated per-tick data if any
                if enable_per_tick_tracking:
                    per_tick_all_runs = tracking_results.get('per_tick_all_runs', {})
                    for run_id, per_tick_data in per_tick_all_runs.items():
                        save_per_tick_to_hdf5(hdf5_path, variant_id, run_id, per_tick_data)
                
                # Write accumulated heatmap data if any
                if enable_heat_map_tracking:
                    heatmaps_all_runs = tracking_results.get('heatmaps_all_runs', {})
                    for run_id, staying_heatmap in heatmaps_all_runs.items():
                        save_heatmaps_to_hdf5(hdf5_path, variant_id, run_id, staying_heatmap)
                    
    return (variant_id, lifespan_results)


def eval_generation(genomes, cfg, EXPERIMENT_FOLDER, SIMULATION_NAME, ENABLE_PER_RUN_TRACKING, ENABLE_PER_TICK_TRACKING,
                                ENABLE_HEAT_MAP_TRACKING, VIZ_ENABLED, VIZ_BRAIN_ENABLED, VIZ_FPS, VIZ_BRAIN_FPS, N_VARIANTS,
                                rng_noise, rng_decision, rng_world, brain_module_name, MAX_TICKS, N_RUNS, grid_width, grid_height, start_pos, worm_speed, worm_energy_capacity, worm_metabolic_rate, worm_movement_cost, sensor_cfg, feeding_cfg, brain_cfg, pre_computed_seeds_dict=None, replay_info=None):
    """Execute all variants for a generation and return lifespan data.
    
    Args:
        genomes: List of pre-generated genomes for each variant
        cfg: Full configuration dict
        EXPERIMENT_FOLDER: Output directory for experiment results
        SIMULATION_NAME: Name of the simulation
        ENABLE_PER_RUN_TRACKING: Whether to track per-run metrics
        ENABLE_PER_TICK_TRACKING: Whether to track per-tick data
        ENABLE_HEAT_MAP_TRACKING: Whether to track heatmaps
        VIZ_ENABLED: Whether world visualization is enabled
        VIZ_BRAIN_ENABLED: Whether brain visualization is enabled
        VIZ_FPS: World visualization FPS
        VIZ_BRAIN_FPS: Brain visualization FPS
        N_VARIANTS: Number of variants
        rng_noise: RNG for neuron noise
        rng_decision: RNG for decision-making
        rng_world: RNG for world food initialization
        brain_module_name: Name of brain module
        MAX_TICKS: Maximum simulation ticks
        N_RUNS: Number of runs per variant
        grid_width: World grid width
        grid_height: World grid height
        start_pos: Worm starting position
        worm_speed: Worm speed parameter
        worm_energy_capacity: Worm energy capacity
        worm_metabolic_rate: Worm metabolic rate
        worm_movement_cost: Worm movement cost
        sensor_cfg: Sensor configuration list
        feeding_cfg: Feed config dict with keys: feeding_paradigm, initial_fraction_per_cell, regrow_time
        brain_cfg: Brain configuration dict
        pre_computed_seeds_dict: Optional dict mapping variant_id to seed arrays (for replay mode).
                                If provided, uses these seeds instead of generating new ones.
                                Format: {variant_id: {'run_seeds': array, 'noise_seeds': array, 'decision_seeds': array}}
    
    Returns:
        Tuple of (all_lifespans, run_seeds)
        - all_lifespans: Dict mapping variant_id to lifespan_vector (1D array of lifetime ticks)
        - run_seeds: Array of N_RUNS seeds drawn for world initialization this generation
    """

    # ============================================================
    # ONE-TIME HDF5 INITIALIZATION (if tracking enabled)
    # ============================================================
    
    if ENABLE_PER_RUN_TRACKING:
        # In replay mode, skip timestamp prefix; otherwise add timestamp
        skip_timestamp = replay_info is not None
        hdf5_path = make_experiment_dir(EXPERIMENT_FOLDER, SIMULATION_NAME, skip_timestamp=skip_timestamp)
        print(f"[batch] writing to {hdf5_path}\n")
        create_hdf5_file(hdf5_path, cfg)
        
        # Save the actual tracking flags used (after validation/user input corrections)
        with h5py.File(hdf5_path, 'a') as f:
            f.attrs['tracking_per_run_enabled'] = int(ENABLE_PER_RUN_TRACKING)
            f.attrs['tracking_per_tick_enabled'] = int(ENABLE_PER_TICK_TRACKING)
            f.attrs['tracking_heatmap_enabled'] = int(ENABLE_HEAT_MAP_TRACKING)
        
        # Create manager and lock for parallel HDF5 writing
        manager = Manager()
        hdf5_lock = manager.Lock()


    # ============================================================
    # PREPARE WORKERS and THEN EITHER PARALLEL OR SERIAL
    # ============================================================

    # Determine run_seeds: use pre-computed if in replay mode, otherwise generate fresh
    if pre_computed_seeds_dict is not None:
        # In replay mode, use pre-computed seeds (all variants share same run_seeds)
        run_seeds = pre_computed_seeds_dict[0]['run_seeds']
    else:
        # Normal mode: draw new RNG seeds from rng_world
        run_seeds = rng_world.integers(0, 2**32, size=N_RUNS, dtype=np.uint32)
    
    # Save per-generation data to HDF5 (only if per-run tracking is enabled)
    if ENABLE_PER_RUN_TRACKING:
        # Save genome generation parameters to HDF5
        save_genome_properties_to_hdf5(hdf5_path, genomes, replay_info=replay_info)
        # Save run_seeds for reproducibility
        with h5py.File(hdf5_path, 'a') as f:
            # If elite_genomes group exists, save run_seeds there; otherwise at root level
            if 'elite_genomes' in f:
                f['elite_genomes'].create_dataset('run_seeds', data=run_seeds)
            else:
                f.create_dataset('run_seeds', data=run_seeds)

    try:
        # Run simulation
        all_lifespans = {}
        num_workers = get_num_workers(VIZ_ENABLED, VIZ_BRAIN_ENABLED)
        
        if num_workers is None:
            for variant_id in range(N_VARIANTS):
                print(f"[variant {variant_id+1:02d}/{N_VARIANTS:02d}] Simulating...", end='', flush=True)
                
                # Use pre-computed seeds if in replay mode; otherwise generate fresh seeds
                if pre_computed_seeds_dict is not None:
                    seeds_noise_variant = pre_computed_seeds_dict[variant_id]['noise_seeds']
                    seeds_decision_variant = pre_computed_seeds_dict[variant_id]['decision_seeds']
                else:
                    seeds_noise_variant = rng_noise.integers(0, 2**32, size=N_RUNS, dtype=np.uint32)
                    seeds_decision_variant = rng_decision.integers(0, 2**32, size=N_RUNS, dtype=np.uint32)

                kwargs = {
                    'variant_id': variant_id,
                    'brain_module_name': brain_module_name,
                    'genome': genomes[variant_id],
                    'viz_enabled': VIZ_ENABLED,
                    'enable_per_run_tracking': ENABLE_PER_RUN_TRACKING,
                    'enable_per_tick_tracking': ENABLE_PER_TICK_TRACKING,
                    'enable_heat_map_tracking': ENABLE_HEAT_MAP_TRACKING,
                    'max_ticks': MAX_TICKS,
                    'n_runs': N_RUNS,
                    'viz_fps': VIZ_FPS,
                    'viz_brain_enabled': VIZ_BRAIN_ENABLED,
                    'viz_brain_fps': VIZ_BRAIN_FPS,
                    'grid_width': grid_width,
                    'grid_height': grid_height,
                    'start_pos': start_pos,
                    'worm_speed': worm_speed,
                    'worm_energy_capacity': worm_energy_capacity,
                    'worm_metabolic_rate': worm_metabolic_rate,
                    'worm_movement_cost': worm_movement_cost,
                    'sensor_cfg': sensor_cfg,
                    'feeding_cfg': feeding_cfg,
                    'brain_cfg': brain_cfg,
                    'seeds_noise_variant': seeds_noise_variant,
                    'seeds_decision_variant': seeds_decision_variant,
                    'run_seeds': run_seeds,
                }
                if ENABLE_PER_RUN_TRACKING:
                    kwargs['hdf5_path'] = hdf5_path
                    kwargs['hdf5_lock'] = hdf5_lock
                
                returned_variant_id, lifespan_results = run_variant_worker(**kwargs)

                all_lifespans[variant_id] = lifespan_results
                print(" done")
        
        else:
            completed = 0            
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = set()
                for variant_id in range(N_VARIANTS):

                    # Use pre-computed seeds if in replay mode; otherwise generate fresh seeds
                    if pre_computed_seeds_dict is not None:
                        seeds_noise_variant = pre_computed_seeds_dict[variant_id]['noise_seeds']
                        seeds_decision_variant = pre_computed_seeds_dict[variant_id]['decision_seeds']
                    else:
                        seeds_noise_variant = rng_noise.integers(0, 2**32, size=N_RUNS, dtype=np.uint32)
                        seeds_decision_variant = rng_decision.integers(0, 2**32, size=N_RUNS, dtype=np.uint32)

                    kwargs = {
                        'variant_id': variant_id,
                        'brain_module_name': brain_module_name,
                        'genome': genomes[variant_id],
                        'viz_enabled': False,  # Never viz in parallel (serial only)
                        'enable_per_run_tracking': ENABLE_PER_RUN_TRACKING,
                        'enable_per_tick_tracking': ENABLE_PER_TICK_TRACKING,
                        'enable_heat_map_tracking': ENABLE_HEAT_MAP_TRACKING,
                        'max_ticks': MAX_TICKS,
                        'n_runs': N_RUNS,
                        'viz_fps': VIZ_FPS,
                        'viz_brain_enabled': VIZ_BRAIN_ENABLED,
                        'viz_brain_fps': VIZ_BRAIN_FPS,
                        'grid_width': grid_width,
                        'grid_height': grid_height,
                        'start_pos': start_pos,
                        'worm_speed': worm_speed,
                        'worm_energy_capacity': worm_energy_capacity,
                        'worm_metabolic_rate': worm_metabolic_rate,
                        'worm_movement_cost': worm_movement_cost,
                        'sensor_cfg': sensor_cfg,
                        'feeding_cfg': feeding_cfg,
                        'brain_cfg': brain_cfg,
                        'seeds_noise_variant': seeds_noise_variant,
                        'seeds_decision_variant': seeds_decision_variant,
                        'run_seeds': run_seeds,
                    }
                    if ENABLE_PER_RUN_TRACKING:
                        kwargs['hdf5_path'] = hdf5_path
                        kwargs['hdf5_lock'] = hdf5_lock
                    
                    future = executor.submit(run_variant_worker, **kwargs)
                    futures.add(future)
                
                for future in as_completed(futures):
                    completed += 1
                    returned_variant_id, lifespan_results = future.result()
                    all_lifespans[returned_variant_id] = lifespan_results
                    print(f"\rProcessing variants... ({completed}/{N_VARIANTS} completed)", end='', flush=True)
            
            print()
        # ============================================================
        # WRAP-UP
        # ============================================================
        if ENABLE_PER_RUN_TRACKING:
            print(f"[batch] Simulation completed. Saved to {hdf5_path.name}")
        else:
            print(f"[batch] Simulation completed. (No data recording)")
    
    except PauseManagerExit:
        print("[EXIT] Batch simulation stopped by user.")
    finally:
        from simulate.pause_manager import cleanup_pause_manager
        if VIZ_ENABLED or VIZ_BRAIN_ENABLED:
            cleanup_pause_manager()
    
    # Sort lifespans by variant ID to ensure deterministic ordering regardless of parallel task completion
    all_lifespans = dict(sorted(all_lifespans.items()))
    return all_lifespans, run_seeds
