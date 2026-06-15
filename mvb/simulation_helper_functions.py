"""Helper functions for the simulation API.

Internal utilities used by the simulation API for tracking, brain loading,
and worker process management. Not intended for direct use by other modules.
"""

import os
import importlib
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager

import numpy as np
import h5py

from mvb.world import World
from mvb.worm import Worm
from mvb.world_renderer_qt import QtRenderer
from mvb.brain_renderer_qt import BrainQtRenderer
from simulate.pause_manager import get_pause_manager
from simulate.hdf5_utils import (
    save_variant_summary_to_hdf5,
    save_wiring_to_hdf5,
    save_modulation_to_hdf5,
    save_heatmaps_to_hdf5,
    save_per_tick_to_hdf5,
)


def load_brain_module(version: str):
    """Load brain decision-making module by version name.
    
    Args:
        version: The brain module version (e.g., "plasticity", "random")
    
    Returns:
        The imported brain module
    
    Raises:
        AttributeError: If module does not have a decide() function
    """
    module_name = f"mvb.brains.decisionmaking_{version}"
    module = importlib.import_module(module_name)
    if not hasattr(module, "decide"):
        raise AttributeError(f"{module_name} has no decide()")
    return module


def get_connection_weight(brain_module, src_neuron_id: int, tgt_neuron_id: int) -> float:
    """Get weight of specific neuron-to-neuron connection from brain state.
    
    Args:
        brain_module: The brain module (e.g., decisionmaking_plasticity)
        src_neuron_id: Source neuron ID
        tgt_neuron_id: Target neuron ID
    
    Returns:
        Current weight of the connection, or 0.0 if not found
    """
    if not hasattr(brain_module, '_brain_state'):
        return 0.0
    
    brain_state = brain_module._brain_state
    if brain_state is None:
        return 0.0
    
    # Search through connections for the one from src to tgt
    for conn in brain_state.connections:
        # Check if this is a neuron-to-neuron connection (not input source)
        if hasattr(conn.source, 'id'):
            if conn.source.id == src_neuron_id:
                # Check target by finding which neuron has this in its incoming list
                for neuron in brain_state.neurons:
                    if neuron.id == tgt_neuron_id and conn in neuron.incoming:
                        return conn.weight
    
    return 0.0


def get_num_workers(viz_enabled, viz_brain_enabled):
    """Determine number of worker processes and handle viz/serial execution.
    
    If visualization is enabled, forces serial execution and initializes pause manager.
    Otherwise, reserves 2 cores for system tasks and returns worker count.
    Returns None if system has ≤2 cores or if visualization is enabled (force serial).
    
    Args:
        viz_enabled: Whether world visualization is enabled
        viz_brain_enabled: Whether brain visualization is enabled
    
    Returns:
        Number of workers to use, or None for serial execution
    """
    # Force serial execution if any visualization is enabled
    if viz_enabled or viz_brain_enabled:
        print("[INFO] Visualization enabled. Running serially.")
        from simulate.pause_manager import init_pause_manager
        init_pause_manager()
        return None
    
    # Determine parallel worker count
    try:
        available_cores = os.cpu_count()
        if available_cores is None or available_cores <= 2:
            print("[INFO] Insufficient CPU cores. Running serially.")
            return None
        
        num_workers = max(1, available_cores - 2)
        print(f"[INFO] Parallel execution on {num_workers} cores ({available_cores} total).")
        return num_workers
    except Exception:
        print("[INFO] Insufficient CPU cores. Running serially.")
        return None


def make_experiment_dir(experiment_folder: str, simulation_name: str, skip_timestamp: bool = False) -> Path:
    """Create HDF5 file path for experiment.
    
    Args:
        experiment_folder: Base folder for experiment output
        simulation_name: Name of the simulation
        skip_timestamp: If True, don't prepend a timestamp (used for replay mode)
    
    Returns:
        Path to HDF5 file for saving all results.
    """
    base = Path(experiment_folder)
    base.mkdir(parents=True, exist_ok=True)

    if skip_timestamp:
        hdf5_path = base / f"{simulation_name}.h5"
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        hdf5_path = base / f"{ts}_{simulation_name}.h5"
    
    return hdf5_path


@dataclass
class MetricsRecorder:
    """Track simulation metrics per tick and across runs.
    
    Handles per-tick data logging, connection weight tracking, heatmap/movement tracking,
    and decision-making accuracy metrics during worm simulation.
    """
    per_tick_data: np.ndarray = None
    per_tick_count: int = 0
    connections_to_track: list[tuple] = None
    start_y: int = 0  # Starting Y position for manhattan distance calculation
    start_x: int = 0  # Starting X position for manhattan distance calculation
    prev_y: int = 0
    prev_x: int = 0
    prev_eats: int = 0  # Track food consumption this tick
    prev_action: tuple = None  # Track which movement happened
    grid_height: int = 0  # World grid height for heatmap indexing
    grid_width: int = 0  # World grid width for heatmap indexing
    staying_heatmap: np.ndarray = None  # 2D array (height, width) - field ticks spent
    moves_north: int = 0
    moves_south: int = 0
    moves_east: int = 0
    moves_west: int = 0
    food_sensed_north: int = 0
    food_sensed_east: int = 0
    food_sensed_south: int = 0
    food_sensed_west: int = 0
    prev_on_food: bool = False  # Track if worm was on food before the action
    prev_action_was_decision: bool = False  # Track if previous action was a decision
    decisions: int = 0
    correct_decisions: int = 0
    enable_per_tick_tracking: bool = True  # Whether to track per-tick metrics
    enable_heat_map_tracking: bool = True  # Whether to track heatmaps

    @classmethod
    def empty(cls, genome, start_y, start_x, grid_height, grid_width, enable_per_run_tracking=True, enable_per_tick_tracking=True, enable_heat_map_tracking=True, max_ticks=2000):
        """Initialize recorder with genome and explicit configuration parameters.
        
        If enable_per_run_tracking is False, only lifetime metrics are tracked.
        
        Args:
            genome: Genome dict with connection_weights
            start_y: Starting Y position of worm
            start_x: Starting X position of worm
            grid_height: Height of world grid
            grid_width: Width of world grid
            enable_per_run_tracking: Whether to track per-run metrics
            enable_per_tick_tracking: Whether to track per-tick data
            enable_heat_map_tracking: Whether to track heatmaps
            max_ticks: Maximum simulation ticks
        """
        connection_weights = genome["connection_weights"]
        
        connections_to_track = []
        for src in range(connection_weights.shape[0]):
            for tgt in range(connection_weights.shape[1]):
                if connection_weights[src, tgt, 0] != 0.0:
                    connections_to_track.append((src, tgt))
                
        if enable_per_run_tracking:
            kwargs = {
            'per_tick_count': 0,
            'connections_to_track': connections_to_track,
            'start_y': start_y,
            'start_x': start_x,
            'prev_y': start_y,
            'prev_x': start_x,
            'prev_eats': 0,
            'prev_action': None,
            'grid_height': grid_height,
            'grid_width': grid_width,
            'moves_north': 0,
            'moves_south': 0,
            'moves_east': 0,
            'moves_west': 0,
            'food_sensed_north': 0,
            'food_sensed_east': 0,
            'food_sensed_south': 0,
            'food_sensed_west': 0,
            'prev_on_food': False,
            'prev_action_was_decision': False,
            'decisions': 0,
            'correct_decisions': 0,
            'enable_per_tick_tracking': enable_per_tick_tracking,
            'enable_heat_map_tracking': enable_heat_map_tracking,
            }
            if enable_per_tick_tracking:
                dtype_fields = [
                    ('tick', 'i4'),
                    ('food_sensed_N', 'u1'),
                    ('food_sensed_E', 'u1'),
                    ('food_sensed_S', 'u1'),
                    ('food_sensed_W', 'u1'),
                    ('movement', 'S4'),
                    ('food_consumed', 'u1'),
                    ('energy', 'f4'),
                    ('manhattan_dist', 'u2'),
                    ('decision_made', 'u1'),
                ]
                for src, tgt in connections_to_track:
                    dtype_fields.append((f'{src}_{tgt}', 'f4'))
                kwargs['per_tick_data'] = np.zeros(max_ticks + 1, dtype=dtype_fields)
            
            if enable_heat_map_tracking:
                staying_heatmap = np.zeros((grid_height, grid_width), dtype=np.int32)
                kwargs['staying_heatmap'] = staying_heatmap
            return cls(**kwargs)
        else:
            return cls()        
        
    def reset(self):
        """Reset all counters and state for a new run while keeping arrays allocated."""
        self.per_tick_count = 0
        self.prev_y = self.start_y
        self.prev_x = self.start_x
        self.prev_eats = 0
        self.prev_action = None
        self.moves_north = 0
        self.moves_south = 0
        self.moves_east = 0
        self.moves_west = 0
        self.food_sensed_north = 0
        self.food_sensed_east = 0
        self.food_sensed_south = 0
        self.food_sensed_west = 0
        self.prev_on_food = False
        self.prev_action_was_decision = False
        self.decisions = 0
        self.correct_decisions = 0
        # Clear array data if allocated
        if self.per_tick_data is not None:
            self.per_tick_data.fill(0)
        if self.staying_heatmap is not None:
            self.staying_heatmap.fill(0)

    def record(self, worm: Worm):
        """Record metrics for this tick."""
        dy = worm.y - self.prev_y
        dx = worm.x - self.prev_x
        
        if dy > 0:
            self.moves_south += 1
        elif dy < 0:
            self.moves_north += 1
        
        if dx > 0:
            self.moves_east += 1
        elif dx < 0:
            self.moves_west += 1
        
        # Check if food was sensed in any direction (regardless of other directions)
        sense = getattr(worm, "sensory_information", {})
        food_north = sense.get("food_north", 0.0) > 0.0
        food_east = sense.get("food_east", 0.0) > 0.0
        food_south = sense.get("food_south", 0.0) > 0.0
        food_west = sense.get("food_west", 0.0) > 0.0
        
        # Count food sensing occurrences
        if food_north:
            self.food_sensed_north += 1
        if food_east:
            self.food_sensed_east += 1
        if food_south:
            self.food_sensed_south += 1
        if food_west:
            self.food_sensed_west += 1
        
        # Track decision-making accuracy using new rules:
        # Rule 1: Stay is a decision if worm is NOT on food CURRENTLY (before the stay action)
        # Rule 2: Movement is a decision if ANY food is sensed on the 5 current sensing fields
        # Rule 3: Decision is correct if it was deemed a decision AND worm is on food in NEXT tick
        on_food = sense.get("on_food", 0) > 0
        any_food_sensed = food_north or food_east or food_south or food_west
        
        # Check if the previous action was correct (in this tick after previous action)
        if self.prev_action_was_decision and on_food:
            self.correct_decisions += 1
        
        # Determine if current action (about to happen) is a decision
        # We use prev_on_food because the decision is made BEFORE the action
        stayed = (dy == 0 and dx == 0)
        
        is_decision = False
        if stayed:
            # Stay is a decision only if worm is NOT on food CURRENTLY (before the stay)
            is_decision = not self.prev_on_food
        else:
            # Movement is a decision if ANY food is sensed on the 5 current sensing fields
            is_decision = any_food_sensed
        
        if is_decision:
            self.decisions += 1
        
        # Update state for next tick
        self.prev_on_food = on_food
        self.prev_action_was_decision = is_decision
        
        # Track comprehensive per-tick data (if enabled)
        if self.enable_per_tick_tracking and self.per_tick_data is not None:
            # Determine movement direction from previous action
            movement_str = "stay"
            if self.prev_action is not None:
                if self.prev_action[0] == "move":
                    move_y, move_x = self.prev_action[1]
                    if move_y < self.prev_y:
                        movement_str = "N"
                    elif move_y > self.prev_y:
                        movement_str = "S"
                    elif move_x > self.prev_x:
                        movement_str = "E"
                    elif move_x < self.prev_x:
                        movement_str = "W"
            
            # Check if food was consumed this tick
            food_consumed = 1 if worm.eats > self.prev_eats else 0
            
            # Calculate manhattan distance from start position
            manhattan_dist = abs(worm.y - self.start_y) + abs(worm.x - self.start_x)
            
            # Check if decision was made (action is not None)
            decision_made = 1 if worm.action is not None else 0
            
            # Populate array at current tick index
            tick_idx = worm.ticks
            self.per_tick_data[tick_idx] = (
                worm.ticks,
                int(food_north),
                int(food_east),
                int(food_south),
                int(food_west),
                movement_str,
                food_consumed,
                worm.energy,
                manhattan_dist,
                decision_made,
            ) + tuple(get_connection_weight(worm.brain, src, tgt) for src, tgt in self.connections_to_track)
            
            self.per_tick_count = tick_idx + 1
        
        if self.enable_heat_map_tracking and self.staying_heatmap is not None:
            self.staying_heatmap[worm.y, worm.x] += 1
        
        self.prev_y = worm.y
        self.prev_x = worm.x
        self.prev_eats = worm.eats
        self.prev_action = worm.action
