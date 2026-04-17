import numpy as np
import importlib
from ..world import World

try:
    from simulate.pause_manager import get_pause_manager, PauseManagerExit
except ImportError:
    # Fallback: create a dummy pause manager that never pauses
    class DummyPauseManager:
        def check_pause(self):
            pass  # Do nothing
        def should_exit(self):
            return False
    
    class PauseManagerExit(Exception):
        pass
    
    _dummy_pm = DummyPauseManager()
    def get_pause_manager():
        return _dummy_pm


# ============================================================
# Module-level persistent state
# ============================================================

_brain_state = None


# ============================================================
# Optional brain visualization (Qt)
# ============================================================

_brain_renderer = None

# ============================================================
# Hardware primitives
# ============================================================

class Neuron:
    def __init__(
        self,
        neuron_id: int,
        threshold: float = 0.5,
        noise_level: float = 0.0,
        tonic_level: float = 0.0,
    ):
        self.id = neuron_id
        self.threshold = threshold
        self.noise_level = noise_level
        self.tonic_level = tonic_level

        self.incoming = []
        self.activity = 0.0
        self.next_activity = 0.0

    def compute_input(self, rng_neuron_noise=None):
        total = self.tonic_level
        for conn in self.incoming:
            total += conn.propagate()
        if self.noise_level > 0.0 and rng_neuron_noise is not None:
            total += rng_neuron_noise.normal(0.0, self.noise_level)
        return total

    def update(self, rng_neuron_noise=None):
        total_input = self.compute_input(rng_neuron_noise)
        self.next_activity = 1.0 if total_input >= self.threshold else 0.0

    def commit(self):
        self.activity = self.next_activity


class InputSource:
    def __init__(self, key: str):
        self.key = key
        self.activity = 0.0

    def update(self, inputs: dict):
        self.activity = float(inputs.get(self.key, 0.0))


class Connection:
    def __init__(self, connection_id: int, source, weight: float, reliability: float, modulating_inputs=None):
        self.connection_id = connection_id
        self.source = source
        self.weight = weight
        self.reliability = reliability
        self.modulating_inputs = modulating_inputs if modulating_inputs is not None else []
        self.next_weight = weight

    def propagate(self):
        return self.weight * self.source.activity * self.reliability
    
    def update(self, eta):
        """Compute next weight based on modulating inputs.
        
        Uses plasticity rule K: w_new = sign(w)·max(0, |w|+η|w|(1-|w|)·modsum)
        
        Optimized for efficiency with early exits to avoid expensive operations.
        
        Parameters
        ----------
        eta : float
            Global plasticity factor (amplitude of weight changes)
        """
        # Check eta first (cheapest)
        if eta == 0.0:
            self.next_weight = self.weight
            return
        
        # Check if weight is zero (very cheap, avoids expensive modulation_sum calculation)
        # If w==0, then magnitude = 0 + 0 = 0, so result is always 0
        if self.weight == 0:
            self.next_weight = 0.0
            return
        
        # Check modulators
        if not self.modulating_inputs:
            self.next_weight = self.weight
            return
        
        # Calculate modulation (most expensive operation—now only done if w != 0)
        modulation_sum = sum(mod_weight * neuron.activity 
                            for neuron, mod_weight in self.modulating_inputs)
        
        if modulation_sum == 0.0:
            self.next_weight = self.weight
            return
        
        # Formula K: w_new = sign(w)·max(0, |w|+η|w|(1-|w|)·modsum)
        sign = np.sign(self.weight)
        abs_w = abs(self.weight)
        magnitude = abs_w + eta * abs_w * (1.0 - abs_w) * modulation_sum
        self.next_weight = sign * max(0.0, magnitude)
    
    def commit(self):
        """Apply the computed weight change."""
        self.weight = self.next_weight


# ============================================================
# Brain state
# ============================================================

class BrainState:
    def __init__(self, neurons, connections, input_sources, warmup_ticks=0, max_ticks=0, max_decision_delay=2.0, eta=0.0, output_mapping=None):
        self.neurons = neurons
        self.connections = connections
        self.input_sources = input_sources
        self.warmup_ticks = warmup_ticks
        self.max_ticks = max_ticks
        self.max_decision_delay = max_decision_delay
        self.eta = eta  # Global plasticity factor
        self.output_mapping = output_mapping if output_mapping is not None else {}  # Maps neuron_id to action_name


# ============================================================
# Initialization
# ============================================================

def init_brain(genome, yaml_config, rng_neuron_noise):
    """
    Initialize brain from genome and YAML configuration.
    
    This is the new clean interface replacing the old init() function.
    
    Parameters
    ----------
    genome : dict
        Dictionary with keys:
        - 'connection_weights': np.ndarray (n_neurons, n_neurons, 2)
        - 'modulation_spec': dict
        - 'tonic_activations': np.ndarray (n_neurons,)
        - 'eta': float
    
    yaml_config : dict
        Configuration with 'brain' section containing:
        - 'n_neurons': int
        - 'threshold': float (applied to all neurons)
        - 'noise_level': float (applied to all neurons)
        - 'sensory_mapping': dict
        - 'output_mapping': dict (neuron_id → action_name)
        - 'max_decision_delay': float
    
    rng_neuron_noise : np.random.Generator
        RNG for neuron noise
    
    Returns
    -------
    None (modifies module-level _brain_state)
    """
    global _brain_state, _brain_renderer
    
    brain_cfg = yaml_config.get("brain", {})
    n_neurons = brain_cfg.get("n_neurons", 10)
    default_threshold = brain_cfg.get("threshold", 0.5)
    default_noise_level = brain_cfg.get("noise_level", 0.05)
    sensory_mapping = brain_cfg.get("sensory_mapping", {})
    output_mapping = brain_cfg.get("output_mapping", {})
    max_decision_delay = brain_cfg.get("max_decision_delay", 2.0)
    
    # Extract genome components
    connection_weights = genome["connection_weights"]
    modulation_spec = genome["modulation_spec"]
    tonic_activations = genome["tonic_activations"]
    eta = genome["eta"]
    
    # Create neurons with properties from YAML (all neurons get same threshold and noise_level)
    neurons = []
    for i in range(n_neurons):
        threshold = default_threshold
        noise_level = default_noise_level
        tonic_level = float(tonic_activations[i])
        
        neuron = Neuron(
            neuron_id=i,
            threshold=threshold,
            noise_level=noise_level,
            tonic_level=tonic_level,
        )
        neurons.append(neuron)
    
    # Create input sources for sensory inputs
    input_keys = list(sensory_mapping.keys())
    input_sources = [InputSource(k) for k in input_keys]
    
    # Wire sensory inputs to neurons
    connections = []
    cid = 0
    for sense_key, (target_neuron_id, weight, reliability) in sensory_mapping.items():
        # Find the InputSource with this key
        source_obj = None
        for inp in input_sources:
            if inp.key == sense_key:
                source_obj = inp
                break
        
        if source_obj is not None:
            conn = Connection(cid, source_obj, weight, reliability)
            neurons[target_neuron_id].incoming.append(conn)
            connections.append(conn)
            cid += 1
    
    # Wire neurons to neurons from genome connection weights
    for src in range(n_neurons):
        for tgt in range(n_neurons):
            weight, reliability = connection_weights[src, tgt]
            if weight == 0.0:
                continue
            
            # Check if this connection has modulators
            modulating_inputs = None
            if (src, tgt) in modulation_spec:
                modulating_inputs = [(neurons[mod_id], mod_weight) 
                                    for mod_id, mod_weight in modulation_spec[(src, tgt)]]
            
            conn = Connection(cid, neurons[src], weight, reliability, modulating_inputs)
            neurons[tgt].incoming.append(conn)
            connections.append(conn)
            cid += 1
    
    # Create brain state
    _brain_state = BrainState(neurons, connections, input_sources, eta=eta, output_mapping=output_mapping)
    _brain_state.max_decision_delay = max_decision_delay
    
    # Calculate warmup and max ticks
    warmup_ticks, max_ticks = _calculate_warmup_and_max_ticks(_brain_state)
    _brain_state.warmup_ticks = warmup_ticks
    _brain_state.max_ticks = max_ticks
    
    # Store RNG for neuron noise
    _brain_state.rng_neuron_noise = rng_neuron_noise


def _calculate_warmup_and_max_ticks(state: BrainState) -> tuple:
    """
    Calculate warmup period and max ticks based on current circuit topology.
    Warmup = number of active neurons (neurons with incoming or outgoing connections).
    Max ticks = warmup * max_decision_delay (from state).
    """
    active_neurons = set()
    for neuron in state.neurons:
        if neuron.incoming:
            active_neurons.add(neuron.id)
    for conn in state.connections:
        if isinstance(conn.source, Neuron):
            active_neurons.add(conn.source.id)
    
    warmup_ticks = len(active_neurons)
    max_ticks = int(warmup_ticks * state.max_decision_delay)
    
    return warmup_ticks, max_ticks


def init(worm, cfg, rng_neuron_noise, brain_init_spec=None):
    global _brain_state, _brain_renderer, _last_init_args

    # Load brain spec (either from parameter or from config)
    eta = 0.0  # Default: no plasticity
    modulator_spec = {}  # Default: no modulators
    
    if brain_init_spec is not None:
        # brain_init_spec can be:
        # - 3-tuple: (neuron_params, conn_matrix, sensory_mapping)
        # - 4-tuple: (neuron_params, conn_matrix, sensory_mapping, max_decision_delay)
        # - 5-tuple: (neuron_params, conn_matrix, sensory_mapping, max_decision_delay, eta)
        # - 6-tuple: (neuron_params, conn_matrix, sensory_mapping, max_decision_delay, eta, modulator_spec)
        if len(brain_init_spec) == 6:
            neuron_params, conn_matrix, sensory_mapping, max_decision_delay, eta, modulator_spec = brain_init_spec
        elif len(brain_init_spec) == 5:
            neuron_params, conn_matrix, sensory_mapping, max_decision_delay, eta = brain_init_spec
        elif len(brain_init_spec) == 4:
            neuron_params, conn_matrix, sensory_mapping, max_decision_delay = brain_init_spec
        else:
            # Backwards compatibility: if only 3 values, use default max_decision_delay
            neuron_params, conn_matrix, sensory_mapping = brain_init_spec
            max_decision_delay = 2.0
    else:
        # Load from config (backwards compatibility)
        brain_cfg = cfg["decisionmaking"]["brain"]
        init_name = brain_cfg["init"]

        init_module = importlib.import_module(
            f"configs.brain_init_{init_name}"
        )

        spec = init_module.build_brain_spec()
        if len(spec) == 6:
            neuron_params, conn_matrix, sensory_mapping, max_decision_delay, eta, modulator_spec = spec
        elif len(spec) == 5:
            neuron_params, conn_matrix, sensory_mapping, max_decision_delay, eta = spec
        elif len(spec) == 4:
            neuron_params, conn_matrix, sensory_mapping, max_decision_delay = spec
        else:
            # Backwards compatibility
            neuron_params, conn_matrix, sensory_mapping = spec
            max_decision_delay = 2.0

    n_neurons = neuron_params.shape[0]

    neurons = [
        Neuron(
            neuron_id=i,
            threshold=neuron_params[i, 0],
            noise_level=neuron_params[i, 1],
            tonic_level=neuron_params[i, 2],
        )
        for i in range(n_neurons)
    ]

    input_keys = [
        "on_food",
        "food_north",
        "food_east",
        "food_south",
        "food_west",
    ]
    input_sources = [InputSource(k) for k in input_keys]

    # ---------------------------------
    # Wire sensory inputs to neurons
    # ---------------------------------
    connections = []
    cid = 0
    for sense_key, (target_neuron_id, weight, reliability) in sensory_mapping.items():
        # Find the InputSource with this key
        source_obj = None
        for inp in input_sources:
            if inp.key == sense_key:
                source_obj = inp
                break
        
        if source_obj is not None:
            conn = Connection(cid, source_obj, weight, reliability)
            neurons[target_neuron_id].incoming.append(conn)
            connections.append(conn)
            cid += 1

    # ---------------------------------
    # Wire neurons to neurons (from connection matrix)
    # ---------------------------------
    for src in range(n_neurons):
        for tgt in range(n_neurons):
            weight, reliability = conn_matrix[src, tgt]
            if weight == 0.0:
                continue

            # Check if this connection has modulators (from brain_init modulator_spec)
            modulating_inputs = None
            if (src, tgt) in modulator_spec:
                # Build modulating_inputs list: convert neuron IDs to neuron objects
                modulating_inputs = [(neurons[mod_id], mod_weight) 
                                    for mod_id, mod_weight in modulator_spec[(src, tgt)]]

            conn = Connection(cid, neurons[src], weight, reliability, modulating_inputs)
            neurons[tgt].incoming.append(conn)
            connections.append(conn)
            cid += 1

    # Build default output_mapping if not provided (for backwards compatibility)
    default_output_mapping = {
        5: "stay",
        6: "move_north",
        7: "move_east",
        8: "move_south",
        9: "move_west",
    }
    
    _brain_state = BrainState(neurons, connections, input_sources, eta=eta, output_mapping=default_output_mapping)

    # Use max_decision_delay from brain_init_spec (already set above)
    _brain_state.max_decision_delay = max_decision_delay

    # Calculate warmup and max ticks based on circuit topology
    warmup_ticks, max_ticks = _calculate_warmup_and_max_ticks(_brain_state)
    _brain_state.warmup_ticks = warmup_ticks
    _brain_state.max_ticks = max_ticks
    
    # Store the neuron noise RNG stream
    _brain_state.rng_neuron_noise = rng_neuron_noise

    # DEBUG OUTPUT (commented out - uncomment to see circuit details)
    # print("\n" + "="*70)
    # print("NEURAL CIRCUIT WIRING (Actual connections after initialization)")
    # print("="*70)
    # for neuron in neurons:
    #     if neuron.incoming:
    #         print(f"\nNeuron {neuron.id}:")
    #         for conn in neuron.incoming:
    #             source_name = conn.source.key if isinstance(conn.source, InputSource) else f"Neuron {conn.source.id}"
    #             print(f"  ← {source_name:20s} (weight={conn.weight:+.1f}, reliability={conn.reliability:.2f})")
    #     else:
    #         print(f"\nNeuron {neuron.id}: (no incoming connections)")
    # print("="*70 + "\n")

    viz_cfg = cfg.get("viz", {})
    if viz_cfg.get("brain_enabled", False):
        from mvb.brain_renderer_qt import BrainQtRenderer
        if _brain_renderer is None:
            _brain_renderer = BrainQtRenderer(
                fps=int(viz_cfg.get("brain_fps", 30))
            )


# ============================================================
# Decision process
# ============================================================

def decide(world: World, worm, rng_decision, inputs: dict):
    state = _brain_state

    # load sensory inputs
    for src in state.input_sources:
        src.update(inputs)

    # Recalculate warmup and max ticks (topology may have changed with plasticity)
    # This is done at decision time to handle dynamic network changes
    warmup_ticks, max_ticks = _calculate_warmup_and_max_ticks(state)
    
    # Track output history during propagation phase
    propagation_history = []
    # Track candidate neurons identified at end of propagation
    candidate_neurons = None
    # Track stability phase history
    stability_history = []
    
    try:
        for tick in range(max_ticks):

            # ---------------- Actual brain beat ----------------
            # 1. All neurons compute next activity based on current state
            for neuron in state.neurons:
                neuron.update(state.rng_neuron_noise)
            
            # 2. All connections compute next weight based on current neuron activities
            for conn in state.connections:
                conn.update(state.eta)
            
            # Get current output state
            current_output = _get_output_state(state)
            
            # Build decision status message for visualization
            if tick < warmup_ticks:
                decision_status = f"PROPAGATION PHASE ({tick+1}/{warmup_ticks})"
            else:
                # Identify candidates at start of stability phase
                if candidate_neurons is None:
                    candidate_neurons = _get_candidate_neurons(propagation_history)
                
                # Display candidate neurons
                candidate_str = ",".join(str(n) for n in sorted(candidate_neurons)) if candidate_neurons else "NONE"
                stability_tick = tick - warmup_ticks + 1
                decision_status = f"STABILITY PHASE ({stability_tick}): candidates={candidate_str}"
            
            # ---------------- Single visualization per tick (after update, before commit) ----------------
            if _brain_renderer is not None:
                sense = getattr(worm, "sensory_information", None) or inputs or {}
                _brain_renderer.draw(
                    state,
                    brain_tick=tick,
                    decision_status=decision_status,
                    sense=sense,
                )
                _brain_renderer.wait_frame()
            
            # 3. Commit all updates simultaneously
            for neuron in state.neurons:
                neuron.commit()
            
            for conn in state.connections:
                conn.commit()

            # Store history in appropriate phase
            if tick < warmup_ticks:
                propagation_history.append(current_output)
                # Keep only last 5 ticks of propagation
                if len(propagation_history) > 5:
                    propagation_history.pop(0)
            else:
                stability_history.append(current_output)
            
            # Check for decision during stability phase
            if tick >= warmup_ticks and candidate_neurons:
                # Check if candidates are stably active during stability phase
                stable_neurons = _check_candidate_stability(candidate_neurons, stability_history)
                if stable_neurons:
                    stable_decision = _stable_outputs_to_decision(stable_neurons, state, world, worm, rng_decision)
                    if stable_decision is not None:
                        return stable_decision
            
            # CHECKPOINT 2: At end of each brain tick iteration
            try:
                pause_mgr = get_pause_manager()
                pause_mgr.check_pause()
            except RuntimeError:
                # Pause manager not initialized (visualization disabled)
                pass
    
    except PauseManagerExit:
        # User exited - re-raise to propagate to simulation loop
        raise

    # Fallback: check if candidates remained stable in stability phase
    if candidate_neurons and stability_history:
        stable_neurons = _check_candidate_stability(candidate_neurons, stability_history)
        if stable_neurons:
            stable_decision = _stable_outputs_to_decision(stable_neurons, state, world, worm, rng_decision)
            if stable_decision is not None:
                return stable_decision
    
    # No decision reached within max_ticks - pick a random movement to avoid getting stuck
    return _get_random_decision(state, world, worm, rng_decision)


# ============================================================
# Output state tracking and stability detection
# ============================================================

def _get_output_state(state: BrainState) -> tuple:
    """
    Get the current output state as a tuple of neuron activities (neurons 5-9).
    Used for stability tracking.
    """
    return tuple(state.neurons[i].activity for i in range(5, 10))


def _get_candidate_neurons(propagation_history: list) -> set:
    """
    Identify candidate output neurons from propagation phase.
    A neuron is a candidate if it was active in 3+ of the last 5 propagation ticks.
    Returns a set of neuron IDs (5-9) that are candidates.
    """
    if len(propagation_history) < 3:
        return set()
    
    # Take last 5 (or fewer if not available)
    recent = propagation_history[-5:] if len(propagation_history) >= 5 else propagation_history
    
    # Count how many times each neuron was active
    neuron_counts = [0] * 5  # neurons 5-9 (indices 0-4 in output tuple)
    for output_state in recent:
        for i in range(5):
            if output_state[i] > 0.0:
                neuron_counts[i] += 1
    
    # A neuron is a candidate if active in 3+ ticks
    candidates = set()
    for i in range(5):
        if neuron_counts[i] >= 3:
            candidates.add(i + 5)  # Convert to actual neuron IDs (5-9)
    
    return candidates


def _check_candidate_stability(candidate_neurons: set, stability_history: list) -> set:
    """
    Check which candidate neurons remain stably active during stability phase.
    A candidate is deemed stable if it's active in the majority of stability phase ticks.
    Returns a set of neuron IDs that are stably active.
    """
    if not candidate_neurons or len(stability_history) < 3:
        return set()
    
    # Count how many times each candidate was active during stability phase
    neuron_counts = {neuron_id: 0 for neuron_id in candidate_neurons}
    
    for output_state in stability_history:
        for neuron_id in candidate_neurons:
            idx = neuron_id - 5  # Convert neuron ID to index in output tuple
            if output_state[idx] > 0.0:
                neuron_counts[neuron_id] += 1
    
    # A candidate is stable if active in majority (>50%) of stability ticks
    stability_threshold = len(stability_history) / 2.0
    stable = set()
    for neuron_id, count in neuron_counts.items():
        if count > stability_threshold:
            stable.add(neuron_id)
    
    return stable


def _stable_outputs_to_decision(stable_neurons: set, state: BrainState, world: World, worm, rng_decision) -> tuple:
    """
    Convert stable output neurons to a decision using the output_mapping from yaml.
    
    Priority order:
    1. If a neuron with "stay" action is stable → stay (regardless of other outputs)
    2. Otherwise → randomly choose from stable movement neurons
    
    Parameters
    ----------
    stable_neurons : set
        Set of neuron IDs that were:
        1. Active in 3+ of last 5 propagation ticks (candidates)
        2. Active in majority of stability phase ticks (confirmed stable)
    
    state : BrainState
        Brain state containing output_mapping (neuron_id → action_name)
    """
    if not stable_neurons:
        return None
    
    # Get output mapping from brain state
    output_mapping = state.output_mapping
    if not output_mapping:
        return None
    
    # Priority 1: Check for "stay" action among stable neurons
    for neuron_id in stable_neurons:
        action = output_mapping.get(str(neuron_id), None) or output_mapping.get(neuron_id, None)
        if action == "stay":
            return ("stay",)
    
    # Priority 2: Collect all stable movement neurons
    active_movements = []
    direction_map = {
        "move_north": "north",
        "move_south": "south",
        "move_east": "east",
        "move_west": "west",
    }
    
    for neuron_id in stable_neurons:
        action = output_mapping.get(str(neuron_id), None) or output_mapping.get(neuron_id, None)
        if action and action.startswith("move_"):
            direction = direction_map.get(action)
            if direction:
                active_movements.append(direction)
    
    if not active_movements:
        return None
    
    # Pick random stable movement
    direction = active_movements[rng_decision.integers(len(active_movements))]
    
    y, x = worm.y, worm.x
    dy, dx = {
        "north": (-1, 0),
        "south": (1, 0),
        "west": (0, -1),
        "east": (0, 1),
    }[direction]
    
    ny = (y + dy) % world.height
    nx = (x + dx) % world.width
    
    return ("move", (ny, nx), direction)


def _get_random_decision(state: BrainState, world: World, worm, rng_decision) -> tuple:
    """
    Generate a random decision (stay or move in random direction).
    Used as fallback when brain hits max_ticks without deciding.
    """
    choices = ["stay", "north", "east", "south", "west"]
    choice = choices[rng_decision.integers(len(choices))]
    
    if choice == "stay":
        return ("stay",)
    
    # Random movement
    y, x = worm.y, worm.x
    dy, dx = {
        "north": (-1, 0),
        "south": (1, 0),
        "west": (0, -1),
        "east": (0, 1),
    }[choice]
    
    ny = (y + dy) % world.height
    nx = (x + dx) % world.width
    
    return ("move", (ny, nx), choice)


# ============================================================
# Helper: Format decision for display
# ============================================================

def _format_decision_display(decision: tuple) -> str:
    """
    Convert decision tuple to human-readable cardinal direction.
    
    - ("stay",) -> "STAY"
    - ("move", (ny, nx), direction) -> "MOVE N/S/E/W"
    """
    if decision[0] == "stay":
        return "STAY"
    
    if decision[0] == "move" and len(decision) > 2:
        direction = decision[2]
        direction_map = {
            "north": "N",
            "south": "S",
            "east": "E",
            "west": "W",
        }
        return f"MOVE {direction_map.get(direction, '?')}"
    
    return str(decision)


