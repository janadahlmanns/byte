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

    def compute_input(self, rng=None):
        total = self.tonic_level
        for conn in self.incoming:
            total += conn.propagate()
        if self.noise_level > 0.0 and rng is not None:
            total += rng.normal(0.0, self.noise_level)
        return total

    def update(self, rng=None):
        total_input = self.compute_input(rng)
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
    def __init__(self, connection_id: int, source, weight: float, reliability: float):
        self.connection_id = connection_id
        self.source = source
        self.weight = weight
        self.reliability = reliability

    def propagate(self):
        return self.weight * self.source.activity * self.reliability


# ============================================================
# Brain state
# ============================================================

class BrainState:
    def __init__(self, neurons, connections, input_sources, warmup_ticks=0, max_ticks=0, max_decision_delay=2.0):
        self.neurons = neurons
        self.connections = connections
        self.input_sources = input_sources
        self.warmup_ticks = warmup_ticks
        self.max_ticks = max_ticks
        self.max_decision_delay = max_decision_delay


# ============================================================
# Initialization
# ============================================================

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


def init(worm, rng, cfg):
    global _brain_state, _brain_renderer, _last_init_args

    brain_cfg = cfg["decisionmaking"]["brain"]
    init_name = brain_cfg["init"]

    init_module = importlib.import_module(
        f"configs.brain_init_{init_name}"
    )

    neuron_params, conn_matrix, sensory_mapping = init_module.build_brain_spec(rng)
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

            conn = Connection(cid, neurons[src], weight, reliability)
            neurons[tgt].incoming.append(conn)
            connections.append(conn)
            cid += 1

    _brain_state = BrainState(neurons, connections, input_sources)

    # Read max_decision_delay from config, default 2.0
    max_decision_delay = cfg.get("decisionmaking", {}).get("brain", {}).get("max_decision_delay", 2.0)
    _brain_state.max_decision_delay = max_decision_delay

    # Calculate warmup and max ticks based on circuit topology
    warmup_ticks, max_ticks = _calculate_warmup_and_max_ticks(_brain_state)
    _brain_state.warmup_ticks = warmup_ticks
    _brain_state.max_ticks = max_ticks

    # Print actual neuron wiring
    print("\n" + "="*70)
    print("NEURAL CIRCUIT WIRING (Actual connections after initialization)")
    print("="*70)
    for neuron in neurons:
        if neuron.incoming:
            print(f"\nNeuron {neuron.id}:")
            for conn in neuron.incoming:
                source_name = conn.source.key if isinstance(conn.source, InputSource) else f"Neuron {conn.source.id}"
                print(f"  ← {source_name:20s} (weight={conn.weight:+.1f}, reliability={conn.reliability:.2f})")
        else:
            print(f"\nNeuron {neuron.id}: (no incoming connections)")
    print("="*70 + "\n")

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

def decide(world: World, worm, rng, inputs: dict):
    state = _brain_state

    # load sensory inputs
    for src in state.input_sources:
        src.update(inputs)

    # Recalculate warmup and max ticks (topology may have changed with plasticity)
    # This is done at decision time to handle dynamic network changes
    warmup_ticks, max_ticks = _calculate_warmup_and_max_ticks(state)
    
    # Track output history for stability detection
    output_history = []
    final_decision = None
    
    try:
        for tick in range(max_ticks):

            # ---------------- Actual brain beat ----------------
            for neuron in state.neurons:
                neuron.update(rng)
            
            # Get current output state (without converting to decision yet)
            current_output = _get_output_state(state)
            
            # Compute decision once from this output state (to ensure consistency)
            current_decision = _output_to_decision(current_output, state, world, worm, rng)
            
            # Build decision status message for visualization
            if tick < warmup_ticks:
                decision_status = f"PROPAGATION PHASE ({tick+1}/{warmup_ticks})"
            else:
                # Count how many ticks in the history match the current output
                stable_count = sum(1 for h in output_history[-5:] if h == current_output) if output_history else 0
                candidate_str = _format_decision_display(current_decision) if current_decision else "NONE"
                decision_status = f"STABILITY CHECK ({stable_count}/3 stable): {candidate_str}"
            
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
            
            for neuron in state.neurons:
                neuron.commit()

            output_history.append(current_output)
            
            # Keep history to last 5 ticks
            if len(output_history) > 5:
                output_history.pop(0)
            
            # Only check for decision after warmup period
            if tick >= warmup_ticks:
                # Check for stability: same output in 3 out of last 5 ticks
                if len(output_history) >= 3 and _is_output_stable(output_history):
                    if current_decision is not None:
                        return current_decision
            
            # CHECKPOINT 2: At end of each brain tick iteration
            pause_mgr = get_pause_manager()
            pause_mgr.check_pause()
    
    except PauseManagerExit:
        # User exited - return a safe default
        return ("stay",)

    # Fallback: return decision based on final output state
    final_output = _output_to_decision(output_history[-1] if output_history else None, state, world, worm, rng)
    return final_output if final_output is not None else ("stay",)


# ============================================================
# Output state tracking and stability detection
# ============================================================

def _get_output_state(state: BrainState) -> tuple:
    """
    Get the current output state as a tuple of neuron activities (neurons 5-9).
    Used for stability tracking.
    """
    return tuple(state.neurons[i].activity for i in range(5, 10))


def _is_output_stable(output_history: list) -> bool:
    """
    Check if outputs are stable: same state in 3 out of last 5 ticks.
    """
    if len(output_history) < 3:
        return False
    
    # Take last 5 (or fewer if not available yet)
    recent = output_history[-5:] if len(output_history) >= 5 else output_history
    
    # Find the most common output state
    from collections import Counter
    state_counts = Counter(recent)
    
    # If any state appears 3+ times, we have stability
    for state, count in state_counts.items():
        if count >= 3:
            return True
    
    return False


def _output_to_decision(output_state: tuple, state: BrainState, world: World, worm, rng) -> tuple:
    """
    Convert output state tuple to a decision.
    """
    if output_state is None:
        return None
    
    # output_state is (neuron_5, neuron_6, neuron_7, neuron_8, neuron_9)
    stay_neuron_activity = output_state[0]  # neuron 5
    
    if stay_neuron_activity > 0.0:
        return ("stay",)

    move_map = {
        1: "north",   # neuron 6
        2: "east",    # neuron 7
        3: "south",   # neuron 8
        4: "west",    # neuron 9
    }

    active = []
    for idx, direction in move_map.items():
        if output_state[idx] > 0.0:
            active.append((idx, direction))
    
    if not active:
        return None

    # Pick random active movement neuron
    idx, direction = active[rng.integers(len(active))]

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


# ============================================================
# (Old check_outputs function replaced by helper functions above)
# ============================================================
