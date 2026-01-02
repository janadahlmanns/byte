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
    def __init__(self, neurons, connections, input_sources):
        self.neurons = neurons
        self.connections = connections
        self.input_sources = input_sources


# ============================================================
# Initialization
# ============================================================

def init(worm, rng, cfg):
    global _brain_state, _brain_renderer, _last_init_args

    brain_cfg = cfg["decisionmaking"]["brain"]
    init_name = brain_cfg["init"]

    init_module = importlib.import_module(
        f"configs.brain_init_{init_name}"
    )

    neuron_params, conn_matrix = init_module.build_brain_spec(rng)
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

    connections = []
    cid = 0
    for src in range(n_neurons):
        for tgt in range(n_neurons):
            weight, reliability = conn_matrix[src, tgt]
            if weight == 0.0:
                continue

            source_obj = (
                input_sources[src] if src < len(input_sources) else neurons[src]
            )

            conn = Connection(cid, source_obj, weight, reliability)
            neurons[tgt].incoming.append(conn)
            connections.append(conn)
            cid += 1

    _brain_state = BrainState(neurons, connections, input_sources)

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

    max_ticks = 10
    try:
        for tick in range(max_ticks):

            # ---------------- Visualization ----------------
            if _brain_renderer is not None:
                sense = getattr(worm, "sensory_information", None) or inputs or {}
                _brain_renderer.draw(
                    state,
                    brain_tick=tick,
                    decision_status=f"THINKING ({tick+1}/{max_ticks})",
                    sense=sense,
                )
            _brain_renderer.wait_frame()

            # ---------------- Actual brain beat ----------------
            for neuron in state.neurons:
                neuron.update(rng)
            for neuron in state.neurons:
                neuron.commit()

            decision = check_outputs(state, world, worm, rng)
            if decision is not None:
                if _brain_renderer is not None:
                    _brain_renderer.draw(
                        state,
                        brain_tick=tick,
                        decision_status=f"DECISION MADE: {decision}",
                        sense=sense,
                    )
                return decision
            
            # CHECKPOINT 2: At end of each brain tick iteration
            pause_mgr = get_pause_manager()
            pause_mgr.check_pause()
    
    except PauseManagerExit:
        # User exited - return a safe default
        return ("stay",)

    return ("stay",)


# ============================================================
# Interpret outputs
# ============================================================

def check_outputs(state: BrainState, world: World, worm, rng):
    stay_neuron = state.neurons[5]
    if stay_neuron.activity > 0.0:
        return ("stay",)

    move_map = {
        6: "north",
        7: "east",
        8: "south",
        9: "west",
    }

    active = [nid for nid in move_map if state.neurons[nid].activity > 0.0]
    if not active:
        return None

    nid = active[rng.integers(len(active))]
    direction = move_map[nid]

    y, x = worm.y, worm.x
    dy, dx = {
        "north": (-1, 0),
        "south": (1, 0),
        "west": (0, -1),
        "east": (0, 1),
    }[direction]

    ny = (y + dy) % world.height
    nx = (x + dx) % world.width

    return ("move", (ny, nx))
