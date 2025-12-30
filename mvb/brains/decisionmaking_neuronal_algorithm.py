import numpy as np
import importlib
from ..world import World


# ============================================================
# Module-level persistent state
# ============================================================

_brain_state = None


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

        self.incoming = []          # list of Connection objects
        self.activity = 0.0         # activity at beginning of thinking tick
        self.next_activity = 0.0    # activity computed during thinking tick

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
    """
    Transduces symbolic sensory input into neural activity.
    Pretends to be a neuron from the network's point of view.
    """
    def __init__(self, key: str):
        self.key = key
        self.activity = 0.0

    def update(self, inputs: dict):
        self.activity = float(inputs.get(self.key, 0.0))


class Connection:
    def __init__(self, connection_id: int, source, weight: float, reliability: float):
        self.connection_id = connection_id
        self.source = source      # Neuron or InputSource
        self.weight = weight
        self.reliability = reliability

    def propagate(self):
        return self.weight * self.source.activity * self.reliability


# ============================================================
# Brain state (persistent hardware)
# ============================================================

class BrainState:
    """
    Holds all persistent neural hardware.
    Survives across days / steps.
    """
    def __init__(self, neurons, connections, input_sources):
        self.neurons = neurons
        self.connections = connections
        self.input_sources = input_sources


# ============================================================
# Initialization (genotype → phenotype)
# ============================================================

def init(worm, rng, cfg):
    """
    Explicit brain initialization.
    Called once by the runner if present.
    """
    global _brain_state

    # --------------------------------------------------------
    # 1) Read YAML config
    # --------------------------------------------------------
    brain_cfg = cfg["decisionmaking"]["brain"]
    init_name = brain_cfg["init"]

    # --------------------------------------------------------
    # 2) Load brain_init module
    # --------------------------------------------------------
    init_module = importlib.import_module(
        f"configs.brain_init_{init_name}"
    )

    neuron_params, conn_matrix = init_module.build_brain_spec(rng)

    n_neurons = neuron_params.shape[0]

    # --------------------------------------------------------
    # 3) Build neurons
    # --------------------------------------------------------
    neurons = []
    for i in range(n_neurons):
        n = Neuron(
            neuron_id=i,
            threshold=neuron_params[i, 0],
            noise_level=neuron_params[i, 1],
            tonic_level = neuron_params[i,2],
        )
        neurons.append(n)

    # --------------------------------------------------------
    # 4) Build input sources
    # --------------------------------------------------------
    input_keys = [
        "on_food",
        "food_north",
        "food_east",
        "food_south",
        "food_west",
    ]
    input_sources = [InputSource(k) for k in input_keys]


    # --------------------------------------------------------
    # 5) Build connections
    # --------------------------------------------------------
    connections = []
    cid = 0

    for src in range(n_neurons):
        for tgt in range(n_neurons):
            weight = conn_matrix[src, tgt, 0]
            reliability = conn_matrix[src, tgt, 1]

            if weight == 0.0:
                continue

            # source can be input or neuron
            if src < len(input_sources):
                source_obj = input_sources[src]
            else:
                source_obj = neurons[src]

            conn = Connection(
                connection_id=cid,
                source=source_obj,
                weight=weight,
                reliability=reliability,
            )
            neurons[tgt].incoming.append(conn)
            connections.append(conn)
            cid += 1

    # --------------------------------------------------------
    # 6) Assemble BrainState
    # --------------------------------------------------------
    _brain_state = BrainState(
        neurons=neurons,
        connections=connections,
        input_sources=input_sources,
    )


# ============================================================
# Decision process 
# ============================================================

def decide(
    world: World,
    worm,
    rng: np.random.Generator,
    inputs: dict,
):
    """
    Decision-making process.
    Mutates BrainState but does not own it.
    """
    state = _brain_state

    # --- 1) Load sensory inputs ---
    for src in state.input_sources:
        src.update(inputs)

    # --- 2) Thinking loop ---
    max_ticks = 10
    for tick in range(max_ticks):

        # update phase
        for neuron in state.neurons:
            neuron.update(rng)

        # commit phase
        for neuron in state.neurons:
            neuron.commit()

        # visualization hook (later)
        # visualize(state, tick)

        # output check (not implemented yet)
        decision = check_outputs(state, world, worm, rng)
        if decision is not None:
            return decision

    return ("stay",)

# ============================================================
# Interpret output 
# ============================================================


def check_outputs(state: BrainState, world: World, worm, rng):
    """
    Reads output neurons and translates activity into an action.
    Priority:
    1) stay neuron
    2) movement neurons
    """

    # --- 1) Stay has absolute priority ---
    stay_neuron = state.neurons[5]
    if stay_neuron.activity > 0.0:
        return ("stay",)

    # --- 2) Check movement neurons ---
    move_map = {
        6: "north",
        7: "east",
        8: "south",
        9: "west",
    }

    active_moves = [
        nid for nid in move_map
        if state.neurons[nid].activity > 0.0
    ]

    if not active_moves:
        return None  # no output yet

    # choose one movement (random if multiple)
    nid = active_moves[rng.integers(len(active_moves))]
    direction = move_map[nid]

    y, x = worm.y, worm.x
    dy_dx = {
        "north": (-1, 0),
        "south": (1, 0),
        "west":  (0, -1),
        "east":  (0, 1),
    }

    dy, dx = dy_dx[direction]
    ny = (y + dy) % world.height
    nx = (x + dx) % world.width

    return ("move", (ny, nx))

