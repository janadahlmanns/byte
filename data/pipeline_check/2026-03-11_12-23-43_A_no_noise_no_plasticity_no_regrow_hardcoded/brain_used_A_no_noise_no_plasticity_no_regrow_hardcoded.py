import numpy as np


def build_brain_spec():
    """
    Builds the initial brain specification (genotype) WITH PLASTICITY.

    Returns
    -------
    neuron_params : np.ndarray, shape (11, 3)
        Column 0: threshold
        Column 1: noise level
        Column 2: tonic level

    connections : np.ndarray, shape (11, 11, 2)
        [:, :, 0] = weight
        [:, :, 1] = reliability
        Rows = source neuron
        Columns = target neuron
    
    sensory_mapping : dict
        Maps sensory input keys to (target_neuron_id, weight, reliability)
    
    max_decision_delay : float
        Maximum decision delay in ticks for the neural circuit
    
    eta : float
        Global plasticity factor (amplitude of weight changes)
    """

    n_neurons = 11
    max_decision_delay = 2.0  # in ticks

    # ---------------------------------
    # sensory input mapping
    # ---------------------------------
    # Specifies which sensory information connects to which neuron
    sensory_mapping = {
        "on_food": (0, 1.0, 1.0),       # sensory key -> (neuron_id, weight, reliability)
        "food_north": (1, 1.0, 1.0),
        "food_east": (2, 1.0, 1.0),
        "food_south": (3, 1.0, 1.0),
        "food_west": (4, 1.0, 1.0),
    }

    # ---------------------------------
    # neuron parameters
    # ---------------------------------
    # column 0: threshold
    # column 1: noise level
    # column 2: tonic
    neuron_params = np.zeros((n_neurons, 3), dtype=float)
    neuron_params[:, 0] = 0.5  # threshold
    neuron_params[:, 1] = 0.0   # noise - all neurons 0-10 have noise
    neuron_params[:, 2] = 0.0   # tonic level

    neuron_params[10, 2] = 1.0 # always on neuron
    # ---------------------------------
    # connection matrix
    # ---------------------------------
    # [:, :, 0] = weight
    # [:, :, 1] = reliability
    connections = np.zeros((n_neurons, n_neurons, 2), dtype=float)

    # set reliability = 1 everywhere
    connections[:, :, 1] = 1.0

    # ---------------------------------
    # manual wiring (input -> output)
    # ---------------------------------
    # Note: input sources are wired separately in decisionmaking_neuronal_algorithm.py
    # This connection matrix is only for neuron-to-neuron connections
    
    # neurons 0-4 feed to neurons 5-9
    # AND neurons 0-4 inhibit neuron 10
    for i in range(5):
        src = i
        tgt = i + 5
        if i == 0:
            # Connection 0→5 (stay/eat): strong, reflexive, plastic
            connections[src, tgt, 0] = 1.0
        else:
            # Connections 1-4→6-9 (directions): weak, for learning
            connections[src, tgt, 0] = 1.0
        connections[src, 10, 0] = -1.0  # inhibition to interneuron

    # interneuron 10 excites output neurons 6-9 (but not 5 which is "stay")
    for i in range(6, 10):
        connections[10, i, 0] = 1.0  # weight

    # ---------------------------------
    # plasticity parameters
    # ---------------------------------
    eta = 0.0  # Global plasticity factor
    
    # TEST SETUP: Connection from neuron 1 (food_north sensor) to neuron 6 (move_north output)
    # should receive modulatory input from neuron 1 itself with mod_weight = +1.0
    # This means: when food_north fires, it strengthens its own connection to move_north
    # Expected behavior: Initially weak response to north food, strengthens over encounters
    
    # Plasticity setup: initialize modulator_spec
    modulator_spec = {}
       
    # neurons 1-4 modulate their respective direction connections
    # Additionally, each direction gets depression (-0.5) from the next direction neuron (circular)
    for i in range(1, 5):
        src = i
        tgt = i + 5
        next_neuron = 1 + ((i - 1 + 1) % 4)  # cycles: 1→2→3→4→1
        modulator_spec[(src, tgt)] = [(next_neuron, -0.5)]
    

    return neuron_params, connections, sensory_mapping, max_decision_delay, eta, modulator_spec


if __name__ == "__main__":
    build_brain_spec()
