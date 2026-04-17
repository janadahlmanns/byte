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
    neuron_params[:, 1] = 0.1   # noise - all neurons 0-10 have noise
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
    # manual wiring (hard-coded connections)
    # ---------------------------------
    # Note: input sources are wired separately in decisionmaking_neuronal_algorithm.py
    # This connection matrix is only for neuron-to-neuron connections
    
    # n0 → n5 (on_food → stay): strong, reflexive, NO plasticity
    connections[0, 5, 0] = 1.0
    
    # n1-4 → n6-9 (directional sensors → directional outputs): weak, for learning
    connections[1, 6, 0] = 0.6
    connections[2, 7, 0] = 0.6
    connections[3, 8, 0] = 0.6
    connections[4, 9, 0] = 0.6
    
    # n1-4 → n10 (directional sensors inhibit interneuron)
    connections[1, 10, 0] = -0.6
    connections[2, 10, 0] = -0.6
    connections[3, 10, 0] = -0.6
    connections[4, 10, 0] = -0.6

    # n10 → n6-9 (interneuron excites all directional outputs)
    connections[10, 6, 0] = 0.6
    connections[10, 7, 0] = 0.6
    connections[10, 8, 0] = 0.6
    connections[10, 9, 0] = 0.6

    # ---------------------------------
    # plasticity parameters
    # ---------------------------------
    eta = 0.01  # Global plasticity factor
    
    # Plasticity setup: initialize modulator_spec for all connections
    modulator_spec = {}
    for src in range(n_neurons):
        for tgt in range(n_neurons):
            if connections[src, tgt, 0] != 0.0:
                modulator_spec[(src, tgt)] = []
    
    # n0 → n5: NO modulation (empty list already set)
    
    # n1-4 → n6-9: POTENTIATION triggered by source neurons
    modulator_spec[(1, 6)].append((1, 0.5))   # n1 potentiates its own connection
    modulator_spec[(2, 7)].append((2, 0.5))   # n2 potentiates its own connection
    modulator_spec[(3, 8)].append((3, 0.5))   # n3 potentiates its own connection
    modulator_spec[(4, 9)].append((4, 0.5))   # n4 potentiates its own connection
    
    # n1-4 → n10: POTENTIATION triggered by source neurons
    modulator_spec[(1, 10)].append((1, 0.5))  # n1 potentiates its inhibitory connection
    modulator_spec[(2, 10)].append((2, 0.5))  # n2 potentiates its inhibitory connection
    modulator_spec[(3, 10)].append((3, 0.5))  # n3 potentiates its inhibitory connection
    modulator_spec[(4, 10)].append((4, 0.5))  # n4 potentiates its inhibitory connection
    
    # n10 → n6-9: POTENTIATION triggered by n10
    modulator_spec[(10, 6)].append((10, 0.5))  # n10 potentiates connection to n6
    modulator_spec[(10, 7)].append((10, 0.5))  # n10 potentiates connection to n7
    modulator_spec[(10, 8)].append((10, 0.5))  # n10 potentiates connection to n8
    modulator_spec[(10, 9)].append((10, 0.5))  # n10 potentiates connection to n9

    return neuron_params, connections, sensory_mapping, max_decision_delay, eta, modulator_spec


if __name__ == "__main__":
    build_brain_spec()
