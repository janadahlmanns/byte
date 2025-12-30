import numpy as np


def build_brain_spec(rng=None):
    """
    Builds the initial brain specification (genotype).

    Returns
    -------
    neuron_params : np.ndarray, shape (10, 2)
        Column 0: threshold
        Column 1: noise level

    connections : np.ndarray, shape (10, 10, 2)
        [:, :, 0] = weight
        [:, :, 1] = reliability
        Rows = source neuron
        Columns = target neuron
    """

    n_neurons = 11

    # ---------------------------------
    # neuron parameters
    # ---------------------------------
    # column 0: threshold
    # column 1: noise level
    # column 2: tonic
    neuron_params = np.zeros((n_neurons, 3), dtype=float)
    neuron_params[:, 0] = 0.5   # threshold
    neuron_params[:, 1] = 0.0   # noise
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
    # input neurons:  0..4
    # output neurons: 5..9
    for i in range(5):
        src = i
        tgt = i + 5
        connections[src, tgt, 0] = 1.0  # weight
        connections[src, 10, 0] = -1.0

    for i in range(6,10):
        tgt = i
        connections [10, tgt, 0] = 1.0

    print("Neuron params (first 5 rows):")
    print(neuron_params[:5])
    print("Connection weights:")
    print(connections[:, :, 0])
    print("Connection reliability:")
    print(connections[:, :, 1])
    print("Non-zero connections (src, tgt, weight):")
    for src in range(10):
        for tgt in range(10):
            w = connections[src, tgt, 0]
            if w != 0:
                print(src, "->", tgt, ":", w)


    return neuron_params, connections

build_brain_spec()