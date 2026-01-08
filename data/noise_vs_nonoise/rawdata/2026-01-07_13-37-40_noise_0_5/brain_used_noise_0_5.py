import numpy as np


def build_brain_spec():
    """
    Builds the initial brain specification (genotype) WITH NEURONAL NOISE.

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
    """

    n_neurons = 11

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
    neuron_params[:, 0] = 0.5   # threshold
    neuron_params[:, 1] = 0.5   # noise - all neurons 0-10 have noise
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
        connections[src, tgt, 0] = 1.0  # weight
        connections[src, 10, 0] = -1.0  # inhibition to interneuron

    # interneuron 10 excites output neurons 6-9 (but not 5 which is "stay")
    for i in range(6, 10):
        connections[10, i, 0] = 1.0

    # DEBUG OUTPUT (commented out - uncomment to see circuit details)
    # print("Neuron params (first 5 rows):")
    # print(neuron_params[:5])
    # print("\n" + "="*60)
    # print("CONNECTION MATRIX (weights)")
    # print("="*60)
    # print("Rows = Source, Columns = Target")
    # print()
    # 
    # # Create a readable adjacency matrix
    # labels = [f"in{i}" for i in range(5)] + [f"out{i}" for i in range(5)] + ["int10"]
    # 
    # print("     ", end="")
    # for label in labels:
    #     print(f"{label:>8}", end="")
    # print()
    # print("-" * 100)
    # 
    # for src in range(n_neurons):
    #     print(f"{labels[src]:>4}", end=" ")
    #     for tgt in range(n_neurons):
    #         w = connections[src, tgt, 0]
    #         if w == 0:
    #             print("       .", end="")
    #         else:
    #             print(f"{w:>8.1f}", end="")
    #     print()
    # 
    # print("\n" + "="*60)
    # print("NON-ZERO CONNECTIONS (src -> tgt : weight)")
    # print("="*60)
    # for src in range(n_neurons):
    #     for tgt in range(n_neurons):
    #         w = connections[src, tgt, 0]
    #         if w != 0:
    #             src_label = f"in{src}" if src < 5 else (f"out{src-5}" if src < 10 else "int10")
    #             tgt_label = f"in{tgt}" if tgt < 5 else (f"out{tgt-5}" if tgt < 10 else "int10")
    #             print(f"{src_label:>5} -> {tgt_label:>5} : {w:>6.1f}")
    # 
    # print("\n" + "="*60)
    # print("SENSORY MAPPING")
    # print("="*60)
    # for sense_key, (neuron_id, weight, reliability) in sensory_mapping.items():
    #     print(f"{sense_key:>15} -> neuron {neuron_id}  (w={weight}, r={reliability})")

    return neuron_params, connections, sensory_mapping, 2.0

build_brain_spec()
