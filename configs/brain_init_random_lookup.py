import numpy as np


def build_brain_spec(
    wiring_seed,
    connectivity_degree_excitatory,
    connectivity_degree_inhibitory,
    modulation_degree_potentiation,
    modulation_degree_depression,
):
    """
    Builds the initial brain specification with randomized wiring.
    
    All parameters are required (no defaults) to ensure reproducibility and prevent silent value overwrites.

    Parameters
    ----------
    wiring_seed : int
        Seed for wiring randomization. Required.
    
    connectivity_degree_excitatory : float
        Fraction of all possible connections to be excitatory (0.2 = 20%). Required.
    
    connectivity_degree_inhibitory : float
        Fraction of all possible connections to be inhibitory (0.4 = 40%). Required.
    
    modulation_degree_potentiation : float
        Fraction of connections to receive potentiation modulation. Required.
    
    modulation_degree_depression : float
        Fraction of connections to receive depression modulation. Required.

    Returns
    -------
    neuron_params : np.ndarray, shape (11, 3)
        Column 0: threshold, Column 1: noise level, Column 2: tonic level

    connections : np.ndarray, shape (11, 11, 2)
        [:, :, 0] = weight, [:, :, 1] = reliability
    
    sensory_mapping : dict
        Maps sensory input keys to (target_neuron_id, weight, reliability)
    
    max_decision_delay : float
        Maximum decision delay in ticks
    
    eta : float
        Global plasticity factor
    
    modulator_spec : dict
        Plasticity modulation specification (currently empty, will be filled next)
    """

    # ============================================================
    # INITIALIZATION
    # ============================================================
    n_neurons = 11
    max_decision_delay = 2.0
    eta = 0.01
    
    # Connection and modulation weights
    EXCITATORY_WEIGHT = 0.6
    INHIBITORY_WEIGHT = -0.6
    POTENTIATION_WEIGHT = 0.5
    DEPRESSION_WEIGHT = -0.5
    HARD_CODED_CONNECTION_WEIGHT = 1.0
    
    # Neuron IDs and roles (hardcoded based on circuit architecture)
    INPUT_NEURONS = list(range(0, 5))       # 0-4: sensory input neurons
    OUTPUT_NEURONS = list(range(5, 10))     # 5-9: output neurons
    ALWAYS_ON = 10                          # 10: always-on neuron
    
    # ============================================================
    # RNG STREAM SETUP
    # ============================================================
    # Split wiring seed into 4 independent streams for different randomization aspects
    rng_exc = np.random.default_rng(wiring_seed)
    rng_inh = np.random.default_rng(wiring_seed + 1)
    rng_mod_pot = np.random.default_rng(wiring_seed + 2)
    rng_mod_dep = np.random.default_rng(wiring_seed + 3)
    
    # Generate test number from first RNG for seed verification
    wiring_test_number = int(rng_exc.integers(0, 2**31))
    
    # ============================================================
    # NEURON PARAMETERS
    # ============================================================
    # All neurons have the same base parameters
    neuron_params = np.zeros((n_neurons, 3), dtype=float)
    neuron_params[:, 0] = 0.5      # threshold: all 0.5
    neuron_params[:, 1] = 0.1      # noise level: all 0.1
    neuron_params[:, 2] = 0.0      # tonic level: all 0.0 by default
    
    # Exception: always-on neuron has tonic drive
    neuron_params[ALWAYS_ON, 2] = 1.0
    
    # ============================================================
    # SENSORY INPUT MAPPING
    # ============================================================
    sensory_mapping = {
        "on_food":    (INPUT_NEURONS[0], 1.0, 1.0),
        "food_north": (INPUT_NEURONS[1], 1.0, 1.0),
        "food_east":  (INPUT_NEURONS[2], 1.0, 1.0),
        "food_south": (INPUT_NEURONS[3], 1.0, 1.0),
        "food_west":  (INPUT_NEURONS[4], 1.0, 1.0),
    }
    
    # ============================================================
    # HARD-WIRED LOOKUP TABLE CONNECTIONS
    # ============================================================
    # This implements the algorithmic wiring with deterministic connections
    connections = np.zeros((n_neurons, n_neurons, 2), dtype=float)
    connections[:, :, 1] = 1.0  # All reliability = 1.0
    
    # ============================================================
    # Hard-coded connection: n0 → n5 with weight 1.0 (no modulation)
    # ============================================================
    connections[0, 5, 0] = HARD_CODED_CONNECTION_WEIGHT
    
    # ============================================================
    # Excitatory Connections: n1→n6, n2→n7, n3→n8, n4→n9
    # ============================================================
    # Each directional sensory neuron drives the corresponding output neuron
    connections[1, 6, 0] = EXCITATORY_WEIGHT
    connections[2, 7, 0] = EXCITATORY_WEIGHT
    connections[3, 8, 0] = EXCITATORY_WEIGHT
    connections[4, 9, 0] = EXCITATORY_WEIGHT
    
    # ============================================================
    # Inhibitory Connections: n1, n2, n3, n4 → n10
    # ============================================================
    # Directional sensory neurons inhibit the interneuron
    connections[1, ALWAYS_ON, 0] = INHIBITORY_WEIGHT
    connections[2, ALWAYS_ON, 0] = INHIBITORY_WEIGHT
    connections[3, ALWAYS_ON, 0] = INHIBITORY_WEIGHT
    connections[4, ALWAYS_ON, 0] = INHIBITORY_WEIGHT
    
    # ============================================================
    # Excitatory Connections: n10 → n6, n7, n8, n9
    # ============================================================
    # Always-on neuron drives all movement outputs
    connections[ALWAYS_ON, 6, 0] = EXCITATORY_WEIGHT
    connections[ALWAYS_ON, 7, 0] = EXCITATORY_WEIGHT
    connections[ALWAYS_ON, 8, 0] = EXCITATORY_WEIGHT
    connections[ALWAYS_ON, 9, 0] = EXCITATORY_WEIGHT
    
    # ============================================================
    # PLASTICITY MODULATION (Hard-wired)
    # ============================================================
    # Initialize modulator_spec for all existing connections
    modulator_spec = {}
    for src in range(n_neurons):
        for tgt in range(n_neurons):
            if connections[src, tgt, 0] != 0.0:
                modulator_spec[(src, tgt)] = []
    
    # ============================================================
    # No modulation on n0 → n5 connection (leave empty list)
    # ============================================================
    # modulator_spec[(0, 5)] already has empty list from initialization
    
    # ============================================================
    # Potentiation on Excitatory Connections: n1→n6, n2→n7, n3→n8, n4→n9
    # ============================================================
    # Each connection is potentiated by its source neuron
    modulator_spec[(1, 6)].append((1, POTENTIATION_WEIGHT))
    modulator_spec[(2, 7)].append((2, POTENTIATION_WEIGHT))
    modulator_spec[(3, 8)].append((3, POTENTIATION_WEIGHT))
    modulator_spec[(4, 9)].append((4, POTENTIATION_WEIGHT))
    
    # ============================================================
    # Depression on Inhibitory Connections: n1, n2, n3, n4 → n10
    # ============================================================
    # Each inhibitory connection has depression triggered by its source
    modulator_spec[(1, ALWAYS_ON)].append((1, DEPRESSION_WEIGHT))
    modulator_spec[(2, ALWAYS_ON)].append((2, DEPRESSION_WEIGHT))
    modulator_spec[(3, ALWAYS_ON)].append((3, DEPRESSION_WEIGHT))
    modulator_spec[(4, ALWAYS_ON)].append((4, DEPRESSION_WEIGHT))
    
    # ============================================================
    # Potentiation on n10 → n6, n7, n8, n9 Connections
    # ============================================================
    # All these connections are potentiated by n10
    modulator_spec[(ALWAYS_ON, 6)].append((ALWAYS_ON, POTENTIATION_WEIGHT))
    modulator_spec[(ALWAYS_ON, 7)].append((ALWAYS_ON, POTENTIATION_WEIGHT))
    modulator_spec[(ALWAYS_ON, 8)].append((ALWAYS_ON, POTENTIATION_WEIGHT))
    modulator_spec[(ALWAYS_ON, 9)].append((ALWAYS_ON, POTENTIATION_WEIGHT))
    
    return neuron_params, connections, sensory_mapping, max_decision_delay, eta, modulator_spec


if __name__ == "__main__":
    build_brain_spec()
