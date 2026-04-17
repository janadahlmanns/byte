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
    # RANDOMIZED NEURON-TO-NEURON CONNECTIONS
    # ============================================================
    connections = np.zeros((n_neurons, n_neurons, 2), dtype=float)
    connections[:, :, 1] = 1.0  # All reliability = 1.0
    
    # Hard-coded connection: On Food → Stay/Eat (neuron 0 → neuron 5)
    # This connection is exempt from randomization and does not count toward connectivity degree
    ON_FOOD_NEURON = INPUT_NEURONS[0]     # 0
    STAY_OUTPUT_NEURON = OUTPUT_NEURONS[0]  # 5
    
    # Calculate target number of connections for each type
    # Full possibility space: 11 x 11 = 121 possible connections
    # Subtract 1 from excitatory for the hard-coded connection
    num_possible = n_neurons * n_neurons
    num_excitatory_target = int(connectivity_degree_excitatory * num_possible) - 1
    num_inhibitory_target = int(connectivity_degree_inhibitory * num_possible)
    
    # ============================================================
    # Random Excitatory Connections
    # ============================================================
    num_excitatory = 0
    while num_excitatory < num_excitatory_target:
        src = int(rng_exc.integers(0, n_neurons))
        tgt = int(rng_exc.integers(0, n_neurons))
        
        # Skip the hard-coded connection (0 → 5)
        if src == ON_FOOD_NEURON and tgt == STAY_OUTPUT_NEURON:
            continue
        
        # Only add if this connection doesn't already exist
        if connections[src, tgt, 0] == 0.0:
            connections[src, tgt, 0] = EXCITATORY_WEIGHT
            num_excitatory += 1
    
    # ============================================================
    # Random Inhibitory Connections
    # ============================================================
    num_inhibitory = 0
    while num_inhibitory < num_inhibitory_target:
        src = int(rng_inh.integers(0, n_neurons))
        tgt = int(rng_inh.integers(0, n_neurons))
        
        # Only add if this connection doesn't already exist
        if connections[src, tgt, 0] == 0.0:
            connections[src, tgt, 0] = INHIBITORY_WEIGHT
            num_inhibitory += 1
    
    # ============================================================
    # HARD-CODED EXEMPT CONNECTION
    # ============================================================
    # On Food → Stay/Eat: neuron 0 → neuron 5, weight 1.0
    # Set after randomization to ensure it's never overwritten
    connections[ON_FOOD_NEURON, STAY_OUTPUT_NEURON, 0] = HARD_CODED_CONNECTION_WEIGHT
    
    # ============================================================
    # PLASTICITY MODULATION
    # ============================================================
    # Randomly assign modulation sources to existing connections
    # Potentiation: mod_weight = +1.0
    # Depression: mod_weight = -0.5
    
    # Find all existing connections
    existing_connections = []
    for src in range(n_neurons):
        for tgt in range(n_neurons):
            if connections[src, tgt, 0] != 0.0:
                existing_connections.append((src, tgt))
    
    num_existing = len(existing_connections)
    
    # Calculate targets for modulation
    # Possibility space: n_neurons * num_existing_connections (for which neuron modulates which connection)
    num_potentiation_target = int(modulation_degree_potentiation * num_existing)
    num_depression_target = int(modulation_degree_depression * num_existing)
    
    # Initialize modulator_spec with empty lists for each connection
    modulator_spec = {}
    for conn_src, conn_tgt in existing_connections:
        modulator_spec[(conn_src, conn_tgt)] = []
    
    # ============================================================
    # Random Potentiation Modulations
    # ============================================================
    num_potentiation = 0
    while num_potentiation < num_potentiation_target:
        mod_src = int(rng_mod_pot.integers(0, n_neurons))
        conn_idx = int(rng_mod_pot.integers(0, num_existing))
        conn_src, conn_tgt = existing_connections[conn_idx]
        
        # Check if this exact modulation (source, weight) doesn't already exist on this connection
        already_exists = any(m[0] == mod_src and m[1] == POTENTIATION_WEIGHT for m in modulator_spec[(conn_src, conn_tgt)])
        if not already_exists:
            modulator_spec[(conn_src, conn_tgt)].append((mod_src, POTENTIATION_WEIGHT))
            num_potentiation += 1
    
    # ============================================================
    # Random Depression Modulations
    # ============================================================
    num_depression = 0
    while num_depression < num_depression_target:
        mod_src = int(rng_mod_dep.integers(0, n_neurons))
        conn_idx = int(rng_mod_dep.integers(0, num_existing))
        conn_src, conn_tgt = existing_connections[conn_idx]
        
        # Check if this exact modulation (source, weight) doesn't already exist on this connection
        already_exists = any(m[0] == mod_src and m[1] == DEPRESSION_WEIGHT for m in modulator_spec[(conn_src, conn_tgt)])
        if not already_exists:
            modulator_spec[(conn_src, conn_tgt)].append((mod_src, DEPRESSION_WEIGHT))
            num_depression += 1
    
    return neuron_params, connections, sensory_mapping, max_decision_delay, eta, modulator_spec


if __name__ == "__main__":
    build_brain_spec()
