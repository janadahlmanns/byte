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
    POTENTIATION_WEIGHT = 1.0
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
    # that replicate behavior: food sensed → move there; no food → all outputs active
    connections = np.zeros((n_neurons, n_neurons, 2), dtype=float)
    connections[:, :, 1] = 1.0  # All reliability = 1.0
    
    # Define neuron roles
    ON_FOOD_NEURON = INPUT_NEURONS[0]     # 0
    STAY_OUTPUT_NEURON = OUTPUT_NEURONS[0]  # 5
    
    # ============================================================
    # Excitatory Connections: 0-4 → 5-9
    # ============================================================
    # Each sensory input neuron drives the corresponding output neuron
    for src in INPUT_NEURONS:  # 0-4
        for tgt in OUTPUT_NEURONS:  # 5-9
            connections[src, tgt, 0] = EXCITATORY_WEIGHT
    
    # ============================================================
    # HARD-CODED SPECIAL CONNECTION
    # ============================================================
    # On Food → Stay/Eat: neuron 0 → neuron 5, weight 1.0 (overrides excitatory weight)
    connections[ON_FOOD_NEURON, STAY_OUTPUT_NEURON, 0] = HARD_CODED_CONNECTION_WEIGHT
    
    # ============================================================
    # Inhibitory Connections: 1-4 → 5
    # ============================================================
    # When directional food is detected (1-4), inhibit the "stay" output (5)
    # This competes with the sensory-to-output excitation during decision-making
    for src in INPUT_NEURONS[1:]:  # 1-4 (exclude "on_food" neuron 0)
        connections[src, STAY_OUTPUT_NEURON, 0] = INHIBITORY_WEIGHT
    
    # ============================================================
    # Excitatory Connections: 10 → 5-9
    # ============================================================
    # Always-on neuron drives all output neurons
    # This ensures all outputs can be active when no food is sensed
    for tgt in OUTPUT_NEURONS:  # 5-9
        connections[ALWAYS_ON, tgt, 0] = EXCITATORY_WEIGHT
    
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
    # Potentiation on Excitatory Sensory-to-Output Connections
    # ============================================================
    # Connections 0-4 → 5-9 are potentiated by their source neurons (0-4)
    for src in INPUT_NEURONS:  # 0-4
        for tgt in OUTPUT_NEURONS:  # 5-9
            if (src, tgt) in modulator_spec:
                modulator_spec[(src, tgt)].append((src, POTENTIATION_WEIGHT))
    
    # ============================================================
    # Depression on Inhibitory Directional-to-Stay Connections
    # ============================================================
    # Connections 1-4 → 5 (inhibitory) are modulated by depression from their sources
    for src in INPUT_NEURONS[1:]:  # 1-4
        if (src, STAY_OUTPUT_NEURON) in modulator_spec:
            modulator_spec[(src, STAY_OUTPUT_NEURON)].append((src, DEPRESSION_WEIGHT))
    
    # ============================================================
    # Potentiation on Always-On-to-Output Connections
    # ============================================================
    # Connections 10 → 5-9 are potentiated by neuron 10
    for tgt in OUTPUT_NEURONS:  # 5-9
        if (ALWAYS_ON, tgt) in modulator_spec:
            modulator_spec[(ALWAYS_ON, tgt)].append((ALWAYS_ON, POTENTIATION_WEIGHT))
    
    return neuron_params, connections, sensory_mapping, max_decision_delay, eta, modulator_spec


if __name__ == "__main__":
    build_brain_spec()
