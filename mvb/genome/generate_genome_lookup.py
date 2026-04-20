"""
Generate deterministic lookup table genomes for CTRNN neural networks.

This module creates hand-crafted genomes with a specific hard-wired architecture
that implements an algorithmic lookup table behavior. This is useful for testing
and validating that the simulation correctly executes a known circuit.

A genome specifies:
- Connection weights (neuron-to-neuron wiring)
- Modulation specification (which connections are modulated by which neurons)
- Tonic activations (baseline activation per neuron)
- Global plasticity factor (eta)

These are separated from structural info (n_neurons, sensory mapping, etc.)
which comes from YAML configuration.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class GenomeLookuParams:
    """Lookup genome generation parameters."""
    n_neurons: int
    description: str


@dataclass
class GenomeLookuResult:
    """Result from lookup genome generation."""
    params: GenomeLookuParams
    connection_weights: np.ndarray
    modulation_spec: dict
    tonic_activations: np.ndarray
    eta: float
    
    def __getitem__(self, key: str):
        """Support dict-like access for backward compatibility."""
        if key == "connection_weights":
            return self.connection_weights
        elif key == "modulation_spec":
            return self.modulation_spec
        elif key == "tonic_activations":
            return self.tonic_activations
        elif key == "eta":
            return self.eta
        elif key == "params":
            return self.params
        else:
            raise KeyError(f"GenomeLookuResult has no key '{key}'")


def generate_lookup_genome(yaml_config, rng_seed):
    """
    Generate a deterministic lookup table genome with hard-crafted wiring.
    
    This function creates a fixed neural circuit that implements a lookup table
    behavior, regardless of the rng_seed (seed is ignored for deterministic generation).
    
    Parameters
    ----------
    yaml_config : dict
        Configuration dict with 'brain' section containing:
        - n_neurons: int, total number of neurons (should be 11 for this circuit)
        - sensory_mapping: dict, for validation only (not used here)
        - output_mapping: dict, for validation only (not used here)
    
    rng_seed : int
        Random seed (ignored; included for API compatibility with generate_genome_random)
    
    Returns
    -------
    genome : dict
        Dictionary with keys:
        - 'connection_weights': np.ndarray shape (n_neurons, n_neurons, 2)
          Each [i, j] contains [weight, reliability]
        - 'modulation_spec': dict mapping (src, tgt) → [(mod_id, mod_weight), ...]
        - 'tonic_activations': np.ndarray shape (n_neurons,)
          Baseline tonic activation per neuron
        - 'eta': float
          Global plasticity factor
    
    Notes
    -----
    This circuit implements the following hard-coded architecture:
    
    **Hard-coded connection:** n0 → n5 with weight 1.0 (no modulation)
    
    **Excitatory connections:** n1→n6, n2→n7, n3→n8, n4→n9
    Each directional sensory neuron drives the corresponding output neuron.
    
    **Inhibitory connections:** n1, n2, n3, n4 → n10 (always-on neuron)
    Directional sensory neurons inhibit the interneuron.
    
    **Excitatory connections:** n10 → n6, n7, n8, n9
    Always-on neuron drives all movement outputs.
    
    **Plasticity modulation:**
    - Excitatory sensory connections (n1→n6, n2→n7, n3→n8, n4→n9)
      are potentiated by their respective source neurons
    - Inhibitory connections to n10 are potentiated by their respective sources
    - Connections from n10 are potentiated by n10 itself
    """
    
    brain_cfg = yaml_config.get("brain", {})
    n_neurons = brain_cfg.get("n_neurons", 11)
    
    # ============================================================
    # CONNECTION AND MODULATION WEIGHTS
    # ============================================================
    EXCITATORY_WEIGHT = 0.6
    INHIBITORY_WEIGHT = -0.6
    POTENTIATION_WEIGHT = 0.5
    HARD_CODED_CONNECTION_WEIGHT = 1.0
    
    # ============================================================
    # NEURON ROLES (hardcoded based on circuit architecture)
    # ============================================================
    INPUT_NEURONS = list(range(0, 5))       # 0-4: sensory input neurons
    OUTPUT_NEURONS = list(range(5, 10))     # 5-9: output neurons
    ALWAYS_ON = 10                          # 10: always-on neuron
    
    # ============================================================
    # INITIALIZE CONNECTION MATRIX
    # ============================================================
    # Shape: (n_neurons, n_neurons, 2)
    # Each [i, j] = [weight, reliability]
    connection_weights = np.zeros((n_neurons, n_neurons, 2), dtype=np.float32)
    connection_weights[:, :, 1] = 1.0  # All reliability = 1.0
    
    # ============================================================
    # HARD-WIRED LOOKUP TABLE CONNECTIONS
    # ============================================================
    
    # Hard-coded connection: n0 → n5 with weight 1.0 (no modulation)
    connection_weights[0, 5, 0] = HARD_CODED_CONNECTION_WEIGHT
    
    # Excitatory Connections: n1→n6, n2→n7, n3→n8, n4→n9
    # Each directional sensory neuron drives the corresponding output neuron
    connection_weights[1, 6, 0] = EXCITATORY_WEIGHT
    connection_weights[2, 7, 0] = EXCITATORY_WEIGHT
    connection_weights[3, 8, 0] = EXCITATORY_WEIGHT
    connection_weights[4, 9, 0] = EXCITATORY_WEIGHT
    
    # Inhibitory Connections: n1, n2, n3, n4 → n10 (always-on neuron)
    # Directional sensory neurons inhibit the interneuron
    connection_weights[1, ALWAYS_ON, 0] = INHIBITORY_WEIGHT
    connection_weights[2, ALWAYS_ON, 0] = INHIBITORY_WEIGHT
    connection_weights[3, ALWAYS_ON, 0] = INHIBITORY_WEIGHT
    connection_weights[4, ALWAYS_ON, 0] = INHIBITORY_WEIGHT
    
    # Excitatory Connections: n10 → n6, n7, n8, n9
    # Always-on neuron drives all movement outputs
    connection_weights[ALWAYS_ON, 6, 0] = EXCITATORY_WEIGHT
    connection_weights[ALWAYS_ON, 7, 0] = EXCITATORY_WEIGHT
    connection_weights[ALWAYS_ON, 8, 0] = EXCITATORY_WEIGHT
    connection_weights[ALWAYS_ON, 9, 0] = EXCITATORY_WEIGHT
    
    # ============================================================
    # PLASTICITY MODULATION (Hard-wired)
    # ============================================================
    # Initialize modulator_spec for all existing connections
    modulation_spec = {}
    for src in range(n_neurons):
        for tgt in range(n_neurons):
            if connection_weights[src, tgt, 0] != 0.0:
                modulation_spec[(src, tgt)] = []
    
    # No modulation on n0 → n5 connection (leave empty list)
    # modulation_spec[(0, 5)] already has empty list from initialization
    
    # Potentiation on Excitatory Connections: n1→n6, n2→n7, n3→n8, n4→n9
    # Each connection is potentiated by its source neuron
    modulation_spec[(1, 6)].append((1, POTENTIATION_WEIGHT))
    modulation_spec[(2, 7)].append((2, POTENTIATION_WEIGHT))
    modulation_spec[(3, 8)].append((3, POTENTIATION_WEIGHT))
    modulation_spec[(4, 9)].append((4, POTENTIATION_WEIGHT))
    
    # Potentiation on Inhibitory Connections: n1, n2, n3, n4 → n10
    # Each inhibitory connection has potentiation triggered by its source
    modulation_spec[(1, ALWAYS_ON)].append((1, POTENTIATION_WEIGHT))
    modulation_spec[(2, ALWAYS_ON)].append((2, POTENTIATION_WEIGHT))
    modulation_spec[(3, ALWAYS_ON)].append((3, POTENTIATION_WEIGHT))
    modulation_spec[(4, ALWAYS_ON)].append((4, POTENTIATION_WEIGHT))
    
    # Potentiation on n10 → n6, n7, n8, n9 Connections
    # All these connections are potentiated by n10
    modulation_spec[(ALWAYS_ON, 6)].append((ALWAYS_ON, POTENTIATION_WEIGHT))
    modulation_spec[(ALWAYS_ON, 7)].append((ALWAYS_ON, POTENTIATION_WEIGHT))
    modulation_spec[(ALWAYS_ON, 8)].append((ALWAYS_ON, POTENTIATION_WEIGHT))
    modulation_spec[(ALWAYS_ON, 9)].append((ALWAYS_ON, POTENTIATION_WEIGHT))
    
    # ============================================================
    # TONIC ACTIVATIONS
    # ============================================================
    tonic_activations = np.zeros(n_neurons, dtype=np.float32)
    
    # Most neurons have zero tonic activation
    # Exception: always-on neuron has constant high tonic drive
    if n_neurons > ALWAYS_ON:
        tonic_activations[ALWAYS_ON] = 1.0
    
    # ============================================================
    # GLOBAL PLASTICITY FACTOR
    # ============================================================
    eta = 0.01
    
    # ============================================================
    # PACKAGE INTO GENOME RESULT
    # ============================================================
    params = GenomeLookuParams(
        n_neurons=n_neurons,
        description='hand-crafted, prioritize staying if on food, move onto food if sensed, otherwise force random movement'
    )
    
    result = GenomeLookuResult(
        params=params,
        connection_weights=connection_weights,
        modulation_spec=modulation_spec,
        tonic_activations=tonic_activations,
        eta=eta,
    )
    
    return result
