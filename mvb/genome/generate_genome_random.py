"""
Generate random genomes for CTRNN neural networks.

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
from typing import Dict, Tuple, List

# ============================================================
# GENOME RANDOMIZATION PARAMETERS
# ============================================================

CONNECTIVITY_DEGREE_EXCITATORY = 0.2       # Fraction of excitatory connections
CONNECTIVITY_DEGREE_INHIBITORY = 0.4       # Fraction of inhibitory connections
MODULATION_DEGREE_POTENTIATION = 0.5       # Fraction for potentiation modulation
MODULATION_DEGREE_DEPRESSION = 0.5         # Fraction for depression modulation
ETA_PLASTICITY = 0.01                      # Global plasticity factor


# ============================================================
# RESULT STRUCTURES
# ============================================================

@dataclass
class GenomeRandomParams:
    """Input parameters for random genome generation."""
    connectivity_degree_excitatory: float
    connectivity_degree_inhibitory: float
    modulation_degree_potentiation: float
    modulation_degree_depression: float
    eta_plasticity: float
    n_neurons: int
    rng_seed: int
    
    def __repr__(self) -> str:
        return (
            f"GenomeRandomParams(\n"
            f"  connectivity_excitatory={self.connectivity_degree_excitatory},\n"
            f"  connectivity_inhibitory={self.connectivity_degree_inhibitory},\n"
            f"  modulation_potentiation={self.modulation_degree_potentiation},\n"
            f"  modulation_depression={self.modulation_degree_depression},\n"
            f"  eta_plasticity={self.eta_plasticity},\n"
            f"  n_neurons={self.n_neurons},\n"
            f"  rng_seed={self.rng_seed}\n"
            f")"
        )


@dataclass
class GenomeRandomResult:
    """Complete result from random genome generation."""
    params: GenomeRandomParams
    connection_weights: np.ndarray  # shape (n_neurons, n_neurons, 2)
    modulation_spec: Dict[Tuple[int, int], List[Tuple[int, float]]]
    tonic_activations: np.ndarray  # shape (n_neurons,)
    eta: float
    
    def to_dict(self) -> dict:
        """Convert to dict format (compatible with old code)."""
        return {
            "connection_weights": self.connection_weights,
            "modulation_spec": self.modulation_spec,
            "tonic_activations": self.tonic_activations,
            "eta": self.eta,
        }
    
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
            raise KeyError(f"GenomeRandomResult has no key '{key}'")


def generate_random_genome(yaml_config, rng_seed):
    """
    Generate a random genome (wiring, modulation, tonic activations, eta).
    
    Parameters
    ----------
    yaml_config : dict
        Configuration dict with 'brain' section containing:
        - n_neurons: int, total number of neurons
        - sensory_mapping: dict, for validation only (not used here)
        - output_mapping: dict, for validation only (not used here)
    
    rng_seed : int
        Random seed for reproducibility
    
    Returns
    -------
    genome : GenomeRandomResult
        Result object containing:
        - params: GenomeRandomParams with all input parameters
        - connection_weights: np.ndarray shape (n_neurons, n_neurons, 2)
          Each [i, j] contains [weight, reliability]
        - modulation_spec: dict mapping (src, tgt) → [(mod_id, mod_weight), ...]
        - tonic_activations: np.ndarray shape (n_neurons,)
          Baseline tonic activation per neuron
        - eta: float
          Global plasticity factor
    """
    rng = np.random.default_rng(rng_seed)
    
    brain_cfg = yaml_config.get("brain", {})
    n_neurons = brain_cfg.get("n_neurons", 10)
    
    # Capture input parameters
    params = GenomeRandomParams(
        connectivity_degree_excitatory=CONNECTIVITY_DEGREE_EXCITATORY,
        connectivity_degree_inhibitory=CONNECTIVITY_DEGREE_INHIBITORY,
        modulation_degree_potentiation=MODULATION_DEGREE_POTENTIATION,
        modulation_degree_depression=MODULATION_DEGREE_DEPRESSION,
        eta_plasticity=ETA_PLASTICITY,
        n_neurons=n_neurons,
        rng_seed=rng_seed,
    )
    
    # Initialize connection matrix: (n_neurons, n_neurons, 2)
    # Each entry [i, j] = [weight, reliability]
    connection_weights = np.zeros((n_neurons, n_neurons, 2), dtype=np.float32)
    
    # Generate excitatory connections
    num_excitatory = int(n_neurons * n_neurons * CONNECTIVITY_DEGREE_EXCITATORY)
    excitatory_pairs = rng.choice(
        n_neurons * n_neurons, 
        size=num_excitatory, 
        replace=False
    )
    for pair_idx in excitatory_pairs:
        src = pair_idx // n_neurons
        tgt = pair_idx % n_neurons
        weight = rng.uniform(0.1, 1.0)  # Positive weight for excitatory
        reliability = rng.uniform(0.0, 1.0)
        connection_weights[src, tgt] = [weight, reliability]
    
    # Generate inhibitory connections
    num_inhibitory = int(n_neurons * n_neurons * CONNECTIVITY_DEGREE_INHIBITORY)
    available = set(range(n_neurons * n_neurons)) - set(excitatory_pairs)
    inhibitory_pairs = rng.choice(
        list(available),
        size=min(num_inhibitory, len(available)),
        replace=False
    )
    for pair_idx in inhibitory_pairs:
        src = pair_idx // n_neurons
        tgt = pair_idx % n_neurons
        weight = rng.uniform(-1.0, -0.1)  # Negative weight for inhibitory
        reliability = rng.uniform(0.0, 1.0)
        connection_weights[src, tgt] = [weight, reliability]
    
    # Generate modulation specification
    # Structure: {(target_src, target_tgt): [(modulator_neuron_id, mod_weight), ...]}
    modulation_spec = {}
    
    # Find all non-zero connections that could be modulated
    non_zero_connections = []
    for src in range(n_neurons):
        for tgt in range(n_neurons):
            if connection_weights[src, tgt, 0] != 0.0:
                non_zero_connections.append((src, tgt))
    
    # For each connection, possibly add modulators
    num_connections_to_modulate = int(
        len(non_zero_connections) * 
        (MODULATION_DEGREE_POTENTIATION + MODULATION_DEGREE_DEPRESSION)
    )
    
    if num_connections_to_modulate > 0 and non_zero_connections:
        connections_to_modulate = rng.choice(
            len(non_zero_connections),
            size=min(num_connections_to_modulate, len(non_zero_connections)),
            replace=False
        )
        
        for conn_idx in connections_to_modulate:
            src, tgt = non_zero_connections[conn_idx]
            
            # Randomly select modulators from all neurons
            num_modulators = rng.integers(1, 3)  # 1-2 modulators per connection
            modulator_ids = rng.choice(n_neurons, size=num_modulators, replace=False)
            
            # Randomly choose potentiation vs depression for each modulator
            modulators = []
            for mod_id in modulator_ids:
                if rng.random() < MODULATION_DEGREE_POTENTIATION:
                    mod_weight = rng.uniform(0.1, 0.5)  # Positive for potentiation
                else:
                    mod_weight = rng.uniform(-0.5, -0.1)  # Negative for depression
                modulators.append((mod_id, mod_weight))
            
            modulation_spec[(src, tgt)] = modulators
    
    # Generate random tonic activations
    tonic_activations = rng.uniform(0.0, 0.3, size=n_neurons).astype(np.float32)
    
   
    # Package into result structure
    result = GenomeRandomResult(
        params=params,
        connection_weights=connection_weights,
        modulation_spec=modulation_spec,
        tonic_activations=tonic_activations,
        eta=ETA_PLASTICITY,
    )
    
    return result
