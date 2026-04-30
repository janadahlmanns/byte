"""
Simple mutation operator for genomes.

Mutates an elite genome by applying stochastic changes to:
- Tonic activations (Gaussian noise, clipped to [0,1])
- Eta plasticity factor (Gaussian noise, clipped to [0,1])
- Connection weights (Gaussian change, sign flip, or resample source/target neuron)
- Modulation specifications (Gaussian change, sign flip, or resample modulator/source/target neurons)
"""

import numpy as np
from typing import Tuple, List, Dict
from .generate_genome_random import GenomeRandomResult


def generate_genome_mutate_simple(elite_genome, mutation_rate, rng_mutation):
    """
    Mutate a single elite genome and return a new mutated genome.

    Parameters
    ----------
    elite_genome : GenomeRandomResult
        The elite genome to mutate (contains connection_weights, modulation_spec, 
        tonic_activations, eta)
    mutation_rate : float
        Fraction of connections to mutate (applied to both connections and modulations)
        Value in [0, 1]
    rng_mutation : np.random.Generator
        RNG for reproducible mutations

    Returns
    -------
    mutated_genome : GenomeRandomResult
        New mutated genome with same structure as input
    """
    
    # Deep copy the genome to avoid modifying the original
    n_neurons = elite_genome.connection_weights.shape[0]
    connection_weights = elite_genome.connection_weights.copy()
    modulation_spec = {k: v.copy() for k, v in elite_genome.modulation_spec.items()}
    tonic_activations = elite_genome.tonic_activations.copy()
    eta = elite_genome.eta
    
    # ============================================================
    # 1. MUTATE TONIC ACTIVATIONS
    # ============================================================
    # Apply Gaussian mutations with mutation_rate determining mutation strength
    gaussian_std = mutation_rate * 0.5  # Scale std by mutation_rate
    tonic_noise = rng_mutation.normal(0, gaussian_std, size=n_neurons).astype(np.float32)
    tonic_activations = tonic_activations + tonic_noise
    tonic_activations = np.clip(tonic_activations, 0.0, 1.0).astype(np.float32)
    
    # ============================================================
    # 2. MUTATE ETA (Global Plasticity Factor)
    # ============================================================
    eta_noise = rng_mutation.normal(0, gaussian_std)
    eta = eta + eta_noise
    eta = np.clip(eta, 0.0, 1.0)
    
    # ============================================================
    # 3. MUTATE CONNECTION WEIGHTS
    # ============================================================
    # Find all non-zero connections
    non_zero_connections = []
    for src in range(n_neurons):
        for tgt in range(n_neurons):
            if connection_weights[src, tgt, 0] != 0.0:
                non_zero_connections.append((src, tgt))
    
    # Determine how many connections to mutate
    num_connections_to_mutate = max(1, int(len(non_zero_connections) * mutation_rate))
    
    if num_connections_to_mutate > 0 and non_zero_connections:
        # Randomly select which connections to mutate
        mutation_indices = rng_mutation.choice(
            len(non_zero_connections),
            size=min(num_connections_to_mutate, len(non_zero_connections)),
            replace=False
        )
        
        for idx in mutation_indices:
            src, tgt = non_zero_connections[idx]
            
            # Choose mutation action: 0=gaussian_change_weight, 1=gaussian_change_reliability, 
            # 2=sign_flip, 3=resample_source, 4=resample_target
            action = rng_mutation.integers(0, 5)
            
            if action == 0:
                # Gaussian change the weight
                weight_noise = rng_mutation.normal(0, gaussian_std)
                connection_weights[src, tgt, 0] += weight_noise
                # Clip to [-1, 1] range (typical for neural weights)
                connection_weights[src, tgt, 0] = np.clip(connection_weights[src, tgt, 0], -1.0, 1.0)
            
            elif action == 1:
                # Gaussian change the reliability
                reliability_noise = rng_mutation.normal(0, gaussian_std)
                connection_weights[src, tgt, 1] += reliability_noise
                # Clip to [0, 1] range
                connection_weights[src, tgt, 1] = np.clip(connection_weights[src, tgt, 1], 0.0, 1.0)
            
            elif action == 2:
                # Sign flip the weight
                connection_weights[src, tgt, 0] = -connection_weights[src, tgt, 0]
            
            elif action == 3:
                # Resample source neuron
                # Find an empty destination or skip if all are occupied
                available_srcs = []
                for candidate_src in range(n_neurons):
                    if connection_weights[candidate_src, tgt, 0] == 0.0:
                        available_srcs.append(candidate_src)
                
                if available_srcs:
                    new_src = available_srcs[rng_mutation.integers(0, len(available_srcs))]
                    # Move the connection: remove from (src, tgt), add to (new_src, tgt)
                    # Also update any modulations that refer to this connection
                    old_key = (src, tgt)
                    new_key = (new_src, tgt)
                    
                    # Transfer modulation spec if it exists
                    if old_key in modulation_spec:
                        if new_key not in modulation_spec:
                            modulation_spec[new_key] = modulation_spec[old_key]
                        else:
                            # Merge if destination already has modulations
                            modulation_spec[new_key].extend(modulation_spec[old_key])
                        del modulation_spec[old_key]
                    
                    # Transfer weight and reliability
                    connection_weights[new_src, tgt] = connection_weights[src, tgt].copy()
                    connection_weights[src, tgt] = [0.0, 0.0]
            
            elif action == 3:
                # Resample target neuron
                # Find an empty destination or skip if all are occupied
                available_tgts = []
                for candidate_tgt in range(n_neurons):
                    if connection_weights[src, candidate_tgt, 0] == 0.0:
                        available_tgts.append(candidate_tgt)
                
                if available_tgts:
                    new_tgt = available_tgts[rng_mutation.integers(0, len(available_tgts))]
                    # Move the connection: remove from (src, tgt), add to (src, new_tgt)
                    # Also update any modulations that refer to this connection
                    old_key = (src, tgt)
                    new_key = (src, new_tgt)
                    
                    # Transfer modulation spec if it exists
                    if old_key in modulation_spec:
                        if new_key not in modulation_spec:
                            modulation_spec[new_key] = modulation_spec[old_key]
                        else:
                            # Merge if destination already has modulations
                            modulation_spec[new_key].extend(modulation_spec[old_key])
                        del modulation_spec[old_key]
                    
                    # Transfer weight and reliability
                    connection_weights[src, new_tgt] = connection_weights[src, tgt].copy()
                    connection_weights[src, tgt] = [0.0, 0.0]
    
    # ============================================================
    # 4. MUTATE MODULATIONS
    # ============================================================
    # Find all modulations
    all_modulations = list(modulation_spec.items())
    
    # Determine how many modulations to mutate
    num_modulations_to_mutate = max(1, int(len(all_modulations) * mutation_rate))
    
    if num_modulations_to_mutate > 0 and all_modulations:
        # Randomly select which modulations to mutate
        mutation_indices = rng_mutation.choice(
            len(all_modulations),
            size=min(num_modulations_to_mutate, len(all_modulations)),
            replace=False
        )
        
        for idx in mutation_indices:
            conn_key, modulators = all_modulations[idx]
            src, tgt = conn_key
            
            # Choose which modulator to mutate (if multiple exist)
            if len(modulators) > 0:
                mod_idx = rng_mutation.integers(0, len(modulators))
                mod_id, mod_weight = modulators[mod_idx]
                
                # Choose mutation action for this modulation
                # 0=gaussian_change, 1=sign_flip, 2=resample_mod_source, 
                # 3=resample_connection_source, 4=resample_connection_target
                action = rng_mutation.integers(0, 5)
                
                if action == 0:
                    # Gaussian change the modulatory weight
                    mod_noise = rng_mutation.normal(0, gaussian_std)
                    modulators[mod_idx] = (mod_id, np.clip(mod_weight + mod_noise, -1.0, 1.0))
                
                elif action == 1:
                    # Sign flip on the modulatory weight
                    modulators[mod_idx] = (mod_id, -mod_weight)
                
                elif action == 2:
                    # Resample the modulation source neuron
                    new_mod_id = rng_mutation.integers(0, n_neurons)
                    modulators[mod_idx] = (new_mod_id, mod_weight)
                
                elif action == 3:
                    # Resample the source neuron of the modulated connection
                    # with restriction that such a connection must exist
                    non_zero_for_resample = [
                        (s, t) for s in range(n_neurons) for t in range(n_neurons)
                        if connection_weights[s, t, 0] != 0.0 and (s, t) != conn_key
                    ]
                    if non_zero_for_resample:
                        new_src, new_tgt = non_zero_for_resample[
                            rng_mutation.integers(0, len(non_zero_for_resample))
                        ]
                        # Move this modulation to the new connection
                        new_key = (new_src, new_tgt)
                        del modulation_spec[conn_key]
                        if new_key not in modulation_spec:
                            modulation_spec[new_key] = []
                        modulation_spec[new_key].append(modulators[mod_idx])
                        # Update reference (this is tricky since we're iterating)
                        # For safety, we'll just break here
                        break
                
                elif action == 4:
                    # Resample the target neuron of the modulated connection
                    # with restriction that such a connection must exist after resampling
                    non_zero_for_resample = [
                        (s, t) for s in range(n_neurons) for t in range(n_neurons)
                        if connection_weights[s, t, 0] != 0.0 and (s, t) != conn_key
                    ]
                    if non_zero_for_resample:
                        new_src, new_tgt = non_zero_for_resample[
                            rng_mutation.integers(0, len(non_zero_for_resample))
                        ]
                        # Move this modulation to the new connection
                        new_key = (new_src, new_tgt)
                        del modulation_spec[conn_key]
                        if new_key not in modulation_spec:
                            modulation_spec[new_key] = []
                        modulation_spec[new_key].append(modulators[mod_idx])
                        # Update reference (this is tricky since we're iterating)
                        # For safety, we'll just break here
                        break
    
    # ============================================================
    # 5. CREATE AND RETURN MUTATED GENOME
    # ============================================================
    mutated_genome = GenomeRandomResult(
        params=elite_genome.params,
        connection_weights=connection_weights,
        modulation_spec=modulation_spec,
        tonic_activations=tonic_activations,
        eta=eta,
    )
    
    return mutated_genome
