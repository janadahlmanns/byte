"""
Simple mutation operator for genomes.

Mutates an elite genome by applying stochastic changes to:
- Tonic activations (Gaussian noise, clipped to [0,1])
- Eta plasticity factor (Gaussian noise, clipped to [0,1])
- Connection weights and topology via 13 possible actions drawn uniformly:
    change_connection_weight, change_connection_reliability, connection_sign_flip,
    connection_new_source, connection_new_target, remove_connection, add_connection,
    change_modulatory_weight, modulation_sign_flip, new_modulating_source,
    change_modulated_connection, remove_modulation, add_modulation,
    resample_tonic_activation, change_eta
"""

import numpy as np
from .generate_genome_random import GenomeRandomResult

_MUTATION_ACTIONS = [
    "change_connection_weight",
    "change_connection_reliability",
    "connection_sign_flip",
    "connection_new_source",
    "connection_new_target",
    "remove_connection",
    "add_connection",
    "change_modulatory_weight",
    "modulation_sign_flip",
    "new_modulating_source",
    "change_modulated_connection",
    "remove_modulation",
    "add_modulation",
    "resample_tonic_activation",
    "change_eta",
]
_N_ACTIONS = len(_MUTATION_ACTIONS)


def generate_genome_mutate_simple(elite_genome, mutation_rate, mutation_strength, rng_mutation):
    """
    Mutate a single elite genome and return a new mutated genome.

    Parameters
    ----------
    elite_genome : GenomeRandomResult
        The elite genome to mutate (contains connection_weights, modulation_spec,
        tonic_activations, eta)
    mutation_rate : float
        Controls how many mutations are applied: n_mutations = n_neurons^2 * mutation_rate.
        Value in [0, 1]
    mutation_strength : float
        Standard deviation for all Gaussian changes (weights, reliability, eta, tonic activations).
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
    # 1. APPLY N_MUTATIONS ACTIONS (connections + modulations + tonic + eta)
    # ============================================================
    n_mutations = max(1, int(n_neurons * n_neurons * mutation_rate))
    action_indices = rng_mutation.integers(0, _N_ACTIONS, size=n_mutations)

    for action_idx in action_indices:
        action = _MUTATION_ACTIONS[action_idx]

        # Rebuild helpers each step so topology changes are visible to later actions
        non_zero = [
            (s, t) for s in range(n_neurons) for t in range(n_neurons)
            if connection_weights[s, t, 0] != 0.0
        ]
        all_mod_entries = [
            (conn_key, i, mod_id, mod_weight)
            for conn_key, modulators in modulation_spec.items()
            for i, (mod_id, mod_weight) in enumerate(modulators)
        ]

        # ---- CONNECTION ACTIONS ----------------------------------

        if action == "change_connection_weight":
            if not non_zero:
                continue
            src, tgt = non_zero[rng_mutation.integers(0, len(non_zero))]
            connection_weights[src, tgt, 0] = float(
                np.clip(connection_weights[src, tgt, 0] + rng_mutation.normal(0, mutation_strength), -1.0, 1.0)
            )

        elif action == "change_connection_reliability":
            if not non_zero:
                continue
            src, tgt = non_zero[rng_mutation.integers(0, len(non_zero))]
            connection_weights[src, tgt, 1] = float(
                np.clip(connection_weights[src, tgt, 1] + rng_mutation.normal(0, mutation_strength), 0.0, 1.0)
            )

        elif action == "connection_sign_flip":
            if not non_zero:
                continue
            src, tgt = non_zero[rng_mutation.integers(0, len(non_zero))]
            connection_weights[src, tgt, 0] = -connection_weights[src, tgt, 0]

        elif action == "connection_new_source":
            if not non_zero:
                continue
            src, tgt = non_zero[rng_mutation.integers(0, len(non_zero))]
            available = [s for s in range(n_neurons) if connection_weights[s, tgt, 0] == 0.0]
            if not available:
                continue
            new_src = available[rng_mutation.integers(0, len(available))]
            old_key, new_key = (src, tgt), (new_src, tgt)
            connection_weights[new_src, tgt] = connection_weights[src, tgt].copy()
            connection_weights[src, tgt] = [0.0, 0.0]
            if old_key in modulation_spec:
                if new_key not in modulation_spec:
                    modulation_spec[new_key] = modulation_spec.pop(old_key)
                else:
                    modulation_spec[new_key].extend(modulation_spec.pop(old_key))

        elif action == "connection_new_target":
            if not non_zero:
                continue
            src, tgt = non_zero[rng_mutation.integers(0, len(non_zero))]
            available = [t for t in range(n_neurons) if connection_weights[src, t, 0] == 0.0]
            if not available:
                continue
            new_tgt = available[rng_mutation.integers(0, len(available))]
            old_key, new_key = (src, tgt), (src, new_tgt)
            connection_weights[src, new_tgt] = connection_weights[src, tgt].copy()
            connection_weights[src, tgt] = [0.0, 0.0]
            if old_key in modulation_spec:
                if new_key not in modulation_spec:
                    modulation_spec[new_key] = modulation_spec.pop(old_key)
                else:
                    modulation_spec[new_key].extend(modulation_spec.pop(old_key))

        elif action == "remove_connection":
            if not non_zero:
                continue
            src, tgt = non_zero[rng_mutation.integers(0, len(non_zero))]
            connection_weights[src, tgt] = [0.0, 0.0]
            modulation_spec.pop((src, tgt), None)

        elif action == "add_connection":
            zero_slots = [
                (s, t) for s in range(n_neurons) for t in range(n_neurons)
                if connection_weights[s, t, 0] == 0.0
            ]
            if not zero_slots:
                continue
            src, tgt = zero_slots[rng_mutation.integers(0, len(zero_slots))]
            connection_weights[src, tgt, 0] = float(rng_mutation.uniform(-1.0, 1.0))
            connection_weights[src, tgt, 1] = float(rng_mutation.uniform(0.0, 1.0))

        # ---- MODULATION ACTIONS ----------------------------------

        elif action == "change_modulatory_weight":
            if not all_mod_entries:
                continue
            conn_key, i, mod_id, mod_weight = all_mod_entries[rng_mutation.integers(0, len(all_mod_entries))]
            modulation_spec[conn_key][i] = (
                mod_id,
                float(np.clip(mod_weight + rng_mutation.normal(0, mutation_strength), -1.0, 1.0))
            )

        elif action == "modulation_sign_flip":
            if not all_mod_entries:
                continue
            conn_key, i, mod_id, mod_weight = all_mod_entries[rng_mutation.integers(0, len(all_mod_entries))]
            modulation_spec[conn_key][i] = (mod_id, -mod_weight)

        elif action == "new_modulating_source":
            if not all_mod_entries:
                continue
            conn_key, i, mod_id, mod_weight = all_mod_entries[rng_mutation.integers(0, len(all_mod_entries))]
            modulation_spec[conn_key][i] = (int(rng_mutation.integers(0, n_neurons)), mod_weight)

        elif action == "change_modulated_connection":
            if not all_mod_entries:
                continue
            conn_key, i, mod_id, mod_weight = all_mod_entries[rng_mutation.integers(0, len(all_mod_entries))]
            candidates = [c for c in non_zero if c != conn_key]
            if not candidates:
                continue
            new_key = candidates[rng_mutation.integers(0, len(candidates))]
            modulation_spec[conn_key].pop(i)
            if not modulation_spec[conn_key]:
                del modulation_spec[conn_key]
            if new_key not in modulation_spec:
                modulation_spec[new_key] = []
            modulation_spec[new_key].append((mod_id, mod_weight))

        elif action == "remove_modulation":
            if not all_mod_entries:
                continue
            conn_key, i, mod_id, mod_weight = all_mod_entries[rng_mutation.integers(0, len(all_mod_entries))]
            modulation_spec[conn_key].pop(i)
            if not modulation_spec[conn_key]:
                del modulation_spec[conn_key]

        elif action == "add_modulation":
            if not non_zero:
                continue
            conn_key = non_zero[rng_mutation.integers(0, len(non_zero))]
            new_mod_id = int(rng_mutation.integers(0, n_neurons))
            new_mod_weight = float(rng_mutation.uniform(-1.0, 1.0))
            if conn_key not in modulation_spec:
                modulation_spec[conn_key] = []
            modulation_spec[conn_key].append((new_mod_id, new_mod_weight))

        # ---- TONIC / ETA ACTIONS --------------------------------

        elif action == "resample_tonic_activation":
            neuron_idx = rng_mutation.integers(0, n_neurons)
            tonic_activations[neuron_idx] = np.float32(
                np.clip(tonic_activations[neuron_idx] + rng_mutation.normal(0, mutation_strength), 0.0, 1.0)
            )

        elif action == "change_eta":
            eta = float(np.clip(eta + rng_mutation.normal(0, mutation_strength), 0.0, 1.0))

    # ============================================================
    # 2. CREATE AND RETURN MUTATED GENOME
    # ============================================================
    mutated_genome = GenomeRandomResult(
        params=elite_genome.params,
        connection_weights=connection_weights,
        modulation_spec=modulation_spec,
        tonic_activations=tonic_activations,
        eta=eta,
    )

    return mutated_genome
