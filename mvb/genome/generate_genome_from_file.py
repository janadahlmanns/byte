"""
Load genomes from HDF5 file (from EA results).

Reads elite genomes saved by run_ea.py and returns them in the same format
as generate_random_genome() for transparent integration into batch replay mode.
"""

import numpy as np
import h5py
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, List


@dataclass
class GenomeFromFileParams:
    """Parameters for genome loaded from file."""
    hdf5_path: str
    elite_id: int


@dataclass
class GenomeFromFileResult:
    """Complete result from HDF5 genome loading."""
    params: GenomeFromFileParams
    connection_weights: np.ndarray  # shape (n_neurons, n_neurons, 2)
    modulation_spec: Dict[Tuple[int, int], List[Tuple[int, float]]]
    tonic_activations: np.ndarray  # shape (n_neurons,)
    eta: float
    
    def to_dict(self) -> dict:
        """Convert to dict format (compatible with simulation_API)."""
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
            raise KeyError(f"GenomeFromFileResult has no key '{key}'")


def generate_genome_from_file(yaml_config, elite_id=None, hdf5_path=None):
    """
    Load a genome from HDF5 file (from EA results).
    
    Parameters
    ----------
    yaml_config : dict
        Configuration dict. If hdf5_path is None, uses cfg["from_file_genome"] to build path.
        Expected keys:
        - from_file_genome:
          - file_folder: str (folder containing HDF5 file)
          - filename: str (name of HDF5 file without extension)
    elite_id : int, optional
        Elite genome ID to load (e.g., 0, 1, 2, ...).
        If None, will raise error.
    hdf5_path : str, optional
        Full path to HDF5 file. If None, constructs from yaml_config.
    
    Returns
    -------
    GenomeFromFileResult
        Genome loaded from HDF5 in same format as generate_random_genome().
    
    Raises
    ------
    ValueError
        If elite_id is None or HDF5 file not found.
    """
    if elite_id is None:
        raise ValueError("[ERROR] elite_id must be specified when loading genome from file")
    
    # Resolve HDF5 path if not provided
    if hdf5_path is None:
        from_file_cfg = yaml_config.get("from_file_genome", {})
        file_folder = from_file_cfg.get("file_folder")
        filename = from_file_cfg.get("filename")
        
        if not file_folder or not filename:
            raise ValueError("[ERROR] 'from_file_genome' config missing 'file_folder' or 'filename'")
        
        # Build path: file_folder/filename.h5
        hdf5_path = str(Path(file_folder) / f"{filename}.h5")
    
    # Validate HDF5 file exists
    if not Path(hdf5_path).exists():
        raise FileNotFoundError(f"[ERROR] HDF5 file not found: {hdf5_path}")
    
    # Load genome from HDF5
    with h5py.File(hdf5_path, 'r') as f:
        elite_group_name = f"elite_genomes/elite_{elite_id}"
        
        if elite_group_name not in f:
            raise ValueError(
                f"[ERROR] Elite genome not found in {hdf5_path}: {elite_group_name}\n"
                f"Available elites: {[key for key in f['elite_genomes'].keys() if key.startswith('elite_')]}"
            )
        
        elite_group = f[elite_group_name]
        
        # Load connection weights
        connection_weights = elite_group['connection_weights'][:]
        
        # Load tonic activations
        tonic_activations = elite_group['tonic_activations'][:]
        
        # Load eta (stored as single-element array)
        eta_array = elite_group['eta'][:]
        eta = float(eta_array[0]) if len(eta_array) > 0 else float(eta_array[()])
        
        # Load modulation spec (stored as structured array, reconstruct dict)
        modulation_spec = {}
        if 'modulation_spec' in elite_group:
            mod_data = elite_group['modulation_spec'][:]
            if len(mod_data) > 0:
                for row in mod_data:
                    src, tgt, mod_neuron, mod_weight = row
                    key = (int(src), int(tgt))
                    if key not in modulation_spec:
                        modulation_spec[key] = []
                    modulation_spec[key].append((int(mod_neuron), float(mod_weight)))
    
    params = GenomeFromFileParams(hdf5_path=hdf5_path, elite_id=elite_id)
    
    return GenomeFromFileResult(
        params=params,
        connection_weights=connection_weights,
        modulation_spec=modulation_spec,
        tonic_activations=tonic_activations,
        eta=eta,
    )
