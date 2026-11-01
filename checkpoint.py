# checkpoint.py
"""Checkpoint serialisation — save and load trained populations."""

import os
import pickle
from typing import List, Optional, Tuple

from config import CHECKPOINT_DIR
from genetics import Genome


def _ensure_dir() -> None:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def save_checkpoint(generation: int, genomes: List[Genome],
                    filepath: Optional[str] = None) -> str:
    """Pickle the current generation's genomes to disk.

    Returns the filepath written.
    """
    _ensure_dir()
    if filepath is None:
        filepath = os.path.join(CHECKPOINT_DIR, f"bird_gen_{generation}.pkl")
    data = {
        "generation": generation,
        "genomes": [
            {
                "weights": g.network.get_flat_weights(),
                "layer_sizes": list(g.network.layer_sizes),
                "fitness": g.fitness,
            }
            for g in genomes
        ],
    }
    with open(filepath, "wb") as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return filepath


def load_checkpoint(filepath: str) -> Tuple[int, List[Genome]]:
    """Load a previously saved checkpoint.

    Returns (generation_number, list_of_Genomes).
    """
    from neural_net import NeuralNetwork  # deferred to avoid circular at module level

    with open(filepath, "rb") as fh:
        data = pickle.load(fh)

    genomes: List[Genome] = []
    for entry in data["genomes"]:
        nn = NeuralNetwork(entry["layer_sizes"])
        nn.set_flat_weights(entry["weights"])
        g = Genome(nn)
        g.fitness = entry.get("fitness", 0.0)
        genomes.append(g)

    return data["generation"], genomes


def list_checkpoints() -> List[Tuple[int, str]]:
    """Return a sorted list of ``(generation, filepath)`` pairs."""
    _ensure_dir()
    results = []
    for name in os.listdir(CHECKPOINT_DIR):
        if name.startswith("bird_gen_") and name.endswith(".pkl"):
            try:
                gen = int(name.replace("bird_gen_", "").replace(".pkl", ""))
                results.append((gen, os.path.join(CHECKPOINT_DIR, name)))
            except ValueError:
                continue
    results.sort()
    return results
