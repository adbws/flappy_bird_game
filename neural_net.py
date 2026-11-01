# neural_net.py
"""Lightweight feed‑forward neural network for neuroevolution.

Zero external dependencies — pure‑Python math only.
The network is small (≈49 parameters for a 4→8→1 topology) so raw Python
lists are fast enough and avoid a numpy install requirement.
"""

import math
import random
from typing import List


class NeuralNetwork:
    """Simple dense feed‑forward network with configurable layer sizes.

    Hidden layers use *tanh*; the output layer uses *sigmoid* so the
    action probability sits in [0, 1].
    """

    def __init__(self, layer_sizes: List[int]):
        self.layer_sizes = list(layer_sizes)
        self.weights: List[List[List[float]]] = []
        self.biases: List[List[float]] = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            # Xavier‑ish initialisation
            scale = 1.0 / math.sqrt(fan_in)
            w = [
                [random.gauss(0, scale) for _ in range(fan_out)]
                for _ in range(fan_in)
            ]
            b = [random.gauss(0, scale) for _ in range(fan_out)]
            self.weights.append(w)
            self.biases.append(b)
        self.last_activations: List[List[float]] = [
            [0.0] * size for size in self.layer_sizes
        ]

    # ── Forward pass ─────────────────────────────────────────────────
    def activate(self, inputs: List[float]) -> List[float]:
        """Run the network and return the output layer activations."""
        current = list(inputs)
        self.last_activations = [current]
        for layer_idx, (w, b) in enumerate(zip(self.weights, self.biases)):
            out_size = len(b)
            in_size = len(current)
            next_layer: List[float] = []
            for j in range(out_size):
                val = sum(current[i] * w[i][j] for i in range(in_size)) + b[j]
                if layer_idx < len(self.weights) - 1:
                    val = math.tanh(val)
                else:
                    val = _sigmoid(val)
                next_layer.append(val)
            current = next_layer
            self.last_activations.append(current)
        return current

    # ── Serialisation helpers ────────────────────────────────────────
    def get_flat_weights(self) -> List[float]:
        """Flatten all weights and biases into a single list."""
        flat: List[float] = []
        for w, b in zip(self.weights, self.biases):
            for row in w:
                flat.extend(row)
            flat.extend(b)
        return flat

    def set_flat_weights(self, flat: List[float]) -> None:
        """Restore weights and biases from a flat list."""
        idx = 0
        for li in range(len(self.weights)):
            for i in range(len(self.weights[li])):
                for j in range(len(self.weights[li][i])):
                    self.weights[li][i][j] = flat[idx]
                    idx += 1
            for j in range(len(self.biases[li])):
                self.biases[li][j] = flat[idx]
                idx += 1

    def copy(self) -> "NeuralNetwork":
        """Return a deep copy of this network."""
        nn = NeuralNetwork.__new__(NeuralNetwork)
        nn.layer_sizes = list(self.layer_sizes)
        nn.weights = [[row[:] for row in w] for w in self.weights]
        nn.biases = [b[:] for b in self.biases]
        return nn

    def param_count(self) -> int:
        return len(self.get_flat_weights())


# ── Activation helpers ───────────────────────────────────────────────
def _sigmoid(x: float) -> float:
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))
