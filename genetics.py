# genetics.py
"""Genetic Algorithm engine for neuroevolution.

Provides tournament selection, uniform crossover, Gaussian mutation,
and elitism.  Completely self‑contained — no neat‑python dependency.
"""

import random
from typing import List, Tuple

from config import (
    POPULATION_SIZE,
    NET_LAYERS,
    ELITE_COUNT,
    TOURNAMENT_SIZE,
    CROSSOVER_RATE,
    MUTATION_RATE,
    MUTATION_STRENGTH,
    WEIGHT_INIT_RANGE,
)
from neural_net import NeuralNetwork


# ── Public dataclass‑like container ──────────────────────────────────
class Genome:
    """Pairs a neural network with its evaluated fitness."""

    __slots__ = ("network", "fitness")

    def __init__(self, network: NeuralNetwork):
        self.network = network
        self.fitness = 0.0


class GeneticAlgorithm:
    """Manages a population of Genomes across generations."""

    def __init__(
        self,
        pop_size: int = POPULATION_SIZE,
        net_shape: List[int] = NET_LAYERS,
        elite_count: int = ELITE_COUNT,
        tournament_size: int = TOURNAMENT_SIZE,
        crossover_rate: float = CROSSOVER_RATE,
        mutation_rate: float = MUTATION_RATE,
        mutation_strength: float = MUTATION_STRENGTH,
    ):
        self.pop_size = pop_size
        self.net_shape = list(net_shape)
        self.elite_count = min(elite_count, pop_size)
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength

    # ── Population creation ──────────────────────────────────────────
    def create_population(self) -> List[Genome]:
        return [Genome(NeuralNetwork(self.net_shape)) for _ in range(self.pop_size)]

    # ── Evolution step ───────────────────────────────────────────────
    def evolve(self, genomes: List[Genome]) -> List[Genome]:
        """Produce the next generation from a fitness‑evaluated population."""
        ranked = sorted(genomes, key=lambda g: g.fitness, reverse=True)

        new_pop: List[Genome] = []

        # 1. Elitism — carry the top performers unchanged
        for i in range(self.elite_count):
            elite = Genome(ranked[i].network.copy())
            elite.fitness = 0.0
            new_pop.append(elite)

        # 2. Fill the rest via selection → crossover → mutation
        while len(new_pop) < self.pop_size:
            p1 = self._tournament_select(ranked)
            p2 = self._tournament_select(ranked)

            if random.random() < self.crossover_rate:
                child_net = self._crossover(p1.network, p2.network)
            else:
                child_net = p1.network.copy()

            self._mutate(child_net)
            new_pop.append(Genome(child_net))

        return new_pop[: self.pop_size]

    # ── Selection ────────────────────────────────────────────────────
    def _tournament_select(self, ranked: List[Genome]) -> Genome:
        contestants = random.sample(ranked, min(self.tournament_size, len(ranked)))
        return max(contestants, key=lambda g: g.fitness)

    # ── Crossover ────────────────────────────────────────────────────
    @staticmethod
    def _crossover(a: NeuralNetwork, b: NeuralNetwork) -> NeuralNetwork:
        """Uniform crossover: each weight is taken from a random parent."""
        child = a.copy()
        flat_a = a.get_flat_weights()
        flat_b = b.get_flat_weights()
        flat_c = [
            fa if random.random() < 0.5 else fb
            for fa, fb in zip(flat_a, flat_b)
        ]
        child.set_flat_weights(flat_c)
        return child

    # ── Mutation ─────────────────────────────────────────────────────
    def _mutate(self, net: NeuralNetwork) -> None:
        """Gaussian perturbation with a small chance of full weight reset."""
        flat = net.get_flat_weights()
        for i in range(len(flat)):
            if random.random() < self.mutation_rate:
                if random.random() < 0.05:
                    # Rare full reset
                    flat[i] = random.uniform(-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE)
                else:
                    flat[i] += random.gauss(0, self.mutation_strength)
        net.set_flat_weights(flat)

    # ── Utility ──────────────────────────────────────────────────────
    @staticmethod
    def best_genome(genomes: List[Genome]) -> Genome:
        return max(genomes, key=lambda g: g.fitness)

    @staticmethod
    def stats(genomes: List[Genome]) -> Tuple[float, float, float]:
        """Return (best_fitness, avg_fitness, worst_fitness)."""
        fits = [g.fitness for g in genomes]
        return max(fits), sum(fits) / len(fits), min(fits)
