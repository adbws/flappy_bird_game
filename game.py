# game.py
"""Generation‑level simulation: N birds sharing one PipeManager."""

from typing import Dict, List, Optional

from config import (
    POPULATION_SIZE,
    PIPE_WIDTH,
    BIRD_START_X,
    BIRD_RADIUS,
    MAX_STEPS_PER_GEN,
)
from bird import Bird
from pipe import PipeManager


class GenerationSim:
    """Run a single generation where *pop_size* birds face the same pipes.

    After calling :meth:`step` repeatedly until :meth:`all_dead` returns
    ``True``, read ``birds[i].fitness`` and ``birds[i].score`` for each
    agent's performance.
    """

    def __init__(self, pop_size: int = POPULATION_SIZE, seed: Optional[int] = None):
        self.pop_size = pop_size
        self.birds: List[Bird] = [Bird() for _ in range(pop_size)]
        self.pipe_mgr = PipeManager(seed=seed)
        self.frame: int = 0
        self.best_score: int = 0

    # ── Main step ────────────────────────────────────────────────────
    def step(self, actions: Dict[int, int]) -> None:
        """Advance the simulation one frame.

        Parameters
        ----------
        actions : dict mapping bird index → 0 (idle) or 1 (flap).
        """
        self.frame += 1

        # 1. Apply actions
        for idx, act in actions.items():
            if act == 1 and self.birds[idx].alive:
                self.birds[idx].flap()

        # 2. Bird physics
        for bird in self.birds:
            bird.update()

        # 3. Pipe movement
        self.pipe_mgr.update()

        # 4. Collisions & scoring
        for bird in self.birds:
            if not bird.alive:
                continue
            bx_max = bird.x + BIRD_RADIUS
            bx_min = bird.x - BIRD_RADIUS
            for pipe in self.pipe_mgr.pipes:
                # Pipes are sorted by X; if this pipe is ahead of the bird,
                # all subsequent pipes are even further right -> break early.
                if pipe.x > bx_max:
                    break
                # Skip pipes completely behind the bird
                if pipe.x + PIPE_WIDTH < bx_min:
                    continue
                if pipe.collides(bird):
                    bird.alive = False
                    # Shaping: reward birds that died closer to the centre of the gap
                    dist_from_gap = abs(bird.y - pipe.gap_center)
                    bird.fitness = max(0.1, bird.fitness - (dist_from_gap / 100.0))
                    break

        # Scoring: when right edge of a pipe passes the bird column
        for pipe in self.pipe_mgr.pipes:
            if not pipe.passed and pipe.x + PIPE_WIDTH < BIRD_START_X:
                pipe.passed = True
                for bird in self.birds:
                    if bird.alive:
                        bird.score += 1
                        bird.fitness += 10.0  # big bonus for clearing a pipe

        self.best_score = max(b.score for b in self.birds)

    # ── Queries ──────────────────────────────────────────────────────
    def get_observations(self) -> Dict[int, List[float]]:
        """Return the 4‑element AI input vector for every living bird."""
        obs: Dict[int, List[float]] = {}
        for i, bird in enumerate(self.birds):
            if bird.alive:
                nxt = self.pipe_mgr.get_next_pipe(bird.x)
                obs[i] = bird.get_inputs(nxt)
        return obs

    def all_dead(self) -> bool:
        return all(not b.alive for b in self.birds)

    def alive_count(self) -> int:
        return sum(1 for b in self.birds if b.alive)
