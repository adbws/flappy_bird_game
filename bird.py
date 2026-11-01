# bird.py
"""Bird entity — physics, collision bounds, and AI observation vector."""

from typing import List, Optional

from config import (
    GRAVITY,
    FLAP_STRENGTH,
    MAX_FALL_SPEED,
    BIRD_START_X,
    BIRD_RADIUS,
    GAME_H,
    GAME_W,
    GROUND_Y,
)


class Bird:
    """Single bird agent.

    Attributes
    ----------
    x, y     : position in game coordinates.
    vel_y    : vertical velocity (positive = falling).
    alive    : whether this bird is still active.
    score    : pipes cleared.
    fitness  : accumulated training reward.
    """

    __slots__ = ("x", "y", "vel_y", "alive", "score", "fitness")

    def __init__(self) -> None:
        self.x: float = float(BIRD_START_X)
        self.y: float = float(GAME_H) / 2.0
        self.vel_y: float = 0.0
        self.alive: bool = True
        self.score: int = 0
        self.fitness: float = 0.0

    # ── Actions ──────────────────────────────────────────────────────
    def flap(self) -> None:
        if self.alive:
            self.vel_y = FLAP_STRENGTH

    # ── Physics step ─────────────────────────────────────────────────
    def update(self) -> None:
        if not self.alive:
            return
        self.vel_y = min(self.vel_y + GRAVITY, MAX_FALL_SPEED)
        self.y += self.vel_y
        self.fitness += 0.1  # survival reward per frame

        # Ceiling / ground death
        if self.y - BIRD_RADIUS <= 0 or self.y + BIRD_RADIUS >= GROUND_Y:
            self.alive = False

    # ── AI observation vector ────────────────────────────────────────
    def get_inputs(self, next_pipe) -> List[float]:
        """Return the normalised 4‑element state vector in ~[-1, 1]."""
        if next_pipe is None:
            return [1.0, 0.0, 0.0, self.vel_y / MAX_FALL_SPEED]

        dx = (next_pipe.x - self.x) / float(GAME_W)
        dy_top = (self.y - next_pipe.gap_top) / float(GAME_H)
        dy_bot = (next_pipe.gap_bottom - self.y) / float(GAME_H)
        vel = self.vel_y / float(MAX_FALL_SPEED)
        return [dx, dy_top, dy_bot, vel]
