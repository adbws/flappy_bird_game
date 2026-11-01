# pipe.py
"""Pipe obstacles and the shared PipeManager for a generation."""

import random as _random
from typing import List, Optional

from config import (
    PIPE_SPEED,
    PIPE_WIDTH,
    PIPE_GAP,
    PIPE_SPACING,
    PIPE_MIN_Y,
    PIPE_MAX_Y,
    GAME_W,
    BIRD_RADIUS,
)


class Pipe:
    """A single pair of top + bottom pipes with a fixed‑size gap.

    Attributes
    ----------
    x          : horizontal position of the pipe's left edge.
    gap_center : vertical centre of the gap (randomised at spawn).
    passed     : whether a bird has already cleared this pipe.
    """

    __slots__ = ("x", "gap_center", "passed")

    def __init__(self, x: float, gap_center: float):
        self.x = x
        self.gap_center = gap_center
        self.passed = False

    @property
    def gap_top(self) -> float:
        """Y of the bottom edge of the *top* pipe (top of the gap)."""
        return self.gap_center - PIPE_GAP / 2.0

    @property
    def gap_bottom(self) -> float:
        """Y of the top edge of the *bottom* pipe (bottom of the gap)."""
        return self.gap_center + PIPE_GAP / 2.0

    # ── Collision ────────────────────────────────────────────────────
    def collides(self, bird) -> bool:
        if not bird.alive:
            return False

        # Quick horizontal bounding-box rejection
        if bird.x + BIRD_RADIUS < self.x or bird.x - BIRD_RADIUS > self.x + PIPE_WIDTH:
            return False

        # Closest X on the pipe's horizontal span to the bird's center
        closest_x = max(self.x, min(bird.x, self.x + PIPE_WIDTH))

        # Check circle collision against top pipe (0 -> gap_top) and bottom pipe (gap_bottom -> inf)
        for rect_top, rect_bot in ((0.0, self.gap_top), (self.gap_bottom, 9999.0)):
            closest_y = max(rect_top, min(bird.y, rect_bot))
            dx = bird.x - closest_x
            dy = bird.y - closest_y
            if (dx * dx + dy * dy) < (BIRD_RADIUS * BIRD_RADIUS):
                return True

        return False


class PipeManager:
    """Spawns, moves, and culls pipes for one generation.

    A single PipeManager is shared by every bird in the generation so
    they all face the same obstacle sequence.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = _random.Random(seed)
        self.pipes: List[Pipe] = []
        self._next_x: float = GAME_W + 60  # first pipe appears off‑screen right
        self._spawn_ahead()

    # ── Public interface ─────────────────────────────────────────────
    def update(self) -> None:
        """Move all pipes left and maintain the spawn buffer."""
        for p in self.pipes:
            p.x -= PIPE_SPEED
        self._next_x -= PIPE_SPEED
        # Remove pipes that have left the screen
        self.pipes = [p for p in self.pipes if p.x + PIPE_WIDTH > -20]
        self._spawn_ahead()

    def get_next_pipe(self, bird_x: float) -> Optional[Pipe]:
        """Return the closest pipe whose right edge is still ahead of *bird_x*."""
        candidates = [p for p in self.pipes if p.x + PIPE_WIDTH >= bird_x]
        if not candidates:
            return None
        return min(candidates, key=lambda p: p.x)

    # ── Internals ────────────────────────────────────────────────────
    def _spawn_ahead(self) -> None:
        """Ensure there are always pipes extending past the right edge."""
        if self.pipes:
            self._next_x = max(self._next_x, max(p.x for p in self.pipes) + PIPE_SPACING)
        else:
            self._next_x = max(self._next_x, GAME_W + 60)

        while self._next_x < GAME_W * 2.5:
            gap_y = self.rng.randint(int(PIPE_MIN_Y), int(PIPE_MAX_Y))
            self.pipes.append(Pipe(self._next_x, float(gap_y)))
            self._next_x += PIPE_SPACING
