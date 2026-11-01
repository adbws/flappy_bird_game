# config.py
"""Centralised constants for the Flappy Bird NEAT training pipeline.

Every tunable parameter lives here so the rest of the codebase can
import exactly what it needs without circular dependencies.
"""

# ── Game‑coordinate space (the "virtual" Flappy Bird world) ──────────
GAME_W = 288
GAME_H = 512
GROUND_Y = 460  # ground surface starts here

# ── Game physics ─────────────────────────────────────────────────────
GRAVITY = 0.35
FLAP_STRENGTH = -6.0
MAX_FALL_SPEED = 9.0
BIRD_START_X = 60
BIRD_RADIUS = 12

# ── Pipe settings ────────────────────────────────────────────────────
PIPE_SPEED = 2.6         # slightly gentler horizontal speed (easier for human play)
PIPE_WIDTH = 48
PIPE_GAP = 125           # vertical gap between top and bottom pipe (generous clearance)
PIPE_SPACING = 230       # increased horizontal distance between consecutive pipes
PIPE_MIN_Y = 70          # minimum gap‑centre Y
PIPE_MAX_Y = GROUND_Y - 70  # maximum gap‑centre Y

# ── Population & GA ─────────────────────────────────────────────────
POPULATION_SIZE = 50
NET_LAYERS = [4, 8, 1]   # input → hidden → output
ELITE_COUNT = 5
TOURNAMENT_SIZE = 3
CROSSOVER_RATE = 0.75
MUTATION_RATE = 0.20      # probability per weight
MUTATION_STRENGTH = 0.4   # Gaussian σ
WEIGHT_INIT_RANGE = 1.0

# ── Training ─────────────────────────────────────────────────────────
MAX_GENERATIONS = 40
MAX_STEPS_PER_GEN = 8000
CHECKPOINT_DIR = "checkpoints"

# ── 10 × 5 training‑grid layout ─────────────────────────────────────
GRID_COLS = 10
GRID_ROWS = 5
CELL_W = 100
CELL_H = 130
CELL_GAP = 2
HUD_H = 44

WINDOW_W = GRID_COLS * CELL_W + (GRID_COLS + 1) * CELL_GAP
WINDOW_H = GRID_ROWS * CELL_H + (GRID_ROWS + 1) * CELL_GAP + HUD_H

# Scale factors: game coords → cell‑pixel coords
SCALE_X = CELL_W / GAME_W
SCALE_Y = CELL_H / GAME_H

# ── Showcase‑mode layout (single‑screen) ─────────────────────────────
SHOWCASE_H = 540
SHOWCASE_W = int(SHOWCASE_H * (GAME_W / GAME_H))  # preserves 288:512 aspect ratio (~303)
SHOWCASE_SX = SHOWCASE_W / GAME_W
SHOWCASE_SY = SHOWCASE_H / GAME_H

# ── Colour palette ───────────────────────────────────────────────────
COL_SKY         = (78, 192, 202)
COL_GROUND      = (222, 216, 149)
COL_GROUND_EDGE = (100, 180, 50)
COL_GROUND_LINE = (92, 64, 51)
COL_PIPE        = (115, 191, 46)
COL_PIPE_CAP    = (85, 142, 33)
COL_PIPE_BORDER = (40, 70, 20)
COL_BIRD        = (215, 35, 45)
COL_BIRD_EYE    = (255, 255, 255)
COL_BIRD_BEAK   = (255, 190, 30)
COL_HUD_BG      = (0, 0, 0)
COL_TEXT         = (255, 255, 255)
COL_TEXT_DIM     = (160, 160, 160)
COL_TEXT_GOLD    = (255, 220, 50)
COL_DEAD_OVERLAY = (255, 0, 0, 80)    # RGBA – semi‑transparent red
COL_MENU_BG      = (0, 0, 0)
COL_BTN          = (60, 160, 80)
COL_BTN_HOVER    = (80, 200, 110)
COL_BTN_ALT      = (50, 120, 210)
COL_BTN_ALT_HOVER = (75, 150, 240)

# ── Rendering ────────────────────────────────────────────────────────
TARGET_FPS = 60
