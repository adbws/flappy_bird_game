# renderer.py
"""All visual rendering — 10×5 training grid, HUD, showcase view,
and a zero‑dependency 5×7 bitmap font engine.

*No* imports of ``pygame.font`` or ``pygame.freetype`` anywhere, so
this module works on every Python / Pygame / SDL combination including
Python 3.14 + pygame 2.6 on macOS.
"""

from typing import Dict, List, Optional, Tuple
import math
import time
import pygame

from config import (
    GAME_W, GAME_H, GROUND_Y,
    PIPE_WIDTH, PIPE_GAP, BIRD_RADIUS,
    GRID_COLS, GRID_ROWS, CELL_W, CELL_H, CELL_GAP, HUD_H,
    WINDOW_W, WINDOW_H,
    SCALE_X, SCALE_Y,
    SHOWCASE_W, SHOWCASE_H, SHOWCASE_SX, SHOWCASE_SY,
    COL_SKY, COL_GROUND, COL_GROUND_EDGE, COL_GROUND_LINE,
    COL_PIPE, COL_PIPE_CAP, COL_PIPE_BORDER,
    COL_BIRD, COL_BIRD_EYE, COL_BIRD_BEAK,
    COL_HUD_BG, COL_TEXT, COL_TEXT_DIM, COL_TEXT_GOLD,
    COL_DEAD_OVERLAY,
)

# =====================================================================
#  5 × 7  BITMAP FONT (arcade style, zero dependencies)
# =====================================================================
_GLYPHS = {
    'A': [" ### ","#   #","#   #","#####","#   #","#   #","#   #"],
    'B': ["#### ","#   #","#   #","#### ","#   #","#   #","#### "],
    'C': [" ####","#    ","#    ","#    ","#    ","#    "," ####"],
    'D': ["#### ","#   #","#   #","#   #","#   #","#   #","#### "],
    'E': ["#####","#    ","#    ","#### ","#    ","#    ","#####"],
    'F': ["#####","#    ","#    ","#### ","#    ","#    ","#    "],
    'G': [" ####","#    ","#    ","# ###","#   #","#   #"," ####"],
    'H': ["#   #","#   #","#   #","#####","#   #","#   #","#   #"],
    'I': ["#####","  #  ","  #  ","  #  ","  #  ","  #  ","#####"],
    'J': ["  ###","    #","    #","    #","    #","#   #"," ### "],
    'K': ["#   #","#  # ","# #  ","##   ","# #  ","#  # ","#   #"],
    'L': ["#    ","#    ","#    ","#    ","#    ","#    ","#####"],
    'M': ["#   #","## ##","# # #","#   #","#   #","#   #","#   #"],
    'N': ["#   #","##  #","# # #","#  ##","#   #","#   #","#   #"],
    'O': [" ### ","#   #","#   #","#   #","#   #","#   #"," ### "],
    'P': ["#### ","#   #","#   #","#### ","#    ","#    ","#    "],
    'Q': [" ### ","#   #","#   #","#   #","# # #","#  # "," ## #"],
    'R': ["#### ","#   #","#   #","#### ","# #  ","#  # ","#   #"],
    'S': [" ####","#    ","#    "," ### ","    #","    #","#### "],
    'T': ["#####","  #  ","  #  ","  #  ","  #  ","  #  ","  #  "],
    'U': ["#   #","#   #","#   #","#   #","#   #","#   #"," ### "],
    'V': ["#   #","#   #","#   #","#   #","#   #"," # # ","  #  "],
    'W': ["#   #","#   #","#   #","# # #","# # #","## ##","#   #"],
    'X': ["#   #","#   #"," # # ","  #  "," # # ","#   #","#   #"],
    'Y': ["#   #","#   #"," # # ","  #  ","  #  ","  #  ","  #  "],
    'Z': ["#####","    #","   # ","  #  "," #   ","#    ","#####"],
    '0': [" ### ","#  ##","# # #","# # #","# # #","##  #"," ### "],
    '1': ["  #  "," ##  ","  #  ","  #  ","  #  ","  #  "," ### "],
    '2': [" ### ","#   #","    #","  ## "," #   ","#    ","#####"],
    '3': ["#### ","    #","    #"," ### ","    #","    #","#### "],
    '4': ["#   #","#   #","#   #","#####","    #","    #","    #"],
    '5': ["#####","#    ","#    ","#### ","    #","    #","#### "],
    '6': [" ### ","#    ","#    ","#### ","#   #","#   #"," ### "],
    '7': ["#####","    #","   # ","  #  "," #   "," #   "," #   "],
    '8': [" ### ","#   #","#   #"," ### ","#   #","#   #"," ### "],
    '9': [" ### ","#   #","#   #"," ####","    #","    #"," ### "],
    ' ': ["     ","     ","     ","     ","     ","     ","     "],
    ':': ["     ","  #  ","     ","     ","     ","  #  ","     "],
    '.': ["     ","     ","     ","     ","     "," ##  "," ##  "],
    '/': ["    #","   # ","   # ","  #  "," #   "," #   ","#    "],
    '-': ["     ","     ","     ","#####","     ","     ","     "],
    '+': ["     ","  #  ","  #  ","#####","  #  ","  #  ","     "],
    '!': ["  #  ","  #  ","  #  ","  #  ","  #  ","     ","  #  "],
    '|': ["  #  ","  #  ","  #  ","  #  ","  #  ","  #  ","  #  "],
    '(': ["  ## "," #   ","#    ","#    ","#    "," #   ","  ## "],
    ')': [" ##  ","   # ","    #","    #","    #","   # "," ##  "],
    '_': ["     ","     ","     ","     ","     ","     ","#####"],
    '>': ["#    "," #   ","  #  ","   # ","  #  "," #   ","#    "],
    '<': ["    #","   # ","  #  "," #   ","  #  ","   # ","    #"],
    '%': ["##  #","## # ","   # ","  #  "," #   "," # ##","#  ##"],
}

_txt_cache: Dict[Tuple[str, Tuple[int,int,int], int], pygame.Surface] = {}
_card_cache: Dict[Tuple[int, int, Tuple[int, ...], Tuple[int, ...], int, int], pygame.Surface] = {}
_showcase_view: Optional[pygame.Surface] = None


def _get_card_surf(w: int, h: int, bg_col: Tuple[int, ...],
                   border_col: Tuple[int, ...], width: int = 2,
                   radius: int = 8) -> pygame.Surface:
    """Return a cached semi-transparent UI card surface."""
    key = (w, h, bg_col, border_col, width, radius)
    surf = _card_cache.get(key)
    if surf is None:
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        surf.fill(bg_col)
        pygame.draw.rect(surf, border_col, (0, 0, w, h), width=width, border_radius=radius)
        _card_cache[key] = surf
    return surf

def _render_text_surf(text: str, color: Tuple[int,int,int], scale: int) -> pygame.Surface:
    key = (text.upper(), color, scale)
    cached = _txt_cache.get(key)
    if cached is not None:
        return cached
    if len(_txt_cache) > 512:
        _txt_cache.clear()
    chars = list(key[0])
    cw = 5 * scale
    sp = max(1, scale)
    ch = 7 * scale
    w = max(1, len(chars) * (cw + sp) - sp)
    surf = pygame.Surface((w, ch), pygame.SRCALPHA)
    cx = 0
    for c in chars:
        glyph = _GLYPHS.get(c, _GLYPHS[' '])
        for ri, row in enumerate(glyph):
            for ci, px in enumerate(row):
                if px == '#':
                    pygame.draw.rect(surf, color,
                                     (cx + ci * scale, ri * scale, scale, scale))
        cx += cw + sp
    _txt_cache[key] = surf
    return surf


def draw_text(surface: pygame.Surface, text: str,
              pos: Tuple[int,int], color: Tuple[int,int,int] = COL_TEXT,
              scale: int = 2, align: str = "topleft",
              shadow: bool = False) -> pygame.Rect:
    """Render text onto *surface*. Alignment uses ``pygame.Rect`` attribute names."""
    if shadow:
        sh = _render_text_surf(text, (20, 20, 20), scale)
        sr = sh.get_rect(**{align: (pos[0] + max(1, scale // 2), pos[1] + max(1, scale // 2))})
        surface.blit(sh, sr)
    ts = _render_text_surf(text, color, scale)
    rect = ts.get_rect(**{align: pos})
    surface.blit(ts, rect)
    return rect


# =====================================================================
#  CELL RENDERING  — one mini Flappy‑Bird viewport
# =====================================================================
def _draw_cell(surf: pygame.Surface, sim, bird_idx: int,
               sx: float, sy: float, cw: int, ch: int,
               show_sensors: bool = False) -> None:
    """Render a single bird's game‑state into a *cw × ch* surface."""
    surf.fill(COL_SKY)

    # ── Pipes (with dark border) ─────────────────────────────────────
    for pipe in sim.pipe_mgr.pipes:
        px = int(pipe.x * sx)
        pw = max(2, int(PIPE_WIDTH * sx))
        top_h = int(pipe.gap_top * sy)
        bot_y = int(pipe.gap_bottom * sy)
        cap_h = max(2, int(8 * sy))

        if top_h > 0:
            pygame.draw.rect(surf, COL_PIPE, (px, 0, pw, top_h))
            pygame.draw.rect(surf, COL_PIPE_BORDER, (px, 0, pw, top_h), 1)
            pygame.draw.rect(surf, COL_PIPE_CAP, (px - 1, top_h - cap_h, pw + 2, cap_h))
            pygame.draw.rect(surf, COL_PIPE_BORDER, (px - 1, top_h - cap_h, pw + 2, cap_h), 1)
        if bot_y < ch:
            pygame.draw.rect(surf, COL_PIPE, (px, bot_y, pw, ch - bot_y))
            pygame.draw.rect(surf, COL_PIPE_BORDER, (px, bot_y, pw, ch - bot_y), 1)
            pygame.draw.rect(surf, COL_PIPE_CAP, (px - 1, bot_y, pw + 2, cap_h))
            pygame.draw.rect(surf, COL_PIPE_BORDER, (px - 1, bot_y, pw + 2, cap_h), 1)

    # ── Ground (with green grass strip) ──────────────────────────────
    gy = int(GROUND_Y * sy)
    pygame.draw.rect(surf, COL_GROUND, (0, gy, cw, ch - gy))
    grass_h = max(2, int(6 * sy))
    pygame.draw.rect(surf, COL_GROUND_EDGE, (0, gy, cw, grass_h))
    pygame.draw.line(surf, COL_GROUND_LINE, (0, gy), (cw, gy), 1)

    # ── Bird ─────────────────────────────────────────────────────────
    bird = sim.birds[bird_idx]
    bx = int(bird.x * sx)
    by = int(bird.y * sy)
    br = max(2, int(BIRD_RADIUS * min(sx, sy)))

    # ── AI Vision Sensor Lines (Showcase mode only) ──────────────────
    if show_sensors and bird.alive:
        nxt = sim.pipe_mgr.get_next_pipe(bird.x)
        if nxt is not None:
            tx = int(nxt.x * sx)
            ty_top = int(nxt.gap_top * sy)
            ty_bot = int(nxt.gap_bottom * sy)
            ty_mid = int(nxt.gap_center * sy)
            pygame.draw.line(surf, (255, 90, 90), (bx, by), (tx, ty_top), 1)
            pygame.draw.line(surf, (255, 90, 90), (bx, by), (tx, ty_bot), 1)
            pygame.draw.line(surf, (100, 240, 255), (bx, by), (tx, ty_mid), 1)
            pygame.draw.circle(surf, (255, 90, 90), (tx, ty_top), 3)
            pygame.draw.circle(surf, (255, 90, 90), (tx, ty_bot), 3)
            pygame.draw.circle(surf, (100, 240, 255), (tx, ty_mid), 3)

    # Body
    pygame.draw.circle(surf, COL_BIRD, (bx, by), br)
    pygame.draw.circle(surf, (0, 0, 0), (bx, by), br, 1)
    # Eye
    pygame.draw.circle(surf, COL_BIRD_EYE, (bx + br // 2, by - br // 3), max(1, br // 3))

    # Smaller beak with black border for visibility
    beak_len = max(2, int(br * 0.65))
    beak_half_h = max(1, int(br * 0.32))
    beak_pts = [
        (bx + br - 1, by - beak_half_h),
        (bx + br + beak_len, by + 1),
        (bx + br - 1, by + beak_half_h + 1),
    ]
    pygame.draw.polygon(surf, COL_BIRD_BEAK, beak_pts)
    pygame.draw.polygon(surf, (0, 0, 0), beak_pts, 1)


# =====================================================================
#  GRID RENDERER — the 10 × 5 training dashboard
# =====================================================================
class GridRenderer:
    """Manages the 10 × 5 live training grid with frozen‑dead‑cell overlays."""

    def __init__(self) -> None:
        # Pre‑allocate cell surfaces (reused every frame)
        self._cells = [pygame.Surface((CELL_W, CELL_H)) for _ in range(GRID_COLS * GRID_ROWS)]
        # Frozen snapshots for dead birds: idx → Surface
        self._frozen: Dict[int, pygame.Surface] = {}
        # Semi‑transparent red overlay (created once)
        self._red_overlay = pygame.Surface((CELL_W, CELL_H), pygame.SRCALPHA)
        self._red_overlay.fill(COL_DEAD_OVERLAY)

    def reset(self) -> None:
        self._frozen.clear()

    # ── Main draw call ───────────────────────────────────────────────
    def draw(self, screen: pygame.Surface, sim, generation: int,
             fast_forward: bool, paused: bool = False) -> None:
        """Render the full training dashboard for one frame."""
        screen.fill(COL_HUD_BG)

        # ── HUD bar ─────────────────────────────────────────────────
        self._draw_hud(screen, sim, generation, fast_forward, paused)
        alive_indices = [i for i, b in enumerate(sim.birds) if b.alive]
        leader_idx = max(alive_indices, key=lambda i: sim.birds[i].fitness) if alive_indices else -1

        # ── Grid cells ──────────────────────────────────────────────
        for idx in range(sim.pop_size):
            col = idx % GRID_COLS
            row = idx // GRID_COLS
            cx = CELL_GAP + col * (CELL_W + CELL_GAP)
            cy = HUD_H + CELL_GAP + row * (CELL_H + CELL_GAP)

            if idx in self._frozen:
                # Already dead — blit the frozen snapshot
                screen.blit(self._frozen[idx], (cx, cy))
                continue

            # Render live cell
            cell = self._cells[idx]
            _draw_cell(cell, sim, idx, SCALE_X, SCALE_Y, CELL_W, CELL_H)

            if not sim.birds[idx].alive:
                # Bird just died this frame (or an earlier un‑rendered one).
                # Freeze it: copy, apply red overlay, store.
                frozen = cell.copy()
                frozen.blit(self._red_overlay, (0, 0))
                # Draw a small "X" in the centre
                draw_text(frozen, "X", (CELL_W // 2, CELL_H // 2),
                          color=(255, 50, 50), scale=3, align="center",
                          shadow=False)
                self._frozen[idx] = frozen
                screen.blit(frozen, (cx, cy))
            else:
                screen.blit(cell, (cx, cy))

            if idx == leader_idx:
                pygame.draw.rect(screen, COL_TEXT_GOLD, (cx, cy, CELL_W, CELL_H), 2)
                draw_text(screen, "BEST", (cx + 4, cy + 4),
                          color=COL_TEXT_GOLD, scale=1, align="topleft", shadow=True)
            else:
                pygame.draw.rect(screen, (60, 60, 60), (cx, cy, CELL_W, CELL_H), 1)

        # ── Optional pause overlay banner ────────────────────────────
        if paused:
            pw, ph = 320, 56
            px = (WINDOW_W - pw) // 2
            py = HUD_H + (WINDOW_H - HUD_H - ph) // 2
            p_surf = _get_card_surf(pw, ph, (15, 18, 32, 220), (255, 90, 90), 2, 8)
            screen.blit(p_surf, (px, py))
            draw_text(screen, "PAUSED", (px + pw // 2, py + 18),
                      color=(255, 90, 90), scale=3, align="center", shadow=True)
            draw_text(screen, "PRESS [P] TO RESUME", (px + pw // 2, py + 42),
                      color=(220, 225, 235), scale=1, align="center")

        pygame.display.flip()

    # ── HUD ──────────────────────────────────────────────────────────
    @staticmethod
    def _draw_hud(screen: pygame.Surface, sim, generation: int,
                  fast_forward: bool, paused: bool = False) -> None:
        pygame.draw.rect(screen, COL_HUD_BG, (0, 0, WINDOW_W, HUD_H))
        pygame.draw.line(screen, (55, 60, 85), (0, HUD_H - 1), (WINDOW_W, HUD_H - 1), 1)

        y = HUD_H // 2

        # All primary metrics at scale=2 (same size as the rest)
        draw_text(screen, f"GEN:{generation:>3}", (14, y),
                  color=COL_TEXT_GOLD, scale=2, align="midleft")

        draw_text(screen, f"ALIVE:{sim.alive_count():>2}/{sim.pop_size}",
                  (175, y), color=COL_TEXT, scale=2, align="midleft")

        draw_text(screen, f"SCORE:{sim.best_score:>3}",
                  (365, y), color=(140, 240, 140), scale=2, align="midleft")

        # Frame counter: scale=2 (same size as the rest), clean cyan color
        display_frame = (sim.frame // 5) * 5
        draw_text(screen, f"FRAME:{display_frame:>5}",
                  (540, y), color=(100, 220, 255), scale=2, align="midleft", shadow=True)

        # Status badge (top-right)
        if paused:
            badge_text = "PAUSED [P]"
            badge_col = (255, 90, 90)
        elif fast_forward:
            badge_text = "FAST (5X)"
            badge_col = (255, 150, 50)
        else:
            badge_text = "60 FPS"
            badge_col = (100, 220, 130)

        draw_text(screen, badge_text, (WINDOW_W - 14, y),
                  color=badge_col, scale=2, align="midright", shadow=True)

    @staticmethod
    def draw_asap_overlay(screen: pygame.Surface, current_gen: int,
                          gen_idx: int, total_gens: int,
                          step: int, alive_count: int, pop_size: int,
                          anim_time: float) -> None:
        """Render a small animated modal overlay while computing generations ASAP."""
        cw, ch = 380, 140
        cx = (WINDOW_W - cw) // 2
        cy = HUD_H + (WINDOW_H - HUD_H - ch) // 2

        card = _get_card_surf(cw, ch, (15, 20, 40, 235), (100, 220, 255), 2, 10)
        screen.blit(card, (cx, cy))

        draw_text(screen, "FAST TRAINING 1 GEN", (cx + cw // 2, cy + 20),
                  color=COL_TEXT_GOLD, scale=2, align="center", shadow=True)

        draw_text(screen, f"GENERATION {current_gen}  ({gen_idx}/{total_gens})",
                  (cx + cw // 2, cy + 46), color=COL_TEXT, scale=2, align="center")

        draw_text(screen, f"ALIVE:{alive_count}/{pop_size}  |  FRAME:{step}",
                  (cx + cw // 2, cy + 72), color=(140, 230, 255), scale=1, align="center")

        # Animated orbital spinner (8 dots rotating with comet trail)
        spinner_cx = cx + cw // 2
        spinner_cy = cy + 104
        num_dots = 8
        for i in range(num_dots):
            theta = anim_time * 8.0 + (i * 2.0 * math.pi / num_dots)
            dot_x = int(spinner_cx + 36 * math.cos(theta))
            dot_y = int(spinner_cy + 9 * math.sin(theta))
            val = int(80 + 175 * (i / (num_dots - 1)))
            pygame.draw.circle(screen, (val, val, 255), (dot_x, dot_y), 3)

        draw_text(screen, "[G] CANCEL FAST SIM", (cx + cw // 2, cy + 125),
                  color=COL_TEXT_DIM, scale=1, align="center")

        pygame.display.flip()


# =====================================================================
#  SHOWCASE RENDERER — single full‑screen bird for checkpoint playback
# =====================================================================
def _draw_neural_net(screen: pygame.Surface, net, rect: pygame.Rect) -> None:
    """Render a live 4 -> 8 -> 1 neural network diagram with weights and activations."""
    card = _get_card_surf(rect.width, rect.height, (0, 0, 0, 220), (70, 85, 120), 2, 8)
    screen.blit(card, rect.topleft)

    draw_text(screen, "NEURAL NETWORK (4-8-1)", (rect.centerx, rect.top + 16),
              color=COL_TEXT_GOLD, scale=2, align="center", shadow=True)

    layers = net.layer_sizes
    acts = getattr(net, "last_activations", [[0.0] * s for s in layers])
    num_layers = len(layers)

    # Calculate node (x, y) coordinates for each layer
    pad_x, top_y, bot_y = 58, rect.top + 48, rect.bottom - 42
    usable_w = rect.width - 2 * pad_x
    usable_h = bot_y - top_y

    positions: List[List[Tuple[int, int]]] = []
    for li, count in enumerate(layers):
        lx = rect.left + pad_x + int(usable_w * (li / max(1, num_layers - 1)))
        layer_pos = []
        for ni in range(count):
            ly = top_y + usable_h // 2 if count == 1 else top_y + int(usable_h * (ni / (count - 1)))
            layer_pos.append((lx, ly))
        positions.append(layer_pos)

    # 1. Draw synapses (green = positive weight, red = negative weight)
    for li, w_mat in enumerate(net.weights):
        for i, row in enumerate(w_mat):
            p1 = positions[li][i]
            for j, weight in enumerate(row):
                p2 = positions[li + 1][j]
                col = (60, 210, 95) if weight >= 0 else (230, 65, 65)
                thickness = max(1, min(4, int(abs(weight) * 2.2)))
                pygame.draw.line(screen, col, p1, p2, thickness)

    # 2. Draw neurons (brightness scales with current activation)
    input_labels = ["DX", "TOP", "BOT", "VEL"]
    for li, layer_pos in enumerate(positions):
        for ni, (nx, ny) in enumerate(layer_pos):
            val = acts[li][ni] if (li < len(acts) and ni < len(acts[li])) else 0.0
            norm = max(-1.0, min(1.0, val))
            intensity = int(min(1.0, abs(norm)) * 200)
            fill = (55 + intensity, 180, 90) if norm >= 0 else (190, 55 + intensity // 2, 65)

            radius = 10 if li != 1 else 8
            pygame.draw.circle(screen, fill, (nx, ny), radius)
            pygame.draw.circle(screen, (255, 255, 255), (nx, ny), radius, 1)

            if li == 0 and ni < len(input_labels):
                draw_text(screen, input_labels[ni], (nx - 14, ny),
                          color=COL_TEXT_DIM, scale=1, align="midright")
            elif li == num_layers - 1:
                flap_on = val > 0.5
                lbl = "FLAP!" if flap_on else "IDLE"
                lbl_col = (100, 255, 120) if flap_on else COL_TEXT_DIM
                draw_text(screen, lbl, (nx + 14, ny),
                          color=lbl_col, scale=1, align="midleft")

    draw_text(screen, "GREEN: +WEIGHT   RED: -WEIGHT", (rect.centerx, rect.bottom - 16),
              color=COL_TEXT_DIM, scale=1, align="center")


def _draw_fitness_chart(screen: pygame.Surface,
                        history: List[Tuple[int, float, float]],
                        rect: pygame.Rect) -> None:
    """Render a Best vs. Avg fitness progression chart across generations."""
    card = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    card.fill((0, 0, 0, 220))
    pygame.draw.rect(card, (70, 85, 120), (0, 0, rect.width, rect.height), width=2, border_radius=8)
    screen.blit(card, rect.topleft)

    draw_text(screen, "FITNESS PROGRESSION", (rect.centerx, rect.top + 16),
              color=COL_TEXT_GOLD, scale=2, align="center", shadow=True)

    gx = rect.left + 44
    gy = rect.top + 48
    gw = rect.width - 62
    gh = rect.height - 96

    # Axes
    pygame.draw.rect(screen, (0, 0, 0), (gx, gy, gw, gh))
    pygame.draw.line(screen, (90, 105, 140), (gx, gy), (gx, gy + gh), 1)
    pygame.draw.line(screen, (90, 105, 140), (gx, gy + gh), (gx + gw, gy + gh), 1)

    if not history:
        draw_text(screen, "NO DATA", (gx + gw // 2, gy + gh // 2),
                  color=COL_TEXT_DIM, scale=2, align="center")
        return

    max_fit = max(10.0, max(h[1] for h in history) * 1.1)

    # Horizontal grid lines & Y-axis labels
    for i in range(5):
        frac = i / 4.0
        ly = int(gy + gh - frac * gh)
        val = int(frac * max_fit)
        pygame.draw.line(screen, (35, 42, 68), (gx + 1, ly), (gx + gw, ly), 1)
        draw_text(screen, f"{val}", (gx - 6, ly),
                  color=COL_TEXT_DIM, scale=1, align="midright")

    # Plot coordinates
    n = len(history)
    best_pts: List[Tuple[int, int]] = []
    avg_pts: List[Tuple[int, int]] = []
    for idx, (g_num, b_fit, a_fit) in enumerate(history):
        px = gx + gw // 2 if n == 1 else int(gx + (idx / (n - 1)) * gw)
        by_pt = int(gy + gh - min(1.0, max(0.0, b_fit / max_fit)) * gh)
        ay_pt = int(gy + gh - min(1.0, max(0.0, a_fit / max_fit)) * gh)
        best_pts.append((px, by_pt))
        avg_pts.append((px, ay_pt))

    if len(avg_pts) >= 2:
        pygame.draw.lines(screen, (100, 200, 255), False, avg_pts, 2)
    if len(best_pts) >= 2:
        pygame.draw.lines(screen, COL_TEXT_GOLD, False, best_pts, 2)

    for px, py in avg_pts:
        pygame.draw.circle(screen, (100, 200, 255), (px, py), 3)
    for px, py in best_pts:
        pygame.draw.circle(screen, COL_TEXT_GOLD, (px, py), 3)

    # X-axis generation bounds
    draw_text(screen, f"G{history[0][0]}", (gx, gy + gh + 8),
              color=COL_TEXT_DIM, scale=1, align="topleft")
    if n > 1:
        draw_text(screen, f"G{history[-1][0]}", (gx + gw, gy + gh + 8),
                  color=COL_TEXT_DIM, scale=1, align="topright")

    # Legend
    draw_text(screen, "GOLD: BEST   CYAN: AVG", (rect.centerx, rect.bottom - 16),
              color=COL_TEXT_DIM, scale=1, align="center")

def draw_showcase(screen: pygame.Surface, sim, generation: int,
                  score: int, title: Optional[str] = None,
                  fast_forward: bool = False, network=None,
                  paused: bool = False,
                  history: Optional[List[Tuple[int, float, float]]] = None) -> None:
    """Render a single‑bird game at showcase scale, centred on screen."""
    sw, sh = screen.get_size()
    # Center the game viewport when both side panels are visible, or shift left if only NN is shown
    if network is not None and history:
        ox = (sw - SHOWCASE_W) // 2
    elif network is not None:
        ox = (sw - SHOWCASE_W) // 2 - 150
    else:
        ox = (sw - SHOWCASE_W) // 2
    oy = (sh - SHOWCASE_H) // 2

    # Background fill
    screen.fill(COL_HUD_BG)
    global _showcase_view
    if _showcase_view is None:
        _showcase_view = pygame.Surface((SHOWCASE_W, SHOWCASE_H))
    _draw_cell(_showcase_view, sim, 0, SHOWCASE_SX, SHOWCASE_SY, SHOWCASE_W, SHOWCASE_H,
               show_sensors=(network is not None))
    screen.blit(_showcase_view, (ox, oy))
    pygame.draw.rect(screen, COL_TEXT_GOLD, (ox, oy, SHOWCASE_W, SHOWCASE_H), 2)

    # Optional pause overlay inside the showcase game viewport
    if paused:
        pw, ph = 260, 56
        px = ox + (SHOWCASE_W - pw) // 2
        py = oy + (SHOWCASE_H - ph) // 2
        p_surf = pygame.Surface((pw, ph), pygame.SRCALPHA)
        p_surf.fill((15, 18, 32, 225))
        pygame.draw.rect(p_surf, (255, 90, 90), (0, 0, pw, ph), width=2, border_radius=8)
        screen.blit(p_surf, (px, py))
        draw_text(screen, "PAUSED", (px + pw // 2, py + 18),
                  color=(255, 90, 90), scale=3, align="center", shadow=True)
        draw_text(screen, "PRESS [P] TO RESUME", (px + pw // 2, py + 42),
                  color=(220, 225, 235), scale=1, align="center")

    # Left panel: Fitness Progression Chart
    if history:
        chart_rect = pygame.Rect(ox - 24 - 290, oy + 40, 290, SHOWCASE_H - 80)
        _draw_fitness_chart(screen, history, chart_rect)

    # Right panel: Live Neural Network HUD
    if network is not None:
        panel_rect = pygame.Rect(ox + SHOWCASE_W + 24, oy + 40, 290, SHOWCASE_H - 80)
        _draw_neural_net(screen, network, panel_rect)

    # Header label (centered above showcase viewport)
    if title is None:
        header = f"GENERATION {generation} - BEST BRAIN" if generation > 0 else "FLAPPY BIRD - HUMAN PLAY"
    else:
        header = title

    draw_text(screen, header,
              (sw // 2, oy - 22), color=COL_TEXT_GOLD, scale=2, align="midbottom",
              shadow=True)

    # Top-right status badge (Paused or Fast-Forward)
    if paused or fast_forward:
        badge_w, badge_h = 125, 26
        bx = sw - badge_w - 20
        by = max(14, oy - 38)
        badge_surf = pygame.Surface((badge_w, badge_h), pygame.SRCALPHA)
        if paused:
            badge_surf.fill((255, 60, 60, 45))
            pygame.draw.rect(badge_surf, (255, 90, 90), (0, 0, badge_w, badge_h), width=1, border_radius=6)
            screen.blit(badge_surf, (bx, by))
            draw_text(screen, "PAUSED [P]", (bx + badge_w // 2, by + badge_h // 2),
                      color=(255, 90, 90), scale=2, align="center", shadow=True)
        else:
            badge_surf.fill((255, 140, 30, 45))
            pygame.draw.rect(badge_surf, (255, 160, 50), (0, 0, badge_w, badge_h), width=1, border_radius=6)
            screen.blit(badge_surf, (bx, by))
            draw_text(screen, "FAST (5X)", (bx + badge_w // 2, by + badge_h // 2),
                      color=(255, 170, 60), scale=2, align="center", shadow=True)

    # Bottom line: Frame counter at bottom (same size as score: scale=3)
    display_frame = (sim.frame // 5) * 5
    stats_str = f"SCORE:{score:>3}    FRAME:{display_frame:>5}"
    draw_text(screen, stats_str,
              (sw // 2, oy + SHOWCASE_H + 18), color=COL_TEXT, scale=3,
              align="midtop", shadow=True)

    pygame.display.flip()
