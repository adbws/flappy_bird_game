# main.py
"""Entry point — training pipeline, showcase mode, and interactive menus.

Usage
-----
    python main.py                  # launch interactive menu
    python main.py --train          # start fresh training immediately
    python main.py --load GEN       # load and showcase a generation (press Space to replay)
    python main.py --load GEN --ar  # load and showcase with auto-restart on death
    python main.py --human          # play as a human (space/up to flap)
    python main.py --fullscreen     # launch directly in fullscreen mode

Controls
--------
    Hold [F]        Fast‑forward animation speed (Standard 60 FPS is default)
    Toggle [G]      Fast-train 1 generation ASAP (with animation overlay)
    Toggle [P]      Pause / resume training animation
    [F11] / [Cmd+F] Toggle fullscreen (or click macOS green maximize button)
    [ESC]           Return to menu / quit
"""

import argparse
import random
import sys
import time
from typing import Dict, List, Optional
import os

import pygame

from config import (
    POPULATION_SIZE, NET_LAYERS,
    MAX_GENERATIONS, MAX_STEPS_PER_GEN,
    CHECKPOINT_DIR,
    WINDOW_W, WINDOW_H,
    SHOWCASE_W, SHOWCASE_H,
    TARGET_FPS,
    COL_MENU_BG, COL_BTN, COL_BTN_HOVER,
    COL_BTN_ALT, COL_BTN_ALT_HOVER,
    COL_TEXT, COL_TEXT_DIM, COL_TEXT_GOLD,
    GAME_W, GAME_H,
)
from neural_net import NeuralNetwork
from genetics import GeneticAlgorithm, Genome
from game import GenerationSim
from renderer import GridRenderer, draw_text, draw_showcase
from checkpoint import save_checkpoint, load_checkpoint, list_checkpoints


def toggle_fullscreen() -> None:
    """Toggle macOS native / desktop fullscreen."""
    try:
        pygame.display.toggle_fullscreen()
    except Exception:
        pass


# =====================================================================
#  MENUS (bitmap‑font based, no pygame.font dependency)
# =====================================================================
class _Button:
    """Clickable rectangle with hover highlight and keyboard shortcut."""

    def __init__(self, rect: pygame.Rect, label: str, key: int,
                 col: tuple, col_h: tuple):
        self.rect = rect
        self.label = label
        self.key = key
        self.col = col
        self.col_h = col_h

    def draw(self, surf: pygame.Surface, mouse: tuple) -> None:
        hover = self.rect.collidepoint(mouse)
        pygame.draw.rect(surf, (20, 20, 20), self.rect.move(2, 3),
                         border_radius=6)
        pygame.draw.rect(surf, self.col_h if hover else self.col,
                         self.rect, border_radius=6)
        pygame.draw.rect(surf, (255, 255, 255), self.rect, 2,
                         border_radius=6)
        draw_text(surf, self.label, self.rect.center,
                  color=COL_TEXT, scale=2, align="center")

    def hit(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self.rect.collidepoint(event.pos)
        if event.type == pygame.KEYDOWN:
            return event.key == self.key
        return False


def main_menu(screen: pygame.Surface) -> str:
    """Show the main menu.  Returns one of 'train', 'load', 'human', 'quit'."""
    clock = pygame.time.Clock()
    cx = WINDOW_W // 2
    bw, bh = 280, 50

    buttons = [
        _Button(pygame.Rect(cx - bw // 2, 220, bw, bh),
                "1. FRESH TRAINING", pygame.K_1, COL_BTN, COL_BTN_HOVER),
        _Button(pygame.Rect(cx - bw // 2, 290, bw, bh),
                "2. LOAD CHECKPOINT", pygame.K_2, COL_BTN_ALT, COL_BTN_ALT_HOVER),
        _Button(pygame.Rect(cx - bw // 2, 360, bw, bh),
                "3. PLAY AS HUMAN", pygame.K_3, (180, 100, 40), (210, 130, 60)),
    ]

    while True:
        mouse = pygame.mouse.get_pos()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return "quit"
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    return "quit"
                if (ev.key == pygame.K_F11) or (ev.key == pygame.K_f and (ev.mod & (pygame.KMOD_META | pygame.KMOD_CTRL))):
                    toggle_fullscreen()
            if buttons[0].hit(ev):
                return "train"
            if buttons[1].hit(ev):
                return "load"
            if buttons[2].hit(ev):
                return "human"

        screen.fill(COL_MENU_BG)
        draw_text(screen, "FLAPPY BIRD", (cx, 70),
                  color=COL_TEXT_GOLD, scale=5, align="center", shadow=True)
        draw_text(screen, "NEUROEVOLUTION TRAINER", (cx, 130),
                  color=COL_TEXT, scale=2, align="center")
        draw_text(screen, f"POPULATION: {POPULATION_SIZE}  |  NET: {NET_LAYERS}",
                  (cx, 170), color=COL_TEXT_DIM, scale=1, align="center")
        for b in buttons:
            b.draw(screen, mouse)
        draw_text(screen, "ESC: QUIT  |  F11 / CMD+F: FULLSCREEN", (cx, WINDOW_H - 30),
                  color=COL_TEXT_DIM, scale=1, align="center")
        pygame.display.flip()
        clock.tick(30)


def checkpoint_menu(screen: pygame.Surface) -> Optional[str]:
    """Let the user pick a checkpoint file. Supports paginated grid for all generations."""
    clock = pygame.time.Clock()
    cx = WINDOW_W // 2
    available = list_checkpoints()

    if not available:
        # No checkpoints — show a brief message then return None
        for _ in range(75):
            screen.fill(COL_MENU_BG)
            draw_text(screen, "NO CHECKPOINTS FOUND", (cx, WINDOW_H // 2 - 20),
                      color=(255, 80, 80), scale=3, align="center")
            draw_text(screen, f"TRAIN FIRST TO CREATE FILES IN ./{CHECKPOINT_DIR}/",
                      (cx, WINDOW_H // 2 + 30), color=COL_TEXT_DIM, scale=1,
                      align="center")
            pygame.display.flip()
            clock.tick(30)
            for ev in pygame.event.get():
                if ev.type in (pygame.QUIT, pygame.KEYDOWN):
                    return None
        return None

    page = 0
    per_page = 28  # 4 columns x 7 rows
    cols = 4
    bw, bh = 150, 36
    gap_x, gap_y = 16, 10
    total_grid_w = cols * bw + (cols - 1) * gap_x
    start_x = (WINDOW_W - total_grid_w) // 2
    start_y = 160

    btn_prev = _Button(pygame.Rect(start_x, WINDOW_H - 65, 140, 34),
                       "< PREV", pygame.K_LEFT, (70, 80, 100), (90, 105, 130))
    btn_next = _Button(pygame.Rect(start_x + total_grid_w - 140, WINDOW_H - 65, 140, 34),
                       "NEXT >", pygame.K_RIGHT, (70, 80, 100), (90, 105, 130))

    while True:
        max_page = max(0, (len(available) - 1) // per_page)
        page = max(0, min(page, max_page))
        page_items = available[page * per_page : (page + 1) * per_page]

        buttons = []
        for i, (gen_num, fpath) in enumerate(page_items):
            c = i % cols
            r = i // cols
            bx = start_x + c * (bw + gap_x)
            by = start_y + r * (bh + gap_y)
            rect = pygame.Rect(bx, by, bw, bh)
            key = (pygame.K_1 + i) if (i < 9 and page == 0) else -1
            btn = _Button(rect, f"GEN {gen_num}", key, COL_BTN_ALT, COL_BTN_ALT_HOVER)
            buttons.append((btn, fpath))

        mouse = pygame.mouse.get_pos()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return None
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    return None
                if (ev.key == pygame.K_F11) or (ev.key == pygame.K_f and (ev.mod & (pygame.KMOD_META | pygame.KMOD_CTRL))):
                    toggle_fullscreen()
                if ev.key == pygame.K_LEFT and page > 0:
                    page -= 1
                if ev.key == pygame.K_RIGHT and page < max_page:
                    page += 1

            if page > 0 and btn_prev.hit(ev):
                page -= 1
            if page < max_page and btn_next.hit(ev):
                page += 1

            for btn, fpath in buttons:
                if btn.hit(ev):
                    return fpath

        screen.fill(COL_MENU_BG)
        draw_text(screen, "SELECT CHECKPOINT TO LOAD", (cx, 60),
                  color=COL_TEXT_GOLD, scale=3, align="center", shadow=True)
        draw_text(screen, f"PAGE {page + 1}/{max_page + 1}  (TOTAL SAVED: {len(available)})",
                  (cx, 105), color=COL_TEXT, scale=2, align="center")
        draw_text(screen, "CLICK ANY GENERATION TO LAUNCH SHOWCASE", (cx, 135),
                  color=COL_TEXT_DIM, scale=1, align="center")

        for btn, _ in buttons:
            btn.draw(screen, mouse)

        if max_page > 0:
            if page > 0:
                btn_prev.draw(screen, mouse)
            if page < max_page:
                btn_next.draw(screen, mouse)

        draw_text(screen, "ESC: BACK  |  LEFT/RIGHT: PAGE", (cx, WINDOW_H - 22),
                  color=COL_TEXT_DIM, scale=1, align="center")
        pygame.display.flip()
        clock.tick(30)


# =====================================================================
#  TRAINING LOOP
# =====================================================================
def run_training(screen: pygame.Surface) -> None:
    """Main GA training loop with the 10×5 grid visualisation."""
    ga = GeneticAlgorithm()
    genomes = ga.create_population()
    grid = GridRenderer()
    clock = pygame.time.Clock()

    print("\n" + "=" * 60)
    print("  NEUROEVOLUTION TRAINING STARTED")
    print(f"  Population: {POPULATION_SIZE}  |  Network: {NET_LAYERS}")
    print("  Controls:")
    print("    Hold [F]        : Fast‑forward animation speed")
    print("    Toggle [G]      : Fast-train 1 generation ASAP (with animation)")
    print("    Toggle [P]      : Pause / Resume animation")
    print("    [F11] / [Cmd+F] : Fullscreen")
    print("    [ESC]           : Abort to menu")
    print("=" * 60 + "\n")

    gen = 1
    paused = False
    asap_gens_left = 0
    asap_total = 1  # Changed from 5 to 1

    while gen <= MAX_GENERATIONS:
        sim = GenerationSim(pop_size=POPULATION_SIZE, seed=gen * 42)
        grid.reset()
        step = 0
        abort = False
        last_overlay_render = 0.0

        # Initial render when starting a visually-rendered generation
        if asap_gens_left == 0 and not paused:
            grid.draw(screen, sim, gen, fast_forward=False, paused=False)

        while step < MAX_STEPS_PER_GEN and not sim.all_dead():
            now = time.time()

            # ── 1. ASAP 1-Generation Mode ────────────────────────────
            if asap_gens_left > 0:
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        print(f"\n  [Gen {gen}] Training aborted by user. Generation {gen} was not saved.")
                        return
                    if ev.type == pygame.KEYDOWN:
                        if ev.key == pygame.K_ESCAPE:
                            print(f"\n  [Gen {gen}] Training aborted by user [ESC]. Generation {gen} was not saved.")
                            return
                        if ev.key == pygame.K_g:
                            asap_gens_left = 0
                            print("  >> ASAP training cancelled by user [G].")
                            break
                        if (ev.key == pygame.K_F11) or (ev.key == pygame.K_f and (ev.mod & (pygame.KMOD_META | pygame.KMOD_CTRL))):
                            toggle_fullscreen()

                # Advance up to 35 steps per cycle at maximum CPU speed
                for _ in range(35):
                    if step >= MAX_STEPS_PER_GEN or sim.all_dead():
                        break
                    observations = sim.get_observations()
                    actions: Dict[int, int] = {}
                    for idx, obs in observations.items():
                        output = genomes[idx].network.activate(obs)
                        actions[idx] = 1 if output[0] > 0.5 else 0
                    sim.step(actions)
                    step += 1

                # Update animated overlay ~40 times per second (smooth spinning & zero freeze)
                if now - last_overlay_render >= 0.025:
                    last_overlay_render = now
                    gen_idx = asap_total - asap_gens_left + 1
                    GridRenderer.draw_asap_overlay(
                        screen=screen,
                        current_gen=gen,
                        gen_idx=gen_idx,
                        total_gens=asap_total,
                        step=step,
                        alive_count=sim.alive_count(),
                        pop_size=POPULATION_SIZE,
                        anim_time=now,
                    )
                continue

            # ── 2. Paused State ──────────────────────────────────────
            if paused:
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        print(f"\n  [Gen {gen}] Training aborted by user. Generation {gen} was not saved.")
                        return
                    if ev.type == pygame.KEYDOWN:
                        if ev.key == pygame.K_ESCAPE:
                            print(f"\n  [Gen {gen}] Training aborted by user [ESC]. Generation {gen} was not saved.")
                            return
                        if ev.key == pygame.K_p:
                            paused = False
                            print("  >> Resumed [P]")
                        if ev.key == pygame.K_g:
                            paused = False
                            asap_gens_left = 1
                            asap_total = 1
                            print(f"  [Gen {gen}] >> Fast-training 1 generation ASAP [G]...")
                        if (ev.key == pygame.K_F11) or (ev.key == pygame.K_f and (ev.mod & (pygame.KMOD_META | pygame.KMOD_CTRL))):
                            toggle_fullscreen()

                grid.draw(screen, sim, gen, fast_forward=False, paused=True)
                clock.tick(30)
                continue

            # ── 3. Normal / Fast-Forward Visual State ────────────────
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    print(f"\n  [Gen {gen}] Training aborted by user. Generation {gen} was not saved.")
                    return
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        print(f"\n  [Gen {gen}] Training aborted by user [ESC]. Generation {gen} was not saved.")
                        return
                    if ev.key == pygame.K_p:
                        paused = True
                        print("  >> Paused [P]")
                        break
                    if ev.key == pygame.K_g:
                        asap_gens_left = 1
                        asap_total = 1
                        print(f"  [Gen {gen}] >> Fast-training 1 generation ASAP [G]...")
                        break
                    if (ev.key == pygame.K_F11) or (ev.key == pygame.K_f and (ev.mod & (pygame.KMOD_META | pygame.KMOD_CTRL))):
                        toggle_fullscreen()

            if paused or (asap_gens_left > 0):
                continue

            # Check if user is holding the 'F' key (without Cmd/Ctrl)
            keys = pygame.key.get_pressed()
            mods = pygame.key.get_mods()
            is_cmd_ctrl = bool(mods & (pygame.KMOD_META | pygame.KMOD_CTRL))
            fast_forward = bool(keys[pygame.K_f]) and not is_cmd_ctrl

            # Standard mode (default): 1 simulation step per frame at 60 FPS
            # Fast-forward mode: 5 simulation steps per frame at high FPS (visible fast animation)
            sub_steps = 5 if fast_forward else 1

            for _ in range(sub_steps):
                if step >= MAX_STEPS_PER_GEN or sim.all_dead():
                    break
                observations = sim.get_observations()
                actions: Dict[int, int] = {}
                for idx, obs in observations.items():
                    output = genomes[idx].network.activate(obs)
                    actions[idx] = 1 if output[0] > 0.5 else 0

                sim.step(actions)
                step += 1

            # Render grid every frame (NEVER freezes)
            grid.draw(screen, sim, gen, fast_forward, paused=False)

            if fast_forward:
                clock.tick(180)  # high FPS rate for fast animation
            else:
                clock.tick(TARGET_FPS)  # standard 60 FPS

        # ── Transfer fitness from birds to genomes ───────────────────
        for i, bird in enumerate(sim.birds):
            genomes[i].fitness = bird.fitness

        best_fit, avg_fit, _ = ga.stats(genomes)
        best_score = max(b.score for b in sim.birds)
        asap_tag = " (ASAP)" if (asap_gens_left > 0) else ""
        print(f"  Gen {gen:>3}{asap_tag} | Best fitness: {best_fit:>8.1f} | "
              f"Avg: {avg_fit:>7.1f} | Best score: {best_score:>3} | "
              f"Frames: {sim.frame}")

        # ── Save checkpoint for EVERY completed generation ───────────
        fp = save_checkpoint(gen, genomes)
        print(f"         Checkpoint saved → {fp}")

        if asap_gens_left > 0:
            asap_gens_left -= 1
            if asap_gens_left == 0:
                print(f"  >> 1 generation ASAP complete! Resumed visual mode at Gen {gen + 1}.")

        # ── Evolve ───────────────────────────────────────────────────
        genomes = ga.evolve(genomes)
        gen += 1

    print(f"\n  Training completed through Generation {MAX_GENERATIONS}!")
    print(f"  All checkpoints stored in ./{CHECKPOINT_DIR}/.\n")


# =====================================================================
#  SHOWCASE MODE  — run the best genome from a checkpoint
# =====================================================================
def run_showcase(screen: pygame.Surface, filepath: str, auto_restart: bool = False) -> None:
    """Load a checkpoint and let the best genome play in a large single view."""
    gen, genomes = load_checkpoint(filepath)
    best = max(genomes, key=lambda g: g.fitness)
    history = []
    for g_num, fpath in list_checkpoints():
        if g_num <= gen:
            try:
                _, g_pop = load_checkpoint(fpath)
                fits = [g.fitness for g in g_pop]
                if fits:
                    history.append((g_num, max(fits), sum(fits) / len(fits)))
            except Exception:
                pass

    clock = pygame.time.Clock()

    print(f"\n  Showcase: Generation {gen} (best fitness {best.fitness:.1f})")
    print("  Controls:")
    print("    Hold [F]        : Fast-forward animation speed")
    print("    Toggle [P]      : Pause / Resume playback")
    print("    [F11] / [Cmd+F] : Toggle Fullscreen")
    print("    [ESC]           : Return to menu")
    if auto_restart:
        print("    Auto-restart    : ENABLED (--ar)")
    else:
        print("    [SPACE / ENTER] : Replay when trial ends (pass --ar to auto-restart)")
    print()

    paused = False

    while True:
        sim = GenerationSim(pop_size=1, seed=random.randint(1, 99999))
        running = True

        while running and not sim.all_dead():
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    return
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        return
                    if ev.key == pygame.K_p:
                        paused = not paused
                        print("  >> Showcase Paused [P]" if paused else "  >> Showcase Resumed [P]")
                    if (ev.key == pygame.K_F11) or (ev.key == pygame.K_f and (ev.mod & (pygame.KMOD_META | pygame.KMOD_CTRL))):
                        toggle_fullscreen()

            if paused:
                draw_showcase(
                    screen, sim, gen, sim.birds[0].score,
                    fast_forward=False, network=best.network,
                    paused=True, history=history
                )
                clock.tick(30)
                continue

            # Fast-forward mode while holding F (same as train mode)
            keys = pygame.key.get_pressed()
            mods = pygame.key.get_mods()
            is_cmd_ctrl = bool(mods & (pygame.KMOD_META | pygame.KMOD_CTRL))
            fast_forward = bool(keys[pygame.K_f]) and not is_cmd_ctrl

            sub_steps = 5 if fast_forward else 1
            for _ in range(sub_steps):
                if not sim.birds[0].alive:
                    running = False
                    break
                obs = sim.get_observations()
                if 0 in obs:
                    out = best.network.activate(obs[0])
                    sim.step({0: 1 if out[0] > 0.5 else 0})
                else:
                    sim.step({})

            draw_showcase(
                screen, sim, gen, sim.birds[0].score,
                fast_forward=fast_forward, network=best.network,
                paused=False, history=history
            )

            if fast_forward:
                clock.tick(180)
            else:
                clock.tick(TARGET_FPS)

        # ── Once trial is over ───────────────────────────────────────
        final_score = sim.birds[0].score
        final_frame = sim.frame

        if auto_restart:
            t0 = time.time()
            while time.time() - t0 < 0.8:
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        return
                    if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                        return
                    if (ev.type == pygame.KEYDOWN and ev.key == pygame.K_F11) or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_f and (ev.mod & (pygame.KMOD_META | pygame.KMOD_CTRL))):
                        toggle_fullscreen()
                clock.tick(60)
        else:
            # Draw game-over overlay and wait for user input (don't autostart)
            sw, sh = screen.get_size()
            cw, ch = 360, 150
            cx = (sw - cw) // 2
            cy = (sh - ch) // 2
            card = pygame.Surface((cw, ch), pygame.SRCALPHA)
            card.fill((15, 20, 40, 230))
            pygame.draw.rect(card, (255, 220, 50), (0, 0, cw, ch), width=2, border_radius=10)
            screen.blit(card, (cx, cy))

            draw_text(screen, "TRIAL OVER", (cx + cw // 2, cy + 22),
                      color=(255, 90, 90), scale=3, align="center", shadow=True)
            draw_text(screen, f"SCORE: {final_score}   FRAME: {final_frame}",
                      (cx + cw // 2, cy + 60), color=COL_TEXT, scale=2, align="center")
            draw_text(screen, "PRESS [SPACE] TO REPLAY", (cx + cw // 2, cy + 95),
                      color=(120, 230, 130), scale=2, align="center")
            draw_text(screen, "ESC: MENU  |  PASS --AR FOR AUTO-RESTART", (cx + cw // 2, cy + 128),
                      color=COL_TEXT_DIM, scale=1, align="center")

            pygame.display.flip()

            waiting = True
            while waiting:
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        return
                    if ev.type == pygame.KEYDOWN:
                        if ev.key == pygame.K_ESCAPE:
                            return
                        if (ev.key == pygame.K_F11) or (ev.key == pygame.K_f and (ev.mod & (pygame.KMOD_META | pygame.KMOD_CTRL))):
                            toggle_fullscreen()
                        if ev.key in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_r):
                            waiting = False
                    if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                        waiting = False
                clock.tick(30)


# =====================================================================
#  HUMAN PLAY MODE
# =====================================================================
def run_human(screen: pygame.Surface) -> None:
    """Let the user play Flappy Bird manually."""
    from renderer import draw_showcase as _draw
    clock = pygame.time.Clock()
    high_score = 0

    while True:
        sim = GenerationSim(pop_size=1, seed=random.randint(1, 99999))
        flap = False

        while not sim.all_dead():
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    return
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        return
                    if (ev.key == pygame.K_F11) or (ev.key == pygame.K_f and (ev.mod & (pygame.KMOD_META | pygame.KMOD_CTRL))):
                        toggle_fullscreen()
                    if ev.key in (pygame.K_SPACE, pygame.K_UP, pygame.K_w):
                        flap = True
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    flap = True

            sim.step({0: 1 if flap else 0})
            flap = False
            _draw(screen, sim, 0, sim.birds[0].score, title="HUMAN PLAY - SPACE TO FLAP")
            clock.tick(TARGET_FPS)

        score = sim.birds[0].score
        if score > high_score:
            high_score = score

        # Game‑over overlay
        sw, sh = screen.get_size()
        ov = pygame.Surface((sw, sh), pygame.SRCALPHA)
        ov.fill((0, 0, 0, 150))
        screen.blit(ov, (0, 0))
        draw_text(screen, "GAME OVER", (sw // 2, sh // 2 - 50),
                  color=(255, 60, 60), scale=4, align="center", shadow=True)
        draw_text(screen, f"SCORE: {score}  BEST: {high_score}",
                  (sw // 2, sh // 2 + 20), color=COL_TEXT, scale=2,
                  align="center")
        draw_text(screen, "SPACE: RETRY  |  ESC: MENU  |  F11: FULLSCREEN",
                  (sw // 2, sh // 2 + 60), color=COL_TEXT_DIM, scale=1,
                  align="center")
        pygame.display.flip()

        # Wait for user to retry or escape
        waiting = True
        while waiting:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    return
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        return
                    if (ev.key == pygame.K_F11) or (ev.key == pygame.K_f and (ev.mod & (pygame.KMOD_META | pygame.KMOD_CTRL))):
                        toggle_fullscreen()
                    if ev.key in (pygame.K_SPACE, pygame.K_RETURN):
                        waiting = False
                if ev.type == pygame.MOUSEBUTTONDOWN:
                    waiting = False
            clock.tick(30)


# =====================================================================
#  CLI & ENTRY POINT
# =====================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Flappy Bird NeuroEvolution Trainer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--train", action="store_true",
                    help="Start fresh training immediately (skip menu)")
    p.add_argument("--load", type=int, default=None, metavar="GEN",
                    help="Load and showcase a specific generation checkpoint")
    p.add_argument("--ar", action="store_true",
                    help="Auto-restart showcase when trial ends (default: wait for Space/Enter)")
    p.add_argument("--human", action="store_true",
                    help="Launch human‑playable mode")
    p.add_argument("--fullscreen", action="store_true",
                    help="Launch directly in fullscreen mode")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    pygame.init()
    # pygame.RESIZABLE | pygame.SCALED enables:
    # 1. Native macOS green maximize/fullscreen window button
    # 2. Hardware scaling to any monitor resolution without aspect ratio distortion
    # 3. Automatic mouse coordinate translation to virtual (WINDOW_W, WINDOW_H)
    flags = pygame.RESIZABLE | pygame.SCALED
    if args.fullscreen:
        flags = pygame.FULLSCREEN | pygame.SCALED
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H), flags)
    pygame.display.set_caption("Flappy Bird — NeuroEvolution Trainer")

    # ── Direct CLI dispatch ──────────────────────────────────────────
    if args.train:
        run_training(screen)
        pygame.quit()
        return

    if args.load is not None:
        fp = os.path.join(CHECKPOINT_DIR, f"bird_gen_{args.load}.pkl")
        if os.path.isfile(fp):
            run_showcase(screen, fp, auto_restart=args.ar)
        else:
            print(f"Checkpoint not found: {fp}")
        pygame.quit()
        return

    if args.human:
        run_human(screen)
        pygame.quit()
        return

    # ── Interactive menu loop ────────────────────────────────────────
    while True:
        choice = main_menu(screen)
        if choice == "quit":
            break
        elif choice == "train":
            run_training(screen)
        elif choice == "load":
            fp = checkpoint_menu(screen)
            if fp:
                run_showcase(screen, fp, auto_restart=args.ar)
        elif choice == "human":
            run_human(screen)

    pygame.quit()


if __name__ == "__main__":
    main()
