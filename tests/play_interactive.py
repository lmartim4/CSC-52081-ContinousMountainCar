"""
Interactive play: control Mario with the keyboard while watching
the pixel render, composite grid, and individual RAM channels update
in real time.

Controls:
  Arrow Right / D   → right
  Arrow Left  / A   → left
  Arrow Up    / W   → jump (A button)
  Space             → right + jump  (right + A)
  X / Shift         → run (B button)
  R                 → reset episode
  Escape / Q        → quit

Run:
    conda run -n mario python tests/play_interactive.py
"""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pygame
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from src.wrappers.ram_wrappers import (
    RAMGridObservation, SkipFrame, grid_to_composite,
    VISIBLE_ROWS, VISIBLE_COLS,
    N_CHANNELS, CH_TILES, CH_ENEMIES, CH_MARIO, CH_POWERUP, CH_FIREBALL,
    EMPTY, SOLID, ENEMY, MARIO, POWERUP, FIREBALL,
)

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
PIXEL_SCALE = 2             # NES output (256x240) scaled by this factor
CELL = 28                   # pixels per grid cell for composite view
CELL_SM = 16                # pixels per grid cell for channel views
GAP = 10                    # gap between panels
FONT_SIZE = 14
INFO_H = 60                 # height of the info bar at the top

# Composite grid colors  {value: (R, G, B)}
COMP_COLORS = {
    EMPTY:    (135, 206, 235),  # skyblue
    SOLID:    (139, 69, 19),    # saddlebrown
    ENEMY:    (220, 20, 20),    # red
    MARIO:    (0, 200, 0),      # lime
    POWERUP:  (255, 215, 0),    # gold
    FIREBALL: (255, 100, 0),    # orange
}

# Per-channel color gradient (dark → bright)
CH_BASE_COLORS = {
    CH_TILES:    (180, 120, 40),   # brown tones
    CH_ENEMIES:  (220, 40, 40),    # red tones
    CH_MARIO:    (40, 200, 40),    # green tones
    CH_POWERUP:  (255, 215, 0),    # gold tones
    CH_FIREBALL: (255, 100, 0),    # orange tones
}
CH_NAMES = {
    CH_TILES:    "Tiles (raw)",
    CH_ENEMIES:  "Enemies",
    CH_MARIO:    "Mario",
    CH_POWERUP:  "Powerup",
    CH_FIREBALL: "Fireball",
}

SYMBOLS = {EMPTY: "", SOLID: "#", ENEMY: "E", MARIO: "M", POWERUP: "?", FIREBALL: "o"}

# SIMPLE_MOVEMENT index:
# 0=NOOP, 1=right, 2=right+A, 3=right+B, 4=right+A+B, 5=A, 6=left
ACTION_NAMES = ["NOOP", "right", "right+A", "right+B", "right+A+B", "A(jump)", "left"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def keys_to_action(keys):
    """Map pressed keys to a SIMPLE_MOVEMENT action index."""
    right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
    left = keys[pygame.K_LEFT] or keys[pygame.K_a]
    jump = keys[pygame.K_UP] or keys[pygame.K_w] or keys[pygame.K_SPACE]
    run = keys[pygame.K_x] or keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]

    if right and jump and run:
        return 4   # right + A + B
    if right and run:
        return 3   # right + B
    if right and jump:
        return 2   # right + A
    if right:
        return 1   # right
    if jump:
        return 5   # A (jump)
    if left:
        return 6   # left
    return 0       # NOOP


def channel_color(base_rgb, value, vmax):
    """Interpolate from dark background to base_rgb based on value/vmax."""
    if vmax == 0 or value == 0:
        return (20, 20, 30)  # dark background
    t = min(value / vmax, 1.0)
    return tuple(int(20 + (c - 20) * t) for c in base_rgb)


def draw_composite_grid(surface, x0, y0, composite, font):
    """Draw the composite grid onto a surface."""
    for r in range(VISIBLE_ROWS):
        for c in range(VISIBLE_COLS):
            val = int(composite[r, c])
            color = COMP_COLORS.get(val, (50, 50, 50))
            rect = pygame.Rect(x0 + c * CELL, y0 + r * CELL, CELL, CELL)
            pygame.draw.rect(surface, color, rect)
            pygame.draw.rect(surface, (60, 60, 60), rect, 1)

            sym = SYMBOLS.get(val, "")
            if sym:
                txt_color = (255, 255, 255) if val in (SOLID, ENEMY) else (0, 0, 0)
                txt = font.render(sym, True, txt_color)
                tx = rect.centerx - txt.get_width() // 2
                ty = rect.centery - txt.get_height() // 2
                surface.blit(txt, (tx, ty))


def draw_channel(surface, x0, y0, ch_data, ch_idx, font_sm):
    """Draw one channel heatmap."""
    vmax = ch_data.max() if ch_data.max() > 0 else 1
    base = CH_BASE_COLORS[ch_idx]
    for r in range(VISIBLE_ROWS):
        for c in range(VISIBLE_COLS):
            val = float(ch_data[r, c])
            color = channel_color(base, val, vmax)
            rect = pygame.Rect(x0 + c * CELL_SM, y0 + r * CELL_SM, CELL_SM, CELL_SM)
            pygame.draw.rect(surface, color, rect)
            pygame.draw.rect(surface, (40, 40, 50), rect, 1)

            if val != 0:
                txt = font_sm.render(str(int(val)), True, (255, 255, 255))
                tx = rect.centerx - txt.get_width() // 2
                ty = rect.centery - txt.get_height() // 2
                surface.blit(txt, (tx, ty))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # -- Create environments --
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipFrame(env, skip=4)
    env = RAMGridObservation(env)

    obs = env.reset()

    # -- Window layout calculations --
    pixel_w = 256 * PIXEL_SCALE
    pixel_h = 240 * PIXEL_SCALE

    comp_w = VISIBLE_COLS * CELL
    comp_h = VISIBLE_ROWS * CELL

    ch_w = VISIBLE_COLS * CELL_SM
    ch_h = VISIBLE_ROWS * CELL_SM

    # Channel grids: 3 columns × 2 rows (5 channels + 1 empty)
    ch_cols = 3
    ch_rows = 2
    channels_block_w = ch_w * ch_cols + GAP * (ch_cols - 1)
    channels_block_h = (ch_h + FONT_SIZE + 6) * ch_rows

    right_w = max(comp_w, channels_block_w)

    win_w = pixel_w + GAP + right_w + GAP
    win_h = INFO_H + max(pixel_h, comp_h + GAP + channels_block_h + FONT_SIZE + 4) + GAP

    # -- Init pygame --
    pygame.init()
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("Super Mario Bros — Interactive RAM Grid Viewer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", FONT_SIZE, bold=True)
    font_sm = pygame.font.SysFont("monospace", 10)
    font_title = pygame.font.SysFont("monospace", 12, bold=True)

    episode = 1
    step = 0
    total_reward = 0.0
    running = True

    print("Controls: Arrow keys / WASD to move, Space = jump+right, Shift = run")
    print("          R = reset, Esc/Q = quit")

    while running:
        # -- Events --
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                if event.key == pygame.K_r:
                    obs = env.reset()
                    episode += 1
                    step = 0
                    total_reward = 0.0

        if not running:
            break

        # -- Action from keyboard --
        keys = pygame.key.get_pressed()
        action = keys_to_action(keys)

        # -- Step environment --
        obs, reward, done, info = env.step(action)
        step += 1
        total_reward += reward

        if done:
            obs = env.reset()
            episode += 1
            step = 0
            total_reward = 0.0

        # -- Get pixel render --
        pixel_frame = env.unwrapped.screen  # (240, 256, 3)

        # -- Build composite --
        composite = grid_to_composite(obs)

        # -- Draw everything --
        screen.fill((15, 15, 25))

        # Info bar
        info_text = (
            f"Ep:{episode}  Step:{step}  Action:{ACTION_NAMES[action]:>12s}  "
            f"Reward:{total_reward:>7.0f}  "
            f"x:{info.get('x_pos', 0):>4d}  y:{info.get('y_pos', 0):>3d}  "
            f"status:{info.get('status', '?')}"
        )
        screen.blit(font.render(info_text, True, (200, 200, 200)), (GAP, 8))

        mario_state = int(obs[:, :, CH_MARIO].max())
        state_txt = {0: "???", 1: "small", 2: "big", 3: "fire"}.get(mario_state, "?")
        screen.blit(
            font.render(f"Mario: {state_txt}   FPS: {clock.get_fps():.0f}", True, (150, 200, 150)),
            (GAP, 28),
        )

        y_start = INFO_H

        # -- Left panel: pixel render --
        pixel_surf = pygame.surfarray.make_surface(
            np.transpose(pixel_frame, (1, 0, 2))
        )
        pixel_surf = pygame.transform.scale(pixel_surf, (pixel_w, pixel_h))
        screen.blit(pixel_surf, (0, y_start))

        # -- Right panel: composite grid --
        rx = pixel_w + GAP
        screen.blit(font_title.render("Composite Grid", True, (200, 200, 200)), (rx, y_start - 2))
        draw_composite_grid(screen, rx, y_start + FONT_SIZE + 2, composite, font_sm)

        # -- Right panel: channel grids (3×2) --
        cy_start = y_start + FONT_SIZE + 2 + comp_h + GAP

        for idx in range(N_CHANNELS):
            col_off = (idx % ch_cols) * (ch_w + GAP)
            row_off = (idx // ch_cols) * (ch_h + FONT_SIZE + 6)
            cx = rx + col_off
            cy = cy_start + row_off

            # Channel title
            vmax = obs[:, :, idx].max()
            title = f"{CH_NAMES[idx]} (max={vmax:.0f})"
            screen.blit(font_sm.render(title, True, (180, 180, 180)), (cx, cy))

            draw_channel(screen, cx, cy + FONT_SIZE, obs[:, :, idx], idx, font_sm)

        pygame.display.flip()
        clock.tick(15)  # ~15 FPS to match SkipFrame=4

    env.close()
    pygame.quit()
    print("Done.")


if __name__ == "__main__":
    main()
