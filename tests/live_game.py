"""
Live Super Mario Bros — Feature Viewer

Left panel : raw NES frame scaled 2×
Right panel: 13×16 RAM tile grid, each cell = one NES tile (aligned to game)
             Scalar features shown as text in the top-right corner

Controls: ← →  move  |  Z jump  |  X run  |  R reset  |  D debug  |  ESC/Q quit
  Debug mode: shows raw hex tile IDs instead of classified symbols
"""

import sys, os, argparse
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pygame
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from src.wrappers.ram_wrappers import (
    RAMGridObservation, AugmentedRAMObservation, SkipFrame,
    EMPTY, SOLID, ENEMY, MARIO, MUSHROOM, FLOWER, STAR, BRICK, QUESTION, ONEUP,
    VISIBLE_ROWS, VISIBLE_COLS, SCALAR_META,
)
from src.utils.smb_utils import smb_grid as smb_grid_fn

# ---------------------------------------------------------------------------
# Layout — grid cells aligned to NES tiles at 2× scale
# ---------------------------------------------------------------------------
SCALE   = 2
GAME_W  = 256 * SCALE          # 512
GAME_H  = 240 * SCALE          # 480
CELL    = 16 * SCALE           # 32 px  — one NES tile at 2×
GRID_W  = VISIBLE_COLS * CELL  # 512
GRID_TOP = 2 * CELL            # 64 px  — skip the 2 top rows absent from RAM
WIN_W   = GAME_W + GRID_W      # 1024
WIN_H   = GAME_H               # 480
FPS     = 60

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
C_BG    = (20,  20,  20)
C_DIV   = (60,  60,  60)
C_WHITE = (255, 255, 255)
C_GRAY  = (140, 140, 140)
C_BLACK = (0,   0,   0)

CELL_COLOR = {
    EMPTY:    (135, 206, 235),   # sky blue
    SOLID:    (101,  67,  33),   # brown       — ground / pipes / platforms
    BRICK:    (210, 105,  30),   # orange      — breakable brick
    QUESTION: (255, 200,   0),   # yellow      — ? block
    ENEMY:    (220,  50,  50),   # red
    MARIO:    ( 50, 200,  80),   # green
    MUSHROOM: (255, 120,   0),   # orange-red  — super mushroom / 1-up
    FLOWER:   (255,  60, 180),   # pink        — fire flower
    STAR:     (255, 240,  50),   # bright gold — super star
}
CELL_SYMBOL = {
    SOLID: "#", BRICK: "B", QUESTION: "?",
    ENEMY: "E", MARIO: "M",
    MUSHROOM: "Mu", FLOWER: "Fl", STAR: "St",
}
CELL_TEXT_COLOR = {
    SOLID: C_WHITE, BRICK: C_WHITE, QUESTION: C_BLACK,
    ENEMY: C_WHITE, MARIO: C_BLACK,
    MUSHROOM: C_WHITE, FLOWER: C_WHITE, STAR: C_BLACK,
}

# ---------------------------------------------------------------------------
# Font
# ---------------------------------------------------------------------------
def _make_font(size, bold=False):
    return pygame.font.SysFont("monospace", size, bold=bold)

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
def keys_to_action(keys):
    r = keys[pygame.K_RIGHT]; l = keys[pygame.K_LEFT]
    j = keys[pygame.K_z];     b = keys[pygame.K_x]
    if r and b and j: return 4
    if r and j:       return 2
    if r and b:       return 3
    if r:             return 1
    if j:             return 5
    if l:             return 6
    return 0

# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
_game_surf  = None   # reused every frame to avoid allocation
_cell_surfs = {}     # pre-rendered tile label surfaces
_val_cache  = {}     # cached scalar text surfaces
_hex_cache  = {}     # cached hex label surfaces for debug mode

def _build_cell_surfs(font):
    for val, sym in CELL_SYMBOL.items():
        _cell_surfs[val] = font.render(sym, True, CELL_TEXT_COLOR[val])


def blit_game_frame(screen, rgb):
    global _game_surf
    raw = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
    if _game_surf is None:
        _game_surf = pygame.Surface((GAME_W, GAME_H))
    pygame.transform.scale(raw, (GAME_W, GAME_H), _game_surf)
    screen.blit(_game_surf, (0, 0))


def draw_grid(screen, grid):
    for r in range(VISIBLE_ROWS):
        for c in range(VISIBLE_COLS):
            val = int(grid[r, c])
            rect = pygame.Rect(
                GAME_W + c * CELL,
                GRID_TOP + r * CELL,
                CELL - 1, CELL - 1,
            )
            pygame.draw.rect(screen, CELL_COLOR.get(val, C_GRAY), rect)
            surf = _cell_surfs.get(val)
            if surf:
                screen.blit(surf, surf.get_rect(center=rect.center))


def draw_grid_debug(screen, raw_grid, font_dbg):
    """Debug mode: show raw hex tile IDs on each cell."""
    for r in range(VISIBLE_ROWS):
        for c in range(VISIBLE_COLS):
            v = int(raw_grid[r, c])
            rect = pygame.Rect(
                GAME_W + c * CELL,
                GRID_TOP + r * CELL,
                CELL - 1, CELL - 1,
            )
            if v == 0:
                color = (135, 206, 235)   # sky
            elif v == 2:                  # MARIO
                color = (50, 200, 80)
            elif v == -1:                 # ENEMY
                color = (220, 50, 50)
            elif v == MUSHROOM:
                color = (255, 120, 0)
            elif v == FLOWER:
                color = (255, 60, 180)
            elif v == STAR:
                color = (255, 240, 50)
            else:
                color = (60, 60, 80)      # dark — unknown tile
            pygame.draw.rect(screen, color, rect)
            if v != 0:
                _pu = {MUSHROOM: "Mu", FLOWER: "Fl", STAR: "St"}
                label = _pu.get(v, hex(v) if v > 0 else str(v))
                if label not in _hex_cache:
                    _hex_cache[label] = font_dbg.render(label, True, C_WHITE)
                surf = _hex_cache[label]
                screen.blit(surf, surf.get_rect(center=rect.center))


def draw_scalars(screen, scalars, font):
    x = GAME_W + GRID_W - 140
    y = 6
    for idx, label, _, vmin, vmax, _ in SCALAR_META:
        val = float(scalars[idx])
        text = f"{label}: {val:+.2f}"
        if text not in _val_cache:
            _val_cache[text] = font.render(text, True, C_WHITE)
        screen.blit(_val_cache[text], (x, y))
        y += 15

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipFrame(env, skip=1)
    env = RAMGridObservation(env)
    env = AugmentedRAMObservation(env)
    obs, _ = env.reset()

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Mario — Feature Viewer  [D = toggle debug hex IDs]")
    clock = pygame.time.Clock()

    font     = _make_font(12)
    font_dbg = _make_font(9)
    _build_cell_surfs(font)

    debug_mode = False
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    obs, _ = env.reset()
                elif event.key == pygame.K_d:
                    debug_mode = not debug_mode
                    title = "Mario — Feature Viewer  [DEBUG: raw hex IDs]" if debug_mode \
                            else "Mario — Feature Viewer  [D = toggle debug hex IDs]"
                    pygame.display.set_caption(title)

        action = keys_to_action(pygame.key.get_pressed())
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

        screen.fill(C_BG)
        blit_game_frame(screen, env.unwrapped.screen)
        pygame.draw.line(screen, C_DIV, (GAME_W, 0), (GAME_W, WIN_H), 1)

        if debug_mode:
            raw = smb_grid_fn(env).rendered_screen
            RAMGridObservation._fill_powerup(raw, env.unwrapped.ram)
            draw_grid_debug(screen, raw, font_dbg)
        else:
            draw_grid(screen, obs["grid"])
            draw_scalars(screen, obs["scalars"], font)

        pygame.display.flip()
        clock.tick(FPS)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
