#!/usr/bin/env python3
"""
RAM Wrapper Debugger — play Mario and inspect the agent's symbolic state.

Layout:
  Left  │ Right
  ──────┼──────────────────────────
  Game  │ 13×16 RAM grid (colored)
  pixels│ with cell-value overlay
  ──────┴──────────────────────────
        Info bar

Controls:
  Arrow Right / Left   Move
  Z  (hold)            Run  (B button)
  X  / Space / Up      Jump (A button)
  R                    Reset episode
  V                    Toggle value overlay on grid
  P                    Pause / unpause
  Q  / Esc             Quit

Run from the project root:
    python debug_ram.py [env_id]
    python debug_ram.py SuperMarioBros-1-2-v3
"""

import os
import sys
from collections import Counter

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pygame
from PIL import Image, ImageDraw, ImageFont

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from src.wrappers.ram_wrappers import (
    CustomRewardRAM, SkipFrame, RAMGridObservation,
    EMPTY, SOLID, ENEMY, MARIO, POWERUP, FIRE,
)
from src.utils.smb_utils import smb_grid

# ── Configuration ─────────────────────────────────────────────────────────────

CELL_PX    = 30         # pixels per grid cell
GAME_SCALE = 3          # scale factor for the 256×240 game render
INFO_H     = 70         # height of the bottom info bar
GAP        = 8          # gap between game and grid panels
FPS        = 60

GRID_ROWS = 13
GRID_COLS = 16

# Color palette for grid cells  (R, G, B)
PALETTE = {
    EMPTY:   ( 30,  40,  80),   # dark blue  – sky
    SOLID:   (130,  80,  30),   # brown      – tiles / ground
    ENEMY:   (210,  40,  40),   # red        – enemy
    MARIO:   (255, 215,   0),   # gold       – Mario
    POWERUP: ( 30, 180,  80),   # green      – mushroom / flower / star
    FIRE:    (255, 120,   0),   # orange     – fire snake bead
}
BORDER_COLOR = (60, 60, 60)

# SIMPLE_MOVEMENT action labels
ACTION_NAMES = ["NOOP", "->", "->+A", "->+B", "->+A+B", "A", "<-"]

# ── PIL-based font (avoids pygame.font / sysfont circular import on Py 3.14) ──

class PilFont:
    """Render text to pygame Surfaces using Pillow — no pygame.font needed."""

    def __init__(self, size: int):
        try:
            # Pillow >= 10.1 supports size argument
            self._font = ImageFont.load_default(size=size)
        except TypeError:
            self._font = ImageFont.load_default()

    def render(self, text: str, color) -> tuple:
        """Return (pygame.Surface, pygame.Rect) with the rendered text."""
        # Measure
        dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        bbox = dummy.textbbox((0, 0), text, font=self._font)
        w = max(1, bbox[2] - bbox[0])
        h = max(1, bbox[3] - bbox[1])

        # Draw with 1px padding
        img = Image.new("RGBA", (w + 2, h + 2), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text((1 - bbox[0], 1 - bbox[1]), text,
                  font=self._font, fill=(*color, 255))

        surf = pygame.image.fromstring(img.tobytes(), img.size, "RGBA").convert_alpha()
        return surf, surf.get_rect()

# ── Key → action mapping ──────────────────────────────────────────────────────

def keys_to_action(keys) -> int:
    """Map currently-held keys to a SIMPLE_MOVEMENT index."""
    right = keys[pygame.K_RIGHT]
    left  = keys[pygame.K_LEFT]
    run   = keys[pygame.K_z]   or keys[pygame.K_LSHIFT]
    jump  = keys[pygame.K_x]   or keys[pygame.K_SPACE] or keys[pygame.K_UP]

    if right and run and jump:
        return 4   # right + A + B
    if right and jump:
        return 2   # right + A
    if right and run:
        return 3   # right + B
    if right:
        return 1   # right
    if left:
        return 6   # left
    if jump:
        return 5   # A  (jump in place)
    return 0       # NOOP

# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_grid(surface, grid, x0, y0, font_cell, show_values: bool):
    """Render the 13x16 grid at (x0, y0) with an optional value overlay."""
    symbols = {SOLID: "#", ENEMY: "E", MARIO: "M", POWERUP: "?", FIRE: "F"}

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            val   = int(grid[r, c])
            color = PALETTE.get(val, (100, 100, 100))
            rect  = pygame.Rect(x0 + c * CELL_PX, y0 + r * CELL_PX,
                                CELL_PX - 1, CELL_PX - 1)
            pygame.draw.rect(surface, color, rect)

            if show_values and val != EMPTY:
                sym = symbols.get(val, "?")
                txt_color = (255, 255, 255) if val in (SOLID, ENEMY) else (0, 0, 0)
                txt, txt_rect = font_cell.render(sym, txt_color)
                surface.blit(txt, (rect.x + (CELL_PX - txt_rect.width) // 2,
                                   rect.y + (CELL_PX - txt_rect.height) // 2))


def draw_ram_dump(surface, x0, y0, font, ram):
    """Render a table of the 5 enemy slots directly from RAM.

    Columns: slot | flag | type (hex) | x_level | y_screen | grid(col,row)
    Highlights active slots in bright yellow; inactive in grey.
    Toggle with [D].
    """
    header = "  Slot  Flag  Type   X-lvl   Y-scr   Grid(c,r)"
    hdr_surf, _ = font.render(header, (120, 180, 255))
    surface.blit(hdr_surf, (x0, y0))

    mario_level_x = int(ram[0x6d]) * 256 + int(ram[0x86])
    mario_x_screen = int(ram[0x3ad])
    x_start = mario_level_x - mario_x_screen   # left edge of screen in level px

    row_h = 18
    for i in range(5):
        flag  = int(ram[0x0F + i])
        etype = int(ram[0x16 + i])
        ex_hi = int(ram[0x6e + i])
        ex_lo = int(ram[0x87 + i])
        ey    = int(ram[0xcf + i])
        ex_level = ex_hi * 256 + ex_lo
        ex_screen = ex_level - x_start
        col = (ex_screen + 8) // 16
        row = (ey + 8 - 32) // 16

        color = (255, 230, 60) if flag == 1 else (90, 90, 100)
        line = (f"  [{i}]    {flag}     0x{etype:02X}   "
                f"{ex_level:<7}  {ey:<7} ({col},{row})")
        surf, _ = font.render(line, color)
        surface.blit(surf, (x0, y0 + row_h * (i + 1)))


def draw_oam_dump(surface, x0, y0, font, ram):
    """Render active OAM sprites (0x0200–0x02FF) for tile identification.

    NES OAM layout (4 bytes per sprite, 64 sprites):
      byte 0: Y position - 1  (0xEF = off-screen)
      byte 1: tile index
      byte 2: attributes
      byte 3: X position

    Shows only sprites that are on-screen (Y < 0xEF).
    Toggle with [O].
    """
    header = "  #   Tile    X-px   Y-px   Grid(c,r)"
    hdr_surf, _ = font.render(header, (120, 255, 180))
    surface.blit(hdr_surf, (x0, y0))

    row_h = 16
    drawn = 0
    for i in range(64):
        base = 0x0200 + i * 4
        spr_y    = int(ram[base + 0])
        tile     = int(ram[base + 1])
        spr_x    = int(ram[base + 3])

        if spr_y >= 0xEF:   # off-screen / inactive
            continue

        spr_y_actual = spr_y + 1   # NES stores Y-1
        col = (spr_x + 8) // 16
        row = (spr_y_actual + 8 - 32) // 16

        line = (f"  {i:<3}  0x{tile:02X}   {spr_x:<6} {spr_y_actual:<6} ({col},{row})")
        surf, _ = font.render(line, (200, 255, 200))
        surface.blit(surf, (x0, y0 + row_h * (drawn + 1)))
        drawn += 1
        if drawn >= 20:   # cap to avoid overflow
            more_surf, _ = font.render(f"  ... {64 - i - 1} more", (150, 150, 150))
            surface.blit(more_surf, (x0, y0 + row_h * (drawn + 1)))
            break


def draw_legend(surface, x0, y0, font):
    items = [
        (ENEMY,   "Enemy"),
        (EMPTY,   "Empty"),
        (SOLID,   "Solid"),
        (MARIO,   "Mario"),
        (POWERUP, "Power"),
        (FIRE,    "Fire"),
    ]
    for i, (val, label) in enumerate(items):
        cx = x0 + i * 82
        pygame.draw.rect(surface, PALETTE[val],       (cx, y0 + 5, 14, 14))
        pygame.draw.rect(surface, (120, 120, 120),    (cx, y0 + 5, 14, 14), 1)
        txt_surf, _ = font.render(label, (200, 200, 200))
        surface.blit(txt_surf, (cx + 18, y0 + 5))


def draw_info(surface, font_big, font_sm, x0, y0, w,
              action, info, ep_reward, paused, show_values):
    pygame.draw.rect(surface, (15, 15, 20), (0, y0, w, INFO_H))
    pygame.draw.line(surface, (60, 60, 80), (0, y0), (w, y0), 1)

    x_pos  = info.get("x_pos", 0)
    score  = info.get("score",  0)
    status = info.get("status", "?")
    time_  = info.get("time",   0)
    aname  = ACTION_NAMES[action]

    line1 = (f"  Action: {aname:<8}  x_pos: {x_pos:<5}  "
             f"score: {score:<6}  time: {time_:<4}  status: {status}")
    phase_deg = np.degrees(smb_grid._FIREBAR_PHASE)
    line2 = (f"  Ep reward: {ep_reward:+.2f}    "
             f"{'[PAUSED]  ' if paused else ''}"
             f"[R]=reset  [V]=values({'on' if show_values else 'off'})  "
             f"[D]=slots  [O]=OAM  [[/]]=fire phase({phase_deg:.0f}°)  [P]=pause  [Q]=quit")

    surf1, _ = font_big.render(line1, (200, 220, 255))
    surface.blit(surf1, (x0, y0 + 8))
    surf2, _ = font_sm.render(line2, (150, 170, 200))
    surface.blit(surf2, (x0, y0 + 36))

# ── Environment builder ───────────────────────────────────────────────────────

def make_env(env_id: str):
    """Build the debug environment (skip=1 for responsive keyboard play)."""
    env = gym_super_mario_bros.make(env_id)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = CustomRewardRAM(env)
    env = SkipFrame(env, skip=1)    # skip=1 -> every frame goes to pygame
    env = RAMGridObservation(env)   # obs becomes (13, 16) grid
    return env

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    env_id = sys.argv[1] if len(sys.argv) > 1 else "SuperMarioBros-1-1-v3"

    print(f"[debug_ram] Building env: {env_id}")
    env = make_env(env_id)
    obs, _ = env.reset()

    # Compute window dimensions
    game_w   = 256 * GAME_SCALE
    game_h   = 240 * GAME_SCALE
    grid_w   = GRID_COLS * CELL_PX
    grid_h   = GRID_ROWS * CELL_PX
    legend_h = 28
    panel_h  = max(game_h, grid_h + legend_h + 24)   # +24 for title
    win_w    = game_w + GAP + grid_w
    win_h    = panel_h + INFO_H

    pygame.init()
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(f"RAM Debugger - {env_id}")
    clock = pygame.time.Clock()

    # PIL-based fonts (size in pixels)
    font_big  = PilFont(16)
    font_sm   = PilFont(14)
    font_cell = PilFont(14)

    ep_reward   = 0.0
    last_info   = {}
    last_action = 0
    show_values  = True
    show_dump    = False
    show_oam     = False
    paused       = False
    running      = True
    tile_recording = False          # T key: accumulate OAM tile IDs
    tile_counter   = Counter()      # tile_id -> frames seen
    tile_frames    = 0              # frames recorded
    _PHASE_STEP  = np.pi / 8       # 22.5° per keypress

    print("[debug_ram] Window open. Arrow keys + Z (run) + X/Space (jump).")
    print("            R=reset  V=toggle values  D=enemy slots  O=OAM sprites")
    print("            T=start/stop tile recording (prints results)  P=pause  Q=quit")

    while running:
        # ── Events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r:
                    obs, _ = env.reset()
                    ep_reward = 0.0
                    last_info = {}
                elif event.key == pygame.K_v:
                    show_values = not show_values
                elif event.key == pygame.K_d:
                    show_dump = not show_dump
                    if show_dump:
                        show_oam = False   # mutually exclusive
                elif event.key == pygame.K_o:
                    show_oam = not show_oam
                    if show_oam:
                        show_dump = False   # mutually exclusive
                elif event.key == pygame.K_t:
                    if not tile_recording:
                        # Start a fresh recording
                        tile_counter.clear()
                        tile_frames = 0
                        tile_recording = True
                        print("[tile-rec] Started — navigate near the fire snake, then press T again.")
                    else:
                        tile_recording = False
                        print(f"[tile-rec] Stopped after {tile_frames} frames.")
                        print("  Tile  | Frames seen")
                        print("  ------+------------")
                        for tile, count in sorted(tile_counter.items(),
                                                  key=lambda x: -x[1])[:20]:
                            marker = " *** current fire tile" if tile == 0x65 else ""
                            print(f"  0x{tile:02X}  |  {count}{marker}")
                elif event.key == pygame.K_LEFTBRACKET:
                    smb_grid._FIREBAR_PHASE -= _PHASE_STEP
                    print(f"[phase] {smb_grid._FIREBAR_PHASE:.3f} rad  "
                          f"({np.degrees(smb_grid._FIREBAR_PHASE):.1f}°)")
                elif event.key == pygame.K_RIGHTBRACKET:
                    smb_grid._FIREBAR_PHASE += _PHASE_STEP
                    print(f"[phase] {smb_grid._FIREBAR_PHASE:.3f} rad  "
                          f"({np.degrees(smb_grid._FIREBAR_PHASE):.1f}°)")
                elif event.key == pygame.K_p:
                    paused = not paused

        # ── Step environment ───────────────────────────────────────────────────
        if not paused:
            keys        = pygame.key.get_pressed()
            last_action = keys_to_action(keys)
            obs, reward, terminated, truncated, info = env.step(last_action)
            ep_reward += reward
            last_info  = info
            if terminated or truncated:
                obs, _ = env.reset()
                ep_reward = 0.0

        # ── Tile frequency recording (T key) ──────────────────────────────────
        if tile_recording and not paused:
            ram = env.unwrapped.ram
            tile_frames += 1
            for i in range(64):
                base = 0x0200 + i * 4
                if int(ram[base]) < 0xEF:          # sprite is on-screen
                    tile_counter[int(ram[base + 1])] += 1

        # ── Render ─────────────────────────────────────────────────────────────
        screen.fill((20, 20, 30))

        # Left panel: game pixels from NES-py screen buffer (240, 256, 3) uint8
        raw_screen = env.unwrapped.screen
        surf = pygame.surfarray.make_surface(raw_screen.transpose(1, 0, 2))
        surf = pygame.transform.scale(surf, (game_w, game_h))
        screen.blit(surf, (0, 0))

        # Separator
        pygame.draw.line(screen, (60, 60, 90),
                         (game_w + GAP // 2, 0),
                         (game_w + GAP // 2, panel_h), 1)

        # Right panel: RAM grid (vertically centered)
        gx = game_w + GAP
        gy = (panel_h - grid_h - legend_h - 24) // 2 + 24

        # Grid title
        title_surf, _ = font_sm.render("Agent's RAM Grid  (13 x 16)", (160, 190, 240))
        screen.blit(title_surf, (gx, gy - 22))

        draw_grid(screen, obs, gx, gy, font_cell, show_values)

        # Grid border
        pygame.draw.rect(screen, BORDER_COLOR,
                         (gx - 1, gy - 1, grid_w + 1, grid_h + 1), 1)

        # Legend
        draw_legend(screen, gx, gy + grid_h + 2, font_sm)

        # Enemy RAM dump (toggled with D) — mutually exclusive with OAM dump
        if show_dump:
            dump_y = gy + grid_h + legend_h + 10
            draw_ram_dump(screen, gx, dump_y, font_sm, env.unwrapped.ram)
        elif show_oam:
            dump_y = gy + grid_h + legend_h + 10
            draw_oam_dump(screen, gx, dump_y, font_sm, env.unwrapped.ram)

        # Info bar
        draw_info(screen, font_big, font_sm, 4, panel_h, win_w,
                  last_action, last_info, ep_reward, paused, show_values)

        pygame.display.flip()
        clock.tick(FPS)

    env.close()
    pygame.quit()
    print("[debug_ram] Done.")


if __name__ == "__main__":
    main()
