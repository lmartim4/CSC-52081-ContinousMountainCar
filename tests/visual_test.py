"""
Test 3: Visual comparison - pixel frame vs symbolic grid side by side.

Run from the project root:
    python tests/visual_test.py
"""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from src.wrappers.ram_wrappers import (
    RAMGridObservation, SkipFrame,
    EMPTY, SOLID, ENEMY, MARIO, POWERUP,
)


def plot_comparison(pixel_obs, grid_obs, info, step, ax_pixel, ax_grid):
    ax_pixel.imshow(pixel_obs)
    ax_pixel.set_title(f"Pixels (240x256x3)\nstep={step}, x={info.get('x_pos', '?')}")
    ax_pixel.axis("off")

    cmap = mcolors.ListedColormap(["red", "skyblue", "saddlebrown", "lime", "gold"])
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    ax_grid.imshow(grid_obs, cmap=cmap, norm=norm, interpolation="nearest")
    ax_grid.set_title(f"RAM Grid (13x16)\n.=empty #=solid E=enemy M=mario ?=powerup")
    ax_grid.set_xticks(range(16))
    ax_grid.set_yticks(range(13))
    ax_grid.grid(True, color="gray", linewidth=0.5, alpha=0.3)

    symbols = {EMPTY: "", SOLID: "#", ENEMY: "E", MARIO: "M", POWERUP: "?"}
    for r in range(13):
        for c in range(16):
            val = int(grid_obs[r, c])
            sym = symbols.get(val, "")
            if sym:
                color = "white" if val in (SOLID, ENEMY) else "black"
                ax_grid.text(c, r, sym, ha="center", va="center",
                             fontsize=6, fontweight="bold", color=color)


def main():
    output_path = os.path.join(PROJECT_ROOT, "results", "comparison.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    n_snapshots = 4
    steps_between = 50

    print("Creating environments...")
    env_raw = gym_super_mario_bros.make("SuperMarioBros-1-1-v3")
    env_raw = JoypadSpace(env_raw, SIMPLE_MOVEMENT)
    env_raw = SkipFrame(env_raw, skip=4)

    env_sym = gym_super_mario_bros.make("SuperMarioBros-1-1-v3")
    env_sym = JoypadSpace(env_sym, SIMPLE_MOVEMENT)
    env_sym = SkipFrame(env_sym, skip=4)
    env_sym = RAMGridObservation(env_sym)

    env_raw.reset()
    env_sym.reset()

    fig, axes = plt.subplots(n_snapshots, 2, figsize=(14, 4 * n_snapshots))

    actions = [1, 1, 2, 1, 1, 1, 2, 1]
    snapshot = 0
    step = 0

    print(f"Running for {n_snapshots} snapshots...")

    while snapshot < n_snapshots:
        action = actions[step % len(actions)]
        pixel_obs, _, d1, d1t, info = env_raw.step(action)
        grid_obs, _, d2, d2t, _ = env_sym.step(action)
        step += 1

        if step % steps_between == 0 or step == 1:
            print(f"  Snapshot {snapshot + 1} at step {step}: x_pos={info.get('x_pos')}")
            plot_comparison(pixel_obs, grid_obs, info, step,
                            axes[snapshot, 0], axes[snapshot, 1])
            snapshot += 1

        if d1 or d1t or d2 or d2t:
            env_raw.reset()
            env_sym.reset()

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"\n[OK] Comparison saved to: {output_path}")

    env_raw.close()
    env_sym.close()


if __name__ == "__main__":
    main()
