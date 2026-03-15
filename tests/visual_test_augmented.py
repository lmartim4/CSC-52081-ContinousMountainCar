"""
Visual test for AugmentedRAMObservation — verifies all 6 scalar features track correctly.

Outputs:
  results/augmented_snapshots.png   — 4 game moments: pixel | grid | scalar bars
  results/augmented_timeseries.png  — each scalar plotted over 600 steps

Run from project root:
    python tests/visual_test_augmented.py
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
import matplotlib.patches as mpatches

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from src.wrappers.ram_wrappers import (
    RAMGridObservation, AugmentedRAMObservation, SkipFrame,
    EMPTY, SOLID, ENEMY, MARIO, POWERUP,
    VISIBLE_ROWS, VISIBLE_COLS,
    SCALAR_META,
)

FLOAT_STATE_LABELS = {0.00: "ground", 0.33: "jump", 0.67: "ledge", 1.00: "pole"}

# ---------------------------------------------------------------------------
# Grid rendering helpers (same convention as visual_test.py)
# ---------------------------------------------------------------------------

_GRID_CMAP  = mcolors.ListedColormap(["red", "skyblue", "saddlebrown", "lime", "gold"])
_GRID_BOUNDS = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
_GRID_NORM   = mcolors.BoundaryNorm(_GRID_BOUNDS, _GRID_CMAP.N)
_SYMBOLS     = {EMPTY: "", SOLID: "#", ENEMY: "E", MARIO: "M", POWERUP: "?"}


def _draw_grid(ax, grid):
    ax.imshow(grid, cmap=_GRID_CMAP, norm=_GRID_NORM, interpolation="nearest")
    ax.set_xticks(range(VISIBLE_COLS))
    ax.set_yticks(range(VISIBLE_ROWS))
    ax.tick_params(labelsize=5)
    ax.grid(True, color="gray", linewidth=0.4, alpha=0.3)
    for r in range(VISIBLE_ROWS):
        for c in range(VISIBLE_COLS):
            val = int(grid[r, c])
            sym = _SYMBOLS.get(val, "")
            if sym:
                color = "white" if val in (SOLID, ENEMY) else "black"
                ax.text(c, r, sym, ha="center", va="center",
                        fontsize=5, fontweight="bold", color=color)


def _draw_scalar_bars(ax, scalars):
    """Horizontal bar chart for one observation's scalar vector."""
    n = len(SCALAR_META)
    y_pos = np.arange(n)
    colors = []
    for i, (idx, label, unit, vmin, vmax, cmap_name) in enumerate(SCALAR_META):
        val = float(scalars[idx])
        # map value to [0,1] for colormap
        t = (val - vmin) / (vmax - vmin)
        c = plt.get_cmap(cmap_name)(np.clip(t, 0, 1))
        colors.append(c)

    values = [float(scalars[i]) for i, *_ in SCALAR_META]
    vmins  = [m[3] for m in SCALAR_META]
    vmaxs  = [m[4] for m in SCALAR_META]

    ax.barh(y_pos, values, color=colors, edgecolor="black", linewidth=0.5,
            left=0 if vmins[0] >= 0 else None)

    # reference line at 0 for signed scalars
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    ax.set_xlim(-1.05, 1.05)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([m[1] for m in SCALAR_META], fontsize=7)
    ax.tick_params(axis="x", labelsize=6)
    ax.set_xlabel("normalised value", fontsize=6)

    # annotate bar with value
    for i, v in enumerate(values):
        ax.text(v + 0.03 * np.sign(v + 1e-9), i, f"{v:.2f}",
                va="center", fontsize=6,
                color="black")

    # float_state human label
    fs = float(scalars[2])
    fs_label = FLOAT_STATE_LABELS.get(round(fs, 2), f"{fs:.2f}")
    ax.text(1.07, 2, fs_label, va="center", fontsize=6, color="gray")


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_data(n_steps=600, snapshot_every=120):
    """Run both a raw env and the augmented env in lockstep, collecting data."""
    env_raw = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    env_raw = JoypadSpace(env_raw, SIMPLE_MOVEMENT)
    env_raw = SkipFrame(env_raw, skip=4)

    env_aug = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    env_aug = JoypadSpace(env_aug, SIMPLE_MOVEMENT)
    env_aug = SkipFrame(env_aug, skip=4)
    env_aug = RAMGridObservation(env_aug)
    env_aug = AugmentedRAMObservation(env_aug)

    env_raw.reset()
    env_aug.reset()

    history = []    # (step, pixel_obs, grid, scalars, info)
    snapshots = []  # subset for the snapshot figure

    actions = [1, 1, 2, 1, 1, 1, 2, 1, 1, 1]

    print(f"Collecting {n_steps} steps...")
    for step in range(1, n_steps + 1):
        action = actions[step % len(actions)]

        pixel_obs, _, d1, d1t, info = env_raw.step(action)
        aug_obs,   _, d2, d2t, _    = env_aug.step(action)

        grid    = aug_obs["grid"]
        scalars = aug_obs["scalars"]

        history.append((step, scalars.copy()))

        if step % snapshot_every == 0 or step == 1:
            snapshots.append((step, pixel_obs.copy(), grid.copy(),
                              scalars.copy(), dict(info)))
            print(f"  step {step:4d} | x_pos={info.get('x_pos'):4} | "
                  f"x_spd={scalars[0]:+.2f} y_vel={scalars[1]:+.2f} "
                  f"float={scalars[2]:.2f} power={scalars[3]:.2f} "
                  f"inv={scalars[4]:.2f} timer={scalars[5]:.2f}")

        if d1 or d1t or d2 or d2t:
            env_raw.reset()
            env_aug.reset()

    env_raw.close()
    env_aug.close()

    return history, snapshots


# ---------------------------------------------------------------------------
# Figure 1: snapshots (pixel | grid | scalar bars)
# ---------------------------------------------------------------------------

def plot_snapshots(snapshots, out_path):
    n = len(snapshots)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))
    if n == 1:
        axes = [axes]

    fig.suptitle("AugmentedRAMObservation — Snapshots\n"
                 "(pixel frame  |  RAM grid  |  scalar feature bars)",
                 fontsize=11, y=1.01)

    for row, (step, pixel, grid, scalars, info) in enumerate(snapshots):
        ax_px, ax_gr, ax_sc = axes[row]

        # --- pixel ---
        ax_px.imshow(pixel)
        ax_px.set_title(
            f"Step {step}  |  x={info.get('x_pos','?')}  status={info.get('status','?')}",
            fontsize=8)
        ax_px.axis("off")

        # --- grid ---
        _draw_grid(ax_gr, grid)
        ax_gr.set_title("RAM Grid (13×16)", fontsize=8)

        # --- scalars ---
        _draw_scalar_bars(ax_sc, scalars)
        ax_sc.set_title("Scalar Features", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"[OK] Snapshots saved → {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: time series of all 6 scalars
# ---------------------------------------------------------------------------

def plot_timeseries(history, out_path):
    steps   = np.array([h[0] for h in history])
    scalars = np.array([h[1] for h in history])  # (T, 6)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    fig.suptitle("AugmentedRAMObservation — Scalar Features Over Time", fontsize=12)
    axes_flat = axes.flatten()

    for i, (idx, label, unit, vmin, vmax, cmap_name) in enumerate(SCALAR_META):
        ax = axes_flat[i]
        vals = scalars[:, idx]
        color = plt.get_cmap(cmap_name)(0.65)

        ax.plot(steps, vals, linewidth=0.8, color=color, alpha=0.9)
        ax.fill_between(steps, vals, alpha=0.15, color=color)

        if vmin < 0:
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)

        ax.set_ylim(vmin - 0.05, vmax + 0.05)
        ax.set_ylabel("value", fontsize=8)
        ax.set_title(f"{label}  {unit}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3, linewidth=0.5)

        # basic stats annotation
        ax.text(0.98, 0.95,
                f"min={vals.min():.2f}  max={vals.max():.2f}  std={vals.std():.3f}",
                transform=ax.transAxes, fontsize=6.5, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    for ax in axes[-1]:
        ax.set_xlabel("step", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"[OK] Time series saved → {out_path}")


# ---------------------------------------------------------------------------
# Sanity checks (terminal)
# ---------------------------------------------------------------------------

def sanity_checks(history):
    scalars = np.array([h[1] for h in history])
    steps   = np.array([h[0] for h in history])
    T = len(history)

    print("\n" + "=" * 55)
    print("  SANITY CHECKS")
    print("=" * 55)

    checks = []

    # 1. x_speed varies (Mario moves)
    std = scalars[:, 0].std()
    ok = std > 0.05
    checks.append(("x_speed varies (Mario moves)", ok, f"std={std:.4f}"))

    # 2. y_velocity has both signs (jump + fall)
    has_up   = np.any(scalars[:, 1] < -0.05)
    has_down = np.any(scalars[:, 1] >  0.05)
    ok = has_up and has_down
    checks.append(("y_velocity: both up and down observed",
                   ok, f"up={has_up} down={has_down}"))

    # 3. float_state leaves ground (Mario jumps)
    off_ground = np.any(scalars[:, 2] > 0.05)
    checks.append(("float_state leaves 0 (airborne)", off_ground,
                   f"max={scalars[:, 2].max():.2f}"))

    # 4. game_timer decreases over time
    first_half = scalars[:T//2, 5].mean()
    second_half = scalars[T//2:, 5].mean()
    ok = second_half < first_half
    checks.append(("game_timer decreases over time",
                   ok, f"first_half={first_half:.3f} second_half={second_half:.3f}"))

    # 5. power_state is in [0, 1]
    ok = scalars[:, 3].min() >= 0.0 and scalars[:, 3].max() <= 1.0
    checks.append(("power_state within [0, 1]",
                   ok, f"range=[{scalars[:,3].min():.2f}, {scalars[:,3].max():.2f}]"))

    # 6. inv_timer is in [0, 1]
    ok = scalars[:, 4].min() >= 0.0 and scalars[:, 4].max() <= 1.0
    checks.append(("inv_timer within [0, 1]",
                   ok, f"range=[{scalars[:,4].min():.2f}, {scalars[:,4].max():.2f}]"))

    all_ok = True
    for desc, ok, detail in checks:
        tag = "[OK]  " if ok else "[FAIL]"
        if not ok:
            all_ok = False
        print(f"  {tag} {desc}")
        print(f"         {detail}")

    print("=" * 55)
    if all_ok:
        print("  All sanity checks PASSED")
    else:
        print("  Some checks FAILED — inspect the plots above")
    print("=" * 55 + "\n")

    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    snap_path   = os.path.join(results_dir, "augmented_snapshots.png")
    series_path = os.path.join(results_dir, "augmented_timeseries.png")

    print("\n" + "=" * 55)
    print("  AugmentedRAMObservation — Visual Test")
    print("=" * 55 + "\n")

    history, snapshots = collect_data(n_steps=600, snapshot_every=120)

    plot_snapshots(snapshots, snap_path)
    plot_timeseries(history, series_path)
    sanity_checks(history)


if __name__ == "__main__":
    main()
