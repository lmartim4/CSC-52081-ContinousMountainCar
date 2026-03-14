"""
Test 2: Super Mario Bros with RAM symbolic grid wrapper.

Run from the project root:
    python tests/test_ram_wrapper.py

Or from anywhere:
    PYTHONPATH=/path/to/project python tests/test_ram_wrapper.py
"""

import sys
import os

# --- Fix imports: add project root to sys.path ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from src.wrappers.ram_wrappers import (
    make_symbolic_env,
    RAMGridObservation,
    SkipFrame,
    EMPTY, SOLID, ENEMY, MARIO, POWERUP,
    VISIBLE_ROWS, VISIBLE_COLS,
)


# --- Pretty-print the grid ------------------------------------------------

SYMBOLS = {
    EMPTY:   ".",
    SOLID:   "#",
    ENEMY:   "E",
    MARIO:   "M",
    POWERUP: "?",
}


def print_grid(grid, step=None):
    header = f"--- Step {step} " if step is not None else "--- "
    header += "-" * (VISIBLE_COLS * 2 - len(header) + 4)
    print(header)
    col_nums = "   " + "".join(f"{c:2d}" for c in range(VISIBLE_COLS))
    print(col_nums)
    for r in range(VISIBLE_ROWS):
        row_str = f"{r:2d} "
        for c in range(VISIBLE_COLS):
            val = int(grid[r, c])
            row_str += f" {SYMBOLS.get(val, '?')}"
        print(row_str)
    print()


# --- Tests ----------------------------------------------------------------

def test_wrapper_shape():
    print("=" * 60)
    print("TEST 2a: Wrapper observation shape & dtype")
    print("=" * 60)

    env = make_symbolic_env(n_stack=1, flatten=False)
    obs = env.reset()
    print(f"  n_stack=1, flatten=False -> shape: {obs.shape}, dtype: {obs.dtype}")
    assert obs.shape == (VISIBLE_ROWS, VISIBLE_COLS), f"Expected (13, 16), got {obs.shape}"
    env.close()

    env = make_symbolic_env(n_stack=4, flatten=False)
    obs = env.reset()
    print(f"  n_stack=4, flatten=False -> shape: {obs.shape}, dtype: {obs.dtype}")
    assert obs.shape == (VISIBLE_ROWS, VISIBLE_COLS, 4), f"Expected (13, 16, 4), got {obs.shape}"
    env.close()

    env = make_symbolic_env(n_stack=4, flatten=True)
    obs = env.reset()
    expected_flat = VISIBLE_ROWS * VISIBLE_COLS * 4
    print(f"  n_stack=4, flatten=True  -> shape: {obs.shape}, dtype: {obs.dtype}")
    assert obs.shape == (expected_flat,), f"Expected ({expected_flat},), got {obs.shape}"
    env.close()

    print("\n[OK] Shape tests PASSED\n")


def test_grid_content(n_steps=300):
    print("=" * 60)
    print("TEST 2b: Grid content validation")
    print("=" * 60)

    env = make_symbolic_env(n_stack=1, flatten=False)
    obs = env.reset()

    found_mario = False
    found_solid = False
    found_enemy = False
    max_x = 0
    actions_sequence = [1, 1, 2, 1, 1, 1, 2, 1]

    for step in range(n_steps):
        action = actions_sequence[step % len(actions_sequence)]
        obs, reward, done, info = env.step(action)
        max_x = max(max_x, info.get("x_pos", 0))

        grid = obs if obs.ndim == 2 else obs[:, :, -1]
        has_mario = np.any(grid == MARIO)
        has_solid = np.any(grid == SOLID)
        has_enemy = np.any(grid == ENEMY)

        if has_mario: found_mario = True
        if has_solid: found_solid = True
        if has_enemy: found_enemy = True

        if step in [0, 10, 50] or (has_enemy and not found_enemy):
            print_grid(grid, step)
            print(f"  Mario: {has_mario} | Solid: {has_solid} | Enemy: {has_enemy}")
            print(f"  info: x_pos={info.get('x_pos')}, y_pos={info.get('y_pos')}, "
                  f"status={info.get('status')}, time={info.get('time')}")

        if done:
            obs = env.reset()

    env.close()

    print(f"\nResults after {n_steps} steps:")
    print(f"  Mario found:   {'YES' if found_mario else 'NO - PROBLEM'}")
    print(f"  Solid tiles:   {'YES' if found_solid else 'NO - PROBLEM'}")
    print(f"  Enemies found: {'YES' if found_enemy else 'maybe (need more steps)'}")
    print(f"  Max x_pos:     {max_x}")

    assert found_mario, "Mario was never found in the grid!"
    assert found_solid, "No solid tiles were found!"
    print("\n[OK] Content tests PASSED\n")


def test_grid_values(n_steps=100):
    print("=" * 60)
    print("TEST 2c: Grid value range")
    print("=" * 60)

    env = make_symbolic_env(n_stack=1, flatten=False)
    obs = env.reset()
    all_values = set()

    for step in range(n_steps):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        all_values.update(np.unique(obs.astype(int)).tolist())
        if done:
            obs = env.reset()

    env.close()

    valid_values = {EMPTY, SOLID, ENEMY, MARIO, POWERUP}
    unexpected = all_values - valid_values
    print(f"  Values observed: {sorted(all_values)}")
    print(f"  Expected values: {sorted(valid_values)}")
    assert all_values.issubset(valid_values), f"Unexpected: {unexpected}"
    print("\n[OK] Value range tests PASSED\n")


def test_frame_stack():
    print("=" * 60)
    print("TEST 2d: Frame stack temporal consistency")
    print("=" * 60)

    env = make_symbolic_env(n_stack=4, n_skip=1, flatten=False)
    obs = env.reset()

    for _ in range(20):
        obs, _, done, _ = env.step(1)
        if done:
            obs = env.reset()

    frames_differ = any(
        not np.array_equal(obs[:, :, i], obs[:, :, i + 1])
        for i in range(3)
    )

    env.close()
    print(f"  Frames differ after movement: {'YES' if frames_differ else 'WARNING'}")
    print("\n[OK] Frame stack test PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  SUPER MARIO BROS - RAM WRAPPER TEST SUITE")
    print("=" * 60 + "\n")

    test_wrapper_shape()
    test_grid_content()
    test_grid_values()
    test_frame_stack()

    print("=" * 60)
    print("  ALL TESTS COMPLETED")
    print("=" * 60)
