"""
Test 1: Super Mario Bros base environment (pixels, no wrapper).

Run from the project root:
    python tests/test_base_env.py
"""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import numpy as np


def test_base_env(n_steps=200):
    print("=" * 60)
    print("TEST 1: Base environment (pixel observations)")
    print("=" * 60)

    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v3")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    print(f"\nObservation space: {env.observation_space}")
    print(f"  shape: {env.observation_space.shape}")
    print(f"  dtype: {env.observation_space.dtype}")
    print(f"Action space: {env.action_space}")
    print(f"  n_actions: {env.action_space.n}")

    obs = env.reset()
    print(f"\nAfter reset:")
    print(f"  obs shape: {obs.shape}")
    print(f"  obs dtype: {obs.dtype}")
    print(f"  obs range: [{obs.min()}, {obs.max()}]")

    print(f"\nRunning {n_steps} random steps...")
    total_reward = 0.0
    max_x = 0

    for step in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        max_x = max(max_x, info.get("x_pos", 0))
        if done:
            obs = env.reset()

    print(f"\nInfo dict keys: {list(info.keys())}")
    print(f"Info dict sample: {info}")
    print(f"Total reward: {total_reward:.1f}")
    print(f"Max x_pos reached: {max_x}")

    ram = env.unwrapped.ram
    print(f"\nRAM access check:")
    print(f"  ram type: {type(ram)}, length: {len(ram)}")
    print(f"  ram[0x0086] (player x on screen): {ram[0x0086]}")
    print(f"  ram[0x00CE] (player y on screen): {ram[0x00CE]}")
    print(f"  tiles at 0x0500-0x069F: {np.count_nonzero(ram[0x0500:0x06A0])}/416 non-zero")

    for i in range(5):
        drawn = ram[0x000F + i]
        if drawn:
            print(f"  enemy slot {i}: type=0x{ram[0x0016+i]:02X}, "
                  f"x={ram[0x0087+i]}, y={ram[0x00CF+i]}")

    env.close()
    print("\n[OK] Base environment test PASSED")


if __name__ == "__main__":
    test_base_env()
