"""Pixel-based wrappers adapted from vietnh1009/Super-mario-bros-PPO-pytorch.

Key improvements over vanilla wrappers:
  - Custom reward shaping (score bonus, flag/death penalty)
  - Max-pooling over last 2 skip frames (reduces NES sprite flickering)
  - Built-in frame stacking inside the skip wrapper
  - SubprocVecEnv support for parallel environments

Reference: https://github.com/vietnh1009/Super-mario-bros-PPO-pytorch
"""

import cv2
import gym_super_mario_bros
import numpy as np
from gym import spaces, Wrapper
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

from src.config import ENV_ID, SIMPLE_MOVEMENT as USE_SIMPLE


def process_frame(frame):
    """Convert RGB frame to grayscale 84x84 normalized float."""
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        return frame.astype(np.float32) / 255.0
    return np.zeros((84, 84), dtype=np.float32)


class CustomReward(Wrapper):
    """Custom reward shaping from vietnh1009's implementation.

    - Adds score-based reward: (score_delta) / 40
    - Flag reached: +50
    - Death / timeout: -50
    - All rewards scaled by /10
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(84, 84, 1), dtype=np.float32
        )
        self.curr_score = 0
        self.current_x = 40

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            state, reward, done, truncated, info = result
        else:
            state, reward, done, info = result
            truncated = False

        state = process_frame(state)[:, :, np.newaxis]

        # Score-based reward
        reward += (info["score"] - self.curr_score) / 40.0
        self.curr_score = info["score"]

        # Flag / death bonus
        if done:
            if info.get("flag_get", False):
                reward += 50
            else:
                reward -= 50

        self.current_x = info.get("x_pos", self.current_x)
        return state, reward / 10.0, done, truncated, info

    def reset(self, **kwargs):
        kwargs.pop('seed', None)
        kwargs.pop('options', None)
        self.curr_score = 0
        self.current_x = 40
        result = self.env.reset(**kwargs)
        obs = result[0] if isinstance(result, tuple) else result
        return process_frame(obs)[:, :, np.newaxis], {}


class CustomSkipFrame(Wrapper):
    """Frame skip with max-pooling + built-in frame stacking.

    - Repeats action for `skip` frames, sums rewards
    - Max-pools over the last skip//2 frames (reduces NES flickering)
    - Maintains a stack of `skip` frames as the observation
    - Output shape: (84, 84, skip)
    """

    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(84, 84, skip), dtype=np.float32
        )
        self.states = np.zeros((84, 84, skip), dtype=np.float32)

    def step(self, action):
        total_reward = 0
        last_states = []
        done = False
        truncated = False
        info = {}

        for i in range(self.skip):
            state, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            if i >= self.skip // 2:
                last_states.append(state)
            if done or truncated:
                return self.states.astype(np.float32), total_reward, done, truncated, info

        # Max-pool over last frames to reduce flickering
        max_state = np.max(np.stack(last_states, axis=0), axis=0)
        self.states[:, :, :-1] = self.states[:, :, 1:]
        self.states[:, :, -1] = max_state[:, :, 0]
        return self.states.astype(np.float32), total_reward, done, truncated, info

    def reset(self, **kwargs):
        kwargs.pop('seed', None)
        kwargs.pop('options', None)
        state, info = self.env.reset(**kwargs)
        self.states = np.stack([state[:, :, 0]] * self.skip, axis=-1).astype(np.float32)
        return self.states.astype(np.float32), info


def make_pixel_env(env_id=ENV_ID, skip=4):
    """Create a single pixel-based Mario environment."""
    env = gym_super_mario_bros.make(env_id)
    actions = SIMPLE_MOVEMENT if USE_SIMPLE else COMPLEX_MOVEMENT
    env = JoypadSpace(env, actions)
    env = CustomReward(env)
    env = CustomSkipFrame(env, skip=skip)
    return env


def make_pixel_vec_env(env_id=ENV_ID, skip=4, num_envs=8):
    """Create parallel pixel-based Mario environments using SubprocVecEnv.

    Each env runs in its own process — 8 envs = ~8x sample throughput.
    """
    from stable_baselines3.common.vec_env import SubprocVecEnv

    def _make_env(env_id, skip):
        def _init():
            return make_pixel_env(env_id=env_id, skip=skip)
        return _init

    return SubprocVecEnv([_make_env(env_id, skip) for _ in range(num_envs)])
